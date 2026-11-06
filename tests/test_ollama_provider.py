from __future__ import annotations

import json

import httpx
import pytest

from retail_guardian.investigation_models import (
    Evidence,
    InvestigationContext,
    ToolDefinition,
)
from retail_guardian.providers import (
    ModelOutputValidationError,
    ModelUnavailableError,
    OllamaInvestigationModel,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


def _context(evidence: list[Evidence] | None = None) -> InvestigationContext:
    return InvestigationContext(
        incident={"incident_id": "INC-TEST", "violations": []},
        hypotheses=[],
        evidence=evidence or [],
        available_tools=[
            ToolDefinition("inspect_dataset", "Inspect data", {}),
            ToolDefinition(
                "sample_problematic_records",
                "Sample records",
                {
                    "check_id": {"type": "string", "required": False},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
            ),
        ],
        remaining_tool_calls=4,
    )


def _tool_decision(name: str = "inspect_dataset", *, limit: int | None = None) -> dict:
    tool_request = {
        "name": name,
        "rationale": "Inspect observed pipeline data",
    }
    if name == "sample_problematic_records":
        tool_request.update({"check_id": None, "limit": limit})
    elif name == "get_pipeline_history":
        tool_request["limit"] = limit
    elif name == "identify_affected_dimensions":
        tool_request["check_id"] = None
    return {
        "action": "tool",
        "hypotheses": [
            {
                "hypothesis_id": "H-1",
                "statement": "More evidence is needed",
                "diagnosis_code": None,
                "status": "active",
                "supporting_evidence": [],
                "contradicting_evidence": [],
            }
        ],
        "tool_request": tool_request,
        "conclusion": None,
    }


def _conclusion(evidence_id: str) -> dict:
    return {
        "action": "conclude",
        "hypotheses": [
            {
                "hypothesis_id": "H-1",
                "statement": "The upstream schema changed",
                "diagnosis_code": "schema_change",
                "status": "confirmed",
                "supporting_evidence": [evidence_id],
                "contradicting_evidence": [],
            }
        ],
        "tool_request": None,
        "conclusion": {
            "diagnosis_code": "schema_change",
            "final_diagnosis": "The upstream schema changed",
            "confidence": "high",
            "evidence_references": [evidence_id],
            "outstanding_questions": [],
            "recommended_actions": ["Confirm the supplier contract"],
        },
    }


def _ollama_response(decision: dict, *, prompt_tokens: int = 12, output_tokens: int = 8) -> dict:
    return {
        "model": "qwen3:4b",
        "message": {"role": "assistant", "content": json.dumps(decision)},
        "done": True,
        "prompt_eval_count": prompt_tokens,
        "eval_count": output_tokens,
    }


def test_ollama_returns_structured_tool_decision() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url == "http://ollama.test:11434/api/chat"
        assert payload["model"] == "qwen3:4b"
        assert payload["stream"] is False
        assert payload["think"] is False
        assert payload["format"]["type"] == "object"
        context = json.loads(payload["messages"][1]["content"])
        assert context["available_evidence_ids"] == []
        return httpx.Response(200, json=_ollama_response(_tool_decision()))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaInvestigationModel(endpoint="http://ollama.test:11434", client=client)
        decision = provider.decide(_context())

    assert decision.action == "tool"
    assert decision.tool_request.name == "inspect_dataset"
    assert decision.token_usage == 20
    assert provider.telemetry.calls == 1
    assert provider.telemetry.provider_name == "ollama"


def test_ollama_configuration_uses_environment(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_ENDPOINT", "http://local-runtime:9999/")
    monkeypatch.setenv("OLLAMA_MODEL", "local-test:latest")
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500))) as client:
        provider = OllamaInvestigationModel(client=client, timeout_seconds=7, max_retries=0)

    assert provider.endpoint == "http://local-runtime:9999"
    assert provider.model == "local-test:latest"
    assert provider.timeout_seconds == 7


def test_malformed_output_is_retried_and_state_is_preserved() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(200, json={"message": {"content": "not JSON"}})
        payload = json.loads(request.content)
        correction = payload["messages"][-1]["content"]
        assert "specific validation failure" in correction.lower()
        assert "invalid structured decision" in correction.lower()
        return httpx.Response(200, json=_ollama_response(_tool_decision()))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=1)
        decision = provider.decide(_context())

    assert decision.action == "tool"
    assert provider.telemetry.validation_failures == 1
    assert provider.telemetry.retries == 1
    assert provider.telemetry.calls == 2
    assert provider.telemetry.rejected_decisions[0].decision is None


def test_tentative_hypothesis_with_fabricated_evidence_is_corrected() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        decision = _tool_decision()
        if attempts == 1:
            decision["hypotheses"][0]["supporting_evidence"] = ["EV-NOT-YET-CREATED"]
        else:
            correction = json.loads(request.content)["messages"][-1]["content"]
            assert "EV-NOT-YET-CREATED" in correction
            assert "Available evidence IDs: none" in correction
            assert "tentative active hypothesis may use empty evidence arrays" in correction
        return httpx.Response(200, json=_ollama_response(decision))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=1)
        decision = provider.decide(_context())

    assert decision.action == "tool"
    assert decision.hypotheses[0].supporting_evidence == []
    assert provider.telemetry.validation_failures == 1
    rejection = provider.telemetry.rejected_decisions[0]
    assert rejection.error_type == "ModelOutputValidationError"
    assert "unavailable evidence IDs" in rejection.error
    assert rejection.decision["hypotheses"][0]["supporting_evidence"] == ["EV-NOT-YET-CREATED"]


def test_invalid_tool_selection_is_retried_before_execution() -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        decision = _tool_decision("run_sql") if attempts == 1 else _tool_decision()
        return httpx.Response(200, json=_ollama_response(decision))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=1)
        decision = provider.decide(_context())

    assert decision.tool_request.name == "inspect_dataset"
    assert provider.telemetry.validation_failures == 1
    assert provider.telemetry.invalid_tool_calls == 1


def test_invalid_tool_arguments_are_rejected() -> None:
    decision = _tool_decision("inspect_dataset")
    decision["tool_request"]["limit"] = 10
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=_ollama_response(decision)))
    with httpx.Client(transport=transport) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=0)
        with pytest.raises(ModelOutputValidationError, match="invalid structured decision"):
            provider.decide(_context())

    assert provider.telemetry.invalid_tool_calls == 1


def test_conclusion_must_reference_real_evidence() -> None:
    evidence = Evidence("EV-REAL", "inspect_dataset", "Observed schema", {"columns": []})
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        evidence_id = "EV-FABRICATED" if attempts == 1 else "EV-REAL"
        return httpx.Response(200, json=_ollama_response(_conclusion(evidence_id)))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=1)
        decision = provider.decide(_context([evidence]))

    assert decision.conclusion.evidence_references == ["EV-REAL"]
    assert provider.telemetry.validation_failures == 1


def test_connection_failure_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=0)
        with pytest.raises(ProviderUnavailableError, match="unavailable"):
            provider.decide(_context())

    assert provider.telemetry.connection_failures == 1


def test_timeout_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow model", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=0)
        with pytest.raises(ProviderTimeoutError, match="timed out"):
            provider.decide(_context())

    assert provider.telemetry.timeouts == 1


def test_unavailable_model_is_not_retried() -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(404, json={"error": "model 'qwen3:4b' not found"})
    )
    with httpx.Client(transport=transport) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=2)
        with pytest.raises(ModelUnavailableError, match="qwen3:4b"):
            provider.decide(_context())

    assert provider.telemetry.calls == 1


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(200, content=b"{"), "malformed response JSON"),
        (httpx.Response(200, json=[]), "JSON object"),
        (httpx.Response(200, json={"message": {"content": ""}}), "message.content"),
    ],
)
def test_invalid_ollama_envelopes_are_reported(response, message) -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda _: response)) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=0)
        with pytest.raises(ModelOutputValidationError, match=message):
            provider.decide(_context())

    assert provider.telemetry.validation_failures == 1


def test_ollama_http_failure_is_reported() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(503, text="runtime is starting"))
    with httpx.Client(transport=transport) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=0)
        with pytest.raises(ProviderUnavailableError, match="HTTP 503"):
            provider.decide(_context())

    assert provider.telemetry.connection_failures == 1


def test_tool_argument_outside_allowed_range_is_rejected() -> None:
    response = _tool_decision("sample_problematic_records", limit=0)
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=_ollama_response(response)))
    with httpx.Client(transport=transport) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=0)
        with pytest.raises(ModelOutputValidationError, match="allowed range"):
            provider.decide(_context())

    assert provider.telemetry.invalid_tool_calls == 1


def test_unsupported_diagnosis_code_is_rejected() -> None:
    evidence = Evidence("EV-REAL", "inspect_dataset", "Observed schema", {})
    decision = _conclusion("EV-REAL")
    decision["hypotheses"][0]["diagnosis_code"] = "database_hack"
    decision["conclusion"]["diagnosis_code"] = "database_hack"
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=_ollama_response(decision)))
    with httpx.Client(transport=transport) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=0)
        with pytest.raises(ModelOutputValidationError, match="unsupported diagnosis"):
            provider.decide(_context([evidence]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timeout_seconds": 0}, "positive"),
        ({"max_retries": 6}, "between 0 and 5"),
        ({"endpoint": "localhost:11434"}, "http"),
    ],
)
def test_invalid_provider_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        OllamaInvestigationModel(**kwargs)
