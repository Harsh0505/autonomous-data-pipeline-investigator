from __future__ import annotations

import json

import httpx
import pytest

from retail_guardian.investigation_models import (
    CompletedToolCall,
    Evidence,
    InvestigationContext,
    InvestigationQuestion,
    QuestionStatus,
    RootCauseHypothesis,
    ToolDefinition,
)
from retail_guardian.providers import (
    ModelOutputValidationError,
    ModelUnavailableError,
    OllamaInvestigationModel,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


def _context(
    evidence: list[Evidence] | None = None,
    hypotheses: list[RootCauseHypothesis] | None = None,
) -> InvestigationContext:
    return InvestigationContext(
        incident={"incident_id": "INC-TEST", "violations": []},
        hypotheses=hypotheses or [],
        evidence=evidence or [],
        available_tools=[
            ToolDefinition("inspect_dataset", "Inspect data", {}),
            ToolDefinition(
                "sample_problematic_records",
                "Sample records",
                {
                    "check_id": {
                        "type": "string",
                        "required": False,
                        "enum": ["schema_contract"],
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
            ),
        ],
        remaining_tool_calls=4,
        available_check_ids=["schema_contract"],
    )


def _tool_decision(name: str = "inspect_dataset", *, limit: int | None = None) -> dict:
    tool_request = {
        "name": name,
        "rationale": "Inspect observed pipeline data",
        "question_id": f"Q-{name}",
        "question": f"What evidence can {name} provide?",
        "hypothesis_ids": [],
    }
    if name == "sample_problematic_records":
        tool_request.update({"check_id": None, "limit": limit})
    elif name == "get_pipeline_history":
        tool_request["limit"] = limit
    elif name == "identify_affected_dimensions":
        tool_request["check_id"] = None
    return {
        "action": "tool",
        "hypothesis_operations": [
            {
                "operation": "create",
                "hypothesis_id": "H-1",
                "statement": "More evidence is needed",
                "diagnosis_code": None,
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
        "hypothesis_operations": [
            {
                "operation": "confirm",
                "hypothesis_id": "H-1",
                "statement": "The upstream schema changed",
                "diagnosis_code": "schema_change",
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
        assert "Decision JSON Schema" not in payload["messages"][0]["content"]
        assert payload["format"]["type"] == "object"
        action_schemas = {
            item["properties"]["action"]["const"]: item for item in payload["format"]["oneOf"]
        }
        assert action_schemas["tool"]["properties"]["conclusion"] == {"type": "null"}
        assert action_schemas["conclude"]["properties"]["tool_request"] == {"type": "null"}
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
    assert provider.telemetry.first_attempt_schema_valid_decisions == 1


def test_same_causal_question_can_use_multiple_distinct_tools() -> None:
    question = "Did an internal retry repeat already committed output?"
    context = InvestigationContext(
        incident={"incident_id": "INC-CAUSAL", "violations": []},
        hypotheses=[],
        evidence=[],
        available_tools=[
            ToolDefinition("get_orchestration_events", "Read task attempts", {}),
        ],
        remaining_tool_calls=3,
        investigation_questions=[
            InvestigationQuestion(
                "q3",
                question,
                [],
                "get_pipeline_logs",
                {},
                QuestionStatus.ANSWERED,
                "EV-LOGS",
            )
        ],
        investigation_objective="identify_underlying_root_cause",
        allowed_diagnosis_codes=["supplier_batch_replay", "failed_job_retry"],
    )
    response = _tool_decision("get_orchestration_events")
    response["hypothesis_operations"] = []
    response["tool_request"].update(
        {
            "question_id": "q4",
            "question": question,
            "hypothesis_ids": [],
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        supplied_context = json.loads(json.loads(request.content)["messages"][1]["content"])
        causal_questions = supplied_context["investigation_memory"]["causal_questions"]
        assert len(causal_questions) == 1
        assert causal_questions[0]["question_id"] == "q3"
        assert causal_questions[0]["evidence_requests"][0]["tool_name"] == "get_pipeline_logs"
        return httpx.Response(200, json=_ollama_response(response))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=0)
        decision = provider.decide(context)

    assert decision.action == "tool"
    assert decision.tool_request.name == "get_orchestration_events"
    assert decision.tool_request.question == question


def test_ollama_context_summarizes_evidence_and_removes_completed_tools() -> None:
    evidence = Evidence(
        "EV-VALIDATION",
        "get_validation_failures",
        "Retrieved 1 validation failure(s)",
        {
            "failures": [
                {
                    "check_id": "supplier_currency",
                    "kind": "deterministic",
                    "severity": "critical",
                    "finding": {
                        "message": "Supplier currency differs from its contract",
                        "affected_rows": 5,
                        "sample_values": [{"supplier_id": "meridian_uk", "currency": "USD"}],
                        "context": {},
                    },
                }
            ]
        },
    )
    contract_evidence = Evidence(
        "EV-CONTRACT",
        "get_data_contract",
        "Retrieved configured contracts",
        {
            "supplier_currencies": {"meridian_uk": "GBP"},
            "unit_price_z_score_threshold": 5.0,
        },
    )
    context = InvestigationContext(
        incident={"incident_id": "INC-TEST"},
        hypotheses=[],
        evidence=[evidence, contract_evidence],
        available_tools=[
            ToolDefinition(
                "sample_problematic_records",
                "Sample records",
                {
                    "check_id": {"type": "string", "required": False},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
            )
        ],
        remaining_tool_calls=3,
        completed_tool_calls=[
            CompletedToolCall("get_validation_failures", {}, "EV-VALIDATION", True),
            CompletedToolCall("get_data_contract", {}, "EV-CONTRACT", True),
        ],
        remaining_invalid_tool_calls=2,
        available_check_ids=["supplier_currency"],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        supplied_context = json.loads(payload["messages"][1]["content"])
        summary = supplied_context["evidence_registry"]["EV-VALIDATION"]
        assert summary["key_findings"]["failed_checks"][0]["affected_rows"] == 5
        assert supplied_context["citable_evidence_ids"] == [
            "EV-VALIDATION",
            "EV-CONTRACT",
        ]
        assert supplied_context["investigation_memory"]["used_tool_calls"][0]["name"] == (
            "get_validation_failures"
        )
        capability = supplied_context["tool_capabilities"]["sample_problematic_records"]
        assert capability["provides"] == "Sample records"
        assert "arguments" in capability
        assert "evidence" not in supplied_context
        assert "evidence_summary" not in supplied_context
        assert "currency_mismatch" in supplied_context["diagnosis_taxonomy"]
        assert supplied_context["remaining_invalid_tool_calls"] == 2
        schema_text = json.dumps(payload["format"])
        assert "get_validation_failures" not in schema_text
        assert "EV-VALIDATION" in schema_text
        assert "supplier_currency" in schema_text
        return httpx.Response(
            200,
            json=_ollama_response(_tool_decision("sample_problematic_records", limit=5)),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=0)
        decision = provider.decide(context)

    assert decision.tool_request.name == "sample_problematic_records"


def test_root_cause_objective_restricts_tools_and_diagnosis_codes() -> None:
    operational_evidence = Evidence(
        "EV-SUBMISSION",
        "get_supplier_submissions",
        "Retrieved supplier records",
        {
            "records": [
                {
                    "record_id": "OP-1",
                    "details": {"field_statistics": {"unit_price": {"null_count": 0}}},
                }
            ]
        },
    )
    context = InvestigationContext(
        incident={
            "incident_id": "INC-CAUSAL",
            "violations": [
                {
                    "check_id": "unique_order_id",
                    "severity": "critical",
                    "finding": {"message": "Order identifiers must be unique"},
                }
            ],
        },
        hypotheses=[],
        evidence=[operational_evidence],
        available_tools=[
            ToolDefinition("get_supplier_submissions", "Read supplier submission manifests", {})
        ],
        remaining_tool_calls=6,
        available_check_ids=["unique_order_id"],
        investigation_objective="identify_underlying_root_cause",
        allowed_diagnosis_codes=[
            "supplier_batch_replay",
            "failed_job_retry",
            "incorrect_join_cardinality",
        ],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        supplied_context = json.loads(payload["messages"][1]["content"])
        schema_text = json.dumps(payload["format"])
        assert supplied_context["investigation_objective"] == ("identify_underlying_root_cause")
        assert set(supplied_context["diagnosis_taxonomy"]) == {
            "supplier_batch_replay",
            "failed_job_retry",
            "incorrect_join_cardinality",
        }
        assert (
            supplied_context["evidence_registry"]["EV-SUBMISSION"]["key_findings"]["records"][0][
                "details"
            ]["field_statistics"]["unit_price"]["null_count"]
            == 0
        )
        assert "supplier_batch_replay" in schema_text
        assert "missing_values" not in schema_text
        return httpx.Response(
            200,
            json=_ollama_response(_tool_decision("get_supplier_submissions")),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=0)
        decision = provider.decide(context)

    assert decision.tool_request.name == "get_supplier_submissions"


def test_initial_causal_schema_allows_hypotheses_but_not_diagnosed_conclusions() -> None:
    context = InvestigationContext(
        incident={"incident_id": "INC-CAUSAL", "violations": []},
        hypotheses=[],
        evidence=[],
        available_tools=[ToolDefinition("get_supplier_submissions", "Read supplier manifests", {})],
        remaining_tool_calls=4,
        investigation_objective="identify_underlying_root_cause",
        allowed_diagnosis_codes=[
            "supplier_batch_replay",
            "failed_job_retry",
            "incorrect_join_cardinality",
        ],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        schema = json.loads(request.content)["format"]
        actions = [item["properties"]["action"]["const"] for item in schema["oneOf"]]
        operation_codes = schema["$defs"]["_HypothesisOperationPayload"]["properties"][
            "diagnosis_code"
        ]["anyOf"]
        assert actions == ["tool"]
        assert any(item.get("enum") == context.allowed_diagnosis_codes for item in operation_codes)
        decision = _tool_decision("get_supplier_submissions")
        decision["hypothesis_operations"][0]["diagnosis_code"] = "failed_job_retry"
        return httpx.Response(200, json=_ollama_response(decision))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=0)
        decision = provider.decide(context)

    assert decision.action == "tool"
    assert decision.hypothesis_operations[0].diagnosis_code == "failed_job_retry"


def test_causal_confirmation_retries_when_cited_facts_do_not_support_claim() -> None:
    evidence = Evidence(
        "EV-SUBMISSION",
        "get_supplier_submissions",
        "Retrieved one supplier submission",
        {
            "records": [
                {
                    "record_id": "OP-1",
                    "occurred_at": "2026-10-20T11:20:00+00:00",
                    "details": {
                        "supplier_id": "atlas_us",
                        "batch_id": "batch-1",
                        "checksum": "checksum-1",
                        "status": "accepted",
                    },
                }
            ],
            "coverage": {"complete": True, "record_count": 1},
        },
    )
    context = InvestigationContext(
        incident={"incident_id": "INC-CAUSAL", "violations": []},
        hypotheses=[
            RootCauseHypothesis(
                "H-replay",
                "The supplier batch may have been replayed",
                "supplier_batch_replay",
            )
        ],
        evidence=[evidence],
        available_tools=[
            ToolDefinition("get_orchestration_events", "Read orchestration history", {})
        ],
        remaining_tool_calls=3,
        investigation_objective="identify_underlying_root_cause",
        allowed_diagnosis_codes=["supplier_batch_replay", "failed_job_retry"],
    )
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        payload = json.loads(request.content)
        if attempts == 1:
            memory = json.loads(payload["messages"][1]["content"])["investigation_memory"]
            replay_check = memory["hypothesis_checks"][0]
            assert replay_check["hypothesis_id"] == "H-replay"
            assert replay_check["contradicting_evidence"] == ["EV-SUBMISSION"]
            assert replay_check["information_needed"]
            assert memory["unexplored_tools"] == ["get_orchestration_events"]
            decision = {
                "action": "conclude",
                "hypothesis_operations": [
                    {
                        "operation": "confirm",
                        "hypothesis_id": "H-replay",
                        "statement": None,
                        "diagnosis_code": "supplier_batch_replay",
                        "supporting_evidence": ["EV-SUBMISSION"],
                        "contradicting_evidence": [],
                    }
                ],
                "tool_request": None,
                "conclusion": {
                    "diagnosis_code": "supplier_batch_replay",
                    "final_diagnosis": "The supplier replayed a batch",
                    "confidence": "high",
                    "evidence_references": ["EV-SUBMISSION"],
                    "outstanding_questions": [],
                    "recommended_actions": [],
                },
            }
        else:
            correction = payload["messages"][-1]["content"]
            assert "rejected and was not applied" in correction
            assert "Hypothesis IDs currently in state: H-replay" in correction
            assert "Available evidence IDs: EV-SUBMISSION" in correction
            assert "get_orchestration_events" in correction
            actions = [item["properties"]["action"]["const"] for item in payload["format"]["oneOf"]]
            assert actions == ["tool"]
            recovery_operations = payload["format"]["oneOf"][0]["properties"][
                "hypothesis_operations"
            ]
            assert recovery_operations["maxItems"] == 0
            assert "Causal recovery mode is active" in correction
            assert "hypothesis_operations: []" in correction
            if attempts == 2:
                assert "at least two accepted submissions" in correction
            else:
                assert "invalid structured decision" in correction.lower()
            decision = {
                "action": "tool",
                "hypothesis_operations": [],
                "tool_request": {
                    "name": "get_orchestration_events",
                    "rationale": "Test the competing retry explanation",
                    "question_id": "Q-orchestration",
                    "question": "Did orchestration retry the execution?",
                    "hypothesis_ids": ["H-replay"],
                },
                "conclusion": None,
            }
            if attempts == 2:
                decision["tool_request"]["query"] = "select *"
        return httpx.Response(200, json=_ollama_response(decision))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=2)
        decision = provider.decide(context)

    assert decision.action == "tool"
    assert decision.tool_request.name == "get_orchestration_events"
    assert provider.telemetry.validation_failures == 2
    assert provider.telemetry.invalid_tool_calls == 1
    assert provider.telemetry.retry_recoveries == 1
    assert "causal evidence is insufficient" in provider.telemetry.rejected_decisions[0].error


def test_causal_inconclusive_requires_exhausting_operational_sources() -> None:
    evidence = Evidence(
        "EV-SUBMISSION",
        "get_supplier_submissions",
        "Retrieved one supplier submission",
        {"records": [], "coverage": {"complete": True}},
    )
    decision = {
        "action": "conclude",
        "hypothesis_operations": [],
        "tool_request": None,
        "conclusion": {
            "diagnosis_code": None,
            "final_diagnosis": "The available evidence is insufficient",
            "confidence": "insufficient_evidence",
            "evidence_references": ["EV-SUBMISSION"],
            "outstanding_questions": [],
            "recommended_actions": [],
        },
    }
    context = InvestigationContext(
        incident={"incident_id": "INC-CAUSAL"},
        hypotheses=[],
        evidence=[evidence],
        available_tools=[
            ToolDefinition("get_orchestration_events", "Read orchestration history", {})
        ],
        remaining_tool_calls=3,
        investigation_objective="identify_underlying_root_cause",
        allowed_diagnosis_codes=["supplier_batch_replay", "failed_job_retry"],
    )
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=_ollama_response(decision)))

    with httpx.Client(transport=transport) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=0)
        with pytest.raises(ModelOutputValidationError, match="remain unexplored"):
            provider.decide(context)


def test_causal_inconclusive_accepts_explicitly_unavailable_exhausted_evidence() -> None:
    evidence = Evidence(
        "EV-ORCHESTRATION",
        "get_orchestration_events",
        "Retrieved unavailable orchestration metadata",
        {
            "records": [
                {
                    "details": {
                        "state": "metadata_unavailable",
                        "reason": "retention_window_expired",
                    }
                }
            ]
        },
    )
    decision = {
        "action": "conclude",
        "hypothesis_operations": [],
        "tool_request": None,
        "conclusion": {
            "diagnosis_code": None,
            "final_diagnosis": "Operational evidence is unavailable",
            "confidence": "insufficient_evidence",
            "evidence_references": ["EV-ORCHESTRATION"],
            "outstanding_questions": [],
            "recommended_actions": [],
        },
    }
    context = InvestigationContext(
        incident={"incident_id": "INC-CAUSAL"},
        hypotheses=[],
        evidence=[evidence],
        available_tools=[ToolDefinition("get_validation_failures", "Read failures", {})],
        remaining_tool_calls=3,
        investigation_objective="identify_underlying_root_cause",
        allowed_diagnosis_codes=["supplier_batch_replay", "failed_job_retry"],
    )
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=_ollama_response(decision)))

    with httpx.Client(transport=transport) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=0)
        result = provider.decide(context)

    assert result.action == "conclude"
    assert result.conclusion.confidence.value == "insufficient_evidence"


def test_completed_tool_selection_gets_specific_correction() -> None:
    evidence = Evidence("EV-REAL", "inspect_dataset", "Observed dataset", {})
    context = InvestigationContext(
        incident={"incident_id": "INC-TEST"},
        hypotheses=[],
        evidence=[evidence],
        available_tools=[ToolDefinition("get_data_contract", "Get contract", {})],
        remaining_tool_calls=3,
        completed_tool_calls=[CompletedToolCall("inspect_dataset", {}, "EV-REAL", True)],
    )
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            decision = _tool_decision("inspect_dataset")
        else:
            correction = json.loads(request.content)["messages"][-1]["content"]
            assert "not currently available" in correction
            assert "Completed successful tools: inspect_dataset" in correction
            assert "Currently available tools: get_data_contract" in correction
            decision = _tool_decision("get_data_contract")
        return httpx.Response(200, json=_ollama_response(decision))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=1)
        decision = provider.decide(context)

    assert decision.tool_request.name == "get_data_contract"
    assert provider.telemetry.invalid_tool_calls == 1


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
    assert provider.telemetry.retry_recoveries == 1
    assert provider.telemetry.first_attempt_schema_valid_decisions == 0
    assert provider.telemetry.calls == 2
    assert provider.telemetry.rejected_decisions[0].decision is None


def test_exhausted_retries_preserve_all_available_token_usage() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {"content": "not JSON"},
                "prompt_eval_count": 11,
                "eval_count": 7,
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=2)
        with pytest.raises(ModelOutputValidationError, match="invalid structured decision"):
            provider.decide(_context())

    assert provider.telemetry.calls == 3
    assert provider.telemetry.validation_failures == 3
    assert provider.telemetry.token_usage_available
    assert provider.telemetry.token_usage == 54


@pytest.mark.parametrize(
    ("response_payload", "usage_available"),
    [
        (
            {
                "message": {"content": "not JSON"},
                "prompt_eval_count": 0,
                "eval_count": 0,
            },
            True,
        ),
        ({"message": {"content": "not JSON"}}, False),
    ],
)
def test_token_usage_distinguishes_zero_from_unavailable(response_payload, usage_available) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=response_payload))
    with httpx.Client(transport=transport) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=0)
        with pytest.raises(ModelOutputValidationError):
            provider.decide(_context())

    assert provider.telemetry.token_usage == 0
    assert provider.telemetry.token_usage_available is usage_available


def test_tentative_hypothesis_with_fabricated_evidence_is_corrected() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        decision = _tool_decision()
        if attempts == 1:
            decision["hypothesis_operations"][0]["supporting_evidence"] = ["EV-NOT-YET-CREATED"]
        else:
            correction = json.loads(request.content)["messages"][-1]["content"]
            assert "EV-NOT-YET-CREATED" in correction
            assert "Available evidence IDs: none" in correction
            assert "Evidence arrays accept bare IDs only" in correction
        return httpx.Response(200, json=_ollama_response(decision))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=1)
        decision = provider.decide(_context())

    assert decision.action == "tool"
    assert decision.hypothesis_operations[0].supporting_evidence == []
    assert provider.telemetry.validation_failures == 1
    assert provider.telemetry.first_attempt_schema_valid_decisions == 1
    assert provider.telemetry.retry_recoveries == 1
    rejection = provider.telemetry.rejected_decisions[0]
    assert rejection.error_type == "ModelOutputValidationError"
    assert "unavailable evidence IDs" in rejection.error
    assert rejection.decision["hypothesis_operations"][0]["supporting_evidence"] == [
        "EV-NOT-YET-CREATED"
    ]


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


def test_invalid_check_id_gets_available_values_in_retry_feedback() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        decision = _tool_decision("sample_problematic_records", limit=5)
        if attempts == 1:
            decision["tool_request"]["check_id"] = "H-1"
        else:
            correction = json.loads(request.content)["messages"][-1]["content"]
            assert "check_id must be one of the available values" in correction
            assert "Valid check_id values: schema_contract" in correction
            decision["tool_request"]["check_id"] = "schema_contract"
        return httpx.Response(200, json=_ollama_response(decision))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=1)
        decision = provider.decide(_context())

    assert decision.tool_request.arguments["check_id"] == "schema_contract"
    assert provider.telemetry.invalid_tool_calls == 1
    assert provider.telemetry.first_attempt_schema_valid_decisions == 1
    assert provider.telemetry.retry_recoveries == 1


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
        decision = provider.decide(
            _context(
                [evidence],
                [RootCauseHypothesis("H-1", "The upstream schema changed", "schema_change")],
            )
        )

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
    decision["hypothesis_operations"][0]["diagnosis_code"] = "database_hack"
    decision["conclusion"]["diagnosis_code"] = "database_hack"
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=_ollama_response(decision)))
    with httpx.Client(transport=transport) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=0)
        with pytest.raises(ModelOutputValidationError, match="invalid structured decision"):
            provider.decide(
                _context(
                    [evidence],
                    [RootCauseHypothesis("H-1", "The upstream schema changed", "schema_change")],
                )
            )


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
