from __future__ import annotations

import json
from datetime import timedelta

import httpx

from retail_guardian.investigation import InvestigationAgent
from retail_guardian.providers import OllamaInvestigationModel
from retail_guardian.scenarios import run_scenario
from retail_guardian.storage import SQLiteHistoryRepository


def _tool_decision() -> dict:
    return {
        "action": "tool",
        "hypotheses": [
            {
                "hypothesis_id": "H-1",
                "statement": "Validation evidence is needed",
                "diagnosis_code": None,
                "status": "active",
                "supporting_evidence": [],
                "contradicting_evidence": [],
            }
        ],
        "tool_request": {
            "name": "get_validation_failures",
            "rationale": "Inspect measured validation failures",
        },
        "conclusion": None,
    }


def _conclusion(evidence_id: str) -> dict:
    return {
        "action": "conclude",
        "hypotheses": [
            {
                "hypothesis_id": "H-1",
                "statement": "The feed contains duplicate order records",
                "diagnosis_code": "duplicate_records",
                "status": "confirmed",
                "supporting_evidence": [evidence_id],
                "contradicting_evidence": [],
            }
        ],
        "tool_request": None,
        "conclusion": {
            "diagnosis_code": "duplicate_records",
            "final_diagnosis": "The feed contains duplicate order records",
            "confidence": "high",
            "evidence_references": [evidence_id],
            "outstanding_questions": [],
            "recommended_actions": ["Audit upstream retry semantics"],
        },
    }


def test_ollama_drives_graph_and_persists_full_trace(tmp_path, as_of) -> None:
    repository = SQLiteHistoryRepository(tmp_path / "history.db")
    run_scenario("healthy", as_of=as_of, repository=repository, ingested_at=as_of)
    incident = run_scenario(
        "duplicate_records",
        as_of=as_of,
        repository=repository,
        ingested_at=as_of + timedelta(seconds=1),
    )
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            decision = _tool_decision()
            decision["hypotheses"][0]["supporting_evidence"] = ["EV-PENDING"]
        elif calls == 2:
            decision = _tool_decision()
        else:
            request_payload = json.loads(request.content)
            context = json.loads(request_payload["messages"][1]["content"])
            evidence_id = context["evidence"][0]["evidence_id"]
            decision = _conclusion(evidence_id)
        return httpx.Response(
            200,
            json={
                "message": {"content": json.dumps(decision)},
                "prompt_eval_count": 20,
                "eval_count": 10,
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=1)
        report = InvestigationAgent(repository, provider).investigate(incident.report.incident_id)

    trace = repository.get_investigation_trace(report.investigation_id)
    assert report.status == "completed"
    assert report.provider_name == "ollama"
    assert report.model_name == "qwen3:4b"
    assert report.conclusion.diagnosis_code == "duplicate_records"
    assert report.token_usage == 90
    assert report.decision_count == 2
    assert report.invalid_tool_calls == 0
    assert report.model_validation_failures == 1
    assert trace is not None
    assert len(trace["decisions"]) == 2
    assert len(trace["tool_calls"]) == 1
    assert len(trace["rejected_model_decisions"]) == 1
    rejected = trace["rejected_model_decisions"][0]
    assert "unavailable evidence IDs" in rejected["error"]
    assert rejected["decision"]["hypotheses"][0]["supporting_evidence"] == ["EV-PENDING"]
    assert trace["evidence"][0]["evidence_id"] in report.conclusion.evidence_references


def test_ollama_connection_failure_produces_failed_report(tmp_path, as_of) -> None:
    repository = SQLiteHistoryRepository(tmp_path / "history.db")
    incident = run_scenario("missing_values", as_of=as_of, repository=repository, ingested_at=as_of)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=0)
        report = InvestigationAgent(repository, provider).investigate(incident.report.incident_id)

    assert report.status == "failed"
    assert report.investigation_failures == 1
    assert report.conclusion.diagnosis_code is None
    assert "Ollama is unavailable" in report.conclusion.final_diagnosis


class _DuplicateToolModel:
    def __init__(self) -> None:
        from retail_guardian.investigation_models import ProviderTelemetry

        self.telemetry = ProviderTelemetry("test", "duplicate-tool")

    def decide(self, context):
        from retail_guardian.investigation_models import (
            ModelDecision,
            RootCauseHypothesis,
            ToolRequest,
        )

        return ModelDecision(
            "tool",
            [RootCauseHypothesis("H-1", "Inspect repeatedly", None)],
            ToolRequest("inspect_dataset", {}, "Request the same evidence"),
        )


def test_duplicate_tools_are_rejected_and_budget_terminates(tmp_path, as_of) -> None:
    repository = SQLiteHistoryRepository(tmp_path / "history.db")
    incident = run_scenario("missing_values", as_of=as_of, repository=repository, ingested_at=as_of)

    report = InvestigationAgent(repository, _DuplicateToolModel(), max_tool_calls=2).investigate(
        incident.report.incident_id
    )

    assert report.status == "inconclusive"
    assert report.tool_call_count == 2
    assert report.invalid_tool_calls == 1
    assert report.evidence[-1].error == "duplicate tool request"
