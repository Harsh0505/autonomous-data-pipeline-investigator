from __future__ import annotations

import json
from datetime import timedelta

import httpx

from retail_guardian.causal_evidence import has_inconclusive_audit_evidence
from retail_guardian.causal_scenarios import run_root_cause_scenario
from retail_guardian.investigation import InvestigationAgent
from retail_guardian.providers import OllamaInvestigationModel
from retail_guardian.scenarios import run_scenario
from retail_guardian.storage import SQLiteHistoryRepository


def _tool_decision() -> dict:
    return {
        "action": "tool",
        "hypothesis_operations": [
            {
                "operation": "create",
                "hypothesis_id": "H-1",
                "statement": "Validation evidence is needed",
                "diagnosis_code": None,
                "supporting_evidence": [],
                "contradicting_evidence": [],
            }
        ],
        "tool_request": {
            "name": "get_validation_failures",
            "rationale": "Inspect measured validation failures",
            "question_id": "Q-validation",
            "question": "Which validation checks failed?",
            "hypothesis_ids": [],
        },
        "conclusion": None,
    }


def _conclusion(evidence_id: str) -> dict:
    return {
        "action": "conclude",
        "hypothesis_operations": [
            {
                "operation": "confirm",
                "hypothesis_id": "H-1",
                "statement": "The feed contains duplicate order records",
                "diagnosis_code": "duplicate_records",
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


def _ollama_response(decision: dict, *, prompt_tokens: int = 20, output_tokens: int = 10) -> dict:
    return {
        "message": {"content": json.dumps(decision)},
        "prompt_eval_count": prompt_tokens,
        "eval_count": output_tokens,
    }


def _causal_operation(
    operation: str,
    hypothesis_id: str,
    diagnosis_code: str,
    *,
    statement: str | None = None,
    supporting: list[str] | None = None,
    contradicting: list[str] | None = None,
) -> dict:
    return {
        "operation": operation,
        "hypothesis_id": hypothesis_id,
        "statement": statement,
        "diagnosis_code": diagnosis_code,
        "supporting_evidence": supporting or [],
        "contradicting_evidence": contradicting or [],
    }


def _causal_tool(
    name: str,
    operations: list[dict] | None = None,
    *,
    question_id: str | None = None,
    hypothesis_ids: list[str] | None = None,
) -> dict:
    return {
        "action": "tool",
        "hypothesis_operations": operations or [],
        "tool_request": {
            "name": name,
            "rationale": f"Investigate with {name}",
            "question_id": question_id or f"Q-{name}",
            "question": f"What causal evidence can {name} provide?",
            "hypothesis_ids": hypothesis_ids or [],
        },
        "conclusion": None,
    }


def _causal_conclusion(
    diagnosis_code: str | None,
    confidence: str,
    evidence_references: list[str],
    operations: list[dict] | None = None,
) -> dict:
    return {
        "action": "conclude",
        "hypothesis_operations": operations or [],
        "tool_request": None,
        "conclusion": {
            "diagnosis_code": diagnosis_code,
            "final_diagnosis": (
                "A failed execution retried after committing rows"
                if diagnosis_code == "failed_job_retry"
                else "Operational audit evidence is unavailable"
            ),
            "confidence": confidence,
            "evidence_references": evidence_references,
            "outstanding_questions": [],
            "recommended_actions": [],
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
            decision["hypothesis_operations"][0]["supporting_evidence"] = ["EV-PENDING"]
        elif calls == 2:
            decision = _tool_decision()
        else:
            request_payload = json.loads(request.content)
            context = json.loads(request_payload["messages"][1]["content"])
            evidence_id = next(iter(context["evidence_registry"]))
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
    assert report.first_attempt_schema_valid_decisions == 2
    assert report.retry_recoveries == 1
    assert trace is not None
    assert len(trace["decisions"]) == 2
    assert len(trace["tool_calls"]) == 1
    assert len(trace["rejected_model_decisions"]) == 1
    rejected = trace["rejected_model_decisions"][0]
    assert "unavailable evidence IDs" in rejected["error"]
    assert rejected["decision"]["hypothesis_operations"][0]["supporting_evidence"] == ["EV-PENDING"]
    assert trace["evidence"][0]["evidence_id"] in report.conclusion.evidence_references
    assert trace["evidence"][0]["result_kind"] == "informative"


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


def test_forced_tool_recovery_preserves_hypotheses_then_resolves_retry(tmp_path, as_of) -> None:
    repository = SQLiteHistoryRepository(tmp_path / "retry.db")
    scenario = run_root_cause_scenario(
        "case-002", seed=42, as_of=as_of, repository=repository, ingested_at=as_of
    )
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content)
        context = json.loads(payload["messages"][1]["content"])
        registry = context["evidence_registry"]
        if calls == 1:
            decision = _causal_tool(
                "get_supplier_submissions",
                [
                    _causal_operation(
                        "create",
                        "H-replay",
                        "supplier_batch_replay",
                        statement="The supplier may have replayed the batch",
                    ),
                    _causal_operation(
                        "create",
                        "H-retry",
                        "failed_job_retry",
                        statement="A failed job may have repeated a durable write",
                    ),
                ],
            )
        elif calls == 2:
            supplier_id = next(iter(registry))
            decision = _causal_conclusion(
                "supplier_batch_replay",
                "high",
                [supplier_id],
                [
                    _causal_operation(
                        "confirm",
                        "H-replay",
                        "supplier_batch_replay",
                        supporting=[supplier_id],
                    )
                ],
            )
        elif calls == 3:
            tool_schema = payload["format"]["oneOf"][0]
            assert tool_schema["properties"]["hypothesis_operations"]["maxItems"] == 0
            assert [item["hypothesis_id"] for item in context["current_hypotheses"]] == [
                "H-replay",
                "H-retry",
            ]
            assert all(
                not item["supporting_evidence"]
                and not item["contradicting_evidence"]
                and item["status"] == "active"
                for item in context["current_hypotheses"]
            )
            decision = _causal_tool("get_orchestration_events")
        elif calls == 4:
            evidence_by_tool = {item["tool_name"]: key for key, item in registry.items()}
            memory = context["investigation_memory"]
            assert (
                "maxItems"
                not in payload["format"]["oneOf"][0]["properties"]["hypothesis_operations"]
            )
            assert all(
                not item["supporting_evidence"] and not item["contradicting_evidence"]
                for item in context["current_hypotheses"]
            )
            assert [item["status"] for item in memory["questions_investigated"]] == [
                "answered",
                "answered",
            ]
            replay_check = next(
                item for item in memory["hypothesis_checks"] if item["hypothesis_id"] == "H-replay"
            )
            assert replay_check["contradicting_evidence"] == [
                evidence_by_tool["get_supplier_submissions"]
            ]
            assert replay_check["information_needed"]
            decision = _causal_tool(
                "get_pipeline_logs",
                [
                    _causal_operation(
                        "update",
                        "H-replay",
                        "supplier_batch_replay",
                        contradicting=[evidence_by_tool["get_supplier_submissions"]],
                    ),
                    _causal_operation(
                        "update",
                        "H-retry",
                        "failed_job_retry",
                        supporting=[evidence_by_tool["get_orchestration_events"]],
                    ),
                ],
            )
        else:
            evidence_by_tool = {item["tool_name"]: key for key, item in registry.items()}
            support = [
                evidence_by_tool["get_orchestration_events"],
                evidence_by_tool["get_pipeline_logs"],
            ]
            decision = _causal_conclusion(
                "failed_job_retry",
                "high",
                support,
                [
                    _causal_operation(
                        "confirm",
                        "H-retry",
                        "failed_job_retry",
                        supporting=[evidence_by_tool["get_pipeline_logs"]],
                    )
                ],
            )
        return httpx.Response(200, json=_ollama_response(decision))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=2)
        report = InvestigationAgent(repository, provider).investigate(
            scenario.pipeline_run.report.incident_id
        )

    trace = repository.get_investigation_trace(report.investigation_id)
    hypotheses = {item.hypothesis_id: item for item in report.hypotheses}
    assert report.status == "completed"
    assert report.conclusion.diagnosis_code == "failed_job_retry"
    assert report.tools_used == [
        "get_supplier_submissions",
        "get_orchestration_events",
        "get_pipeline_logs",
    ]
    assert hypotheses["H-replay"].contradicting_evidence
    assert hypotheses["H-retry"].status.value == "confirmed"
    assert [item.question_id for item in report.investigation_questions] == [
        "Q-get_supplier_submissions",
        "Q-get_orchestration_events",
        "Q-get_pipeline_logs",
    ]
    assert provider.telemetry.validation_failures == 1
    assert provider.telemetry.retry_recoveries == 1
    assert report.token_usage == 150
    assert len(trace["rejected_model_decisions"]) == 1


def test_forced_tool_recovery_can_establish_genuine_insufficient_evidence(tmp_path, as_of) -> None:
    repository = SQLiteHistoryRepository(tmp_path / "inconclusive.db")
    scenario = run_root_cause_scenario(
        "case-010", seed=42, as_of=as_of, repository=repository, ingested_at=as_of
    )
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content)
        context = json.loads(payload["messages"][1]["content"])
        registry = context["evidence_registry"]
        if calls == 1:
            decision = _causal_tool(
                "get_supplier_submissions",
                [
                    _causal_operation(
                        "create",
                        "H-replay",
                        "supplier_batch_replay",
                        statement="The supplier may have replayed the batch",
                    )
                ],
            )
        elif calls == 2:
            supplier_id = next(iter(registry))
            decision = _causal_conclusion(
                "supplier_batch_replay",
                "high",
                [supplier_id],
                [
                    _causal_operation(
                        "confirm",
                        "H-replay",
                        "supplier_batch_replay",
                        supporting=[supplier_id],
                    )
                ],
            )
        elif calls == 3:
            assert (
                payload["format"]["oneOf"][0]["properties"]["hypothesis_operations"]["maxItems"]
                == 0
            )
            decision = _causal_tool("get_orchestration_events")
        elif calls == 4:
            decision = _causal_tool("get_transformation_metadata")
        elif calls == 5:
            decision = _causal_tool("get_pipeline_logs")
        elif calls == 6:
            decision = _causal_tool("get_configuration_changes")
        else:
            decision = _causal_conclusion(
                None,
                "insufficient_evidence",
                list(registry),
            )
        return httpx.Response(200, json=_ollama_response(decision))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=2)
        report = InvestigationAgent(repository, provider).investigate(
            scenario.pipeline_run.report.incident_id
        )

    assert report.status == "inconclusive"
    assert report.conclusion.diagnosis_code is None
    assert report.tools_used == [
        "get_supplier_submissions",
        "get_orchestration_events",
        "get_transformation_metadata",
        "get_pipeline_logs",
        "get_configuration_changes",
    ]
    assert has_inconclusive_audit_evidence(report.evidence)
    question_statuses = {
        item.tool_name: item.status.value for item in report.investigation_questions
    }
    assert question_statuses["get_orchestration_events"] == "unavailable"
    assert question_statuses["get_transformation_metadata"] == "unavailable"
    assert report.hypotheses[0].status.value == "active"
    assert not report.hypotheses[0].supporting_evidence
    assert provider.telemetry.validation_failures == 1
    assert provider.telemetry.retry_recoveries == 1
    assert report.token_usage == 210


def test_failed_provider_decision_reports_consumed_tokens(tmp_path, as_of) -> None:
    repository = SQLiteHistoryRepository(tmp_path / "failed-usage.db")
    incident = run_scenario("missing_values", as_of=as_of, repository=repository, ingested_at=as_of)

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {"content": "not JSON"},
                "prompt_eval_count": 5,
                "eval_count": 2,
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaInvestigationModel(client=client, max_retries=1)
        report = InvestigationAgent(repository, provider).investigate(incident.report.incident_id)

    assert report.status == "failed"
    assert report.model_validation_failures == 2
    assert report.token_usage == 14


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

    report = InvestigationAgent(
        repository,
        _DuplicateToolModel(),
        max_tool_calls=2,
        max_invalid_tool_calls=1,
        max_duplicate_tool_calls=1,
    ).investigate(incident.report.incident_id)

    assert report.status == "failed"
    assert report.tool_call_count == 2
    assert report.invalid_tool_calls == 0
    assert report.duplicate_tool_calls == 1
    assert report.successful_tool_calls == 1
    assert report.evidence[-1].error == "duplicate tool request"
    assert "duplicate tool-request limit" in report.conclusion.final_diagnosis


class _EquivalentDuplicateToolModel:
    def __init__(self) -> None:
        from retail_guardian.investigation_models import ProviderTelemetry

        self.telemetry = ProviderTelemetry("test", "equivalent-duplicate")

    def decide(self, context):
        from retail_guardian.investigation_models import ModelDecision, ToolRequest

        if context.completed_tool_calls:
            assert "get_pipeline_history" not in {
                definition.name for definition in context.available_tools
            }
        arguments = {} if not context.completed_tool_calls else {"limit": 10}
        return ModelDecision(
            "tool",
            [],
            ToolRequest(
                "get_pipeline_history",
                arguments,
                "Inspect prior executions",
            ),
        )


def test_semantically_equivalent_default_arguments_are_duplicates(tmp_path, as_of) -> None:
    repository = SQLiteHistoryRepository(tmp_path / "equivalent-duplicate.db")
    incident = run_scenario("missing_values", as_of=as_of, repository=repository, ingested_at=as_of)

    report = InvestigationAgent(
        repository,
        _EquivalentDuplicateToolModel(),
        max_duplicate_tool_calls=1,
    ).investigate(incident.report.incident_id)

    assert report.status == "failed"
    assert report.tool_call_count == 2
    assert report.duplicate_tool_calls == 1
    assert report.informative_tool_calls == 0
    assert report.empty_tool_calls == 1
    assert report.investigation_questions[-1].status.value == "failed"
