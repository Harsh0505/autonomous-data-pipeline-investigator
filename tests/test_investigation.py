from __future__ import annotations

from datetime import timedelta

import pytest

from retail_guardian.failures import FailureType
from retail_guardian.investigation import InvestigationAgent
from retail_guardian.investigation_models import (
    ConfidenceLevel,
    HypothesisStatus,
    InvestigationConclusion,
    ModelDecision,
    RootCauseHypothesis,
    ToolRequest,
)
from retail_guardian.providers import HeuristicInvestigationModel
from retail_guardian.scenarios import run_scenario
from retail_guardian.storage import SQLiteHistoryRepository


def _persist_case(tmp_path, as_of, scenario: str):
    repository = SQLiteHistoryRepository(tmp_path / f"{scenario}.db")
    run_scenario("healthy", as_of=as_of, repository=repository, ingested_at=as_of)
    incident = run_scenario(
        scenario,
        as_of=as_of,
        repository=repository,
        ingested_at=as_of + timedelta(seconds=1),
    )
    return repository, incident


@pytest.mark.parametrize("failure_type", list(FailureType))
def test_mock_agent_investigates_each_failure(tmp_path, as_of, failure_type) -> None:
    repository, incident = _persist_case(tmp_path, as_of, failure_type.value)

    report = InvestigationAgent(repository, HeuristicInvestigationModel()).investigate(
        incident.report.incident_id
    )

    assert report.status == "completed"
    assert report.conclusion.diagnosis_code == failure_type.value
    assert report.conclusion.evidence_references
    assert set(report.conclusion.evidence_references) <= {
        item.evidence_id for item in report.evidence
    }
    assert report.tool_call_count == 2
    assert report.tools_used[0] == "get_validation_failures"


def test_repeated_investigations_are_independently_persisted(tmp_path, as_of) -> None:
    repository, incident = _persist_case(tmp_path, as_of, "duplicate_records")
    agent = InvestigationAgent(repository, HeuristicInvestigationModel())

    first = agent.investigate(incident.report.incident_id)
    second = agent.investigate(incident.report.incident_id)
    persisted = repository.get_investigations(incident.report.incident_id)

    assert first.investigation_id != second.investigation_id
    assert len(persisted) == 2
    assert all(item["status"] == "completed" for item in persisted)


class _EndlessToolModel:
    def decide(self, context):
        tool_name = (
            context.available_tools[0].name if context.available_tools else "inspect_dataset"
        )
        return ModelDecision(
            "tool",
            [RootCauseHypothesis("H-1", "More evidence is needed", None)],
            ToolRequest(tool_name, {}, "Continue collecting evidence"),
        )


def test_tool_budget_forces_termination(tmp_path, as_of) -> None:
    repository, incident = _persist_case(tmp_path, as_of, "missing_values")

    report = InvestigationAgent(repository, _EndlessToolModel(), max_tool_calls=2).investigate(
        incident.report.incident_id
    )

    assert report.status == "inconclusive"
    assert report.tool_call_count == 2
    assert report.successful_tool_calls == 2
    assert report.conclusion.confidence == ConfidenceLevel.INSUFFICIENT
    assert "budget" in report.conclusion.final_diagnosis


class _EmptyToolModel:
    def decide(self, context):
        requests = [
            ToolRequest(
                "sample_problematic_records",
                {"check_id": "schema_contract", "limit": 5},
                "Look for affected records",
            ),
            ToolRequest(
                "identify_affected_dimensions",
                {"check_id": "schema_contract"},
                "Look for affected dimensions",
            ),
            ToolRequest("inspect_dataset", {}, "Try another tool"),
        ]
        request = requests[min(len(context.evidence), len(requests) - 1)]
        return ModelDecision(
            "tool",
            [RootCauseHypothesis("H-1", "More evidence may be needed", None)],
            request,
        )


def test_empty_results_have_a_separate_bounded_budget(tmp_path, as_of) -> None:
    repository, incident = _persist_case(tmp_path, as_of, "schema_change")

    report = InvestigationAgent(
        repository,
        _EmptyToolModel(),
        max_tool_calls=3,
        max_empty_tool_calls=2,
    ).investigate(incident.report.incident_id)

    assert report.status == "inconclusive"
    assert report.tool_call_count == 2
    assert report.informative_tool_calls == 0
    assert report.empty_tool_calls == 2
    assert "empty tool-result limit" in report.conclusion.final_diagnosis


class _EvidenceIntegratingModel:
    def __init__(self) -> None:
        self.contexts = []

    def decide(self, context):
        self.contexts.append(context)
        if not context.evidence:
            return ModelDecision(
                "tool",
                [RootCauseHypothesis("H-1", "A required value may be absent", "missing_values")],
                ToolRequest("get_validation_failures", {}, "Inspect measured failures"),
            )
        evidence_id = context.evidence[0].evidence_id
        return ModelDecision(
            "conclude",
            [
                RootCauseHypothesis(
                    "H-1",
                    "A required value may be absent",
                    "missing_values",
                    status=HypothesisStatus.CONFIRMED,
                    supporting_evidence=[evidence_id],
                ),
                RootCauseHypothesis(
                    "H-2",
                    "The schema may have changed",
                    "schema_change",
                    status=HypothesisStatus.REJECTED,
                    contradicting_evidence=[evidence_id],
                ),
            ],
            conclusion=InvestigationConclusion(
                "missing_values",
                "Required sale values are missing",
                ConfidenceLevel.HIGH,
                [evidence_id],
                [],
                ["Request a corrected supplier feed"],
            ),
        )


def test_completed_tools_and_evidence_are_available_for_hypothesis_updates(tmp_path, as_of) -> None:
    repository, incident = _persist_case(tmp_path, as_of, "missing_values")
    model = _EvidenceIntegratingModel()

    report = InvestigationAgent(repository, model).investigate(incident.report.incident_id)

    follow_up = model.contexts[1]
    assert report.status == "completed"
    assert report.successful_tool_calls == 1
    assert follow_up.completed_tool_calls[0].name == "get_validation_failures"
    assert follow_up.completed_tool_calls[0].evidence_id == follow_up.evidence[0].evidence_id
    assert "get_validation_failures" not in {tool.name for tool in follow_up.available_tools}
    assert {item.status.value for item in report.hypotheses} == {"confirmed", "rejected"}


class _InsufficientEvidenceModel:
    def decide(self, context):
        if not context.evidence:
            return ModelDecision(
                "tool",
                [RootCauseHypothesis("H-1", "Historical context may identify a cause", None)],
                ToolRequest("get_pipeline_history", {"limit": 10}, "Check available history"),
            )
        return ModelDecision(
            "conclude",
            context.hypotheses,
            conclusion=InvestigationConclusion(
                None,
                "The available history does not establish a root cause",
                ConfidenceLevel.INSUFFICIENT,
                [],
                ["Which upstream event preceded this execution?"],
                ["Collect upstream delivery logs"],
            ),
        )


def test_model_can_terminate_with_genuinely_insufficient_evidence(tmp_path, as_of) -> None:
    repository, incident = _persist_case(tmp_path, as_of, "missing_values")

    report = InvestigationAgent(repository, _InsufficientEvidenceModel()).investigate(
        incident.report.incident_id
    )

    assert report.status == "inconclusive"
    assert report.tool_call_count == 1
    assert report.successful_tool_calls == 1
    assert report.conclusion.diagnosis_code is None
    assert "does not establish" in report.conclusion.final_diagnosis


class _FailedToolModel:
    def decide(self, context):
        if not context.evidence:
            return ModelDecision(
                "tool",
                [],
                ToolRequest("run_sql", {"query": "select *"}, "Try an unavailable tool"),
            )
        return ModelDecision(
            "conclude",
            [],
            conclusion=InvestigationConclusion(
                None,
                "The failed tool left the evidence insufficient",
                ConfidenceLevel.INSUFFICIENT,
                [],
                ["Can the required evidence be collected through an approved tool?"],
                ["Use a scoped investigation tool"],
            ),
        )


def test_failed_tool_is_recorded_without_crashing(tmp_path, as_of) -> None:
    repository, incident = _persist_case(tmp_path, as_of, "missing_values")

    report = InvestigationAgent(repository, _FailedToolModel()).investigate(
        incident.report.incident_id
    )

    assert report.status == "inconclusive"
    assert len(report.evidence) == 1
    assert not report.evidence[0].success
    assert report.evidence[0].error


class _UnsupportedConclusionModel:
    def decide(self, context):
        return ModelDecision(
            "conclude",
            [],
            conclusion=InvestigationConclusion(
                "schema_change",
                "The schema changed",
                ConfidenceLevel.HIGH,
                ["EV-MADE-UP"],
                [],
                ["Update the schema"],
            ),
        )


def test_unsupported_diagnosis_is_downgraded_to_inconclusive(tmp_path, as_of) -> None:
    repository, incident = _persist_case(tmp_path, as_of, "schema_change")

    report = InvestigationAgent(repository, _UnsupportedConclusionModel()).investigate(
        incident.report.incident_id
    )

    assert report.status == "inconclusive"
    assert report.conclusion.diagnosis_code is None
    assert report.conclusion.confidence == ConfidenceLevel.INSUFFICIENT


class _UnconfirmedConclusionModel:
    def decide(self, context):
        if not context.evidence:
            return ModelDecision(
                "tool",
                [RootCauseHypothesis("H-1", "The schema changed", "schema_change")],
                ToolRequest("get_validation_failures", {}, "Retrieve observed failures"),
            )
        evidence_id = context.evidence[0].evidence_id
        return ModelDecision(
            "conclude",
            [RootCauseHypothesis("H-1", "The schema changed", "schema_change")],
            conclusion=InvestigationConclusion(
                "schema_change",
                "The schema changed",
                ConfidenceLevel.HIGH,
                [evidence_id],
                [],
                [],
            ),
        )


def test_conclusion_requires_a_confirmed_supported_hypothesis(tmp_path, as_of) -> None:
    repository, incident = _persist_case(tmp_path, as_of, "schema_change")

    report = InvestigationAgent(repository, _UnconfirmedConclusionModel()).investigate(
        incident.report.incident_id
    )

    assert report.status == "inconclusive"
    assert report.conclusion.diagnosis_code is None
    assert "confirmed hypothesis" in report.conclusion.final_diagnosis
