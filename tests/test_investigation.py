from __future__ import annotations

from datetime import timedelta

import pytest

from retail_guardian.failures import FailureType
from retail_guardian.investigation import InvestigationAgent
from retail_guardian.investigation_models import (
    ConfidenceLevel,
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
        return ModelDecision(
            "tool",
            [RootCauseHypothesis("H-1", "More evidence is needed", None)],
            ToolRequest("inspect_dataset", {}, "Continue collecting evidence"),
        )


def test_tool_budget_forces_termination(tmp_path, as_of) -> None:
    repository, incident = _persist_case(tmp_path, as_of, "missing_values")

    report = InvestigationAgent(repository, _EndlessToolModel(), max_tool_calls=2).investigate(
        incident.report.incident_id
    )

    assert report.status == "inconclusive"
    assert report.tool_call_count == 2
    assert report.conclusion.confidence == ConfidenceLevel.INSUFFICIENT
    assert "budget" in report.conclusion.final_diagnosis


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
