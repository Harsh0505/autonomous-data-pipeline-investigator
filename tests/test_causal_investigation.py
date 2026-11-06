from __future__ import annotations

import json
from datetime import datetime

import pytest

from retail_guardian.causal_evidence import (
    CausalEvidenceError,
    assess_causal_evidence,
    has_inconclusive_audit_evidence,
    require_causal_evidence_alignment,
)
from retail_guardian.causal_scenarios import ROOT_CAUSE_CASES, run_root_cause_scenario
from retail_guardian.investigation import InvestigationAgent
from retail_guardian.investigation_models import (
    ConfidenceLevel,
    HypothesisOperation,
    HypothesisOperationType,
    InvestigationConclusion,
    ModelDecision,
    ProviderTelemetry,
    ToolRequest,
    ToolResultKind,
)
from retail_guardian.investigation_tools import InvestigationToolbox
from retail_guardian.root_cause_evaluation import RootCauseBenchmarkEvaluator
from retail_guardian.storage import SQLiteHistoryRepository

OPERATIONAL_TOOLS = (
    "get_pipeline_logs",
    "get_supplier_submissions",
    "get_orchestration_events",
    "get_transformation_metadata",
    "get_configuration_changes",
)


@pytest.mark.parametrize("case_id", ROOT_CAUSE_CASES)
def test_root_cause_cases_are_reproducible_and_do_not_persist_truth(
    tmp_path, as_of, case_id
) -> None:
    first_repository = SQLiteHistoryRepository(tmp_path / f"{case_id}-first.db")
    second_repository = SQLiteHistoryRepository(tmp_path / f"{case_id}-second.db")
    first = run_root_cause_scenario(
        case_id, seed=91, as_of=as_of, repository=first_repository, ingested_at=as_of
    )
    second = run_root_cause_scenario(
        case_id, seed=91, as_of=as_of, repository=second_repository, ingested_at=as_of
    )

    execution_id = first.pipeline_run.execution_id
    first_records = first_repository.get_operational_records(execution_id)
    second_records = second_repository.get_operational_records(second.pipeline_run.execution_id)
    persisted = json.dumps(
        {
            "incident": first_repository.get_incident(first.pipeline_run.report.incident_id),
            "records": first_records,
        },
        sort_keys=True,
    )

    assert first.pipeline_run.report.status == "incident"
    assert first.pipeline_run.report.to_dict() == second.pipeline_run.report.to_dict()
    assert first_records == second_records
    assert case_id not in persisted
    if first.ground_truth.root_cause is not None:
        assert first.ground_truth.root_cause.value not in persisted


@pytest.mark.parametrize(
    ("case_ids", "failed_check"),
    [
        (("case-001", "case-002", "case-003"), "unique_order_id"),
        (("case-004", "case-005"), "required_values"),
        (("case-006", "case-007"), "known_category"),
        (("case-008", "case-009"), "delivery_freshness"),
    ],
)
def test_competing_causes_share_the_same_observable_failure(
    tmp_path, as_of, case_ids, failed_check
) -> None:
    observed: list[set[str]] = []
    roots: set[str] = set()
    for index, case_id in enumerate(case_ids):
        repository = SQLiteHistoryRepository(tmp_path / f"shared-{index}.db")
        scenario = run_root_cause_scenario(
            case_id,
            seed=40 + index,
            as_of=as_of,
            repository=repository,
            ingested_at=as_of,
        )
        results = repository.get_validation_results(scenario.pipeline_run.execution_id)
        observed.append({item["check_id"] for item in results if not item["passed"]})
        roots.add(scenario.ground_truth.root_cause.value)

    assert all(failed_check in failures for failures in observed)
    assert len(roots) == len(case_ids)


def test_operational_tools_are_scoped_typed_and_causal(tmp_path, as_of) -> None:
    repository = SQLiteHistoryRepository(tmp_path / "tools.db")
    scenario = run_root_cause_scenario(
        "case-005", seed=18, as_of=as_of, repository=repository, ingested_at=as_of
    )
    toolbox = InvestigationToolbox(repository, scenario.pipeline_run.report.incident_id)

    supplier = toolbox.execute("get_supplier_submissions")
    transformation = toolbox.execute("get_transformation_metadata")
    invalid = toolbox.execute("get_pipeline_logs", {"query": "all"})

    assert toolbox.investigation_objective == "identify_underlying_root_cause"
    assert "transformation_logic_error" in toolbox.allowed_diagnosis_codes
    assert "missing_values" not in toolbox.allowed_diagnosis_codes
    assert supplier.result_kind == ToolResultKind.INFORMATIVE
    assert supplier.data["records"][0]["details"]["field_statistics"]["unit_price"] == {
        "null_count": 0
    }
    assert transformation.data["records"][0]["details"]["conversion_errors"] == 5
    assert invalid.result_kind == ToolResultKind.INVALID
    assert "unsupported arguments" in invalid.error


def test_diagnosis_taxonomy_order_is_not_fixed_across_incidents(tmp_path, as_of) -> None:
    orders: list[tuple[str, ...]] = []
    for index, case_id in enumerate(ROOT_CAUSE_CASES[:4]):
        repository = SQLiteHistoryRepository(tmp_path / f"taxonomy-{case_id}.db")
        scenario = run_root_cause_scenario(
            case_id,
            seed=120 + index,
            as_of=as_of,
            repository=repository,
            ingested_at=as_of,
        )
        toolbox = InvestigationToolbox(repository, scenario.pipeline_run.report.incident_id)
        orders.append(tuple(toolbox.allowed_diagnosis_codes))

    assert all(set(order) == set(orders[0]) for order in orders)
    assert len(set(orders)) > 1


@pytest.mark.parametrize("case_id", ROOT_CAUSE_CASES[:-1])
def test_selected_root_cause_requires_its_actual_causal_facts(tmp_path, as_of, case_id) -> None:
    repository = SQLiteHistoryRepository(tmp_path / f"facts-{case_id}.db")
    scenario = run_root_cause_scenario(
        case_id, seed=64, as_of=as_of, repository=repository, ingested_at=as_of
    )
    toolbox = InvestigationToolbox(repository, scenario.pipeline_run.report.incident_id)
    evidence = [toolbox.execute(name) for name in OPERATIONAL_TOOLS]
    references = [item.evidence_id for item in evidence if item.success]

    assessment = assess_causal_evidence(
        scenario.ground_truth.root_cause.value,
        references,
        evidence,
    )

    assert assessment.supported
    assert assessment.supporting_evidence
    assert not assessment.missing_requirements


def test_generic_validation_evidence_cannot_confirm_a_causal_mechanism(tmp_path, as_of) -> None:
    repository = SQLiteHistoryRepository(tmp_path / "generic-evidence.db")
    scenario = run_root_cause_scenario(
        "case-001", seed=14, as_of=as_of, repository=repository, ingested_at=as_of
    )
    toolbox = InvestigationToolbox(repository, scenario.pipeline_run.report.incident_id)
    evidence = toolbox.execute("get_validation_failures")

    assessment = assess_causal_evidence("supplier_batch_replay", [evidence.evidence_id], [evidence])

    assert not assessment.supported
    assert assessment.supporting_evidence == ()
    assert assessment.missing_requirements


def test_inconclusive_case_records_counterevidence_and_missing_audit_data(tmp_path, as_of) -> None:
    repository = SQLiteHistoryRepository(tmp_path / "inconclusive-facts.db")
    scenario = run_root_cause_scenario(
        "case-010", seed=29, as_of=as_of, repository=repository, ingested_at=as_of
    )
    toolbox = InvestigationToolbox(repository, scenario.pipeline_run.report.incident_id)
    evidence = [
        toolbox.execute("get_supplier_submissions"),
        toolbox.execute("get_orchestration_events"),
        toolbox.execute("get_transformation_metadata"),
    ]
    references = [item.evidence_id for item in evidence]

    assessment = assess_causal_evidence("supplier_batch_replay", references, evidence)

    assert not assessment.supported
    assert assessment.contradicting_evidence == (evidence[0].evidence_id,)
    assert set(assessment.unavailable_evidence) == {
        evidence[1].evidence_id,
        evidence[2].evidence_id,
    }
    assert has_inconclusive_audit_evidence(evidence)


def test_hypothesis_evidence_cannot_be_assigned_against_the_facts(tmp_path, as_of) -> None:
    repository = SQLiteHistoryRepository(tmp_path / "evidence-direction.db")
    scenario = run_root_cause_scenario(
        "case-010", seed=31, as_of=as_of, repository=repository, ingested_at=as_of
    )
    toolbox = InvestigationToolbox(repository, scenario.pipeline_run.report.incident_id)
    supplier = toolbox.execute("get_supplier_submissions")
    validation = toolbox.execute("get_validation_failures")

    with pytest.raises(CausalEvidenceError, match="cannot be recorded as supporting"):
        require_causal_evidence_alignment(
            "supplier_batch_replay",
            [supplier.evidence_id],
            [],
            [supplier],
        )

    with pytest.raises(CausalEvidenceError, match="requires cited facts that contradict"):
        require_causal_evidence_alignment(
            "supplier_batch_replay",
            [],
            [validation.evidence_id],
            [validation],
            require_contradiction=True,
        )


@pytest.mark.parametrize("case_id", ROOT_CAUSE_CASES)
def test_causal_scenarios_have_consistent_supplier_and_row_metadata(
    tmp_path, as_of, case_id
) -> None:
    repository = SQLiteHistoryRepository(tmp_path / f"consistency-{case_id}.db")
    scenario = run_root_cause_scenario(
        case_id,
        row_count=240,
        seed=73,
        as_of=as_of,
        repository=repository,
        ingested_at=as_of,
    )
    execution_id = scenario.pipeline_run.execution_id
    incident = repository.get_incident(scenario.pipeline_run.report.incident_id)
    samples = repository.get_problem_samples(execution_id, limit=50)
    submissions = repository.get_operational_records(execution_id, "supplier_submission")
    sample_suppliers = {
        item["record"]["supplier_id"] for item in samples if item["record"].get("supplier_id")
    }
    submission_suppliers = {
        item["details"]["supplier_id"] for item in submissions if item["details"].get("supplier_id")
    }

    assert incident["dataset"]["supplier_count"] == 1
    assert len(sample_suppliers) == 1
    assert submission_suppliers == sample_suppliers
    assert all(
        0
        < record["details"].get("row_count", incident["dataset"]["row_count"])
        <= incident["dataset"]["row_count"]
        for record in submissions
    )


def test_source_and_transformation_facts_agree(tmp_path, as_of) -> None:
    for case_id, expected_source_nulls, expected_errors in (
        ("case-004", 5, 0),
        ("case-005", 0, 5),
    ):
        repository = SQLiteHistoryRepository(tmp_path / f"nulls-{case_id}.db")
        scenario = run_root_cause_scenario(
            case_id, seed=83, as_of=as_of, repository=repository, ingested_at=as_of
        )
        execution_id = scenario.pipeline_run.execution_id
        submission = repository.get_operational_records(execution_id, "supplier_submission")[0][
            "details"
        ]
        transformation = repository.get_operational_records(
            execution_id, "transformation_metadata"
        )[0]["details"]

        assert submission["field_statistics"]["unit_price"]["null_count"] == (expected_source_nulls)
        assert transformation["input_null_count"] == expected_source_nulls
        assert transformation["output_null_count"] == 5
        assert transformation["conversion_errors"] == expected_errors


@pytest.mark.parametrize(
    ("case_id", "supplier_is_late", "minimum_queue_minutes"),
    [
        ("case-008", True, 0),
        ("case-009", False, 60),
    ],
)
def test_delivery_timestamps_distinguish_supplier_delay_from_queueing(
    tmp_path, as_of, case_id, supplier_is_late, minimum_queue_minutes
) -> None:
    repository = SQLiteHistoryRepository(tmp_path / f"timing-{case_id}.db")
    scenario = run_root_cause_scenario(
        case_id, seed=92, as_of=as_of, repository=repository, ingested_at=as_of
    )
    execution_id = scenario.pipeline_run.execution_id
    submission = repository.get_operational_records(execution_id, "supplier_submission")[0][
        "details"
    ]
    orchestration = repository.get_operational_records(execution_id, "orchestration_event")
    received = datetime.fromisoformat(submission["received_at"])
    expected = datetime.fromisoformat(submission["expected_by"])
    queue_minutes = max(int(item["details"].get("queue_wait_minutes", 0)) for item in orchestration)

    assert (received > expected) is supplier_is_late
    assert queue_minutes >= minimum_queue_minutes
    if supplier_is_late:
        assert queue_minutes <= 30


class _CausalPatternModel:
    """Small test double that exercises evidence-driven multi-turn state transitions."""

    def __init__(self) -> None:
        self.telemetry = ProviderTelemetry("test", "causal-pattern")

    def decide(self, context):
        self.telemetry.calls += 1
        self.telemetry.first_attempt_schema_valid_decisions += 1
        assert context.investigation_objective == "identify_underlying_root_cause"
        evidence_by_tool = {item.tool_name: item for item in context.evidence}
        if not context.hypotheses:
            return ModelDecision(
                "tool",
                [],
                ToolRequest(
                    "get_supplier_submissions", {}, "Test whether the input was submitted twice"
                ),
                hypothesis_operations=[
                    HypothesisOperation(
                        HypothesisOperationType.CREATE,
                        "H-replay",
                        "The same supplier batch may have been accepted twice",
                        "supplier_batch_replay",
                    ),
                    HypothesisOperation(
                        HypothesisOperationType.CREATE,
                        "H-retry",
                        "A partial job retry may have repeated writes",
                        "failed_job_retry",
                    ),
                    HypothesisOperation(
                        HypothesisOperationType.CREATE,
                        "H-join",
                        "A join may have multiplied rows",
                        "incorrect_join_cardinality",
                    ),
                ],
            )

        supplier = evidence_by_tool.get("get_supplier_submissions")
        records = [] if supplier is None else supplier.data.get("records", [])
        duplicate_submission = (
            len(records) >= 2 and len({item["details"].get("checksum") for item in records}) == 1
        )
        if duplicate_submission and "get_pipeline_logs" not in evidence_by_tool:
            return ModelDecision(
                "tool",
                [],
                ToolRequest(
                    "get_pipeline_logs", {}, "Verify that both matching batches were accepted"
                ),
                hypothesis_operations=[
                    HypothesisOperation(
                        HypothesisOperationType.UPDATE,
                        "H-replay",
                        supporting_evidence=[supplier.evidence_id],
                    )
                ],
            )
        if duplicate_submission:
            log = evidence_by_tool["get_pipeline_logs"]
            support = [supplier.evidence_id, log.evidence_id]
            return ModelDecision(
                "conclude",
                [],
                conclusion=InvestigationConclusion(
                    "supplier_batch_replay",
                    "Two matching supplier manifests were accepted and ingested",
                    ConfidenceLevel.HIGH,
                    support,
                    [],
                    ["Reject a previously accepted batch checksum"],
                ),
                hypothesis_operations=[
                    HypothesisOperation(
                        HypothesisOperationType.CONFIRM,
                        "H-replay",
                        supporting_evidence=[log.evidence_id],
                    )
                ],
            )

        if "get_orchestration_events" not in evidence_by_tool:
            return ModelDecision(
                "tool",
                [],
                ToolRequest("get_orchestration_events", {}, "Check retry and queue history"),
            )
        if "get_transformation_metadata" not in evidence_by_tool:
            return ModelDecision(
                "tool",
                [],
                ToolRequest("get_transformation_metadata", {}, "Check row-changing lineage"),
            )
        return ModelDecision(
            "conclude",
            [],
            conclusion=InvestigationConclusion(
                None,
                "Retained operational evidence does not distinguish replay, retry, or join causes",
                ConfidenceLevel.INSUFFICIENT,
                [],
                ["Can missing orchestration and lineage records be recovered?"],
                ["Restore the operational audit trail before proposing a repair"],
            ),
        )


def test_multi_step_investigation_compares_hypotheses_and_concludes(tmp_path, as_of) -> None:
    repository = SQLiteHistoryRepository(tmp_path / "multi-step.db")
    scenario = run_root_cause_scenario(
        "case-001", seed=57, as_of=as_of, repository=repository, ingested_at=as_of
    )

    report = InvestigationAgent(repository, _CausalPatternModel()).investigate(
        scenario.pipeline_run.report.incident_id
    )

    assert report.status == "completed"
    assert report.conclusion.diagnosis_code == "supplier_batch_replay"
    assert report.tools_used == ["get_supplier_submissions", "get_pipeline_logs"]
    assert {item.status.value for item in report.hypotheses} == {"active", "confirmed"}
    assert len(report.conclusion.evidence_references) == 2


def test_root_cause_benchmark_scores_resolution_and_inconclusive_cases(tmp_path, as_of) -> None:
    repository = SQLiteHistoryRepository(tmp_path / "benchmark.db")
    evaluator = RootCauseBenchmarkEvaluator(
        repository,
        lambda: InvestigationAgent(repository, _CausalPatternModel()),
    )

    summary = evaluator.run(
        row_count=100,
        seed=12,
        as_of=as_of,
        case_ids=("case-001", "case-010"),
    )

    assert summary.root_cause_accuracy == 1.0
    assert summary.evidence_support_rate == 1.0
    assert summary.inconclusive_accuracy == 1.0
    assert summary.workflow_completion_rate == 1.0
    assert summary.invalid_tool_calls == 0
    assert summary.duplicate_tool_calls == 0
    assert summary.false_confident_conclusions == 0
    assert summary.false_confident_conclusion_rate == 0.0
    assert summary.benchmark_scope == "ambiguous_upstream_root_cause"
