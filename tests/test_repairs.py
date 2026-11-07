from __future__ import annotations

import pytest
from pydantic import ValidationError

from retail_guardian.causal_scenarios import run_root_cause_scenario
from retail_guardian.investigation import InvestigationAgent
from retail_guardian.investigation_models import (
    ConfidenceLevel,
    Evidence,
    InvestigationConclusion,
    InvestigationReport,
    ToolResultKind,
)
from retail_guardian.providers import HeuristicInvestigationModel
from retail_guardian.repair import RepairWorkflow, RepairWorkflowError
from retail_guardian.repair_models import (
    RepairProposal,
    RepairStatus,
    RepairType,
)
from retail_guardian.repair_operations import RepairOperationRegistry, frame_hash
from retail_guardian.repair_providers import (
    DeterministicRepairPlanningModel,
    RepairPlanningContext,
)
from retail_guardian.repair_verification import verify_repair
from retail_guardian.scenarios import run_scenario
from retail_guardian.storage import SQLiteHistoryRepository


def _investigated_scenario(tmp_path, as_of, scenario: str):
    repository = SQLiteHistoryRepository(tmp_path / f"{scenario}.db")
    run = run_scenario(
        scenario,
        row_count=100,
        seed=42,
        as_of=as_of,
        ingested_at=as_of,
        repository=repository,
    )
    report = InvestigationAgent(repository, HeuristicInvestigationModel()).investigate(
        run.report.incident_id
    )
    assert report.status == "completed"
    return repository, run, report


def test_duplicate_repair_end_to_end_isolated_and_recommended(tmp_path, as_of) -> None:
    repository, run, investigation = _investigated_scenario(tmp_path, as_of, "duplicate_records")
    workflow = RepairWorkflow(repository, DeterministicRepairPlanningModel())
    original = repository.get_dataset_snapshot(run.execution_id)
    assert original is not None
    original_hash = frame_hash(original)

    proposed = workflow.propose(investigation.investigation_id)
    assert proposed.status == RepairStatus.PROPOSED
    assert proposed.proposal.repair_type == RepairType.DEDUPLICATE_RECORDS
    with pytest.raises(RepairWorkflowError, match="requires explicit simulation approval"):
        workflow.simulate(proposed.repair_id)

    approved = workflow.approve_for_simulation(proposed.repair_id)
    completed = workflow.simulate(approved.repair_id)

    assert completed.status == RepairStatus.RECOMMENDED
    assert completed.verification is not None and completed.verification.passed
    assert completed.verification.resolved_checks == ["unique_order_id"]
    assert completed.simulation is not None and completed.simulation.isolated
    assert completed.simulation.change_summary.rows_removed == 5
    persisted_original = repository.get_dataset_snapshot(run.execution_id)
    assert persisted_original is not None
    assert frame_hash(persisted_original) == original_hash
    trace = repository.get_repair_trace(completed.repair_id)
    assert trace is not None
    assert [event["status"] for event in trace["events"]] == [
        "proposed",
        "approved_for_simulation",
        "simulated",
        "verification_completed",
        "recommended",
    ]


def test_currency_repair_is_rejected_when_price_anomaly_remains(tmp_path, as_of) -> None:
    repository, _, investigation = _investigated_scenario(tmp_path, as_of, "currency_mismatch")
    workflow = RepairWorkflow(repository, DeterministicRepairPlanningModel())

    proposed = workflow.propose(investigation.investigation_id)
    assert proposed.proposal.repair_type == RepairType.CORRECT_SUPPLIER_CURRENCY
    workflow.approve_for_simulation(proposed.repair_id)
    completed = workflow.simulate(proposed.repair_id)

    assert completed.status == RepairStatus.REJECTED
    assert completed.verification is not None
    assert not completed.verification.passed
    assert "unit_price_z_score" in completed.verification.remaining_checks
    assert "required checks still fail" in (completed.rejection_reason or "")


def test_approved_taxonomy_repair_end_to_end(tmp_path, as_of) -> None:
    repository = SQLiteHistoryRepository(tmp_path / "taxonomy.db")
    scenario = run_root_cause_scenario(
        "case-007",
        row_count=100,
        seed=42,
        as_of=as_of,
        repository=repository,
        ingested_at=as_of,
    )
    run = scenario.pipeline_run
    evidence = Evidence(
        "EV-APPROVAL",
        "get_configuration_changes",
        "Retrieved approved taxonomy metadata",
        {"records": repository.get_operational_records(run.execution_id)},
        result_kind=ToolResultKind.INFORMATIVE,
    )
    investigation_id = "INV-TAXONOMY-TEST"
    conclusion = InvestigationConclusion(
        "legitimate_taxonomy_update",
        "An approved taxonomy version was not loaded by the runtime",
        ConfidenceLevel.HIGH,
        [evidence.evidence_id],
        [],
        [],
    )
    report = InvestigationReport(
        investigation_id=investigation_id,
        incident_id=run.report.incident_id,
        status="completed",
        started_at=as_of,
        completed_at=as_of,
        hypotheses=[],
        evidence=[evidence],
        tools_used=["get_configuration_changes"],
        conclusion=conclusion,
        tool_call_count=1,
    )
    repository.start_investigation(investigation_id, run.report.incident_id, as_of)
    repository.complete_investigation(report)
    workflow = RepairWorkflow(repository, DeterministicRepairPlanningModel())

    proposed = workflow.propose(investigation_id)
    assert proposed.proposal.repair_type == RepairType.ACTIVATE_APPROVED_CATEGORY
    workflow.approve_for_simulation(proposed.repair_id)
    completed = workflow.simulate(proposed.repair_id)

    assert completed.status == RepairStatus.RECOMMENDED
    assert completed.verification is not None and completed.verification.passed
    assert completed.verification.resolved_checks == ["known_category"]
    assert completed.simulation is not None
    original_hash = completed.simulation.original_snapshot_hash
    repaired_hash = completed.simulation.repaired_snapshot_hash
    assert original_hash == repaired_hash


def test_unsupported_diagnosis_has_no_repair_capability(tmp_path, as_of) -> None:
    repository, _, investigation = _investigated_scenario(tmp_path, as_of, "missing_values")
    workflow = RepairWorkflow(repository, DeterministicRepairPlanningModel())

    with pytest.raises(RepairWorkflowError, match="no safe repair capability"):
        workflow.propose(investigation.investigation_id)


class _FabricatedEvidencePlanner:
    def propose(self, context: RepairPlanningContext) -> RepairProposal:
        capability = context.capabilities[0]
        return RepairProposal.model_validate(
            {
                "repair_type": capability.repair_type,
                "target_component": capability.target_component,
                "parameters": capability.candidate_parameters[0],
                "explanation": "Use the available scoped operation.",
                "supporting_evidence_ids": ["EV-FABRICATED"],
                "expected_effect": "Resolve the failure.",
                "risks_and_limitations": [capability.limitations],
                "verification_requirements": {
                    "checks_to_pass": context.failed_check_ids,
                    "maximum_rows_removed": capability.maximum_expected_rows_removed,
                    "expected_changed_columns": capability.expected_changed_columns,
                },
                "rollback_approach": "Discard the isolated copy.",
            }
        )


def test_fabricated_repair_evidence_is_rejected(tmp_path, as_of) -> None:
    repository, _, investigation = _investigated_scenario(tmp_path, as_of, "duplicate_records")
    workflow = RepairWorkflow(repository, _FabricatedEvidencePlanner())

    with pytest.raises(RepairWorkflowError, match="fabricated evidence"):
        workflow.propose(investigation.investigation_id)


def test_invalid_repair_parameters_fail_type_validation() -> None:
    with pytest.raises(ValidationError):
        RepairProposal.model_validate(
            {
                "repair_type": "deduplicate_records",
                "target_component": "dataset",
                "parameters": {"identifier_column": "customer_id", "identifiers": []},
                "explanation": "Invalid proposal",
                "supporting_evidence_ids": ["EV-1"],
                "expected_effect": "None",
                "risks_and_limitations": ["Unsafe"],
                "verification_requirements": {"checks_to_pass": ["unique_order_id"]},
                "rollback_approach": "Discard",
            }
        )


def test_verification_detects_unexpected_data_loss(tmp_path, as_of) -> None:
    repository, run, investigation = _investigated_scenario(tmp_path, as_of, "duplicate_records")
    workflow = RepairWorkflow(repository, DeterministicRepairPlanningModel())
    record = workflow.propose(investigation.investigation_id)
    original = repository.get_dataset_snapshot(run.execution_id)
    assert original is not None
    operation = RepairOperationRegistry().execute(
        record.proposal,
        original,
        repository.config,
        [],
    )
    damaged = operation.frame.iloc[1:].reset_index(drop=True)

    verification = verify_repair(
        record.proposal,
        original,
        damaged,
        operation.config,
        operation.summary,
        repository.get_validation_results(run.execution_id),
        as_of=as_of,
        original_snapshot_hash=frame_hash(original),
    )

    assert not verification.passed
    assert verification.unexpected_data_loss
    assert not verification.unaffected_records_preserved
