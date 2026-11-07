from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, TypedDict
from uuid import uuid4

import pandas as pd
from langgraph.graph import END, START, StateGraph

from retail_guardian.config import PipelineConfig
from retail_guardian.repair_models import (
    RepairProposal,
    RepairRecord,
    RepairStatus,
    SimulationResult,
    VerificationReport,
)
from retail_guardian.repair_operations import (
    OperationResult,
    RepairOperationRegistry,
    RepairPreconditionError,
    frame_hash,
)
from retail_guardian.repair_providers import (
    RepairPlanningContext,
    RepairPlanningError,
    RepairPlanningModel,
    validate_repair_proposal,
)
from retail_guardian.repair_verification import verify_repair
from retail_guardian.storage import HistoryRepository


class RepairWorkflowError(RuntimeError):
    """The requested repair transition is invalid or unsafe."""


class _PlanningState(TypedDict, total=False):
    investigation_id: str
    context: RepairPlanningContext
    proposal: RepairProposal
    record: RepairRecord


class _SimulationState(TypedDict, total=False):
    repair_id: str
    record: RepairRecord
    investigation_report: dict[str, Any]
    original: pd.DataFrame
    operational_records: list[dict[str, Any]]
    validation_results: list[dict[str, Any]]
    as_of: datetime
    original_hash: str
    operation_result: OperationResult
    simulation: SimulationResult
    verification: VerificationReport


class RepairWorkflow:
    """LangGraph repair planner with deterministic isolated execution and verification."""

    def __init__(
        self,
        repository: HistoryRepository,
        planner: RepairPlanningModel,
        config: PipelineConfig | None = None,
    ) -> None:
        self.repository = repository
        self.planner = planner
        self.config = config or PipelineConfig()
        self.operations = RepairOperationRegistry()
        self._planning_graph = self._build_planning_graph()
        self._simulation_graph = self._build_simulation_graph()

    def propose(self, investigation_id: str) -> RepairRecord:
        state = self._planning_graph.invoke({"investigation_id": investigation_id})
        record = state.get("record")
        if record is None:
            raise RepairWorkflowError("repair planning completed without a proposal")
        return record

    def approve_for_simulation(self, repair_id: str) -> RepairRecord:
        record = self._get_repair(repair_id)
        if record.status != RepairStatus.PROPOSED:
            raise RepairWorkflowError(
                f"repair {repair_id} cannot be approved from status {record.status.value!r}"
            )
        approved = record.model_copy(
            update={
                "status": RepairStatus.APPROVED_FOR_SIMULATION,
                "approved_at": datetime.now(UTC),
            }
        )
        self.repository.update_repair(approved, {"approval": "simulation_only"})
        return approved

    def simulate(self, repair_id: str) -> RepairRecord:
        record = self._get_repair(repair_id)
        if record.status != RepairStatus.APPROVED_FOR_SIMULATION:
            raise RepairWorkflowError(
                f"repair {repair_id} requires explicit simulation approval; current status is "
                f"{record.status.value!r}"
            )
        try:
            state = self._simulation_graph.invoke({"repair_id": repair_id})
        except RepairPreconditionError as exc:
            rejected = record.model_copy(
                update={
                    "status": RepairStatus.REJECTED,
                    "completed_at": datetime.now(UTC),
                    "rejection_reason": str(exc),
                }
            )
            self.repository.update_repair(
                rejected,
                {"precondition_failure": str(exc)},
            )
            raise RepairWorkflowError(f"repair simulation was rejected: {exc}") from exc
        completed = state.get("record")
        if completed is None:
            raise RepairWorkflowError("repair simulation completed without a result")
        return completed

    def get(self, repair_id: str) -> RepairRecord:
        return self._get_repair(repair_id)

    def _build_planning_graph(self):
        builder = StateGraph(_PlanningState)
        builder.add_node("prepare", self._prepare_plan)
        builder.add_node("plan", self._plan)
        builder.add_node("persist", self._persist_proposal)
        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "plan")
        builder.add_edge("plan", "persist")
        builder.add_edge("persist", END)
        return builder.compile()

    def _build_simulation_graph(self):
        builder = StateGraph(_SimulationState)
        builder.add_node("load", self._load_simulation)
        builder.add_node("execute", self._execute_simulation)
        builder.add_node("verify", self._verify_simulation)
        builder.add_node("finalize", self._finalize_simulation)
        builder.add_edge(START, "load")
        builder.add_edge("load", "execute")
        builder.add_edge("execute", "verify")
        builder.add_edge("verify", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile()

    def _prepare_plan(self, state: _PlanningState) -> dict[str, Any]:
        investigation_id = state["investigation_id"]
        report = self.repository.get_investigation_report(investigation_id)
        if report is None:
            raise RepairWorkflowError(f"completed investigation not found: {investigation_id}")
        if report.get("status") != "completed":
            raise RepairWorkflowError("only a completed investigation can produce a repair")
        conclusion = report.get("conclusion") or {}
        diagnosis = conclusion.get("diagnosis_code")
        if not diagnosis:
            raise RepairWorkflowError("an evidence-backed diagnosis is required before repair")
        incident = self.repository.get_incident(report["incident_id"])
        execution = self.repository.get_execution(report["incident_id"])
        if incident is None or execution is None:
            raise RepairWorkflowError("the investigation incident or execution is unavailable")
        frame = self.repository.get_dataset_snapshot(execution["execution_id"])
        if frame is None:
            raise RepairWorkflowError("the immutable dataset snapshot is unavailable")
        operational_records = self.repository.get_operational_records(
            execution["execution_id"], limit=100
        )
        capabilities = [
            item
            for item in self.operations.capabilities(frame, self.config, operational_records)
            if diagnosis in item.compatible_diagnoses
        ]
        if not capabilities:
            raise RepairWorkflowError(
                f"no safe repair capability supports diagnosis {diagnosis!r} and this dataset"
            )
        references = set(conclusion.get("evidence_references", []))
        evidence = {
            item["evidence_id"]: {
                "tool_name": item["tool_name"],
                "summary": item["summary"],
                "key_facts": item.get("data", {}),
            }
            for item in report.get("evidence", [])
            if item.get("evidence_id") in references
            and item.get("success")
            and item.get("result_kind") == "informative"
        }
        if references != set(evidence):
            raise RepairWorkflowError(
                "the investigation conclusion does not have complete informative evidence"
            )
        validation = self.repository.get_validation_results(execution["execution_id"])
        failed_checks = sorted(item["check_id"] for item in validation if not item["passed"])
        context = RepairPlanningContext(
            investigation_id=investigation_id,
            incident_id=report["incident_id"],
            diagnosis_code=diagnosis,
            diagnosis_summary=str(conclusion.get("final_diagnosis", "")),
            supporting_evidence=evidence,
            failed_check_ids=failed_checks,
            pipeline_metadata={
                "execution_id": execution["execution_id"],
                "pipeline_stage": incident.get("pipeline_stage"),
                "dataset": incident.get("dataset", {}),
            },
            capabilities=capabilities,
        )
        return {"context": context}

    def _plan(self, state: _PlanningState) -> dict[str, Any]:
        context = state["context"]
        proposal = self.planner.propose(context)
        self._validate_proposal(proposal, context)
        return {"proposal": proposal}

    def _persist_proposal(self, state: _PlanningState) -> dict[str, Any]:
        record = RepairRecord(
            repair_id=f"REP-{uuid4().hex[:12].upper()}",
            investigation_id=state["investigation_id"],
            status=RepairStatus.PROPOSED,
            proposal=state["proposal"],
            proposed_at=datetime.now(UTC),
        )
        self.repository.create_repair(record)
        return {"record": record}

    def _validate_proposal(
        self,
        proposal: RepairProposal,
        context: RepairPlanningContext,
    ) -> None:
        try:
            validate_repair_proposal(proposal, context)
        except RepairPlanningError as exc:
            raise RepairWorkflowError(str(exc)) from exc

    def _load_simulation(self, state: _SimulationState) -> dict[str, Any]:
        record = self._get_repair(state["repair_id"])
        report = self.repository.get_investigation_report(record.investigation_id)
        if report is None:
            raise RepairWorkflowError("originating investigation is unavailable")
        execution = self.repository.get_execution(report["incident_id"])
        incident = self.repository.get_incident(report["incident_id"])
        if execution is None or incident is None:
            raise RepairWorkflowError("originating execution is unavailable")
        original = self.repository.get_dataset_snapshot(execution["execution_id"])
        if original is None:
            raise RepairWorkflowError("the immutable dataset snapshot is unavailable")
        return {
            "record": record,
            "investigation_report": report,
            "original": original,
            "operational_records": self.repository.get_operational_records(
                execution["execution_id"], limit=100
            ),
            "validation_results": self.repository.get_validation_results(execution["execution_id"]),
            "as_of": datetime.fromisoformat(incident["detected_at"]),
            "original_hash": frame_hash(original),
        }

    def _execute_simulation(self, state: _SimulationState) -> dict[str, Any]:
        record = state["record"]
        result = self.operations.execute(
            record.proposal,
            state["original"],
            self.config,
            state["operational_records"],
        )
        simulated_at = datetime.now(UTC)
        simulation = SimulationResult(
            isolated=True,
            original_snapshot_hash=state["original_hash"],
            repaired_snapshot_hash=frame_hash(result.frame),
            change_summary=result.summary,
            simulated_at=simulated_at,
        )
        simulated = record.model_copy(
            update={
                "status": RepairStatus.SIMULATED,
                "simulated_at": simulated_at,
                "simulation": simulation,
            }
        )
        self.repository.update_repair(
            simulated,
            {"change_summary": result.summary.model_dump(mode="json")},
        )
        return {"record": simulated, "operation_result": result, "simulation": simulation}

    def _verify_simulation(self, state: _SimulationState) -> dict[str, Any]:
        result = state["operation_result"]
        verification = verify_repair(
            state["record"].proposal,
            state["original"],
            result.frame,
            result.config,
            result.summary,
            state["validation_results"],
            as_of=state["as_of"],
            original_snapshot_hash=state["original_hash"],
        )
        verified = state["record"].model_copy(
            update={
                "status": RepairStatus.VERIFICATION_COMPLETED,
                "verified_at": verification.completed_at,
                "verification": verification,
            }
        )
        self.repository.update_repair(
            verified,
            {"verification": verification.model_dump(mode="json")},
        )
        return {"record": verified, "verification": verification}

    def _finalize_simulation(self, state: _SimulationState) -> dict[str, Any]:
        verification = state["verification"]
        status = RepairStatus.RECOMMENDED if verification.passed else RepairStatus.REJECTED
        reason = None if verification.passed else "; ".join(verification.reasons)
        completed = state["record"].model_copy(
            update={
                "status": status,
                "completed_at": verification.completed_at,
                "verification": verification,
                "rejection_reason": reason,
            }
        )
        self.repository.update_repair(
            completed,
            {"recommendation": status.value, "reasons": verification.reasons},
        )
        return {"record": completed}

    def _get_repair(self, repair_id: str) -> RepairRecord:
        record = self.repository.get_repair(repair_id)
        if record is None:
            raise RepairWorkflowError(f"repair not found: {repair_id}")
        return record
