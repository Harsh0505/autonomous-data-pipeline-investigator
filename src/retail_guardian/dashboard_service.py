from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from retail_guardian.investigation import InvestigationAgent
from retail_guardian.investigation_models import InvestigationReport
from retail_guardian.pipeline import PipelineRun
from retail_guardian.providers import HeuristicInvestigationModel, OllamaInvestigationModel
from retail_guardian.repair import RepairWorkflow
from retail_guardian.repair_models import RepairRecord
from retail_guardian.repair_providers import (
    DeterministicRepairPlanningModel,
    OllamaRepairPlanningModel,
)
from retail_guardian.scenarios import run_scenario
from retail_guardian.storage import SQLiteHistoryRepository

ProviderName = Literal["mock", "ollama"]


@dataclass(frozen=True)
class DashboardSnapshot:
    pipeline_health: str
    execution_count: int
    incident_count: int
    affected_records: int
    active_investigations: int
    recommended_repairs: int
    rejected_repairs: int
    recent_executions: list[dict[str, Any]]
    incidents: list[dict[str, Any]]
    investigations: list[dict[str, Any]]
    repairs: list[RepairRecord]


class DashboardService:
    """Application-service boundary used by the Streamlit presentation layer."""

    def __init__(self, database: Path | str) -> None:
        self.repository = SQLiteHistoryRepository(database)

    def overview(self, *, limit: int = 25) -> DashboardSnapshot:
        executions = self.repository.list_executions(limit=limit)
        incidents = self.repository.list_incidents(limit=limit)
        investigations = self.repository.list_investigation_sessions(limit=limit)
        repairs = self.repository.list_repairs(limit=limit)
        latest_status = executions[0]["status"] if executions else "no data"
        affected = sum(_incident_affected_records(item["report"]) for item in incidents)
        return DashboardSnapshot(
            pipeline_health=("healthy" if latest_status == "healthy" else latest_status),
            execution_count=len(executions),
            incident_count=len(incidents),
            affected_records=affected,
            active_investigations=sum(item["status"] == "running" for item in investigations),
            recommended_repairs=sum(item.status.value == "recommended" for item in repairs),
            rejected_repairs=sum(item.status.value == "rejected" for item in repairs),
            recent_executions=executions,
            incidents=incidents,
            investigations=investigations,
            repairs=repairs,
        )

    def run_scenario(
        self,
        scenario: str,
        *,
        row_count: int,
        seed: int,
        as_of: datetime,
    ) -> PipelineRun:
        return run_scenario(
            scenario,
            row_count=row_count,
            seed=seed,
            as_of=as_of,
            ingested_at=as_of,
            repository=self.repository,
        )

    def investigate(
        self,
        incident_id: str,
        *,
        provider: ProviderName,
        model: str = "qwen3:4b",
        endpoint: str = "http://localhost:11434",
        timeout_seconds: float = 180.0,
    ) -> InvestigationReport:
        investigation_model = (
            OllamaInvestigationModel(
                model,
                endpoint=endpoint,
                timeout_seconds=timeout_seconds,
                max_retries=2,
            )
            if provider == "ollama"
            else HeuristicInvestigationModel()
        )
        return InvestigationAgent(self.repository, investigation_model).investigate(incident_id)

    def propose_repair(
        self,
        investigation_id: str,
        *,
        provider: ProviderName,
        model: str = "qwen3:4b",
        endpoint: str = "http://localhost:11434",
        timeout_seconds: float = 180.0,
    ) -> RepairRecord:
        planner = (
            OllamaRepairPlanningModel(
                model,
                endpoint=endpoint,
                timeout_seconds=timeout_seconds,
                max_retries=2,
            )
            if provider == "ollama"
            else DeterministicRepairPlanningModel()
        )
        return RepairWorkflow(self.repository, planner).propose(investigation_id)

    def approve_repair(self, repair_id: str) -> RepairRecord:
        return RepairWorkflow(
            self.repository, DeterministicRepairPlanningModel()
        ).approve_for_simulation(repair_id)

    def simulate_repair(self, repair_id: str) -> RepairRecord:
        return RepairWorkflow(self.repository, DeterministicRepairPlanningModel()).simulate(
            repair_id
        )


def _incident_affected_records(report: dict[str, Any]) -> int:
    counts = [
        int((item.get("finding") or {}).get("affected_rows", 0))
        for item in report.get("violations", [])
    ]
    return max(counts, default=0)
