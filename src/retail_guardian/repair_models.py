from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RepairType(StrEnum):
    DEDUPLICATE_RECORDS = "deduplicate_records"
    CORRECT_SUPPLIER_CURRENCY = "correct_supplier_currency"
    ACTIVATE_APPROVED_CATEGORY = "activate_approved_category"


class RepairStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED_FOR_SIMULATION = "approved_for_simulation"
    SIMULATED = "simulated"
    VERIFICATION_COMPLETED = "verification_completed"
    RECOMMENDED = "recommended"
    REJECTED = "rejected"


class DeduplicateParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identifier_column: Literal["order_id"] = "order_id"
    identifiers: list[str] = Field(min_length=1, max_length=100)
    keep: Literal["first", "last"] = "first"


class CorrectCurrencyParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supplier_id: str = Field(min_length=1, max_length=100)
    expected_currency: Literal["GBP", "USD", "EUR"]


class ActivateCategoryParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str = Field(min_length=1, max_length=100)
    approved_change_id: str = Field(min_length=1, max_length=100)


RepairParameters = Annotated[
    DeduplicateParameters | CorrectCurrencyParameters | ActivateCategoryParameters,
    Field(union_mode="smart"),
]


class VerificationRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checks_to_pass: list[str] = Field(min_length=1)
    preserve_unaffected_records: bool = True
    maximum_rows_removed: int = Field(default=0, ge=0)
    expected_changed_columns: list[str] = Field(default_factory=list)


class RepairProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repair_type: RepairType
    target_component: Literal["dataset", "pipeline_configuration"]
    parameters: RepairParameters
    explanation: str = Field(min_length=1, max_length=1000)
    supporting_evidence_ids: list[str] = Field(min_length=1)
    expected_effect: str = Field(min_length=1, max_length=500)
    risks_and_limitations: list[str] = Field(min_length=1)
    verification_requirements: VerificationRequirements
    rollback_approach: str = Field(min_length=1, max_length=500)
    planner_provider: str = "unknown"
    planner_model: str = "unknown"
    token_usage: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def parameters_match_repair_type(self) -> RepairProposal:
        expected = {
            RepairType.DEDUPLICATE_RECORDS: DeduplicateParameters,
            RepairType.CORRECT_SUPPLIER_CURRENCY: CorrectCurrencyParameters,
            RepairType.ACTIVATE_APPROVED_CATEGORY: ActivateCategoryParameters,
        }[self.repair_type]
        if not isinstance(self.parameters, expected):
            raise ValueError(f"parameters do not match repair_type {self.repair_type.value!r}")
        return self


class RepairCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repair_type: RepairType
    target_component: Literal["dataset", "pipeline_configuration"]
    description: str
    limitations: str
    resolves_checks: list[str]
    compatible_diagnoses: list[str]
    candidate_parameters: list[dict[str, object]]
    parameter_schema: dict[str, object]
    maximum_expected_rows_removed: int = Field(default=0, ge=0)
    expected_changed_columns: list[str] = Field(default_factory=list)


class ChangeSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repair_type: RepairType
    rows_before: int
    rows_after: int
    rows_changed: int
    rows_removed: int
    affected_identifiers: list[str]
    changed_columns: list[str]
    configuration_changes: dict[str, object] = Field(default_factory=dict)


class SimulationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    isolated: bool
    original_snapshot_hash: str
    repaired_snapshot_hash: str
    change_summary: ChangeSummary
    simulated_at: datetime


class VerificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    checks_required: list[str]
    original_failed_checks: list[str]
    repaired_failed_checks: list[str]
    resolved_checks: list[str]
    remaining_checks: list[str]
    new_failures: list[str]
    original_row_count: int
    repaired_row_count: int
    unexpected_data_loss: bool
    unaffected_records_preserved: bool
    original_snapshot_preserved: bool
    business_aggregate_comparison: dict[str, object]
    reasons: list[str]
    completed_at: datetime


class RepairRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repair_id: str
    investigation_id: str
    status: RepairStatus
    proposal: RepairProposal
    proposed_at: datetime
    approved_at: datetime | None = None
    simulated_at: datetime | None = None
    verified_at: datetime | None = None
    completed_at: datetime | None = None
    simulation: SimulationResult | None = None
    verification: VerificationReport | None = None
    rejection_reason: str | None = None

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)
