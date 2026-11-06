from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from retail_guardian.models import _json_ready


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient_evidence"


class HypothesisStatus(StrEnum):
    ACTIVE = "active"
    REJECTED = "rejected"
    CONFIRMED = "confirmed"


class HypothesisOperationType(StrEnum):
    CREATE = "create"
    UPDATE = "update"
    CONFIRM = "confirm"
    REJECT = "reject"


class ToolResultKind(StrEnum):
    INFORMATIVE = "informative"
    EMPTY = "empty"
    INVALID = "invalid"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    tool_name: str
    summary: str
    data: dict[str, Any]
    success: bool = True
    error: str | None = None
    result_kind: ToolResultKind = ToolResultKind.INFORMATIVE


@dataclass(frozen=True)
class RootCauseHypothesis:
    hypothesis_id: str
    statement: str
    diagnosis_code: str | None
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HypothesisOperation:
    operation: HypothesisOperationType
    hypothesis_id: str
    statement: str | None = None
    diagnosis_code: str | None = None
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class InvestigationConclusion:
    diagnosis_code: str | None
    final_diagnosis: str
    confidence: ConfidenceLevel
    evidence_references: list[str]
    outstanding_questions: list[str]
    recommended_actions: list[str]


@dataclass(frozen=True)
class InvestigationReport:
    investigation_id: str
    incident_id: str
    status: str
    started_at: datetime
    completed_at: datetime
    hypotheses: list[RootCauseHypothesis]
    evidence: list[Evidence]
    tools_used: list[str]
    conclusion: InvestigationConclusion
    tool_call_count: int
    token_usage: int | None = None
    duration_seconds: float = 0.0
    decision_count: int = 0
    invalid_tool_calls: int = 0
    successful_tool_calls: int = 0
    informative_tool_calls: int = 0
    empty_tool_calls: int = 0
    duplicate_tool_calls: int = 0
    model_validation_failures: int = 0
    first_attempt_schema_valid_decisions: int = 0
    retry_recoveries: int = 0
    investigation_failures: int = 0
    timeouts: int = 0
    provider_name: str = "unknown"
    model_name: str = "unknown"
    schema_version: str = "3.0"

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolRequest:
    name: str
    arguments: dict[str, Any]
    rationale: str


@dataclass(frozen=True)
class CompletedToolCall:
    name: str
    arguments: dict[str, Any]
    evidence_id: str
    success: bool
    error: str | None = None
    result_kind: ToolResultKind = ToolResultKind.INFORMATIVE


@dataclass(frozen=True)
class ModelDecision:
    action: Literal["tool", "conclude"]
    hypotheses: list[RootCauseHypothesis]
    tool_request: ToolRequest | None = None
    conclusion: InvestigationConclusion | None = None
    token_usage: int | None = None
    hypothesis_operations: list[HypothesisOperation] = field(default_factory=list)


@dataclass(frozen=True)
class RejectedDecisionDiagnostic:
    attempt: int
    error_type: str
    error: str
    decision: dict[str, Any] | None
    content_length: int


@dataclass
class ProviderTelemetry:
    provider_name: str
    model_name: str
    calls: int = 0
    validation_failures: int = 0
    invalid_tool_calls: int = 0
    connection_failures: int = 0
    timeouts: int = 0
    retries: int = 0
    duration_seconds: float = 0.0
    first_attempt_schema_valid_decisions: int = 0
    retry_recoveries: int = 0
    rejected_decisions: list[RejectedDecisionDiagnostic] = field(default_factory=list)


@dataclass(frozen=True)
class InvestigationContext:
    incident: dict[str, Any]
    hypotheses: list[RootCauseHypothesis]
    evidence: list[Evidence]
    available_tools: list[ToolDefinition]
    remaining_tool_calls: int
    completed_tool_calls: list[CompletedToolCall] = field(default_factory=list)
    remaining_invalid_tool_calls: int = 3
    available_check_ids: list[str] = field(default_factory=list)
    remaining_empty_tool_calls: int = 2
