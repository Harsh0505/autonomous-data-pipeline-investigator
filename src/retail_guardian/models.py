from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class Severity(StrEnum):
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CheckKind(StrEnum):
    DETERMINISTIC = "deterministic"
    STATISTICAL = "statistical"


@dataclass(frozen=True)
class Finding:
    message: str
    affected_rows: int
    sample_values: list[Any] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    kind: CheckKind
    passed: bool
    severity: Severity
    finding: Finding | None = None


@dataclass(frozen=True)
class IncidentReport:
    incident_id: str
    execution_id: str
    source: str
    status: str
    detected_at: datetime
    pipeline_stage: str
    dataset: dict[str, Any]
    violations: list[CheckResult]
    metrics: dict[str, Any]
    pipeline_error: str | None = None
    schema_version: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    if hasattr(value, "item"):
        return value.item()
    return value
