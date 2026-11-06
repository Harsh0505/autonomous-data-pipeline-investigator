from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from retail_guardian.config import PipelineConfig
from retail_guardian.investigation_models import (
    Evidence,
    InvestigationReport,
    ModelDecision,
    RejectedDecisionDiagnostic,
)
from retail_guardian.models import CheckResult, _json_ready
from retail_guardian.pipeline import PipelineRun


class HistoryRepository(Protocol):
    """Storage boundary used by pipeline history and investigation components."""

    def save_pipeline_run(
        self, run: PipelineRun, frame: pd.DataFrame, *, ingested_at: datetime | None = None
    ) -> None: ...

    def get_incident(self, incident_id: str) -> dict[str, Any] | None: ...

    def get_execution(self, incident_id: str) -> dict[str, Any] | None: ...

    def get_schema(self, execution_id: str) -> list[dict[str, Any]]: ...

    def get_profile(self, execution_id: str) -> dict[str, Any] | None: ...

    def get_validation_results(self, execution_id: str) -> list[dict[str, Any]]: ...

    def get_problem_samples(
        self, execution_id: str, check_id: str | None = None, *, limit: int = 10
    ) -> list[dict[str, Any]]: ...

    def list_executions(
        self, *, before_ingested_at: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]: ...

    def start_investigation(
        self, investigation_id: str, incident_id: str, started_at: datetime
    ) -> None: ...

    def record_tool_call(
        self,
        investigation_id: str,
        sequence: int,
        tool_name: str,
        arguments: dict[str, Any],
        rationale: str,
        evidence: Evidence,
        started_at: datetime,
        completed_at: datetime,
    ) -> None: ...

    def record_model_decision(
        self,
        investigation_id: str,
        sequence: int,
        decision: ModelDecision,
        created_at: datetime,
    ) -> None: ...

    def record_model_rejection(
        self,
        investigation_id: str,
        sequence: int,
        rejection: RejectedDecisionDiagnostic,
        created_at: datetime,
    ) -> None: ...

    def complete_investigation(self, report: InvestigationReport) -> None: ...


class SQLiteHistoryRepository:
    """SQLite implementation that stores profiles and evidence, not full datasets."""

    def __init__(self, path: Path | str, config: PipelineConfig | None = None) -> None:
        self.path = Path(path)
        self.config = config or PipelineConfig()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS pipeline_execution (
                    execution_id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    pipeline_stage TEXT NOT NULL,
                    ingestion_timestamp TEXT NOT NULL,
                    event_time_min TEXT,
                    event_time_max TEXT,
                    row_count INTEGER NOT NULL,
                    profile_json TEXT NOT NULL,
                    aggregate_metrics_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dataset_schema (
                    execution_id TEXT NOT NULL REFERENCES pipeline_execution(execution_id),
                    ordinal INTEGER NOT NULL,
                    column_name TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    null_count INTEGER NOT NULL,
                    distinct_count INTEGER NOT NULL,
                    PRIMARY KEY (execution_id, column_name)
                );
                CREATE TABLE IF NOT EXISTS validation_result (
                    execution_id TEXT NOT NULL REFERENCES pipeline_execution(execution_id),
                    check_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    severity TEXT NOT NULL,
                    finding_json TEXT,
                    PRIMARY KEY (execution_id, check_id)
                );
                CREATE TABLE IF NOT EXISTS incident (
                    incident_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL REFERENCES pipeline_execution(execution_id),
                    observable_report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS problem_sample (
                    execution_id TEXT NOT NULL REFERENCES pipeline_execution(execution_id),
                    check_id TEXT NOT NULL,
                    sample_index INTEGER NOT NULL,
                    record_json TEXT NOT NULL,
                    PRIMARY KEY (execution_id, check_id, sample_index)
                );
                CREATE TABLE IF NOT EXISTS investigation_session (
                    investigation_id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL REFERENCES incident(incident_id),
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    report_json TEXT
                );
                CREATE TABLE IF NOT EXISTS agent_tool_call (
                    investigation_id TEXT NOT NULL
                        REFERENCES investigation_session(investigation_id),
                    sequence INTEGER NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    error TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    PRIMARY KEY (investigation_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS agent_decision (
                    investigation_id TEXT NOT NULL
                        REFERENCES investigation_session(investigation_id),
                    sequence INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (investigation_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS model_rejection (
                    investigation_id TEXT NOT NULL
                        REFERENCES investigation_session(investigation_id),
                    sequence INTEGER NOT NULL,
                    attempt INTEGER NOT NULL,
                    error_type TEXT NOT NULL,
                    error TEXT NOT NULL,
                    decision_json TEXT,
                    content_length INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (investigation_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY,
                    investigation_id TEXT NOT NULL
                        REFERENCES investigation_session(investigation_id),
                    tool_sequence INTEGER NOT NULL,
                    tool_name TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS execution_ingested_idx
                    ON pipeline_execution(ingestion_timestamp DESC);
                CREATE INDEX IF NOT EXISTS investigation_incident_idx
                    ON investigation_session(incident_id, started_at DESC);
                """
            )

    def save_pipeline_run(
        self, run: PipelineRun, frame: pd.DataFrame, *, ingested_at: datetime | None = None
    ) -> None:
        ingestion_time = ingested_at or datetime.now(UTC)
        if ingestion_time.tzinfo is None:
            raise ValueError("ingested_at must include a timezone")
        profile = _build_profile(frame)
        aggregate_metrics = _build_aggregate_metrics(frame)
        report = run.report

        with self._connect() as connection:
            connection.execute(
                """INSERT INTO pipeline_execution
                   (execution_id, incident_id, source, status, pipeline_stage,
                    ingestion_timestamp, event_time_min, event_time_max, row_count,
                    profile_json, aggregate_metrics_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(execution_id) DO UPDATE SET
                     source=excluded.source,
                     status=excluded.status,
                     pipeline_stage=excluded.pipeline_stage,
                     ingestion_timestamp=excluded.ingestion_timestamp,
                     event_time_min=excluded.event_time_min,
                     event_time_max=excluded.event_time_max,
                     row_count=excluded.row_count,
                     profile_json=excluded.profile_json,
                     aggregate_metrics_json=excluded.aggregate_metrics_json""",
                (
                    run.execution_id,
                    report.incident_id,
                    report.source,
                    report.status,
                    report.pipeline_stage,
                    ingestion_time.isoformat(),
                    report.dataset.get("event_time_min"),
                    report.dataset.get("event_time_max"),
                    len(frame),
                    _dump(profile),
                    _dump(aggregate_metrics),
                ),
            )
            connection.execute(
                "DELETE FROM dataset_schema WHERE execution_id = ?", (run.execution_id,)
            )
            connection.executemany(
                """INSERT INTO dataset_schema
                   (execution_id, ordinal, column_name, data_type, null_count, distinct_count)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (
                        run.execution_id,
                        position,
                        str(column),
                        str(frame[column].dtype),
                        int(frame[column].isna().sum()),
                        int(frame[column].nunique(dropna=True)),
                    )
                    for position, column in enumerate(frame.columns)
                ],
            )
            connection.execute(
                "DELETE FROM validation_result WHERE execution_id = ?", (run.execution_id,)
            )
            connection.executemany(
                """INSERT INTO validation_result
                   (execution_id, check_id, kind, passed, severity, finding_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [_validation_row(run.execution_id, result) for result in run.validation_results],
            )
            connection.execute(
                "DELETE FROM problem_sample WHERE execution_id = ?", (run.execution_id,)
            )
            for result in run.validation_results:
                if result.passed:
                    continue
                samples = _problem_samples(frame, result.check_id, self.config)
                connection.executemany(
                    """INSERT INTO problem_sample
                       (execution_id, check_id, sample_index, record_json)
                       VALUES (?, ?, ?, ?)""",
                    [
                        (run.execution_id, result.check_id, index, _dump(record))
                        for index, record in enumerate(samples)
                    ],
                )
            if report.status == "incident":
                connection.execute(
                    """INSERT INTO incident
                       (incident_id, execution_id, observable_report_json, created_at)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(incident_id) DO UPDATE SET
                         observable_report_json=excluded.observable_report_json,
                         created_at=excluded.created_at""",
                    (
                        report.incident_id,
                        run.execution_id,
                        report.to_json(indent=0),
                        report.detected_at.isoformat(),
                    ),
                )

    def get_incident(self, incident_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT observable_report_json FROM incident WHERE incident_id = ?", (incident_id,)
            ).fetchone()
        return None if row is None else json.loads(row["observable_report_json"])

    def get_execution(self, incident_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM pipeline_execution WHERE incident_id = ?", (incident_id,)
            ).fetchone()
        return None if row is None else _execution_dict(row)

    def get_schema(self, execution_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT column_name, data_type, null_count, distinct_count
                   FROM dataset_schema WHERE execution_id = ? ORDER BY ordinal""",
                (execution_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_profile(self, execution_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT profile_json, aggregate_metrics_json
                   FROM pipeline_execution WHERE execution_id = ?""",
                (execution_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "profile": json.loads(row["profile_json"]),
            "aggregate_metrics": json.loads(row["aggregate_metrics_json"]),
        }

    def get_validation_results(self, execution_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT check_id, kind, passed, severity, finding_json
                   FROM validation_result WHERE execution_id = ? ORDER BY check_id""",
                (execution_id,),
            ).fetchall()
        return [
            {
                "check_id": row["check_id"],
                "kind": row["kind"],
                "passed": bool(row["passed"]),
                "severity": row["severity"],
                "finding": json.loads(row["finding_json"]) if row["finding_json"] else None,
            }
            for row in rows
        ]

    def get_problem_samples(
        self, execution_id: str, check_id: str | None = None, *, limit: int = 10
    ) -> list[dict[str, Any]]:
        query = "SELECT check_id, record_json FROM problem_sample WHERE execution_id = ?"
        parameters: list[Any] = [execution_id]
        if check_id:
            query += " AND check_id = ?"
            parameters.append(check_id)
        query += " ORDER BY check_id, sample_index LIMIT ?"
        parameters.append(max(1, min(limit, 50)))
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            {"check_id": row["check_id"], "record": json.loads(row["record_json"])} for row in rows
        ]

    def list_executions(
        self, *, before_ingested_at: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM pipeline_execution"
        parameters: list[Any] = []
        if before_ingested_at:
            query += " WHERE ingestion_timestamp < ?"
            parameters.append(before_ingested_at)
        query += " ORDER BY ingestion_timestamp DESC LIMIT ?"
        parameters.append(max(1, min(limit, 100)))
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [_execution_dict(row) for row in rows]

    def start_investigation(
        self, investigation_id: str, incident_id: str, started_at: datetime
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO investigation_session
                   (investigation_id, incident_id, status, started_at)
                   VALUES (?, ?, 'running', ?)""",
                (investigation_id, incident_id, started_at.isoformat()),
            )

    def record_tool_call(
        self,
        investigation_id: str,
        sequence: int,
        tool_name: str,
        arguments: dict[str, Any],
        rationale: str,
        evidence: Evidence,
        started_at: datetime,
        completed_at: datetime,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO agent_tool_call
                   (investigation_id, sequence, tool_name, arguments_json, rationale,
                    success, error, started_at, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    investigation_id,
                    sequence,
                    tool_name,
                    _dump(arguments),
                    rationale,
                    int(evidence.success),
                    evidence.error,
                    started_at.isoformat(),
                    completed_at.isoformat(),
                ),
            )
            connection.execute(
                """INSERT INTO evidence
                   (evidence_id, investigation_id, tool_sequence, tool_name,
                    summary, data_json, success, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    evidence.evidence_id,
                    investigation_id,
                    sequence,
                    evidence.tool_name,
                    evidence.summary,
                    _dump(evidence.data),
                    int(evidence.success),
                    evidence.error,
                ),
            )

    def record_model_decision(
        self,
        investigation_id: str,
        sequence: int,
        decision: ModelDecision,
        created_at: datetime,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO agent_decision
                   (investigation_id, sequence, action, decision_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    investigation_id,
                    sequence,
                    decision.action,
                    _dump(decision),
                    created_at.isoformat(),
                ),
            )

    def record_model_rejection(
        self,
        investigation_id: str,
        sequence: int,
        rejection: RejectedDecisionDiagnostic,
        created_at: datetime,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO model_rejection
                   (investigation_id, sequence, attempt, error_type, error,
                    decision_json, content_length, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    investigation_id,
                    sequence,
                    rejection.attempt,
                    rejection.error_type,
                    rejection.error,
                    None if rejection.decision is None else _dump(rejection.decision),
                    rejection.content_length,
                    created_at.isoformat(),
                ),
            )

    def complete_investigation(self, report: InvestigationReport) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE investigation_session
                   SET status = ?, completed_at = ?, report_json = ?
                   WHERE investigation_id = ?""",
                (
                    report.status,
                    report.completed_at.isoformat(),
                    report.to_json(indent=0),
                    report.investigation_id,
                ),
            )

    def get_investigations(self, incident_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT investigation_id, status, started_at, completed_at, report_json
                   FROM investigation_session WHERE incident_id = ? ORDER BY started_at""",
                (incident_id,),
            ).fetchall()
        return [
            {
                "investigation_id": row["investigation_id"],
                "status": row["status"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "report": json.loads(row["report_json"]) if row["report_json"] else None,
            }
            for row in rows
        ]

    def get_investigation_trace(self, investigation_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            session = connection.execute(
                "SELECT * FROM investigation_session WHERE investigation_id = ?",
                (investigation_id,),
            ).fetchone()
            if session is None:
                return None
            decisions = connection.execute(
                """SELECT sequence, action, decision_json, created_at
                   FROM agent_decision WHERE investigation_id = ? ORDER BY sequence""",
                (investigation_id,),
            ).fetchall()
            tool_calls = connection.execute(
                """SELECT sequence, tool_name, arguments_json, rationale, success, error,
                          started_at, completed_at
                   FROM agent_tool_call WHERE investigation_id = ? ORDER BY sequence""",
                (investigation_id,),
            ).fetchall()
            evidence = connection.execute(
                """SELECT evidence_id, tool_sequence, tool_name, summary, data_json,
                          success, error
                   FROM evidence WHERE investigation_id = ? ORDER BY tool_sequence""",
                (investigation_id,),
            ).fetchall()
            rejections = connection.execute(
                """SELECT sequence, attempt, error_type, error, decision_json,
                          content_length, created_at
                   FROM model_rejection WHERE investigation_id = ? ORDER BY sequence""",
                (investigation_id,),
            ).fetchall()
        return {
            "investigation_id": investigation_id,
            "incident_id": session["incident_id"],
            "status": session["status"],
            "started_at": session["started_at"],
            "completed_at": session["completed_at"],
            "decisions": [
                {
                    "sequence": row["sequence"],
                    "action": row["action"],
                    "created_at": row["created_at"],
                    "decision": json.loads(row["decision_json"]),
                }
                for row in decisions
            ],
            "tool_calls": [
                {
                    "sequence": row["sequence"],
                    "tool_name": row["tool_name"],
                    "arguments": json.loads(row["arguments_json"]),
                    "rationale": row["rationale"],
                    "success": bool(row["success"]),
                    "error": row["error"],
                    "started_at": row["started_at"],
                    "completed_at": row["completed_at"],
                }
                for row in tool_calls
            ],
            "evidence": [
                {
                    "evidence_id": row["evidence_id"],
                    "tool_sequence": row["tool_sequence"],
                    "tool_name": row["tool_name"],
                    "summary": row["summary"],
                    "data": json.loads(row["data_json"]),
                    "success": bool(row["success"]),
                    "error": row["error"],
                }
                for row in evidence
            ],
            "rejected_model_decisions": [
                {
                    "sequence": row["sequence"],
                    "attempt": row["attempt"],
                    "error_type": row["error_type"],
                    "error": row["error"],
                    "decision": (
                        json.loads(row["decision_json"])
                        if row["decision_json"] is not None
                        else None
                    ),
                    "content_length": row["content_length"],
                    "created_at": row["created_at"],
                }
                for row in rejections
            ],
            "report": json.loads(session["report_json"]) if session["report_json"] else None,
        }


def _validation_row(execution_id: str, result: CheckResult) -> tuple[Any, ...]:
    finding = None if result.finding is None else _dump(result.finding)
    return (
        execution_id,
        result.check_id,
        result.kind.value,
        int(result.passed),
        result.severity.value,
        finding,
    )


def _build_profile(frame: pd.DataFrame) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for column_name in frame.columns:
        series = frame[column_name]
        summary: dict[str, Any] = {
            "dtype": str(series.dtype),
            "count": int(series.notna().sum()),
            "null_count": int(series.isna().sum()),
            "distinct_count": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric.empty:
                summary["numeric"] = {
                    "mean": float(numeric.mean()),
                    "std": float(numeric.std(ddof=0)),
                    "min": float(numeric.min()),
                    "p25": float(numeric.quantile(0.25)),
                    "median": float(numeric.median()),
                    "p75": float(numeric.quantile(0.75)),
                    "max": float(numeric.max()),
                }
        elif column_name != "order_id":
            counts = series.dropna().astype(str).value_counts().head(10)
            summary["top_values"] = {str(key): int(value) for key, value in counts.items()}
        columns[str(column_name)] = summary
    return {"row_count": len(frame), "columns": columns}


def _build_aggregate_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"row_count": len(frame), "dimensions": {}}
    for dimension in ("supplier_id", "product_category", "currency"):
        if dimension not in frame:
            continue
        groups: dict[str, Any] = {}
        for value, group in frame.groupby(dimension, dropna=False):
            metrics: dict[str, Any] = {"record_count": len(group)}
            if "quantity" in group:
                metrics["quantity_sum"] = _optional_float(group["quantity"].sum(min_count=1))
            if "unit_price" in group:
                metrics["unit_price_mean"] = _optional_float(group["unit_price"].mean())
                metrics["unit_price_std"] = _optional_float(group["unit_price"].std(ddof=0))
                metrics["unit_price_nulls"] = int(group["unit_price"].isna().sum())
            groups[str(value)] = metrics
        result["dimensions"][dimension] = groups
    return result


def _problem_samples(
    frame: pd.DataFrame, check_id: str, config: PipelineConfig
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    mask = pd.Series(False, index=frame.index)
    if check_id == "unique_order_id" and "order_id" in frame:
        mask = frame.duplicated("order_id", keep=False)
    elif check_id == "required_values":
        available = [column for column in config.required_columns if column in frame]
        mask = frame[available].isna().any(axis=1) if available else mask
    elif check_id == "supplier_currency" and {"supplier_id", "currency"}.issubset(frame.columns):
        expected = frame["supplier_id"].map(config.supplier_currencies)
        mask = expected.notna() & frame["currency"].ne(expected)
    elif check_id == "known_category" and "product_category" in frame:
        mask = ~frame["product_category"].isin(config.category_profiles)
    elif check_id == "known_supplier" and "supplier_id" in frame:
        mask = ~frame["supplier_id"].isin(config.supplier_currencies)
    elif check_id == "positive_sale_values" and {"quantity", "unit_price"}.issubset(frame.columns):
        mask = frame["quantity"].le(0) | frame["unit_price"].le(0)
    elif check_id == "unit_price_z_score" and {
        "product_category",
        "unit_price",
    }.issubset(frame.columns):
        means = frame["product_category"].map(
            {key: value.mean_price for key, value in config.category_profiles.items()}
        )
        stds = frame["product_category"].map(
            {key: value.price_std for key, value in config.category_profiles.items()}
        )
        mask = ((frame["unit_price"] - means).abs() / stds).gt(config.anomaly_z_score_threshold)
    elif check_id == "delivery_freshness":
        timestamps = pd.to_datetime(frame.get("event_timestamp"), utc=True, errors="coerce")
        indexes = timestamps.sort_values(ascending=False).head(10).index
        mask.loc[indexes] = True
    records = frame.loc[mask.fillna(False)].head(10).to_dict(orient="records")
    return [{str(key): _safe_value(value) for key, value in record.items()} for record in records]


def _execution_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "execution_id": row["execution_id"],
        "incident_id": row["incident_id"],
        "source": row["source"],
        "status": row["status"],
        "pipeline_stage": row["pipeline_stage"],
        "ingestion_timestamp": row["ingestion_timestamp"],
        "event_time_min": row["event_time_min"],
        "event_time_max": row["event_time_max"],
        "row_count": row["row_count"],
        "aggregate_metrics": json.loads(row["aggregate_metrics_json"]),
    }


def _dump(value: Any) -> str:
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        value = asdict(value)
    return json.dumps(_json_ready(value), sort_keys=True, allow_nan=False)


def _safe_value(value: Any) -> Any:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _optional_float(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)
