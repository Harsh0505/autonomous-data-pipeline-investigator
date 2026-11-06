from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import TYPE_CHECKING, Any

import pandas as pd

from retail_guardian.config import PipelineConfig
from retail_guardian.failures import FailureInjection, FailureType, inject_failure
from retail_guardian.pipeline import PipelineRun, RetailSalesPipeline
from retail_guardian.synthetic import SyntheticSalesGenerator

if TYPE_CHECKING:
    from retail_guardian.storage import HistoryRepository


class OperationalRecordType(StrEnum):
    PIPELINE_LOG = "pipeline_log"
    SUPPLIER_SUBMISSION = "supplier_submission"
    ORCHESTRATION_EVENT = "orchestration_event"
    TRANSFORMATION_METADATA = "transformation_metadata"
    CONFIGURATION_CHANGE = "configuration_change"


class RootCause(StrEnum):
    SUPPLIER_BATCH_REPLAY = "supplier_batch_replay"
    FAILED_JOB_RETRY = "failed_job_retry"
    INCORRECT_JOIN_CARDINALITY = "incorrect_join_cardinality"
    SOURCE_SYSTEM_OMISSION = "source_system_omission"
    TRANSFORMATION_LOGIC_ERROR = "transformation_logic_error"
    SUPPLIER_DATA_ERROR = "supplier_data_error"
    LEGITIMATE_TAXONOMY_UPDATE = "legitimate_taxonomy_update"
    LATE_SUPPLIER_DELIVERY = "late_supplier_delivery"
    SCHEDULER_BACKLOG = "scheduler_backlog"


@dataclass(frozen=True)
class OperationalRecord:
    record_id: str
    record_type: OperationalRecordType
    occurred_at: datetime
    component: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "record_type": self.record_type.value,
            "occurred_at": self.occurred_at.isoformat(),
            "component": self.component,
            "details": self.details,
        }


@dataclass(frozen=True)
class RootCauseGroundTruth:
    """Evaluation-only causal label that never crosses the repository boundary."""

    root_cause: RootCause | None
    observable_failure: FailureType
    required_evidence_tools: tuple[str, ...]
    seed: int


@dataclass(frozen=True)
class RootCauseScenarioRun:
    pipeline_run: PipelineRun
    ground_truth: RootCauseGroundTruth


@dataclass(frozen=True)
class _CaseSpec:
    observable_failure: FailureType
    root_cause: RootCause | None
    required_evidence_tools: tuple[str, ...]


ROOT_CAUSE_CASES = tuple(f"case-{index:03d}" for index in range(1, 11))

_CASE_SPECS = {
    "case-001": _CaseSpec(
        FailureType.DUPLICATE_RECORDS,
        RootCause.SUPPLIER_BATCH_REPLAY,
        ("get_supplier_submissions",),
    ),
    "case-002": _CaseSpec(
        FailureType.DUPLICATE_RECORDS,
        RootCause.FAILED_JOB_RETRY,
        ("get_orchestration_events", "get_pipeline_logs"),
    ),
    "case-003": _CaseSpec(
        FailureType.DUPLICATE_RECORDS,
        RootCause.INCORRECT_JOIN_CARDINALITY,
        ("get_transformation_metadata",),
    ),
    "case-004": _CaseSpec(
        FailureType.MISSING_VALUES,
        RootCause.SOURCE_SYSTEM_OMISSION,
        ("get_supplier_submissions", "get_transformation_metadata"),
    ),
    "case-005": _CaseSpec(
        FailureType.MISSING_VALUES,
        RootCause.TRANSFORMATION_LOGIC_ERROR,
        ("get_supplier_submissions", "get_transformation_metadata"),
    ),
    "case-006": _CaseSpec(
        FailureType.UNEXPECTED_CATEGORIES,
        RootCause.SUPPLIER_DATA_ERROR,
        ("get_supplier_submissions", "get_configuration_changes"),
    ),
    "case-007": _CaseSpec(
        FailureType.UNEXPECTED_CATEGORIES,
        RootCause.LEGITIMATE_TAXONOMY_UPDATE,
        ("get_configuration_changes", "get_supplier_submissions"),
    ),
    "case-008": _CaseSpec(
        FailureType.DELAYED_DATA,
        RootCause.LATE_SUPPLIER_DELIVERY,
        ("get_supplier_submissions", "get_orchestration_events"),
    ),
    "case-009": _CaseSpec(
        FailureType.DELAYED_DATA,
        RootCause.SCHEDULER_BACKLOG,
        ("get_supplier_submissions", "get_orchestration_events"),
    ),
    "case-010": _CaseSpec(
        FailureType.DUPLICATE_RECORDS,
        None,
        (
            "get_supplier_submissions",
            "get_orchestration_events",
            "get_transformation_metadata",
        ),
    ),
}


def run_root_cause_scenario(
    case_id: str,
    *,
    row_count: int = 500,
    seed: int = 42,
    as_of: datetime,
    config: PipelineConfig | None = None,
    repository: HistoryRepository | None = None,
    ingested_at: datetime | None = None,
) -> RootCauseScenarioRun:
    """Create one opaque causal case and persist only its observable facts."""
    try:
        spec = _CASE_SPECS[case_id]
    except KeyError as exc:
        raise ValueError(f"unknown root-cause case: {case_id}") from exc

    pipeline_config = config or PipelineConfig()
    frame = SyntheticSalesGenerator(pipeline_config).generate(row_count, seed=seed, as_of=as_of)
    supplier_id = str(frame["supplier_id"].value_counts().index[0])
    frame = frame.loc[frame["supplier_id"] == supplier_id].reset_index(drop=True)
    corrupted, injection = inject_failure(
        frame,
        spec.observable_failure,
        seed=seed + int(case_id[-3:]) * 17,
        affected_rows=5,
    )
    run = RetailSalesPipeline(pipeline_config).run(
        corrupted,
        source_name="synthetic_supplier_feed",
        as_of=as_of,
    )
    records = _operational_records(run, corrupted, injection, spec.root_cause, as_of)
    if repository is not None:
        repository.save_pipeline_run(run, corrupted, ingested_at=ingested_at)
        repository.save_operational_records(
            run.execution_id, [record.to_dict() for record in records]
        )
    truth = RootCauseGroundTruth(
        spec.root_cause,
        spec.observable_failure,
        spec.required_evidence_tools,
        seed,
    )
    return RootCauseScenarioRun(run, truth)


def _operational_records(
    run: PipelineRun,
    frame: pd.DataFrame,
    injection: FailureInjection,
    root_cause: RootCause | None,
    as_of: datetime,
) -> list[OperationalRecord]:
    affected = _affected_records(frame, injection)
    supplier_id = str(affected.iloc[0]["supplier_id"]) if not affected.empty else "atlas_us"
    batch_id = f"batch-{as_of:%Y%m%d}-{supplier_id}"
    checksum = sha256(
        "|".join(sorted(str(value) for value in affected.get("order_id", []))).encode()
    ).hexdigest()[:16]
    records: list[tuple[OperationalRecordType, int, str, dict[str, Any]]] = [
        (
            OperationalRecordType.PIPELINE_LOG,
            -8,
            "ingestion-gateway",
            {
                "level": "info",
                "event": "execution_started",
                "execution_id": run.execution_id,
                "worker": "ingest-03",
            },
        ),
        (
            OperationalRecordType.CONFIGURATION_CHANGE,
            -180,
            "feature-flags",
            {
                "change_id": "CFG-observability-sampling",
                "status": "applied",
                "setting": "trace_sample_rate",
                "old_value": 0.1,
                "new_value": 0.15,
            },
        ),
    ]
    records.extend(
        _cause_records(root_cause, frame, affected, supplier_id, batch_id, checksum, as_of)
    )
    return [
        _record(
            run.execution_id,
            record_type,
            index,
            as_of + timedelta(minutes=offset),
            component,
            details,
        )
        for index, (record_type, offset, component, details) in enumerate(records)
    ]


def _cause_records(
    root_cause: RootCause | None,
    frame: pd.DataFrame,
    affected: pd.DataFrame,
    supplier_id: str,
    batch_id: str,
    checksum: str,
    as_of: datetime,
) -> list[tuple[OperationalRecordType, int, str, dict[str, Any]]]:
    row_count = len(frame)
    affected_count = max(1, len(affected))
    if root_cause == RootCause.SUPPLIER_BATCH_REPLAY:
        submission = {
            "supplier_id": supplier_id,
            "batch_id": batch_id,
            "checksum": checksum,
            "row_count": affected_count,
            "status": "accepted",
        }
        return [
            (OperationalRecordType.SUPPLIER_SUBMISSION, -42, "supplier-api", submission),
            (OperationalRecordType.SUPPLIER_SUBMISSION, -37, "supplier-api", submission),
            (
                OperationalRecordType.PIPELINE_LOG,
                -36,
                "ingestion-gateway",
                {"event": "batch_accepted", "batch_id": batch_id, "checksum": checksum},
            ),
            (
                OperationalRecordType.PIPELINE_LOG,
                -35,
                "ingestion-gateway",
                {"event": "batch_accepted", "batch_id": batch_id, "checksum": checksum},
            ),
            (
                OperationalRecordType.ORCHESTRATION_EVENT,
                -10,
                "retail-sales-dag",
                {"attempt": 1, "state": "succeeded", "retry_count": 0},
            ),
        ]
    if root_cause == RootCause.FAILED_JOB_RETRY:
        return [
            (
                OperationalRecordType.SUPPLIER_SUBMISSION,
                -45,
                "supplier-api",
                {
                    "supplier_id": supplier_id,
                    "batch_id": batch_id,
                    "checksum": checksum,
                    "row_count": row_count - affected_count,
                    "status": "accepted",
                },
            ),
            (
                OperationalRecordType.PIPELINE_LOG,
                -18,
                "orders-sink",
                {"event": "rows_committed", "attempt": 1, "committed_rows": affected_count},
            ),
            (
                OperationalRecordType.ORCHESTRATION_EVENT,
                -17,
                "retail-sales-dag",
                {
                    "attempt": 1,
                    "state": "failed",
                    "failure_stage": "post_write_ack",
                    "retry_scheduled": True,
                },
            ),
            (
                OperationalRecordType.ORCHESTRATION_EVENT,
                -12,
                "retail-sales-dag",
                {"attempt": 2, "state": "succeeded", "trigger": "automatic_retry"},
            ),
            (
                OperationalRecordType.PIPELINE_LOG,
                -11,
                "orders-sink",
                {"event": "rows_committed", "attempt": 2, "committed_rows": row_count},
            ),
        ]
    if root_cause == RootCause.INCORRECT_JOIN_CARDINALITY:
        return [
            (
                OperationalRecordType.SUPPLIER_SUBMISSION,
                -40,
                "supplier-api",
                {
                    "supplier_id": supplier_id,
                    "batch_id": batch_id,
                    "checksum": checksum,
                    "row_count": row_count - affected_count,
                    "status": "accepted",
                },
            ),
            (
                OperationalRecordType.TRANSFORMATION_METADATA,
                -15,
                "orders-customer-enrichment",
                {
                    "operation": "join",
                    "join_key": "customer_country",
                    "declared_cardinality": "many_to_one",
                    "observed_cardinality": "many_to_many",
                    "right_key_unique": False,
                    "input_rows": row_count - affected_count,
                    "output_rows": row_count,
                },
            ),
            (
                OperationalRecordType.ORCHESTRATION_EVENT,
                -10,
                "retail-sales-dag",
                {"attempt": 1, "state": "succeeded", "retry_count": 0},
            ),
        ]
    if root_cause == RootCause.SOURCE_SYSTEM_OMISSION:
        return [
            (
                OperationalRecordType.SUPPLIER_SUBMISSION,
                -38,
                "supplier-api",
                {
                    "supplier_id": supplier_id,
                    "batch_id": batch_id,
                    "row_count": row_count,
                    "field_statistics": {"unit_price": {"null_count": affected_count}},
                    "status": "accepted_with_warnings",
                },
            ),
            (
                OperationalRecordType.TRANSFORMATION_METADATA,
                -14,
                "sales-normalizer",
                {
                    "operation": "unit_price_passthrough",
                    "input_null_count": affected_count,
                    "output_null_count": affected_count,
                    "conversion_errors": 0,
                },
            ),
        ]
    if root_cause == RootCause.TRANSFORMATION_LOGIC_ERROR:
        return [
            (
                OperationalRecordType.SUPPLIER_SUBMISSION,
                -38,
                "supplier-api",
                {
                    "supplier_id": supplier_id,
                    "batch_id": batch_id,
                    "row_count": row_count,
                    "field_statistics": {"unit_price": {"null_count": 0}},
                    "status": "accepted",
                },
            ),
            (
                OperationalRecordType.TRANSFORMATION_METADATA,
                -14,
                "sales-normalizer",
                {
                    "operation": "parse_unit_price",
                    "build": "normalizer-2.14.0",
                    "input_null_count": 0,
                    "output_null_count": affected_count,
                    "conversion_errors": affected_count,
                    "error_pattern": "decimal_comma_not_supported",
                },
            ),
            (
                OperationalRecordType.CONFIGURATION_CHANGE,
                -75,
                "deployment-controller",
                {
                    "change_id": "DEP-normalizer-2.14.0",
                    "status": "deployed",
                    "component": "sales-normalizer",
                    "old_build": "2.13.4",
                    "new_build": "2.14.0",
                },
            ),
        ]
    if root_cause == RootCause.SUPPLIER_DATA_ERROR:
        return [
            (
                OperationalRecordType.SUPPLIER_SUBMISSION,
                -37,
                "supplier-api",
                {
                    "supplier_id": supplier_id,
                    "batch_id": batch_id,
                    "row_count": row_count,
                    "category_counts": {"collectibles": affected_count},
                    "supplier_schema_version": "2026.10",
                    "status": "accepted_with_warnings",
                    "warning": "category_not_in_registered_manifest",
                },
            ),
            (
                OperationalRecordType.CONFIGURATION_CHANGE,
                -1440,
                "taxonomy-service",
                {"change_id": "CFG-home-alias", "status": "applied", "category": "home"},
            ),
        ]
    if root_cause == RootCause.LEGITIMATE_TAXONOMY_UPDATE:
        return [
            (
                OperationalRecordType.CONFIGURATION_CHANGE,
                -240,
                "taxonomy-service",
                {
                    "change_id": "TAX-2841",
                    "status": "approved",
                    "effective_at": (as_of - timedelta(hours=2)).isoformat(),
                    "added_categories": ["collectibles"],
                    "target_version": "taxonomy-19",
                },
            ),
            (
                OperationalRecordType.CONFIGURATION_CHANGE,
                -20,
                "pipeline-runtime",
                {
                    "event": "configuration_loaded",
                    "loaded_version": "taxonomy-18",
                    "latest_approved_version": "taxonomy-19",
                },
            ),
            (
                OperationalRecordType.SUPPLIER_SUBMISSION,
                -35,
                "supplier-api",
                {
                    "supplier_id": supplier_id,
                    "batch_id": batch_id,
                    "row_count": row_count,
                    "category_counts": {"collectibles": affected_count},
                    "declared_taxonomy_version": "taxonomy-19",
                    "status": "accepted",
                },
            ),
        ]
    if root_cause == RootCause.LATE_SUPPLIER_DELIVERY:
        return [
            (
                OperationalRecordType.SUPPLIER_SUBMISSION,
                -12,
                "supplier-api",
                {
                    "supplier_id": supplier_id,
                    "batch_id": batch_id,
                    "generated_at": (as_of - timedelta(hours=72)).isoformat(),
                    "expected_by": (as_of - timedelta(hours=70)).isoformat(),
                    "received_at": (as_of - timedelta(minutes=12)).isoformat(),
                    "status": "accepted_late",
                },
            ),
            (
                OperationalRecordType.ORCHESTRATION_EVENT,
                -11,
                "retail-sales-dag",
                {"state": "started", "queue_wait_minutes": 1, "trigger": "file_arrival"},
            ),
            (
                OperationalRecordType.PIPELINE_LOG,
                -10,
                "ingestion-gateway",
                {"event": "freshness_warning", "event_age_hours": 72},
            ),
        ]
    if root_cause == RootCause.SCHEDULER_BACKLOG:
        received_at = as_of - timedelta(hours=72)
        return [
            (
                OperationalRecordType.SUPPLIER_SUBMISSION,
                -4320,
                "supplier-api",
                {
                    "supplier_id": supplier_id,
                    "batch_id": batch_id,
                    "generated_at": received_at.isoformat(),
                    "expected_by": (received_at + timedelta(hours=2)).isoformat(),
                    "received_at": received_at.isoformat(),
                    "status": "accepted_on_time",
                },
            ),
            (
                OperationalRecordType.ORCHESTRATION_EVENT,
                -4319,
                "retail-sales-dag",
                {"state": "queued", "queue": "supplier-ingestion", "priority": "normal"},
            ),
            (
                OperationalRecordType.ORCHESTRATION_EVENT,
                -15,
                "retail-sales-dag",
                {
                    "state": "started",
                    "queue_wait_minutes": 4304,
                    "scheduler_health": "backlogged",
                },
            ),
        ]
    return [
        (
            OperationalRecordType.SUPPLIER_SUBMISSION,
            -35,
            "supplier-api",
            {
                "supplier_id": supplier_id,
                "batch_id": batch_id,
                "row_count": row_count - affected_count,
                "status": "accepted",
            },
        ),
        (
            OperationalRecordType.PIPELINE_LOG,
            -9,
            "validation-gate",
            {"event": "duplicate_keys_detected", "affected_rows": affected_count},
        ),
        (
            OperationalRecordType.ORCHESTRATION_EVENT,
            -8,
            "retail-sales-dag",
            {"state": "metadata_unavailable", "reason": "retention_window_expired"},
        ),
        (
            OperationalRecordType.TRANSFORMATION_METADATA,
            -8,
            "lineage-catalog",
            {"status": "unavailable", "reason": "collector_disabled"},
        ),
    ]


def _affected_records(frame: pd.DataFrame, injection: FailureInjection) -> pd.DataFrame:
    order_ids = injection.details.get("order_ids") or injection.details.get("duplicated_order_ids")
    if isinstance(order_ids, list) and "order_id" in frame:
        return frame.loc[frame["order_id"].isin(order_ids)].drop_duplicates("order_id")
    return frame.head(min(injection.affected_rows, len(frame)))


def _record(
    execution_id: str,
    record_type: OperationalRecordType,
    ordinal: int,
    occurred_at: datetime,
    component: str,
    details: dict[str, Any],
) -> OperationalRecord:
    identity = json.dumps(
        [execution_id, record_type.value, ordinal, occurred_at.isoformat(), component, details],
        sort_keys=True,
    )
    record_id = f"OP-{sha256(identity.encode()).hexdigest()[:12].upper()}"
    return OperationalRecord(record_id, record_type, occurred_at, component, details)
