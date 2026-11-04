from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from retail_guardian.config import PipelineConfig
from retail_guardian.failures import BenchmarkGroundTruth
from retail_guardian.models import CheckResult, IncidentReport, Severity
from retail_guardian.quality import DataQualityEngine


@dataclass(frozen=True)
class PipelineRun:
    execution_id: str
    report: IncidentReport
    validation_results: list[CheckResult]
    transformed: pd.DataFrame | None
    aggregates: pd.DataFrame | None
    benchmark_truth: BenchmarkGroundTruth | None = None


class RetailSalesPipeline:
    """Ingest, validate, transform, and aggregate supplier sales."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.quality = DataQualityEngine(self.config)

    def ingest(self, source: pd.DataFrame | Path) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return source.copy(deep=True)
        if source.suffix.lower() != ".csv":
            raise ValueError(f"unsupported input format: {source.suffix}")
        if not source.is_file():
            raise FileNotFoundError(source)
        return pd.read_csv(source)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = frame.copy(deep=True)
        transformed["event_timestamp"] = pd.to_datetime(
            transformed["event_timestamp"], utc=True, errors="raise"
        )
        transformed["gross_revenue"] = transformed["quantity"] * transformed["unit_price"]
        transformed["fx_rate_to_gbp"] = transformed["currency"].map(self.config.fx_to_gbp)
        if transformed["fx_rate_to_gbp"].isna().any():
            missing_rate = transformed["fx_rate_to_gbp"].isna()
            unsupported = sorted(transformed.loc[missing_rate, "currency"].unique())
            raise ValueError(f"unsupported currencies: {unsupported}")
        transformed["gross_revenue_gbp"] = (
            transformed["gross_revenue"] * transformed["fx_rate_to_gbp"]
        ).round(2)
        transformed["sale_date"] = transformed["event_timestamp"].dt.date
        return transformed

    def aggregate(self, transformed: pd.DataFrame) -> pd.DataFrame:
        aggregates = (
            transformed.groupby(["sale_date", "supplier_id", "product_category"], as_index=False)
            .agg(
                order_count=("order_id", "nunique"),
                units_sold=("quantity", "sum"),
                gross_revenue_gbp=("gross_revenue_gbp", "sum"),
            )
            .sort_values(["sale_date", "supplier_id", "product_category"])
            .reset_index(drop=True)
        )
        aggregates["average_order_value_gbp"] = (
            aggregates["gross_revenue_gbp"] / aggregates["order_count"]
        ).round(2)
        aggregates["gross_revenue_gbp"] = aggregates["gross_revenue_gbp"].round(2)
        return aggregates

    def run(
        self,
        source: pd.DataFrame | Path,
        *,
        source_name: str = "external_supplier_feed",
        as_of: datetime | None = None,
    ) -> PipelineRun:
        reference_time = as_of or datetime.now(UTC)
        if reference_time.tzinfo is None:
            raise ValueError("as_of must include a timezone")

        frame = self.ingest(source)
        execution_id = _execution_id(frame, reference_time)
        results = self.quality.evaluate(frame, as_of=reference_time)
        violations = [result for result in results if not result.passed]
        blocking = [
            result
            for result in violations
            if result.severity in {Severity.ERROR, Severity.CRITICAL}
        ]
        transformed: pd.DataFrame | None = None
        aggregates: pd.DataFrame | None = None
        pipeline_error: str | None = None
        stage = "validation"

        if not blocking:
            try:
                stage = "transformation"
                transformed = self.transform(frame)
                stage = "aggregation"
                aggregates = self.aggregate(transformed)
            except (KeyError, TypeError, ValueError) as exc:
                pipeline_error = str(exc)

        status = "healthy" if not violations and pipeline_error is None else "incident"
        if blocking:
            pipeline_error = "Validation gate rejected the dataset"

        report = IncidentReport(
            incident_id=f"INC-{execution_id.removeprefix('RUN-')}",
            execution_id=execution_id,
            source=source_name,
            status=status,
            detected_at=reference_time,
            pipeline_stage=stage,
            dataset=_dataset_summary(frame),
            violations=violations,
            metrics={
                "checks_run": len(results),
                "checks_failed": len(violations),
                "blocking_violations": len(blocking),
                "aggregate_rows": 0 if aggregates is None else len(aggregates),
            },
            pipeline_error=pipeline_error,
        )
        return PipelineRun(execution_id, report, results, transformed, aggregates)


def _execution_id(frame: pd.DataFrame, detected_at: datetime) -> str:
    canonical = frame.copy()
    canonical.columns = canonical.columns.astype(str)
    content_hash = pd.util.hash_pandas_object(canonical.astype(str), index=True).values.tobytes()
    schema_hash = "|".join(canonical.columns).encode()
    digest = sha256(detected_at.isoformat().encode() + schema_hash + content_hash)
    digest = digest.hexdigest()[:12].upper()
    return f"RUN-{digest}"


def _dataset_summary(frame: pd.DataFrame) -> dict[str, Any]:
    timestamps = None
    if "event_timestamp" in frame:
        timestamps = pd.to_datetime(frame["event_timestamp"], utc=True, errors="coerce")
    has_timestamps = timestamps is not None and not timestamps.isna().all()
    return {
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "columns": frame.columns.tolist(),
        "supplier_count": int(frame["supplier_id"].nunique()) if "supplier_id" in frame else 0,
        "event_time_min": timestamps.min().isoformat() if has_timestamps else None,
        "event_time_max": timestamps.max().isoformat() if has_timestamps else None,
    }
