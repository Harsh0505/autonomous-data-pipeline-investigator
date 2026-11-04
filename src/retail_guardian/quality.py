from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

import pandas as pd

from retail_guardian.config import PipelineConfig
from retail_guardian.models import CheckKind, CheckResult, Finding, Severity


class DataQualityEngine:
    """Run deterministic contract checks and statistical price anomaly detection."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def evaluate(self, frame: pd.DataFrame, *, as_of: datetime) -> list[CheckResult]:
        checks: list[Callable[[], CheckResult]] = [
            lambda: self._schema_check(frame),
            lambda: self._missing_values_check(frame),
            lambda: self._duplicate_check(frame),
            lambda: self._supplier_check(frame),
            lambda: self._currency_check(frame),
            lambda: self._category_check(frame),
            lambda: self._value_range_check(frame),
            lambda: self._freshness_check(frame, as_of),
            lambda: self._price_anomaly_check(frame),
        ]
        return [check() for check in checks]

    def _schema_check(self, frame: pd.DataFrame) -> CheckResult:
        expected = set(self.config.required_columns)
        actual = set(frame.columns)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        finding = None
        if missing or unexpected:
            finding = Finding(
                message="Dataset schema does not match the supplier contract",
                affected_rows=len(frame),
                context={"missing_columns": missing, "unexpected_columns": unexpected},
            )
        return _result("schema_contract", CheckKind.DETERMINISTIC, finding, Severity.CRITICAL)

    def _missing_values_check(self, frame: pd.DataFrame) -> CheckResult:
        available = [column for column in self.config.required_columns if column in frame]
        if not available:
            return _result("required_values", CheckKind.DETERMINISTIC, None, Severity.ERROR)
        missing_by_column = frame[available].isna().sum()
        failures = {column: int(count) for column, count in missing_by_column.items() if count}
        finding = None
        if failures:
            finding = Finding(
                message="Required fields contain missing values",
                affected_rows=int(frame[available].isna().any(axis=1).sum()),
                context={"missing_by_column": failures},
            )
        return _result("required_values", CheckKind.DETERMINISTIC, finding, Severity.ERROR)

    def _duplicate_check(self, frame: pd.DataFrame) -> CheckResult:
        if "order_id" not in frame:
            return _result("unique_order_id", CheckKind.DETERMINISTIC, None, Severity.ERROR)
        mask = frame.duplicated("order_id", keep=False)
        finding = None
        if mask.any():
            values = frame.loc[mask, "order_id"].drop_duplicates().head(10).tolist()
            finding = Finding(
                message="Duplicate order identifiers were received",
                affected_rows=int(mask.sum()),
                sample_values=values,
                context={"duplicate_id_count": int(frame.loc[mask, "order_id"].nunique())},
            )
        return _result("unique_order_id", CheckKind.DETERMINISTIC, finding, Severity.ERROR)

    def _supplier_check(self, frame: pd.DataFrame) -> CheckResult:
        if "supplier_id" not in frame:
            return _result("known_supplier", CheckKind.DETERMINISTIC, None, Severity.ERROR)
        mask = ~frame["supplier_id"].isin(self.config.supplier_currencies)
        finding = _mask_finding(frame, mask, "supplier_id", "Unknown suppliers were received")
        return _result("known_supplier", CheckKind.DETERMINISTIC, finding, Severity.ERROR)

    def _currency_check(self, frame: pd.DataFrame) -> CheckResult:
        required = {"supplier_id", "currency"}
        if not required.issubset(frame.columns):
            return _result("supplier_currency", CheckKind.DETERMINISTIC, None, Severity.ERROR)
        expected = frame["supplier_id"].map(self.config.supplier_currencies)
        mask = expected.notna() & frame["currency"].ne(expected)
        finding = None
        if mask.any():
            samples = (
                frame.loc[mask, ["supplier_id", "currency"]]
                .drop_duplicates()
                .head(10)
                .to_dict(orient="records")
            )
            finding = Finding(
                message="Supplier currency differs from its contract",
                affected_rows=int(mask.sum()),
                sample_values=samples,
            )
        return _result("supplier_currency", CheckKind.DETERMINISTIC, finding, Severity.CRITICAL)

    def _category_check(self, frame: pd.DataFrame) -> CheckResult:
        if "product_category" not in frame:
            return _result("known_category", CheckKind.DETERMINISTIC, None, Severity.ERROR)
        mask = ~frame["product_category"].isin(self.config.category_profiles)
        finding = _mask_finding(
            frame, mask, "product_category", "Unexpected product categories were received"
        )
        return _result("known_category", CheckKind.DETERMINISTIC, finding, Severity.ERROR)

    def _value_range_check(self, frame: pd.DataFrame) -> CheckResult:
        required = {"quantity", "unit_price"}
        if not required.issubset(frame.columns):
            return _result("positive_sale_values", CheckKind.DETERMINISTIC, None, Severity.ERROR)
        mask = frame["quantity"].le(0) | frame["unit_price"].le(0)
        finding = None
        if mask.fillna(False).any():
            finding = Finding(
                message="Sale quantities and prices must be positive",
                affected_rows=int(mask.fillna(False).sum()),
                sample_values=frame.loc[mask.fillna(False), "order_id"].head(10).tolist(),
            )
        return _result("positive_sale_values", CheckKind.DETERMINISTIC, finding, Severity.ERROR)

    def _freshness_check(self, frame: pd.DataFrame, as_of: datetime) -> CheckResult:
        if "event_timestamp" not in frame or frame.empty:
            return _result("delivery_freshness", CheckKind.DETERMINISTIC, None, Severity.CRITICAL)
        timestamps = pd.to_datetime(frame["event_timestamp"], utc=True, errors="coerce")
        latest = timestamps.max()
        maximum_age = timedelta(hours=self.config.maximum_delivery_delay_hours)
        is_stale = pd.isna(latest) or pd.Timestamp(as_of) - latest > maximum_age
        finding = None
        if is_stale:
            age_hours = (
                None
                if pd.isna(latest)
                else round((pd.Timestamp(as_of) - latest).total_seconds() / 3600, 2)
            )
            finding = Finding(
                message="Latest supplier data arrived outside the freshness SLA",
                affected_rows=len(frame),
                context={
                    "latest_event_timestamp": None if pd.isna(latest) else latest.isoformat(),
                    "age_hours": age_hours,
                    "maximum_age_hours": self.config.maximum_delivery_delay_hours,
                },
            )
        return _result("delivery_freshness", CheckKind.DETERMINISTIC, finding, Severity.CRITICAL)

    def _price_anomaly_check(self, frame: pd.DataFrame) -> CheckResult:
        required = {"product_category", "unit_price"}
        if not required.issubset(frame.columns):
            return _result("unit_price_z_score", CheckKind.STATISTICAL, None, Severity.WARNING)
        category_mean = frame["product_category"].map(
            {name: profile.mean_price for name, profile in self.config.category_profiles.items()}
        )
        category_std = frame["product_category"].map(
            {name: profile.price_std for name, profile in self.config.category_profiles.items()}
        )
        z_scores = (frame["unit_price"] - category_mean).abs() / category_std
        mask = z_scores.gt(self.config.anomaly_z_score_threshold) & category_mean.notna()
        finding = None
        if mask.any():
            samples = []
            for index in frame.index[mask][:10]:
                samples.append(
                    {
                        "order_id": (
                            frame.at[index, "order_id"] if "order_id" in frame else str(index)
                        ),
                        "category": frame.at[index, "product_category"],
                        "unit_price": float(frame.at[index, "unit_price"]),
                        "z_score": round(float(z_scores.at[index]), 2),
                    }
                )
            finding = Finding(
                message="Unit prices exceed the category statistical baseline",
                affected_rows=int(mask.sum()),
                sample_values=samples,
                context={"z_score_threshold": self.config.anomaly_z_score_threshold},
            )
        return _result("unit_price_z_score", CheckKind.STATISTICAL, finding, Severity.WARNING)


def _result(
    check_id: str,
    kind: CheckKind,
    finding: Finding | None,
    severity: Severity,
) -> CheckResult:
    return CheckResult(check_id, kind, finding is None, severity, finding)


def _mask_finding(
    frame: pd.DataFrame, mask: pd.Series, column: str, message: str
) -> Finding | None:
    if not mask.fillna(False).any():
        return None
    return Finding(
        message=message,
        affected_rows=int(mask.fillna(False).sum()),
        sample_values=frame.loc[mask.fillna(False), column].drop_duplicates().head(10).tolist(),
    )
