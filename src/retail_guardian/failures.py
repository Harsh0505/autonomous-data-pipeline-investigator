from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

import numpy as np
import pandas as pd


class FailureType(StrEnum):
    SCHEMA_CHANGE = "schema_change"
    DUPLICATE_RECORDS = "duplicate_records"
    CURRENCY_MISMATCH = "currency_mismatch"
    MISSING_VALUES = "missing_values"
    UNEXPECTED_CATEGORIES = "unexpected_categories"
    DELAYED_DATA = "delayed_data"


@dataclass(frozen=True)
class FailureInjection:
    failure_type: FailureType
    affected_rows: int
    details: dict[str, object]


@dataclass(frozen=True)
class BenchmarkGroundTruth:
    """Hidden labels used by the evaluation harness, never exposed to investigators."""

    failure_types: tuple[FailureType, ...]
    injections: tuple[FailureInjection, ...]
    seed: int


def inject_failure(
    frame: pd.DataFrame,
    failure_type: FailureType,
    *,
    seed: int = 42,
    affected_rows: int = 5,
) -> tuple[pd.DataFrame, FailureInjection]:
    """Return a corrupted copy and reproducible metadata describing the mutation."""
    if frame.empty:
        raise ValueError("cannot inject a failure into an empty dataset")

    corrupted = frame.copy(deep=True)
    count = min(max(1, affected_rows), len(corrupted))
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(corrupted), size=count, replace=False))
    indexes = corrupted.index[positions]
    details: dict[str, object]

    if failure_type == FailureType.SCHEMA_CHANGE:
        corrupted = corrupted.rename(columns={"quantity": "units_sold"})
        count = len(corrupted)
        details = {"renamed_column": {"quantity": "units_sold"}}
    elif failure_type == FailureType.DUPLICATE_RECORDS:
        duplicates = corrupted.loc[indexes].copy()
        corrupted = pd.concat([corrupted, duplicates], ignore_index=True)
        details = {"duplicated_order_ids": duplicates["order_id"].tolist()}
    elif failure_type == FailureType.CURRENCY_MISMATCH:
        replacements = {"USD": "EUR", "EUR": "GBP", "GBP": "USD"}
        original = corrupted.loc[indexes, "currency"].copy()
        corrupted.loc[indexes, "currency"] = original.map(replacements)
        # Misinterpreting minor units is a common companion to a currency feed defect.
        corrupted.loc[indexes, "unit_price"] *= 20
        details = {"order_ids": corrupted.loc[indexes, "order_id"].tolist()}
    elif failure_type == FailureType.MISSING_VALUES:
        corrupted.loc[indexes, "unit_price"] = np.nan
        details = {"column": "unit_price", "order_ids": corrupted.loc[indexes, "order_id"].tolist()}
    elif failure_type == FailureType.UNEXPECTED_CATEGORIES:
        corrupted.loc[indexes, "product_category"] = "collectibles"
        details = {
            "category": "collectibles",
            "order_ids": corrupted.loc[indexes, "order_id"].tolist(),
        }
    elif failure_type == FailureType.DELAYED_DATA:
        corrupted["event_timestamp"] = pd.to_datetime(
            corrupted["event_timestamp"], utc=True
        ) - timedelta(hours=72)
        count = len(corrupted)
        details = {"delay_hours": 72}
    else:  # pragma: no cover - StrEnum guards normal callers
        raise ValueError(f"unsupported failure type: {failure_type}")

    return corrupted, FailureInjection(failure_type, count, details)


def inject_all_failures(
    frame: pd.DataFrame, *, seed: int = 42
) -> tuple[pd.DataFrame, list[FailureInjection]]:
    corrupted = frame.copy(deep=True)
    injections: list[FailureInjection] = []
    for offset, failure_type in enumerate(FailureType):
        corrupted, injection = inject_failure(
            corrupted, failure_type, seed=seed + offset, affected_rows=5
        )
        injections.append(injection)
    return corrupted, injections
