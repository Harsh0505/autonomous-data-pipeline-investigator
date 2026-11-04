from __future__ import annotations

import pytest

from retail_guardian.config import PipelineConfig
from retail_guardian.failures import FailureType, inject_failure
from retail_guardian.quality import DataQualityEngine
from retail_guardian.synthetic import SyntheticSalesGenerator

EXPECTED_CHECK = {
    FailureType.SCHEMA_CHANGE: "schema_contract",
    FailureType.DUPLICATE_RECORDS: "unique_order_id",
    FailureType.CURRENCY_MISMATCH: "supplier_currency",
    FailureType.MISSING_VALUES: "required_values",
    FailureType.UNEXPECTED_CATEGORIES: "known_category",
    FailureType.DELAYED_DATA: "delivery_freshness",
}


@pytest.mark.parametrize("failure_type", list(FailureType))
def test_each_failure_is_reproducible_and_detected(
    failure_type: FailureType, config: PipelineConfig, as_of
) -> None:
    healthy = SyntheticSalesGenerator(config).generate(250, seed=15, as_of=as_of)
    first, first_metadata = inject_failure(healthy, failure_type, seed=9)
    second, second_metadata = inject_failure(healthy, failure_type, seed=9)

    assert first.equals(second)
    assert first_metadata == second_metadata
    failures = {
        result.check_id
        for result in DataQualityEngine(config).evaluate(first, as_of=as_of)
        if not result.passed
    }
    assert EXPECTED_CHECK[failure_type] in failures


def test_healthy_data_passes_all_checks(config: PipelineConfig, as_of) -> None:
    frame = SyntheticSalesGenerator(config).generate(1_000, seed=2, as_of=as_of)
    results = DataQualityEngine(config).evaluate(frame, as_of=as_of)

    assert len(results) == 9
    assert all(result.passed for result in results)


def test_statistical_price_anomaly_is_detected(config: PipelineConfig, as_of) -> None:
    frame = SyntheticSalesGenerator(config).generate(100, as_of=as_of)
    frame.loc[0, "unit_price"] = 50_000.0

    result = next(
        item
        for item in DataQualityEngine(config).evaluate(frame, as_of=as_of)
        if item.check_id == "unit_price_z_score"
    )

    assert not result.passed
    assert result.finding is not None
    assert result.finding.affected_rows == 1
