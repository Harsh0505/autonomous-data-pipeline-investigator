from __future__ import annotations

import json

import pytest

from retail_guardian.config import PipelineConfig
from retail_guardian.failures import FailureType
from retail_guardian.pipeline import RetailSalesPipeline
from retail_guardian.scenarios import run_scenario
from retail_guardian.synthetic import SyntheticSalesGenerator


def test_healthy_pipeline_transforms_and_aggregates(config: PipelineConfig, as_of) -> None:
    frame = SyntheticSalesGenerator(config).generate(300, seed=8, as_of=as_of)
    run = RetailSalesPipeline(config).run(frame, source_name="test_feed", as_of=as_of)

    assert run.report.status == "healthy"
    assert run.report.pipeline_stage == "aggregation"
    assert run.transformed is not None
    assert run.aggregates is not None
    assert len(run.transformed) == 300
    assert run.aggregates["order_count"].sum() == 300
    assert run.transformed["gross_revenue_gbp"].sum() == pytest.approx(
        run.aggregates["gross_revenue_gbp"].sum(), abs=0.1
    )


@pytest.mark.parametrize("scenario", [item.value for item in FailureType])
def test_corrupted_scenario_returns_structured_incident(scenario: str, as_of) -> None:
    run = run_scenario(scenario, row_count=200, seed=12, as_of=as_of)

    assert run.report.status == "incident"
    assert run.report.violations
    assert run.report.pipeline_error == "Validation gate rejected the dataset"
    assert run.transformed is None
    assert run.benchmark_truth is not None
    assert run.benchmark_truth.failure_types[0].value == scenario
    assert "failure_context" not in run.report.to_dict()
    assert "scenario" not in run.report.to_dict()


def test_incident_report_is_json_serializable(tmp_path, as_of) -> None:
    run = run_scenario("all_failures", row_count=100, seed=4, as_of=as_of)
    path = tmp_path / "report.json"
    run.report.write(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "2.0"
    assert payload["incident_id"].startswith("INC-")
    assert payload["dataset"]["row_count"] == 105
    assert "failure_context" not in payload
    assert "scenario" not in payload
    assert run.benchmark_truth is not None
    assert len(run.benchmark_truth.injections) == 6
    assert {item["kind"] for item in payload["violations"]} >= {"deterministic"}


def test_csv_ingestion(config: PipelineConfig, as_of, tmp_path) -> None:
    frame = SyntheticSalesGenerator(config).generate(25, as_of=as_of)
    path = tmp_path / "sales.csv"
    frame.to_csv(path, index=False)

    ingested = RetailSalesPipeline(config).ingest(path)

    assert len(ingested) == 25
    assert list(ingested.columns) == list(frame.columns)


def test_unsupported_ingestion_format(config: PipelineConfig, tmp_path) -> None:
    path = tmp_path / "sales.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        RetailSalesPipeline(config).ingest(path)
