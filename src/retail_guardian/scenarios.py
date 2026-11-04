from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING

from retail_guardian.config import PipelineConfig
from retail_guardian.failures import (
    BenchmarkGroundTruth,
    FailureType,
    inject_all_failures,
    inject_failure,
)
from retail_guardian.pipeline import PipelineRun, RetailSalesPipeline
from retail_guardian.synthetic import SyntheticSalesGenerator

if TYPE_CHECKING:
    from retail_guardian.storage import HistoryRepository

HEALTHY_SCENARIO = "healthy"
ALL_FAILURES_SCENARIO = "all_failures"
SCENARIOS = (HEALTHY_SCENARIO, *(item.value for item in FailureType), ALL_FAILURES_SCENARIO)


def run_scenario(
    scenario: str,
    *,
    row_count: int = 500,
    seed: int = 42,
    as_of: datetime,
    config: PipelineConfig | None = None,
    repository: HistoryRepository | None = None,
    ingested_at: datetime | None = None,
) -> PipelineRun:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")

    pipeline_config = config or PipelineConfig()
    frame = SyntheticSalesGenerator(pipeline_config).generate(row_count, seed=seed, as_of=as_of)
    injections = []
    if scenario == ALL_FAILURES_SCENARIO:
        frame, injections = inject_all_failures(frame, seed=seed)
    elif scenario != HEALTHY_SCENARIO:
        frame, injection = inject_failure(frame, FailureType(scenario), seed=seed)
        injections = [injection]

    run = RetailSalesPipeline(pipeline_config).run(
        frame, source_name="synthetic_supplier_feed", as_of=as_of
    )
    truth = BenchmarkGroundTruth(
        failure_types=tuple(item.failure_type for item in injections),
        injections=tuple(injections),
        seed=seed,
    )
    run = replace(run, benchmark_truth=truth)
    if repository is not None:
        repository.save_pipeline_run(run, frame, ingested_at=ingested_at)
    return run
