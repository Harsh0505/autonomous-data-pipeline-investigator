from __future__ import annotations

from retail_guardian.evaluation import BenchmarkEvaluator
from retail_guardian.investigation import InvestigationAgent
from retail_guardian.providers import HeuristicInvestigationModel
from retail_guardian.storage import SQLiteHistoryRepository


def test_benchmark_evaluates_all_hidden_failure_labels(tmp_path, as_of) -> None:
    repository = SQLiteHistoryRepository(tmp_path / "benchmark.db")
    evaluator = BenchmarkEvaluator(
        repository,
        lambda: InvestigationAgent(repository, HeuristicInvestigationModel()),
    )

    summary = evaluator.run(row_count=150, seed=11, as_of=as_of)

    assert len(summary.cases) == 6
    assert summary.accuracy == 1.0
    assert summary.evidence_support_rate == 1.0
    assert summary.completion_rate == 1.0
    assert summary.average_tool_calls == 2.0
    assert summary.total_tokens is None
    assert summary.baseline_accuracy == 1.0
    assert summary.evaluation_type == "deterministic_mock"
    assert summary.provider_name == "mock"
    assert summary.model_name == "heuristic"
    assert summary.average_duration_seconds >= 0
    assert summary.invalid_tool_calls == 0
    assert summary.model_validation_failures == 0
    assert summary.investigation_failures == 0
    assert summary.timeouts == 0
