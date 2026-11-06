from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

from retail_guardian.failures import FailureType
from retail_guardian.investigation import InvestigationAgent
from retail_guardian.pipeline import PipelineRun
from retail_guardian.scenarios import run_scenario
from retail_guardian.storage import HistoryRepository


@dataclass(frozen=True)
class EvaluationCase:
    incident_id: str
    expected_root_cause: str
    predicted_root_cause: str | None
    correct: bool
    evidence_supported: bool
    tool_calls: int
    token_usage: int | None
    completed: bool
    baseline_prediction: str | None
    baseline_correct: bool
    status: str
    duration_seconds: float
    invalid_tool_calls: int
    model_validation_failures: int
    investigation_failures: int
    timeouts: int
    provider_name: str
    model_name: str


@dataclass(frozen=True)
class EvaluationSummary:
    cases: list[EvaluationCase]
    accuracy: float
    evidence_support_rate: float
    completion_rate: float
    average_tool_calls: float
    total_tokens: int | None
    baseline_accuracy: float
    average_duration_seconds: float
    invalid_tool_calls: int
    model_validation_failures: int
    investigation_failures: int
    timeouts: int
    evaluation_type: str
    provider_name: str
    model_name: str

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "cases": [asdict(case) for case in self.cases],
        }


class BenchmarkEvaluator:
    """Evaluate investigations only after the agent has completed without truth access."""

    def __init__(
        self,
        repository: HistoryRepository,
        agent_factory: Callable[[], InvestigationAgent],
    ) -> None:
        self.repository = repository
        self.agent_factory = agent_factory

    def run(
        self,
        *,
        row_count: int = 500,
        seed: int = 42,
        as_of: datetime | None = None,
    ) -> EvaluationSummary:
        reference_time = as_of or datetime.now(UTC)
        ingestion_start = datetime.now(UTC)
        run_scenario(
            "healthy",
            row_count=row_count,
            seed=seed,
            as_of=reference_time,
            repository=self.repository,
            ingested_at=ingestion_start,
        )
        cases: list[EvaluationCase] = []
        for index, failure_type in enumerate(FailureType, start=1):
            run = run_scenario(
                failure_type.value,
                row_count=row_count,
                seed=seed,
                as_of=reference_time,
                repository=self.repository,
                ingested_at=ingestion_start + timedelta(microseconds=index),
            )
            cases.append(self._evaluate_run(run))
        return _summarize(cases)

    def _evaluate_run(self, run: PipelineRun) -> EvaluationCase:
        truth = run.benchmark_truth
        if truth is None or len(truth.failure_types) != 1:
            raise ValueError("evaluation requires exactly one hidden failure label")

        # Only the opaque incident ID crosses into the investigation boundary.
        report = self.agent_factory().investigate(run.report.incident_id)
        expected = truth.failure_types[0].value
        predicted = report.conclusion.diagnosis_code
        valid_evidence = {item.evidence_id for item in report.evidence if item.success}
        evidence_supported = (
            predicted == expected
            and bool(report.conclusion.evidence_references)
            and set(report.conclusion.evidence_references).issubset(valid_evidence)
        )
        baseline = validation_baseline(run)
        return EvaluationCase(
            incident_id=run.report.incident_id,
            expected_root_cause=expected,
            predicted_root_cause=predicted,
            correct=predicted == expected,
            evidence_supported=evidence_supported,
            tool_calls=report.tool_call_count,
            token_usage=report.token_usage,
            completed=report.status in {"completed", "inconclusive"},
            baseline_prediction=baseline,
            baseline_correct=baseline == expected,
            status=report.status,
            duration_seconds=report.duration_seconds,
            invalid_tool_calls=report.invalid_tool_calls,
            model_validation_failures=report.model_validation_failures,
            investigation_failures=report.investigation_failures,
            timeouts=report.timeouts,
            provider_name=report.provider_name,
            model_name=report.model_name,
        )


def validation_baseline(run: PipelineRun) -> str | None:
    """Simple non-agent baseline using only the first failed validation check."""
    mapping = {
        "schema_contract": "schema_change",
        "unique_order_id": "duplicate_records",
        "supplier_currency": "currency_mismatch",
        "required_values": "missing_values",
        "known_category": "unexpected_categories",
        "delivery_freshness": "delayed_data",
    }
    return next(
        (
            mapping[result.check_id]
            for result in run.validation_results
            if not result.passed and result.check_id in mapping
        ),
        None,
    )


def _summarize(cases: list[EvaluationCase]) -> EvaluationSummary:
    count = len(cases)
    tokens = [case.token_usage for case in cases if case.token_usage is not None]
    provider_names = {case.provider_name for case in cases}
    model_names = {case.model_name for case in cases}
    provider_name = next(iter(provider_names)) if len(provider_names) == 1 else "mixed"
    model_name = next(iter(model_names)) if len(model_names) == 1 else "mixed"
    evaluation_type = {
        "mock": "deterministic_mock",
        "ollama": "real_local_llm",
        "openai": "external_llm",
    }.get(provider_name, "unknown")
    return EvaluationSummary(
        cases=cases,
        accuracy=sum(case.correct for case in cases) / count,
        evidence_support_rate=sum(case.evidence_supported for case in cases) / count,
        completion_rate=sum(case.completed for case in cases) / count,
        average_tool_calls=sum(case.tool_calls for case in cases) / count,
        total_tokens=sum(tokens) if tokens else None,
        baseline_accuracy=sum(case.baseline_correct for case in cases) / count,
        average_duration_seconds=sum(case.duration_seconds for case in cases) / count,
        invalid_tool_calls=sum(case.invalid_tool_calls for case in cases),
        model_validation_failures=sum(case.model_validation_failures for case in cases),
        investigation_failures=sum(case.investigation_failures for case in cases),
        timeouts=sum(case.timeouts for case in cases),
        evaluation_type=evaluation_type,
        provider_name=provider_name,
        model_name=model_name,
    )
