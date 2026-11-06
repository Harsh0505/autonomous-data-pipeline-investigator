from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

from retail_guardian.causal_evidence import (
    assess_causal_evidence,
    has_inconclusive_audit_evidence,
)
from retail_guardian.causal_scenarios import ROOT_CAUSE_CASES, run_root_cause_scenario
from retail_guardian.investigation import InvestigationAgent
from retail_guardian.scenarios import run_scenario
from retail_guardian.storage import HistoryRepository


@dataclass(frozen=True)
class RootCauseEvaluationCase:
    case_id: str
    incident_id: str
    observable_failure: str
    expected_root_cause: str | None
    predicted_root_cause: str | None
    correct: bool
    evidence_supported: bool
    correct_inconclusive: bool
    false_confident_conclusion: bool
    status: str
    tool_calls: int
    informative_tool_calls: int
    empty_tool_calls: int
    invalid_tool_calls: int
    duplicate_tool_calls: int
    token_usage: int | None
    duration_seconds: float
    investigation_failures: int
    timeouts: int
    provider_name: str
    model_name: str


@dataclass(frozen=True)
class RootCauseEvaluationSummary:
    cases: list[RootCauseEvaluationCase]
    root_cause_accuracy: float
    evidence_support_rate: float
    inconclusive_accuracy: float
    false_confident_conclusions: int
    false_confident_conclusion_rate: float
    workflow_completion_rate: float
    average_tool_calls: float
    informative_tool_calls: int
    empty_tool_calls: int
    invalid_tool_calls: int
    duplicate_tool_calls: int
    total_tokens: int | None
    average_duration_seconds: float
    investigation_failures: int
    timeouts: int
    provider_name: str
    model_name: str
    benchmark_scope: str = "ambiguous_upstream_root_cause"

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "cases": [asdict(case) for case in self.cases]}


class RootCauseBenchmarkEvaluator:
    """Evaluate causal investigations after agents finish without access to hidden truth."""

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
        case_ids: tuple[str, ...] = ROOT_CAUSE_CASES,
    ) -> RootCauseEvaluationSummary:
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
        cases: list[RootCauseEvaluationCase] = []
        for index, case_id in enumerate(case_ids, start=1):
            scenario = run_root_cause_scenario(
                case_id,
                row_count=row_count,
                seed=seed + index * 101,
                as_of=reference_time,
                repository=self.repository,
                ingested_at=ingestion_start + timedelta(microseconds=index),
            )
            report = self.agent_factory().investigate(scenario.pipeline_run.report.incident_id)
            truth = scenario.ground_truth
            expected = None if truth.root_cause is None else truth.root_cause.value
            predicted = report.conclusion.diagnosis_code
            cited = set(report.conclusion.evidence_references)
            assessment = (
                None
                if predicted is None
                else assess_causal_evidence(predicted, list(cited), report.evidence)
            )
            evidence_supported = (
                predicted is not None
                and report.status == "completed"
                and bool(cited)
                and assessment is not None
                and assessment.supported
            )
            investigated_tools = {item.tool_name for item in report.evidence}
            correct_inconclusive = (
                expected is None
                and predicted is None
                and report.status == "inconclusive"
                and set(truth.required_evidence_tools).issubset(investigated_tools)
                and has_inconclusive_audit_evidence(report.evidence)
            )
            false_confident = (
                predicted is not None
                and report.status == "completed"
                and report.conclusion.confidence.value in {"high", "medium"}
                and not evidence_supported
            )
            cases.append(
                RootCauseEvaluationCase(
                    case_id=case_id,
                    incident_id=report.incident_id,
                    observable_failure=truth.observable_failure.value,
                    expected_root_cause=expected,
                    predicted_root_cause=predicted,
                    correct=(
                        expected is not None
                        and predicted == expected
                        and report.status == "completed"
                    )
                    or correct_inconclusive,
                    evidence_supported=evidence_supported,
                    correct_inconclusive=correct_inconclusive,
                    false_confident_conclusion=false_confident,
                    status=report.status,
                    tool_calls=report.tool_call_count,
                    informative_tool_calls=report.informative_tool_calls,
                    empty_tool_calls=report.empty_tool_calls,
                    invalid_tool_calls=report.invalid_tool_calls,
                    duplicate_tool_calls=report.duplicate_tool_calls,
                    token_usage=report.token_usage,
                    duration_seconds=report.duration_seconds,
                    investigation_failures=report.investigation_failures,
                    timeouts=report.timeouts,
                    provider_name=report.provider_name,
                    model_name=report.model_name,
                )
            )
        return _summarize(cases)


def _summarize(cases: list[RootCauseEvaluationCase]) -> RootCauseEvaluationSummary:
    if not cases:
        raise ValueError("root-cause benchmark requires at least one case")
    resolvable = [case for case in cases if case.expected_root_cause is not None]
    inconclusive = [case for case in cases if case.expected_root_cause is None]
    diagnosed = [case for case in cases if case.predicted_root_cause is not None]
    tokens = [case.token_usage for case in cases if case.token_usage is not None]
    providers = {case.provider_name for case in cases}
    models = {case.model_name for case in cases}
    return RootCauseEvaluationSummary(
        cases=cases,
        root_cause_accuracy=(
            sum(case.correct for case in resolvable) / len(resolvable) if resolvable else 0.0
        ),
        evidence_support_rate=(
            sum(case.evidence_supported for case in diagnosed) / len(diagnosed)
            if diagnosed
            else 0.0
        ),
        inconclusive_accuracy=(
            sum(case.correct_inconclusive for case in inconclusive) / len(inconclusive)
            if inconclusive
            else 0.0
        ),
        false_confident_conclusions=sum(case.false_confident_conclusion for case in cases),
        false_confident_conclusion_rate=(
            sum(case.false_confident_conclusion for case in cases) / len(cases)
        ),
        workflow_completion_rate=(
            sum(case.status in {"completed", "inconclusive"} for case in cases) / len(cases)
        ),
        average_tool_calls=sum(case.tool_calls for case in cases) / len(cases),
        informative_tool_calls=sum(case.informative_tool_calls for case in cases),
        empty_tool_calls=sum(case.empty_tool_calls for case in cases),
        invalid_tool_calls=sum(case.invalid_tool_calls for case in cases),
        duplicate_tool_calls=sum(case.duplicate_tool_calls for case in cases),
        total_tokens=sum(tokens) if tokens else None,
        average_duration_seconds=sum(case.duration_seconds for case in cases) / len(cases),
        investigation_failures=sum(case.investigation_failures for case in cases),
        timeouts=sum(case.timeouts for case in cases),
        provider_name=next(iter(providers)) if len(providers) == 1 else "mixed",
        model_name=next(iter(models)) if len(models) == 1 else "mixed",
    )
