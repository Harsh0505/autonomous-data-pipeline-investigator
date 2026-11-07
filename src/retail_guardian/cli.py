from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from retail_guardian.causal_scenarios import ROOT_CAUSE_CASES, run_root_cause_scenario
from retail_guardian.evaluation import BenchmarkEvaluator
from retail_guardian.investigation import InvestigationAgent
from retail_guardian.providers import (
    HeuristicInvestigationModel,
    OllamaInvestigationModel,
    OpenAIInvestigationModel,
)
from retail_guardian.repair import RepairWorkflow, RepairWorkflowError
from retail_guardian.repair_providers import (
    DeterministicRepairPlanningModel,
    OllamaRepairPlanningModel,
    RepairPlanningError,
)
from retail_guardian.root_cause_evaluation import RootCauseBenchmarkEvaluator
from retail_guardian.scenarios import SCENARIOS, run_scenario
from retail_guardian.storage import SQLiteHistoryRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retail-guardian",
        description="Run reproducible retail data-pipeline scenarios.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Generate and execute a pipeline scenario")
    run_parser.add_argument("--scenario", choices=SCENARIOS, default="healthy")
    run_parser.add_argument("--rows", type=int, default=500)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument(
        "--as-of",
        type=_parse_datetime,
        default=None,
        help="ISO-8601 reference time (default: current UTC time)",
    )
    run_parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    run_parser.add_argument("--database", type=Path, default=None)
    causal_parser = subparsers.add_parser(
        "run-root-cause", help="Generate an opaque upstream root-cause scenario"
    )
    causal_parser.add_argument("--case", choices=ROOT_CAUSE_CASES, default=ROOT_CAUSE_CASES[0])
    causal_parser.add_argument("--rows", type=int, default=500)
    causal_parser.add_argument("--seed", type=int, default=42)
    causal_parser.add_argument("--as-of", type=_parse_datetime, default=None)
    causal_parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    causal_parser.add_argument("--database", type=Path, default=None)
    subparsers.add_parser("list-scenarios", help="List available example scenarios")
    investigate_parser = subparsers.add_parser(
        "investigate", help="Autonomously investigate a persisted incident"
    )
    investigate_parser.add_argument("--incident-id", required=True)
    investigate_parser.add_argument("--database", type=Path, default=Path("artifacts/history.db"))
    investigate_parser.add_argument(
        "--provider", choices=("mock", "ollama", "openai"), default="ollama"
    )
    investigate_parser.add_argument("--model", default=None)
    investigate_parser.add_argument("--endpoint", default=None)
    investigate_parser.add_argument("--timeout", type=float, default=120.0)
    investigate_parser.add_argument("--model-retries", type=int, default=2)
    investigate_parser.add_argument("--max-tool-calls", type=int, default=8)
    investigate_parser.add_argument("--max-invalid-tool-calls", type=int, default=3)
    investigate_parser.add_argument("--max-duplicate-tool-calls", type=int, default=2)
    investigate_parser.add_argument("--max-empty-tool-calls", type=int, default=2)
    investigate_parser.add_argument("--output", type=Path, default=None)
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Evaluate investigations against hidden synthetic ground truth"
    )
    benchmark_parser.add_argument("--database", type=Path, default=Path("artifacts/benchmark.db"))
    benchmark_parser.add_argument(
        "--provider", choices=("mock", "ollama", "openai"), default="ollama"
    )
    benchmark_parser.add_argument("--model", default=None)
    benchmark_parser.add_argument("--endpoint", default=None)
    benchmark_parser.add_argument("--timeout", type=float, default=120.0)
    benchmark_parser.add_argument("--model-retries", type=int, default=2)
    benchmark_parser.add_argument("--rows", type=int, default=500)
    benchmark_parser.add_argument("--seed", type=int, default=42)
    benchmark_parser.add_argument("--max-tool-calls", type=int, default=8)
    benchmark_parser.add_argument("--max-invalid-tool-calls", type=int, default=3)
    benchmark_parser.add_argument("--max-duplicate-tool-calls", type=int, default=2)
    benchmark_parser.add_argument("--max-empty-tool-calls", type=int, default=2)
    benchmark_parser.add_argument("--output", type=Path, default=None)
    root_benchmark_parser = subparsers.add_parser(
        "root-cause-benchmark",
        help="Evaluate ambiguous upstream causal investigations",
    )
    root_benchmark_parser.add_argument(
        "--database", type=Path, default=Path("artifacts/root-cause-benchmark.db")
    )
    root_benchmark_parser.add_argument(
        "--provider", choices=("mock", "ollama", "openai"), default="ollama"
    )
    root_benchmark_parser.add_argument("--model", default=None)
    root_benchmark_parser.add_argument("--endpoint", default=None)
    root_benchmark_parser.add_argument("--timeout", type=float, default=120.0)
    root_benchmark_parser.add_argument("--model-retries", type=int, default=2)
    root_benchmark_parser.add_argument("--rows", type=int, default=500)
    root_benchmark_parser.add_argument("--seed", type=int, default=42)
    root_benchmark_parser.add_argument("--max-tool-calls", type=int, default=8)
    root_benchmark_parser.add_argument("--max-invalid-tool-calls", type=int, default=3)
    root_benchmark_parser.add_argument("--max-duplicate-tool-calls", type=int, default=2)
    root_benchmark_parser.add_argument("--max-empty-tool-calls", type=int, default=2)
    root_benchmark_parser.add_argument("--output", type=Path, default=None)
    trace_parser = subparsers.add_parser(
        "trace", help="Print a persisted investigation decision and tool trace"
    )
    trace_parser.add_argument("--investigation-id", required=True)
    trace_parser.add_argument("--database", type=Path, default=Path("artifacts/history.db"))
    propose_parser = subparsers.add_parser(
        "propose-repair", help="Generate a safe repair proposal for an investigation"
    )
    propose_parser.add_argument("--investigation-id", required=True)
    propose_parser.add_argument("--database", type=Path, default=Path("artifacts/history.db"))
    propose_parser.add_argument("--provider", choices=("mock", "ollama"), default="ollama")
    propose_parser.add_argument("--model", default=None)
    propose_parser.add_argument("--endpoint", default=None)
    propose_parser.add_argument("--timeout", type=float, default=120.0)
    propose_parser.add_argument("--model-retries", type=int, default=2)
    propose_parser.add_argument("--output", type=Path, default=None)
    for command, help_text in (
        ("show-repair", "Display a persisted repair proposal"),
        ("approve-repair", "Approve a repair for isolated simulation"),
        ("simulate-repair", "Run an approved repair in isolation and verify it"),
        ("repair-result", "Display repair transitions and verification results"),
    ):
        repair_parser = subparsers.add_parser(command, help=help_text)
        repair_parser.add_argument("--repair-id", required=True)
        repair_parser.add_argument("--database", type=Path, default=Path("artifacts/history.db"))
        repair_parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list-scenarios":
        print("\n".join(SCENARIOS))
        return 0
    if args.command == "investigate":
        return _run_investigation(args)
    if args.command == "benchmark":
        return _run_benchmark(args)
    if args.command == "root-cause-benchmark":
        return _run_root_cause_benchmark(args)
    if args.command == "trace":
        return _show_trace(args)
    if args.command == "propose-repair":
        return _propose_repair(args)
    if args.command == "show-repair":
        return _show_repair(args)
    if args.command == "approve-repair":
        return _approve_repair(args)
    if args.command == "simulate-repair":
        return _simulate_repair(args)
    if args.command == "repair-result":
        return _show_repair_result(args)
    if args.command == "run-root-cause":
        return _run_root_cause_scenario(args)

    as_of = args.as_of or datetime.now(UTC)
    output_dir: Path = args.output_dir
    database = args.database or output_dir / "history.db"
    repository = SQLiteHistoryRepository(database)
    try:
        run = run_scenario(
            args.scenario,
            row_count=args.rows,
            seed=args.seed,
            as_of=as_of,
            repository=repository,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report_path = output_dir / f"{args.scenario}-incident.json"
    run.report.write(report_path)
    if run.aggregates is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        run.aggregates.to_csv(output_dir / f"{args.scenario}-aggregates.csv", index=False)

    print(run.report.to_json())
    print(f"Report written to {report_path}", file=sys.stderr)
    return 0 if run.report.status == "healthy" else 1


def _run_investigation(args: argparse.Namespace) -> int:
    try:
        repository = SQLiteHistoryRepository(args.database)
        model = _build_model(
            args.provider,
            args.model,
            endpoint=args.endpoint,
            timeout=args.timeout,
            model_retries=args.model_retries,
        )
        report = InvestigationAgent(
            repository,
            model,
            max_tool_calls=args.max_tool_calls,
            max_invalid_tool_calls=args.max_invalid_tool_calls,
            max_duplicate_tool_calls=args.max_duplicate_tool_calls,
            max_empty_tool_calls=args.max_empty_tool_calls,
        ).investigate(args.incident_id)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.output:
        report.write(args.output)
    print(report.to_json())
    return 0 if report.status == "completed" else 1


def _run_benchmark(args: argparse.Namespace) -> int:
    try:
        repository = SQLiteHistoryRepository(args.database)

        def agent_factory() -> InvestigationAgent:
            return InvestigationAgent(
                repository,
                _build_model(
                    args.provider,
                    args.model,
                    endpoint=args.endpoint,
                    timeout=args.timeout,
                    model_retries=args.model_retries,
                ),
                max_tool_calls=args.max_tool_calls,
                max_invalid_tool_calls=args.max_invalid_tool_calls,
                max_duplicate_tool_calls=args.max_duplicate_tool_calls,
                max_empty_tool_calls=args.max_empty_tool_calls,
            )

        summary = BenchmarkEvaluator(repository, agent_factory).run(
            row_count=args.rows,
            seed=args.seed,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output = json.dumps(summary.to_dict(), indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 1 if summary.investigation_failures or summary.timeouts else 0


def _run_root_cause_scenario(args: argparse.Namespace) -> int:
    as_of = args.as_of or datetime.now(UTC)
    output_dir: Path = args.output_dir
    database = args.database or output_dir / "history.db"
    repository = SQLiteHistoryRepository(database)
    try:
        scenario = run_root_cause_scenario(
            args.case,
            row_count=args.rows,
            seed=args.seed,
            as_of=as_of,
            repository=repository,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report_path = output_dir / f"{args.case}-incident.json"
    scenario.pipeline_run.report.write(report_path)
    print(scenario.pipeline_run.report.to_json())
    print(f"Report written to {report_path}", file=sys.stderr)
    return 1


def _run_root_cause_benchmark(args: argparse.Namespace) -> int:
    try:
        repository = SQLiteHistoryRepository(args.database)

        def agent_factory() -> InvestigationAgent:
            return InvestigationAgent(
                repository,
                _build_model(
                    args.provider,
                    args.model,
                    endpoint=args.endpoint,
                    timeout=args.timeout,
                    model_retries=args.model_retries,
                ),
                max_tool_calls=args.max_tool_calls,
                max_invalid_tool_calls=args.max_invalid_tool_calls,
                max_duplicate_tool_calls=args.max_duplicate_tool_calls,
                max_empty_tool_calls=args.max_empty_tool_calls,
            )

        summary = RootCauseBenchmarkEvaluator(repository, agent_factory).run(
            row_count=args.rows,
            seed=args.seed,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output = json.dumps(summary.to_dict(), indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 1 if summary.investigation_failures or summary.timeouts else 0


def _show_trace(args: argparse.Namespace) -> int:
    repository = SQLiteHistoryRepository(args.database)
    trace = repository.get_investigation_trace(args.investigation_id)
    if trace is None:
        print(f"error: investigation not found: {args.investigation_id}", file=sys.stderr)
        return 2
    print(json.dumps(trace, indent=2, sort_keys=True))
    return 0


def _propose_repair(args: argparse.Namespace) -> int:
    try:
        repository = SQLiteHistoryRepository(args.database)
        planner = (
            OllamaRepairPlanningModel(
                args.model,
                endpoint=args.endpoint,
                timeout_seconds=args.timeout,
                max_retries=args.model_retries,
            )
            if args.provider == "ollama"
            else DeterministicRepairPlanningModel()
        )
        record = RepairWorkflow(repository, planner).propose(args.investigation_id)
    except (RepairPlanningError, RepairWorkflowError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output = record.to_json()
    _write_optional_output(args.output, output)
    print(output)
    return 0


def _show_repair(args: argparse.Namespace) -> int:
    repository = SQLiteHistoryRepository(args.database)
    record = repository.get_repair(args.repair_id)
    if record is None:
        print(f"error: repair not found: {args.repair_id}", file=sys.stderr)
        return 2
    output = record.to_json()
    _write_optional_output(args.output, output)
    print(output)
    return 0


def _approve_repair(args: argparse.Namespace) -> int:
    try:
        repository = SQLiteHistoryRepository(args.database)
        workflow = RepairWorkflow(repository, DeterministicRepairPlanningModel())
        record = workflow.approve_for_simulation(args.repair_id)
    except (RepairWorkflowError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output = record.to_json()
    _write_optional_output(args.output, output)
    print(output)
    return 0


def _simulate_repair(args: argparse.Namespace) -> int:
    try:
        repository = SQLiteHistoryRepository(args.database)
        record = RepairWorkflow(repository, DeterministicRepairPlanningModel()).simulate(
            args.repair_id
        )
    except (RepairWorkflowError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output = record.to_json()
    _write_optional_output(args.output, output)
    print(output)
    return 0 if record.verification and record.verification.passed else 1


def _show_repair_result(args: argparse.Namespace) -> int:
    repository = SQLiteHistoryRepository(args.database)
    trace = repository.get_repair_trace(args.repair_id)
    if trace is None:
        print(f"error: repair not found: {args.repair_id}", file=sys.stderr)
        return 2
    output = json.dumps(trace, indent=2, sort_keys=True)
    _write_optional_output(args.output, output)
    print(output)
    return 0


def _write_optional_output(path: Path | None, content: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def _build_model(
    provider: str,
    model_name: str | None,
    *,
    endpoint: str | None = None,
    timeout: float = 120.0,
    model_retries: int = 2,
):
    if provider == "openai":
        return OpenAIInvestigationModel(model_name)
    if provider == "ollama":
        return OllamaInvestigationModel(
            model_name,
            endpoint=endpoint,
            timeout_seconds=timeout,
            max_retries=model_retries,
        )
    return HeuristicInvestigationModel()


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("--as-of must include a timezone")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
