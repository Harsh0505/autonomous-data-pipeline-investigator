from __future__ import annotations

import json

from retail_guardian.cli import main
from retail_guardian.providers import OllamaInvestigationModel


def test_cli_writes_healthy_outputs(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "run",
            "--scenario",
            "healthy",
            "--rows",
            "50",
            "--seed",
            "3",
            "--as-of",
            "2026-10-20T12:00:00Z",
            "--output-dir",
            str(tmp_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "healthy"
    assert (tmp_path / "healthy-incident.json").is_file()
    assert (tmp_path / "healthy-aggregates.csv").is_file()


def test_cli_uses_nonzero_exit_for_incident(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "run",
            "--scenario",
            "duplicate_records",
            "--as-of",
            "2026-10-20T12:00:00+00:00",
            "--output-dir",
            str(tmp_path),
        ]
    )
    capsys.readouterr()

    assert exit_code == 1
    assert (tmp_path / "duplicate_records-incident.json").is_file()


def test_cli_builds_configured_ollama_provider() -> None:
    from retail_guardian.cli import _build_model

    provider = _build_model(
        "ollama",
        "qwen3:4b",
        endpoint="http://runtime.test:11434",
        timeout=45,
        model_retries=1,
    )

    assert isinstance(provider, OllamaInvestigationModel)
    assert provider.endpoint == "http://runtime.test:11434"
    assert provider.timeout_seconds == 45
    assert provider.max_retries == 1


def test_cli_mock_benchmark_writes_distinguished_results(tmp_path, capsys) -> None:
    output = tmp_path / "evaluation.json"

    exit_code = main(
        [
            "benchmark",
            "--provider",
            "mock",
            "--rows",
            "50",
            "--database",
            str(tmp_path / "benchmark.db"),
            "--output",
            str(output),
        ]
    )
    capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["evaluation_type"] == "deterministic_mock"
    assert payload["provider_name"] == "mock"
    assert len(payload["cases"]) == 6


def test_cli_generates_opaque_root_cause_case(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "run-root-cause",
            "--case",
            "case-003",
            "--rows",
            "50",
            "--seed",
            "9",
            "--as-of",
            "2026-10-20T12:00:00Z",
            "--output-dir",
            str(tmp_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["status"] == "incident"
    assert "incorrect_join_cardinality" not in json.dumps(payload)
    assert (tmp_path / "case-003-incident.json").is_file()


def test_cli_root_cause_benchmark_is_separate_from_regression_suite(tmp_path, capsys) -> None:
    output = tmp_path / "root-cause-evaluation.json"

    exit_code = main(
        [
            "root-cause-benchmark",
            "--provider",
            "mock",
            "--rows",
            "50",
            "--database",
            str(tmp_path / "root-cause.db"),
            "--output",
            str(output),
        ]
    )
    capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["benchmark_scope"] == "ambiguous_upstream_root_cause"
    assert len(payload["cases"]) == 10
    assert payload["provider_name"] == "mock"
