from __future__ import annotations

import json

from retail_guardian.cli import main


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
