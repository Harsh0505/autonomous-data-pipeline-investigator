from __future__ import annotations

import sqlite3
from datetime import timedelta

from retail_guardian.investigation_tools import InvestigationToolbox
from retail_guardian.scenarios import run_scenario
from retail_guardian.storage import SQLiteHistoryRepository


def test_repository_persists_profiles_not_full_datasets(tmp_path, as_of) -> None:
    database = tmp_path / "history.db"
    repository = SQLiteHistoryRepository(database)
    ingested_at = as_of + timedelta(minutes=10)
    run = run_scenario(
        "missing_values",
        row_count=100,
        seed=7,
        as_of=as_of,
        repository=repository,
        ingested_at=ingested_at,
    )

    execution = repository.get_execution(run.report.incident_id)
    assert execution is not None
    assert execution["ingestion_timestamp"] == ingested_at.isoformat()
    assert execution["event_time_max"] != execution["ingestion_timestamp"]
    assert repository.get_schema(run.execution_id)
    assert repository.get_profile(run.execution_id)["profile"]["row_count"] == 100
    assert any(not item["passed"] for item in repository.get_validation_results(run.execution_id))
    assert repository.get_problem_samples(run.execution_id, "required_values")

    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert "raw_dataset" not in tables
    assert "pipeline_execution" in tables
    assert "investigation_session" in tables


def test_schema_tool_compares_with_healthy_history(tmp_path, as_of) -> None:
    repository = SQLiteHistoryRepository(tmp_path / "history.db")
    run_scenario(
        "healthy",
        as_of=as_of,
        repository=repository,
        ingested_at=as_of,
    )
    incident = run_scenario(
        "schema_change",
        as_of=as_of,
        repository=repository,
        ingested_at=as_of + timedelta(seconds=1),
    )

    evidence = InvestigationToolbox(repository, incident.report.incident_id).execute(
        "compare_schemas"
    )

    assert evidence.success
    assert evidence.data["sufficient_history"]
    assert evidence.data["added_columns"] == ["units_sold"]
    assert evidence.data["removed_columns"] == ["quantity"]


def test_comparison_tool_reports_missing_history(tmp_path, as_of) -> None:
    repository = SQLiteHistoryRepository(tmp_path / "history.db")
    incident = run_scenario(
        "schema_change",
        as_of=as_of,
        repository=repository,
        ingested_at=as_of,
    )

    evidence = InvestigationToolbox(repository, incident.report.incident_id).execute(
        "compare_schemas"
    )

    assert evidence.success
    assert not evidence.data["sufficient_history"]
    assert evidence.data["reason"] == "no_prior_execution"


def test_toolbox_returns_structured_failure_for_unknown_tool(tmp_path, as_of) -> None:
    repository = SQLiteHistoryRepository(tmp_path / "history.db")
    incident = run_scenario(
        "duplicate_records", as_of=as_of, repository=repository, ingested_at=as_of
    )

    evidence = InvestigationToolbox(repository, incident.report.incident_id).execute(
        "arbitrary_sql", {"query": "DROP TABLE incident"}
    )

    assert not evidence.success
    assert evidence.data == {}
    assert "unknown investigation tool" in evidence.error


def test_all_scoped_tools_return_structured_evidence(tmp_path, as_of) -> None:
    repository = SQLiteHistoryRepository(tmp_path / "history.db")
    run_scenario("healthy", as_of=as_of, repository=repository, ingested_at=as_of)
    incident = run_scenario(
        "currency_mismatch",
        as_of=as_of,
        repository=repository,
        ingested_at=as_of + timedelta(seconds=1),
    )
    toolbox = InvestigationToolbox(repository, incident.report.incident_id)
    requests = {
        "inspect_dataset": {},
        "get_validation_failures": {},
        "compare_schemas": {},
        "compare_distributions": {},
        "sample_problematic_records": {"check_id": "supplier_currency", "limit": 3},
        "get_pipeline_history": {"limit": 3},
        "identify_affected_dimensions": {"check_id": "supplier_currency"},
        "get_data_contract": {},
    }

    evidence = [toolbox.execute(name, arguments) for name, arguments in requests.items()]

    assert len(evidence) == 8
    assert all(item.success for item in evidence)
    assert all(item.data for item in evidence)
