from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from retail_guardian.dashboard_service import DashboardService


def test_dashboard_service_runs_complete_deterministic_workflow(tmp_path, as_of) -> None:
    database = tmp_path / "dashboard.db"
    service = DashboardService(database)
    assert service.overview().pipeline_health == "no data"

    run = service.run_scenario(
        "duplicate_records",
        row_count=100,
        seed=42,
        as_of=as_of,
    )
    investigation = service.investigate(run.report.incident_id, provider="mock")
    repair = service.propose_repair(investigation.investigation_id, provider="mock")
    service.approve_repair(repair.repair_id)
    completed = service.simulate_repair(repair.repair_id)

    snapshot = service.overview()
    assert snapshot.pipeline_health == "incident"
    assert snapshot.execution_count == 1
    assert snapshot.incident_count == 1
    assert snapshot.affected_records == 10
    assert snapshot.recommended_repairs == 1
    assert completed.verification is not None and completed.verification.passed


def test_streamlit_application_smoke_test(tmp_path, as_of) -> None:
    database = tmp_path / "streamlit.db"
    service = DashboardService(database)
    service.run_scenario("duplicate_records", row_count=50, seed=9, as_of=as_of)

    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=20)
    app.run()
    assert not app.exception
    app.sidebar.text_input[0].set_value(str(database)).run()

    assert not app.exception
    assert any(item.value == "Operational overview" for item in app.subheader)
    assert any(metric.label == "Incidents" and metric.value == "1" for metric in app.metric)
