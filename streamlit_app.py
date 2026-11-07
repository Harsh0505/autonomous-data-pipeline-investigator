from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from retail_guardian.dashboard_service import DashboardService
from retail_guardian.repair_models import RepairRecord, RepairStatus
from retail_guardian.scenarios import SCENARIOS

st.set_page_config(
    page_title="Retail Pipeline Guardian",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    _apply_style()
    _render_header()
    database_path, page = _render_sidebar()
    try:
        service = DashboardService(database_path)
    except (OSError, ValueError) as exc:
        st.error(f"The history database could not be opened: {exc}")
        st.info("Choose a writable database path in the sidebar.")
        return

    if page == "Overview":
        _render_overview(service)
    elif page == "Run scenario":
        _render_scenario(service)
    elif page == "Investigate and repair":
        _render_workflow(service)
    else:
        _render_history(service)


def _render_header() -> None:
    st.title("Retail Pipeline Guardian")
    st.caption(
        "Evidence-driven incident investigation, bounded repair planning, and deterministic "
        "verification for synthetic retail data pipelines."
    )


def _render_sidebar() -> tuple[Path, str]:
    with st.sidebar:
        st.subheader("Workspace")
        database = st.text_input("SQLite database", value="artifacts/dashboard.db")
        page = st.radio(
            "View",
            ("Overview", "Run scenario", "Investigate and repair", "History"),
        )
        st.divider()
        st.caption("All repairs are simulation-only. No production mutation is available.")
    return Path(database), page


def _render_overview(service: DashboardService) -> None:
    snapshot = service.overview()
    st.subheader("Operational overview")
    columns = st.columns(6)
    columns[0].metric("Pipeline health", snapshot.pipeline_health.title())
    columns[1].metric("Recent executions", snapshot.execution_count)
    columns[2].metric("Incidents", snapshot.incident_count)
    columns[3].metric("Affected records", snapshot.affected_records)
    columns[4].metric("Repairs recommended", snapshot.recommended_repairs)
    columns[5].metric("Repairs rejected", snapshot.rejected_repairs)

    left, right = st.columns((1.7, 1))
    with left:
        st.markdown("#### Recent executions")
        if snapshot.recent_executions:
            rows = [
                {
                    "Execution": item["execution_id"],
                    "Status": item["status"],
                    "Stage": item["pipeline_stage"],
                    "Rows": item["row_count"],
                    "Ingested": item["ingestion_timestamp"],
                }
                for item in snapshot.recent_executions
            ]
            st.dataframe(rows, hide_index=True, use_container_width=True)
        else:
            st.info("No executions have been recorded yet. Run a scenario to begin.")
    with right:
        st.markdown("#### Incident severity")
        severity = _severity_counts(snapshot.incidents)
        if severity:
            st.bar_chart(pd.DataFrame({"Incidents": severity}))
        else:
            st.info("No incident severity data is available.")

    st.markdown("#### Investigation and repair state")
    state_rows = _workflow_state_rows(snapshot.investigations, snapshot.repairs)
    if state_rows:
        st.dataframe(state_rows, hide_index=True, use_container_width=True)
    else:
        st.info("No investigations have been started.")


def _render_scenario(service: DashboardService) -> None:
    st.subheader("Reproducible incident demonstration")
    st.write(
        "Generate a deterministic supplier feed, execute the existing pipeline, and persist the "
        "actual validation results for investigation."
    )
    with st.form("scenario-form"):
        first, second, third = st.columns(3)
        scenario = first.selectbox(
            "Scenario",
            SCENARIOS,
            format_func=lambda value: value.replace("_", " ").title(),
        )
        rows = second.number_input("Rows", min_value=25, max_value=5_000, value=500, step=25)
        seed = third.number_input("Seed", min_value=0, max_value=1_000_000, value=42)
        submitted = st.form_submit_button("Run pipeline", type="primary")
    if submitted:
        try:
            with st.spinner("Generating data and running validation..."):
                run = service.run_scenario(
                    scenario,
                    row_count=int(rows),
                    seed=int(seed),
                    as_of=datetime.now(UTC),
                )
            st.session_state["selected_incident_id"] = run.report.incident_id
            st.session_state["latest_pipeline_report"] = run.report.to_dict()
        except (OSError, TypeError, ValueError) as exc:
            st.error(_friendly_error(exc))

    report = st.session_state.get("latest_pipeline_report")
    if report:
        _render_pipeline_report(report, service)


def _render_pipeline_report(report: dict[str, Any], service: DashboardService) -> None:
    status = report["status"]
    if status == "healthy":
        st.success("The pipeline completed without validation failures.")
    else:
        st.warning("The validation gate created an incident for investigation.")
    metrics = report.get("metrics", {})
    columns = st.columns(4)
    columns[0].metric("Status", status.title())
    columns[1].metric("Rows", report.get("dataset", {}).get("row_count", 0))
    columns[2].metric("Checks failed", metrics.get("checks_failed", 0))
    columns[3].metric("Blocking failures", metrics.get("blocking_violations", 0))

    violations = report.get("violations", [])
    if violations:
        st.markdown("#### Validation findings")
        st.dataframe(
            [
                {
                    "Check": item["check_id"],
                    "Kind": item["kind"],
                    "Severity": item["severity"],
                    "Affected rows": (item.get("finding") or {}).get("affected_rows", 0),
                    "Finding": (item.get("finding") or {}).get("message", ""),
                }
                for item in violations
            ],
            hide_index=True,
            use_container_width=True,
        )
        execution = service.repository.get_execution(report["incident_id"])
        if execution:
            samples = service.repository.get_problem_samples(execution["execution_id"], limit=20)
            if samples:
                st.markdown("#### Affected data sample")
                st.dataframe(
                    [dict(Check=item["check_id"], **item["record"]) for item in samples],
                    hide_index=True,
                    use_container_width=True,
                )


def _render_workflow(service: DashboardService) -> None:
    st.subheader("Investigation and repair workflow")
    incidents = service.repository.list_incidents(limit=100)
    if not incidents:
        st.info("No incidents are available. Run a failing scenario first.")
        return
    incident_by_id = {item["incident_id"]: item for item in incidents}
    default_id = st.session_state.get("selected_incident_id")
    options = list(incident_by_id)
    default_index = options.index(default_id) if default_id in options else 0
    incident_id = st.selectbox(
        "Incident",
        options,
        index=default_index,
        format_func=lambda value: _incident_label(incident_by_id[value]),
    )
    st.session_state["selected_incident_id"] = incident_id
    _render_incident_summary(incident_by_id[incident_id]["report"])

    st.markdown("#### Investigation provider")
    provider_label = st.radio(
        "Provider",
        (
            "Deterministic demo provider",
            "Local Ollama — autonomous qwen3:4b reasoning",
        ),
        horizontal=True,
    )
    provider = "ollama" if provider_label.startswith("Local") else "mock"
    endpoint = "http://localhost:11434"
    if provider == "ollama":
        endpoint = st.text_input("Ollama endpoint", value=endpoint)
        st.caption("If Ollama is unavailable, select the deterministic provider and continue.")
    if st.button("Launch investigation", type="primary"):
        try:
            with st.spinner("The investigation agent is collecting and validating evidence..."):
                report = service.investigate(
                    incident_id,
                    provider=provider,
                    endpoint=endpoint,
                )
            st.session_state["selected_investigation_id"] = report.investigation_id
            st.success(f"Investigation finished with status: {report.status}")
        except Exception as exc:  # Streamlit must remain usable after provider failures.
            st.error(_friendly_error(exc))

    investigations = service.repository.get_investigations(incident_id)
    if not investigations:
        return
    investigation_ids = [item["investigation_id"] for item in investigations]
    selected = st.session_state.get("selected_investigation_id")
    index = (
        investigation_ids.index(selected)
        if selected in investigation_ids
        else len(investigation_ids) - 1
    )
    investigation_id = st.selectbox(
        "Investigation result",
        investigation_ids,
        index=index,
    )
    st.session_state["selected_investigation_id"] = investigation_id
    trace = service.repository.get_investigation_trace(investigation_id)
    if trace:
        _render_investigation(trace)
        _render_repair_workflow(service, trace)


def _render_investigation(trace: dict[str, Any]) -> None:
    report = trace.get("report") or {}
    conclusion = report.get("conclusion") or {}
    st.markdown("#### Investigation result")
    columns = st.columns(5)
    columns[0].metric("Status", str(report.get("status", trace.get("status", "unknown"))).title())
    columns[1].metric("Diagnosis", conclusion.get("diagnosis_code") or "Inconclusive")
    columns[2].metric("Confidence", conclusion.get("confidence", "unknown"))
    columns[3].metric("Tool calls", report.get("tool_call_count", len(trace["tool_calls"])))
    token_usage = report.get("token_usage")
    columns[4].metric("Tokens", token_usage if token_usage is not None else "Unavailable")
    st.write(conclusion.get("final_diagnosis", "No conclusion was recorded."))
    if conclusion.get("outstanding_questions"):
        st.markdown("**Outstanding questions**")
        for question in conclusion["outstanding_questions"]:
            st.write(f"- {question}")

    tool_tab, evidence_tab, hypothesis_tab = st.tabs(("Tool calls", "Evidence", "Hypotheses"))
    with tool_tab:
        if trace["tool_calls"]:
            st.dataframe(trace["tool_calls"], hide_index=True, use_container_width=True)
        else:
            st.info("No tools were executed.")
    with evidence_tab:
        for evidence in trace["evidence"]:
            with st.expander(
                f"{evidence['evidence_id']} · {evidence['tool_name']} · {evidence['result_kind']}"
            ):
                st.write(evidence["summary"])
                st.json(evidence["data"], expanded=False)
    with hypothesis_tab:
        hypotheses = report.get("hypotheses", [])
        if hypotheses:
            st.dataframe(hypotheses, hide_index=True, use_container_width=True)
        else:
            st.info("No hypotheses were persisted.")
    st.caption(
        f"Duration: {report.get('duration_seconds', 0):.2f}s · "
        f"Validation failures: {report.get('model_validation_failures', 0)}"
    )


def _render_repair_workflow(service: DashboardService, trace: dict[str, Any]) -> None:
    report = trace.get("report") or {}
    if report.get("status") != "completed" or not (report.get("conclusion") or {}).get(
        "diagnosis_code"
    ):
        st.info("A completed, evidence-backed diagnosis is required before repair planning.")
        return
    st.markdown("#### Repair planning")
    planner_label = st.selectbox(
        "Repair planner",
        ("Deterministic demo provider", "Local Ollama — qwen3:4b"),
    )
    planner_provider = "ollama" if planner_label.startswith("Local") else "mock"
    if st.button("Request repair proposal"):
        try:
            with st.spinner("Generating and validating a scoped repair proposal..."):
                repair = service.propose_repair(
                    trace["investigation_id"], provider=planner_provider
                )
            st.session_state["selected_repair_id"] = repair.repair_id
            st.success("A repair proposal was persisted. Review it before approval.")
        except Exception as exc:  # Provider and unsupported-repair failures are user-facing.
            st.error(_friendly_error(exc))

    repairs = service.repository.list_repairs(investigation_id=trace["investigation_id"], limit=50)
    if not repairs:
        return
    repair_by_id = {item.repair_id: item for item in repairs}
    selected = st.session_state.get("selected_repair_id")
    repair_id = st.selectbox(
        "Repair proposal",
        list(repair_by_id),
        index=list(repair_by_id).index(selected) if selected in repair_by_id else 0,
    )
    st.session_state["selected_repair_id"] = repair_id
    repair = repair_by_id[repair_id]
    _render_repair(repair)

    first, second = st.columns(2)
    if repair.status == RepairStatus.PROPOSED:
        if first.button("Approve for isolated simulation", type="primary"):
            try:
                service.approve_repair(repair_id)
                st.rerun()
            except Exception as exc:
                st.error(_friendly_error(exc))
    elif repair.status == RepairStatus.APPROVED_FOR_SIMULATION and second.button(
        "Simulate and verify", type="primary"
    ):
        try:
            with st.spinner("Executing the isolated repair and rerunning validation..."):
                service.simulate_repair(repair_id)
            st.rerun()
        except Exception as exc:
            st.error(_friendly_error(exc))


def _render_repair(repair: RepairRecord) -> None:
    proposal = repair.proposal
    columns = st.columns(3)
    columns[0].metric("Repair status", repair.status.value.replace("_", " ").title())
    columns[1].metric("Operation", proposal.repair_type.value)
    columns[2].metric("Target", proposal.target_component.replace("_", " "))
    st.write(proposal.explanation)
    details, safety = st.columns(2)
    with details:
        st.markdown("**Parameters**")
        st.json(proposal.parameters.model_dump(mode="json"), expanded=False)
        st.markdown("**Expected effect**")
        st.write(proposal.expected_effect)
        st.markdown("**Supporting evidence**")
        st.write(", ".join(proposal.supporting_evidence_ids))
    with safety:
        st.markdown("**Risks and limitations**")
        for risk in proposal.risks_and_limitations:
            st.write(f"- {risk}")
        st.markdown("**Verification requirements**")
        st.json(proposal.verification_requirements.model_dump(mode="json"), expanded=False)
        st.markdown("**Rollback**")
        st.write(proposal.rollback_approach)
    if repair.verification:
        verification = repair.verification
        if verification.passed:
            st.success("Deterministic verification passed. The simulated repair is recommended.")
        else:
            st.error("Deterministic verification failed. The simulated repair is rejected.")
        columns = st.columns(4)
        columns[0].metric("Original rows", verification.original_row_count)
        columns[1].metric("Repaired rows", verification.repaired_row_count)
        columns[2].metric(
            "Changed records",
            repair.simulation.change_summary.rows_changed if repair.simulation else 0,
        )
        columns[3].metric("New failures", len(verification.new_failures))
        st.dataframe(
            [
                {"Measure": "Resolved checks", "Result": ", ".join(verification.resolved_checks)},
                {"Measure": "Remaining checks", "Result": ", ".join(verification.remaining_checks)},
                {"Measure": "New failures", "Result": ", ".join(verification.new_failures)},
                {
                    "Measure": "Unaffected records preserved",
                    "Result": verification.unaffected_records_preserved,
                },
                {
                    "Measure": "Original snapshot preserved",
                    "Result": verification.original_snapshot_preserved,
                },
            ],
            hide_index=True,
            use_container_width=True,
        )


def _render_history(service: DashboardService) -> None:
    st.subheader("Investigation history")
    sessions = service.repository.list_investigation_sessions(limit=100)
    if not sessions:
        st.info("No investigation history is available in this database.")
        return
    session_by_id = {item["investigation_id"]: item for item in sessions}
    investigation_id = st.selectbox("Investigation", list(session_by_id))
    trace = service.repository.get_investigation_trace(investigation_id)
    if trace is None:
        st.warning("The selected investigation trace is unavailable.")
        return
    incident = service.repository.get_incident(trace["incident_id"])
    if incident:
        _render_incident_summary(incident)
    _render_investigation(trace)
    with st.expander("Agent decisions"):
        st.json(trace["decisions"], expanded=False)
    with st.expander("Rejected model decisions"):
        st.json(trace["rejected_model_decisions"], expanded=False)
    repairs = service.repository.list_repairs(investigation_id=investigation_id, limit=100)
    if repairs:
        st.markdown("#### Repair history")
        for repair in repairs:
            with st.expander(f"{repair.repair_id} · {repair.status.value}"):
                _render_repair(repair)


def _render_incident_summary(report: dict[str, Any]) -> None:
    violations = report.get("violations", [])
    st.markdown("#### Incident")
    columns = st.columns(4)
    columns[0].metric("Incident", report.get("incident_id", "unknown"))
    columns[1].metric("Stage", report.get("pipeline_stage", "unknown"))
    columns[2].metric("Rows", report.get("dataset", {}).get("row_count", 0))
    columns[3].metric("Failed checks", len(violations))
    if violations:
        st.dataframe(
            [
                {
                    "Check": item["check_id"],
                    "Severity": item["severity"],
                    "Affected": (item.get("finding") or {}).get("affected_rows", 0),
                    "Finding": (item.get("finding") or {}).get("message", ""),
                }
                for item in violations
            ],
            hide_index=True,
            use_container_width=True,
        )


def _incident_label(item: dict[str, Any]) -> str:
    report = item["report"]
    checks = ", ".join(value["check_id"] for value in report.get("violations", []))
    return f"{item['incident_id']} · {checks or 'no failed checks'}"


def _severity_counts(incidents: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for incident in incidents:
        severities = [
            item.get("severity", "unknown") for item in incident["report"].get("violations", [])
        ]
        severity = _highest_severity(severities)
        counts[severity] = counts.get(severity, 0) + 1
    return counts


def _highest_severity(values: list[str]) -> str:
    rank = {"unknown": 0, "warning": 1, "error": 2, "critical": 3}
    return max(values, key=lambda value: rank.get(value, 0), default="unknown")


def _workflow_state_rows(
    investigations: list[dict[str, Any]], repairs: list[RepairRecord]
) -> list[dict[str, Any]]:
    repair_by_investigation: dict[str, RepairRecord] = {}
    for repair in repairs:
        repair_by_investigation.setdefault(repair.investigation_id, repair)
    return [
        {
            "Investigation": item["investigation_id"],
            "Incident": item["incident_id"],
            "Investigation status": item["status"],
            "Diagnosis": ((item.get("report") or {}).get("conclusion") or {}).get("diagnosis_code"),
            "Repair status": (
                repair_by_investigation[item["investigation_id"]].status.value
                if item["investigation_id"] in repair_by_investigation
                else "not proposed"
            ),
        }
        for item in investigations
    ]


def _friendly_error(error: Exception) -> str:
    message = str(error)
    lowered = message.lower()
    if "ollama" in lowered and ("unavailable" in lowered or "connect" in lowered):
        return (
            "Ollama is unavailable. Start the local Ollama service or switch to the deterministic "
            "demo provider."
        )
    if "timeout" in lowered or "timed out" in lowered:
        return "The model request timed out. Retry or use the deterministic demo provider."
    if "no safe repair capability" in lowered:
        return "This diagnosis does not yet have a supported safe repair operation."
    if "not found" in lowered:
        return "The selected record is no longer available. Refresh the view and select another."
    return message[:500] or "The operation could not be completed."


def _apply_style() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f7f8fa; color: #172033; }
        [data-testid="stSidebar"] { background: #eef1f5; border-right: 1px solid #d8dee8; }
        .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 4rem; }
        h1, h2, h3, h4 { color: #172033; letter-spacing: -0.02em; }
        [data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #dfe4ec; border-radius: 8px; padding: 0.9rem;
        }
        [data-testid="stDataFrame"] { border: 1px solid #dfe4ec; border-radius: 8px; }
        .stButton > button[kind="primary"] { background: #245b85; border-color: #245b85; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
