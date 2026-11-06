from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from retail_guardian.investigation_models import (
    ConfidenceLevel,
    Evidence,
    InvestigationConclusion,
    InvestigationContext,
    InvestigationReport,
    ModelDecision,
    RootCauseHypothesis,
)
from retail_guardian.investigation_tools import InvestigationToolbox
from retail_guardian.providers import InvestigationModel
from retail_guardian.storage import HistoryRepository


class _InvestigationState(TypedDict):
    investigation_id: str
    incident: dict[str, Any]
    started_at: datetime
    hypotheses: list[RootCauseHypothesis]
    evidence: list[Evidence]
    tools_used: list[str]
    tool_call_count: int
    decision_count: int
    invalid_tool_calls: int
    tool_request_signatures: list[str]
    persisted_rejection_count: int
    token_usage: int
    fatal_model_error: bool
    pending_decision: ModelDecision | None
    report: InvestigationReport | None


class InvestigationAgent:
    """Bounded LangGraph workflow that alternates model decisions and scoped tools."""

    def __init__(
        self,
        repository: HistoryRepository,
        model: InvestigationModel,
        *,
        max_tool_calls: int = 8,
    ) -> None:
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")
        self.repository = repository
        self.model = model
        self.max_tool_calls = max_tool_calls
        self._toolbox: InvestigationToolbox | None = None
        self.graph = self._build_graph()

    def investigate(self, incident_id: str) -> InvestigationReport:
        incident = self.repository.get_incident(incident_id)
        if incident is None:
            raise ValueError(f"incident not found: {incident_id}")
        self._toolbox = InvestigationToolbox(self.repository, incident_id)
        started_at = datetime.now(UTC)
        investigation_id = f"INV-{uuid4().hex[:12].upper()}"
        self.repository.start_investigation(investigation_id, incident_id, started_at)
        initial_state: _InvestigationState = {
            "investigation_id": investigation_id,
            "incident": incident,
            "started_at": started_at,
            "hypotheses": [],
            "evidence": [],
            "tools_used": [],
            "tool_call_count": 0,
            "decision_count": 0,
            "invalid_tool_calls": 0,
            "tool_request_signatures": [],
            "persisted_rejection_count": 0,
            "token_usage": 0,
            "fatal_model_error": False,
            "pending_decision": None,
            "report": None,
        }
        final_state = self.graph.invoke(
            initial_state,
            config={"recursion_limit": self.max_tool_calls * 3 + 6},
        )
        report = final_state.get("report")
        if report is None:
            raise RuntimeError("investigation graph completed without a report")
        return report

    def _build_graph(self):
        builder = StateGraph(_InvestigationState)
        builder.add_node("reason", self._reason)
        builder.add_node("call_tool", self._call_tool)
        builder.add_node("finalize", self._finalize)
        builder.add_edge(START, "reason")
        builder.add_conditional_edges(
            "reason",
            self._route_decision,
            {"tool": "call_tool", "finalize": "finalize"},
        )
        builder.add_edge("call_tool", "reason")
        builder.add_edge("finalize", END)
        return builder.compile()

    def _reason(self, state: _InvestigationState) -> dict[str, Any]:
        toolbox = self._require_toolbox()
        remaining = self.max_tool_calls - state["tool_call_count"]
        context = InvestigationContext(
            incident=state["incident"],
            hypotheses=state["hypotheses"],
            evidence=state["evidence"],
            available_tools=toolbox.definitions,
            remaining_tool_calls=remaining,
        )
        try:
            decision = self.model.decide(context)
            fatal_model_error = False
        except Exception as exc:  # Provider errors become an explicit inconclusive result.
            fatal_model_error = True
            decision = ModelDecision(
                action="conclude",
                hypotheses=state["hypotheses"],
                conclusion=_inconclusive_conclusion(
                    state["evidence"], f"The investigation model failed: {exc}"
                ),
            )
        if decision.action == "tool" and (decision.tool_request is None or remaining <= 0):
            decision = ModelDecision(
                action="conclude",
                hypotheses=decision.hypotheses,
                conclusion=_inconclusive_conclusion(
                    state["evidence"], "The investigation reached its tool-call budget"
                ),
                token_usage=decision.token_usage,
            )
        telemetry = getattr(self.model, "telemetry", None)
        rejections = getattr(telemetry, "rejected_decisions", [])
        for sequence, rejection in enumerate(
            rejections[state["persisted_rejection_count"] :],
            start=state["persisted_rejection_count"] + 1,
        ):
            self.repository.record_model_rejection(
                state["investigation_id"], sequence, rejection, datetime.now(UTC)
            )
        decision_sequence = state["decision_count"] + 1
        self.repository.record_model_decision(
            state["investigation_id"], decision_sequence, decision, datetime.now(UTC)
        )
        return {
            "hypotheses": _sanitize_hypotheses(decision.hypotheses, state["evidence"]),
            "pending_decision": decision,
            "token_usage": state["token_usage"] + (decision.token_usage or 0),
            "decision_count": decision_sequence,
            "fatal_model_error": state["fatal_model_error"] or fatal_model_error,
            "persisted_rejection_count": len(rejections),
        }

    def _route_decision(self, state: _InvestigationState) -> Literal["tool", "finalize"]:
        decision = state["pending_decision"]
        return "tool" if decision is not None and decision.action == "tool" else "finalize"

    def _call_tool(self, state: _InvestigationState) -> dict[str, Any]:
        decision = state["pending_decision"]
        if decision is None or decision.tool_request is None:
            raise RuntimeError("tool node received no tool request")
        request = decision.tool_request
        signature = json.dumps(
            {"name": request.name, "arguments": request.arguments}, sort_keys=True
        )
        validation_error = self._require_toolbox().validate_request(request.name, request.arguments)
        is_duplicate = signature in state["tool_request_signatures"]
        started_at = datetime.now(UTC)
        if is_duplicate:
            evidence = Evidence(
                evidence_id=f"EV-{uuid4().hex[:12].upper()}",
                tool_name=request.name,
                summary="Duplicate investigation tool request was rejected",
                data={},
                success=False,
                error="duplicate tool request",
            )
        else:
            evidence = self._require_toolbox().execute(request.name, request.arguments)
        completed_at = datetime.now(UTC)
        sequence = state["tool_call_count"] + 1
        self.repository.record_tool_call(
            state["investigation_id"],
            sequence,
            request.name,
            request.arguments,
            request.rationale,
            evidence,
            started_at,
            completed_at,
        )
        return {
            "evidence": [*state["evidence"], evidence],
            "tools_used": [*state["tools_used"], request.name],
            "tool_call_count": sequence,
            "invalid_tool_calls": state["invalid_tool_calls"]
            + int(validation_error is not None or is_duplicate),
            "tool_request_signatures": [*state["tool_request_signatures"], signature],
            "pending_decision": None,
        }

    def _finalize(self, state: _InvestigationState) -> dict[str, Any]:
        decision = state["pending_decision"]
        conclusion = None if decision is None else decision.conclusion
        if conclusion is None:
            conclusion = _inconclusive_conclusion(
                state["evidence"], "The model did not provide a conclusion"
            )
        conclusion = _sanitize_conclusion(conclusion, state["evidence"])
        status = "completed"
        if state["fatal_model_error"]:
            status = "failed"
        elif conclusion.confidence == ConfidenceLevel.INSUFFICIENT:
            status = "inconclusive"
        completed_at = datetime.now(UTC)
        telemetry = getattr(self.model, "telemetry", None)
        report = InvestigationReport(
            investigation_id=state["investigation_id"],
            incident_id=str(state["incident"]["incident_id"]),
            status=status,
            started_at=state["started_at"],
            completed_at=completed_at,
            hypotheses=state["hypotheses"],
            evidence=state["evidence"],
            tools_used=state["tools_used"],
            conclusion=conclusion,
            tool_call_count=state["tool_call_count"],
            token_usage=state["token_usage"] or None,
            duration_seconds=round((completed_at - state["started_at"]).total_seconds(), 6),
            decision_count=state["decision_count"],
            invalid_tool_calls=state["invalid_tool_calls"]
            + getattr(telemetry, "invalid_tool_calls", 0),
            model_validation_failures=getattr(telemetry, "validation_failures", 0),
            investigation_failures=int(state["fatal_model_error"]),
            timeouts=getattr(telemetry, "timeouts", 0),
            provider_name=getattr(telemetry, "provider_name", "unknown"),
            model_name=getattr(telemetry, "model_name", "unknown"),
        )
        self.repository.complete_investigation(report)
        return {"report": report}

    def _require_toolbox(self) -> InvestigationToolbox:
        if self._toolbox is None:
            raise RuntimeError("investigation toolbox has not been initialized")
        return self._toolbox


def _sanitize_hypotheses(
    hypotheses: list[RootCauseHypothesis], evidence: list[Evidence]
) -> list[RootCauseHypothesis]:
    valid_ids = {item.evidence_id for item in evidence}
    return [
        RootCauseHypothesis(
            item.hypothesis_id,
            item.statement,
            item.diagnosis_code,
            item.status,
            [reference for reference in item.supporting_evidence if reference in valid_ids],
            [reference for reference in item.contradicting_evidence if reference in valid_ids],
        )
        for item in hypotheses
    ]


def _sanitize_conclusion(
    conclusion: InvestigationConclusion, evidence: list[Evidence]
) -> InvestigationConclusion:
    valid_ids = {item.evidence_id for item in evidence}
    references = [item for item in conclusion.evidence_references if item in valid_ids]
    if conclusion.diagnosis_code is not None and not references:
        return _inconclusive_conclusion(
            evidence, "The proposed diagnosis did not reference any available evidence"
        )
    return InvestigationConclusion(
        conclusion.diagnosis_code,
        conclusion.final_diagnosis,
        conclusion.confidence,
        references,
        conclusion.outstanding_questions,
        conclusion.recommended_actions,
    )


def _inconclusive_conclusion(evidence: list[Evidence], reason: str) -> InvestigationConclusion:
    return InvestigationConclusion(
        None,
        reason,
        ConfidenceLevel.INSUFFICIENT,
        [item.evidence_id for item in evidence if item.success],
        ["What additional upstream or historical evidence can establish the cause?"],
        ["Collect the missing evidence before proposing a repair"],
    )
