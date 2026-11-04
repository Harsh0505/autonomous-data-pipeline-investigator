from __future__ import annotations

import json
import os
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from retail_guardian.investigation_models import (
    ConfidenceLevel,
    HypothesisStatus,
    InvestigationConclusion,
    InvestigationContext,
    ModelDecision,
    RootCauseHypothesis,
    ToolRequest,
)


class InvestigationModel(Protocol):
    """Provider boundary for one evidence-aware investigation decision."""

    def decide(self, context: InvestigationContext) -> ModelDecision: ...


class HeuristicInvestigationModel:
    """Deterministic evidence-driven model for tests and benchmark baselines."""

    _diagnoses = {
        "schema_contract": ("schema_change", "The upstream dataset schema changed"),
        "unique_order_id": ("duplicate_records", "The feed contains duplicate order records"),
        "supplier_currency": (
            "currency_mismatch",
            "Reported currencies conflict with supplier contracts",
        ),
        "required_values": ("missing_values", "Required sale values are missing"),
        "known_category": (
            "unexpected_categories",
            "The feed contains categories outside the data contract",
        ),
        "delivery_freshness": ("delayed_data", "The supplier feed missed its freshness SLA"),
        "unit_price_z_score": (
            "statistical_price_anomaly",
            "Unit prices depart materially from category baselines",
        ),
    }
    _target_tools = {
        "schema_contract": "compare_schemas",
        "unique_order_id": "sample_problematic_records",
        "supplier_currency": "get_data_contract",
        "required_values": "sample_problematic_records",
        "known_category": "get_data_contract",
        "delivery_freshness": "get_pipeline_history",
        "unit_price_z_score": "compare_distributions",
    }

    def decide(self, context: InvestigationContext) -> ModelDecision:
        validation_evidence = next(
            (item for item in context.evidence if item.tool_name == "get_validation_failures"),
            None,
        )
        if validation_evidence is None and context.remaining_tool_calls > 0:
            hypothesis = RootCauseHypothesis(
                "H-1",
                "A data-contract or distribution defect caused validation to reject the feed",
                None,
            )
            return ModelDecision(
                action="tool",
                hypotheses=[hypothesis],
                tool_request=ToolRequest(
                    "get_validation_failures",
                    {},
                    "Start with directly observed validation outcomes before narrowing the cause",
                ),
            )

        failures = (
            [] if validation_evidence is None else validation_evidence.data.get("failures", [])
        )
        failed_ids = [item["check_id"] for item in failures]
        primary = next((check for check in self._diagnoses if check in failed_ids), None)
        if primary is None:
            return ModelDecision(
                action="conclude",
                hypotheses=context.hypotheses,
                conclusion=InvestigationConclusion(
                    None,
                    "The available evidence does not establish a root cause",
                    ConfidenceLevel.INSUFFICIENT,
                    [item.evidence_id for item in context.evidence],
                    ["Which upstream change or delivery event preceded this incident?"],
                    ["Collect additional validated history and retry the investigation"],
                ),
            )

        diagnosis_code, statement = self._diagnoses[primary]
        target_tool = self._target_tools[primary]
        used_tools = {item.tool_name for item in context.evidence}
        hypothesis = RootCauseHypothesis(
            "H-1",
            statement,
            diagnosis_code,
            supporting_evidence=[validation_evidence.evidence_id] if validation_evidence else [],
        )
        if target_tool not in used_tools and context.remaining_tool_calls > 0:
            arguments = (
                {"check_id": primary, "limit": 10}
                if target_tool == "sample_problematic_records"
                else {"limit": 10}
                if target_tool == "get_pipeline_history"
                else {}
            )
            return ModelDecision(
                action="tool",
                hypotheses=[hypothesis],
                tool_request=ToolRequest(
                    target_tool,
                    arguments,
                    f"Seek independent evidence for the {diagnosis_code} hypothesis",
                ),
            )

        supporting = [item.evidence_id for item in context.evidence if item.success]
        hypothesis = RootCauseHypothesis(
            "H-1",
            statement,
            diagnosis_code,
            HypothesisStatus.CONFIRMED,
            supporting_evidence=supporting,
        )
        confidence = ConfidenceLevel.HIGH
        if any(not item.success for item in context.evidence):
            confidence = ConfidenceLevel.MEDIUM
        return ModelDecision(
            action="conclude",
            hypotheses=[hypothesis],
            conclusion=InvestigationConclusion(
                diagnosis_code,
                statement,
                confidence,
                supporting,
                [],
                _recommended_actions(diagnosis_code),
            ),
        )


class OpenAIInvestigationModel:
    """OpenAI Responses API provider using a strict structured decision schema."""

    def __init__(self, model: str | None = None) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI provider")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5.5")

    def decide(self, context: InvestigationContext) -> ModelDecision:
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(_context_payload(context), sort_keys=True),
                },
            ],
            text_format=_DecisionPayload,
        )
        payload = response.output_parsed
        if payload is None:
            raise RuntimeError("model returned no structured investigation decision")
        usage = getattr(response, "usage", None)
        token_usage = getattr(usage, "total_tokens", None) if usage else None
        return payload.to_domain(token_usage)


class _HypothesisPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    statement: str
    diagnosis_code: str | None
    status: Literal["active", "rejected", "confirmed"]
    supporting_evidence: list[str]
    contradicting_evidence: list[str]

    def to_domain(self) -> RootCauseHypothesis:
        return RootCauseHypothesis(
            self.hypothesis_id,
            self.statement,
            self.diagnosis_code,
            HypothesisStatus(self.status),
            self.supporting_evidence,
            self.contradicting_evidence,
        )


class _ToolRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    check_id: str | None
    limit: int | None
    rationale: str


class _ConclusionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnosis_code: str | None
    final_diagnosis: str
    confidence: Literal["high", "medium", "low", "insufficient_evidence"]
    evidence_references: list[str]
    outstanding_questions: list[str]
    recommended_actions: list[str]

    def to_domain(self) -> InvestigationConclusion:
        return InvestigationConclusion(
            self.diagnosis_code,
            self.final_diagnosis,
            ConfidenceLevel(self.confidence),
            self.evidence_references,
            self.outstanding_questions,
            self.recommended_actions,
        )


class _DecisionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["tool", "conclude"]
    hypotheses: list[_HypothesisPayload]
    tool_request: _ToolRequestPayload | None
    conclusion: _ConclusionPayload | None

    def to_domain(self, token_usage: int | None) -> ModelDecision:
        tool_request = None
        if self.tool_request is not None:
            arguments = {}
            if self.tool_request.check_id is not None:
                arguments["check_id"] = self.tool_request.check_id
            if self.tool_request.limit is not None:
                arguments["limit"] = self.tool_request.limit
            tool_request = ToolRequest(
                self.tool_request.name,
                arguments,
                self.tool_request.rationale,
            )
        return ModelDecision(
            self.action,
            [item.to_domain() for item in self.hypotheses],
            tool_request,
            None if self.conclusion is None else self.conclusion.to_domain(),
            token_usage,
        )


def _context_payload(context: InvestigationContext) -> dict[str, object]:
    return {
        "observable_incident": context.incident,
        "current_hypotheses": [
            {
                "hypothesis_id": item.hypothesis_id,
                "statement": item.statement,
                "diagnosis_code": item.diagnosis_code,
                "status": item.status.value,
                "supporting_evidence": item.supporting_evidence,
                "contradicting_evidence": item.contradicting_evidence,
            }
            for item in context.hypotheses
        ],
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "tool_name": item.tool_name,
                "summary": item.summary,
                "data": item.data,
                "success": item.success,
                "error": item.error,
            }
            for item in context.evidence
        ],
        "available_tools": [
            {
                "name": item.name,
                "description": item.description,
                "parameters": item.parameters,
            }
            for item in context.available_tools
        ],
        "remaining_tool_calls": context.remaining_tool_calls,
    }


def _recommended_actions(diagnosis_code: str) -> list[str]:
    actions = {
        "schema_change": [
            "Confirm the supplier contract change before updating the ingestion mapping"
        ],
        "duplicate_records": ["Quarantine duplicate order IDs and audit upstream retry semantics"],
        "currency_mismatch": [
            "Quarantine affected prices and confirm supplier currency configuration"
        ],
        "missing_values": ["Request a corrected feed and trace the producing field transformation"],
        "unexpected_categories": [
            "Confirm whether the new category is valid before updating taxonomy"
        ],
        "delayed_data": [
            "Check supplier delivery status and defer downstream freshness-sensitive jobs"
        ],
        "statistical_price_anomaly": ["Review sampled prices before allowing revenue aggregation"],
    }
    return actions.get(diagnosis_code, ["Escalate for additional upstream evidence"])


_SYSTEM_PROMPT = """You investigate data-pipeline incidents using only the supplied observable
incident, tool evidence, and data contracts. Maintain explicit hypotheses and select at most one
scoped tool per decision when evidence is insufficient. Investigation paths must follow findings,
not a fixed sequence. Never invent evidence. Every conclusion and hypothesis must cite evidence IDs.
Treat confidence as a qualitative uncertainty label, not a probability. Conclude with
insufficient_evidence when the available observations do not support a root cause. Tool arguments
must match a listed tool. Return the requested structured decision."""
