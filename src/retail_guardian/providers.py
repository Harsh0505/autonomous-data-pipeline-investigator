from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from retail_guardian.investigation_models import (
    ConfidenceLevel,
    HypothesisStatus,
    InvestigationConclusion,
    InvestigationContext,
    ModelDecision,
    ProviderTelemetry,
    RejectedDecisionDiagnostic,
    RootCauseHypothesis,
    ToolRequest,
)


class InvestigationModel(Protocol):
    """Provider boundary for one evidence-aware investigation decision."""

    def decide(self, context: InvestigationContext) -> ModelDecision: ...


class ProviderError(RuntimeError):
    """Base error for a model provider failure."""


class ProviderUnavailableError(ProviderError):
    pass


class ModelUnavailableError(ProviderError):
    pass


class ProviderTimeoutError(ProviderError):
    pass


class ModelOutputValidationError(ProviderError):
    pass


class InvalidToolSelectionError(ModelOutputValidationError):
    pass


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

    def __init__(self) -> None:
        self.telemetry = ProviderTelemetry("mock", "heuristic")

    def decide(self, context: InvestigationContext) -> ModelDecision:
        self.telemetry.calls += 1
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
        self.telemetry = ProviderTelemetry("openai", self.model)

    def decide(self, context: InvestigationContext) -> ModelDecision:
        started_at = perf_counter()
        self.telemetry.calls += 1
        try:
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
        finally:
            self.telemetry.duration_seconds += perf_counter() - started_at
        payload = response.output_parsed
        if payload is None:
            self.telemetry.validation_failures += 1
            raise RuntimeError("model returned no structured investigation decision")
        usage = getattr(response, "usage", None)
        token_usage = getattr(usage, "total_tokens", None) if usage else None
        decision = payload.to_domain(token_usage)
        try:
            _validate_decision_for_context(decision, context)
        except InvalidToolSelectionError:
            self.telemetry.validation_failures += 1
            self.telemetry.invalid_tool_calls += 1
            raise
        except ModelOutputValidationError:
            self.telemetry.validation_failures += 1
            raise
        return decision


class OllamaInvestigationModel:
    """Local Ollama provider using schema-constrained, validated decisions."""

    def __init__(
        self,
        model: str | None = None,
        *,
        endpoint: str | None = None,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0 or max_retries > 5:
            raise ValueError("max_retries must be between 0 and 5")
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen3:4b")
        self.endpoint = (
            endpoint or os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")
        ).rstrip("/")
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("Ollama endpoint must use http:// or https://")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.client = client or httpx.Client()
        self.telemetry = ProviderTelemetry("ollama", self.model)

    def decide(self, context: InvestigationContext) -> ModelDecision:
        token_usage = 0
        last_error: ProviderError | None = None
        correction: str | None = None
        for attempt in range(self.max_retries + 1):
            content: str | None = None
            if attempt:
                self.telemetry.retries += 1
            try:
                response_payload = self._request(context, correction)
                token_usage += _ollama_token_usage(response_payload)
                content = response_payload.get("message", {}).get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ModelOutputValidationError(
                        "Ollama response did not contain message.content"
                    )
                payload = _DecisionPayload.model_validate_json(content)
                decision = payload.to_domain(token_usage or None)
                _validate_decision_for_context(decision, context)
                return decision
            except (ValidationError, json.JSONDecodeError) as exc:
                self.telemetry.validation_failures += 1
                error_type = (
                    InvalidToolSelectionError
                    if isinstance(exc, ValidationError) and _is_tool_request_validation_error(exc)
                    else ModelOutputValidationError
                )
                if error_type is InvalidToolSelectionError:
                    self.telemetry.invalid_tool_calls += 1
                last_error = error_type(f"Ollama returned an invalid structured decision: {exc}")
            except InvalidToolSelectionError as exc:
                self.telemetry.validation_failures += 1
                self.telemetry.invalid_tool_calls += 1
                last_error = exc
            except ModelOutputValidationError as exc:
                self.telemetry.validation_failures += 1
                last_error = exc
            except (ProviderUnavailableError, ProviderTimeoutError) as exc:
                last_error = exc
            if isinstance(last_error, ModelOutputValidationError):
                self.telemetry.rejected_decisions.append(
                    _rejection_diagnostic(attempt + 1, last_error, content)
                )
                correction = _correction_feedback(last_error, context)
            if attempt == self.max_retries:
                break
        assert last_error is not None
        raise last_error

    def _request(self, context: InvestigationContext, correction: str | None) -> dict[str, Any]:
        self.telemetry.calls += 1
        started_at = perf_counter()
        messages = [
            {
                "role": "system",
                "content": (
                    f"{_OLLAMA_SYSTEM_PROMPT}\nDecision JSON Schema:\n"
                    f"{json.dumps(_DecisionPayload.model_json_schema(), sort_keys=True)}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(_context_payload(context), sort_keys=True),
            },
        ]
        if correction:
            messages.append(
                {
                    "role": "user",
                    "content": correction,
                }
            )
        try:
            response = self.client.post(
                f"{self.endpoint}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "format": _DecisionPayload.model_json_schema(),
                    "options": {"temperature": 0},
                },
                timeout=self.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            self.telemetry.timeouts += 1
            raise ProviderTimeoutError(
                f"Ollama request timed out after {self.timeout_seconds:g} seconds"
            ) from exc
        except httpx.RequestError as exc:
            self.telemetry.connection_failures += 1
            raise ProviderUnavailableError(
                f"Ollama is unavailable at {self.endpoint}: {exc}"
            ) from exc
        finally:
            self.telemetry.duration_seconds += perf_counter() - started_at

        if response.status_code >= 400:
            error = _ollama_error(response)
            if response.status_code == 404 or (
                "model" in error.lower() and "not found" in error.lower()
            ):
                raise ModelUnavailableError(f"Ollama model {self.model!r} is unavailable: {error}")
            self.telemetry.connection_failures += 1
            raise ProviderUnavailableError(f"Ollama returned HTTP {response.status_code}: {error}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ModelOutputValidationError("Ollama returned malformed response JSON") from exc
        if not isinstance(payload, dict):
            raise ModelOutputValidationError("Ollama response must be a JSON object")
        return payload


class _HypothesisPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    hypothesis_id: str
    statement: str
    diagnosis_code: str | None = Field(
        description="Tentative diagnosis code, or null while gathering evidence."
    )
    status: Literal["active", "rejected", "confirmed"]
    supporting_evidence: list[str] = Field(
        description=(
            "Existing evidence IDs supporting this hypothesis. Use [] for a tentative hypothesis "
            "when no supporting evidence exists yet."
        )
    )
    contradicting_evidence: list[str] = Field(
        description="Existing evidence IDs contradicting this hypothesis, or [] when none exist."
    )

    def to_domain(self) -> RootCauseHypothesis:
        return RootCauseHypothesis(
            self.hypothesis_id,
            self.statement,
            self.diagnosis_code,
            HypothesisStatus(self.status),
            self.supporting_evidence,
            self.contradicting_evidence,
        )


class _NoArgumentToolRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: Literal[
        "inspect_dataset",
        "get_validation_failures",
        "compare_schemas",
        "compare_distributions",
        "get_data_contract",
    ]
    rationale: str


class _SampleRecordsToolRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: Literal["sample_problematic_records"]
    check_id: str | None
    limit: int | None
    rationale: str


class _PipelineHistoryToolRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: Literal["get_pipeline_history"]
    limit: int | None
    rationale: str


class _AffectedDimensionsToolRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: Literal["identify_affected_dimensions"]
    check_id: str | None
    rationale: str


_ToolRequestPayload = (
    _NoArgumentToolRequestPayload
    | _SampleRecordsToolRequestPayload
    | _PipelineHistoryToolRequestPayload
    | _AffectedDimensionsToolRequestPayload
)


class _ConclusionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    diagnosis_code: str | None
    final_diagnosis: str
    confidence: Literal["high", "medium", "low", "insufficient_evidence"]
    evidence_references: list[str] = Field(
        description=(
            "Existing evidence IDs supporting the final diagnosis. A non-null diagnosis must cite "
            "at least one available evidence ID."
        )
    )
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
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["tool", "conclude"]
    hypotheses: list[_HypothesisPayload]
    tool_request: _ToolRequestPayload | None
    conclusion: _ConclusionPayload | None

    @model_validator(mode="after")
    def validate_action_payload(self) -> _DecisionPayload:
        if self.action == "tool" and (self.tool_request is None or self.conclusion is not None):
            raise ValueError("tool decisions require tool_request and no conclusion")
        if self.action == "conclude" and (self.conclusion is None or self.tool_request is not None):
            raise ValueError("conclude decisions require conclusion and no tool_request")
        return self

    def to_domain(self, token_usage: int | None) -> ModelDecision:
        tool_request = None
        if self.tool_request is not None:
            payload = self.tool_request.model_dump()
            name = payload.pop("name")
            rationale = payload.pop("rationale")
            arguments = {key: value for key, value in payload.items() if value is not None}
            tool_request = ToolRequest(
                name,
                arguments,
                rationale,
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
        "available_evidence_ids": [item.evidence_id for item in context.evidence],
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


def _validate_decision_for_context(decision: ModelDecision, context: InvestigationContext) -> None:
    evidence_ids = {item.evidence_id for item in context.evidence}
    for hypothesis in decision.hypotheses:
        references = {
            *hypothesis.supporting_evidence,
            *hypothesis.contradicting_evidence,
        }
        unavailable = references - evidence_ids
        if unavailable:
            available = sorted(evidence_ids) or ["none"]
            raise ModelOutputValidationError(
                f"hypothesis {hypothesis.hypothesis_id!r} referenced unavailable evidence IDs: "
                f"{sorted(unavailable)}. Available evidence IDs: {available}. Tentative active "
                "hypotheses may use empty supporting_evidence and contradicting_evidence arrays."
            )

    if decision.action == "tool":
        if decision.tool_request is None:
            raise ModelOutputValidationError("tool decision omitted tool_request")
        definitions = {item.name: item for item in context.available_tools}
        definition = definitions.get(decision.tool_request.name)
        if definition is None:
            raise InvalidToolSelectionError(
                f"unauthorised investigation tool: {decision.tool_request.name}"
            )
        allowed_arguments = set(definition.parameters)
        unexpected = set(decision.tool_request.arguments) - allowed_arguments
        if unexpected:
            raise InvalidToolSelectionError(
                f"unsupported arguments for {definition.name}: {sorted(unexpected)}"
            )
        try:
            _validate_tool_arguments(decision.tool_request.arguments, definition.parameters)
        except ModelOutputValidationError as exc:
            raise InvalidToolSelectionError(str(exc)) from exc
        return

    conclusion = decision.conclusion
    if conclusion is None:
        raise ModelOutputValidationError("conclude decision omitted conclusion")
    allowed_diagnoses = {
        "schema_change",
        "duplicate_records",
        "currency_mismatch",
        "missing_values",
        "unexpected_categories",
        "delayed_data",
        "statistical_price_anomaly",
    }
    if conclusion.diagnosis_code is not None and conclusion.diagnosis_code not in allowed_diagnoses:
        raise ModelOutputValidationError(f"unsupported diagnosis code: {conclusion.diagnosis_code}")
    references = set(conclusion.evidence_references)
    if not references.issubset(evidence_ids):
        raise ModelOutputValidationError(
            "conclusion referenced evidence that is not in the investigation state"
        )
    if conclusion.diagnosis_code is not None and not references:
        raise ModelOutputValidationError("a diagnosis must cite at least one evidence ID")


def _validate_tool_arguments(arguments: dict[str, Any], parameters: dict[str, Any]) -> None:
    for name, value in arguments.items():
        specification = parameters[name]
        if specification.get("type") == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ModelOutputValidationError(f"{name} must be an integer")
            minimum = specification.get("minimum")
            maximum = specification.get("maximum")
            if minimum is not None and value < minimum or maximum is not None and value > maximum:
                raise ModelOutputValidationError(f"{name} is outside the allowed range")
        elif specification.get("type") == "string" and not isinstance(value, str):
            raise ModelOutputValidationError(f"{name} must be a string")


def _ollama_token_usage(payload: dict[str, Any]) -> int:
    prompt_tokens = payload.get("prompt_eval_count", 0)
    completion_tokens = payload.get("eval_count", 0)
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        return 0
    return prompt_tokens + completion_tokens


def _is_tool_request_validation_error(error: ValidationError) -> bool:
    return any("tool_request" in item.get("loc", ()) for item in error.errors())


def _ollama_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or "unknown Ollama error"
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return payload["error"]
    return response.text.strip() or "unknown Ollama error"


def _correction_feedback(error: ModelOutputValidationError, context: InvestigationContext) -> str:
    evidence_ids = [item.evidence_id for item in context.evidence]
    available = ", ".join(evidence_ids) if evidence_ids else "none"
    return (
        "The previous structured decision was rejected.\n"
        f"Specific validation failure: {str(error)[:1000]}\n"
        f"Available evidence IDs: {available}.\n"
        "Correct that specific error and return exactly one decision matching the JSON Schema. "
        "A tentative active hypothesis may use empty evidence arrays. You may select any listed "
        "investigation tool before establishing a diagnosis. Never invent an evidence ID."
    )


def _rejection_diagnostic(
    attempt: int, error: ModelOutputValidationError, content: str | None
) -> RejectedDecisionDiagnostic:
    return RejectedDecisionDiagnostic(
        attempt=attempt,
        error_type=type(error).__name__,
        error=str(error)[:2000],
        decision=_safe_rejected_decision(content),
        content_length=0 if content is None else len(content),
    )


def _safe_rejected_decision(content: str | None) -> dict[str, Any] | None:
    if not content:
        return None
    try:
        payload = json.loads(content)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    sanitized: dict[str, Any] = {"action": payload.get("action")}
    hypotheses = payload.get("hypotheses")
    if isinstance(hypotheses, list):
        sanitized["hypotheses"] = [
            {
                key: _safe_diagnostic_value(item.get(key))
                for key in (
                    "hypothesis_id",
                    "statement",
                    "diagnosis_code",
                    "status",
                    "supporting_evidence",
                    "contradicting_evidence",
                )
            }
            for item in hypotheses[:20]
            if isinstance(item, dict)
        ]
    tool_request = payload.get("tool_request")
    if isinstance(tool_request, dict):
        sanitized["tool_request"] = {
            key: _safe_diagnostic_value(tool_request.get(key))
            for key in ("name", "check_id", "limit", "rationale")
        }
    conclusion = payload.get("conclusion")
    if isinstance(conclusion, dict):
        sanitized["conclusion"] = {
            key: _safe_diagnostic_value(conclusion.get(key))
            for key in (
                "diagnosis_code",
                "final_diagnosis",
                "confidence",
                "evidence_references",
                "outstanding_questions",
                "recommended_actions",
            )
        }
    return sanitized


def _safe_diagnostic_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:1000]
    if isinstance(value, list):
        return [_safe_diagnostic_value(item) for item in value[:20]]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:1000]


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
not a fixed sequence. Hypotheses are tentative while evidence is being gathered: an active
hypothesis may have a null diagnosis_code and empty supporting_evidence and contradicting_evidence
arrays. Never invent evidence IDs; only use IDs listed in available_evidence_ids. A supported final
diagnosis must cite actual evidence IDs, while an insufficient-evidence conclusion may cite none.
You may select any listed investigation tool before establishing a diagnosis. Treat confidence as a
qualitative uncertainty label, not a probability. Conclude with insufficient_evidence when the
available observations do not support a root cause. Do not repeat a tool when successful evidence
from that tool is already present. If current evidence supports a diagnosis, conclude instead of
collecting redundant evidence. Tool arguments must match a listed tool. Return the requested
structured decision."""


_OLLAMA_SYSTEM_PROMPT = f"""{_SYSTEM_PROMPT}
Return only the structured decision object. Do not include private reasoning, markdown, or prose
outside the object. A tool decision must set conclusion to null. A conclude decision must set
tool_request to null. Optional tool fields that do not apply must be null. When
available_evidence_ids is empty, tentative hypotheses must use empty evidence-reference arrays."""
