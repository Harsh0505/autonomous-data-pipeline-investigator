from __future__ import annotations

import json
import os
from copy import deepcopy
from time import perf_counter
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from retail_guardian.hypothesis_state import (
    HypothesisTransitionError,
    apply_hypothesis_operations,
    validate_supported_conclusion,
)
from retail_guardian.investigation_models import (
    ConfidenceLevel,
    Evidence,
    HypothesisOperation,
    HypothesisOperationType,
    InvestigationConclusion,
    InvestigationContext,
    ModelDecision,
    ProviderTelemetry,
    RejectedDecisionDiagnostic,
    ToolRequest,
    ToolResultKind,
)


class InvestigationModel(Protocol):
    """Provider boundary for one evidence-aware investigation decision."""

    def decide(self, context: InvestigationContext) -> ModelDecision: ...


DiagnosisCode = Literal[
    "schema_change",
    "duplicate_records",
    "currency_mismatch",
    "missing_values",
    "unexpected_categories",
    "delayed_data",
    "statistical_price_anomaly",
]


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
        self.telemetry.first_attempt_schema_valid_decisions += 1
        validation_evidence = next(
            (item for item in context.evidence if item.tool_name == "get_validation_failures"),
            None,
        )
        if validation_evidence is None and context.remaining_tool_calls > 0:
            return ModelDecision(
                action="tool",
                hypotheses=[],
                tool_request=ToolRequest(
                    "get_validation_failures",
                    {},
                    "Start with directly observed validation outcomes before narrowing the cause",
                ),
                hypothesis_operations=[
                    HypothesisOperation(
                        HypothesisOperationType.CREATE,
                        "H-1",
                        "A data-contract or distribution defect caused validation to reject "
                        "the feed",
                    )
                ],
            )

        failures = (
            [] if validation_evidence is None else validation_evidence.data.get("failures", [])
        )
        failed_ids = [item["check_id"] for item in failures]
        primary = next((check for check in self._diagnoses if check in failed_ids), None)
        if primary is None:
            return ModelDecision(
                action="conclude",
                hypotheses=[],
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
                hypotheses=[],
                tool_request=ToolRequest(
                    target_tool,
                    arguments,
                    f"Seek independent evidence for the {diagnosis_code} hypothesis",
                ),
                hypothesis_operations=[
                    HypothesisOperation(
                        HypothesisOperationType.UPDATE,
                        "H-1",
                        statement,
                        diagnosis_code,
                        [validation_evidence.evidence_id] if validation_evidence else [],
                    )
                ],
            )

        supporting = [item.evidence_id for item in context.evidence if item.success]
        confidence = ConfidenceLevel.HIGH
        if any(not item.success for item in context.evidence):
            confidence = ConfidenceLevel.MEDIUM
        return ModelDecision(
            action="conclude",
            hypotheses=[],
            conclusion=InvestigationConclusion(
                diagnosis_code,
                statement,
                confidence,
                supporting,
                [],
                _recommended_actions(diagnosis_code),
            ),
            hypothesis_operations=[
                HypothesisOperation(
                    HypothesisOperationType.CONFIRM,
                    "H-1",
                    statement,
                    diagnosis_code,
                    supporting,
                )
            ],
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
        self.telemetry.first_attempt_schema_valid_decisions += 1
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
                if attempt == 0:
                    self.telemetry.first_attempt_schema_valid_decisions += 1
                decision = payload.to_domain(token_usage or None)
                _validate_decision_for_context(decision, context)
                if attempt > 0:
                    self.telemetry.retry_recoveries += 1
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
                last_error = error_type(
                    "Ollama returned an invalid structured decision: "
                    f"{_validation_error_message(exc)}"
                )
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
        decision_schema = _decision_schema(context)
        messages = [
            {
                "role": "system",
                "content": _OLLAMA_SYSTEM_PROMPT,
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
                    "format": decision_schema,
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


class _HypothesisOperationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    operation: Literal["create", "update", "confirm", "reject"]
    hypothesis_id: str
    statement: str | None
    diagnosis_code: DiagnosisCode | None = Field(
        description="Tentative diagnosis code, or null while gathering evidence."
    )
    supporting_evidence: list[str] = Field(
        description=(
            "Bare evidence IDs to add as support. Use only IDs permitted by this schema; never "
            "include summaries or annotations."
        )
    )
    contradicting_evidence: list[str] = Field(
        description=(
            "Bare evidence IDs to add as contradictions. Use only IDs permitted by this schema."
        )
    )

    @model_validator(mode="after")
    def validate_operation_payload(self) -> _HypothesisOperationPayload:
        if self.operation == "create" and not self.statement:
            raise ValueError("create hypothesis operation requires statement")
        if self.operation == "update" and not any(
            (
                self.statement,
                self.diagnosis_code,
                self.supporting_evidence,
                self.contradicting_evidence,
            )
        ):
            raise ValueError("update hypothesis operation must change state")
        return self

    def to_domain(self) -> HypothesisOperation:
        return HypothesisOperation(
            HypothesisOperationType(self.operation),
            self.hypothesis_id,
            self.statement,
            self.diagnosis_code,
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

    diagnosis_code: DiagnosisCode | None
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
    hypothesis_operations: list[_HypothesisOperationPayload]
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
            action=self.action,
            hypotheses=[],
            tool_request=tool_request,
            conclusion=None if self.conclusion is None else self.conclusion.to_domain(),
            token_usage=token_usage,
            hypothesis_operations=[item.to_domain() for item in self.hypothesis_operations],
        )


def _context_payload(context: InvestigationContext) -> dict[str, object]:
    evidence_registry = {
        item.evidence_id: {
            "tool_name": item.tool_name,
            "result_kind": item.result_kind.value,
            "summary": item.summary,
            "key_findings": _evidence_key_findings(item),
            "error": item.error,
        }
        for item in context.evidence
    }
    return {
        "observable_incident": _compact_incident(context.incident),
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
        "evidence_registry": evidence_registry,
        "available_evidence_ids": [item.evidence_id for item in context.evidence],
        "supportable_evidence_ids": [
            item.evidence_id
            for item in context.evidence
            if item.result_kind == ToolResultKind.INFORMATIVE
        ],
        "available_check_ids": context.available_check_ids,
        "completed_tool_calls": [
            {
                "name": item.name,
                "arguments": item.arguments,
                "evidence_id": item.evidence_id,
                "result_kind": item.result_kind.value,
                "error": item.error,
            }
            for item in context.completed_tool_calls
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
        "remaining_invalid_tool_calls": context.remaining_invalid_tool_calls,
        "remaining_empty_tool_calls": context.remaining_empty_tool_calls,
        "decision_guidance": (
            "Apply only the hypothesis operations needed for this turn. Confirm a supported "
            "hypothesis and conclude as soon as evidence is sufficient; unused budget is not a "
            "reason to continue. Otherwise select one currently available tool. Use only bare "
            "evidence IDs and check IDs allowed by the schema."
        ),
        "diagnosis_taxonomy": {
            "schema_change": "Observed dataset columns or types changed from the expected schema.",
            "duplicate_records": "Order identifiers or records are duplicated.",
            "currency_mismatch": (
                "Observed supplier currencies conflict with configured supplier currencies."
            ),
            "missing_values": "Required sale fields contain missing values.",
            "unexpected_categories": "Observed categories are outside the configured taxonomy.",
            "delayed_data": "Data delivery exceeded its freshness or arrival-time contract.",
            "statistical_price_anomaly": (
                "Unit prices depart materially from established statistical baselines."
            ),
        },
    }


def _evidence_key_findings(evidence: Evidence) -> dict[str, Any]:
    if not evidence.success:
        return {"tool_error": evidence.error or "unknown error"}
    data = evidence.data
    if evidence.tool_name == "get_validation_failures":
        failures = []
        for result in data.get("failures", [])[:20]:
            finding = result.get("finding") or {}
            failures.append(
                {
                    "check_id": result.get("check_id"),
                    "kind": result.get("kind"),
                    "severity": result.get("severity"),
                    "message": finding.get("message"),
                    "affected_rows": finding.get("affected_rows", finding.get("affected_count")),
                    "sample_values": finding.get("sample_values", [])[:10],
                    "context": _compact_value(finding.get("context", {})),
                }
            )
        return {"failed_checks": failures, "failure_count": len(failures)}
    if evidence.tool_name == "get_data_contract":
        return {
            "required_columns": data.get("required_columns", []),
            "supplier_currencies": data.get("supplier_currencies", {}),
            "allowed_categories": data.get("allowed_categories", []),
            "maximum_delivery_delay_hours": data.get("maximum_delivery_delay_hours"),
            "unit_price_z_score_threshold": data.get("unit_price_z_score_threshold"),
        }
    if evidence.tool_name == "sample_problematic_records":
        samples = data.get("samples", [])
        return {
            "check_id": data.get("check_id"),
            "sample_count": len(samples),
            "samples": _compact_value(samples[:3]),
        }
    if evidence.tool_name == "inspect_dataset":
        profile = data.get("profile", {})
        columns = profile.get("columns", {})
        return {
            "row_count": profile.get("row_count"),
            "schema": data.get("schema", [])[:20],
            "columns_with_nulls": {
                name: details.get("null_count")
                for name, details in columns.items()
                if details.get("null_count", 0)
            },
            "numeric_summaries": {
                name: details["numeric"]
                for name, details in list(columns.items())[:20]
                if details.get("numeric")
            },
        }
    if evidence.tool_name == "get_pipeline_history":
        return {
            "execution_count": len(data.get("executions", [])),
            "executions": data.get("executions", [])[:5],
            "sufficient_history": data.get("sufficient_history"),
        }
    return _compact_value(data)


def _compact_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 3:
        return "[nested value omitted]"
    if isinstance(value, dict):
        return {
            str(key): _compact_value(item, depth=depth + 1)
            for key, item in list(value.items())[:20]
        }
    if isinstance(value, list):
        return [_compact_value(item, depth=depth + 1) for item in value[:10]]
    if isinstance(value, str):
        return value[:500]
    return value


def _compact_incident(incident: dict[str, Any]) -> dict[str, Any]:
    return {
        "incident_id": incident.get("incident_id"),
        "execution_id": incident.get("execution_id"),
        "source": incident.get("source"),
        "pipeline_stage": incident.get("pipeline_stage"),
        "pipeline_error": incident.get("pipeline_error"),
        "dataset": incident.get("dataset"),
        "metrics": incident.get("metrics"),
        "detected_checks": [
            {
                "check_id": item.get("check_id"),
                "severity": item.get("severity"),
                "message": (item.get("finding") or {}).get("message"),
            }
            for item in incident.get("violations", [])
        ],
    }


def _decision_schema(context: InvestigationContext) -> dict[str, Any]:
    """Constrain Ollama to one valid action and currently executable tools."""
    schema = deepcopy(_DecisionPayload.model_json_schema())
    supportable_ids = [
        item.evidence_id
        for item in context.evidence
        if item.result_kind == ToolResultKind.INFORMATIVE
    ]
    operation_properties = schema["$defs"]["_HypothesisOperationPayload"]["properties"]
    _restrict_array_values(operation_properties["supporting_evidence"], supportable_ids)
    _restrict_array_values(operation_properties["contradicting_evidence"], supportable_ids)
    conclusion_properties = schema["$defs"]["_ConclusionPayload"]["properties"]
    _restrict_array_values(conclusion_properties["evidence_references"], supportable_ids)
    for definition_name in (
        "_SampleRecordsToolRequestPayload",
        "_AffectedDimensionsToolRequestPayload",
    ):
        _restrict_optional_string(
            schema["$defs"][definition_name]["properties"]["check_id"],
            context.available_check_ids,
        )
    allowed_tools = {item.name for item in context.available_tools}
    tool_property = schema["properties"]["tool_request"]
    variants = tool_property["anyOf"]
    retained_variants: list[dict[str, Any]] = []
    tool_definition_names: set[str] = set()
    retained_definition_names: set[str] = set()
    for variant in variants:
        reference = variant.get("$ref")
        if reference is None:
            retained_variants.append(variant)
            continue
        definition_name = reference.rsplit("/", 1)[-1]
        tool_definition_names.add(definition_name)
        name_schema = schema["$defs"][definition_name]["properties"]["name"]
        names = set(name_schema.get("enum", []))
        if "const" in name_schema:
            names = {name_schema["const"]}
        available_names = sorted(names & allowed_tools)
        if not available_names:
            continue
        if "enum" in name_schema:
            name_schema["enum"] = available_names
        retained_variants.append(variant)
        retained_definition_names.add(definition_name)
    tool_property["anyOf"] = retained_variants
    for definition_name in tool_definition_names - retained_definition_names:
        del schema["$defs"][definition_name]

    operations_property = schema["properties"]["hypothesis_operations"]
    conclusion_variants = [
        variant
        for variant in schema["properties"]["conclusion"]["anyOf"]
        if variant.get("type") != "null"
    ]
    action_variants: list[dict[str, Any]] = []
    available_tool_variants = [
        variant for variant in retained_variants if variant.get("type") != "null"
    ]
    if available_tool_variants:
        action_variants.append(
            _action_schema(
                "tool",
                operations_property,
                {"anyOf": available_tool_variants},
                {"type": "null"},
            )
        )
    action_variants.append(
        _action_schema(
            "conclude",
            operations_property,
            {"type": "null"},
            {"anyOf": conclusion_variants},
        )
    )
    for key in ("properties", "required", "additionalProperties"):
        schema.pop(key, None)
    schema["oneOf"] = action_variants
    return schema


def _action_schema(
    action: Literal["tool", "conclude"],
    operations_schema: dict[str, Any],
    tool_request_schema: dict[str, Any],
    conclusion_schema: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"const": action, "type": "string"},
            "hypothesis_operations": operations_schema,
            "tool_request": tool_request_schema,
            "conclusion": conclusion_schema,
        },
        "required": ["action", "hypothesis_operations", "tool_request", "conclusion"],
    }


def _restrict_array_values(schema: dict[str, Any], values: list[str]) -> None:
    if values:
        schema["items"] = {"enum": values, "type": "string"}
    else:
        schema["maxItems"] = 0


def _restrict_optional_string(schema: dict[str, Any], values: list[str]) -> None:
    variants = schema.get("anyOf", [])
    if values:
        for variant in variants:
            if variant.get("type") == "string":
                variant["enum"] = values
    else:
        schema["anyOf"] = [variant for variant in variants if variant.get("type") == "null"]


def _validate_decision_for_context(decision: ModelDecision, context: InvestigationContext) -> None:
    evidence_ids = {item.evidence_id for item in context.evidence}
    supportable_evidence_ids = {
        item.evidence_id
        for item in context.evidence
        if item.result_kind == ToolResultKind.INFORMATIVE
    }
    try:
        hypotheses = apply_hypothesis_operations(
            context.hypotheses,
            decision.hypothesis_operations,
            context.evidence,
        )
    except HypothesisTransitionError as exc:
        raise ModelOutputValidationError(f"invalid hypothesis operation: {exc}") from exc

    if decision.action == "tool":
        if decision.tool_request is None:
            raise ModelOutputValidationError("tool decision omitted tool_request")
        definitions = {item.name: item for item in context.available_tools}
        definition = definitions.get(decision.tool_request.name)
        if definition is None:
            completed = sorted({item.name for item in context.completed_tool_calls if item.success})
            raise InvalidToolSelectionError(
                f"tool {decision.tool_request.name!r} is not currently available. "
                f"Completed successful tools: {completed or ['none']}. Select only a tool in "
                "available_tools, or conclude from the evidence already retrieved."
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
    if conclusion.diagnosis_code is not None and not references.issubset(supportable_evidence_ids):
        failed_references = sorted(references - supportable_evidence_ids)
        raise ModelOutputValidationError(
            "a diagnosis may cite only informative evidence IDs; invalid references: "
            f"{failed_references}"
        )
    if conclusion.diagnosis_code is not None:
        try:
            validate_supported_conclusion(
                conclusion.diagnosis_code,
                conclusion.evidence_references,
                hypotheses,
            )
        except HypothesisTransitionError as exc:
            raise ModelOutputValidationError(f"unsupported conclusion: {exc}") from exc


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
        allowed_values = specification.get("enum")
        if allowed_values is not None and value not in allowed_values:
            raise ModelOutputValidationError(
                f"{name} must be one of the available values: {allowed_values or ['none']}"
            )


def _ollama_token_usage(payload: dict[str, Any]) -> int:
    prompt_tokens = payload.get("prompt_eval_count", 0)
    completion_tokens = payload.get("eval_count", 0)
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        return 0
    return prompt_tokens + completion_tokens


def _is_tool_request_validation_error(error: ValidationError) -> bool:
    return any("tool_request" in item.get("loc", ()) for item in error.errors())


def _validation_error_message(error: ValidationError | json.JSONDecodeError) -> str:
    if isinstance(error, json.JSONDecodeError):
        return f"invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}"
    messages = []
    for item in error.errors()[:5]:
        location = ".".join(str(part) for part in item.get("loc", ())) or "decision"
        messages.append(f"{location}: {item.get('msg', 'invalid value')}")
    return "; ".join(messages)


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
    tools = ", ".join(item.name for item in context.available_tools) or "none"
    completed = (
        ", ".join(item.name for item in context.completed_tool_calls if item.success) or "none"
    )
    check_ids = ", ".join(context.available_check_ids) or "none"
    return (
        "The previous structured decision was rejected.\n"
        f"Specific validation failure: {str(error)[:1000]}\n"
        f"Available evidence IDs: {available}.\n"
        f"Currently available tools: {tools}.\n"
        f"Completed successful tools: {completed}.\n"
        f"Valid check_id values: {check_ids}.\n"
        "Correct that specific error and return exactly one decision matching the JSON Schema. "
        "Evidence arrays accept bare IDs only, without summaries or annotations. Use create only "
        "for a new hypothesis ID and update, confirm, or reject for an existing ID. Confirm a "
        "supported hypothesis before concluding with a diagnosis. Do not repeat a completed tool."
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
    operations = payload.get("hypothesis_operations")
    if isinstance(operations, list):
        sanitized["hypothesis_operations"] = [
            {
                key: _safe_diagnostic_value(item.get(key))
                for key in (
                    "operation",
                    "hypothesis_id",
                    "statement",
                    "diagnosis_code",
                    "supporting_evidence",
                    "contradicting_evidence",
                )
            }
            for item in operations[:20]
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
incident, canonical evidence registry, and contracts. Maintain hypotheses through explicit create,
update, confirm, and reject operations; never restate the full hypothesis list. IDs are stable after
creation. Evidence fields contain bare IDs only, selected from supportable_evidence_ids. Never put
prose, summaries, or annotated IDs in evidence fields. Confirming requires supporting evidence, and
a final diagnosis must match a confirmed hypothesis and cite its support. Select at most one
currently available tool when evidence is insufficient. check_id must come from available_check_ids.
Investigation paths follow findings rather than a fixed sequence. Conclude as soon as evidence is
sufficient; unused budget is not a reason to collect more. Conclude with insufficient_evidence when
available tools cannot resolve the uncertainty. Choose diagnosis_code only from diagnosis_taxonomy.
Prefer a directly established defect that explains secondary symptoms over treating every validation
symptom as an independent cause. Completed calls are an execution ledger, and absent tools must not
be selected. Return the requested structured decision."""


_OLLAMA_SYSTEM_PROMPT = f"""{_SYSTEM_PROMPT}
Return only the structured decision object. Do not include private reasoning, markdown, or prose
outside the object. A tool decision must set conclusion to null. A conclude decision must set
tool_request to null. Optional tool fields that do not apply must be null."""
