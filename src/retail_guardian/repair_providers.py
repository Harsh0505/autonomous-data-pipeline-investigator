from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from retail_guardian.repair_models import RepairCapability, RepairProposal


class RepairPlanningError(RuntimeError):
    """The repair planner could not produce a safe structured proposal."""


@dataclass(frozen=True)
class RepairPlanningContext:
    investigation_id: str
    incident_id: str
    diagnosis_code: str
    diagnosis_summary: str
    supporting_evidence: dict[str, dict[str, Any]]
    failed_check_ids: list[str]
    pipeline_metadata: dict[str, Any]
    capabilities: list[RepairCapability]

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "investigation_id": self.investigation_id,
            "incident_id": self.incident_id,
            "selected_root_cause": {
                "diagnosis_code": self.diagnosis_code,
                "summary": self.diagnosis_summary,
            },
            "supporting_evidence": self.supporting_evidence,
            "failed_check_ids": self.failed_check_ids,
            "pipeline_metadata": self.pipeline_metadata,
            "available_repair_capabilities": [
                capability.model_dump(mode="json") for capability in self.capabilities
            ],
        }


class RepairPlanningModel(Protocol):
    def propose(self, context: RepairPlanningContext) -> RepairProposal: ...


class DeterministicRepairPlanningModel:
    """Reproducible planner used for integration tests and local demonstrations."""

    provider_name = "mock"
    model_name = "deterministic-repair-planner"

    def propose(self, context: RepairPlanningContext) -> RepairProposal:
        failed = set(context.failed_check_ids)
        ranked = sorted(
            context.capabilities,
            key=lambda item: len(failed & set(item.resolves_checks)),
            reverse=True,
        )
        if not ranked or not (failed & set(ranked[0].resolves_checks)):
            raise RepairPlanningError(
                "no available repair capability addresses the observed validation failures"
            )
        capability = ranked[0]
        if not capability.candidate_parameters:
            raise RepairPlanningError("the selected repair capability has no validated parameters")
        parameters = capability.candidate_parameters[0]
        return RepairProposal.model_validate(
            {
                "repair_type": capability.repair_type.value,
                "target_component": capability.target_component,
                "parameters": parameters,
                "explanation": (
                    f"Apply the scoped {capability.repair_type.value} capability because it "
                    f"addresses {capability.resolves_checks} using validated candidate parameters."
                ),
                "supporting_evidence_ids": list(context.supporting_evidence),
                "expected_effect": (
                    "Resolve the reported checks while preserving records outside the selected "
                    "repair scope."
                ),
                "risks_and_limitations": [capability.limitations],
                "verification_requirements": {
                    "checks_to_pass": context.failed_check_ids,
                    "preserve_unaffected_records": True,
                    "maximum_rows_removed": capability.maximum_expected_rows_removed,
                    "expected_changed_columns": capability.expected_changed_columns,
                },
                "rollback_approach": (
                    "Discard the isolated repaired copy; the immutable original is never replaced."
                ),
                "planner_provider": self.provider_name,
                "planner_model": self.model_name,
            }
        )


class OllamaRepairPlanningModel:
    """Structured local repair planner backed by Ollama."""

    def __init__(
        self,
        model: str | None = None,
        *,
        endpoint: str | None = None,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model or "qwen3:4b"
        self.endpoint = (endpoint or "http://localhost:11434").rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._client = client

    def propose(self, context: RepairPlanningContext) -> RepairProposal:
        schema = _repair_schema(context)
        messages = [
            {"role": "system", "content": _REPAIR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(context.to_prompt_payload(), sort_keys=True),
            },
        ]
        token_usage = 0
        usage_available = False
        last_error = "unknown structured-output error"
        for attempt in range(self.max_retries + 1):
            if attempt:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous proposal was rejected. Correct this specific error: "
                            f"{last_error[:800]}. Return only a proposal matching the schema."
                        ),
                    }
                )
            payload = self._request(messages, schema)
            prompt_tokens = payload.get("prompt_eval_count")
            output_tokens = payload.get("eval_count")
            if isinstance(prompt_tokens, int) and isinstance(output_tokens, int):
                token_usage += prompt_tokens + output_tokens
                usage_available = True
            content = (payload.get("message") or {}).get("content")
            try:
                proposal = RepairProposal.model_validate_json(content)
            except (ValidationError, ValueError, TypeError) as exc:
                last_error = str(exc)
                continue
            proposal = proposal.model_copy(
                update={
                    "planner_provider": "ollama",
                    "planner_model": self.model,
                    "token_usage": token_usage if usage_available else None,
                }
            )
            try:
                validate_repair_proposal(proposal, context)
            except RepairPlanningError as exc:
                last_error = str(exc)
                continue
            return proposal
        raise RepairPlanningError(
            f"Ollama exhausted its structured repair-proposal retries: {last_error}"
        )

    def _request(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        request = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": 0, "seed": 42},
        }
        try:
            if self._client is not None:
                response = self._client.post("/api/chat", json=request)
            else:
                with httpx.Client(base_url=self.endpoint, timeout=self.timeout_seconds) as client:
                    response = client.post("/api/chat", json=request)
        except httpx.HTTPError as exc:
            raise RepairPlanningError(f"Ollama is unavailable at {self.endpoint}: {exc}") from exc
        if response.status_code >= 400:
            raise RepairPlanningError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:500]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RepairPlanningError("Ollama returned malformed response JSON") from exc
        if not isinstance(payload, dict):
            raise RepairPlanningError("Ollama response must be a JSON object")
        return payload


def validate_repair_proposal(
    proposal: RepairProposal,
    context: RepairPlanningContext,
) -> None:
    """Validate model choices against incident-scoped repair capabilities."""
    capability = next(
        (
            item
            for item in context.capabilities
            if item.repair_type == proposal.repair_type
            and item.target_component == proposal.target_component
        ),
        None,
    )
    if capability is None:
        raise RepairPlanningError("the proposal selected an unavailable repair capability")
    parameters = proposal.parameters.model_dump(mode="json")
    if parameters not in capability.candidate_parameters:
        raise RepairPlanningError(
            "repair parameters are outside the validated candidates for this incident"
        )
    references = set(proposal.supporting_evidence_ids)
    if not references.issubset(context.supporting_evidence):
        invalid = sorted(references - set(context.supporting_evidence))
        raise RepairPlanningError(f"repair proposal fabricated evidence IDs: {invalid}")
    if set(proposal.verification_requirements.checks_to_pass) != set(context.failed_check_ids):
        raise RepairPlanningError(
            "verification requirements must include every current failed check"
        )
    if (
        proposal.verification_requirements.maximum_rows_removed
        != capability.maximum_expected_rows_removed
    ):
        raise RepairPlanningError(
            "maximum_rows_removed must equal the capability's incident-scoped safe value: "
            f"{capability.maximum_expected_rows_removed}"
        )
    if set(proposal.verification_requirements.expected_changed_columns) != set(
        capability.expected_changed_columns
    ):
        raise RepairPlanningError(
            "expected_changed_columns must exactly match the selected capability"
        )


def _repair_schema(context: RepairPlanningContext) -> dict[str, Any]:
    """Constrain structured output to incident-scoped operations and safety bounds."""
    schema = deepcopy(RepairProposal.model_json_schema())
    properties = schema["properties"]
    for model_owned_field in ("planner_provider", "planner_model", "token_usage"):
        properties.pop(model_owned_field, None)
    repair_types = sorted({item.repair_type.value for item in context.capabilities})
    schema["$defs"]["RepairType"]["enum"] = repair_types
    properties["target_component"]["enum"] = sorted(
        {item.target_component for item in context.capabilities}
    )
    candidates = [
        parameters
        for capability in context.capabilities
        for parameters in capability.candidate_parameters
    ]
    properties["parameters"] = {
        "oneOf": [{"const": candidate} for candidate in candidates],
        "title": "Incident-scoped repair parameters",
    }
    properties["supporting_evidence_ids"]["items"] = {
        "enum": sorted(context.supporting_evidence),
        "type": "string",
    }
    verification = schema["$defs"]["VerificationRequirements"]["properties"]
    verification["checks_to_pass"] = {
        "const": context.failed_check_ids,
        "type": "array",
    }
    verification["preserve_unaffected_records"] = {"const": True, "type": "boolean"}
    safe_row_limits = sorted({item.maximum_expected_rows_removed for item in context.capabilities})
    verification["maximum_rows_removed"] = {
        "enum": safe_row_limits,
        "type": "integer",
    }
    changed_column_sets = []
    for capability in context.capabilities:
        if capability.expected_changed_columns not in changed_column_sets:
            changed_column_sets.append(capability.expected_changed_columns)
    verification["expected_changed_columns"] = {
        "oneOf": [{"const": value} for value in changed_column_sets],
        "type": "array",
    }
    schema["$defs"]["VerificationRequirements"]["required"] = [
        "checks_to_pass",
        "preserve_unaffected_records",
        "maximum_rows_removed",
        "expected_changed_columns",
    ]
    return schema


_REPAIR_SYSTEM_PROMPT = """You are a repair-planning agent for a retail data pipeline. Select one
safe operation from available_repair_capabilities and use one of its candidate_parameters exactly.
Use only supplied evidence IDs. Do not generate code, SQL, shell commands, or unlisted operations.
The proposal will run only against an isolated snapshot and deterministic verification decides its
outcome. Set checks_to_pass to all failed_check_ids. Keep the rollback approach limited to
discarding the isolated copy. Copy maximum_expected_rows_removed and expected_changed_columns
exactly from the selected capability. Return only the structured proposal object."""
