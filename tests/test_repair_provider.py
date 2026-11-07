from __future__ import annotations

import json

import httpx

from retail_guardian.repair_models import RepairCapability
from retail_guardian.repair_providers import (
    OllamaRepairPlanningModel,
    RepairPlanningContext,
)


def _context() -> RepairPlanningContext:
    capability = RepairCapability.model_validate(
        {
            "repair_type": "deduplicate_records",
            "target_component": "dataset",
            "description": "Remove selected duplicate order rows.",
            "limitations": "Does not fix the upstream retry mechanism.",
            "resolves_checks": ["unique_order_id"],
            "compatible_diagnoses": ["failed_job_retry"],
            "candidate_parameters": [
                {
                    "identifier_column": "order_id",
                    "identifiers": ["ORD-1"],
                    "keep": "first",
                }
            ],
            "parameter_schema": {},
            "maximum_expected_rows_removed": 1,
            "expected_changed_columns": [],
        }
    )
    return RepairPlanningContext(
        investigation_id="INV-1",
        incident_id="INC-1",
        diagnosis_code="failed_job_retry",
        diagnosis_summary="A retry repeated a durable write",
        supporting_evidence={"EV-1": {"summary": "Retry and write evidence"}},
        failed_check_ids=["unique_order_id"],
        pipeline_metadata={"execution_id": "RUN-1"},
        capabilities=[capability],
    )


def _proposal() -> dict:
    return {
        "repair_type": "deduplicate_records",
        "target_component": "dataset",
        "parameters": {
            "identifier_column": "order_id",
            "identifiers": ["ORD-1"],
            "keep": "first",
        },
        "explanation": "Remove only the verified duplicate identifier.",
        "supporting_evidence_ids": ["EV-1"],
        "expected_effect": "Restore identifier uniqueness.",
        "risks_and_limitations": ["The upstream retry still requires correction."],
        "verification_requirements": {
            "checks_to_pass": ["unique_order_id"],
            "preserve_unaffected_records": True,
            "maximum_rows_removed": 1,
            "expected_changed_columns": [],
        },
        "rollback_approach": "Discard the isolated repaired copy.",
        "planner_provider": "unknown",
        "planner_model": "unknown",
        "token_usage": None,
    }


def test_ollama_repair_planner_returns_structured_proposal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "qwen3:4b"
        assert payload["think"] is False
        assert payload["format"]["type"] == "object"
        verification = payload["format"]["$defs"]["VerificationRequirements"]["properties"]
        assert verification["maximum_rows_removed"]["enum"] == [1]
        assert verification["checks_to_pass"]["const"] == ["unique_order_id"]
        assert (
            "maximum_rows_removed"
            in payload["format"]["$defs"]["VerificationRequirements"]["required"]
        )
        assert payload["format"]["properties"]["parameters"]["oneOf"][0]["const"] == {
            "identifier_column": "order_id",
            "identifiers": ["ORD-1"],
            "keep": "first",
        }
        supplied = json.loads(payload["messages"][1]["content"])
        assert supplied["selected_root_cause"]["diagnosis_code"] == "failed_job_retry"
        return httpx.Response(
            200,
            json={
                "message": {"content": json.dumps(_proposal())},
                "prompt_eval_count": 100,
                "eval_count": 30,
            },
        )

    with httpx.Client(
        base_url="http://ollama.test:11434",
        transport=httpx.MockTransport(handler),
    ) as client:
        planner = OllamaRepairPlanningModel(client=client)
        proposal = planner.propose(_context())

    assert proposal.repair_type.value == "deduplicate_records"
    assert proposal.planner_provider == "ollama"
    assert proposal.token_usage == 130


def test_ollama_repair_planner_recovers_from_malformed_output() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        content = "not-json" if attempts == 1 else json.dumps(_proposal())
        return httpx.Response(
            200,
            json={
                "message": {"content": content},
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )

    with httpx.Client(
        base_url="http://ollama.test:11434",
        transport=httpx.MockTransport(handler),
    ) as client:
        planner = OllamaRepairPlanningModel(client=client, max_retries=1)
        proposal = planner.propose(_context())

    assert attempts == 2
    assert proposal.token_usage == 30


def test_ollama_repair_planner_retries_semantically_unsafe_bounds() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        proposal = _proposal()
        if attempts == 1:
            proposal["verification_requirements"]["maximum_rows_removed"] = 0
        else:
            correction = json.loads(request.content)["messages"][-1]["content"]
            assert "maximum_rows_removed" in correction
        return httpx.Response(
            200,
            json={
                "message": {"content": json.dumps(proposal)},
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )

    with httpx.Client(
        base_url="http://ollama.test:11434",
        transport=httpx.MockTransport(handler),
    ) as client:
        planner = OllamaRepairPlanningModel(client=client, max_retries=1)
        proposal = planner.propose(_context())

    assert attempts == 2
    assert proposal.verification_requirements.maximum_rows_removed == 1
