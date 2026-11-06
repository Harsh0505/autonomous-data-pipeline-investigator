from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from hashlib import sha256
from typing import Any
from uuid import uuid4

from retail_guardian.causal_scenarios import OperationalRecordType
from retail_guardian.config import PipelineConfig
from retail_guardian.investigation_models import (
    QUALITY_DIAGNOSIS_CODES,
    ROOT_CAUSE_DIAGNOSIS_CODES,
    Evidence,
    ToolDefinition,
    ToolResultKind,
)
from retail_guardian.storage import HistoryRepository


class InvestigationToolbox:
    """Incident-bound tools exposing only scoped, persisted pipeline evidence."""

    def __init__(
        self,
        repository: HistoryRepository,
        incident_id: str,
        config: PipelineConfig | None = None,
    ) -> None:
        self.repository = repository
        self.incident_id = incident_id
        self.config = config or PipelineConfig()
        execution = repository.get_execution(incident_id)
        if execution is None:
            raise ValueError(f"incident not found: {incident_id}")
        self.execution = execution
        self.execution_id = str(execution["execution_id"])
        self.operational_evidence_available = bool(
            repository.get_operational_records(self.execution_id, limit=1)
        )
        self._tools: dict[str, Callable[[dict[str, Any]], tuple[str, dict[str, Any]]]] = {
            "inspect_dataset": self._inspect_dataset,
            "get_validation_failures": self._get_validation_failures,
            "compare_schemas": self._compare_schemas,
            "compare_distributions": self._compare_distributions,
            "sample_problematic_records": self._sample_problematic_records,
            "get_pipeline_history": self._get_pipeline_history,
            "identify_affected_dimensions": self._identify_affected_dimensions,
            "get_data_contract": self._get_data_contract,
            "get_pipeline_logs": self._get_pipeline_logs,
            "get_supplier_submissions": self._get_supplier_submissions,
            "get_orchestration_events": self._get_orchestration_events,
            "get_transformation_metadata": self._get_transformation_metadata,
            "get_configuration_changes": self._get_configuration_changes,
        }

    @property
    def definitions(self) -> list[ToolDefinition]:
        check_id_parameter = {
            "type": "string",
            "required": False,
            "enum": self.available_check_ids,
        }
        definitions = [
            ToolDefinition(
                "inspect_dataset",
                "Current dataset profile: schema, nulls, cardinality, numeric summaries, "
                "and aggregates.",
                {},
                ("What does the current dataset contain?",),
                "Describes symptoms, not upstream execution causes.",
            ),
            ToolDefinition(
                "get_validation_failures",
                "Measured validation failures for the current execution.",
                {},
                ("Which deterministic or statistical checks failed?",),
                "Identifies observed symptoms but not their upstream cause.",
            ),
            ToolDefinition(
                "compare_schemas",
                "Current schema compared with the most relevant prior execution.",
                {},
                ("Did columns or data types change from a prior run?",),
                "Requires a prior comparable execution and covers schema only.",
            ),
            ToolDefinition(
                "compare_distributions",
                "Numeric distributions and dimension aggregates compared with a prior execution.",
                {},
                ("Did values or dimension volumes shift from a prior run?",),
                "Requires a prior comparable execution; correlation is not causal proof.",
            ),
            ToolDefinition(
                "sample_problematic_records",
                "Bounded records implicated by one validation check.",
                {
                    "check_id": check_id_parameter,
                    "limit": {
                        "type": "integer",
                        "required": False,
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                ("What values appear in records affected by a failed check?",),
                "A sample is not a complete population or operational audit trail.",
            ),
            ToolDefinition(
                "get_pipeline_history",
                "Prior healthy executions with separate event and ingestion timestamps.",
                {
                    "limit": {
                        "type": "integer",
                        "required": False,
                        "minimum": 1,
                        "maximum": 20,
                    }
                },
                ("Is the current execution unusual relative to earlier healthy runs?",),
                "Does not expose task retries, writes, lineage, or supplier submissions.",
            ),
            ToolDefinition(
                "identify_affected_dimensions",
                "Supplier and category counts from sampled problematic records.",
                {"check_id": check_id_parameter},
                ("Which sampled dimensions are affected?",),
                "Sample-scoped; does not establish the causal mechanism.",
            ),
            ToolDefinition(
                "get_data_contract",
                "Configured columns, currencies, categories, and validation thresholds.",
                {},
                ("What data rules currently apply?",),
                "Static contract state does not show execution, retry, or delivery events.",
            ),
        ]
        if self.operational_evidence_available:
            definitions.extend(
                [
                    ToolDefinition(
                        "get_pipeline_logs",
                        "Execution log events, including component writes and timestamps.",
                        {},
                        ("What did pipeline components execute or persist?",),
                        "Bounded to the current execution; orchestration state is separate.",
                    ),
                    ToolDefinition(
                        "get_supplier_submissions",
                        "Supplier manifests, batch IDs, checksums, arrival times, and field "
                        "statistics.",
                        {},
                        ("What payloads arrived, when, and with which source-level properties?",),
                        "Does not show internal task retries, writes, or transformations.",
                    ),
                    ToolDefinition(
                        "get_orchestration_events",
                        "Task attempts, failures, retry triggers, queue timing, and "
                        "scheduler state.",
                        {},
                        ("Did orchestration retry, fail, or queue the execution?",),
                        "Does not prove which rows were written or transformed.",
                    ),
                    ToolDefinition(
                        "get_transformation_metadata",
                        "Transformation lineage, row counts, join cardinality, and "
                        "conversion outcomes.",
                        {},
                        ("Did transformation logic multiply, lose, or alter source values?",),
                        "Does not show supplier delivery or scheduler behavior.",
                    ),
                    ToolDefinition(
                        "get_configuration_changes",
                        "Approved, deployed, and runtime-loaded configuration versions.",
                        {},
                        ("Was a relevant configuration change approved and loaded?",),
                        "Does not describe source payload contents or task execution.",
                    ),
                ]
            )
        return definitions

    @property
    def investigation_objective(self) -> str:
        if self.operational_evidence_available:
            return "identify_underlying_root_cause"
        return "classify_observable_failure"

    @property
    def allowed_diagnosis_codes(self) -> list[str]:
        codes = (
            ROOT_CAUSE_DIAGNOSIS_CODES
            if self.operational_evidence_available
            else QUALITY_DIAGNOSIS_CODES
        )
        return sorted(
            codes,
            key=lambda code: sha256(f"{self.incident_id}:{code}".encode()).digest(),
        )

    @property
    def available_check_ids(self) -> list[str]:
        return sorted(
            result["check_id"]
            for result in self.repository.get_validation_results(self.execution_id)
            if not result["passed"]
        )

    @property
    def non_repeatable_tools(self) -> set[str]:
        """Tools whose incident-scoped result cannot change during an investigation."""
        return {
            "inspect_dataset",
            "get_validation_failures",
            "compare_schemas",
            "compare_distributions",
            "get_pipeline_history",
            "get_data_contract",
            "get_pipeline_logs",
            "get_supplier_submissions",
            "get_orchestration_events",
            "get_transformation_metadata",
            "get_configuration_changes",
        }

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> Evidence:
        evidence_id = f"EV-{uuid4().hex[:12].upper()}"
        arguments = arguments or {}
        validation_error = self.validate_request(name, arguments)
        if validation_error:
            return Evidence(
                evidence_id=evidence_id,
                tool_name=name,
                summary="Investigation tool request was rejected",
                data={},
                success=False,
                error=validation_error,
                result_kind=ToolResultKind.INVALID,
            )
        try:
            summary, data = self._tools[name](arguments)
        except Exception as exc:  # Storage and decoding failures must become observable evidence.
            return Evidence(
                evidence_id=evidence_id,
                tool_name=name,
                summary="Investigation tool call failed",
                data={},
                success=False,
                error=str(exc),
                result_kind=ToolResultKind.INVALID,
            )
        return Evidence(
            evidence_id,
            name,
            summary,
            data,
            result_kind=_classify_result(name, data),
        )

    def validate_request(self, name: str, arguments: dict[str, Any]) -> str | None:
        definitions = {item.name: item for item in self.definitions}
        definition = definitions.get(name)
        if definition is None:
            return f"unknown investigation tool (unauthorised): {name}"
        unexpected = set(arguments) - set(definition.parameters)
        if unexpected:
            return f"unsupported arguments for {name}: {sorted(unexpected)}"
        for argument_name, value in arguments.items():
            specification = definition.parameters[argument_name]
            if specification.get("type") == "integer":
                if isinstance(value, bool) or not isinstance(value, int):
                    return f"{argument_name} must be an integer"
                if value < specification.get("minimum", value):
                    return f"{argument_name} is below the allowed minimum"
                if value > specification.get("maximum", value):
                    return f"{argument_name} exceeds the allowed maximum"
            if specification.get("type") == "string" and not isinstance(value, str):
                return f"{argument_name} must be a string"
            allowed_values = specification.get("enum")
            if allowed_values is not None and value not in allowed_values:
                return (
                    f"{argument_name} must be one of the available values: "
                    f"{allowed_values or ['none']}"
                )
        return None

    def _inspect_dataset(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        profile = self.repository.get_profile(self.execution_id)
        if profile is None:
            raise ValueError("dataset profile is unavailable")
        data = {
            "execution_id": self.execution_id,
            "schema": self.repository.get_schema(self.execution_id),
            **profile,
        }
        return "Retrieved current schema, distributions, and aggregate metrics", data

    def _get_validation_failures(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        failures = [
            result
            for result in self.repository.get_validation_results(self.execution_id)
            if not result["passed"]
        ]
        return f"Retrieved {len(failures)} validation failure(s)", {"failures": failures}

    def _compare_schemas(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        previous = self._previous_execution()
        current_schema = self.repository.get_schema(self.execution_id)
        if previous is None:
            return "No prior execution is available for schema comparison", {
                "sufficient_history": False,
                "current_schema": current_schema,
                "reason": "no_prior_execution",
            }
        previous_schema = self.repository.get_schema(str(previous["execution_id"]))
        current_by_name = {item["column_name"]: item for item in current_schema}
        previous_by_name = {item["column_name"]: item for item in previous_schema}
        shared = current_by_name.keys() & previous_by_name.keys()
        data = {
            "sufficient_history": True,
            "comparison_execution_id": previous["execution_id"],
            "added_columns": sorted(current_by_name.keys() - previous_by_name.keys()),
            "removed_columns": sorted(previous_by_name.keys() - current_by_name.keys()),
            "type_changes": [
                {
                    "column": name,
                    "previous": previous_by_name[name]["data_type"],
                    "current": current_by_name[name]["data_type"],
                }
                for name in sorted(shared)
                if previous_by_name[name]["data_type"] != current_by_name[name]["data_type"]
            ],
        }
        return "Compared current and previous dataset schemas", data

    def _compare_distributions(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        previous = self._previous_execution()
        current = self.repository.get_profile(self.execution_id)
        if current is None:
            raise ValueError("current dataset profile is unavailable")
        if previous is None:
            return "No prior execution is available for distribution comparison", {
                "sufficient_history": False,
                "reason": "no_prior_execution",
            }
        prior = self.repository.get_profile(str(previous["execution_id"]))
        if prior is None:
            return "Prior execution has no statistical profile", {
                "sufficient_history": False,
                "reason": "prior_profile_unavailable",
            }
        data = {
            "sufficient_history": True,
            "comparison_execution_id": previous["execution_id"],
            "row_count_delta": (current["profile"]["row_count"] - prior["profile"]["row_count"]),
            "numeric_changes": _numeric_changes(current["profile"], prior["profile"]),
            "aggregate_changes": _aggregate_changes(
                current["aggregate_metrics"], prior["aggregate_metrics"]
            ),
        }
        return "Compared current and previous statistical profiles", data

    def _sample_problematic_records(self, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        check_id = _optional_string(arguments.get("check_id"))
        limit = _bounded_limit(arguments.get("limit", 10), maximum=20)
        samples = self.repository.get_problem_samples(self.execution_id, check_id, limit=limit)
        return f"Retrieved {len(samples)} sampled problematic record(s)", {
            "check_id": check_id,
            "samples": samples,
            "sample_is_bounded": True,
        }

    def _get_pipeline_history(self, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        limit = _bounded_limit(arguments.get("limit", 10), maximum=20)
        candidates = self.repository.list_executions(
            before_ingested_at=str(self.execution["ingestion_timestamp"]), limit=100
        )
        history = [
            _execution_summary(item)
            for item in candidates
            if item["status"] == "healthy" and item["source"] == self.execution["source"]
        ][:limit]
        return f"Retrieved {len(history)} prior pipeline execution(s)", {
            "executions": history,
            "sufficient_history": bool(history),
        }

    def _identify_affected_dimensions(
        self, arguments: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        check_id = _optional_string(arguments.get("check_id"))
        samples = self.repository.get_problem_samples(self.execution_id, check_id, limit=50)
        suppliers: Counter[str] = Counter()
        categories: Counter[str] = Counter()
        for sample in samples:
            record = sample["record"]
            if record.get("supplier_id") is not None:
                suppliers[str(record["supplier_id"])] += 1
            if record.get("product_category") is not None:
                categories[str(record["product_category"])] += 1
        return "Identified dimensions represented in bounded problem samples", {
            "check_id": check_id,
            "sample_count": len(samples),
            "suppliers": dict(suppliers),
            "categories": dict(categories),
            "scope": "sampled_problem_records",
        }

    def _get_data_contract(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return "Retrieved the configured supplier and dataset contracts", {
            "required_columns": list(self.config.required_columns),
            "supplier_currencies": self.config.supplier_currencies,
            "allowed_categories": sorted(self.config.category_profiles),
            "maximum_delivery_delay_hours": self.config.maximum_delivery_delay_hours,
            "unit_price_z_score_threshold": self.config.anomaly_z_score_threshold,
        }

    def _get_pipeline_logs(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return self._operational_records(OperationalRecordType.PIPELINE_LOG, "pipeline log")

    def _get_supplier_submissions(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return self._operational_records(
            OperationalRecordType.SUPPLIER_SUBMISSION, "supplier submission"
        )

    def _get_orchestration_events(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return self._operational_records(
            OperationalRecordType.ORCHESTRATION_EVENT, "orchestration event"
        )

    def _get_transformation_metadata(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return self._operational_records(
            OperationalRecordType.TRANSFORMATION_METADATA, "transformation record"
        )

    def _get_configuration_changes(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return self._operational_records(
            OperationalRecordType.CONFIGURATION_CHANGE, "configuration change"
        )

    def _operational_records(
        self, record_type: OperationalRecordType, label: str
    ) -> tuple[str, dict[str, Any]]:
        records = self.repository.get_operational_records(
            self.execution_id, record_type.value, limit=50
        )
        return f"Retrieved {len(records)} {label} record(s)", {
            "records": records,
            "coverage": {
                "complete": True,
                "execution_id": self.execution_id,
                "record_type": record_type.value,
                "record_count": len(records),
            },
        }

    def _previous_execution(self) -> dict[str, Any] | None:
        history = self.repository.list_executions(
            before_ingested_at=str(self.execution["ingestion_timestamp"]), limit=20
        )
        return next(
            (
                item
                for item in history
                if item["status"] == "healthy" and item["source"] == self.execution["source"]
            ),
            None,
        )


def _numeric_changes(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    current_columns = current.get("columns", {})
    previous_columns = previous.get("columns", {})
    for column in current_columns.keys() & previous_columns.keys():
        current_numeric = current_columns[column].get("numeric")
        previous_numeric = previous_columns[column].get("numeric")
        if not current_numeric or not previous_numeric:
            continue
        changes[column] = {
            key: {
                "previous": previous_numeric[key],
                "current": current_numeric[key],
                "delta": current_numeric[key] - previous_numeric[key],
            }
            for key in ("mean", "std", "min", "median", "max")
        }
    return changes


def _aggregate_changes(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    current_dimensions = current.get("dimensions", {})
    previous_dimensions = previous.get("dimensions", {})
    for dimension in current_dimensions.keys() | previous_dimensions.keys():
        current_groups = current_dimensions.get(dimension, {})
        previous_groups = previous_dimensions.get(dimension, {})
        dimension_changes: dict[str, Any] = {}
        for group in current_groups.keys() | previous_groups.keys():
            current_count = current_groups.get(group, {}).get("record_count", 0)
            previous_count = previous_groups.get(group, {}).get("record_count", 0)
            dimension_changes[group] = {
                "previous_record_count": previous_count,
                "current_record_count": current_count,
                "record_count_delta": current_count - previous_count,
            }
        changes[dimension] = dimension_changes
    return changes


def _bounded_limit(value: Any, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("limit must be an integer")
    return max(1, min(value, maximum))


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("check_id must be a string")
    return value


def _classify_result(name: str, data: dict[str, Any]) -> ToolResultKind:
    if name == "get_validation_failures" and not data.get("failures"):
        return ToolResultKind.EMPTY
    if name == "sample_problematic_records" and not data.get("samples"):
        return ToolResultKind.EMPTY
    if name == "identify_affected_dimensions" and data.get("sample_count", 0) == 0:
        return ToolResultKind.EMPTY
    if name == "get_pipeline_history" and not data.get("executions"):
        return ToolResultKind.EMPTY
    if name in {"compare_schemas", "compare_distributions"} and not data.get(
        "sufficient_history", False
    ):
        return ToolResultKind.EMPTY
    if name in {
        "get_pipeline_logs",
        "get_supplier_submissions",
        "get_orchestration_events",
        "get_transformation_metadata",
        "get_configuration_changes",
    } and not data.get("records"):
        return ToolResultKind.EMPTY
    return ToolResultKind.INFORMATIVE


def _execution_summary(execution: dict[str, Any]) -> dict[str, Any]:
    return {
        key: execution[key]
        for key in (
            "execution_id",
            "incident_id",
            "source",
            "status",
            "pipeline_stage",
            "ingestion_timestamp",
            "event_time_min",
            "event_time_max",
            "row_count",
        )
    }
