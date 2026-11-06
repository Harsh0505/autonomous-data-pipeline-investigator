from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from retail_guardian.investigation_models import (
    ROOT_CAUSE_DIAGNOSIS_CODES,
    Evidence,
    ToolResultKind,
)


class CausalEvidenceError(ValueError):
    """A selected causal diagnosis is not established by its cited evidence."""


@dataclass(frozen=True)
class CausalEvidenceAssessment:
    supported: bool
    supporting_evidence: tuple[str, ...]
    contradicting_evidence: tuple[str, ...]
    relevant_evidence: tuple[str, ...]
    unavailable_evidence: tuple[str, ...]
    missing_requirements: tuple[str, ...]


def assess_causal_evidence(
    diagnosis_code: str,
    evidence_references: list[str],
    evidence: list[Evidence],
) -> CausalEvidenceAssessment:
    """Assess a model-selected cause without selecting a diagnosis for the model."""
    selected = [item for item in evidence if item.evidence_id in evidence_references]
    unavailable = {
        item.evidence_id
        for item in selected
        if item.result_kind != ToolResultKind.INFORMATIVE or _records_unavailable(item)
    }
    usable = [item for item in selected if item.evidence_id not in unavailable]
    by_tool = {item.tool_name: item for item in usable}

    if diagnosis_code == "supplier_batch_replay":
        result = _supplier_batch_replay(by_tool)
    elif diagnosis_code == "failed_job_retry":
        result = _failed_job_retry(by_tool)
    elif diagnosis_code == "incorrect_join_cardinality":
        result = _incorrect_join_cardinality(by_tool)
    elif diagnosis_code == "source_system_omission":
        result = _source_system_omission(by_tool)
    elif diagnosis_code == "transformation_logic_error":
        result = _transformation_logic_error(by_tool)
    elif diagnosis_code == "supplier_data_error":
        result = _supplier_data_error(by_tool)
    elif diagnosis_code == "legitimate_taxonomy_update":
        result = _legitimate_taxonomy_update(by_tool)
    elif diagnosis_code == "late_supplier_delivery":
        result = _late_supplier_delivery(by_tool)
    elif diagnosis_code == "scheduler_backlog":
        result = _scheduler_backlog(by_tool)
    else:
        return CausalEvidenceAssessment(False, (), (), (), tuple(sorted(unavailable)), ())

    supported, supporting, contradicting, relevant, missing = result
    return CausalEvidenceAssessment(
        supported,
        tuple(sorted(supporting)),
        tuple(sorted(contradicting)),
        tuple(sorted(relevant - supporting - contradicting)),
        tuple(sorted(unavailable)),
        tuple(missing),
    )


def require_causal_support(
    diagnosis_code: str,
    evidence_references: list[str],
    evidence: list[Evidence],
) -> None:
    if diagnosis_code not in ROOT_CAUSE_DIAGNOSIS_CODES:
        return
    assessment = assess_causal_evidence(diagnosis_code, evidence_references, evidence)
    if assessment.supported:
        return
    details = "; ".join(assessment.missing_requirements) or "cited facts do not establish the claim"
    raise CausalEvidenceError(
        f"causal evidence is insufficient for {diagnosis_code!r}: {details}. "
        f"Contradicting evidence IDs: {list(assessment.contradicting_evidence) or ['none']}; "
        f"relevant but nondiscriminating IDs: "
        f"{list(assessment.relevant_evidence) or ['none']}; unavailable IDs: "
        f"{list(assessment.unavailable_evidence) or ['none']}"
    )


def require_causal_evidence_alignment(
    diagnosis_code: str,
    supporting_references: list[str],
    contradicting_references: list[str],
    evidence: list[Evidence],
    *,
    require_contradiction: bool = False,
) -> None:
    """Reject evidence assignments that conflict with the cited operational facts."""
    if diagnosis_code not in ROOT_CAUSE_DIAGNOSIS_CODES:
        return
    references = list(dict.fromkeys([*supporting_references, *contradicting_references]))
    assessment = assess_causal_evidence(diagnosis_code, references, evidence)
    factual_contradictions = set(assessment.contradicting_evidence)
    misclassified_support = sorted(set(supporting_references) & factual_contradictions)
    if misclassified_support:
        raise CausalEvidenceError(
            f"evidence IDs {misclassified_support} contradict {diagnosis_code!r} and cannot be "
            "recorded as supporting evidence"
        )
    if require_contradiction and not (set(contradicting_references) & factual_contradictions):
        raise CausalEvidenceError(
            f"rejecting {diagnosis_code!r} requires cited facts that contradict it; "
            f"the supplied contradiction IDs are relevant or nondiscriminating: "
            f"{contradicting_references or ['none']}"
        )


def has_inconclusive_audit_evidence(evidence: list[Evidence]) -> bool:
    """Recognise that decisive retry and lineage records were explicitly unavailable."""
    tools = {item.tool_name for item in evidence}
    required = {
        "get_supplier_submissions",
        "get_orchestration_events",
        "get_transformation_metadata",
    }
    if not required.issubset(tools):
        return False
    unavailable_tools = {item.tool_name for item in evidence if evidence_is_unavailable(item)}
    return {
        "get_orchestration_events",
        "get_transformation_metadata",
    }.issubset(unavailable_tools)


def evidence_is_unavailable(evidence: Evidence) -> bool:
    """Return whether a tool established that its underlying records are unavailable."""
    return evidence.result_kind == ToolResultKind.EMPTY or _records_unavailable(evidence)


def _supplier_batch_replay(by_tool: dict[str, Evidence]):
    supplier = by_tool.get("get_supplier_submissions")
    records = _records(supplier)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        details = record.get("details", {})
        if details.get("status") != "accepted":
            continue
        key = (
            str(details.get("supplier_id", "")),
            str(details.get("batch_id", "")),
            str(details.get("checksum", "")),
        )
        if all(key):
            groups.setdefault(key, []).append(record)
    repeated = any(
        len(items) >= 2 and len({item.get("occurred_at") for item in items}) >= 2
        for items in groups.values()
    )
    supporting = {supplier.evidence_id} if supplier and repeated else set()
    contradicting = (
        {supplier.evidence_id}
        if supplier and _coverage_complete(supplier) and records and not repeated
        else set()
    )
    missing = (
        []
        if repeated
        else [
            "cite complete supplier history containing at least two accepted submissions "
            "with matching supplier, batch ID, and checksum at distinct times"
        ]
    )
    return repeated, supporting, contradicting, _ids(supplier), missing


def _failed_job_retry(by_tool: dict[str, Evidence]):
    orchestration = by_tool.get("get_orchestration_events")
    logs = by_tool.get("get_pipeline_logs")
    events = _records(orchestration)
    log_records = _records(logs)
    failed_attempts = {
        int(details["attempt"])
        for details in _details(events)
        if details.get("state") == "failed"
        and details.get("retry_scheduled") is True
        and isinstance(details.get("attempt"), int)
    }
    retry_attempts = {
        int(details["attempt"])
        for details in _details(events)
        if details.get("trigger") == "automatic_retry" and isinstance(details.get("attempt"), int)
    }
    writes = {
        int(details["attempt"])
        for details in _details(log_records)
        if details.get("event") == "rows_committed"
        and int(details.get("committed_rows", 0)) > 0
        and isinstance(details.get("attempt"), int)
    }
    retry_after_failure = any(
        retry > failed for failed in failed_attempts for retry in retry_attempts
    )
    repeated_write = bool(failed_attempts & writes and retry_attempts & writes)
    supported = retry_after_failure and repeated_write
    missing = []
    if not retry_after_failure:
        missing.append("cite a failed attempt followed by an explicitly scheduled retry")
    if not repeated_write:
        missing.append("cite durable writes from both the failed attempt and its retry")
    support = _ids(orchestration, logs) if supported else set()
    return supported, support, set(), _ids(orchestration, logs), missing


def _incorrect_join_cardinality(by_tool: dict[str, Evidence]):
    transformation = by_tool.get("get_transformation_metadata")
    records = _records(transformation)
    expanded = any(
        details.get("operation") == "join"
        and (
            details.get("observed_cardinality") == "many_to_many"
            or details.get("right_key_unique") is False
        )
        and int(details.get("output_rows", 0)) > int(details.get("input_rows", 0))
        for details in _details(records)
    )
    missing = (
        []
        if expanded
        else ["cite transformation metadata showing unintended join multiplicity and row expansion"]
    )
    support = _ids(transformation) if expanded else set()
    return expanded, support, set(), _ids(transformation), missing


def _source_system_omission(by_tool: dict[str, Evidence]):
    supplier = by_tool.get("get_supplier_submissions")
    transformation = by_tool.get("get_transformation_metadata")
    source_nulls = _supplier_null_count(_records(supplier))
    passthrough = any(
        int(details.get("input_null_count", 0)) > 0
        and details.get("output_null_count") == details.get("input_null_count")
        and int(details.get("conversion_errors", 0)) == 0
        for details in _details(_records(transformation))
    )
    supported = source_nulls > 0 and passthrough
    missing = []
    if source_nulls <= 0:
        missing.append("cite source submission statistics showing required values already missing")
    if not passthrough:
        missing.append(
            "cite transformation metadata showing nulls passed through without conversion loss"
        )
    contradicting = {supplier.evidence_id} if supplier and source_nulls == 0 else set()
    support = _ids(supplier, transformation) if supported else set()
    return supported, support, contradicting, _ids(supplier, transformation), missing


def _transformation_logic_error(by_tool: dict[str, Evidence]):
    supplier = by_tool.get("get_supplier_submissions")
    transformation = by_tool.get("get_transformation_metadata")
    source_nulls = _supplier_null_count(_records(supplier))
    loss = any(
        int(details.get("input_null_count", 0)) == 0
        and int(details.get("output_null_count", 0)) > 0
        and int(details.get("conversion_errors", 0)) > 0
        for details in _details(_records(transformation))
    )
    supported = supplier is not None and source_nulls == 0 and loss
    missing = []
    if supplier is None or source_nulls != 0:
        missing.append("cite source statistics showing the required values were present")
    if not loss:
        missing.append(
            "cite transformation metadata showing conversion failures introduced missing values"
        )
    contradicting = {supplier.evidence_id} if supplier and source_nulls > 0 else set()
    support = _ids(supplier, transformation) if supported else set()
    return supported, support, contradicting, _ids(supplier, transformation), missing


def _supplier_data_error(by_tool: dict[str, Evidence]):
    supplier = by_tool.get("get_supplier_submissions")
    configuration = by_tool.get("get_configuration_changes")
    supplier_records = _records(supplier)
    categories = _reported_categories(supplier_records)
    warning = any(
        details.get("warning") == "category_not_in_registered_manifest"
        for details in _details(supplier_records)
    )
    approved = _approved_categories(_records(configuration))
    no_approval = (
        configuration is not None
        and _coverage_complete(configuration)
        and not (categories & approved)
    )
    supported = bool(categories) and warning and no_approval
    missing = []
    if not categories or not warning:
        missing.append("cite a supplier manifest reporting the unexpected category as unregistered")
    if not no_approval:
        missing.append("cite complete configuration history showing no approval for that category")
    contradicting = (
        {configuration.evidence_id} if configuration and bool(categories & approved) else set()
    )
    support = _ids(supplier, configuration) if supported else set()
    return supported, support, contradicting, _ids(supplier, configuration), missing


def _legitimate_taxonomy_update(by_tool: dict[str, Evidence]):
    supplier = by_tool.get("get_supplier_submissions")
    configuration = by_tool.get("get_configuration_changes")
    supplier_records = _records(supplier)
    configuration_records = _records(configuration)
    categories = _reported_categories(supplier_records)
    approved = _approved_categories(configuration_records)
    declared_versions = {
        str(details["declared_taxonomy_version"])
        for details in _details(supplier_records)
        if details.get("declared_taxonomy_version")
    }
    stale_runtime = any(
        details.get("event") == "configuration_loaded"
        and details.get("loaded_version") != details.get("latest_approved_version")
        and details.get("latest_approved_version") in declared_versions
        for details in _details(configuration_records)
    )
    supported = bool(categories & approved) and stale_runtime
    missing = []
    if not categories & approved:
        missing.append("cite matching supplier taxonomy and approved category-change records")
    if not stale_runtime:
        missing.append("cite runtime metadata showing an older taxonomy version was loaded")
    support = _ids(supplier, configuration) if supported else set()
    return supported, support, set(), _ids(supplier, configuration), missing


def _late_supplier_delivery(by_tool: dict[str, Evidence]):
    supplier = by_tool.get("get_supplier_submissions")
    orchestration = by_tool.get("get_orchestration_events")
    late = any(_received_after_expected(details) for details in _details(_records(supplier)))
    short_queue = any(
        0 <= int(details.get("queue_wait_minutes", -1)) <= 30
        for details in _details(_records(orchestration))
    )
    supported = late and short_queue
    missing = []
    if not late:
        missing.append("cite supplier timing showing receipt after the agreed delivery deadline")
    if not short_queue:
        missing.append("cite orchestration timing that rules out a material internal queue delay")
    contradicting = {
        supplier.evidence_id
        for details in _details(_records(supplier))
        if supplier and _received_on_time(details)
    }
    support = _ids(supplier, orchestration) if supported else set()
    return supported, support, contradicting, _ids(supplier, orchestration), missing


def _scheduler_backlog(by_tool: dict[str, Evidence]):
    supplier = by_tool.get("get_supplier_submissions")
    orchestration = by_tool.get("get_orchestration_events")
    on_time = any(_received_on_time(details) for details in _details(_records(supplier)))
    backlog = any(
        int(details.get("queue_wait_minutes", 0)) >= 60
        and details.get("scheduler_health") == "backlogged"
        for details in _details(_records(orchestration))
    )
    supported = on_time and backlog
    missing = []
    if not on_time:
        missing.append("cite supplier timing showing the batch arrived by its deadline")
    if not backlog:
        missing.append("cite orchestration evidence of substantial internal queueing")
    contradicting = (
        {orchestration.evidence_id}
        if orchestration
        and any(
            0 <= int(details.get("queue_wait_minutes", -1)) <= 30
            for details in _details(_records(orchestration))
        )
        else set()
    )
    support = _ids(supplier, orchestration) if supported else set()
    return supported, support, contradicting, _ids(supplier, orchestration), missing


def _records(evidence: Evidence | None) -> list[dict[str, Any]]:
    return [] if evidence is None else list(evidence.data.get("records", []))


def _details(records: list[dict[str, Any]]):
    return [record.get("details", {}) for record in records]


def _ids(*evidence: Evidence | None) -> set[str]:
    return {item.evidence_id for item in evidence if item is not None}


def _coverage_complete(evidence: Evidence) -> bool:
    return evidence.data.get("coverage", {}).get("complete") is True


def _records_unavailable(evidence: Evidence) -> bool:
    records = _records(evidence)
    return bool(records) and all(
        record.get("details", {}).get("status") in {"unavailable", "metadata_unavailable"}
        or record.get("details", {}).get("state") in {"unavailable", "metadata_unavailable"}
        for record in records
    )


def _supplier_null_count(records: list[dict[str, Any]]) -> int:
    return sum(
        int(details.get("field_statistics", {}).get("unit_price", {}).get("null_count", 0))
        for details in _details(records)
    )


def _reported_categories(records: list[dict[str, Any]]) -> set[str]:
    return {
        str(category)
        for details in _details(records)
        for category, count in details.get("category_counts", {}).items()
        if int(count) > 0
    }


def _approved_categories(records: list[dict[str, Any]]) -> set[str]:
    return {
        str(category)
        for details in _details(records)
        if details.get("status") == "approved"
        for category in details.get("added_categories", [])
    }


def _received_after_expected(details: dict[str, Any]) -> bool:
    received = _parse_time(details.get("received_at"))
    expected = _parse_time(details.get("expected_by"))
    return bool(received and expected and received > expected)


def _received_on_time(details: dict[str, Any]) -> bool:
    received = _parse_time(details.get("received_at"))
    expected = _parse_time(details.get("expected_by"))
    return bool(received and expected and received <= expected)


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
