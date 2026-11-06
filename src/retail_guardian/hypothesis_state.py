from __future__ import annotations

from retail_guardian.investigation_models import (
    Evidence,
    HypothesisOperation,
    HypothesisOperationType,
    HypothesisStatus,
    RootCauseHypothesis,
    ToolResultKind,
)


class HypothesisTransitionError(ValueError):
    """A requested hypothesis operation is inconsistent with investigation state."""


def apply_hypothesis_operations(
    current: list[RootCauseHypothesis],
    operations: list[HypothesisOperation],
    evidence: list[Evidence],
) -> list[RootCauseHypothesis]:
    """Apply validated operations while preserving stable hypothesis state."""
    hypotheses = {item.hypothesis_id: item for item in current}
    order = [item.hypothesis_id for item in current]
    evidence_ids = {item.evidence_id for item in evidence}
    successful_ids = {
        item.evidence_id for item in evidence if item.result_kind == ToolResultKind.INFORMATIVE
    }

    for operation in operations:
        _validate_references(operation, evidence_ids, successful_ids)
        existing = hypotheses.get(operation.hypothesis_id)
        if operation.operation == HypothesisOperationType.CREATE:
            if existing is not None:
                raise HypothesisTransitionError(
                    f"hypothesis {operation.hypothesis_id!r} already exists; use update"
                )
            if not operation.statement:
                raise HypothesisTransitionError("create operation requires a statement")
            hypotheses[operation.hypothesis_id] = RootCauseHypothesis(
                operation.hypothesis_id,
                operation.statement,
                operation.diagnosis_code,
                supporting_evidence=_unique(operation.supporting_evidence),
                contradicting_evidence=_unique(operation.contradicting_evidence),
            )
            order.append(operation.hypothesis_id)
            continue

        if existing is None:
            raise HypothesisTransitionError(
                f"hypothesis {operation.hypothesis_id!r} does not exist; use create"
            )
        if existing.status != HypothesisStatus.ACTIVE:
            raise HypothesisTransitionError(
                f"hypothesis {operation.hypothesis_id!r} is already {existing.status.value}"
            )

        supporting = _unique([*existing.supporting_evidence, *operation.supporting_evidence])
        contradicting = _unique(
            [*existing.contradicting_evidence, *operation.contradicting_evidence]
        )
        overlap = set(supporting) & set(contradicting)
        if overlap:
            raise HypothesisTransitionError(
                f"evidence IDs cannot both support and contradict a hypothesis: {sorted(overlap)}"
            )
        diagnosis_code = operation.diagnosis_code or existing.diagnosis_code
        statement = operation.statement or existing.statement
        status = existing.status
        if operation.operation == HypothesisOperationType.CONFIRM:
            if diagnosis_code is None:
                raise HypothesisTransitionError("confirm operation requires a diagnosis code")
            if not supporting:
                raise HypothesisTransitionError(
                    "confirm operation requires at least one supporting evidence ID"
                )
            status = HypothesisStatus.CONFIRMED
        elif operation.operation == HypothesisOperationType.REJECT:
            if not contradicting:
                raise HypothesisTransitionError(
                    "reject operation requires at least one contradicting evidence ID"
                )
            status = HypothesisStatus.REJECTED

        hypotheses[operation.hypothesis_id] = RootCauseHypothesis(
            operation.hypothesis_id,
            statement,
            diagnosis_code,
            status,
            supporting,
            contradicting,
        )

    return [hypotheses[hypothesis_id] for hypothesis_id in order]


def validate_supported_conclusion(
    diagnosis_code: str,
    evidence_references: list[str],
    hypotheses: list[RootCauseHypothesis],
) -> None:
    """Require a conclusion to match a confirmed hypothesis and its evidence."""
    references = set(evidence_references)
    matching = [
        item
        for item in hypotheses
        if item.status == HypothesisStatus.CONFIRMED
        and item.diagnosis_code == diagnosis_code
        and references.issubset(item.supporting_evidence)
    ]
    if not matching:
        raise HypothesisTransitionError(
            "a diagnosis must match a confirmed hypothesis whose supporting evidence includes "
            "every conclusion evidence reference"
        )


def _validate_references(
    operation: HypothesisOperation,
    evidence_ids: set[str],
    successful_ids: set[str],
) -> None:
    references = {
        *operation.supporting_evidence,
        *operation.contradicting_evidence,
    }
    unavailable = references - evidence_ids
    if unavailable:
        raise HypothesisTransitionError(
            f"hypothesis {operation.hypothesis_id!r} referenced unavailable evidence IDs: "
            f"{sorted(unavailable)}"
        )
    unsuccessful = references - successful_ids
    if unsuccessful:
        raise HypothesisTransitionError(
            f"hypothesis {operation.hypothesis_id!r} referenced unsuccessful evidence IDs: "
            f"{sorted(unsuccessful)}"
        )


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
