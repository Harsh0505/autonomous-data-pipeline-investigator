from __future__ import annotations

import pytest

from retail_guardian.hypothesis_state import (
    HypothesisTransitionError,
    apply_hypothesis_operations,
    validate_supported_conclusion,
)
from retail_guardian.investigation_models import (
    Evidence,
    HypothesisOperation,
    HypothesisOperationType,
    HypothesisStatus,
)


def test_hypothesis_operations_preserve_identity_and_accumulate_evidence() -> None:
    first = Evidence("EV-1", "inspect_dataset", "Observed a defect", {})
    second = Evidence("EV-2", "get_data_contract", "Retrieved the contract", {})
    hypotheses = apply_hypothesis_operations(
        [],
        [
            HypothesisOperation(
                HypothesisOperationType.CREATE,
                "H-1",
                "A source field may violate its contract",
            )
        ],
        [],
    )
    hypotheses = apply_hypothesis_operations(
        hypotheses,
        [
            HypothesisOperation(
                HypothesisOperationType.UPDATE,
                "H-1",
                diagnosis_code="schema_change",
                supporting_evidence=["EV-1"],
            )
        ],
        [first],
    )
    hypotheses = apply_hypothesis_operations(
        hypotheses,
        [
            HypothesisOperation(
                HypothesisOperationType.CONFIRM,
                "H-1",
                supporting_evidence=["EV-2"],
            )
        ],
        [first, second],
    )

    hypothesis = hypotheses[0]
    assert hypothesis.hypothesis_id == "H-1"
    assert hypothesis.status == HypothesisStatus.CONFIRMED
    assert hypothesis.supporting_evidence == ["EV-1", "EV-2"]
    validate_supported_conclusion("schema_change", ["EV-1", "EV-2"], hypotheses)


def test_hypothesis_operations_reject_fabricated_and_contradictory_transitions() -> None:
    hypotheses = apply_hypothesis_operations(
        [],
        [HypothesisOperation(HypothesisOperationType.CREATE, "H-1", "A tentative cause")],
        [],
    )

    with pytest.raises(HypothesisTransitionError, match="unavailable evidence"):
        apply_hypothesis_operations(
            hypotheses,
            [
                HypothesisOperation(
                    HypothesisOperationType.CONFIRM,
                    "H-1",
                    diagnosis_code="missing_values",
                    supporting_evidence=["EV-FABRICATED"],
                )
            ],
            [],
        )

    evidence = Evidence("EV-1", "inspect_dataset", "Observed a value", {})
    with pytest.raises(HypothesisTransitionError, match="both support and contradict"):
        apply_hypothesis_operations(
            hypotheses,
            [
                HypothesisOperation(
                    HypothesisOperationType.UPDATE,
                    "H-1",
                    supporting_evidence=["EV-1"],
                    contradicting_evidence=["EV-1"],
                )
            ],
            [evidence],
        )


def test_rejected_hypothesis_cannot_be_confirmed() -> None:
    evidence = Evidence("EV-1", "get_data_contract", "Contract contradicts hypothesis", {})
    hypotheses = apply_hypothesis_operations(
        [],
        [HypothesisOperation(HypothesisOperationType.CREATE, "H-1", "A tentative cause")],
        [],
    )
    hypotheses = apply_hypothesis_operations(
        hypotheses,
        [
            HypothesisOperation(
                HypothesisOperationType.REJECT,
                "H-1",
                contradicting_evidence=["EV-1"],
            )
        ],
        [evidence],
    )

    with pytest.raises(HypothesisTransitionError, match="already rejected"):
        apply_hypothesis_operations(
            hypotheses,
            [
                HypothesisOperation(
                    HypothesisOperationType.CONFIRM,
                    "H-1",
                    diagnosis_code="schema_change",
                    supporting_evidence=["EV-1"],
                )
            ],
            [evidence],
        )
