from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from retail_guardian.config import PipelineConfig
from retail_guardian.pipeline import RetailSalesPipeline
from retail_guardian.repair_models import (
    ChangeSummary,
    RepairProposal,
    VerificationReport,
)
from retail_guardian.repair_operations import frame_hash


def verify_repair(
    proposal: RepairProposal,
    original: pd.DataFrame,
    repaired: pd.DataFrame,
    repaired_config: PipelineConfig,
    change_summary: ChangeSummary,
    original_validation_results: list[dict[str, Any]],
    *,
    as_of: datetime,
    original_snapshot_hash: str,
) -> VerificationReport:
    """Apply deterministic acceptance criteria to an isolated repair result."""
    repaired_run = RetailSalesPipeline(repaired_config).run(
        repaired,
        source_name="isolated_repair_simulation",
        as_of=as_of,
    )
    original_failed = sorted(
        item["check_id"] for item in original_validation_results if not item["passed"]
    )
    repaired_failed = sorted(
        item.check_id for item in repaired_run.validation_results if not item.passed
    )
    required = sorted(set(proposal.verification_requirements.checks_to_pass))
    resolved = sorted(set(original_failed) - set(repaired_failed))
    remaining = sorted(set(original_failed) & set(repaired_failed))
    new_failures = sorted(set(repaired_failed) - set(original_failed))

    rows_removed = len(original) - len(repaired)
    unexpected_data_loss = rows_removed < 0 or (
        rows_removed > proposal.verification_requirements.maximum_rows_removed
    )
    unaffected_preserved = _unaffected_records_preserved(
        original,
        repaired,
        change_summary.affected_identifiers,
    )
    original_preserved = frame_hash(original) == original_snapshot_hash
    changed_columns_allowed = set(change_summary.changed_columns).issubset(
        proposal.verification_requirements.expected_changed_columns
    )

    reasons: list[str] = []
    unresolved_required = sorted(set(required) & set(repaired_failed))
    if unresolved_required:
        reasons.append(f"required checks still fail: {unresolved_required}")
    if new_failures:
        reasons.append(f"repair introduced new validation failures: {new_failures}")
    if unexpected_data_loss:
        reasons.append(
            "row-count change exceeded the approved repair boundary: "
            f"removed={rows_removed}, allowed="
            f"{proposal.verification_requirements.maximum_rows_removed}"
        )
    if proposal.verification_requirements.preserve_unaffected_records and not unaffected_preserved:
        reasons.append("records outside the selected repair scope changed")
    if not original_preserved:
        reasons.append("the immutable original snapshot was modified")
    if not changed_columns_allowed:
        reasons.append("the repair changed columns outside its verification requirements")
    if repaired_run.report.pipeline_error and not repaired_failed:
        reasons.append(f"the repaired pipeline failed: {repaired_run.report.pipeline_error}")

    passed = not reasons
    return VerificationReport(
        passed=passed,
        checks_required=required,
        original_failed_checks=original_failed,
        repaired_failed_checks=repaired_failed,
        resolved_checks=resolved,
        remaining_checks=remaining,
        new_failures=new_failures,
        original_row_count=len(original),
        repaired_row_count=len(repaired),
        unexpected_data_loss=unexpected_data_loss,
        unaffected_records_preserved=unaffected_preserved,
        original_snapshot_preserved=original_preserved,
        business_aggregate_comparison={
            "before": _business_metrics(original),
            "after": _business_metrics(repaired),
        },
        reasons=reasons,
        completed_at=datetime.now(UTC),
    )


def _unaffected_records_preserved(
    original: pd.DataFrame,
    repaired: pd.DataFrame,
    affected_identifiers: list[str],
) -> bool:
    if "order_id" not in original or "order_id" not in repaired:
        return original.equals(repaired)
    affected = set(affected_identifiers)
    original_unaffected = original.loc[
        ~original["order_id"].astype(str).isin(affected)
    ].reset_index(drop=True)
    repaired_unaffected = repaired.loc[
        ~repaired["order_id"].astype(str).isin(affected)
    ].reset_index(drop=True)
    if list(original_unaffected.columns) != list(repaired_unaffected.columns):
        return False
    return frame_hash(original_unaffected) == frame_hash(repaired_unaffected)


def _business_metrics(frame: pd.DataFrame) -> dict[str, int | float | None]:
    result: dict[str, int | float | None] = {
        "row_count": len(frame),
        "distinct_orders": (
            int(frame["order_id"].nunique(dropna=True)) if "order_id" in frame else None
        ),
    }
    result["quantity_total"] = _numeric_sum(frame, "quantity")
    if {"quantity", "unit_price"}.issubset(frame.columns):
        quantity = pd.to_numeric(frame["quantity"], errors="coerce")
        price = pd.to_numeric(frame["unit_price"], errors="coerce")
        result["gross_revenue_local"] = round(float((quantity * price).sum()), 2)
    else:
        result["gross_revenue_local"] = None
    return result


def _numeric_sum(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    return round(float(pd.to_numeric(frame[column], errors="coerce").sum()), 2)
