from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any

import pandas as pd

from retail_guardian.config import CategoryProfile, PipelineConfig
from retail_guardian.repair_models import (
    ActivateCategoryParameters,
    ChangeSummary,
    CorrectCurrencyParameters,
    DeduplicateParameters,
    RepairCapability,
    RepairProposal,
    RepairType,
)


class RepairPreconditionError(ValueError):
    """A proposed repair cannot safely operate on the selected data."""


@dataclass(frozen=True)
class OperationResult:
    frame: pd.DataFrame
    config: PipelineConfig
    summary: ChangeSummary


class RepairOperationRegistry:
    """Safe, deterministic repair operations available to the planning agent."""

    def capabilities(
        self,
        frame: pd.DataFrame,
        config: PipelineConfig,
        operational_records: list[dict[str, Any]],
    ) -> list[RepairCapability]:
        capabilities: list[RepairCapability] = []
        duplicate_ids = _duplicate_identifiers(frame)
        if duplicate_ids:
            capabilities.append(
                RepairCapability(
                    repair_type=RepairType.DEDUPLICATE_RECORDS,
                    target_component="dataset",
                    description=(
                        "Remove repeated rows only for explicitly selected duplicate order IDs."
                    ),
                    limitations="Does not correct the upstream replay or retry mechanism.",
                    resolves_checks=["unique_order_id"],
                    compatible_diagnoses=[
                        "duplicate_records",
                        "supplier_batch_replay",
                        "failed_job_retry",
                        "incorrect_join_cardinality",
                    ],
                    candidate_parameters=[
                        DeduplicateParameters(identifiers=duplicate_ids).model_dump()
                    ],
                    parameter_schema=DeduplicateParameters.model_json_schema(),
                    maximum_expected_rows_removed=_duplicate_excess_count(frame),
                )
            )

        for supplier_id, expected_currency in config.supplier_currencies.items():
            if not {"supplier_id", "currency"}.issubset(frame.columns):
                break
            mismatch = frame["supplier_id"].eq(supplier_id) & frame["currency"].ne(
                expected_currency
            )
            if mismatch.fillna(False).any():
                capabilities.append(
                    RepairCapability(
                        repair_type=RepairType.CORRECT_SUPPLIER_CURRENCY,
                        target_component="dataset",
                        description=(
                            "Replace mismatched currency codes for one supplier with its "
                            "configured contract currency."
                        ),
                        limitations=(
                            "Does not infer or reverse price scaling introduced outside the "
                            "code field."
                        ),
                        resolves_checks=["supplier_currency"],
                        compatible_diagnoses=["currency_mismatch"],
                        candidate_parameters=[
                            CorrectCurrencyParameters(
                                supplier_id=supplier_id,
                                expected_currency=expected_currency,
                            ).model_dump()
                        ],
                        parameter_schema=CorrectCurrencyParameters.model_json_schema(),
                        expected_changed_columns=["currency"],
                    )
                )

        for change in _approved_category_changes(operational_records):
            categories = change["details"].get("added_categories", [])
            for category in categories:
                if (
                    isinstance(category, str)
                    and category not in config.category_profiles
                    and "product_category" in frame
                    and frame["product_category"].eq(category).any()
                ):
                    capabilities.append(
                        RepairCapability(
                            repair_type=RepairType.ACTIVATE_APPROVED_CATEGORY,
                            target_component="pipeline_configuration",
                            description=(
                                "Activate a category present in an approved taxonomy change."
                            ),
                            limitations=(
                                "Changes only the isolated validation configuration and requires "
                                "an approved operational record."
                            ),
                            resolves_checks=["known_category"],
                            compatible_diagnoses=["legitimate_taxonomy_update"],
                            candidate_parameters=[
                                ActivateCategoryParameters(
                                    category=category,
                                    approved_change_id=str(change["details"]["change_id"]),
                                ).model_dump()
                            ],
                            parameter_schema=ActivateCategoryParameters.model_json_schema(),
                        )
                    )
        return capabilities

    def execute(
        self,
        proposal: RepairProposal,
        frame: pd.DataFrame,
        config: PipelineConfig,
        operational_records: list[dict[str, Any]],
    ) -> OperationResult:
        working = frame.copy(deep=True)
        if proposal.repair_type == RepairType.DEDUPLICATE_RECORDS:
            return self._deduplicate(proposal.parameters, working, config)
        if proposal.repair_type == RepairType.CORRECT_SUPPLIER_CURRENCY:
            return self._correct_currency(proposal.parameters, working, config)
        if proposal.repair_type == RepairType.ACTIVATE_APPROVED_CATEGORY:
            return self._activate_category(
                proposal.parameters,
                working,
                config,
                operational_records,
            )
        raise RepairPreconditionError(f"unsupported repair operation: {proposal.repair_type}")

    def _deduplicate(
        self,
        parameters: object,
        frame: pd.DataFrame,
        config: PipelineConfig,
    ) -> OperationResult:
        if not isinstance(parameters, DeduplicateParameters):
            raise RepairPreconditionError("deduplication parameters are invalid")
        column = parameters.identifier_column
        if column not in frame:
            raise RepairPreconditionError(f"identifier column is missing: {column}")
        requested = set(parameters.identifiers)
        actual = set(_duplicate_identifiers(frame))
        unavailable = sorted(requested - actual)
        if unavailable:
            raise RepairPreconditionError(
                f"selected identifiers are not currently duplicated: {unavailable}"
            )
        selected = frame[column].astype(str).isin(requested)
        remove = selected & frame.duplicated(column, keep=parameters.keep)
        if not remove.any():
            raise RepairPreconditionError("selected identifiers contain no removable duplicates")
        repaired = frame.loc[~remove].copy().reset_index(drop=True)
        identifiers = sorted(frame.loc[remove, column].astype(str).unique())
        return OperationResult(
            repaired,
            config,
            ChangeSummary(
                repair_type=RepairType.DEDUPLICATE_RECORDS,
                rows_before=len(frame),
                rows_after=len(repaired),
                rows_changed=int(remove.sum()),
                rows_removed=int(remove.sum()),
                affected_identifiers=identifiers,
                changed_columns=[],
            ),
        )

    def _correct_currency(
        self,
        parameters: object,
        frame: pd.DataFrame,
        config: PipelineConfig,
    ) -> OperationResult:
        if not isinstance(parameters, CorrectCurrencyParameters):
            raise RepairPreconditionError("currency correction parameters are invalid")
        required = {"order_id", "supplier_id", "currency"}
        if not required.issubset(frame.columns):
            raise RepairPreconditionError("currency correction requires order and supplier fields")
        configured = config.supplier_currencies.get(parameters.supplier_id)
        if configured != parameters.expected_currency:
            raise RepairPreconditionError(
                "expected currency does not match the configured supplier contract"
            )
        selected = frame["supplier_id"].eq(parameters.supplier_id) & frame["currency"].ne(
            parameters.expected_currency
        )
        selected = selected.fillna(False)
        if not selected.any():
            raise RepairPreconditionError("no mismatched rows exist for the selected supplier")
        identifiers = sorted(frame.loc[selected, "order_id"].astype(str).tolist())
        frame.loc[selected, "currency"] = parameters.expected_currency
        return OperationResult(
            frame,
            config,
            ChangeSummary(
                repair_type=RepairType.CORRECT_SUPPLIER_CURRENCY,
                rows_before=len(frame),
                rows_after=len(frame),
                rows_changed=int(selected.sum()),
                rows_removed=0,
                affected_identifiers=identifiers,
                changed_columns=["currency"],
            ),
        )

    def _activate_category(
        self,
        parameters: object,
        frame: pd.DataFrame,
        config: PipelineConfig,
        operational_records: list[dict[str, Any]],
    ) -> OperationResult:
        if not isinstance(parameters, ActivateCategoryParameters):
            raise RepairPreconditionError("category activation parameters are invalid")
        approved_record = next(
            (
                record
                for record in operational_records
                if record.get("details", {}).get("change_id") == parameters.approved_change_id
                and record.get("details", {}).get("status") == "approved"
                and parameters.category in record.get("details", {}).get("added_categories", [])
            ),
            None,
        )
        if approved_record is None:
            raise RepairPreconditionError("the category is not backed by an approved change record")
        if parameters.category in config.category_profiles:
            raise RepairPreconditionError("the category is already active")
        if "product_category" not in frame or "unit_price" not in frame:
            raise RepairPreconditionError("category activation requires category and price fields")
        profile = (
            approved_record.get("details", {}).get("category_profiles", {}).get(parameters.category)
        )
        if not isinstance(profile, dict):
            raise RepairPreconditionError("the approved change has no category price profile")
        mean_price = profile.get("mean_price")
        price_std = profile.get("price_std")
        if not isinstance(mean_price, (int, float)) or not isinstance(price_std, (int, float)):
            raise RepairPreconditionError("the approved category price profile is invalid")
        if mean_price <= 0 or price_std <= 0:
            raise RepairPreconditionError("approved category price values must be positive")
        profiles = dict(config.category_profiles)
        profiles[parameters.category] = CategoryProfile(mean_price, price_std)
        repaired_config = replace(config, category_profiles=profiles)
        return OperationResult(
            frame,
            repaired_config,
            ChangeSummary(
                repair_type=RepairType.ACTIVATE_APPROVED_CATEGORY,
                rows_before=len(frame),
                rows_after=len(frame),
                rows_changed=0,
                rows_removed=0,
                affected_identifiers=[],
                changed_columns=[],
                configuration_changes={
                    "approved_change_id": parameters.approved_change_id,
                    "activated_category": parameters.category,
                },
            ),
        )


def frame_hash(frame: pd.DataFrame) -> str:
    canonical = frame.copy()
    canonical.columns = canonical.columns.astype(str)
    values = pd.util.hash_pandas_object(canonical.astype(str), index=True).values.tobytes()
    return sha256("|".join(canonical.columns).encode() + values).hexdigest()


def _duplicate_identifiers(frame: pd.DataFrame) -> list[str]:
    if "order_id" not in frame:
        return []
    mask = frame.duplicated("order_id", keep=False)
    return sorted(frame.loc[mask, "order_id"].astype(str).unique().tolist())


def _duplicate_excess_count(frame: pd.DataFrame) -> int:
    if "order_id" not in frame:
        return 0
    return int(frame.duplicated("order_id", keep="first").sum())


def _approved_category_changes(
    operational_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        record
        for record in operational_records
        if record.get("record_type") == "configuration_change"
        and record.get("details", {}).get("status") == "approved"
        and record.get("details", {}).get("change_id")
    ]
