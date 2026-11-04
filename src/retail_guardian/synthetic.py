from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from retail_guardian.config import PipelineConfig


class SyntheticSalesGenerator:
    """Generate deterministic supplier sales that conform to the pipeline contract."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def generate(
        self,
        row_count: int = 500,
        *,
        seed: int = 42,
        as_of: datetime | None = None,
    ) -> pd.DataFrame:
        if row_count < 1:
            raise ValueError("row_count must be at least 1")

        reference_time = as_of or datetime.now(UTC)
        if reference_time.tzinfo is None:
            raise ValueError("as_of must include a timezone")

        rng = np.random.default_rng(seed)
        supplier_ids = list(self.config.supplier_currencies)
        categories = list(self.config.category_profiles)
        suppliers = rng.choice(supplier_ids, size=row_count, p=[0.38, 0.32, 0.30])
        selected_categories = rng.choice(
            categories, size=row_count, p=[0.18, 0.23, 0.25, 0.17, 0.17]
        )
        quantities = rng.choice([1, 2, 3, 4, 5], size=row_count, p=[0.55, 0.25, 0.12, 0.05, 0.03])

        unit_prices = np.array(
            [
                max(
                    1.0,
                    rng.normal(
                        self.config.category_profiles[category].mean_price,
                        self.config.category_profiles[category].price_std,
                    ),
                )
                for category in selected_categories
            ]
        )
        age_minutes = rng.integers(5, 4 * 60, size=row_count)
        timestamps = pd.to_datetime(reference_time, utc=True) - pd.to_timedelta(
            age_minutes, unit="m"
        )
        country_by_supplier = {
            "atlas_us": ["US", "CA", "MX"],
            "europa_de": ["DE", "FR", "NL", "ES"],
            "meridian_uk": ["GB", "IE"],
        }
        countries = [rng.choice(country_by_supplier[supplier]) for supplier in suppliers]

        return pd.DataFrame(
            {
                "order_id": [f"ORD-{seed:04d}-{index:07d}" for index in range(row_count)],
                "supplier_id": suppliers,
                "event_timestamp": timestamps,
                "product_category": selected_categories,
                "quantity": quantities.astype("int64"),
                "unit_price": np.round(unit_prices, 2),
                "currency": [self.config.supplier_currencies[item] for item in suppliers],
                "customer_country": countries,
            },
            columns=self.config.required_columns,
        )
