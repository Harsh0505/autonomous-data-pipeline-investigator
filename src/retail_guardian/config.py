from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SupplierContract:
    """The data contract for one upstream supplier."""

    supplier_id: str
    currency: str


@dataclass(frozen=True)
class CategoryProfile:
    """Expected pricing characteristics used by generation and anomaly detection."""

    mean_price: float
    price_std: float


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration shared by the generator, checks, and pipeline stages."""

    required_columns: tuple[str, ...] = (
        "order_id",
        "supplier_id",
        "event_timestamp",
        "product_category",
        "quantity",
        "unit_price",
        "currency",
        "customer_country",
    )
    suppliers: tuple[SupplierContract, ...] = (
        SupplierContract("atlas_us", "USD"),
        SupplierContract("europa_de", "EUR"),
        SupplierContract("meridian_uk", "GBP"),
    )
    category_profiles: dict[str, CategoryProfile] = field(
        default_factory=lambda: {
            "electronics": CategoryProfile(280.0, 45.0),
            "home": CategoryProfile(72.0, 14.0),
            "apparel": CategoryProfile(48.0, 9.0),
            "beauty": CategoryProfile(31.0, 6.0),
            "sports": CategoryProfile(86.0, 17.0),
        }
    )
    fx_to_gbp: dict[str, float] = field(
        default_factory=lambda: {"GBP": 1.0, "USD": 0.79, "EUR": 0.84}
    )
    maximum_delivery_delay_hours: int = 6
    anomaly_z_score_threshold: float = 5.0

    @property
    def supplier_currencies(self) -> dict[str, str]:
        return {supplier.supplier_id: supplier.currency for supplier in self.suppliers}
