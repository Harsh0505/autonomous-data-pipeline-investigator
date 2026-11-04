from __future__ import annotations

import pandas as pd
import pytest

from retail_guardian.config import PipelineConfig
from retail_guardian.synthetic import SyntheticSalesGenerator


def test_generator_is_reproducible(config: PipelineConfig, as_of) -> None:
    generator = SyntheticSalesGenerator(config)
    first = generator.generate(100, seed=7, as_of=as_of)
    second = generator.generate(100, seed=7, as_of=as_of)

    pd.testing.assert_frame_equal(first, second)


def test_generator_conforms_to_contract(config: PipelineConfig, as_of) -> None:
    frame = SyntheticSalesGenerator(config).generate(200, as_of=as_of)

    assert tuple(frame.columns) == config.required_columns
    assert frame["order_id"].is_unique
    assert not frame.isna().any().any()
    assert set(frame["supplier_id"]) == set(config.supplier_currencies)
    assert (frame["quantity"] > 0).all()
    assert (frame["unit_price"] > 0).all()


def test_generator_rejects_invalid_arguments(config: PipelineConfig, as_of) -> None:
    generator = SyntheticSalesGenerator(config)
    with pytest.raises(ValueError, match="row_count"):
        generator.generate(0, as_of=as_of)
    with pytest.raises(ValueError, match="timezone"):
        generator.generate(5, as_of=as_of.replace(tzinfo=None))
