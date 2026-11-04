from __future__ import annotations

from datetime import UTC, datetime

import pytest

from retail_guardian.config import PipelineConfig


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig()


@pytest.fixture
def as_of() -> datetime:
    return datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
