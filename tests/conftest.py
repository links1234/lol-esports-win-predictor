from pathlib import Path

import pytest

from lolpredictor.fixtures import generate_synthetic_matches
from lolpredictor.schemas import HistoricalMatch
from lolpredictor.settings import ExperimentSettings, load_settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def settings() -> ExperimentSettings:
    return load_settings(REPOSITORY_ROOT / "configs" / "baselines.yaml")


@pytest.fixture(scope="session")
def synthetic_matches() -> list[HistoricalMatch]:
    return generate_synthetic_matches()
