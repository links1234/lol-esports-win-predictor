from pathlib import Path

import pandas as pd

from lolpredictor.fixtures import build_fixture_database
from lolpredictor.optimization.data import (
    load_cached_development_frame,
    replay_development_frame,
)
from lolpredictor.optimization.settings import load_optimization_configuration
from lolpredictor.training import materialize_historical_features

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_default_replay_matches_the_materialized_bounded_feature_table(
    tmp_path: Path,
) -> None:
    configuration = load_optimization_configuration(
        REPOSITORY_ROOT / "configs" / "v5-optimization-fixture.yaml"
    )
    database = tmp_path / "fixture.duckdb"
    build_fixture_database(database)
    materialize_historical_features(database, configuration.experiment)
    cutoff = configuration.optimization.development_cutoff_timestamp

    cached, _ = load_cached_development_frame(
        database,
        configuration.experiment,
        cutoff=cutoff,
    )
    replayed, replay_settings = replay_development_frame(
        database,
        configuration.experiment,
        cutoff=cutoff,
        feature_parameters={},
    )

    assert replay_settings == configuration.experiment.features
    pd.testing.assert_frame_equal(
        cached.reset_index(drop=True),
        replayed.reset_index(drop=True),
        check_dtype=False,
    )
