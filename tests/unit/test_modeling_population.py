import pandas as pd
import pytest

from lolpredictor.features import generate_historical_features
from lolpredictor.schemas import HistoricalMatch
from lolpredictor.settings import ExperimentSettings, ModelingPopulationSettings
from lolpredictor.training import filter_modeling_population


def test_modeling_population_selection_is_independent_of_outcomes(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    configured = settings.model_copy(
        update={
            "modeling_population": ModelingPopulationSettings(
                require_official=True,
                tournament_levels=("Primary",),
            )
        }
    )
    altered = frame.copy()
    altered["blue_win"] = 1 - altered["blue_win"]

    selected, summary = filter_modeling_population(frame, configured)
    altered_selected, altered_summary = filter_modeling_population(altered, configured)

    assert selected["match_id"].tolist() == altered_selected["match_id"].tolist()
    assert summary == altered_summary


def test_population_filter_rejects_an_empty_structural_population(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    configured = settings.model_copy(
        update={
            "modeling_population": ModelingPopulationSettings(
                require_official=True,
                tournament_levels=("NotARealLevel",),
            )
        }
    )

    with pytest.raises(ValueError, match="excluded every feature row"):
        filter_modeling_population(frame, configured)


def test_population_filter_preserves_chronological_order(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    shuffled = frame.sample(frac=1.0, random_state=7)

    selected, _ = filter_modeling_population(shuffled, settings)

    timestamps = pd.to_datetime(selected["match_timestamp"], utc=True)
    assert timestamps.is_monotonic_increasing
