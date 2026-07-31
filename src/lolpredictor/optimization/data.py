"""Cutoff-bounded optimizer data loading and point-in-time replay."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lolpredictor.features import generate_historical_features, validate_feature_frame
from lolpredictor.schemas import HistoricalMatch
from lolpredictor.settings import ExperimentSettings, FeatureSettings
from lolpredictor.storage import (
    get_metadata,
    load_feature_table,
    load_matches,
)
from lolpredictor.training import filter_modeling_population


def _assert_exclusive_cutoff(frame: pd.DataFrame, cutoff: datetime) -> None:
    timestamps = pd.to_datetime(frame["match_timestamp"], utc=True)
    if frame.empty or bool((timestamps >= pd.Timestamp(cutoff)).any()):
        raise ValueError("Optimizer data violated its exclusive development cutoff")


def load_cached_development_frame(
    database_path: Path,
    settings: ExperimentSettings,
    *,
    cutoff: datetime,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only pre-cutoff labels from the materialized feature table."""
    metadata = get_metadata(database_path)
    if metadata.get("feature_schema_version") != settings.features.feature_schema_version:
        raise ValueError(
            "Feature table is missing or was generated with a different schema version"
        )
    source = load_feature_table(database_path, before_timestamp=cutoff)
    _assert_exclusive_cutoff(source, cutoff)
    validate_feature_frame(source)
    selected, population = filter_modeling_population(source, settings)
    _assert_exclusive_cutoff(selected, cutoff)
    return selected, population


def replay_development_frame(
    database_path: Path,
    settings: ExperimentSettings,
    *,
    cutoff: datetime,
    feature_parameters: dict[str, float | int],
) -> tuple[pd.DataFrame, FeatureSettings]:
    """Replay bounded source matches under one trial's feature-state parameters."""
    matches = load_matches(database_path, before_timestamp=cutoff)
    return replay_development_matches(
        matches,
        settings,
        cutoff=cutoff,
        feature_parameters=feature_parameters,
    )


def replay_development_matches(
    matches: Sequence[HistoricalMatch],
    settings: ExperimentSettings,
    *,
    cutoff: datetime,
    feature_parameters: dict[str, float | int],
) -> tuple[pd.DataFrame, FeatureSettings]:
    """Replay an already SQL-bounded match sequence under one feature configuration."""
    replay_settings = FeatureSettings.model_validate(
        {
            **settings.features.model_dump(),
            **feature_parameters,
        }
    )
    if not matches:
        raise ValueError("No pre-cutoff matches are available for feature replay")
    if any(match.match_timestamp >= cutoff for match in matches):
        raise ValueError("Feature replay loaded a match outside the development boundary")
    source, _ = generate_historical_features(matches, replay_settings)
    _assert_exclusive_cutoff(source, cutoff)
    selected, _ = filter_modeling_population(source, settings)
    _assert_exclusive_cutoff(selected, cutoff)
    return selected, replay_settings
