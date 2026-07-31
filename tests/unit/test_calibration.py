from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from lolpredictor.calibration import (
    CALIBRATION_METHODS,
    ProbabilityCalibrator,
    fit_probability_calibrator,
)
from lolpredictor.features import generate_historical_features
from lolpredictor.models import (
    BlendedCandidate,
    candidate_requires_calibration,
    fit_candidate,
)
from lolpredictor.schemas import HistoricalMatch
from lolpredictor.settings import ExperimentSettings
from lolpredictor.splits import chronological_split
from lolpredictor.training import _refit_partitions


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_calibrators_are_monotone_serializable_and_record_their_interval(
    method: str,
    tmp_path: Path,
) -> None:
    raw = np.linspace(0.03, 0.97, 80, dtype=float)
    labels = np.array(
        [int(probability + 0.08 * np.sin(index) >= 0.5) for index, probability in enumerate(raw)],
        dtype=int,
    )
    timestamps = pd.Series(pd.date_range("2025-01-01T00:00:00Z", periods=len(raw), freq="12h"))

    calibrator = fit_probability_calibrator(
        raw,
        labels,
        timestamps,
        method=method,
        random_seed=17,
    )

    assert isinstance(calibrator, ProbabilityCalibrator)
    path = tmp_path / f"{method}.joblib"
    joblib.dump(calibrator, path)
    restored = joblib.load(path)
    grid = np.linspace(0.001, 0.999, 501, dtype=float)
    mapped = restored.predict(grid)
    assert np.isfinite(mapped).all()
    assert ((mapped > 0.0) & (mapped < 1.0)).all()
    assert np.diff(mapped).min() >= -1e-12
    np.testing.assert_allclose(mapped, calibrator.predict(grid))

    summary = restored.summary()
    assert summary["method"] == method
    assert summary["sample_count"] == len(raw)
    assert summary["positive_count"] == int(labels.sum())
    assert summary["fit_start_timestamp"] == timestamps.min().isoformat()
    assert summary["fit_end_timestamp"] == timestamps.max().isoformat()


@pytest.mark.parametrize(
    ("labels", "raw"),
    [
        (np.zeros(8, dtype=int), np.linspace(0.1, 0.8, 8)),
        (np.array([0, 1] * 4, dtype=int), np.full(8, 0.5)),
    ],
)
def test_calibration_abstains_when_the_interval_has_no_identifiable_mapping(
    labels: np.ndarray,
    raw: np.ndarray,
) -> None:
    timestamps = pd.Series(pd.date_range("2025-01-01T00:00:00Z", periods=len(raw), freq="1h"))

    calibrator = fit_probability_calibrator(
        raw,
        labels,
        timestamps,
        method="platt",
        random_seed=17,
    )

    assert calibrator is None


def test_calibration_rejects_misaligned_or_nonfinite_inputs() -> None:
    timestamps = pd.Series(pd.date_range("2025-01-01T00:00:00Z", periods=4, freq="1h"))
    with pytest.raises(ValueError, match="aligned vectors"):
        fit_probability_calibrator(
            np.array([0.2, 0.8]),
            np.array([0, 1, 0]),
            timestamps.iloc[:2],
            method="platt",
            random_seed=17,
        )
    with pytest.raises(ValueError, match="non-finite"):
        fit_probability_calibrator(
            np.array([0.2, np.nan, 0.4, 0.8]),
            np.array([0, 1, 0, 1]),
            timestamps,
            method="platt",
            random_seed=17,
        )


@pytest.mark.parametrize(
    ("candidate_name", "method"),
    [
        ("catboost_numeric", "platt"),
        ("catboost_numeric_beta", "beta"),
        ("catboost_numeric_isotonic", "isotonic"),
    ],
)
def test_numeric_catboost_calibration_uses_only_the_trailing_interval(
    candidate_name: str,
    method: str,
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    splits = chronological_split(frame, settings.splits)

    candidate = fit_candidate(
        candidate_name,
        splits.fit,
        splits.calibration,
        settings,
    )

    assert isinstance(candidate.calibrator, ProbabilityCalibrator)
    assert candidate.calibrator.method == method
    assert candidate.calibration_summary()["output_mapping_method"] == method
    assert candidate.calibrator.sample_count == len(splits.calibration)
    calibration_end = pd.to_datetime(
        candidate.calibrator.fit_end_timestamp,
        utc=True,
    )
    validation_start = pd.to_datetime(
        splits.validation["match_timestamp"],
        utc=True,
    ).min()
    assert calibration_end < validation_start


@pytest.mark.parametrize("method", ("platt", "beta", "isotonic"))
def test_blend_mapper_is_fitted_after_raw_components_without_label_reuse(
    method: str,
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    splits = chronological_split(frame, settings.splits)
    candidate = fit_candidate(
        f"elo_catboost_numeric_raw_blend_50_{method}",
        splits.fit,
        splits.calibration,
        settings,
    )

    assert isinstance(candidate, BlendedCandidate)
    assert isinstance(candidate.calibrator, ProbabilityCalibrator)
    assert candidate.calibrator.method == method
    assert candidate.calibration_summary()["output_mapping_method"] == method
    assert all(component.calibrator is None for component in candidate.components)
    assert candidate.calibration_sample_count == len(splits.calibration)
    raw_blend = candidate.raw_probability(splits.validation)
    np.testing.assert_allclose(
        candidate.predict_probability(splits.validation),
        candidate.calibrator.predict(raw_blend),
    )


def test_changing_calibration_labels_cannot_refit_blend_components(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    splits = chronological_split(frame, settings.splits)
    flipped_calibration = splits.calibration.copy()
    flipped_calibration["blue_win"] = 1 - flipped_calibration["blue_win"]

    original = fit_candidate(
        "elo_catboost_numeric_raw_blend_50_platt",
        splits.fit,
        splits.calibration,
        settings,
    )
    altered = fit_candidate(
        "elo_catboost_numeric_raw_blend_50_platt",
        splits.fit,
        flipped_calibration,
        settings,
    )

    assert isinstance(original, BlendedCandidate)
    assert isinstance(altered, BlendedCandidate)
    for original_component, altered_component in zip(
        original.components,
        altered.components,
        strict=True,
    ):
        np.testing.assert_allclose(
            original_component.predict_probability(splits.validation),
            altered_component.predict_probability(splits.validation),
        )
    assert not np.allclose(
        original.predict_probability(splits.validation),
        altered.predict_probability(splits.validation),
    )


def test_native_refit_uses_all_rows_but_mapped_refit_reserves_calibration(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)

    native_fit, native_calibration = _refit_partitions(
        frame,
        "elo_catboost_numeric_raw_blend_50",
        settings,
    )
    mapped_fit, mapped_calibration = _refit_partitions(
        frame,
        "elo_catboost_numeric_raw_blend_50_beta",
        settings,
    )

    assert not candidate_requires_calibration("elo_catboost_numeric_raw_blend_50")
    assert candidate_requires_calibration("elo_catboost_numeric_raw_blend_50_beta")
    assert len(native_fit) == len(frame)
    assert native_calibration.empty
    assert len(mapped_fit) + len(mapped_calibration) == len(frame)
    mapped_fit_end = pd.to_datetime(mapped_fit["match_timestamp"], utc=True).max()
    mapped_calibration_start = pd.to_datetime(
        mapped_calibration["match_timestamp"],
        utc=True,
    ).min()
    assert mapped_fit_end < mapped_calibration_start


def test_native_blend_summary_explicitly_records_no_output_mapping(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    splits = chronological_split(frame, settings.splits)
    candidate = fit_candidate(
        "elo_catboost_numeric_raw_blend_50",
        splits.fit,
        splits.calibration,
        settings,
    )

    summary = candidate.calibration_summary()
    assert summary["output_mapping_method"] == "native"
    assert summary["calibrator"] is None
    assert all(
        component["output_mapping_method"] == "native" for component in summary["components"]
    )
