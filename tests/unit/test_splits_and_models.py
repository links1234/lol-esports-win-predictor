from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from lolpredictor.features import (
    FEATURE_NAMES,
    LEGACY_FEATURE_NAMES,
    MODEL_CONTEXT_NAMES,
    PRE_REGIONAL_FEATURE_NAMES,
    REGIONAL_FEATURE_NAMES,
    STATE_SPACE_FEATURE_NAMES,
    V4_FEATURE_NAMES,
    generate_historical_features,
)
from lolpredictor.models import (
    CANDIDATE_NAMES,
    SIMPLE_CONTROL_NAMES,
    BlendedCandidate,
    fit_all_candidates,
    fit_candidate,
    recency_sample_weights,
)
from lolpredictor.schemas import HistoricalMatch
from lolpredictor.settings import ExperimentSettings, SplitSettings
from lolpredictor.splits import chronological_split, rolling_origin_splits


def test_splits_are_contiguous_and_keep_timestamp_groups_together(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    splits = chronological_split(frame, settings.splits)
    partitions = [splits.fit, splits.calibration, splits.validation, splits.test]
    timestamp_sets = [
        set(pd.to_datetime(partition["match_timestamp"], utc=True)) for partition in partitions
    ]

    for left_index, left in enumerate(timestamp_sets):
        for right in timestamp_sets[left_index + 1 :]:
            assert left.isdisjoint(right)
    assert max(timestamp_sets[0]) < min(timestamp_sets[1])
    assert max(timestamp_sets[1]) < min(timestamp_sets[2])
    assert max(timestamp_sets[2]) < min(timestamp_sets[3])
    assert sum(len(partition) for partition in partitions) == len(frame)


def test_timestamp_boundaries_define_validation_and_final_holdout(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    split_settings = SplitSettings(
        validation_start_timestamp=datetime(2024, 2, 15, tzinfo=UTC),
        test_start_timestamp=datetime(2024, 3, 15, tzinfo=UTC),
        calibration_fraction_within_train=0.1,
        refit_before_test=True,
    )

    splits = chronological_split(frame, split_settings)
    validation_timestamps = pd.to_datetime(splits.validation["match_timestamp"], utc=True)
    test_timestamps = pd.to_datetime(splits.test["match_timestamp"], utc=True)

    assert validation_timestamps.min() >= pd.Timestamp("2024-02-15T00:00:00Z")
    assert validation_timestamps.max() < pd.Timestamp("2024-03-15T00:00:00Z")
    assert test_timestamps.min() >= pd.Timestamp("2024-03-15T00:00:00Z")
    assert len(splits.fit) + len(splits.calibration) + len(splits.validation) + len(
        splits.test
    ) == len(frame)


def test_split_settings_reject_mixed_or_naive_modes() -> None:
    with pytest.raises(ValueError, match="exactly one split mode"):
        SplitSettings(
            train_fraction=0.6,
            validation_fraction=0.2,
            validation_start_timestamp=datetime(2024, 2, 1, tzinfo=UTC),
            test_start_timestamp=datetime(2024, 3, 1, tzinfo=UTC),
            calibration_fraction_within_train=0.1,
        )
    with pytest.raises(ValueError, match="must include a timezone"):
        SplitSettings(
            validation_start_timestamp=datetime(2024, 2, 1),
            test_start_timestamp=datetime(2024, 3, 1),
            calibration_fraction_within_train=0.1,
        )


def test_logistic_preprocessing_is_fitted_only_on_fit_interval(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    splits = chronological_split(frame, settings.splits)
    altered_calibration = splits.calibration.copy()
    altered_calibration.loc[:, FEATURE_NAMES] = 1_000_000.0

    candidate = fit_candidate(
        "logistic_regression",
        splits.fit,
        altered_calibration,
        settings,
    )
    scaler = candidate.estimator.named_steps["scaler"]
    expected_means = splits.fit.loc[:, PRE_REGIONAL_FEATURE_NAMES].mean().to_numpy(dtype=float)
    np.testing.assert_allclose(scaler.mean_, expected_means)
    assert int(scaler.n_samples_seen_) == len(splits.fit)


def test_legacy_controls_use_only_the_pre_v3_feature_contract(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    splits = chronological_split(frame, settings.splits)

    legacy_logistic = fit_candidate(
        "legacy_logistic_regression",
        splits.fit,
        splits.calibration,
        settings,
    )
    legacy_booster = fit_candidate(
        "legacy_gradient_boosted_trees",
        splits.fit,
        splits.calibration,
        settings,
    )

    assert legacy_logistic.feature_names == LEGACY_FEATURE_NAMES
    assert legacy_booster.feature_names == LEGACY_FEATURE_NAMES
    assert set(legacy_logistic.feature_names) < set(FEATURE_NAMES)


def test_existing_v3_candidates_keep_the_pre_regional_contract(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    splits = chronological_split(frame, settings.splits)

    candidates = [
        fit_candidate(name, splits.fit, splits.calibration, settings)
        for name in (
            "logistic_regression",
            "gradient_boosted_trees",
            "catboost_numeric_raw",
            "catboost_context",
            "elo_catboost_numeric_raw_blend_50",
        )
    ]

    assert candidates[0].feature_names == PRE_REGIONAL_FEATURE_NAMES
    assert candidates[1].feature_names == PRE_REGIONAL_FEATURE_NAMES
    assert candidates[2].feature_names == PRE_REGIONAL_FEATURE_NAMES
    assert candidates[3].feature_names == (
        *PRE_REGIONAL_FEATURE_NAMES,
        *MODEL_CONTEXT_NAMES,
    )
    assert candidates[4].feature_names == PRE_REGIONAL_FEATURE_NAMES


def test_regional_candidates_use_only_the_frozen_feature_contracts(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    splits = chronological_split(frame, settings.splits)

    regional_elo = fit_candidate(
        "regional_elo_logistic",
        splits.fit,
        splits.calibration,
        settings,
    )
    full_regional = fit_candidate(
        "catboost_regional_raw",
        splits.fit,
        splits.calibration,
        settings,
    )
    blend = fit_candidate(
        "elo_catboost_regional_raw_blend_50",
        splits.fit,
        splits.calibration,
        settings,
    )

    assert "adjusted_elo_diff" in regional_elo.feature_names
    assert set(REGIONAL_FEATURE_NAMES) < set(regional_elo.feature_names)
    assert full_regional.feature_names == V4_FEATURE_NAMES
    assert full_regional.calibrator is None
    assert blend.feature_names == V4_FEATURE_NAMES
    assert blend.calibrator is None
    assert isinstance(blend, BlendedCandidate)
    assert blend.weights == (0.5, 0.5)
    assert set(blend.feature_names).isdisjoint(STATE_SPACE_FEATURE_NAMES)


def test_candidate_and_control_contracts_are_complete_and_unique() -> None:
    assert len(CANDIDATE_NAMES) == len(set(CANDIDATE_NAMES))
    assert set(SIMPLE_CONTROL_NAMES) < set(CANDIDATE_NAMES)


def test_experiment_can_freeze_a_candidate_subset(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    selected_names = ("blue_side_base_rate", "elo_only", "team_roster_logistic")
    configured = settings.model_copy(
        update={"models": settings.models.model_copy(update={"candidate_names": selected_names})}
    )
    frame, _ = generate_historical_features(synthetic_matches, configured.features)
    splits = chronological_split(frame, configured.splits)
    candidates = fit_all_candidates(splits.fit, splits.calibration, configured)
    assert tuple(candidates) == selected_names


def test_native_and_platt_calibrated_catboost_are_separate_candidates(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    splits = chronological_split(frame, settings.splits)

    raw = fit_candidate(
        "catboost_team_roster_raw",
        splits.fit,
        splits.calibration,
        settings,
    )
    calibrated = fit_candidate(
        "catboost_team_roster",
        splits.fit,
        splits.calibration,
        settings,
    )

    assert raw.calibrator is None
    assert calibrated.calibrator is not None
    assert raw.feature_names == calibrated.feature_names


def test_elo_catboost_blends_use_fixed_declared_weights(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    splits = chronological_split(frame, settings.splits)

    blend_25 = fit_candidate(
        "elo_catboost_numeric_blend_25",
        splits.fit,
        splits.calibration,
        settings,
    )
    blend_50 = fit_candidate(
        "elo_catboost_numeric_blend_50",
        splits.fit,
        splits.calibration,
        settings,
    )

    assert isinstance(blend_25, BlendedCandidate)
    assert isinstance(blend_50, BlendedCandidate)
    assert blend_25.weights == (0.25, 0.75)
    assert blend_50.weights == (0.50, 0.50)
    for blend in (blend_25, blend_50):
        component_probabilities = [
            component.predict_probability(splits.validation) for component in blend.components
        ]
        expected = sum(
            weight * probabilities
            for weight, probabilities in zip(
                blend.weights,
                component_probabilities,
                strict=True,
            )
        )
        np.testing.assert_allclose(
            blend.predict_probability(splits.validation),
            expected,
        )


def test_recency_weighting_uses_only_fit_interval_timestamps(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    splits = chronological_split(frame, settings.splits)
    weights = recency_sample_weights(
        splits.fit,
        half_life_days=settings.models.recency_half_life_days,
    )
    altered_calibration = splits.calibration.copy()
    altered_calibration["match_timestamp"] = pd.Timestamp("2099-01-01T00:00:00Z")

    original = fit_candidate(
        "recency_weighted_logistic",
        splits.fit,
        splits.calibration,
        settings,
    )
    altered = fit_candidate(
        "recency_weighted_logistic",
        splits.fit,
        altered_calibration,
        settings,
    )

    assert len(weights) == len(splits.fit)
    assert weights.max() == pytest.approx(1.0)
    np.testing.assert_allclose(
        original.estimator.named_steps["model"].coef_,
        altered.estimator.named_steps["model"].coef_,
    )


def test_context_catboost_handles_categories_unseen_during_fit(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    splits = chronological_split(frame, settings.splits)
    candidate = fit_candidate(
        "catboost_context",
        splits.fit,
        splits.calibration,
        settings,
    )
    unseen = splits.validation.copy()
    for index, column in enumerate(MODEL_CONTEXT_NAMES):
        unseen[column] = f"<unseen-{index}>"

    probabilities = candidate.predict_probability(unseen)

    assert np.isfinite(probabilities).all()
    assert ((probabilities > 0.0) & (probabilities < 1.0)).all()


def test_all_required_baselines_emit_probabilities(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    splits = chronological_split(frame, settings.splits)
    candidates = fit_all_candidates(splits.fit, splits.calibration, settings)
    assert tuple(candidates) == CANDIDATE_NAMES
    for candidate in candidates.values():
        probabilities = candidate.predict_probability(splits.validation)
        assert probabilities.shape == (len(splits.validation),)
        assert np.isfinite(probabilities).all()
        assert ((probabilities > 0) & (probabilities < 1)).all()


def test_rolling_origin_folds_expand_without_touching_final_holdout(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    release = chronological_split(frame, settings.splits)
    development = pd.concat(
        [release.fit, release.calibration, release.validation],
        ignore_index=True,
    )
    folds = rolling_origin_splits(
        development,
        settings.splits,
        settings.backtest,
    )
    assert len(folds) == settings.backtest.fold_count

    test_timestamp_sets: list[set[pd.Timestamp]] = []
    previous_training_count = 0
    for fold in folds:
        splits = fold.splits
        fit_end = pd.to_datetime(splits.fit["match_timestamp"], utc=True).max()
        calibration_start = pd.to_datetime(
            splits.calibration["match_timestamp"],
            utc=True,
        ).min()
        validation_start = pd.to_datetime(
            splits.validation["match_timestamp"],
            utc=True,
        ).min()
        test_start = pd.to_datetime(splits.test["match_timestamp"], utc=True).min()
        assert fit_end < calibration_start < validation_start < test_start
        assert len(splits.training) > previous_training_count
        previous_training_count = len(splits.training)
        test_timestamp_sets.append(set(pd.to_datetime(splits.test["match_timestamp"], utc=True)))

    for left_index, left in enumerate(test_timestamp_sets):
        for right in test_timestamp_sets[left_index + 1 :]:
            assert left.isdisjoint(right)
    assert (
        pd.to_datetime(folds[-1].splits.test["match_timestamp"], utc=True).max()
        < pd.to_datetime(release.test["match_timestamp"], utc=True).min()
    )
