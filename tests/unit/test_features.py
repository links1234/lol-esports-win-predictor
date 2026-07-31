import math
from datetime import timedelta

import pandas as pd
import pytest

from lolpredictor.features import (
    FEATURE_NAMES,
    MODEL_CONTEXT_NAMES,
    PRE_REGIONAL_FEATURE_NAMES,
    PROVENANCE_COLUMNS,
    REGIONAL_FEATURE_NAMES,
    FeatureState,
    TemporalLeakageError,
    build_model_row,
    compute_features,
    generate_historical_features,
    prediction_warnings,
    update_state_batch,
    validate_feature_frame,
)
from lolpredictor.schemas import DraftRequest, HistoricalMatch
from lolpredictor.settings import ExperimentSettings


def test_elo_is_captured_before_current_match_update(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches[:8], settings.features)
    first_timestamp = frame["match_timestamp"].min()
    first_group = frame[frame["match_timestamp"] == first_timestamp]
    assert (first_group["blue_elo"] == settings.features.elo_initial_rating).all()
    assert (first_group["red_elo"] == settings.features.elo_initial_rating).all()

    second_group = frame[frame["match_timestamp"] > first_timestamp]
    assert (second_group["elo_diff"].abs() > 0).any()


def test_seasonal_patch_identifiers_have_stable_numeric_features(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    payload = synthetic_matches[0].model_dump(mode="python")
    payload["patch"] = "25.S1.2"
    match = HistoricalMatch.model_validate(payload)

    frame, _ = generate_historical_features([match], settings.features)

    assert frame.loc[0, "patch"] == "25.S1.2"
    assert frame.loc[0, "patch_major"] == 25.0
    assert frame.loc[0, "patch_minor"] == 2.0


def test_changing_a_future_outcome_cannot_change_earlier_features(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    matches = synthetic_matches[:20]
    changed = list(matches)
    changed[-1] = changed[-1].model_copy(update={"blue_win": not changed[-1].blue_win})

    original_frame, _ = generate_historical_features(matches, settings.features)
    changed_frame, _ = generate_historical_features(changed, settings.features)

    pd.testing.assert_frame_equal(
        original_frame.loc[:, [*FEATURE_NAMES, *PROVENANCE_COLUMNS]],
        changed_frame.loc[:, [*FEATURE_NAMES, *PROVENANCE_COLUMNS]],
    )
    assert (
        original_frame.loc[len(original_frame) - 1, "blue_win"]
        != changed_frame.loc[len(changed_frame) - 1, "blue_win"]
    )


def test_simultaneous_matches_cannot_influence_each_other(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    first_group = synthetic_matches[:4]
    changed = list(first_group)
    changed[0] = changed[0].model_copy(update={"blue_win": not changed[0].blue_win})

    original_frame, _ = generate_historical_features(first_group, settings.features)
    changed_frame, _ = generate_historical_features(changed, settings.features)

    pd.testing.assert_frame_equal(
        original_frame.loc[:, FEATURE_NAMES],
        changed_frame.loc[:, FEATURE_NAMES],
    )
    assert (original_frame["blue_elo"] == settings.features.elo_initial_rating).all()


def test_unknown_home_regions_emit_neutral_regional_features(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    source = synthetic_matches[0]
    request = DraftRequest.model_validate(source.model_dump(exclude={"match_id", "blue_win"}))

    values = compute_features(
        request,
        FeatureState(settings=settings.features),
    ).values

    assert values["blue_home_region_known"] == 0.0
    assert values["red_home_region_known"] == 0.0
    assert values["cross_region_match"] == 0.0
    assert values["blue_region_elo"] == settings.features.elo_initial_rating
    assert values["red_region_elo"] == settings.features.elo_initial_rating
    assert values["region_elo_diff"] == 0.0
    assert values["region_elo_blue_win_probability"] == pytest.approx(0.5)
    assert values["pooled_elo_blue_win_probability"] == pytest.approx(0.5)
    assert all(math.isfinite(values[name]) for name in REGIONAL_FEATURE_NAMES)


def test_regional_elo_uses_only_prior_cross_region_results(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    state = FeatureState(settings=settings.features)
    update_state_batch(state, [synthetic_matches[4], synthetic_matches[7]])
    target = synthetic_matches[16]
    request = DraftRequest.model_validate(target.model_dump(exclude={"match_id", "blue_win"}))

    before = compute_features(request, state)
    update_state_batch(state, [target])
    future = request.model_copy(
        update={"match_timestamp": target.match_timestamp + timedelta(days=1)}
    )
    after = compute_features(future, state)

    assert before.values["cross_region_match"] == 1.0
    assert before.values["region_elo_diff"] == 0.0
    assert before.provenance["regional_elo_history_max_timestamp"] is None
    expected_sign = 1.0 if target.blue_win else -1.0
    assert after.values["region_elo_diff"] * expected_sign > 0.0
    assert after.provenance["regional_elo_history_max_timestamp"] == target.match_timestamp
    assert sorted(state.region_cross_region_games.values()) == [1, 1]


def test_simultaneous_cross_region_outcomes_cannot_influence_each_other(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    domestic_history = [
        synthetic_matches[4],
        synthetic_matches[7],
        *synthetic_matches[8:12],
    ]
    target_group = synthetic_matches[16:20]
    changed_group = list(target_group)
    changed_group[0] = changed_group[0].model_copy(
        update={"blue_win": not changed_group[0].blue_win}
    )

    original_frame, _ = generate_historical_features(
        [*domestic_history, *target_group],
        settings.features,
    )
    changed_frame, _ = generate_historical_features(
        [*domestic_history, *changed_group],
        settings.features,
    )
    target_ids = {match.match_id for match in target_group}
    original_target = original_frame.loc[
        original_frame["match_id"].isin(target_ids),
        [*REGIONAL_FEATURE_NAMES, "regional_elo_history_max_timestamp"],
    ].reset_index(drop=True)
    changed_target = changed_frame.loc[
        changed_frame["match_id"].isin(target_ids),
        [*REGIONAL_FEATURE_NAMES, "regional_elo_history_max_timestamp"],
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(original_target, changed_target)
    assert (original_target["cross_region_match"] == 1.0).all()


@pytest.mark.parametrize(
    ("tournament_level", "is_official"),
    [
        ("Secondary", True),
        ("Primary", False),
        (None, True),
    ],
)
def test_ineligible_matches_cannot_update_regional_ratings(
    tournament_level: str | None,
    is_official: bool,
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    state = FeatureState(settings=settings.features)
    update_state_batch(state, [synthetic_matches[4], synthetic_matches[7]])
    target = synthetic_matches[16].model_copy(
        update={
            "tournament_level": tournament_level,
            "is_official": is_official,
        }
    )

    update_state_batch(state, [target])

    assert state.region_elo == {}
    assert state.region_cross_region_games == {}
    assert state.region_last_seen == {}


def test_provenance_is_strictly_before_every_target(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    target = pd.to_datetime(frame["match_timestamp"], utc=True)
    for column in PROVENANCE_COLUMNS:
        source = pd.to_datetime(frame[column], utc=True)
        assert bool((source.dropna() < target[source.notna()]).all())


def test_leakage_validator_rejects_equal_source_timestamp(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches[:8], settings.features)
    frame.loc[0, "feature_history_max_timestamp"] = frame.loc[0, "match_timestamp"]
    with pytest.raises(TemporalLeakageError, match="strictly before"):
        validate_feature_frame(frame)


def test_feature_state_round_trip_is_prediction_equivalent(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    _, state = generate_historical_features(synthetic_matches[:40], settings.features)
    restored = FeatureState.from_dict(state.to_dict())
    source = synthetic_matches[40]
    request = DraftRequest.model_validate(source.model_dump(exclude={"match_id", "blue_win"}))
    assert compute_features(request, state).values == compute_features(request, restored).values


def test_pre_regional_feature_state_payload_remains_loadable(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    _, state = generate_historical_features(synthetic_matches[:40], settings.features)
    payload = state.to_dict()
    for field_name in (
        "region_elo",
        "region_cross_region_games",
        "team_home_region",
        "region_last_seen",
    ):
        payload.pop(field_name)
    for setting_name in (
        "region_elo_k_factor",
        "regional_rating_levels",
        "regional_rating_excluded_regions",
    ):
        payload["settings"].pop(setting_name)
    restored = FeatureState.from_dict(payload)
    source = synthetic_matches[40]
    request = DraftRequest.model_validate(source.model_dump(exclude={"match_id", "blue_win"}))

    original = compute_features(request, state).values
    restored_values = compute_features(request, restored).values

    assert restored.region_elo == {}
    assert restored.region_cross_region_games == {}
    assert restored.team_home_region == {}
    assert restored.region_last_seen == {}
    for feature_name in PRE_REGIONAL_FEATURE_NAMES:
        assert restored_values[feature_name] == original[feature_name]


def test_training_row_and_prediction_row_use_the_same_feature_pipeline(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    ordered = sorted(
        synthetic_matches[:20],
        key=lambda match: (match.match_timestamp, match.match_id),
    )
    target = ordered[12]
    history = [match for match in ordered if match.match_timestamp < target.match_timestamp]
    _, state = generate_historical_features(history, settings.features)
    full_frame, _ = generate_historical_features(ordered, settings.features)
    historical_row = full_frame.loc[full_frame["match_id"].eq(target.match_id)].iloc[0]
    request = DraftRequest.model_validate(target.model_dump(exclude={"match_id", "blue_win"}))

    prediction_row = build_model_row(request, state)

    assert set(prediction_row) == {*FEATURE_NAMES, *MODEL_CONTEXT_NAMES}
    for feature_name in FEATURE_NAMES:
        assert prediction_row[feature_name] == pytest.approx(historical_row[feature_name])
    for context_name in MODEL_CONTEXT_NAMES:
        assert prediction_row[context_name] == historical_row[context_name]


def test_prediction_rejects_timestamp_at_artifact_cutoff(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    _, state = generate_historical_features(synthetic_matches[:8], settings.features)
    source = synthetic_matches[7]
    request = DraftRequest.model_validate(source.model_dump(exclude={"match_id", "blue_win"}))
    with pytest.raises(TemporalLeakageError, match="strictly later"):
        prediction_warnings(request, state, stale_after_days=45)


def test_prediction_warns_for_unknown_and_stale_inputs(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    _, state = generate_historical_features(synthetic_matches[:8], settings.features)
    cutoff = state.data_cutoff_timestamp
    assert cutoff is not None
    source = synthetic_matches[8]
    payload = source.model_dump(exclude={"match_id", "blue_win"})
    payload.update(
        {
            "match_timestamp": cutoff + timedelta(days=90),
            "league": "INTL",
            "region": "International",
            "blue_team": "UNKNOWN_TEAM",
            "blue_players": [
                "unknown_top",
                "unknown_jungle",
                "unknown_mid",
                "unknown_bottom",
                "unknown_support",
            ],
        }
    )
    request = DraftRequest.model_validate(payload)
    warnings = prediction_warnings(request, state, stale_after_days=45)
    assert any("Unknown teams" in warning for warning in warnings)
    assert any("Unknown players" in warning for warning in warnings)
    assert any("Unknown home regions" in warning for warning in warnings)
    assert any("Stale data" in warning for warning in warnings)


def test_stable_ids_can_be_resolved_from_live_display_names(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    matches_with_ids = [
        match.model_copy(
            update={
                "blue_team_id": f"team:{match.blue_team}",
                "red_team_id": f"team:{match.red_team}",
                "blue_player_ids": tuple(f"player:{player}" for player in match.blue_players),
                "red_player_ids": tuple(f"player:{player}" for player in match.red_players),
            }
        )
        for match in synthetic_matches[:12]
    ]
    _, state = generate_historical_features(matches_with_ids[:8], settings.features)
    source = matches_with_ids[8]
    with_ids = DraftRequest.model_validate(source.model_dump(exclude={"match_id", "blue_win"}))
    by_display_name = with_ids.model_copy(
        update={
            "blue_team_id": None,
            "red_team_id": None,
            "blue_player_ids": None,
            "red_player_ids": None,
        }
    )
    assert (
        compute_features(with_ids, state).values
        == compute_features(
            by_display_name,
            state,
        ).values
    )


def test_series_and_first_pick_features_use_pregame_context(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    payload = synthetic_matches[0].model_dump(exclude={"match_id", "blue_win"})
    payload.update(
        {
            "series_id": "series-1",
            "game_number": 2,
            "blue_series_wins_before": 1,
            "red_series_wins_before": 0,
            "first_pick_side": "red",
            "fearless_bans": [f"Fearless{index}" for index in range(5)],
        }
    )
    request = DraftRequest.model_validate(payload)
    computation = compute_features(
        request,
        FeatureState(settings=settings.features),
    )
    assert computation.values["game_number"] == 2.0
    assert computation.values["series_score_diff"] == 1.0
    assert computation.values["blue_first_pick"] == 0.0
    assert computation.values["fearless_ban_count"] == 5.0
