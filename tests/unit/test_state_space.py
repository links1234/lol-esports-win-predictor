import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from lolpredictor.artifacts import config_fingerprint
from lolpredictor.features import (
    STATE_SPACE_FEATURE_NAMES,
    FeatureState,
    compute_features,
    generate_historical_features,
    update_state_batch,
)
from lolpredictor.schemas import DraftRequest, HistoricalMatch
from lolpredictor.settings import (
    ExperimentSettings,
    FeatureSettings,
    StateSpaceFeatureSettings,
    load_settings,
)
from lolpredictor.state_space.filter import (
    GaussianObservation,
    GaussianSkill,
    predict_gaussian_design,
    project_gaussian_skill,
    skill_key,
    update_gaussian_skills,
)
from lolpredictor.state_space.settings import V6StudySettings, load_v6_configuration


def _enabled_feature_settings(settings: ExperimentSettings) -> FeatureSettings:
    return FeatureSettings.model_validate(
        {
            **settings.features.model_dump(mode="python"),
            "feature_schema_version": "point-in-time-v6-state-space-test",
            "state_space": StateSpaceFeatureSettings(enabled=True).model_dump(mode="python"),
        }
    )


def test_disabled_state_space_preserves_v4_config_fingerprint() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    v4 = load_settings(repository_root / "configs" / "v4-regional-development.yaml")
    v6 = load_v6_configuration(repository_root / "configs" / "v6-state-space.yaml")

    assert "state_space" not in v4.resolved()["features"]
    assert config_fingerprint(v4) == (
        "3dc6f839c7c4403ec0eaafcacf9917aec5548f599d13eedd7c4f2bedb4e66407"
    )
    assert "state_space" in v6.experiment.resolved()["features"]


def test_unknown_state_space_design_is_neutral_and_uncertain() -> None:
    settings = StateSpaceFeatureSettings(enabled=True)
    design = {
        skill_key("global_side", "blue"): 1.0,
        skill_key("team", "blue"): 1.0,
        skill_key("team", "red"): -1.0,
    }

    prediction = predict_gaussian_design(
        {},
        design,
        datetime(2025, 1, 1, tzinfo=UTC),
        settings,
    )

    assert prediction.probability == pytest.approx(0.5)
    assert prediction.linear_mean == 0.0
    assert prediction.linear_variance == pytest.approx(
        settings.global_side_prior_variance + 2.0 * settings.team_prior_variance
    )
    assert all(skill.games == 0 for skill in prediction.projected_skills.values())


def test_evidence_moves_mean_and_contracts_uncertainty() -> None:
    settings = StateSpaceFeatureSettings(enabled=True)
    timestamp = datetime(2025, 1, 1, tzinfo=UTC)
    blue_key = skill_key("team", "blue")
    red_key = skill_key("team", "red")
    design = {
        skill_key("global_side", "blue"): 1.0,
        blue_key: 1.0,
        red_key: -1.0,
    }
    skills: dict[str, GaussianSkill] = {}

    update_gaussian_skills(
        skills,
        [
            GaussianObservation(
                observation_id="game-1",
                design=design,
                outcome=True,
            )
        ],
        timestamp,
        settings,
    )

    assert skills[blue_key].mean > 0.0
    assert skills[red_key].mean < 0.0
    assert skills[blue_key].variance < settings.team_prior_variance
    assert skills[red_key].variance < settings.team_prior_variance
    prediction = predict_gaussian_design(
        skills,
        design,
        timestamp + timedelta(days=1),
        settings,
    )
    assert prediction.probability > 0.5


def test_inactivity_shrinks_mean_and_expands_variance() -> None:
    settings = StateSpaceFeatureSettings(enabled=True)
    key = skill_key("team", "team-a")
    observed = GaussianSkill(
        mean=0.8,
        variance=0.1,
        last_seen=datetime(2025, 1, 1, tzinfo=UTC),
        games=20,
    )

    projected = project_gaussian_skill(
        observed,
        key,
        observed.last_seen + timedelta(days=360),
        settings,
    )

    assert 0.0 < projected.mean < observed.mean
    assert observed.variance < projected.variance < settings.team_prior_variance
    assert projected.games == observed.games


def test_projection_rejects_evidence_at_target_timestamp() -> None:
    settings = StateSpaceFeatureSettings(enabled=True)
    timestamp = datetime(2025, 1, 1, tzinfo=UTC)
    key = skill_key("team", "team-a")
    observed = GaussianSkill(
        mean=0.2,
        variance=0.2,
        last_seen=timestamp,
        games=4,
    )

    with pytest.raises(ValueError, match="strictly after"):
        project_gaussian_skill(
            observed,
            key,
            timestamp,
            settings,
        )


def test_timestamp_batch_update_is_exactly_order_independent() -> None:
    settings = StateSpaceFeatureSettings(enabled=True)
    timestamp = datetime(2025, 1, 1, tzinfo=UTC)
    observations = [
        GaussianObservation(
            observation_id="game-b",
            design={
                skill_key("team", "b"): 1.0,
                skill_key("team", "c"): -1.0,
            },
            outcome=False,
        ),
        GaussianObservation(
            observation_id="game-a",
            design={
                skill_key("team", "a"): 1.0,
                skill_key("team", "b"): -1.0,
            },
            outcome=True,
        ),
    ]
    forward: dict[str, GaussianSkill] = {}
    reverse: dict[str, GaussianSkill] = {}

    forward_probabilities = update_gaussian_skills(
        forward,
        observations,
        timestamp,
        settings,
    )
    reverse_probabilities = update_gaussian_skills(
        reverse,
        list(reversed(observations)),
        timestamp,
        settings,
    )

    assert forward == reverse
    assert forward_probabilities == reverse_probabilities


def test_v6_features_are_target_and_same_timestamp_safe(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    feature_settings = _enabled_feature_settings(settings)
    matches = synthetic_matches[:20]
    changed_target = list(matches)
    changed_target[-1] = changed_target[-1].model_copy(
        update={"blue_win": not changed_target[-1].blue_win}
    )
    original, _ = generate_historical_features(matches, feature_settings)
    changed, _ = generate_historical_features(changed_target, feature_settings)

    pd.testing.assert_frame_equal(
        original.loc[:, STATE_SPACE_FEATURE_NAMES],
        changed.loc[:, STATE_SPACE_FEATURE_NAMES],
    )

    first_group = matches[:4]
    changed_group = list(first_group)
    changed_group[0] = changed_group[0].model_copy(
        update={"blue_win": not changed_group[0].blue_win}
    )
    original_group, _ = generate_historical_features(first_group, feature_settings)
    changed_group_frame, _ = generate_historical_features(
        changed_group,
        feature_settings,
    )
    pd.testing.assert_frame_equal(
        original_group.loc[:, STATE_SPACE_FEATURE_NAMES],
        changed_group_frame.loc[:, STATE_SPACE_FEATURE_NAMES],
    )


def test_v6_feature_state_round_trip_is_exact(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    feature_settings = _enabled_feature_settings(settings)
    _, state = generate_historical_features(
        synthetic_matches[:40],
        feature_settings,
    )
    restored = FeatureState.from_dict(json.loads(json.dumps(state.to_dict(), allow_nan=False)))
    source = synthetic_matches[40]
    request = DraftRequest.model_validate(source.model_dump(exclude={"match_id", "blue_win"}))

    original = compute_features(request, state).values
    round_trip = compute_features(request, restored).values

    assert {name: original[name] for name in STATE_SPACE_FEATURE_NAMES} == {
        name: round_trip[name] for name in STATE_SPACE_FEATURE_NAMES
    }


def test_feature_batch_order_does_not_change_latent_state(
    synthetic_matches: list[HistoricalMatch],
    settings: ExperimentSettings,
) -> None:
    feature_settings = _enabled_feature_settings(settings)
    first_group = synthetic_matches[:4]
    forward = FeatureState(settings=feature_settings)
    reverse = FeatureState(settings=feature_settings)

    update_state_batch(forward, first_group)
    update_state_batch(reverse, list(reversed(first_group)))

    assert forward.state_space_skills == reverse.state_space_skills


def test_v6_config_rejects_a_changed_frozen_parameter() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load(
        (repository_root / "configs" / "v6-state-space.yaml").read_text(encoding="utf-8")
    )
    payload["state_space"]["team_prior_variance"] = 0.65

    with pytest.raises(
        ValidationError,
        match="differ from the frozen protocol",
    ):
        V6StudySettings.model_validate(payload)
