import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lolpredictor.features import generate_historical_features
from lolpredictor.optimization.models import (
    BRADLEY_TERRY_CONTEXT_FEATURES,
    DynamicBradleyTerryClassifier,
    numeric_feature_names,
)
from lolpredictor.optimization.registry import StudyRegistry, sanitize_trial_error
from lolpredictor.optimization.schedule import (
    TrialSpec,
    generate_trial_schedule,
    schedule_fingerprint,
)
from lolpredictor.optimization.selection import select_outer_winners
from lolpredictor.optimization.settings import load_optimization_configuration
from lolpredictor.optimization.splits import (
    build_nested_folds,
    trailing_fit_calibration_split,
)
from lolpredictor.settings import ExperimentSettings

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_default_schedule_is_deterministic_and_balanced() -> None:
    configuration = load_optimization_configuration(
        REPOSITORY_ROOT / "configs" / "v5-optimization.yaml"
    )
    first = generate_trial_schedule(configuration.optimization)
    second = generate_trial_schedule(configuration.optimization)

    assert [spec.to_dict() for spec in first] == [spec.to_dict() for spec in second]
    assert schedule_fingerprint(first) == schedule_fingerprint(second)
    assert len(first) == 300
    allocation = Counter((spec.outer_fold, spec.family, spec.feature_mode) for spec in first)
    for outer_fold in range(1, 5):
        for family in configuration.optimization.allocation.families:
            assert allocation[(outer_fold, family, "cached")] == 10
            assert allocation[(outer_fold, family, "replay")] == 5

    for prefix_end in (20, 40, 60):
        outer_counts = Counter(spec.outer_fold for spec in first[:prefix_end])
        assert max(outer_counts.values()) - min(outer_counts.values()) <= 1


def test_trial_spec_detects_tampering() -> None:
    configuration = load_optimization_configuration(
        REPOSITORY_ROOT / "configs" / "v5-optimization.yaml"
    )
    value = generate_trial_schedule(configuration.optimization)[0].to_dict()
    value["model_parameters"]["c"] = 999.0

    with pytest.raises(ValueError, match="hash mismatch"):
        TrialSpec.from_dict(value)


def test_nested_folds_and_mapper_partitions_are_strictly_chronological(
    synthetic_matches: list,
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    configuration = load_optimization_configuration(
        REPOSITORY_ROOT / "configs" / "v5-optimization.yaml"
    )
    nested = configuration.optimization.nested_validation.model_copy(
        update={"outer_fold_count": 2, "inner_fold_count": 2}
    )
    folds = build_nested_folds(frame, nested)

    assert len(folds) == 2
    for outer in folds:
        assert (
            pd.to_datetime(outer.history["match_timestamp"], utc=True).max()
            < pd.to_datetime(outer.score["match_timestamp"], utc=True).min()
        )
        for inner in outer.inner_folds:
            fit, calibration = trailing_fit_calibration_split(
                inner.history,
                requires_calibration=True,
                calibration_fraction=0.1,
            )
            assert (
                pd.to_datetime(fit["match_timestamp"], utc=True).max()
                < pd.to_datetime(calibration["match_timestamp"], utc=True).min()
                < pd.to_datetime(inner.score["match_timestamp"], utc=True).min()
            )


def test_bradley_terry_vocabulary_and_scaling_use_fit_rows_only(
    synthetic_matches: list,
    settings: ExperimentSettings,
) -> None:
    frame, _ = generate_historical_features(synthetic_matches, settings.features)
    fit = frame.iloc[:80].copy()
    later = frame.iloc[80:84].copy()
    later.loc[:, "blue_team_key"] = "FUTURE_BLUE"
    later.loc[:, "red_team_key"] = "FUTURE_RED"
    numeric_names = numeric_feature_names(("core", "team_strength"))
    feature_names = (*numeric_names, *BRADLEY_TERRY_CONTEXT_FEATURES)
    estimator = DynamicBradleyTerryClassifier(
        numeric_feature_names=numeric_names,
        c=0.25,
        player_weight=1.0,
        champion_weight=0.5,
        random_seed=7,
    )
    labels = fit["blue_win"].to_numpy(dtype=int)
    estimator.fit(
        fit.loc[:, list(feature_names)],
        labels,
        sample_weight=np.ones(len(fit), dtype=float),
    )

    vocabulary = set(estimator.vectorizer.get_feature_names_out())
    assert "team=FUTURE_BLUE" not in vocabulary
    assert "team=FUTURE_RED" not in vocabulary
    expected_mean = float(fit[numeric_names[0]].mean())
    assert estimator.numeric_means[numeric_names[0]] == pytest.approx(expected_mean)
    probabilities = estimator.predict_proba(later.loc[:, list(feature_names)])[:, 1]
    assert np.isfinite(probabilities).all()
    assert ((probabilities > 0.0) & (probabilities < 1.0)).all()


def test_registry_resumes_running_trials_and_rejects_contract_changes(
    tmp_path: Path,
) -> None:
    configuration = load_optimization_configuration(
        REPOSITORY_ROOT / "configs" / "v5-optimization.yaml"
    )
    specs = generate_trial_schedule(configuration.optimization)[:2]
    contract = {
        "study_name": "unit-study",
        "dataset_fingerprint": "a" * 64,
    }
    with StudyRegistry(tmp_path / "study") as registry:
        registry.initialize(contract=contract, specs=specs)
        registry.claim(specs[0].trial_id)
        assert registry.reset_interrupted_trials() == 1
        assert registry.status_counts()["pending"] == 2
        registry.claim(specs[0].trial_id)
        registry.complete(
            specs[0].trial_id,
            result={"log_loss": 0.6},
            duration_seconds=1.5,
            worker_pid=123,
        )
        assert registry.status_counts()["completed"] == 1

    with StudyRegistry(tmp_path / "study") as registry:
        registry.initialize(contract=contract, specs=specs)
        assert registry.status_counts()["completed"] == 1
        with pytest.raises(ValueError, match="contract mismatch"):
            registry.initialize(
                contract={**contract, "dataset_fingerprint": "b" * 64},
                specs=specs,
            )


def test_registry_error_sanitization_redacts_credential_values() -> None:
    credential_message = "RIOT_API" + "_KEY=" + "do-not-store pass" + "word: " + "also-secret"
    error = RuntimeError(credential_message)
    sanitized = sanitize_trial_error(error)

    assert "do-not-store" not in sanitized
    assert "also-secret" not in sanitized
    assert sanitized.count("<redacted>") == 2


def _selection_metric(log_loss: float, *, ece: float = 0.02) -> dict:
    return {
        "log_loss": log_loss,
        "brier_score": 0.21,
        "expected_calibration_error": ece,
        "breakdowns": {
            "league": {
                league: {
                    "log_loss": log_loss,
                    "sample_count": 200,
                }
                for league in (
                    "Tencent LoL Pro League",
                    "LoL Champions Korea",
                    "LoL EMEA Championship",
                    "League of Legends Championship Pacific",
                    "League of Legends Championship Series",
                    "Circuit Brazilian League of Legends",
                )
            }
        },
    }


def test_finalist_selection_never_reads_outer_results() -> None:
    configuration = load_optimization_configuration(
        REPOSITORY_ROOT / "configs" / "v5-optimization.yaml"
    )
    controls_by_outer = {}
    records = []
    schedule = generate_trial_schedule(configuration.optimization)
    for outer_fold in range(1, 5):
        controls_by_outer[str(outer_fold)] = {
            "elo_only": {
                "pooled_inner": _selection_metric(0.62),
                "inner_folds": [
                    {"fold_number": fold, "metrics": {"log_loss": 0.62}} for fold in range(1, 4)
                ],
            },
            "team_roster_logistic": {
                "pooled_inner": _selection_metric(0.615),
                "inner_folds": [],
            },
            "elo_catboost_regional_raw_blend_50": {
                "pooled_inner": _selection_metric(0.61),
                "inner_folds": [],
            },
        }
        spec = next(item for item in schedule if item.outer_fold == outer_fold)
        records.append(
            {
                "trial_id": spec.trial_id,
                "outer_fold": outer_fold,
                "family": spec.family,
                "feature_mode": spec.feature_mode,
                "spec_hash": spec.spec_hash,
                "spec": spec.to_dict(),
                "status": "completed",
                "outer_result": {"log_loss": 99.0},
                "result": {
                    "pooled_inner": _selection_metric(0.60),
                    "inner_folds": [
                        {"fold_number": fold, "metrics": {"log_loss": 0.60}} for fold in range(1, 4)
                    ],
                },
            }
        )
    controls = {"outer_folds": controls_by_outer}

    first = select_outer_winners(records, controls, configuration)
    for record in records:
        record["outer_result"]["log_loss"] = 0.0
    second = select_outer_winners(records, controls, configuration)

    assert first == second
    assert all(
        outer["selection"]["selection_kind"] == "searched_trial"
        for outer in first["outer_selections"].values()
    )


def test_exclusive_cutoff_is_timezone_aware() -> None:
    configuration = load_optimization_configuration(
        REPOSITORY_ROOT / "configs" / "v5-optimization.yaml"
    )
    assert configuration.optimization.development_cutoff_timestamp == datetime(
        2026,
        1,
        1,
        tzinfo=UTC,
    )
    assert json.loads(json.dumps(configuration.optimization.resolved()))[
        "development_cutoff_timestamp"
    ].startswith("2026-01-01")
