"""Auditable probabilistic baseline models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lolpredictor.calibration import (
    CalibrationMethod,
    ProbabilityCalibrator,
    clip_probabilities,
    fit_probability_calibrator,
)
from lolpredictor.features import (
    LEGACY_FEATURE_NAMES,
    MODEL_CONTEXT_NAMES,
    PRE_REGIONAL_FEATURE_NAMES,
    REGIONAL_FEATURE_NAMES,
    ROLE_NAMES,
    V4_FEATURE_NAMES,
)
from lolpredictor.settings import ExperimentSettings

CANDIDATE_NAMES = (
    "blue_side_base_rate",
    "elo_only",
    "team_roster_logistic",
    "team_roster_gradient_boosted_trees",
    "draft_only_logistic",
    "legacy_logistic_regression",
    "logistic_regression",
    "regional_elo_logistic",
    "team_roster_regional_logistic",
    "regional_logistic_regression",
    "legacy_gradient_boosted_trees",
    "gradient_boosted_trees",
    "regional_gradient_boosted_trees",
    "recency_weighted_logistic",
    "catboost_team_roster_raw",
    "catboost_team_roster",
    "catboost_legacy_raw",
    "catboost_legacy",
    "catboost_numeric_raw",
    "catboost_numeric",
    "catboost_numeric_beta",
    "catboost_numeric_isotonic",
    "catboost_regional_raw",
    "catboost_context",
    "elo_catboost_numeric_blend_25",
    "elo_catboost_numeric_blend_50",
    "elo_catboost_numeric_raw_blend_50",
    "elo_catboost_numeric_raw_blend_50_platt",
    "elo_catboost_numeric_raw_blend_50_beta",
    "elo_catboost_numeric_raw_blend_50_isotonic",
    "elo_catboost_regional_raw_blend_50",
    "elo_logistic_blend",
)

SIMPLE_CONTROL_NAMES = (
    "blue_side_base_rate",
    "elo_only",
    "team_roster_logistic",
    "team_roster_gradient_boosted_trees",
    "draft_only_logistic",
    "legacy_logistic_regression",
    "legacy_gradient_boosted_trees",
)


def configured_candidate_names(settings: ExperimentSettings) -> tuple[str, ...]:
    """Return the candidate set frozen by the experiment configuration."""
    names = settings.models.candidate_names or CANDIDATE_NAMES
    unknown = sorted(set(names) - set(CANDIDATE_NAMES))
    if unknown:
        raise ValueError(f"Unknown configured candidate names: {', '.join(unknown)}")
    required_controls = {"blue_side_base_rate", "elo_only"}
    missing_controls = sorted(required_controls - set(names))
    if missing_controls:
        raise ValueError(
            "Configured candidates must include controls: " + ", ".join(missing_controls)
        )
    return names


TEAM_ROSTER_FEATURES = (
    "blue_side_prior",
    "league_blue_side_prior",
    "blue_first_pick",
    "first_pick_known",
    "blue_first_pick_prior",
    "elo_diff",
    "elo_blue_win_probability",
    "player_elo_diff",
    "blue_team_form",
    "red_team_form",
    "team_form_diff",
    "team_games_log_diff",
    "roster_experience_diff",
    "roster_continuity_diff",
    "team_roster_experience_diff",
    "head_to_head_blue_rate",
    "head_to_head_games",
    "game_number",
    "series_score_diff",
    "series_game_progress",
    "patch_major",
    "patch_minor",
)
REGIONAL_ELO_FEATURES = (
    "adjusted_elo_diff",
    *REGIONAL_FEATURE_NAMES,
)
TEAM_ROSTER_REGIONAL_FEATURES = (
    *TEAM_ROSTER_FEATURES,
    *REGIONAL_FEATURE_NAMES,
)

DRAFT_ONLY_FEATURES = (
    "blue_side_prior",
    "league_blue_side_prior",
    "blue_first_pick",
    "first_pick_known",
    "blue_first_pick_prior",
    "champion_strength_diff",
    "blue_draft_known_fraction",
    "red_draft_known_fraction",
    "ban_strength_diff",
    "patch_champion_strength_diff",
    "synergy_strength_diff",
    "blue_role_matchup_strength",
    "role_matchup_advantage",
    "fearless_ban_count",
    "patch_major",
    "patch_minor",
    *tuple(
        feature_name
        for role in ROLE_NAMES
        for feature_name in (
            f"blue_{role}_champion_strength",
            f"red_{role}_champion_strength",
            f"{role}_champion_strength_diff",
            f"blue_{role}_champion_games_log",
            f"red_{role}_champion_games_log",
            f"{role}_champion_games_log_diff",
            f"{role}_matchup_strength",
            f"{role}_matchup_games_log",
        )
    ),
)
CATBOOST_CONTEXT_FEATURES = (*PRE_REGIONAL_FEATURE_NAMES, *MODEL_CONTEXT_NAMES)


class ConstantProbabilityEstimator:
    """Minimal sklearn-like estimator for the base-rate baseline."""

    def __init__(self, probability: float) -> None:
        self.probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(features), self.probability, dtype=float)
        return np.column_stack([1.0 - positive, positive])


@dataclass
class CalibratedCandidate:
    name: str
    feature_names: tuple[str, ...]
    estimator: Any
    calibrator: ProbabilityCalibrator | LogisticRegression | None
    fit_sample_count: int
    calibration_sample_count: int

    @property
    def probability_calibration_applied(self) -> bool:
        return self.calibrator is not None

    @property
    def output_mapping_method(self) -> str:
        if isinstance(self.calibrator, ProbabilityCalibrator):
            return self.calibrator.method
        if self.calibrator is not None:
            return "platt"
        return "native"

    def _calibrator_summary(self) -> dict[str, Any] | None:
        if isinstance(self.calibrator, ProbabilityCalibrator):
            return self.calibrator.summary()
        if self.calibrator is not None:
            return {
                "method": "platt",
                "estimator": type(self.calibrator).__name__,
                "metadata_status": "unavailable_for_legacy_artifact",
            }
        return None

    def calibration_summary(self) -> dict[str, Any]:
        return {
            "candidate": self.name,
            "output_mapping_method": self.output_mapping_method,
            "calibrator": self._calibrator_summary(),
            "components": [],
        }

    def raw_probability(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        features = frame.loc[:, list(self.feature_names)]
        raw = np.asarray(self.estimator.predict_proba(features)[:, 1], dtype=float)
        return clip_probabilities(raw)

    def predict_probability(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        raw = self.raw_probability(frame)
        if self.calibrator is None:
            return raw
        if isinstance(self.calibrator, ProbabilityCalibrator):
            return self.calibrator.predict(raw)
        # Artifacts created before the explicit calibration contract stored the
        # Platt logistic estimator directly.
        logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
        calibrated = self.calibrator.predict_proba(logits)[:, 1]
        return clip_probabilities(np.asarray(calibrated, dtype=float))


@dataclass
class BlendedCandidate(CalibratedCandidate):
    components: tuple[CalibratedCandidate, ...]
    weights: tuple[float, ...]

    @property
    def probability_calibration_applied(self) -> bool:
        return self.calibrator is not None or any(
            component.probability_calibration_applied for component in self.components
        )

    def calibration_summary(self) -> dict[str, Any]:
        return {
            "candidate": self.name,
            "output_mapping_method": self.output_mapping_method,
            "calibrator": self._calibrator_summary(),
            "components": [
                {
                    "weight": weight,
                    **component.calibration_summary(),
                }
                for weight, component in zip(
                    self.weights,
                    self.components,
                    strict=True,
                )
            ],
        }

    def raw_probability(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        probabilities = sum(
            weight * component.predict_probability(frame)
            for weight, component in zip(
                self.weights,
                self.components,
                strict=True,
            )
        )
        return clip_probabilities(np.asarray(probabilities, dtype=float))


def candidate_requires_calibration(name: str) -> bool:
    """Return whether a final refit must reserve later rows for a mapper."""
    if name not in {*CANDIDATE_NAMES, "elo_only_raw"}:
        raise ValueError(f"Unknown candidate: {name}")
    return name not in {
        "blue_side_base_rate",
        "catboost_team_roster_raw",
        "catboost_legacy_raw",
        "catboost_numeric_raw",
        "catboost_regional_raw",
        "elo_only_raw",
        "elo_catboost_numeric_raw_blend_50",
        "elo_catboost_regional_raw_blend_50",
    }


def _build_estimator(
    name: str,
    settings: ExperimentSettings,
) -> tuple[Any, tuple[str, ...]]:
    if name in {"elo_only", "elo_only_raw"}:
        return (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=settings.models.logistic_c,
                            max_iter=2000,
                            random_state=settings.random_seed,
                        ),
                    ),
                ]
            ),
            ("elo_diff",),
        )
    if name in {
        "team_roster_logistic",
        "draft_only_logistic",
        "legacy_logistic_regression",
        "logistic_regression",
        "regional_elo_logistic",
        "team_roster_regional_logistic",
        "regional_logistic_regression",
        "recency_weighted_logistic",
    }:
        feature_names: tuple[str, ...]
        if name == "team_roster_logistic":
            feature_names = TEAM_ROSTER_FEATURES
        elif name == "draft_only_logistic":
            feature_names = DRAFT_ONLY_FEATURES
        elif name == "legacy_logistic_regression":
            feature_names = LEGACY_FEATURE_NAMES
        elif name == "regional_elo_logistic":
            feature_names = REGIONAL_ELO_FEATURES
        elif name == "team_roster_regional_logistic":
            feature_names = TEAM_ROSTER_REGIONAL_FEATURES
        elif name == "regional_logistic_regression":
            feature_names = V4_FEATURE_NAMES
        else:
            feature_names = PRE_REGIONAL_FEATURE_NAMES
        return (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=settings.models.logistic_c,
                            max_iter=2000,
                            random_state=settings.random_seed,
                        ),
                    ),
                ]
            ),
            feature_names,
        )
    if name.startswith("catboost_"):
        if name in {"catboost_team_roster", "catboost_team_roster_raw"}:
            feature_names = TEAM_ROSTER_FEATURES
        elif name in {"catboost_legacy", "catboost_legacy_raw"}:
            feature_names = LEGACY_FEATURE_NAMES
        elif name == "catboost_regional_raw":
            feature_names = V4_FEATURE_NAMES
        elif name == "catboost_context":
            feature_names = CATBOOST_CONTEXT_FEATURES
        else:
            feature_names = PRE_REGIONAL_FEATURE_NAMES
        categorical_features = list(MODEL_CONTEXT_NAMES) if name == "catboost_context" else None
        return (
            CatBoostClassifier(
                iterations=settings.models.catboost_iterations,
                depth=settings.models.catboost_depth,
                learning_rate=settings.models.catboost_learning_rate,
                l2_leaf_reg=settings.models.catboost_l2_leaf_regularization,
                random_strength=settings.models.catboost_random_strength,
                loss_function="Logloss",
                eval_metric="Logloss",
                random_seed=settings.random_seed,
                has_time=True,
                boosting_type="Plain",
                cat_features=categorical_features,
                allow_writing_files=False,
                verbose=False,
                thread_count=1,
            ),
            feature_names,
        )
    if name in {
        "team_roster_gradient_boosted_trees",
        "legacy_gradient_boosted_trees",
        "gradient_boosted_trees",
        "regional_gradient_boosted_trees",
    }:
        if name == "team_roster_gradient_boosted_trees":
            feature_names = TEAM_ROSTER_FEATURES
        elif name == "legacy_gradient_boosted_trees":
            feature_names = LEGACY_FEATURE_NAMES
        elif name == "regional_gradient_boosted_trees":
            feature_names = V4_FEATURE_NAMES
        else:
            feature_names = PRE_REGIONAL_FEATURE_NAMES
        return (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            learning_rate=settings.models.gradient_boosting_learning_rate,
                            max_iter=settings.models.gradient_boosting_max_iter,
                            max_leaf_nodes=(settings.models.gradient_boosting_max_leaf_nodes),
                            l2_regularization=(settings.models.gradient_boosting_l2_regularization),
                            random_state=settings.random_seed,
                            early_stopping=False,
                        ),
                    ),
                ]
            ),
            feature_names,
        )
    raise ValueError(f"Unknown candidate: {name}")


def _fit_estimator_probability_calibrator(
    estimator: Any,
    feature_names: tuple[str, ...],
    calibration: pd.DataFrame,
    *,
    method: CalibrationMethod,
    random_seed: int,
) -> ProbabilityCalibrator | None:
    labels = calibration["blue_win"].to_numpy(dtype=int)
    raw = np.asarray(
        estimator.predict_proba(calibration.loc[:, list(feature_names)])[:, 1],
        dtype=float,
    )
    return fit_probability_calibrator(
        raw,
        labels,
        calibration["match_timestamp"],
        method=method,
        random_seed=random_seed,
    )


def fit_candidate(
    name: str,
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    settings: ExperimentSettings,
) -> CalibratedCandidate:
    raw_component_blends: dict[
        str,
        tuple[str, tuple[str, ...], CalibrationMethod | None],
    ] = {
        "elo_catboost_numeric_raw_blend_50": (
            "catboost_numeric_raw",
            PRE_REGIONAL_FEATURE_NAMES,
            None,
        ),
        "elo_catboost_numeric_raw_blend_50_platt": (
            "catboost_numeric_raw",
            PRE_REGIONAL_FEATURE_NAMES,
            "platt",
        ),
        "elo_catboost_numeric_raw_blend_50_beta": (
            "catboost_numeric_raw",
            PRE_REGIONAL_FEATURE_NAMES,
            "beta",
        ),
        "elo_catboost_numeric_raw_blend_50_isotonic": (
            "catboost_numeric_raw",
            PRE_REGIONAL_FEATURE_NAMES,
            "isotonic",
        ),
        "elo_catboost_regional_raw_blend_50": (
            "catboost_regional_raw",
            V4_FEATURE_NAMES,
            None,
        ),
    }
    if name in raw_component_blends:
        catboost_name, blend_feature_names, calibration_method = raw_component_blends[name]
        elo = fit_candidate("elo_only_raw", fit, calibration, settings)
        catboost = fit_candidate(
            catboost_name,
            fit,
            calibration,
            settings,
        )
        candidate = BlendedCandidate(
            name=name,
            feature_names=blend_feature_names,
            estimator=None,
            calibrator=None,
            fit_sample_count=len(fit),
            calibration_sample_count=0,
            components=(elo, catboost),
            weights=(0.50, 0.50),
        )
        if calibration_method is None:
            return candidate
        calibrator = fit_probability_calibrator(
            candidate.raw_probability(calibration),
            calibration["blue_win"].to_numpy(dtype=int),
            calibration["match_timestamp"],
            method=calibration_method,
            random_seed=settings.random_seed,
        )
        candidate.calibrator = calibrator
        candidate.calibration_sample_count = len(calibration) if calibrator is not None else 0
        return candidate

    catboost_blend_elo_weights = {
        "elo_catboost_numeric_blend_25": 0.25,
        "elo_catboost_numeric_blend_50": 0.50,
    }
    if name in catboost_blend_elo_weights:
        elo = fit_candidate("elo_only", fit, calibration, settings)
        catboost = fit_candidate(
            "catboost_numeric_raw",
            fit,
            calibration,
            settings,
        )
        elo_weight = catboost_blend_elo_weights[name]
        return BlendedCandidate(
            name=name,
            feature_names=PRE_REGIONAL_FEATURE_NAMES,
            estimator=None,
            calibrator=None,
            fit_sample_count=len(fit),
            calibration_sample_count=elo.calibration_sample_count,
            components=(elo, catboost),
            weights=(elo_weight, 1.0 - elo_weight),
        )
    if name == "elo_logistic_blend":
        elo = fit_candidate("elo_only", fit, calibration, settings)
        logistic = fit_candidate("logistic_regression", fit, calibration, settings)
        elo_weight = settings.models.elo_logistic_blend_elo_weight
        return BlendedCandidate(
            name=name,
            feature_names=PRE_REGIONAL_FEATURE_NAMES,
            estimator=None,
            calibrator=None,
            fit_sample_count=len(fit),
            calibration_sample_count=max(
                elo.calibration_sample_count,
                logistic.calibration_sample_count,
            ),
            components=(elo, logistic),
            weights=(elo_weight, 1.0 - elo_weight),
        )

    labels = fit["blue_win"].to_numpy(dtype=int)
    if len(np.unique(labels)) < 2:
        raise ValueError(f"Fit interval for {name} must contain both outcome classes")

    if name == "blue_side_base_rate":
        estimator = ConstantProbabilityEstimator(float(np.mean(labels)))
        return CalibratedCandidate(
            name=name,
            feature_names=("blue_side_prior",),
            estimator=estimator,
            calibrator=None,
            fit_sample_count=len(fit),
            calibration_sample_count=0,
        )

    estimator, feature_names = _build_estimator(name, settings)
    fit_arguments: dict[str, Any] = {}
    if name == "recency_weighted_logistic":
        fit_arguments["model__sample_weight"] = recency_sample_weights(
            fit,
            half_life_days=settings.models.recency_half_life_days,
        )
    estimator.fit(
        fit.loc[:, list(feature_names)],
        labels,
        **fit_arguments,
    )
    estimator_calibration_method: CalibrationMethod | None
    if name.endswith("_raw"):
        estimator_calibration_method = None
    elif name in {
        "catboost_numeric_beta",
    }:
        estimator_calibration_method = "beta"
    elif name in {
        "catboost_numeric_isotonic",
    }:
        estimator_calibration_method = "isotonic"
    else:
        estimator_calibration_method = "platt"
    calibrator = (
        _fit_estimator_probability_calibrator(
            estimator,
            feature_names,
            calibration,
            method=estimator_calibration_method,
            random_seed=settings.random_seed,
        )
        if estimator_calibration_method is not None
        else None
    )
    return CalibratedCandidate(
        name=name,
        feature_names=feature_names,
        estimator=estimator,
        calibrator=calibrator,
        fit_sample_count=len(fit),
        calibration_sample_count=len(calibration) if calibrator is not None else 0,
    )


def recency_sample_weights(
    fit: pd.DataFrame,
    *,
    half_life_days: float,
) -> NDArray[np.float64]:
    """Compute weights relative only to the latest timestamp in the fit interval."""
    if fit.empty:
        raise ValueError("Cannot compute recency weights for an empty fit interval")
    if half_life_days <= 0.0:
        raise ValueError("Recency half-life must be positive")
    timestamps = pd.to_datetime(fit["match_timestamp"], utc=True)
    age_days = (timestamps.max() - timestamps).dt.total_seconds().to_numpy() / 86400.0
    return cast(
        NDArray[np.float64],
        np.power(0.5, age_days / half_life_days),
    )


def fit_all_candidates(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    settings: ExperimentSettings,
) -> dict[str, CalibratedCandidate]:
    return {
        name: fit_candidate(name, fit, calibration, settings)
        for name in configured_candidate_names(settings)
    }
