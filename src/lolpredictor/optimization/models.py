"""Searchable model families with one fit-only preprocessing contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lolpredictor.calibration import CalibrationMethod, fit_probability_calibrator
from lolpredictor.features import (
    FEATURE_NAMES,
    MODEL_CONTEXT_NAMES,
    REGIONAL_FEATURE_NAMES,
    ROLE_NAMES,
)
from lolpredictor.models import (
    BlendedCandidate,
    CalibratedCandidate,
    recency_sample_weights,
)
from lolpredictor.optimization.schedule import FeatureGroup, TrialSpec
from lolpredictor.settings import ExperimentSettings

CORE_FEATURES = (
    "blue_side_prior",
    "league_blue_side_prior",
    "blue_first_pick",
    "first_pick_known",
    "blue_first_pick_prior",
    "game_number",
    "series_score_diff",
    "series_game_progress",
    "patch_major",
    "patch_minor",
)
TEAM_STRENGTH_FEATURES = (
    "elo_diff",
    "elo_blue_win_probability",
    "adjusted_elo_diff",
    "team_inactivity_log_days_diff",
    "head_to_head_blue_rate",
    "head_to_head_games",
)
ROSTER_FORM_FEATURES = (
    "player_elo_diff",
    "blue_team_form",
    "red_team_form",
    "team_form_diff",
    "team_games_log_diff",
    "blue_roster_experience",
    "red_roster_experience",
    "roster_experience_diff",
    "blue_roster_continuity",
    "red_roster_continuity",
    "roster_continuity_diff",
    "blue_team_roster_experience",
    "red_team_roster_experience",
    "team_roster_experience_diff",
    "roster_min_games_log_diff",
    *tuple(
        feature_name
        for role in ROLE_NAMES
        for feature_name in (
            f"{role}_player_elo_diff",
            f"{role}_adjusted_player_elo_diff",
        )
    ),
)
CHAMPION_META_FEATURES = (
    "blue_champion_strength",
    "red_champion_strength",
    "champion_strength_diff",
    "blue_draft_known_fraction",
    "red_draft_known_fraction",
    "blue_patch_champion_strength",
    "red_patch_champion_strength",
    "patch_champion_strength_diff",
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
        )
    ),
)
PLAYER_CHAMPION_FEATURES = (
    "blue_player_champion_strength",
    "red_player_champion_strength",
    "player_champion_strength_diff",
    *tuple(
        feature_name
        for role in ROLE_NAMES
        for feature_name in (
            f"blue_{role}_player_champion_strength",
            f"red_{role}_player_champion_strength",
            f"{role}_player_champion_strength_diff",
            f"blue_{role}_player_champion_games_log",
            f"red_{role}_player_champion_games_log",
            f"{role}_player_champion_games_log_diff",
        )
    ),
)
DRAFT_INTERACTION_FEATURES = (
    "blue_ban_strength",
    "red_ban_strength",
    "ban_strength_diff",
    "blue_synergy_strength",
    "red_synergy_strength",
    "synergy_strength_diff",
    "blue_role_matchup_strength",
    "role_matchup_advantage",
    "fearless_ban_count",
    *tuple(
        feature_name
        for role in ROLE_NAMES
        for feature_name in (
            f"{role}_matchup_strength",
            f"{role}_matchup_games_log",
        )
    ),
)
FEATURE_GROUPS: dict[FeatureGroup, tuple[str, ...]] = {
    "core": CORE_FEATURES,
    "team_strength": TEAM_STRENGTH_FEATURES,
    "roster_form": ROSTER_FORM_FEATURES,
    "champion_meta": CHAMPION_META_FEATURES,
    "player_champion": PLAYER_CHAMPION_FEATURES,
    "draft_interactions": DRAFT_INTERACTION_FEATURES,
    "regional": REGIONAL_FEATURE_NAMES,
    "categorical_context": (),
}

BRADLEY_TERRY_CONTEXT_FEATURES = (
    "league",
    "region",
    "tournament_level",
    "blue_team_key",
    "red_team_key",
    *tuple(
        feature_name
        for side in ("blue", "red")
        for role in ROLE_NAMES
        for feature_name in (
            f"{side}_{role}_player",
            f"{side}_{role}_champion",
        )
    ),
)


def _estimator_seed(seed: int) -> int:
    """Map a stable 64-bit trial seed into sklearn and CatBoost's accepted range."""
    return seed % (2**32)


def _deduplicated(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def numeric_feature_names(groups: tuple[FeatureGroup, ...]) -> tuple[str, ...]:
    """Resolve coherent groups into a stable numeric feature contract."""
    values = _deduplicated(
        tuple(feature_name for group in groups for feature_name in FEATURE_GROUPS[group])
    )
    unknown = set(values) - set(FEATURE_NAMES)
    if unknown:
        raise AssertionError(f"Optimization feature groups contain unknown fields: {unknown}")
    if not values:
        raise ValueError("At least one numeric feature group is required")
    return values


@dataclass
class DynamicBradleyTerryClassifier:
    """Regularized signed-identity model with fit-only vocabulary and scaling."""

    numeric_feature_names: tuple[str, ...]
    c: float
    player_weight: float
    champion_weight: float
    random_seed: int
    vectorizer: DictVectorizer = field(init=False)
    model: LogisticRegression = field(init=False)
    numeric_medians: dict[str, float] = field(init=False, default_factory=dict)
    numeric_means: dict[str, float] = field(init=False, default_factory=dict)
    numeric_scales: dict[str, float] = field(init=False, default_factory=dict)
    fit_sample_count: int = field(init=False, default=0)

    def _fit_numeric_statistics(self, frame: pd.DataFrame) -> None:
        for name in self.numeric_feature_names:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if finite.size else 0.0
            filled = np.where(np.isfinite(values), values, median)
            mean = float(np.mean(filled))
            scale = float(np.std(filled))
            self.numeric_medians[name] = median
            self.numeric_means[name] = mean
            self.numeric_scales[name] = scale if scale > 1e-12 else 1.0

    def _row_dictionary(self, row: Any) -> dict[str, float]:
        values: dict[str, float] = {
            f"team={row.blue_team_key}": 1.0,
            f"team={row.red_team_key}": -1.0,
            f"league_blue_side={row.league}": 1.0,
            f"region_blue_side={row.region}": 1.0,
            f"tournament_level_blue_side={row.tournament_level}": 1.0,
        }
        role_scale = 1.0 / len(ROLE_NAMES)
        for role in ROLE_NAMES:
            blue_player = getattr(row, f"blue_{role}_player")
            red_player = getattr(row, f"red_{role}_player")
            blue_champion = getattr(row, f"blue_{role}_champion")
            red_champion = getattr(row, f"red_{role}_champion")
            values[f"player={blue_player}"] = self.player_weight * role_scale
            values[f"player={red_player}"] = -self.player_weight * role_scale
            values[f"champion={blue_champion}"] = self.champion_weight * role_scale
            values[f"champion={red_champion}"] = -self.champion_weight * role_scale
        for name in self.numeric_feature_names:
            raw_value = getattr(row, name)
            value = float(raw_value) if pd.notna(raw_value) else self.numeric_medians[name]
            values[f"numeric={name}"] = (value - self.numeric_means[name]) / self.numeric_scales[
                name
            ]
        return values

    def _dictionaries(self, frame: pd.DataFrame) -> list[dict[str, float]]:
        required = (*BRADLEY_TERRY_CONTEXT_FEATURES, *self.numeric_feature_names)
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"Bradley-Terry input is missing fields: {sorted(missing)}")
        return [
            self._row_dictionary(row)
            for row in frame.loc[:, list(required)].itertuples(index=False)
        ]

    @staticmethod
    def _sklearn_sparse_indexes(matrix: Any) -> Any:
        """Normalize SciPy's platform index width for sklearn's liblinear binding."""
        matrix.indices = np.asarray(matrix.indices, dtype=np.int32)
        matrix.indptr = np.asarray(matrix.indptr, dtype=np.int32)
        return matrix

    def fit(
        self,
        frame: pd.DataFrame,
        labels: NDArray[np.int64],
        *,
        sample_weight: NDArray[np.float64],
    ) -> DynamicBradleyTerryClassifier:
        outcomes = np.asarray(labels, dtype=int)
        if len(frame) != len(outcomes) or len(frame) != len(sample_weight):
            raise ValueError("Bradley-Terry fit arrays must be aligned")
        if len(np.unique(outcomes)) < 2:
            raise ValueError("Bradley-Terry fit interval must contain both outcome classes")
        self._fit_numeric_statistics(frame)
        self.vectorizer = DictVectorizer(sparse=True, sort=True)
        matrix = self._sklearn_sparse_indexes(
            self.vectorizer.fit_transform(self._dictionaries(frame))
        )
        self.model = LogisticRegression(
            C=self.c,
            max_iter=2000,
            random_state=self.random_seed,
            solver="liblinear",
        )
        self.model.fit(matrix, outcomes, sample_weight=sample_weight)
        self.fit_sample_count = len(frame)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        if not hasattr(self, "vectorizer") or not hasattr(self, "model"):
            raise ValueError("Bradley-Terry estimator has not been fitted")
        matrix = self._sklearn_sparse_indexes(self.vectorizer.transform(self._dictionaries(frame)))
        return cast(NDArray[np.float64], self.model.predict_proba(matrix))


def _fit_mapper(
    candidate: CalibratedCandidate,
    calibration: pd.DataFrame,
    spec: TrialSpec,
) -> None:
    if spec.output_mapping == "native":
        return
    if calibration.empty:
        raise ValueError("A learned output mapping requires a calibration interval")
    method: CalibrationMethod = spec.output_mapping
    candidate.calibrator = fit_probability_calibrator(
        candidate.raw_probability(calibration),
        calibration["blue_win"].to_numpy(dtype=int),
        calibration["match_timestamp"],
        method=method,
        random_seed=_estimator_seed(spec.seed),
    )
    candidate.calibration_sample_count = len(calibration) if candidate.calibrator is not None else 0


def _catboost_estimator(
    spec: TrialSpec,
    *,
    categorical_features: list[str],
) -> CatBoostClassifier:
    parameters = spec.model_parameters
    return CatBoostClassifier(
        iterations=int(parameters["iterations"]),
        depth=int(parameters["depth"]),
        learning_rate=float(parameters["learning_rate"]),
        l2_leaf_reg=float(parameters["l2_leaf_regularization"]),
        random_strength=float(parameters["random_strength"]),
        rsm=float(parameters["rsm"]),
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=_estimator_seed(spec.seed),
        has_time=True,
        boosting_type="Plain",
        cat_features=categorical_features or None,
        allow_writing_files=False,
        verbose=False,
        thread_count=1,
    )


def _plain_candidate(
    spec: TrialSpec,
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> CalibratedCandidate:
    labels = fit["blue_win"].to_numpy(dtype=int)
    if len(np.unique(labels)) < 2:
        raise ValueError("Optimization fit interval must contain both outcome classes")
    numeric_names = numeric_feature_names(spec.feature_groups)
    categorical_names = MODEL_CONTEXT_NAMES if "categorical_context" in spec.feature_groups else ()

    estimator: Any
    feature_names: tuple[str, ...]
    fit_arguments: dict[str, Any] = {}
    if spec.family == "logistic_regression":
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(spec.model_parameters["c"]),
                        max_iter=2000,
                        random_state=_estimator_seed(spec.seed),
                    ),
                ),
            ]
        )
        feature_names = numeric_names
    elif spec.family == "histogram_gradient_boosting":
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=float(spec.model_parameters["learning_rate"]),
                        max_iter=int(spec.model_parameters["iterations"]),
                        max_leaf_nodes=int(spec.model_parameters["max_leaf_nodes"]),
                        l2_regularization=float(spec.model_parameters["l2_regularization"]),
                        random_state=_estimator_seed(spec.seed),
                        early_stopping=False,
                    ),
                ),
            ]
        )
        feature_names = numeric_names
    elif spec.family == "catboost":
        feature_names = (*numeric_names, *categorical_names)
        estimator = _catboost_estimator(
            spec,
            categorical_features=list(categorical_names),
        )
    elif spec.family == "dynamic_bradley_terry":
        feature_names = _deduplicated((*numeric_names, *BRADLEY_TERRY_CONTEXT_FEATURES))
        estimator = DynamicBradleyTerryClassifier(
            numeric_feature_names=numeric_names,
            c=float(spec.model_parameters["c"]),
            player_weight=float(spec.model_parameters["player_weight"]),
            champion_weight=float(spec.model_parameters["champion_weight"]),
            random_seed=_estimator_seed(spec.seed),
        )
        fit_arguments["sample_weight"] = recency_sample_weights(
            fit,
            half_life_days=float(spec.model_parameters["recency_half_life_days"]),
        )
    else:
        raise ValueError(f"Family does not define a plain candidate: {spec.family}")

    estimator.fit(
        fit.loc[:, list(feature_names)],
        labels,
        **fit_arguments,
    )
    candidate = CalibratedCandidate(
        name=f"v5_{spec.family}_{spec.spec_hash[:12]}",
        feature_names=feature_names,
        estimator=estimator,
        calibrator=None,
        fit_sample_count=len(fit),
        calibration_sample_count=0,
    )
    _fit_mapper(candidate, calibration, spec)
    return candidate


def _elo_catboost_blend(
    spec: TrialSpec,
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    settings: ExperimentSettings,
) -> BlendedCandidate:
    labels = fit["blue_win"].to_numpy(dtype=int)
    numeric_names = numeric_feature_names(spec.feature_groups)
    categorical_names = MODEL_CONTEXT_NAMES if "categorical_context" in spec.feature_groups else ()
    elo_features = ("elo_diff",)
    elo_estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=settings.models.logistic_c,
                    max_iter=2000,
                    random_state=_estimator_seed(spec.seed),
                ),
            ),
        ]
    )
    elo_estimator.fit(fit.loc[:, list(elo_features)], labels)
    elo = CalibratedCandidate(
        name=f"v5_elo_raw_{spec.spec_hash[:12]}",
        feature_names=elo_features,
        estimator=elo_estimator,
        calibrator=None,
        fit_sample_count=len(fit),
        calibration_sample_count=0,
    )

    catboost_features = (*numeric_names, *categorical_names)
    catboost_estimator = _catboost_estimator(
        spec,
        categorical_features=list(categorical_names),
    )
    catboost_estimator.fit(fit.loc[:, list(catboost_features)], labels)
    catboost = CalibratedCandidate(
        name=f"v5_catboost_raw_{spec.spec_hash[:12]}",
        feature_names=catboost_features,
        estimator=catboost_estimator,
        calibrator=None,
        fit_sample_count=len(fit),
        calibration_sample_count=0,
    )
    elo_weight = float(spec.model_parameters["elo_weight"])
    candidate = BlendedCandidate(
        name=f"v5_{spec.family}_{spec.spec_hash[:12]}",
        feature_names=_deduplicated((*elo_features, *catboost_features)),
        estimator=None,
        calibrator=None,
        fit_sample_count=len(fit),
        calibration_sample_count=0,
        components=(elo, catboost),
        weights=(elo_weight, 1.0 - elo_weight),
    )
    _fit_mapper(candidate, calibration, spec)
    return candidate


def fit_optimization_candidate(
    spec: TrialSpec,
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    settings: ExperimentSettings,
) -> CalibratedCandidate:
    """Fit one frozen trial without observing any scoring interval."""
    if spec.family == "elo_catboost_blend":
        return _elo_catboost_blend(spec, fit, calibration, settings)
    return _plain_candidate(spec, fit, calibration)
