"""Fixed v6 challengers built on the shared point-in-time feature frame."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lolpredictor.calibration import fit_probability_calibrator
from lolpredictor.features import (
    REGIONAL_FEATURE_NAMES,
    STATE_SPACE_FEATURE_NAMES,
)
from lolpredictor.models import (
    BlendedCandidate,
    CalibratedCandidate,
    fit_candidate,
)
from lolpredictor.optimization.models import (
    CORE_FEATURES,
    DRAFT_INTERACTION_FEATURES,
    ROSTER_FORM_FEATURES,
    TEAM_STRENGTH_FEATURES,
)
from lolpredictor.optimization.splits import trailing_fit_calibration_split
from lolpredictor.state_space.settings import (
    V4_CONTROL_NAME,
    V6CandidateName,
    V6Configuration,
)

STATE_SPACE_PROBABILITY_FEATURE = "state_space_blue_win_probability"
STATE_SPACE_AUGMENTED_FEATURES = tuple(
    dict.fromkeys(
        (
            *CORE_FEATURES,
            *TEAM_STRENGTH_FEATURES,
            *ROSTER_FORM_FEATURES,
            *DRAFT_INTERACTION_FEATURES,
            *REGIONAL_FEATURE_NAMES,
            *STATE_SPACE_FEATURE_NAMES,
        )
    )
)


class ColumnProbabilityEstimator:
    """Use one precomputed point-in-time probability as an estimator."""

    def __init__(self, column: str) -> None:
        self.column = column

    def predict_proba(self, features: pd.DataFrame) -> NDArray[np.float64]:
        if tuple(features.columns) != (self.column,):
            raise ValueError("Column-probability input does not match its contract")
        positive = np.clip(
            pd.to_numeric(features[self.column], errors="raise").to_numpy(dtype=float),
            1e-6,
            1.0 - 1e-6,
        )
        return cast(
            NDArray[np.float64],
            np.column_stack([1.0 - positive, positive]),
        )


def _native_state_candidate(
    history: pd.DataFrame,
    *,
    name: str,
) -> CalibratedCandidate:
    return CalibratedCandidate(
        name=name,
        feature_names=(STATE_SPACE_PROBABILITY_FEATURE,),
        estimator=ColumnProbabilityEstimator(STATE_SPACE_PROBABILITY_FEATURE),
        calibrator=None,
        fit_sample_count=len(history),
        calibration_sample_count=0,
    )


def _platt_state_candidate(
    history: pd.DataFrame,
    configuration: V6Configuration,
) -> CalibratedCandidate:
    fit, calibration = trailing_fit_calibration_split(
        history,
        requires_calibration=True,
        calibration_fraction=(configuration.study.candidates.state_calibration_fraction),
    )
    candidate = _native_state_candidate(fit, name="state_space_platt")
    candidate.calibrator = fit_probability_calibrator(
        candidate.raw_probability(calibration),
        calibration["blue_win"].to_numpy(dtype=int),
        calibration["match_timestamp"],
        method="platt",
        random_seed=configuration.study.random_seed,
    )
    candidate.calibration_sample_count = len(calibration) if candidate.calibrator is not None else 0
    return candidate


def _augmented_logistic_candidate(
    history: pd.DataFrame,
    configuration: V6Configuration,
) -> CalibratedCandidate:
    labels = history["blue_win"].to_numpy(dtype=int)
    if len(np.unique(labels)) < 2:
        raise ValueError("V6 fit history must contain both outcome classes")
    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=configuration.study.candidates.augmented_logistic_c,
                    max_iter=2000,
                    random_state=configuration.study.random_seed,
                ),
            ),
        ]
    )
    estimator.fit(
        history.loc[:, list(STATE_SPACE_AUGMENTED_FEATURES)],
        labels,
    )
    return CalibratedCandidate(
        name="state_space_augmented_logistic",
        feature_names=STATE_SPACE_AUGMENTED_FEATURES,
        estimator=estimator,
        calibrator=None,
        fit_sample_count=len(history),
        calibration_sample_count=0,
    )


def fit_v4_control(
    history: pd.DataFrame,
    configuration: V6Configuration,
) -> CalibratedCandidate:
    """Fit the behaviorally frozen v4 control on one history interval."""
    ordered = history.sort_values(["match_timestamp", "match_id"]).reset_index(drop=True)
    return fit_candidate(
        V4_CONTROL_NAME,
        ordered,
        ordered.iloc[0:0].copy(),
        configuration.experiment,
    )


def fit_v6_candidates(
    history: pd.DataFrame,
    configuration: V6Configuration,
) -> dict[str, CalibratedCandidate]:
    """Fit the complete frozen candidate set once for one history interval."""
    missing = {
        STATE_SPACE_PROBABILITY_FEATURE,
        *STATE_SPACE_AUGMENTED_FEATURES,
        "blue_win",
        "match_timestamp",
    } - set(history.columns)
    if missing:
        raise ValueError(f"V6 fit history is missing fields: {sorted(missing)}")
    ordered = history.sort_values(["match_timestamp", "match_id"]).reset_index(drop=True)
    native = _native_state_candidate(ordered, name="state_space_native")
    platt = _platt_state_candidate(ordered, configuration)
    augmented = _augmented_logistic_candidate(ordered, configuration)
    v4 = fit_v4_control(ordered, configuration)
    state_weight_15 = configuration.study.candidates.blend_15_state_weight
    state_weight_30 = configuration.study.candidates.blend_30_state_weight
    blend_features = tuple(dict.fromkeys((*v4.feature_names, *augmented.feature_names)))
    blend_15 = BlendedCandidate(
        name="state_space_v4_blend_15",
        feature_names=blend_features,
        estimator=None,
        calibrator=None,
        fit_sample_count=len(ordered),
        calibration_sample_count=0,
        components=(v4, augmented),
        weights=(1.0 - state_weight_15, state_weight_15),
    )
    blend_30 = BlendedCandidate(
        name="state_space_v4_blend_30",
        feature_names=blend_features,
        estimator=None,
        calibrator=None,
        fit_sample_count=len(ordered),
        calibration_sample_count=0,
        components=(v4, augmented),
        weights=(1.0 - state_weight_30, state_weight_30),
    )
    candidates: dict[str, CalibratedCandidate] = {
        native.name: native,
        platt.name: platt,
        augmented.name: augmented,
        blend_15.name: blend_15,
        blend_30.name: blend_30,
    }
    if tuple(candidates) != configuration.study.candidates.names:
        raise AssertionError("Fitted v6 candidates do not match the frozen ordering")
    return candidates


def fit_v6_candidate(
    name: V6CandidateName,
    history: pd.DataFrame,
    configuration: V6Configuration,
) -> CalibratedCandidate:
    """Fit one named challenger using the same shared candidate construction."""
    return fit_v6_candidates(history, configuration)[name]
