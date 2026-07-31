"""Chronology-compatible probability calibration primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CalibrationMethod = Literal["platt", "beta", "isotonic"]
CALIBRATION_METHODS: tuple[CalibrationMethod, ...] = (
    "platt",
    "beta",
    "isotonic",
)
PROBABILITY_EPSILON = 1e-6


def clip_probabilities(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return finite probabilities bounded away from log-loss singularities."""
    values = np.asarray(probabilities, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Probability calibration received a non-finite value")
    return cast(
        NDArray[np.float64],
        np.clip(values, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON),
    )


def _platt_features(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = clip_probabilities(probabilities)
    return cast(
        NDArray[np.float64],
        np.log(clipped / (1.0 - clipped)).reshape(-1, 1),
    )


def _beta_features(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = clip_probabilities(probabilities)
    return cast(
        NDArray[np.float64],
        np.column_stack(
            [
                np.log(clipped),
                -np.log1p(-clipped),
            ]
        ),
    )


@dataclass(frozen=True)
class BetaCalibrationEstimator:
    """Monotone beta calibrator fitted by bounded maximum likelihood."""

    coefficients: tuple[float, float]
    intercept: float

    @classmethod
    def fit(
        cls,
        probabilities: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> BetaCalibrationEstimator:
        features = _beta_features(probabilities)
        outcomes = np.asarray(labels, dtype=float)

        def objective(parameters: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            scores = features @ parameters[:2] + parameters[2]
            losses = np.logaddexp(0.0, scores) - outcomes * scores
            errors = expit(scores) - outcomes
            gradient = np.array(
                [
                    np.mean(errors * features[:, 0]),
                    np.mean(errors * features[:, 1]),
                    np.mean(errors),
                ],
                dtype=float,
            )
            return float(np.mean(losses)), gradient

        result = minimize(
            objective,
            np.array([1.0, 1.0, 0.0], dtype=float),
            method="L-BFGS-B",
            jac=True,
            bounds=((0.0, None), (0.0, None), (None, None)),
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 2000},
        )
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"Beta calibration optimization failed: {result.message}")
        return cls(
            coefficients=(float(result.x[0]), float(result.x[1])),
            intercept=float(result.x[2]),
        )

    def predict(self, probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
        features = _beta_features(probabilities)
        scores = (
            features[:, 0] * self.coefficients[0]
            + features[:, 1] * self.coefficients[1]
            + self.intercept
        )
        return cast(NDArray[np.float64], expit(scores))


@dataclass(frozen=True)
class ProbabilityCalibrator:
    """Serializable mapper plus the exact chronology used to fit it."""

    method: CalibrationMethod
    estimator: LogisticRegression | BetaCalibrationEstimator | IsotonicRegression
    sample_count: int
    positive_count: int
    fit_start_timestamp: str
    fit_end_timestamp: str
    raw_probability_min: float
    raw_probability_max: float

    def predict(self, probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
        clipped = clip_probabilities(probabilities)
        if self.method == "platt":
            calibrated = cast(
                LogisticRegression,
                self.estimator,
            ).predict_proba(_platt_features(clipped))[:, 1]
        elif self.method == "beta":
            calibrated = cast(
                BetaCalibrationEstimator,
                self.estimator,
            ).predict(clipped)
        else:
            calibrated = cast(
                IsotonicRegression,
                self.estimator,
            ).predict(clipped)
        return clip_probabilities(np.asarray(calibrated, dtype=float))

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "estimator": type(self.estimator).__name__,
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "negative_count": self.sample_count - self.positive_count,
            "fit_start_timestamp": self.fit_start_timestamp,
            "fit_end_timestamp": self.fit_end_timestamp,
            "raw_probability_min": self.raw_probability_min,
            "raw_probability_max": self.raw_probability_max,
        }


def fit_probability_calibrator(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
    timestamps: pd.Series,
    *,
    method: CalibrationMethod,
    random_seed: int,
) -> ProbabilityCalibrator | None:
    """Fit a mapping on an already-separated trailing calibration interval."""
    raw = clip_probabilities(np.asarray(probabilities, dtype=float))
    outcomes = np.asarray(labels, dtype=int)
    if raw.ndim != 1 or outcomes.ndim != 1 or len(raw) != len(outcomes):
        raise ValueError("Calibration probabilities and labels must be aligned vectors")
    if len(raw) != len(timestamps):
        raise ValueError("Calibration timestamps must align with probabilities")
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"Unsupported probability calibration method: {method}")
    if len(np.unique(outcomes)) < 2 or float(np.std(raw)) < 1e-9:
        return None

    normalized_timestamps = pd.to_datetime(timestamps, utc=True)
    if normalized_timestamps.isna().any():
        raise ValueError("Calibration timestamps cannot be missing")

    estimator: LogisticRegression | BetaCalibrationEstimator | IsotonicRegression
    if method == "platt":
        estimator = LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=random_seed,
        )
        estimator.fit(_platt_features(raw), outcomes)
    elif method == "beta":
        estimator = BetaCalibrationEstimator.fit(raw, outcomes)
    else:
        estimator = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip",
            y_min=0.0,
            y_max=1.0,
        )
        estimator.fit(raw, outcomes)

    return ProbabilityCalibrator(
        method=method,
        estimator=estimator,
        sample_count=len(outcomes),
        positive_count=int(np.sum(outcomes)),
        fit_start_timestamp=normalized_timestamps.min().isoformat(),
        fit_end_timestamp=normalized_timestamps.max().isoformat(),
        raw_probability_min=float(np.min(raw)),
        raw_probability_max=float(np.max(raw)),
    )
