"""Inner-trial and outer-policy evaluation without outcome reuse."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from lolpredictor.evaluation import evaluate_probabilities
from lolpredictor.models import candidate_requires_calibration, fit_candidate
from lolpredictor.optimization.models import fit_optimization_candidate
from lolpredictor.optimization.schedule import TrialSpec
from lolpredictor.optimization.settings import OptimizationConfiguration
from lolpredictor.optimization.splits import (
    OptimizationInnerFold,
    OptimizationOuterFold,
    build_nested_folds,
    frame_summary,
    trailing_fit_calibration_split,
)

CONTROL_NAMES = (
    "elo_only",
    "team_roster_logistic",
    "elo_catboost_regional_raw_blend_50",
)
TRIAL_RESULT_SCHEMA_VERSION = "1"


def _prediction_fingerprint(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for match_id, probability in zip(
        frame["match_id"].astype(str),
        probabilities,
        strict=True,
    ):
        digest.update(f"{match_id}\x1f{float(probability):.17g}\n".encode())
    return digest.hexdigest()


def _trial_fold_result(
    spec: TrialSpec,
    fold: OptimizationInnerFold,
    configuration: OptimizationConfiguration,
) -> tuple[dict[str, Any], np.ndarray]:
    fit, calibration = trailing_fit_calibration_split(
        fold.history,
        requires_calibration=spec.output_mapping != "native",
        calibration_fraction=(
            configuration.optimization.nested_validation.calibration_fraction_within_train
        ),
    )
    candidate = fit_optimization_candidate(
        spec,
        fit,
        calibration,
        configuration.experiment,
    )
    probabilities = candidate.predict_probability(fold.score)
    metrics = evaluate_probabilities(
        fold.score,
        probabilities,
        include_breakdowns=False,
    )
    return (
        {
            "fold_number": fold.fold_number,
            "fit": frame_summary(fit),
            "calibration": frame_summary(calibration) if not calibration.empty else None,
            "score": frame_summary(fold.score),
            "metrics": metrics,
            "calibration_summary": candidate.calibration_summary(),
            "prediction_fingerprint": _prediction_fingerprint(
                fold.score,
                probabilities,
            ),
        },
        probabilities,
    )


def evaluate_trial(
    spec: TrialSpec,
    frame: pd.DataFrame,
    configuration: OptimizationConfiguration,
) -> dict[str, Any]:
    """Score one specification only on its three inner chronological folds."""
    outer_folds = build_nested_folds(
        frame,
        configuration.optimization.nested_validation,
    )
    outer = outer_folds[spec.outer_fold - 1]
    fold_results: list[dict[str, Any]] = []
    score_frames: list[pd.DataFrame] = []
    probabilities: list[np.ndarray] = []
    for fold in outer.inner_folds:
        fold_result, fold_probabilities = _trial_fold_result(
            spec,
            fold,
            configuration,
        )
        fold_results.append(fold_result)
        score_frames.append(fold.score)
        probabilities.append(fold_probabilities)
    pooled_frame = pd.concat(score_frames, ignore_index=True)
    pooled_probabilities = np.concatenate(probabilities)
    return {
        "schema_version": TRIAL_RESULT_SCHEMA_VERSION,
        "trial_id": spec.trial_id,
        "spec_hash": spec.spec_hash,
        "outer_fold": spec.outer_fold,
        "family": spec.family,
        "feature_mode": spec.feature_mode,
        "output_mapping": spec.output_mapping,
        "inner_folds": fold_results,
        "pooled_inner": evaluate_probabilities(
            pooled_frame,
            pooled_probabilities,
            include_breakdowns=True,
        ),
        "pooled_prediction_fingerprint": _prediction_fingerprint(
            pooled_frame,
            pooled_probabilities,
        ),
    }


def _fit_control_on_history(
    name: str,
    history: pd.DataFrame,
    configuration: OptimizationConfiguration,
) -> Any:
    fit, calibration = trailing_fit_calibration_split(
        history,
        requires_calibration=candidate_requires_calibration(name),
        calibration_fraction=(
            configuration.optimization.nested_validation.calibration_fraction_within_train
        ),
    )
    return fit_candidate(
        name,
        fit,
        calibration,
        configuration.experiment,
    )


def evaluate_inner_controls(
    frame: pd.DataFrame,
    configuration: OptimizationConfiguration,
) -> dict[str, Any]:
    """Evaluate fixed controls once per inner fold for later eligibility checks."""
    outer_folds = build_nested_folds(
        frame,
        configuration.optimization.nested_validation,
    )
    outer_reports: dict[str, Any] = {}
    for outer in outer_folds:
        control_reports: dict[str, Any] = {}
        for name in CONTROL_NAMES:
            score_frames: list[pd.DataFrame] = []
            probabilities: list[np.ndarray] = []
            fold_reports: list[dict[str, Any]] = []
            for fold in outer.inner_folds:
                candidate = _fit_control_on_history(name, fold.history, configuration)
                fold_probabilities = candidate.predict_probability(fold.score)
                fold_metrics = evaluate_probabilities(
                    fold.score,
                    fold_probabilities,
                    include_breakdowns=False,
                )
                fold_reports.append(
                    {
                        "fold_number": fold.fold_number,
                        "metrics": fold_metrics,
                        "prediction_fingerprint": _prediction_fingerprint(
                            fold.score,
                            fold_probabilities,
                        ),
                    }
                )
                score_frames.append(fold.score)
                probabilities.append(fold_probabilities)
            pooled_frame = pd.concat(score_frames, ignore_index=True)
            pooled_probabilities = np.concatenate(probabilities)
            control_reports[name] = {
                "inner_folds": fold_reports,
                "pooled_inner": evaluate_probabilities(
                    pooled_frame,
                    pooled_probabilities,
                    include_breakdowns=True,
                ),
                "pooled_prediction_fingerprint": _prediction_fingerprint(
                    pooled_frame,
                    pooled_probabilities,
                ),
            }
        outer_reports[str(outer.fold_number)] = control_reports
    return {
        "schema_version": "1",
        "control_names": list(CONTROL_NAMES),
        "outer_folds": outer_reports,
    }


def evaluate_outer_spec(
    spec: TrialSpec,
    frame: pd.DataFrame,
    configuration: OptimizationConfiguration,
) -> dict[str, Any]:
    """Refit an inner-selected trial and touch its outer score interval once."""
    outer_folds = build_nested_folds(
        frame,
        configuration.optimization.nested_validation,
    )
    outer = outer_folds[spec.outer_fold - 1]
    fit, calibration = trailing_fit_calibration_split(
        outer.history,
        requires_calibration=spec.output_mapping != "native",
        calibration_fraction=(
            configuration.optimization.nested_validation.calibration_fraction_within_train
        ),
    )
    candidate = fit_optimization_candidate(
        spec,
        fit,
        calibration,
        configuration.experiment,
    )
    probabilities = candidate.predict_probability(outer.score)
    return {
        "schema_version": "1",
        "trial_id": spec.trial_id,
        "spec_hash": spec.spec_hash,
        "outer_fold": spec.outer_fold,
        "family": spec.family,
        "feature_mode": spec.feature_mode,
        "fit": frame_summary(fit),
        "calibration": frame_summary(calibration) if not calibration.empty else None,
        "score": frame_summary(outer.score),
        "metrics": evaluate_probabilities(
            outer.score,
            probabilities,
            include_breakdowns=True,
        ),
        "calibration_summary": candidate.calibration_summary(),
        "match_ids": outer.score["match_id"].astype(str).tolist(),
        "probabilities": probabilities.astype(float).tolist(),
        "prediction_fingerprint": _prediction_fingerprint(
            outer.score,
            probabilities,
        ),
    }


def evaluate_outer_control(
    name: str,
    outer: OptimizationOuterFold,
    configuration: OptimizationConfiguration,
) -> dict[str, Any]:
    candidate = _fit_control_on_history(name, outer.history, configuration)
    probabilities = candidate.predict_probability(outer.score)
    return {
        "name": name,
        "outer_fold": outer.fold_number,
        "metrics": evaluate_probabilities(
            outer.score,
            probabilities,
            include_breakdowns=True,
        ),
        "match_ids": outer.score["match_id"].astype(str).tolist(),
        "probabilities": probabilities.astype(float).tolist(),
        "prediction_fingerprint": _prediction_fingerprint(
            outer.score,
            probabilities,
        ),
    }


def controls_fingerprint(controls: dict[str, Any]) -> str:
    encoded = json.dumps(
        controls,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
