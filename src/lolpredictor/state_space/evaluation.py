"""Nested chronological evaluation for the fixed v6 candidate policy."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from lolpredictor.evaluation import (
    clustered_log_loss_interval,
    clustered_paired_log_loss_difference_interval,
    evaluate_probabilities,
)
from lolpredictor.models import (
    BlendedCandidate,
    CalibratedCandidate,
    candidate_requires_calibration,
    fit_candidate,
)
from lolpredictor.optimization.splits import (
    OptimizationOuterFold,
    build_nested_folds,
    frame_summary,
    trailing_fit_calibration_split,
)
from lolpredictor.state_space.models import fit_v6_candidates
from lolpredictor.state_space.settings import (
    V4_CONTROL_NAME,
    V6CandidateName,
    V6Configuration,
)

CONTROL_NAMES = (
    V4_CONTROL_NAME,
    "elo_only",
    "team_roster_logistic",
)


def prediction_fingerprint(
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


def _fit_simple_control(
    name: str,
    history: pd.DataFrame,
    configuration: V6Configuration,
) -> CalibratedCandidate:
    fit, calibration = trailing_fit_calibration_split(
        history,
        requires_calibration=candidate_requires_calibration(name),
        calibration_fraction=(
            configuration.study.nested_validation.calibration_fraction_within_train
        ),
    )
    return fit_candidate(
        name,
        fit,
        calibration,
        configuration.experiment,
    )


def _v4_from_candidates(
    candidates: dict[str, CalibratedCandidate],
) -> CalibratedCandidate:
    blend = candidates["state_space_v4_blend_15"]
    if not isinstance(blend, BlendedCandidate):
        raise TypeError("The frozen v6 blend has an unexpected candidate type")
    v4 = blend.components[0]
    if v4.name != V4_CONTROL_NAME:
        raise ValueError("The frozen v6 blend does not contain the fixed v4 control")
    return v4


def _score_model_set(
    candidates: dict[str, CalibratedCandidate],
    controls: dict[str, CalibratedCandidate],
    score: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return (
        {name: candidate.predict_probability(score) for name, candidate in candidates.items()},
        {name: candidate.predict_probability(score) for name, candidate in controls.items()},
    )


def _major_league_differences(
    candidate_metrics: dict[str, Any],
    v4_metrics: dict[str, Any],
    configuration: V6Configuration,
) -> dict[str, dict[str, float | int]]:
    differences: dict[str, dict[str, float | int]] = {}
    candidate_leagues = candidate_metrics["breakdowns"]["league"]
    v4_leagues = v4_metrics["breakdowns"]["league"]
    minimum_count = configuration.study.selection.minimum_breakdown_sample_count
    for league in configuration.study.decision_gate.major_leagues:
        candidate = candidate_leagues.get(league)
        control = v4_leagues.get(league)
        if candidate is None or control is None:
            continue
        sample_count = int(candidate["sample_count"])
        if sample_count < minimum_count:
            continue
        differences[league] = {
            "sample_count": sample_count,
            "log_loss_difference_vs_v4": (
                float(candidate["log_loss"]) - float(control["log_loss"])
            ),
        }
    return differences


def _candidate_eligibility(
    name: str,
    candidate_metrics: dict[str, Any],
    candidate_fold_metrics: list[dict[str, Any]],
    control_metrics: dict[str, dict[str, Any]],
    control_fold_metrics: dict[str, list[dict[str, Any]]],
    configuration: V6Configuration,
) -> dict[str, Any]:
    v4_metrics = control_metrics[V4_CONTROL_NAME]
    elo_metrics = control_metrics["elo_only"]
    v4_folds = control_fold_metrics[V4_CONTROL_NAME]
    fold_differences = [
        {
            "fold_number": fold_index + 1,
            "log_loss_difference_vs_v4": (
                float(candidate_fold["log_loss"]) - float(v4_fold["log_loss"])
            ),
        }
        for fold_index, (candidate_fold, v4_fold) in enumerate(
            zip(candidate_fold_metrics, v4_folds, strict=True)
        )
    ]
    league_differences = _major_league_differences(
        candidate_metrics,
        v4_metrics,
        configuration,
    )
    selection = configuration.study.selection
    checks = {
        "beats_v4": (float(candidate_metrics["log_loss"]) < float(v4_metrics["log_loss"])),
        "beats_elo": (float(candidate_metrics["log_loss"]) < float(elo_metrics["log_loss"])),
        "calibration_within_limit": (
            float(candidate_metrics["expected_calibration_error"])
            <= selection.maximum_expected_calibration_error
        ),
        "inner_fold_regressions_within_limit": all(
            float(item["log_loss_difference_vs_v4"])
            <= selection.maximum_inner_fold_log_loss_regression_vs_v4
            for item in fold_differences
        ),
        "major_league_regressions_within_limit": all(
            float(item["log_loss_difference_vs_v4"])
            <= selection.maximum_major_league_log_loss_regression_vs_v4
            for item in league_differences.values()
        ),
    }
    return {
        "candidate": name,
        "eligible": all(checks.values()),
        "checks": checks,
        "fold_log_loss_differences_vs_v4": fold_differences,
        "major_league_log_loss_differences_vs_v4": league_differences,
    }


def evaluate_inner_policy(
    outer: OptimizationOuterFold,
    configuration: V6Configuration,
) -> tuple[dict[str, Any], str]:
    """Select a challenger using only one outer fold's inner evidence."""
    candidate_probabilities: dict[str, list[np.ndarray]] = {
        name: [] for name in configuration.study.candidates.names
    }
    control_probabilities: dict[str, list[np.ndarray]] = {name: [] for name in CONTROL_NAMES}
    candidate_fold_metrics: dict[str, list[dict[str, Any]]] = {
        name: [] for name in configuration.study.candidates.names
    }
    control_fold_metrics: dict[str, list[dict[str, Any]]] = {name: [] for name in CONTROL_NAMES}
    score_frames: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []

    for fold in outer.inner_folds:
        candidates = fit_v6_candidates(fold.history, configuration)
        controls = {
            V4_CONTROL_NAME: _v4_from_candidates(candidates),
            "elo_only": _fit_simple_control(
                "elo_only",
                fold.history,
                configuration,
            ),
            "team_roster_logistic": _fit_simple_control(
                "team_roster_logistic",
                fold.history,
                configuration,
            ),
        }
        candidate_scores, control_scores = _score_model_set(
            candidates,
            controls,
            fold.score,
        )
        for name, probabilities in candidate_scores.items():
            candidate_probabilities[name].append(probabilities)
            candidate_fold_metrics[name].append(
                evaluate_probabilities(
                    fold.score,
                    probabilities,
                    include_breakdowns=False,
                )
            )
        for name, probabilities in control_scores.items():
            control_probabilities[name].append(probabilities)
            control_fold_metrics[name].append(
                evaluate_probabilities(
                    fold.score,
                    probabilities,
                    include_breakdowns=False,
                )
            )
        fold_summaries.append(
            {
                "fold_number": fold.fold_number,
                "history": frame_summary(fold.history),
                "score": frame_summary(fold.score),
                "candidates": {
                    name: {
                        "metrics": candidate_fold_metrics[name][-1],
                        "prediction_fingerprint": prediction_fingerprint(
                            fold.score,
                            candidate_scores[name],
                        ),
                    }
                    for name in configuration.study.candidates.names
                },
                "controls": {
                    name: {
                        "metrics": control_fold_metrics[name][-1],
                        "prediction_fingerprint": prediction_fingerprint(
                            fold.score,
                            control_scores[name],
                        ),
                    }
                    for name in CONTROL_NAMES
                },
            }
        )
        score_frames.append(fold.score)

    pooled_frame = pd.concat(score_frames, ignore_index=True)
    pooled_candidate_metrics = {
        name: evaluate_probabilities(
            pooled_frame,
            np.concatenate(candidate_probabilities[name]),
            include_breakdowns=True,
        )
        for name in configuration.study.candidates.names
    }
    pooled_control_metrics = {
        name: evaluate_probabilities(
            pooled_frame,
            np.concatenate(control_probabilities[name]),
            include_breakdowns=True,
        )
        for name in CONTROL_NAMES
    }
    eligibility = {
        name: _candidate_eligibility(
            name,
            pooled_candidate_metrics[name],
            candidate_fold_metrics[name],
            pooled_control_metrics,
            control_fold_metrics,
            configuration,
        )
        for name in configuration.study.candidates.names
    }
    eligible_names: list[V6CandidateName] = [
        name for name in configuration.study.candidates.names if eligibility[name]["eligible"]
    ]

    def candidate_rank(name: V6CandidateName) -> tuple[float, int]:
        return (
            float(pooled_candidate_metrics[name]["log_loss"]),
            configuration.study.candidates.names.index(name),
        )

    selected: str
    if eligible_names:
        selected = min(
            eligible_names,
            key=candidate_rank,
        )
    else:
        selected = configuration.study.selection.fallback_candidate
    return (
        {
            "outer_fold": outer.fold_number,
            "folds": fold_summaries,
            "pooled_score": frame_summary(pooled_frame),
            "candidates": {
                name: {
                    "metrics": pooled_candidate_metrics[name],
                    "eligibility": eligibility[name],
                    "prediction_fingerprint": prediction_fingerprint(
                        pooled_frame,
                        np.concatenate(candidate_probabilities[name]),
                    ),
                }
                for name in configuration.study.candidates.names
            },
            "controls": {
                name: {
                    "metrics": pooled_control_metrics[name],
                    "prediction_fingerprint": prediction_fingerprint(
                        pooled_frame,
                        np.concatenate(control_probabilities[name]),
                    ),
                }
                for name in CONTROL_NAMES
            },
            "selected_candidate": selected,
            "fallback_used": selected == V4_CONTROL_NAME,
        },
        selected,
    )


def _evaluate_outer_selection(
    outer: OptimizationOuterFold,
    selected_name: str,
    configuration: V6Configuration,
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    candidates = fit_v6_candidates(outer.history, configuration)
    controls = {
        V4_CONTROL_NAME: _v4_from_candidates(candidates),
        "elo_only": _fit_simple_control(
            "elo_only",
            outer.history,
            configuration,
        ),
        "team_roster_logistic": _fit_simple_control(
            "team_roster_logistic",
            outer.history,
            configuration,
        ),
    }
    selected = (
        controls[V4_CONTROL_NAME] if selected_name == V4_CONTROL_NAME else candidates[selected_name]
    )
    selected_probabilities = selected.predict_probability(outer.score)
    control_probabilities = {
        name: control.predict_probability(outer.score) for name, control in controls.items()
    }
    return (
        {
            "outer_fold": outer.fold_number,
            "selected_candidate": selected_name,
            "fallback_used": selected_name == V4_CONTROL_NAME,
            "history": frame_summary(outer.history),
            "score": frame_summary(outer.score),
            "selected_metrics": evaluate_probabilities(
                outer.score,
                selected_probabilities,
                include_breakdowns=True,
            ),
            "selected_prediction_fingerprint": prediction_fingerprint(
                outer.score,
                selected_probabilities,
            ),
            "controls": {
                name: {
                    "metrics": evaluate_probabilities(
                        outer.score,
                        probabilities,
                        include_breakdowns=True,
                    ),
                    "prediction_fingerprint": prediction_fingerprint(
                        outer.score,
                        probabilities,
                    ),
                }
                for name, probabilities in control_probabilities.items()
            },
            "match_ids": outer.score["match_id"].astype(str).tolist(),
            "selected_probabilities": selected_probabilities.astype(float).tolist(),
            "control_probabilities": {
                name: probabilities.astype(float).tolist()
                for name, probabilities in control_probabilities.items()
            },
        },
        selected_probabilities,
        control_probabilities,
    )


def _cluster_values(frame: pd.DataFrame) -> np.ndarray:
    if "series_id" not in frame:
        return frame["match_id"].astype(str).to_numpy()
    return frame["series_id"].fillna(frame["match_id"]).astype(str).to_numpy()


def _decision_gate(
    pooled_frame: pd.DataFrame,
    selected_metrics: dict[str, Any],
    v4_metrics: dict[str, Any],
    selected_probabilities: np.ndarray,
    v4_probabilities: np.ndarray,
    paired_interval: dict[str, float | int] | None,
    outer_reports: list[dict[str, Any]],
    configuration: V6Configuration,
) -> dict[str, Any]:
    gate = configuration.study.decision_gate
    series_count = len(np.unique(_cluster_values(pooled_frame)))
    fold_differences = [
        {
            "fold_number": report["outer_fold"],
            "log_loss_difference_vs_v4": (
                float(report["selected_metrics"]["log_loss"])
                - float(report["controls"][V4_CONTROL_NAME]["metrics"]["log_loss"])
            ),
        }
        for report in outer_reports
    ]
    league_differences = _major_league_differences(
        selected_metrics,
        v4_metrics,
        configuration,
    )
    improvement = float(v4_metrics["log_loss"]) - float(selected_metrics["log_loss"])
    checks = {
        "minimum_games": len(pooled_frame) >= gate.minimum_outer_games,
        "minimum_series": series_count >= gate.minimum_outer_series,
        "minimum_log_loss_improvement_vs_v4": (
            improvement >= gate.minimum_log_loss_improvement_vs_v4
        ),
        "paired_interval_upper_below_zero": (
            paired_interval is not None and float(paired_interval["upper"]) < 0.0
        ),
        "calibration_within_limit": (
            float(selected_metrics["expected_calibration_error"])
            <= gate.maximum_expected_calibration_error
        ),
        "outer_fold_regressions_within_limit": all(
            float(item["log_loss_difference_vs_v4"])
            <= gate.maximum_outer_fold_log_loss_regression_vs_v4
            for item in fold_differences
        ),
        "major_league_regressions_within_limit": all(
            float(item["log_loss_difference_vs_v4"])
            <= gate.maximum_major_league_log_loss_regression_vs_v4
            for item in league_differences.values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "sample_count": len(pooled_frame),
        "series_count": series_count,
        "log_loss_improvement_vs_v4": improvement,
        "paired_log_loss_difference_vs_v4_interval": paired_interval,
        "outer_fold_log_loss_differences_vs_v4": fold_differences,
        "major_league_log_loss_differences_vs_v4": league_differences,
        "recommendation": ("replace_v4_shadow" if all(checks.values()) else "retain_v4_shadow"),
    }


def evaluate_v6_nested(
    frame: pd.DataFrame,
    configuration: V6Configuration,
) -> dict[str, Any]:
    """Evaluate the complete inner-selection policy on untouched outer folds."""
    outer_folds = build_nested_folds(
        frame,
        configuration.study.nested_validation,
    )
    inner_reports: list[dict[str, Any]] = []
    outer_reports: list[dict[str, Any]] = []
    score_frames: list[pd.DataFrame] = []
    selected_probabilities: list[np.ndarray] = []
    control_probabilities: dict[str, list[np.ndarray]] = {name: [] for name in CONTROL_NAMES}
    selected_names: list[str] = []

    for outer in outer_folds:
        inner_report, selected_name = evaluate_inner_policy(
            outer,
            configuration,
        )
        outer_report, outer_selected, outer_controls = _evaluate_outer_selection(
            outer,
            selected_name,
            configuration,
        )
        inner_reports.append(inner_report)
        outer_reports.append(outer_report)
        score_frames.append(outer.score)
        selected_probabilities.append(outer_selected)
        for name in CONTROL_NAMES:
            control_probabilities[name].append(outer_controls[name])
        selected_names.append(selected_name)

    pooled_frame = pd.concat(score_frames, ignore_index=True)
    pooled_selected = np.concatenate(selected_probabilities)
    pooled_controls = {
        name: np.concatenate(values) for name, values in control_probabilities.items()
    }
    pooled_selected_metrics = evaluate_probabilities(
        pooled_frame,
        pooled_selected,
        include_breakdowns=True,
    )
    pooled_control_metrics = {
        name: evaluate_probabilities(
            pooled_frame,
            probabilities,
            include_breakdowns=True,
        )
        for name, probabilities in pooled_controls.items()
    }
    clusters = _cluster_values(pooled_frame)
    bootstrap_iterations = configuration.study.bootstrap_iterations
    random_seed = configuration.study.random_seed
    selected_interval = clustered_log_loss_interval(
        pooled_frame["blue_win"].to_numpy(dtype=int),
        pooled_selected,
        clusters,
        iterations=bootstrap_iterations,
        random_seed=random_seed,
    )
    paired_intervals = {
        name: clustered_paired_log_loss_difference_interval(
            pooled_frame["blue_win"].to_numpy(dtype=int),
            pooled_selected,
            probabilities,
            clusters,
            iterations=bootstrap_iterations,
            random_seed=random_seed,
        )
        for name, probabilities in pooled_controls.items()
    }
    decision = _decision_gate(
        pooled_frame,
        pooled_selected_metrics,
        pooled_control_metrics[V4_CONTROL_NAME],
        pooled_selected,
        pooled_controls[V4_CONTROL_NAME],
        paired_intervals[V4_CONTROL_NAME],
        outer_reports,
        configuration,
    )
    locked_candidate = selected_names[-1]
    locked_source = inner_reports[-1]
    locked_metrics = (
        locked_source["controls"][V4_CONTROL_NAME]["metrics"]
        if locked_candidate == V4_CONTROL_NAME
        else locked_source["candidates"][locked_candidate]["metrics"]
    )
    return {
        "schema_version": "1",
        "inner_reports": inner_reports,
        "outer_reports": outer_reports,
        "selection_policy": {
            "selected_candidate_counts": dict(sorted(Counter(selected_names).items())),
            "selected_candidates_by_outer_fold": selected_names,
            "metrics": pooled_selected_metrics,
            "series_clustered_log_loss_interval": selected_interval,
            "paired_log_loss_difference_intervals": paired_intervals,
            "prediction_fingerprint": prediction_fingerprint(
                pooled_frame,
                pooled_selected,
            ),
        },
        "controls": {
            name: {
                "metrics": pooled_control_metrics[name],
                "prediction_fingerprint": prediction_fingerprint(
                    pooled_frame,
                    pooled_controls[name],
                ),
            }
            for name in CONTROL_NAMES
        },
        "pooled_outer_score": frame_summary(pooled_frame),
        "decision_gate": decision,
        "locked_candidate": {
            "name": locked_candidate,
            "source_outer_fold": outer_folds[-1].fold_number,
            "selection_metric": "pooled_inner_log_loss",
            "selection_value": float(locked_metrics["log_loss"]),
        },
    }
