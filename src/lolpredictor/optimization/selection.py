"""Preregistered trial eligibility and inner-only finalist selection."""

from __future__ import annotations

from typing import Any

from lolpredictor.optimization.evaluation import CONTROL_NAMES
from lolpredictor.optimization.settings import OptimizationConfiguration


def trial_eligibility(
    trial_result: dict[str, Any],
    controls: dict[str, Any],
    configuration: OptimizationConfiguration,
) -> dict[str, Any]:
    """Apply frozen checks using inner results from one outer fold only."""
    candidate = trial_result["pooled_inner"]
    elo = controls["elo_only"]["pooled_inner"]
    team_roster = controls["team_roster_logistic"]["pooled_inner"]
    current = controls["elo_catboost_regional_raw_blend_50"]["pooled_inner"]
    simple_reference_loss = min(float(elo["log_loss"]), float(team_roster["log_loss"]))

    candidate_folds = trial_result["inner_folds"]
    elo_folds = controls["elo_only"]["inner_folds"]
    if len(candidate_folds) != len(elo_folds):
        raise ValueError("Candidate and Elo inner-fold counts do not match")
    fold_differences = [
        {
            "fold_number": int(candidate_fold["fold_number"]),
            "log_loss_difference_vs_elo": (
                float(candidate_fold["metrics"]["log_loss"])
                - float(elo_fold["metrics"]["log_loss"])
            ),
        }
        for candidate_fold, elo_fold in zip(
            candidate_folds,
            elo_folds,
            strict=True,
        )
    ]

    selection = configuration.optimization.selection
    candidate_leagues = candidate["breakdowns"]["league"]
    elo_leagues = elo["breakdowns"]["league"]
    league_differences: dict[str, dict[str, float | int]] = {}
    for league in configuration.experiment.release_gate.major_leagues:
        candidate_metrics = candidate_leagues.get(league)
        elo_metrics = elo_leagues.get(league)
        if candidate_metrics is None or elo_metrics is None:
            continue
        sample_count = int(candidate_metrics["sample_count"])
        if sample_count < selection.minimum_breakdown_sample_count:
            continue
        league_differences[league] = {
            "sample_count": sample_count,
            "log_loss_difference_vs_elo": (
                float(candidate_metrics["log_loss"]) - float(elo_metrics["log_loss"])
            ),
        }

    no_fold_regression = all(item["log_loss_difference_vs_elo"] <= 0.0 for item in fold_differences)
    checks = {
        "beats_elo": float(candidate["log_loss"]) < float(elo["log_loss"]),
        "beats_best_simple_control": float(candidate["log_loss"]) < simple_reference_loss,
        "beats_current_v4_control": (float(candidate["log_loss"]) < float(current["log_loss"])),
        "calibration_within_limit": (
            float(candidate["expected_calibration_error"])
            <= selection.maximum_expected_calibration_error
        ),
        "no_inner_fold_regression_vs_elo": (
            no_fold_regression if selection.require_no_inner_fold_regression_vs_elo else True
        ),
        "major_league_regressions_within_limit": all(
            float(metrics["log_loss_difference_vs_elo"])
            <= selection.maximum_major_league_log_loss_regression
            for metrics in league_differences.values()
        ),
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "fold_log_loss_differences_vs_elo": fold_differences,
        "major_league_log_loss_differences_vs_elo": league_differences,
        "simple_reference_log_loss": simple_reference_loss,
        "current_v4_log_loss": float(current["log_loss"]),
    }


def select_outer_winners(
    records: list[dict[str, Any]],
    controls_report: dict[str, Any],
    configuration: OptimizationConfiguration,
) -> dict[str, Any]:
    """Select each outer finalist without consulting any outer score."""
    outer_selections: dict[str, Any] = {}
    for outer_fold in range(
        1,
        configuration.optimization.nested_validation.outer_fold_count + 1,
    ):
        controls = controls_report["outer_folds"][str(outer_fold)]
        candidates: list[dict[str, Any]] = []
        for record in records:
            if (
                record["outer_fold"] != outer_fold
                or record["status"] != "completed"
                or record["result"] is None
            ):
                continue
            eligibility = trial_eligibility(
                record["result"],
                controls,
                configuration,
            )
            candidates.append(
                {
                    "trial_id": record["trial_id"],
                    "spec_hash": record["spec_hash"],
                    "family": record["family"],
                    "feature_mode": record["feature_mode"],
                    "output_mapping": record["spec"]["output_mapping"],
                    "pooled_inner_log_loss": float(record["result"]["pooled_inner"]["log_loss"]),
                    "pooled_inner_brier_score": float(
                        record["result"]["pooled_inner"]["brier_score"]
                    ),
                    "pooled_inner_expected_calibration_error": float(
                        record["result"]["pooled_inner"]["expected_calibration_error"]
                    ),
                    "eligibility": eligibility,
                }
            )
        candidates.sort(
            key=lambda item: (
                item["pooled_inner_log_loss"],
                item["trial_id"],
            )
        )
        eligible = [item for item in candidates if item["eligibility"]["eligible"]]
        if eligible:
            winner = eligible[0]
            selection = {
                "selection_kind": "searched_trial",
                "trial_id": winner["trial_id"],
                "spec_hash": winner["spec_hash"],
                "family": winner["family"],
                "feature_mode": winner["feature_mode"],
                "output_mapping": winner["output_mapping"],
                "pooled_inner_log_loss": winner["pooled_inner_log_loss"],
                "fallback_candidate": None,
            }
        else:
            fallback = configuration.optimization.selection.fallback_candidate
            selection = {
                "selection_kind": "fallback_control",
                "trial_id": None,
                "spec_hash": None,
                "family": None,
                "feature_mode": "cached",
                "output_mapping": "native",
                "pooled_inner_log_loss": float(controls[fallback]["pooled_inner"]["log_loss"]),
                "fallback_candidate": fallback,
            }
        outer_selections[str(outer_fold)] = {
            "selection": selection,
            "completed_candidate_count": len(candidates),
            "eligible_candidate_count": len(eligible),
            "ranked_candidates": candidates,
            "control_log_losses": {
                name: float(controls[name]["pooled_inner"]["log_loss"]) for name in CONTROL_NAMES
            },
        }
    return {
        "selection_metric": "pooled_inner_log_loss",
        "outer_selections": outer_selections,
        "locked_finalist_outer_fold": (
            configuration.optimization.nested_validation.outer_fold_count
        ),
    }
