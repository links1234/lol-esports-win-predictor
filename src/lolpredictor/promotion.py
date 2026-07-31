"""Preregistered release gates for a current chronological holdout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lolpredictor.settings import ExperimentSettings

PROMOTION_REPORT_SCHEMA_VERSION = "1"
BASE_RATE_CANDIDATE = "blue_side_base_rate"
ELO_CANDIDATE = "elo_only"


def _candidate_metrics(
    report: dict[str, Any],
    candidate: str,
    interval: str,
) -> dict[str, Any]:
    try:
        value = report["candidates"][candidate][interval]
    except KeyError as error:
        raise ValueError(
            f"Report is missing {interval} metrics for candidate {candidate}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"Candidate {candidate} {interval} metrics must be an object")
    return value


def _aggregate_metrics(
    report: dict[str, Any],
    candidate: str,
) -> dict[str, Any]:
    try:
        value = report["aggregate_candidates"][candidate]
    except KeyError as error:
        raise ValueError(f"Backtest report is missing candidate {candidate}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Backtest candidate {candidate} metrics must be an object")
    return value


def _number(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"Metric {key} is missing or non-numeric")
    return float(value)


def _gate(
    passed: bool,
    *,
    observed: object,
    requirement: str,
) -> dict[str, object]:
    return {
        "passed": bool(passed),
        "observed": observed,
        "requirement": requirement,
    }


def evaluate_release_promotion(
    training_report: dict[str, Any],
    backtest_report: dict[str, Any],
    settings: ExperimentSettings,
) -> dict[str, Any]:
    """Evaluate the fixed promotion rules without fitting or tuning a model."""
    gate_settings = settings.release_gate
    if not gate_settings.enabled:
        raise ValueError("Release promotion gates are disabled in this experiment config")
    if training_report.get("experiment_name") != settings.experiment_name:
        raise ValueError("Training report experiment does not match the release config")
    if backtest_report.get("experiment_name") != settings.experiment_name:
        raise ValueError("Backtest report experiment does not match the release config")
    training_fingerprint = training_report.get("dataset_fingerprint")
    if not isinstance(training_fingerprint, str) or not training_fingerprint:
        raise ValueError("Training report has no dataset fingerprint")
    if backtest_report.get("dataset_fingerprint") != training_fingerprint:
        raise ValueError("Training and backtest reports use different datasets")
    if not training_report.get("release_refit_before_test"):
        raise ValueError("Release training report did not refit before the final holdout")

    selection = training_report.get("selection")
    if not isinstance(selection, dict) or not isinstance(selection.get("candidate"), str):
        raise ValueError("Training report has no selected candidate")
    selected_name = selection["candidate"]

    selected_validation = _candidate_metrics(training_report, selected_name, "validation")
    base_validation = _candidate_metrics(
        training_report,
        BASE_RATE_CANDIDATE,
        "validation",
    )
    elo_validation = _candidate_metrics(training_report, ELO_CANDIDATE, "validation")
    selected_test = _candidate_metrics(training_report, selected_name, "test")
    elo_test = _candidate_metrics(training_report, ELO_CANDIDATE, "test")
    selected_rolling = _aggregate_metrics(backtest_report, selected_name)
    elo_rolling = _aggregate_metrics(backtest_report, ELO_CANDIDATE)

    selected_validation_log_loss = _number(selected_validation, "log_loss")
    base_validation_log_loss = _number(base_validation, "log_loss")
    elo_validation_log_loss = _number(elo_validation, "log_loss")
    selected_test_log_loss = _number(selected_test, "log_loss")
    elo_test_log_loss = _number(elo_test, "log_loss")
    selected_rolling_log_loss = _number(selected_rolling, "log_loss")
    elo_rolling_log_loss = _number(elo_rolling, "log_loss")
    minimum_improvement = gate_settings.minimum_log_loss_improvement

    absolute_interval = selected_test.get("series_clustered_log_loss_interval")
    if not isinstance(absolute_interval, dict):
        raise ValueError("Selected holdout metrics have no series-clustered interval")
    holdout_sample_count = int(_number(selected_test, "sample_count"))
    holdout_series_count = int(_number(absolute_interval, "cluster_count"))

    paired_interval = selected_test.get("paired_log_loss_difference_vs_elo_interval")
    paired_upper = _number(paired_interval, "upper") if isinstance(paired_interval, dict) else None

    selected_leagues = selected_test.get("breakdowns", {}).get("league", {})
    elo_leagues = elo_test.get("breakdowns", {}).get("league", {})
    if not isinstance(selected_leagues, dict) or not isinstance(elo_leagues, dict):
        raise ValueError("Holdout league breakdowns are missing")
    major_league_evaluations: dict[str, dict[str, object]] = {}
    major_league_passed = True
    for league in gate_settings.major_leagues:
        selected_league = selected_leagues.get(league)
        elo_league = elo_leagues.get(league)
        if not isinstance(selected_league, dict) or not isinstance(elo_league, dict):
            major_league_evaluations[league] = {
                "evaluated": False,
                "sample_count": 0,
                "reason": "league_absent_from_holdout",
            }
            continue
        sample_count = int(_number(selected_league, "sample_count"))
        if sample_count < 100:
            major_league_evaluations[league] = {
                "evaluated": False,
                "sample_count": sample_count,
                "reason": "fewer_than_100_holdout_games",
            }
            continue
        regression = _number(selected_league, "log_loss") - _number(
            elo_league,
            "log_loss",
        )
        passed = regression <= gate_settings.maximum_major_league_log_loss_regression
        major_league_passed = major_league_passed and passed
        major_league_evaluations[league] = {
            "evaluated": True,
            "sample_count": sample_count,
            "selected_minus_elo_log_loss": regression,
            "passed": passed,
        }

    enough_holdout_data = (
        holdout_sample_count >= gate_settings.minimum_holdout_games
        and holdout_series_count >= gate_settings.minimum_holdout_series
    )
    gates = {
        "holdout_sample_sufficiency": _gate(
            enough_holdout_data,
            observed={
                "games": holdout_sample_count,
                "series": holdout_series_count,
            },
            requirement=(
                f"at least {gate_settings.minimum_holdout_games} games and "
                f"{gate_settings.minimum_holdout_series} series"
            ),
        ),
        "validation_beats_base_rate": _gate(
            selected_validation_log_loss < base_validation_log_loss,
            observed=selected_validation_log_loss - base_validation_log_loss,
            requirement="selected-minus-base-rate validation log loss is below zero",
        ),
        "validation_beats_elo": _gate(
            selected_validation_log_loss < elo_validation_log_loss,
            observed=selected_validation_log_loss - elo_validation_log_loss,
            requirement="selected-minus-Elo validation log loss is below zero",
        ),
        "rolling_log_loss_improvement": _gate(
            elo_rolling_log_loss - selected_rolling_log_loss >= minimum_improvement,
            observed=elo_rolling_log_loss - selected_rolling_log_loss,
            requirement=f"at least {minimum_improvement} lower log loss than Elo",
        ),
        "holdout_log_loss_improvement": _gate(
            elo_test_log_loss - selected_test_log_loss >= minimum_improvement,
            observed=elo_test_log_loss - selected_test_log_loss,
            requirement=f"at least {minimum_improvement} lower log loss than Elo",
        ),
        "paired_series_interval": _gate(
            paired_upper is not None and paired_upper < 0.0,
            observed=paired_interval,
            requirement="95 percent upper bound for selected-minus-Elo is below zero",
        ),
        "holdout_brier": _gate(
            _number(selected_test, "brier_score") <= _number(elo_test, "brier_score"),
            observed={
                "selected": _number(selected_test, "brier_score"),
                "elo": _number(elo_test, "brier_score"),
            },
            requirement="selected Brier score is no worse than Elo",
        ),
        "holdout_calibration": _gate(
            _number(selected_test, "expected_calibration_error")
            <= gate_settings.maximum_expected_calibration_error,
            observed=_number(selected_test, "expected_calibration_error"),
            requirement=(
                "expected calibration error is at most "
                f"{gate_settings.maximum_expected_calibration_error}"
            ),
        ),
        "major_league_regression": _gate(
            major_league_passed,
            observed=major_league_evaluations,
            requirement=(
                "no configured major league with at least 100 games regresses "
                f"by more than {gate_settings.maximum_major_league_log_loss_regression}"
            ),
        ),
        "probability_contract": _gate(
            selected_test.get("probability_contract_valid") is True,
            observed=selected_test.get("probability_contract_valid"),
            requirement="all holdout probabilities are finite and within zero and one",
        ),
    }
    promotion_passed = all(bool(gate["passed"]) for gate in gates.values())
    return {
        "promotion_report_schema_version": PROMOTION_REPORT_SCHEMA_VERSION,
        "experiment_name": settings.experiment_name,
        "dataset_fingerprint": training_fingerprint,
        "selected_candidate": selected_name,
        "promotion_passed": promotion_passed,
        "provisional": not enough_holdout_data,
        "recommended_candidate": selected_name if promotion_passed else ELO_CANDIDATE,
        "fallback_policy": "use Elo when any preregistered promotion gate fails",
        "gates": gates,
    }


def write_release_promotion_report(
    training_report_path: Path,
    backtest_report_path: Path,
    settings: ExperimentSettings,
    output_path: Path,
) -> dict[str, Any]:
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    backtest_report = json.loads(backtest_report_path.read_text(encoding="utf-8"))
    if not isinstance(training_report, dict) or not isinstance(backtest_report, dict):
        raise ValueError("Training and backtest reports must contain JSON objects")
    report = evaluate_release_promotion(
        training_report,
        backtest_report,
        settings,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
