from pathlib import Path
from typing import Any

from lolpredictor.promotion import evaluate_release_promotion
from lolpredictor.settings import load_settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SELECTED = "logistic_regression"
ELO = "elo_only"
BASE = "blue_side_base_rate"


def _test_metrics(
    *,
    log_loss: float,
    league_log_loss: float,
    paired_upper: float,
) -> dict[str, Any]:
    return {
        "log_loss": log_loss,
        "brier_score": 0.20 if log_loss < 0.61 else 0.21,
        "expected_calibration_error": 0.03,
        "sample_count": 1000,
        "probability_contract_valid": True,
        "series_clustered_log_loss_interval": {
            "cluster_count": 400,
            "lower": log_loss - 0.01,
            "upper": log_loss + 0.01,
        },
        "paired_log_loss_difference_vs_elo_interval": {
            "point_estimate": log_loss - 0.62,
            "lower": -0.03,
            "upper": paired_upper,
        },
        "breakdowns": {
            "league": {
                "Tencent LoL Pro League": {
                    "log_loss": league_log_loss,
                    "sample_count": 150,
                }
            }
        },
    }


def _reports() -> tuple[dict[str, Any], dict[str, Any]]:
    validation = {
        SELECTED: {"log_loss": 0.60},
        ELO: {"log_loss": 0.63},
        BASE: {"log_loss": 0.69},
    }
    training = {
        "experiment_name": "leaguepedia-2020-2026-current-release",
        "dataset_fingerprint": "abc123",
        "release_refit_before_test": True,
        "selection": {"candidate": SELECTED},
        "candidates": {
            name: {
                "validation": metrics,
                "test": (
                    _test_metrics(
                        log_loss=0.60,
                        league_log_loss=0.59,
                        paired_upper=-0.002,
                    )
                    if name == SELECTED
                    else _test_metrics(
                        log_loss=0.62 if name == ELO else 0.68,
                        league_log_loss=0.60 if name == ELO else 0.67,
                        paired_upper=0.0,
                    )
                ),
            }
            for name, metrics in validation.items()
        },
    }
    backtest = {
        "experiment_name": training["experiment_name"],
        "dataset_fingerprint": training["dataset_fingerprint"],
        "aggregate_candidates": {
            SELECTED: {"log_loss": 0.61},
            ELO: {"log_loss": 0.62},
        },
    }
    return training, backtest


def test_all_preregistered_gates_must_pass() -> None:
    settings = load_settings(REPOSITORY_ROOT / "configs" / "current-release.yaml")
    training, backtest = _reports()

    report = evaluate_release_promotion(training, backtest, settings)

    assert report["promotion_passed"] is True
    assert report["provisional"] is False
    assert report["recommended_candidate"] == SELECTED
    assert all(gate["passed"] for gate in report["gates"].values())


def test_failed_gate_falls_back_to_elo() -> None:
    settings = load_settings(REPOSITORY_ROOT / "configs" / "current-release.yaml")
    training, backtest = _reports()
    training["candidates"][SELECTED]["test"]["log_loss"] = 0.619

    report = evaluate_release_promotion(training, backtest, settings)

    assert report["promotion_passed"] is False
    assert report["recommended_candidate"] == ELO
    assert report["gates"]["holdout_log_loss_improvement"]["passed"] is False
