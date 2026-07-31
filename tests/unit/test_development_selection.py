from typing import Any

from lolpredictor.settings import ExperimentSettings, ReleaseGateSettings
from lolpredictor.training import _development_candidate_eligibility


def _candidate_metrics(
    *,
    log_loss: float,
    league_log_loss: float,
    paired_upper: float,
) -> dict[str, Any]:
    return {
        "log_loss": log_loss,
        "expected_calibration_error": 0.02,
        "paired_log_loss_difference_vs_elo_interval": {
            "lower": -0.02,
            "point_estimate": -0.01,
            "upper": paired_upper,
        },
        "breakdowns": {
            "league": {
                "League A": {
                    "sample_count": 200,
                    "log_loss": league_log_loss,
                }
            }
        },
    }


def test_development_eligibility_rejects_hidden_major_league_regression(
    settings: ExperimentSettings,
) -> None:
    configured = settings.model_copy(
        update={
            "release_gate": ReleaseGateSettings(
                enabled=True,
                maximum_expected_calibration_error=0.04,
                maximum_major_league_log_loss_regression=0.01,
                major_leagues=("League A",),
            )
        }
    )
    aggregate_candidates = {
        "elo_only": _candidate_metrics(
            log_loss=0.620,
            league_log_loss=0.600,
            paired_upper=0.0,
        ),
        "team_roster_logistic": _candidate_metrics(
            log_loss=0.615,
            league_log_loss=0.605,
            paired_upper=-0.001,
        ),
        "candidate": _candidate_metrics(
            log_loss=0.610,
            league_log_loss=0.620,
            paired_upper=-0.001,
        ),
    }
    fold_reports = [
        {
            "fold_number": fold_number,
            "candidates": {
                "elo_only": {"test": {"log_loss": 0.620}},
                "candidate": {"test": {"log_loss": 0.610}},
            },
        }
        for fold_number in range(1, 4)
    ]

    result = _development_candidate_eligibility(
        "candidate",
        aggregate_candidates=aggregate_candidates,
        fold_reports=fold_reports,
        best_simple_control="team_roster_logistic",
        settings=configured,
    )

    assert result["checks"]["major_league_regressions_within_limit"] is False
    assert result["eligible"] is False


def test_development_eligibility_accepts_stable_robust_improvement(
    settings: ExperimentSettings,
) -> None:
    configured = settings.model_copy(
        update={
            "release_gate": ReleaseGateSettings(
                enabled=True,
                maximum_expected_calibration_error=0.04,
                maximum_major_league_log_loss_regression=0.01,
                major_leagues=("League A",),
            )
        }
    )
    aggregate_candidates = {
        "elo_only": _candidate_metrics(
            log_loss=0.620,
            league_log_loss=0.600,
            paired_upper=0.0,
        ),
        "team_roster_logistic": _candidate_metrics(
            log_loss=0.615,
            league_log_loss=0.605,
            paired_upper=-0.001,
        ),
        "candidate": _candidate_metrics(
            log_loss=0.610,
            league_log_loss=0.605,
            paired_upper=-0.001,
        ),
    }
    fold_reports = [
        {
            "fold_number": fold_number,
            "candidates": {
                "elo_only": {"test": {"log_loss": 0.620}},
                "candidate": {"test": {"log_loss": 0.610}},
            },
        }
        for fold_number in range(1, 4)
    ]

    result = _development_candidate_eligibility(
        "candidate",
        aggregate_candidates=aggregate_candidates,
        fold_reports=fold_reports,
        best_simple_control="team_roster_logistic",
        settings=configured,
    )

    assert all(result["checks"].values())
    assert result["eligible"] is True
