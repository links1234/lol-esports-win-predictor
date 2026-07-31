import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from lolpredictor.artifacts import ARTIFACT_PAYLOAD_FILES, load_artifact
from lolpredictor.fixtures import build_fixture_database
from lolpredictor.models import (
    CANDIDATE_NAMES,
    SIMPLE_CONTROL_NAMES,
    candidate_requires_calibration,
)
from lolpredictor.prediction import (
    load_draft_request,
    predict_file,
    predict_jsonl,
    predict_with_artifact,
)
from lolpredictor.settings import ExperimentSettings, SplitSettings
from lolpredictor.training import (
    backtest_baselines,
    confirm_backtest_selection,
    evaluate_saved_artifact,
    materialize_historical_features,
    refit_development_artifact,
    refit_selected_artifact,
    train_baselines,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def trained_run(
    tmp_path_factory: pytest.TempPathFactory,
    settings: ExperimentSettings,
) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("trained-run")
    database = root / "fixture.duckdb"
    fixture_summary = build_fixture_database(database)
    feature_summary = materialize_historical_features(database, settings)
    training_report = train_baselines(
        database,
        settings,
        registry_directory=root / "artifacts",
        reports_directory=root / "reports",
    )
    return {
        "root": root,
        "database": database,
        "fixture": fixture_summary,
        "features": feature_summary,
        "training": training_report,
        "artifact": Path(training_report["artifact_directory"]),
    }


@pytest.mark.integration
def test_complete_vertical_slice(trained_run: dict[str, Any]) -> None:
    assert trained_run["fixture"]["match_count"] == 120
    assert trained_run["features"]["feature_row_count"] == 120

    report = trained_run["training"]
    assert set(report["candidates"]) == set(CANDIDATE_NAMES)
    selected = report["selection"]["candidate"]
    selected_loss = report["selection"]["value"]
    assert selected_loss == min(
        candidate["validation"]["log_loss"] for candidate in report["candidates"].values()
    )

    artifact = load_artifact(trained_run["artifact"])
    assert artifact.manifest["model_kind"] == selected
    assert artifact.manifest["dataset_fingerprint"][:8] in artifact.manifest["model_version"]
    assert artifact.manifest["data_cutoff_timestamp"] == report["data_cutoff_timestamp"]
    for filename in (*ARTIFACT_PAYLOAD_FILES, "checksums.json"):
        assert (trained_run["artifact"] / filename).is_file()

    evaluation = evaluate_saved_artifact(
        trained_run["database"],
        trained_run["artifact"],
        output_path=trained_run["root"] / "reports" / "evaluation.json",
    )
    expected_test = report["candidates"][selected]["test"]
    assert evaluation["test"]["log_loss"] == expected_test["log_loss"]
    assert evaluation["test"]["sample_count"] == 24
    assert set(evaluation["test"]["breakdowns"]) == {
        "time_period",
        "patch",
        "league",
        "region",
        "tournament_level",
        "official_status",
    }

    prediction = predict_file(
        trained_run["artifact"],
        REPOSITORY_ROOT / "examples" / "sample_draft.json",
    )
    assert 0.0 <= prediction.blue_win_probability <= 1.0
    assert prediction.blue_win_probability + prediction.red_win_probability == 1.0
    assert prediction.model_version == artifact.manifest["model_version"]


@pytest.mark.integration
def test_artifact_checksum_detects_tampering(
    trained_run: dict[str, Any],
    tmp_path: Path,
) -> None:
    copied = tmp_path / "tampered"
    shutil.copytree(trained_run["artifact"], copied)
    with (copied / "model.joblib").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_artifact(copied)


@pytest.mark.integration
def test_batch_prediction_uses_the_same_artifact_pipeline(
    trained_run: dict[str, Any],
    tmp_path: Path,
) -> None:
    request = json.loads(
        (REPOSITORY_ROOT / "examples" / "sample_draft.json").read_text(encoding="utf-8")
    )
    input_path = tmp_path / "requests.jsonl"
    input_path.write_text(
        json.dumps(request) + "\n" + json.dumps(request) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "predictions.jsonl"
    summary = predict_jsonl(trained_run["artifact"], input_path, output_path)
    assert summary["prediction_count"] == 2
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0] == rows[1]


@pytest.mark.integration
def test_rolling_backtest_reserves_the_final_holdout(
    trained_run: dict[str, Any],
    settings: ExperimentSettings,
) -> None:
    confirmation_settings = settings.model_copy(
        update={
            "splits": SplitSettings(
                validation_start_timestamp=datetime(2024, 3, 1, tzinfo=UTC),
                test_start_timestamp=datetime(2024, 3, 20, tzinfo=UTC),
                calibration_fraction_within_train=0.1,
                refit_before_test=True,
            )
        }
    )
    report = backtest_baselines(
        trained_run["database"],
        confirmation_settings,
        reports_directory=trained_run["root"] / "backtest-reports",
    )
    assert report["fold_count"] == confirmation_settings.backtest.fold_count
    assert report["reserved_final_holdout_evaluated"] is False
    assert report["reserved_final_holdout"]["start_timestamp"] >= ("2024-03-20T00:00:00+00:00")
    assert report["selection_policy"]["test"]["sample_count"] > 0
    assert report["selection_policy"]["test"]["series_clustered_log_loss_interval"] is not None
    assert report["aggregate_selection"]["candidate"] in CANDIDATE_NAMES
    assert report["aggregate_selection"]["best_simple_control"] in SIMPLE_CONTROL_NAMES
    assert set(report["development_eligibility"]) == set(CANDIDATE_NAMES)
    selected = report["aggregate_selection"]["candidate"]
    assert report["development_eligibility"][selected]["eligible"] or selected == "elo_only"
    assert (trained_run["root"] / "backtest-reports" / "backtest-report.json").is_file()

    confirmation = confirm_backtest_selection(
        trained_run["database"],
        confirmation_settings,
        backtest_report_path=(trained_run["root"] / "backtest-reports" / "backtest-report.json"),
        output_path=(trained_run["root"] / "backtest-reports" / "confirmation-report.json"),
    )
    assert confirmation["selection"]["locked_before_confirmation"] is True
    assert confirmation["candidate_validation"]["sample_count"] > 0
    assert confirmation["reserved_final_holdout_evaluated"] is False
    assert "test" not in confirmation
    candidate_refit = confirmation["candidate_refit_split"]
    assert (
        candidate_refit["fit"]["sample_count"] + candidate_refit["calibration"]["sample_count"]
        == candidate_refit["train"]["sample_count"]
    )
    if candidate_requires_calibration(selected):
        assert candidate_refit["calibration"]["sample_count"] > 0
    else:
        assert candidate_refit["fit"] == candidate_refit["train"]
        assert candidate_refit["calibration"]["sample_count"] == 0

    development = refit_development_artifact(
        trained_run["database"],
        confirmation_settings,
        backtest_report_path=(trained_run["root"] / "backtest-reports" / "backtest-report.json"),
        confirmation_report_path=(
            trained_run["root"] / "backtest-reports" / "confirmation-report.json"
        ),
        registry_directory=trained_run["root"] / "development-artifacts",
        reports_directory=trained_run["root"] / "development-reports",
    )
    artifact = load_artifact(Path(development["artifact_directory"]))
    assert artifact.manifest["artifact_purpose"] == "development"
    assert artifact.manifest["selection"]["metric"] == "rolling_test_log_loss"
    assert "probability_calibration" in artifact.manifest
    assert (
        artifact.manifest["fit_sample_count"] == development["split_summary"]["fit"]["sample_count"]
    )
    if not candidate_requires_calibration(selected):
        assert (
            artifact.manifest["fit_sample_count"]
            == development["split_summary"]["train"]["sample_count"]
        )
        assert artifact.manifest["calibration_sample_count"] == 0
    assert development["prospective_promotion_passed"] is False
    shadow_request = load_draft_request(
        REPOSITORY_ROOT / "examples" / "sample_draft.json",
    ).model_copy(
        update={
            "match_timestamp": (artifact.feature_state.data_cutoff_timestamp + timedelta(days=1))
        }
    )
    shadow_prediction = predict_with_artifact(artifact, shadow_request)
    assert any("Development model" in warning for warning in shadow_prediction.warnings)


@pytest.mark.integration
def test_production_refit_uses_all_available_feature_state(
    trained_run: dict[str, Any],
    settings: ExperimentSettings,
) -> None:
    report = refit_selected_artifact(
        trained_run["database"],
        settings,
        training_report_path=trained_run["root"] / "reports" / "training-report.json",
        registry_directory=trained_run["root"] / "production-artifacts",
        reports_directory=trained_run["root"] / "production-reports",
    )
    artifact = load_artifact(Path(report["artifact_directory"]))
    assert artifact.manifest["artifact_purpose"] == "production"
    assert (
        artifact.manifest["data_cutoff_timestamp"]
        == trained_run["features"]["history_cutoff_timestamp"]
    )
    assert report["model_kind"] == trained_run["training"]["selection"]["candidate"]
    with pytest.raises(ValueError, match="do not have an untouched"):
        evaluate_saved_artifact(
            trained_run["database"],
            artifact.directory,
        )

    promotion_report_path = trained_run["root"] / "reports" / "promotion-fallback.json"
    promotion_report_path.write_text(
        json.dumps(
            {
                "dataset_fingerprint": trained_run["training"]["dataset_fingerprint"],
                "experiment_name": settings.experiment_name,
                "promotion_passed": False,
                "recommended_candidate": "elo_only",
            }
        ),
        encoding="utf-8",
    )
    fallback = refit_selected_artifact(
        trained_run["database"],
        settings,
        training_report_path=trained_run["root"] / "reports" / "training-report.json",
        promotion_report_path=promotion_report_path,
        registry_directory=trained_run["root"] / "fallback-artifacts",
        reports_directory=trained_run["root"] / "fallback-reports",
    )
    assert fallback["model_kind"] == "elo_only"


@pytest.mark.integration
def test_release_training_refits_before_a_timestamp_holdout(
    trained_run: dict[str, Any],
    settings: ExperimentSettings,
    tmp_path: Path,
) -> None:
    payload = settings.resolved()
    payload["experiment_name"] = "synthetic-date-release"
    payload["model_version"] = "synthetic-date-release-v1"
    payload["splits"] = {
        "validation_start_timestamp": "2024-02-15T00:00:00+00:00",
        "test_start_timestamp": "2024-03-15T00:00:00+00:00",
        "calibration_fraction_within_train": 0.10,
        "refit_before_test": True,
    }
    payload["models"]["production_calibration_fraction"] = 0.10
    release_settings = ExperimentSettings.model_validate(payload)

    report = train_baselines(
        trained_run["database"],
        release_settings,
        registry_directory=tmp_path / "artifacts",
        reports_directory=tmp_path / "reports",
    )

    assert report["release_refit_before_test"] is True
    assert (
        report["selection_split_summary"]["train"]["end_timestamp"]
        < report["selection_split_summary"]["validation"]["start_timestamp"]
    )
    assert report["data_cutoff_timestamp"] == report["split_summary"]["train"]["end_timestamp"]
    assert (
        report["data_cutoff_timestamp"]
        == report["selection_split_summary"]["validation"]["end_timestamp"]
    )
    selected = report["selection"]["candidate"]
    assert (
        report["candidates"][selected]["test"]["paired_log_loss_difference_vs_elo_interval"]
        is not None
    )

    evaluation = evaluate_saved_artifact(
        trained_run["database"],
        Path(report["artifact_directory"]),
    )
    assert evaluation["test"]["log_loss"] == report["candidates"][selected]["test"]["log_loss"]
