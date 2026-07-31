import json
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def run_cli(*arguments: str) -> dict:
    completed = subprocess.run(
        [sys.executable, "-m", "lolpredictor", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.e2e
def test_complete_workflow_runs_in_separate_processes(tmp_path: Path) -> None:
    database = tmp_path / "fixture.duckdb"
    registry = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    config = REPOSITORY_ROOT / "configs" / "baselines.yaml"

    fixture = run_cli("fixture", "--database", str(database))
    assert fixture["match_count"] == 120

    features = run_cli(
        "features",
        "--database",
        str(database),
        "--config",
        str(config),
    )
    assert features["feature_row_count"] == 120

    backtest = run_cli(
        "backtest",
        "--database",
        str(database),
        "--config",
        str(config),
        "--reports",
        str(reports / "backtest"),
    )
    assert backtest["fold_count"] == 3
    assert backtest["reserved_final_holdout_evaluated"] is False

    training = run_cli(
        "train",
        "--database",
        str(database),
        "--config",
        str(config),
        "--registry",
        str(registry),
        "--reports",
        str(reports),
    )
    artifact = Path(training["artifact_directory"])
    assert artifact.is_dir()

    evaluation = run_cli(
        "evaluate",
        "--database",
        str(database),
        "--artifact",
        str(artifact),
        "--output",
        str(reports / "evaluation.json"),
    )
    assert evaluation["test"]["sample_count"] == 24

    prediction = run_cli(
        "predict",
        "--artifact",
        str(artifact),
        "--input",
        str(REPOSITORY_ROOT / "examples" / "sample_draft.json"),
        "--output",
        str(reports / "prediction.json"),
    )
    assert prediction["blue_win_probability"] + prediction["red_win_probability"] == 1.0
    assert prediction["model_version"] == evaluation["model_version"]
    assert (reports / "prediction.json").is_file()

    vision_directory = tmp_path / "vision"
    vision = run_cli(
        "vision-fixture",
        "--directory",
        str(vision_directory),
        "--sample",
        str(REPOSITORY_ROOT / "examples" / "sample_draft.json"),
    )
    screenshot_prediction = run_cli(
        "predict-image",
        "--artifact",
        str(artifact),
        "--input",
        vision["screenshot"],
        "--profile",
        vision["profile"],
        "--catalog",
        vision["catalog"],
        "--context",
        vision["context"],
        "--output",
        str(reports / "screenshot-prediction.json"),
    )
    assert screenshot_prediction["status"] == "predicted"
    assert (
        screenshot_prediction["prediction"]["blue_win_probability"]
        + screenshot_prediction["prediction"]["red_win_probability"]
        == 1.0
    )
    assert screenshot_prediction["prediction"]["estimate_type"] == "post_draft_pregame"
