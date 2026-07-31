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
def test_parallel_nested_optimizer_completes_and_is_resumable(tmp_path: Path) -> None:
    database = tmp_path / "fixture.duckdb"
    study = tmp_path / "study"
    base_config = REPOSITORY_ROOT / "configs" / "v5-optimization-fixture-base.yaml"
    optimization_config = REPOSITORY_ROOT / "configs" / "v5-optimization-fixture.yaml"

    run_cli("fixture", "--database", str(database))
    features = run_cli(
        "features",
        "--database",
        str(database),
        "--config",
        str(base_config),
    )
    assert features["feature_row_count"] == 120

    result = run_cli(
        "optimize",
        "--database",
        str(database),
        "--config",
        str(optimization_config),
        "--study",
        str(study),
    )
    assert result["status_counts"] == {
        "completed": 30,
        "failed": 0,
        "pending": 0,
        "running": 0,
    }
    locked_before = (study / "locked-finalist.json").read_bytes()
    report = json.loads((study / "nested-report.json").read_text(encoding="utf-8"))
    assert report["development_cutoff_timestamp"] == "2026-01-01T00:00:00+00:00"
    assert report["interpretation"]["outer_outcomes_used_for_selection"] is False
    assert report["interpretation"]["outcomes_at_or_after_cutoff_loaded"] is False
    assert report["outer_policy_evaluation"]["score_sample_count"] > 0
    assert report["outer_policy_evaluation"]["selected_policy"]["sample_count"] > 0

    resumed = run_cli(
        "optimize",
        "--database",
        str(database),
        "--config",
        str(optimization_config),
        "--study",
        str(study),
    )
    assert resumed["status_counts"]["completed"] == 30
    assert (study / "locked-finalist.json").read_bytes() == locked_before
    status = run_cli("optimize-status", "--study", str(study))
    assert status["status_counts"]["completed"] == 30
    assert status["reset_interrupted_trial_count"] == 0

    refit = run_cli(
        "optimize-refit",
        "--database",
        str(database),
        "--config",
        str(optimization_config),
        "--locked-finalist",
        str(study / "locked-finalist.json"),
        "--nested-report",
        str(study / "nested-report.json"),
        "--registry",
        str(tmp_path / "artifacts"),
        "--output",
        str(study / "refit-report.json"),
    )
    assert refit["artifact_purpose"] == "development"
    assert refit["promotion_status"] == "development_only"

    request = json.loads(
        (REPOSITORY_ROOT / "examples" / "sample_draft.json").read_text(encoding="utf-8")
    )
    request["match_timestamp"] = "2024-04-15T12:00:00Z"
    request_path = tmp_path / "future-draft.json"
    request_path.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prediction = run_cli(
        "predict",
        "--artifact",
        refit["artifact_directory"],
        "--input",
        str(request_path),
    )
    assert prediction["blue_win_probability"] + prediction["red_win_probability"] == 1.0
    assert prediction["warnings"][0].startswith("Development model:")
