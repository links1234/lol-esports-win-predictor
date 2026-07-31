import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

from lolpredictor.artifacts import load_artifact
from lolpredictor.fixtures import build_fixture_database, generate_synthetic_matches
from lolpredictor.state_space.refit import refit_v6_finalist
from lolpredictor.state_space.runner import run_v6_study


def test_state_space_study_refit_and_separate_prediction_process(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    database = tmp_path / "fixture.duckdb"
    study = tmp_path / "study"
    registry = tmp_path / "artifacts"
    refit_report = tmp_path / "refit-report.json"
    config = repository_root / "configs" / "v6-state-space-fixture.yaml"
    build_fixture_database(database)

    summary = run_v6_study(database, config, study)

    assert summary["status"] == "complete"
    assert summary["source_sample_count"] == 120
    assert summary["pooled_outer_sample_count"] > 0
    assert summary["development_cutoff_exclusive"] == "2026-01-01T00:00:00+00:00"
    study_report = json.loads((study / "study-report.json").read_text(encoding="utf-8"))
    assert (
        study_report["nested_evaluation"]["selection_policy"]["metrics"]["sample_count"]
        == summary["pooled_outer_sample_count"]
    )
    assert set(study_report["nested_evaluation"]["inner_reports"][0]["candidates"]) == {
        "state_space_native",
        "state_space_platt",
        "state_space_augmented_logistic",
        "state_space_v4_blend_15",
        "state_space_v4_blend_30",
    }

    refit = refit_v6_finalist(
        database,
        config,
        locked_finalist_path=study / "locked-finalist.json",
        study_report_path=study / "study-report.json",
        registry_directory=registry,
        output_path=refit_report,
    )
    artifact_path = Path(refit["artifact_directory"])
    artifact = load_artifact(artifact_path)
    assert artifact.manifest["artifact_purpose"] == "development"
    assert artifact.feature_state.settings.state_space.enabled
    assert artifact.feature_state.state_space_skills

    source = generate_synthetic_matches()[-1]
    request_payload = source.model_dump(
        mode="json",
        exclude={"match_id", "blue_win"},
    )
    request_payload["match_timestamp"] = (source.match_timestamp + timedelta(days=1)).isoformat()
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "prediction.json"
    request_path.write_text(
        json.dumps(request_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "lolpredictor",
            "predict",
            "--artifact",
            str(artifact_path),
            "--input",
            str(request_path),
            "--output",
            str(output_path),
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    prediction = json.loads(output_path.read_text(encoding="utf-8"))

    assert process.returncode == 0
    assert 0.0 < prediction["blue_win_probability"] < 1.0
    assert prediction["blue_win_probability"] + prediction["red_win_probability"] == 1.0
    assert any("Development model" in warning for warning in prediction["warnings"])
