"""Post-lock development refit for a v6 state-space finalist."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from lolpredictor.artifacts import config_fingerprint, save_artifact
from lolpredictor.features import generate_historical_features
from lolpredictor.models import CalibratedCandidate
from lolpredictor.state_space.models import (
    fit_v4_control,
    fit_v6_candidates,
)
from lolpredictor.state_space.runner import (
    validate_locked_finalist,
    value_fingerprint,
)
from lolpredictor.state_space.settings import (
    V4_CONTROL_NAME,
    V6CandidateName,
    load_v6_configuration,
)
from lolpredictor.storage import load_matches
from lolpredictor.training import dataset_fingerprint, filter_modeling_population


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    timestamps = pd.to_datetime(frame["match_timestamp"], utc=True)
    return {
        "sample_count": len(frame),
        "timestamp_count": int(timestamps.nunique()),
        "start_timestamp": timestamps.min().isoformat(),
        "end_timestamp": timestamps.max().isoformat(),
    }


def refit_v6_finalist(
    database_path: Path,
    config_path: Path,
    *,
    locked_finalist_path: Path,
    study_report_path: Path,
    registry_directory: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate the lock and create a clearly non-promoted current artifact."""
    configuration = load_v6_configuration(config_path)
    lock = _load_json(locked_finalist_path)
    candidate_name = validate_locked_finalist(lock)
    expected_configuration_fingerprint = value_fingerprint(configuration.resolved())
    if lock["configuration_fingerprint"] != expected_configuration_fingerprint:
        raise ValueError("V6 lock was created from a different configuration")
    if (
        lock["development_cutoff_timestamp"]
        != configuration.study.development_cutoff_timestamp.isoformat()
    ):
        raise ValueError("V6 lock uses a different development cutoff")

    study_report = _load_json(study_report_path)
    embedded_lock = study_report.get("locked_finalist")
    if embedded_lock != lock:
        raise ValueError("V6 study report and locked-finalist file disagree")
    if study_report.get("status") != "complete":
        raise ValueError("V6 study report is not complete")

    matches = load_matches(database_path)
    if not matches:
        raise ValueError("No matches are available for the v6 development refit")
    frame, feature_state = generate_historical_features(
        matches,
        configuration.experiment.features,
    )
    modeling_frame, population = filter_modeling_population(
        frame,
        configuration.experiment,
    )
    if candidate_name == V4_CONTROL_NAME:
        candidate = fit_v4_control(modeling_frame, configuration)
    else:
        candidate = fit_v6_candidates(
            modeling_frame,
            configuration,
        )[cast(V6CandidateName, candidate_name)]
    if not isinstance(candidate, CalibratedCandidate):
        raise TypeError("V6 development refit produced an unexpected model type")

    cutoff = feature_state.data_cutoff_timestamp
    if cutoff is None:
        raise ValueError("V6 development refit has no feature-state cutoff")
    fingerprint = dataset_fingerprint(database_path)
    split_summary = {
        "development_refit": _frame_summary(modeling_frame),
        "post_lock_outcomes_used_for_estimation_only": True,
    }
    metrics = {
        "schema_version": "1",
        "artifact_purpose": "development",
        "candidate_name": candidate_name,
        "prospective_promotion_passed": False,
        "selection_decision": study_report["nested_evaluation"]["decision_gate"],
        "locked_finalist_fingerprint": lock["lock_fingerprint"],
        "development_cutoff_timestamp": cutoff.isoformat(),
        "modeling_population": population,
        "fit": _frame_summary(modeling_frame),
    }
    artifact = save_artifact(
        registry_directory,
        candidate=candidate,
        feature_state=feature_state,
        metrics=metrics,
        settings=configuration.experiment,
        split_summary=split_summary,
        dataset_fingerprint=fingerprint,
        selected_validation_log_loss=float(lock["selection_source"]["value"]),
        artifact_purpose="development",
        selection_metric=str(lock["selection_source"]["metric"]),
        model_specification={
            "locked_finalist": lock,
            "study_report_sha256": value_fingerprint(study_report),
            "post_lock_refit": True,
            "experiment_config_sha256": config_fingerprint(configuration.experiment),
        },
    )
    report = {
        **metrics,
        "artifact_directory": str(artifact),
        "model_version": json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))[
            "model_version"
        ],
        "dataset_fingerprint": fingerprint,
    }
    _write_json(output_path, report)
    return report
