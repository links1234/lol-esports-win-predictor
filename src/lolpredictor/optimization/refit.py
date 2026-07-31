"""Refit a locked optimization finalist as a warned development artifact."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from lolpredictor.artifacts import load_artifact, save_artifact
from lolpredictor.features import generate_historical_features
from lolpredictor.models import candidate_requires_calibration, fit_candidate
from lolpredictor.optimization.models import fit_optimization_candidate
from lolpredictor.optimization.schedule import TrialSpec
from lolpredictor.optimization.settings import (
    OptimizationConfiguration,
    load_optimization_configuration,
)
from lolpredictor.optimization.splits import (
    frame_summary,
    trailing_fit_calibration_split,
)
from lolpredictor.settings import ExperimentSettings, FeatureSettings
from lolpredictor.storage import load_matches
from lolpredictor.training import dataset_fingerprint, filter_modeling_population


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _configuration_fingerprint(configuration: OptimizationConfiguration) -> str:
    return _fingerprint(
        {
            "optimization": configuration.optimization.resolved(),
            "experiment": configuration.experiment.resolved(),
        }
    )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_locked_inputs(
    database_path: Path,
    configuration: OptimizationConfiguration,
    locked: dict[str, Any],
    nested_report: dict[str, Any],
) -> None:
    if locked.get("schema_version") != "1":
        raise ValueError("Unsupported locked-finalist schema")
    if locked.get("study_name") != configuration.optimization.study_name:
        raise ValueError("Locked finalist belongs to a different study")
    if (
        locked.get("development_cutoff_timestamp")
        != configuration.optimization.development_cutoff_timestamp.isoformat()
    ):
        raise ValueError("Locked finalist uses a different development cutoff")
    if locked.get("configuration_fingerprint") != _configuration_fingerprint(configuration):
        raise ValueError("Locked finalist configuration fingerprint does not match")
    bounded_hash = dataset_fingerprint(
        database_path,
        before_timestamp=configuration.optimization.development_cutoff_timestamp,
    )
    if locked.get("dataset_fingerprint") != bounded_hash:
        raise ValueError("Locked finalist pre-cutoff dataset fingerprint does not match")
    lock_core = {
        key: value for key, value in locked.items() if key not in {"locked_at", "lock_fingerprint"}
    }
    if locked.get("lock_fingerprint") != _fingerprint(lock_core):
        raise ValueError("Locked finalist fingerprint is invalid")
    if nested_report.get("locked_finalist") != locked:
        raise ValueError("Nested report does not contain the exact locked finalist")
    report_without_fingerprint = {
        key: value for key, value in nested_report.items() if key != "report_fingerprint"
    }
    if nested_report.get("report_fingerprint") != _fingerprint(report_without_fingerprint):
        raise ValueError("Nested optimization report fingerprint is invalid")
    if nested_report.get("interpretation", {}).get("outer_outcomes_used_for_selection"):
        raise ValueError("Refusing a report that used outer outcomes for selection")


def _refit_settings(
    configuration: OptimizationConfiguration,
    feature_settings: FeatureSettings,
    *,
    specification_hash: str,
) -> ExperimentSettings:
    resolved = configuration.experiment.resolved()
    resolved["experiment_name"] = (
        f"{configuration.experiment.experiment_name}-v5-locked-{specification_hash[:8]}"
    )
    resolved["model_version"] = f"v5-locked-{specification_hash[:12]}"
    resolved["features"] = feature_settings.model_dump(mode="json")
    return ExperimentSettings.model_validate(resolved)


def refit_locked_finalist(
    database_path: Path,
    optimization_config_path: Path,
    *,
    locked_finalist_path: Path,
    nested_report_path: Path,
    registry_directory: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Refit after locking, without using the refit data to revise the specification."""
    database_path = database_path.resolve()
    configuration = load_optimization_configuration(optimization_config_path)
    locked = _read_object(locked_finalist_path.resolve())
    nested_report = _read_object(nested_report_path.resolve())
    _validate_locked_inputs(
        database_path,
        configuration,
        locked,
        nested_report,
    )

    trial_value = locked.get("trial_spec")
    selected = locked["selection"]
    if trial_value is None:
        specification_hash = str(locked["lock_fingerprint"])
        feature_settings = configuration.experiment.features
        spec = None
    else:
        if not isinstance(trial_value, dict):
            raise ValueError("Locked trial specification is malformed")
        spec = TrialSpec.from_dict(trial_value)
        if spec.spec_hash != selected.get("spec_hash"):
            raise ValueError("Locked selection and trial specification do not match")
        specification_hash = spec.spec_hash
        feature_settings = FeatureSettings.model_validate(
            {
                **configuration.experiment.features.model_dump(),
                **spec.feature_parameters,
            }
        )

    refit_settings = _refit_settings(
        configuration,
        feature_settings,
        specification_hash=specification_hash,
    )
    matches = load_matches(database_path)
    source_frame, state = generate_historical_features(matches, feature_settings)
    frame, population = filter_modeling_population(source_frame, refit_settings)

    if spec is None:
        fallback_name = str(selected["fallback_candidate"])
        requires_calibration = candidate_requires_calibration(fallback_name)
    else:
        fallback_name = ""
        requires_calibration = spec.output_mapping != "native"
    fit, calibration = trailing_fit_calibration_split(
        frame,
        requires_calibration=requires_calibration,
        calibration_fraction=(
            configuration.optimization.nested_validation.calibration_fraction_within_train
        ),
    )
    candidate = (
        fit_candidate(
            fallback_name,
            fit,
            calibration,
            refit_settings,
        )
        if spec is None
        else fit_optimization_candidate(
            spec,
            fit,
            calibration,
            refit_settings,
        )
    )
    full_dataset_hash = dataset_fingerprint(database_path)
    split_summary = {
        "fit": frame_summary(fit),
        "calibration": frame_summary(calibration) if not calibration.empty else None,
        "train": frame_summary(frame),
        "test": None,
    }
    metrics = {
        "evaluation_source": "frozen_pre2026_nested_report",
        "development_cutoff_timestamp": locked["development_cutoff_timestamp"],
        "locked_selection": selected,
        "nested_outer_policy_evaluation": nested_report["outer_policy_evaluation"],
        "refit_population": population,
        "refit_split_summary": split_summary,
        "warning": (
            "Refit coefficients may use already-opened post-cutoff outcomes, but the "
            "model and feature specification was locked using pre-cutoff data only."
        ),
    }
    model_specification = {
        "source": "locked_v5_nested_optimization",
        "study_name": configuration.optimization.study_name,
        "lock_fingerprint": locked["lock_fingerprint"],
        "selection_source": locked["selection_source"],
        "development_cutoff_timestamp": locked["development_cutoff_timestamp"],
        "selection": selected,
        "trial_spec": spec.to_dict() if spec is not None else None,
        "post_lock_refit": True,
        "promotion_status": "development_only",
    }
    artifact_directory = save_artifact(
        registry_directory,
        candidate=candidate,
        feature_state=state,
        metrics=metrics,
        settings=refit_settings,
        split_summary=split_summary,
        dataset_fingerprint=full_dataset_hash,
        selected_validation_log_loss=float(selected["pooled_inner_log_loss"]),
        artifact_purpose="development",
        selection_metric="outer_4_pooled_inner_log_loss",
        model_specification=model_specification,
    )
    artifact = load_artifact(artifact_directory)
    result = {
        "schema_version": "1",
        "artifact_directory": str(artifact_directory),
        "artifact_purpose": artifact.manifest["artifact_purpose"],
        "model_version": artifact.manifest["model_version"],
        "model_kind": artifact.manifest["model_kind"],
        "feature_schema_version": artifact.manifest["feature_schema_version"],
        "data_cutoff_timestamp": artifact.manifest["data_cutoff_timestamp"],
        "development_selection_cutoff_timestamp": (locked["development_cutoff_timestamp"]),
        "fit_sample_count": artifact.manifest["fit_sample_count"],
        "calibration_sample_count": artifact.manifest["calibration_sample_count"],
        "refit_population": population,
        "locked_finalist_fingerprint": locked["lock_fingerprint"],
        "promotion_status": "development_only",
    }
    _write_json_atomic(output_path, result)
    return result
