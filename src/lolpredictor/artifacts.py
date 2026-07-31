"""Versioned, self-describing local model artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib

from lolpredictor.features import (
    FEATURE_NAMES,
    LEGACY_FEATURE_NAMES,
    PRE_REGIONAL_FEATURE_NAMES,
    V4_FEATURE_NAMES,
    FeatureState,
)
from lolpredictor.models import CalibratedCandidate
from lolpredictor.settings import ExperimentSettings

ARTIFACT_SCHEMA_VERSION = "2"
ARTIFACT_PAYLOAD_FILES = (
    "manifest.json",
    "model.joblib",
    "feature-state.json",
    "metrics.json",
    "training-config.json",
)


@dataclass(frozen=True)
class LoadedArtifact:
    directory: Path
    manifest: dict[str, Any]
    model: CalibratedCandidate
    feature_state: FeatureState
    metrics: dict[str, Any]
    training_config: dict[str, Any]


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_fingerprint(settings: ExperimentSettings) -> str:
    encoded = json.dumps(
        settings.resolved(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _dependency_versions() -> dict[str, str]:
    dependencies = {
        "python": platform.python_version(),
        "catboost": version("catboost"),
        "duckdb": version("duckdb"),
        "joblib": version("joblib"),
        "numpy": version("numpy"),
        "openpyxl": version("openpyxl"),
        "pandas": version("pandas"),
        "pillow": version("pillow"),
        "pydantic": version("pydantic"),
        "pytz": version("pytz"),
        "pyyaml": version("pyyaml"),
        "scipy": version("scipy"),
        "scikit-learn": version("scikit-learn"),
    }
    return dependencies


def save_artifact(
    registry_directory: Path,
    *,
    candidate: CalibratedCandidate,
    feature_state: FeatureState,
    metrics: dict[str, Any],
    settings: ExperimentSettings,
    split_summary: dict[str, Any],
    dataset_fingerprint: str,
    selected_validation_log_loss: float,
    artifact_purpose: str,
    selection_metric: str = "validation_log_loss",
    model_specification: dict[str, Any] | None = None,
) -> Path:
    cutoff = feature_state.data_cutoff_timestamp
    if cutoff is None:
        raise ValueError("Cannot save an artifact without a feature-state cutoff")
    if artifact_purpose not in {"development", "evaluation", "production"}:
        raise ValueError("artifact_purpose must be development, evaluation, or production")

    registry_directory = registry_directory.resolve()
    registry_directory.mkdir(parents=True, exist_ok=True)
    configuration_hash = config_fingerprint(settings)
    model_version = (
        f"{settings.model_version}+cfg.{configuration_hash[:8]}.data.{dataset_fingerprint[:8]}"
    )
    created_at = datetime.now(UTC)
    artifact_id = (
        f"{settings.experiment_name}-{artifact_purpose}-{candidate.name}-"
        f"{created_at.strftime('%Y%m%dT%H%M%SZ')}-{configuration_hash[:8]}"
    )
    target = registry_directory / artifact_id
    if target.exists():
        raise FileExistsError(f"Artifact already exists: {target}")

    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=str(registry_directory))).resolve()
    try:
        calibration_summary = candidate.calibration_summary()
        manifest = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_purpose": artifact_purpose,
            "model_version": model_version,
            "model_kind": candidate.name,
            "feature_schema_version": settings.features.feature_schema_version,
            "all_feature_names": list(FEATURE_NAMES),
            "model_feature_names": list(candidate.feature_names),
            "created_at": created_at.isoformat(),
            "data_cutoff_timestamp": cutoff.isoformat(),
            "selection": {
                "metric": selection_metric,
                "value": selected_validation_log_loss,
            },
            "split_summary": split_summary,
            "dataset_fingerprint": dataset_fingerprint,
            "training_config_sha256": configuration_hash,
            "fit_sample_count": candidate.fit_sample_count,
            "calibration_sample_count": candidate.calibration_sample_count,
            "probability_calibration_applied": (candidate.probability_calibration_applied),
            "probability_calibration": calibration_summary,
            "stale_after_days": settings.prediction.stale_after_days,
            "dependencies": _dependency_versions(),
            "model_specification": model_specification,
        }
        (temporary / "manifest.json").write_text(_json_text(manifest), encoding="utf-8")
        joblib.dump(candidate, temporary / "model.joblib", compress=3)
        (temporary / "feature-state.json").write_text(
            _json_text(feature_state.to_dict()), encoding="utf-8"
        )
        (temporary / "metrics.json").write_text(_json_text(metrics), encoding="utf-8")
        (temporary / "training-config.json").write_text(
            _json_text(settings.resolved()), encoding="utf-8"
        )
        checksums = {
            filename: _sha256_file(temporary / filename) for filename in ARTIFACT_PAYLOAD_FILES
        }
        (temporary / "checksums.json").write_text(_json_text(checksums), encoding="utf-8")
        temporary.rename(target)
    except Exception:
        if temporary.exists() and temporary.parent == registry_directory:
            shutil.rmtree(temporary)
        raise
    return target


def load_artifact(directory: Path) -> LoadedArtifact:
    directory = directory.resolve()
    required = {*ARTIFACT_PAYLOAD_FILES, "checksums.json"}
    missing = [name for name in sorted(required) if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"Artifact is missing required files: {missing}")

    checksums = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
    if set(checksums) != set(ARTIFACT_PAYLOAD_FILES):
        raise ValueError("Artifact checksum manifest has an unexpected file set")
    for filename, expected in checksums.items():
        if Path(filename).name != filename:
            raise ValueError("Artifact checksum paths must be simple file names")
        actual = _sha256_file(directory / filename)
        if actual != expected:
            raise ValueError(f"Artifact checksum mismatch: {filename}")

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported artifact schema version")
    if manifest.get("artifact_purpose") not in {
        "development",
        "evaluation",
        "production",
    }:
        raise ValueError("Artifact purpose is missing or unsupported")
    supported_feature_contracts = {
        tuple(FEATURE_NAMES),
        tuple(V4_FEATURE_NAMES),
        tuple(PRE_REGIONAL_FEATURE_NAMES),
        tuple(LEGACY_FEATURE_NAMES),
    }
    if tuple(manifest.get("all_feature_names", ())) not in supported_feature_contracts:
        raise ValueError("Artifact feature contract does not match this package")

    feature_state = FeatureState.from_dict(
        json.loads((directory / "feature-state.json").read_text(encoding="utf-8"))
    )
    if feature_state.settings.feature_schema_version != manifest["feature_schema_version"]:
        raise ValueError("Artifact feature-state schema does not match its manifest")
    model = joblib.load(directory / "model.joblib")
    if not isinstance(model, CalibratedCandidate):
        raise TypeError("Artifact model has an unexpected type")
    if list(model.feature_names) != manifest["model_feature_names"]:
        raise ValueError("Artifact model feature list does not match its manifest")

    return LoadedArtifact(
        directory=directory,
        manifest=manifest,
        model=model,
        feature_state=feature_state,
        metrics=json.loads((directory / "metrics.json").read_text(encoding="utf-8")),
        training_config=json.loads(
            (directory / "training-config.json").read_text(encoding="utf-8")
        ),
    )
