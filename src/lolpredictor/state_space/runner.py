"""Atomic orchestration for the preregistered v6 state-space study."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from lolpredictor.features import (
    FEATURE_NAMES,
    STATE_SPACE_FEATURE_NAMES,
    V4_FEATURE_NAMES,
    generate_historical_features,
)
from lolpredictor.schemas import HistoricalMatch
from lolpredictor.state_space.evaluation import evaluate_v6_nested
from lolpredictor.state_space.filter import skill_kind
from lolpredictor.state_space.models import STATE_SPACE_AUGMENTED_FEATURES
from lolpredictor.state_space.settings import (
    V4_CONTROL_NAME,
    V6Configuration,
    load_v6_configuration,
)
from lolpredictor.storage import load_matches
from lolpredictor.training import filter_modeling_population


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def value_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _matches_fingerprint(matches: list[HistoricalMatch]) -> str:
    digest = hashlib.sha256()
    for match in matches:
        digest.update(canonical_json_bytes(match.model_dump(mode="json")))
        digest.update(b"\n")
    return digest.hexdigest()


def _source_summary(matches: list[HistoricalMatch]) -> dict[str, Any]:
    timestamps = pd.to_datetime(
        [match.match_timestamp for match in matches],
        utc=True,
    )
    return {
        "sample_count": len(matches),
        "timestamp_count": int(timestamps.nunique()),
        "start_timestamp": timestamps.min().isoformat(),
        "end_timestamp": timestamps.max().isoformat(),
    }


def _configuration_fingerprint(configuration: V6Configuration) -> str:
    return value_fingerprint(configuration.resolved())


def _candidate_specification(
    candidate_name: str,
    configuration: V6Configuration,
) -> dict[str, Any]:
    candidate = configuration.study.candidates
    common = {
        "candidate_name": candidate_name,
        "state_space": configuration.study.state_space.model_dump(mode="json"),
        "feature_schema_version": (configuration.experiment.features.feature_schema_version),
        "v4_control_name": V4_CONTROL_NAME,
        "v4_control_feature_names": list(V4_FEATURE_NAMES),
    }
    if candidate_name == "state_space_native":
        return {
            **common,
            "model_family": "column_probability",
            "model_feature_names": ["state_space_blue_win_probability"],
            "output_mapping": "native",
        }
    if candidate_name == "state_space_platt":
        return {
            **common,
            "model_family": "column_probability",
            "model_feature_names": ["state_space_blue_win_probability"],
            "output_mapping": "platt",
            "calibration_fraction": candidate.state_calibration_fraction,
        }
    if candidate_name == "state_space_augmented_logistic":
        return {
            **common,
            "model_family": "l2_logistic_regression",
            "model_feature_names": list(STATE_SPACE_AUGMENTED_FEATURES),
            "logistic_c": candidate.augmented_logistic_c,
            "output_mapping": "native",
        }
    if candidate_name == "state_space_v4_blend_15":
        return {
            **common,
            "model_family": "fixed_probability_blend",
            "v4_weight": 1.0 - candidate.blend_15_state_weight,
            "augmented_logistic_weight": candidate.blend_15_state_weight,
            "augmented_logistic_c": candidate.augmented_logistic_c,
            "model_feature_names": list(
                dict.fromkeys((*V4_FEATURE_NAMES, *STATE_SPACE_AUGMENTED_FEATURES))
            ),
            "output_mapping": "native",
        }
    if candidate_name == "state_space_v4_blend_30":
        return {
            **common,
            "model_family": "fixed_probability_blend",
            "v4_weight": 1.0 - candidate.blend_30_state_weight,
            "augmented_logistic_weight": candidate.blend_30_state_weight,
            "augmented_logistic_c": candidate.augmented_logistic_c,
            "model_feature_names": list(
                dict.fromkeys((*V4_FEATURE_NAMES, *STATE_SPACE_AUGMENTED_FEATURES))
            ),
            "output_mapping": "native",
        }
    if candidate_name == V4_CONTROL_NAME:
        return {
            **common,
            "model_family": "fixed_v4_fallback",
            "model_feature_names": list(V4_FEATURE_NAMES),
            "output_mapping": "native",
        }
    raise ValueError(f"Unsupported locked v6 candidate: {candidate_name}")


def validate_locked_finalist(value: dict[str, Any]) -> str:
    """Validate a lock's canonical fingerprint and return its candidate name."""
    if value.get("schema_version") != "1":
        raise ValueError("Unsupported v6 locked-finalist schema")
    expected = value.get("lock_fingerprint")
    if not isinstance(expected, str):
        raise ValueError("V6 locked finalist is missing its fingerprint")
    unsigned = {key: item for key, item in value.items() if key != "lock_fingerprint"}
    if value_fingerprint(unsigned) != expected:
        raise ValueError("V6 locked-finalist fingerprint mismatch")
    candidate_name = value.get("specification", {}).get("candidate_name")
    if not isinstance(candidate_name, str):
        raise ValueError("V6 locked finalist is missing its candidate name")
    return candidate_name


def run_v6_study(
    database_path: Path,
    config_path: Path,
    study_directory: Path,
) -> dict[str, Any]:
    """Replay bounded data, run nested selection, and atomically write its lock."""
    started = time.monotonic()
    configuration = load_v6_configuration(config_path)
    cutoff = configuration.study.development_cutoff_timestamp
    matches = load_matches(database_path, before_timestamp=cutoff)
    if not matches:
        raise ValueError("No matches are available before the v6 cutoff")
    if any(match.match_timestamp >= cutoff for match in matches):
        raise ValueError("V6 storage boundary returned an outcome at or after its cutoff")
    source_fingerprint = _matches_fingerprint(matches)
    source_summary = _source_summary(matches)
    feature_frame, feature_state = generate_historical_features(
        matches,
        configuration.experiment.features,
    )
    timestamps = pd.to_datetime(feature_frame["match_timestamp"], utc=True)
    if bool((timestamps >= pd.Timestamp(cutoff)).any()):
        raise ValueError("V6 feature replay crossed its exclusive cutoff")
    modeling_frame, population = filter_modeling_population(
        feature_frame,
        configuration.experiment,
    )
    nested = evaluate_v6_nested(modeling_frame, configuration)
    configuration_fingerprint = _configuration_fingerprint(configuration)
    locked_summary = nested["locked_candidate"]
    candidate_name = str(locked_summary["name"])
    specification = _candidate_specification(candidate_name, configuration)
    unsigned_lock = {
        "schema_version": "1",
        "study_name": configuration.study.study_name,
        "development_cutoff_timestamp": cutoff.isoformat(),
        "configuration_fingerprint": configuration_fingerprint,
        "bounded_dataset_fingerprint": source_fingerprint,
        "selection_source": {
            "outer_fold": int(locked_summary["source_outer_fold"]),
            "metric": str(locked_summary["selection_metric"]),
            "value": float(locked_summary["selection_value"]),
        },
        "specification": specification,
    }
    locked_finalist = {
        **unsigned_lock,
        "lock_fingerprint": value_fingerprint(unsigned_lock),
    }
    latent_counts = Counter(skill_kind(key) for key in feature_state.state_space_skills)
    report = {
        "schema_version": "1",
        "study_name": configuration.study.study_name,
        "status": "complete",
        "development_cutoff_exclusive": cutoff.isoformat(),
        "bounded_dataset_fingerprint": source_fingerprint,
        "configuration_fingerprint": configuration_fingerprint,
        "source": source_summary,
        "modeling_population": population,
        "feature_replay": {
            "feature_schema_version": (configuration.experiment.features.feature_schema_version),
            "all_feature_count": len(FEATURE_NAMES),
            "state_space_feature_names": list(STATE_SPACE_FEATURE_NAMES),
            "state_cutoff_timestamp": (
                feature_state.data_cutoff_timestamp.isoformat()
                if feature_state.data_cutoff_timestamp is not None
                else None
            ),
            "latent_component_counts": dict(sorted(latent_counts.items())),
            "latent_component_count": len(feature_state.state_space_skills),
        },
        "nested_evaluation": nested,
        "locked_finalist": locked_finalist,
        "elapsed_seconds": time.monotonic() - started,
    }

    target = study_directory.resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite an existing v6 study: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".v6-building-", dir=str(target.parent))).resolve()
    try:
        _write_json(temporary / "resolved-study.json", configuration.resolved())
        _write_json(temporary / "locked-finalist.json", locked_finalist)
        _write_json(temporary / "study-report.json", report)
        temporary.rename(target)
    except Exception:
        if temporary.exists() and temporary.parent == target.parent:
            shutil.rmtree(temporary)
        raise
    return {
        "status": "complete",
        "study_directory": str(target),
        "report": str(target / "study-report.json"),
        "locked_finalist": str(target / "locked-finalist.json"),
        "development_cutoff_exclusive": cutoff.isoformat(),
        "source_sample_count": source_summary["sample_count"],
        "modeling_sample_count": population["selected_sample_count"],
        "pooled_outer_sample_count": nested["pooled_outer_score"]["sample_count"],
        "selection_policy_log_loss": nested["selection_policy"]["metrics"]["log_loss"],
        "v4_control_log_loss": nested["controls"][V4_CONTROL_NAME]["metrics"]["log_loss"],
        "decision_gate": nested["decision_gate"],
        "locked_candidate": candidate_name,
        "lock_fingerprint": locked_finalist["lock_fingerprint"],
        "elapsed_seconds": report["elapsed_seconds"],
    }
