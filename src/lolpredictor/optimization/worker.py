"""Isolated optimizer worker entrypoints."""

from __future__ import annotations

import gc
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from threadpoolctl import threadpool_limits

from lolpredictor.optimization.data import (
    load_cached_development_frame,
    replay_development_matches,
)
from lolpredictor.optimization.evaluation import evaluate_outer_spec, evaluate_trial
from lolpredictor.optimization.schedule import TrialSpec
from lolpredictor.optimization.settings import (
    OptimizationConfiguration,
    load_optimization_configuration,
)
from lolpredictor.schemas import HistoricalMatch
from lolpredictor.storage import load_matches

_CONFIGURATIONS: dict[str, OptimizationConfiguration] = {}
_CACHED_FRAMES: dict[tuple[str, str], pd.DataFrame] = {}
_BOUNDED_MATCHES: dict[tuple[str, str], list[HistoricalMatch]] = {}


def _configuration(path: str) -> OptimizationConfiguration:
    resolved = str(Path(path).resolve())
    if resolved not in _CONFIGURATIONS:
        _CONFIGURATIONS[resolved] = load_optimization_configuration(Path(resolved))
    return _CONFIGURATIONS[resolved]


def _cache_key(database_path: str, cutoff: datetime) -> tuple[str, str]:
    return str(Path(database_path).resolve()), cutoff.isoformat()


def _cached_frame(
    database_path: str,
    configuration: OptimizationConfiguration,
) -> pd.DataFrame:
    cutoff = configuration.optimization.development_cutoff_timestamp
    key = _cache_key(database_path, cutoff)
    if key not in _CACHED_FRAMES:
        frame, _ = load_cached_development_frame(
            Path(database_path),
            configuration.experiment,
            cutoff=cutoff,
        )
        _CACHED_FRAMES[key] = frame
    return _CACHED_FRAMES[key]


def _bounded_matches(
    database_path: str,
    configuration: OptimizationConfiguration,
) -> list[HistoricalMatch]:
    cutoff = configuration.optimization.development_cutoff_timestamp
    key = _cache_key(database_path, cutoff)
    if key not in _BOUNDED_MATCHES:
        _BOUNDED_MATCHES[key] = load_matches(
            Path(database_path),
            before_timestamp=cutoff,
        )
    return _BOUNDED_MATCHES[key]


def _trial_frame(
    database_path: str,
    configuration: OptimizationConfiguration,
    spec: TrialSpec,
) -> pd.DataFrame:
    cached = _cached_frame(database_path, configuration)
    if spec.feature_mode == "cached":
        return cached
    replayed, _ = replay_development_matches(
        _bounded_matches(database_path, configuration),
        configuration.experiment,
        cutoff=configuration.optimization.development_cutoff_timestamp,
        feature_parameters=spec.feature_parameters,
    )
    cached_ids = cached["match_id"].astype(str).tolist()
    replayed_ids = replayed["match_id"].astype(str).tolist()
    if replayed_ids != cached_ids:
        raise ValueError("Feature replay changed the bounded modeling population or row order")
    return replayed


def execute_trial_worker(
    database_path: str,
    optimization_config_path: str,
    spec_value: dict[str, Any],
) -> dict[str, Any]:
    """Run one inner-only trial and return a main-process registry envelope."""
    started = time.monotonic()
    spec = TrialSpec.from_dict(spec_value)
    configuration = _configuration(optimization_config_path)
    with threadpool_limits(limits=1):
        result = evaluate_trial(
            spec,
            _trial_frame(database_path, configuration, spec),
            configuration,
        )
    gc.collect()
    return {
        "trial_id": spec.trial_id,
        "spec_hash": spec.spec_hash,
        "worker_pid": os.getpid(),
        "duration_seconds": time.monotonic() - started,
        "result": result,
    }


def execute_outer_worker(
    database_path: str,
    optimization_config_path: str,
    spec_value: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one inner-selected trial on its assigned outer interval."""
    started = time.monotonic()
    spec = TrialSpec.from_dict(spec_value)
    configuration = _configuration(optimization_config_path)
    with threadpool_limits(limits=1):
        result = evaluate_outer_spec(
            spec,
            _trial_frame(database_path, configuration, spec),
            configuration,
        )
    gc.collect()
    return {
        "trial_id": spec.trial_id,
        "spec_hash": spec.spec_hash,
        "worker_pid": os.getpid(),
        "duration_seconds": time.monotonic() - started,
        "result": result,
    }
