"""Resumable orchestration for the frozen nested optimization study."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    as_completed,
    wait,
)
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from lolpredictor.evaluation import (
    clustered_log_loss_interval,
    clustered_paired_log_loss_difference_interval,
    evaluate_probabilities,
)
from lolpredictor.optimization.data import load_cached_development_frame
from lolpredictor.optimization.evaluation import (
    CONTROL_NAMES,
    controls_fingerprint,
    evaluate_inner_controls,
    evaluate_outer_control,
)
from lolpredictor.optimization.registry import StudyRegistry
from lolpredictor.optimization.schedule import (
    TrialSpec,
    generate_trial_schedule,
    schedule_fingerprint,
)
from lolpredictor.optimization.selection import select_outer_winners
from lolpredictor.optimization.settings import (
    OptimizationConfiguration,
    load_optimization_configuration,
)
from lolpredictor.optimization.splits import build_nested_folds
from lolpredictor.optimization.worker import (
    execute_outer_worker,
    execute_trial_worker,
)
from lolpredictor.training import dataset_fingerprint

OPTIMIZATION_REPORT_SCHEMA_VERSION = "1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _configuration_fingerprint(configuration: OptimizationConfiguration) -> str:
    return _fingerprint(
        {
            "optimization": configuration.optimization.resolved(),
            "experiment": configuration.experiment.resolved(),
        }
    )


def _code_fingerprint() -> str:
    package = Path(__file__).resolve().parents[1]
    paths = (
        package / "calibration.py",
        package / "evaluation.py",
        package / "features.py",
        package / "models.py",
        package / "settings.py",
        package / "splits.py",
        package / "storage.py",
        package / "training.py",
        *sorted((package / "optimization").glob("*.py")),
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _status_payload(
    registry: StudyRegistry,
    configuration: OptimizationConfiguration,
    *,
    reset_trial_count: int,
) -> dict[str, Any]:
    records = registry.trial_records()
    best_by_outer: dict[str, Any] = {}
    for outer_fold in range(
        1,
        configuration.optimization.nested_validation.outer_fold_count + 1,
    ):
        completed = [
            record
            for record in records
            if record["outer_fold"] == outer_fold
            and record["status"] == "completed"
            and record["result"] is not None
        ]
        if completed:
            winner = min(
                completed,
                key=lambda record: (
                    record["result"]["pooled_inner"]["log_loss"],
                    record["trial_id"],
                ),
            )
            best_by_outer[str(outer_fold)] = {
                "trial_id": winner["trial_id"],
                "family": winner["family"],
                "feature_mode": winner["feature_mode"],
                "log_loss": winner["result"]["pooled_inner"]["log_loss"],
            }
        else:
            best_by_outer[str(outer_fold)] = None
    state = registry.state()
    elapsed = float(state.get("active_elapsed_seconds", 0.0))
    timeout_seconds = configuration.optimization.timeout_hours * 3600.0
    return {
        "study_name": configuration.optimization.study_name,
        "development_cutoff_timestamp": (
            configuration.optimization.development_cutoff_timestamp.isoformat()
        ),
        "status_counts": registry.status_counts(),
        "best_completed_by_outer_fold": best_by_outer,
        "active_elapsed_seconds": elapsed,
        "remaining_budget_seconds": max(0.0, timeout_seconds - elapsed),
        "reset_interrupted_trial_count": reset_trial_count,
        "last_updated_at": datetime.now(UTC).isoformat(),
    }


def _ensure_controls(
    study_directory: Path,
    frame: pd.DataFrame,
    configuration: OptimizationConfiguration,
    registry: StudyRegistry,
    *,
    dataset_hash: str,
    configuration_hash: str,
) -> dict[str, Any]:
    path = study_directory / "inner-controls.json"
    if path.exists():
        envelope = _read_json(path)
        if envelope.get("dataset_fingerprint") != dataset_hash:
            raise ValueError("Cached inner controls use a different dataset")
        if envelope.get("configuration_fingerprint") != configuration_hash:
            raise ValueError("Cached inner controls use a different configuration")
        report = envelope.get("controls")
        if not isinstance(report, dict):
            raise ValueError("Cached inner controls are malformed")
    else:
        print("Computing fixed inner-fold controls...", file=sys.stderr, flush=True)
        with threadpool_limits(limits=1):
            report = evaluate_inner_controls(frame, configuration)
        envelope = {
            "schema_version": "1",
            "dataset_fingerprint": dataset_hash,
            "configuration_fingerprint": configuration_hash,
            "development_cutoff_timestamp": (
                configuration.optimization.development_cutoff_timestamp.isoformat()
            ),
            "created_at": datetime.now(UTC).isoformat(),
            "controls": report,
        }
        _write_json_atomic(path, envelope)
    fingerprint = controls_fingerprint(report)
    registry.ensure_state_value("inner_controls_fingerprint", fingerprint)
    return report


def _record_worker_envelope(
    registry: StudyRegistry,
    spec: TrialSpec,
    envelope: dict[str, Any],
) -> None:
    if envelope.get("trial_id") != spec.trial_id:
        raise ValueError("Worker returned the wrong trial identifier")
    if envelope.get("spec_hash") != spec.spec_hash:
        raise ValueError("Worker returned the wrong trial specification hash")
    result = envelope.get("result")
    if not isinstance(result, dict):
        raise ValueError("Worker returned a malformed trial result")
    registry.complete(
        spec.trial_id,
        result=result,
        duration_seconds=float(envelope["duration_seconds"]),
        worker_pid=int(envelope["worker_pid"]),
    )


def _run_trials_synchronously(
    database_path: Path,
    configuration: OptimizationConfiguration,
    registry: StudyRegistry,
    study_directory: Path,
    *,
    deadline: float,
    reset_trial_count: int,
) -> None:
    for spec in registry.pending_specs():
        if time.monotonic() >= deadline:
            break
        registry.claim(spec.trial_id)
        started = time.monotonic()
        try:
            envelope = execute_trial_worker(
                str(database_path),
                str(configuration.optimization_path),
                spec.to_dict(),
            )
            _record_worker_envelope(registry, spec, envelope)
        except Exception as error:
            registry.fail(
                spec.trial_id,
                error=error,
                duration_seconds=time.monotonic() - started,
                worker_pid=os.getpid(),
            )
        elapsed = time.monotonic() - started
        registry.add_active_elapsed(elapsed)
        status = _status_payload(
            registry,
            configuration,
            reset_trial_count=reset_trial_count,
        )
        _write_json_atomic(study_directory / "status.json", status)
        counts = status["status_counts"]
        print(
            f"Trials completed={counts['completed']} failed={counts['failed']} "
            f"pending={counts['pending']}",
            file=sys.stderr,
            flush=True,
        )


def _run_trials_parallel(
    database_path: Path,
    configuration: OptimizationConfiguration,
    registry: StudyRegistry,
    study_directory: Path,
    *,
    deadline: float,
    reset_trial_count: int,
) -> None:
    pending = iter(registry.pending_specs())
    active: dict[Future[dict[str, Any]], tuple[TrialSpec, float]] = {}
    context = multiprocessing.get_context("spawn")
    last_elapsed_update = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=configuration.optimization.worker_count,
        mp_context=context,
    ) as executor:

        def submit_available() -> None:
            while (
                len(active) < configuration.optimization.worker_count
                and time.monotonic() < deadline
            ):
                try:
                    spec = next(pending)
                except StopIteration:
                    return
                registry.claim(spec.trial_id)
                started = time.monotonic()
                try:
                    future = executor.submit(
                        execute_trial_worker,
                        str(database_path),
                        str(configuration.optimization_path),
                        spec.to_dict(),
                    )
                except Exception as error:
                    registry.fail(
                        spec.trial_id,
                        error=error,
                        duration_seconds=time.monotonic() - started,
                        worker_pid=None,
                    )
                    continue
                active[future] = (spec, started)

        submit_available()
        while active:
            timeout = max(0.1, min(30.0, deadline - time.monotonic()))
            done, _ = wait(
                active,
                timeout=timeout,
                return_when=FIRST_COMPLETED,
            )
            now = time.monotonic()
            registry.add_active_elapsed(now - last_elapsed_update)
            last_elapsed_update = now
            for future in done:
                spec, started = active.pop(future)
                try:
                    _record_worker_envelope(registry, spec, future.result())
                except Exception as error:
                    registry.fail(
                        spec.trial_id,
                        error=error,
                        duration_seconds=time.monotonic() - started,
                        worker_pid=None,
                    )
            if done:
                status = _status_payload(
                    registry,
                    configuration,
                    reset_trial_count=reset_trial_count,
                )
                _write_json_atomic(study_directory / "status.json", status)
                counts = status["status_counts"]
                print(
                    f"Trials completed={counts['completed']} failed={counts['failed']} "
                    f"running={counts['running']} pending={counts['pending']}",
                    file=sys.stderr,
                    flush=True,
                )
            submit_available()


def _run_pending_trials(
    database_path: Path,
    configuration: OptimizationConfiguration,
    registry: StudyRegistry,
    study_directory: Path,
    *,
    reset_trial_count: int,
) -> None:
    state = registry.state()
    elapsed = float(state.get("active_elapsed_seconds", 0.0))
    remaining = max(
        0.0,
        configuration.optimization.timeout_hours * 3600.0 - elapsed,
    )
    deadline = time.monotonic() + remaining
    if configuration.optimization.worker_count == 1:
        _run_trials_synchronously(
            database_path,
            configuration,
            registry,
            study_directory,
            deadline=deadline,
            reset_trial_count=reset_trial_count,
        )
    else:
        _run_trials_parallel(
            database_path,
            configuration,
            registry,
            study_directory,
            deadline=deadline,
            reset_trial_count=reset_trial_count,
        )


def _locked_finalist(
    study_directory: Path,
    selection: dict[str, Any],
    specs_by_id: dict[int, TrialSpec],
    configuration: OptimizationConfiguration,
    *,
    dataset_hash: str,
    configuration_hash: str,
) -> dict[str, Any]:
    final_outer = str(selection["locked_finalist_outer_fold"])
    selected = selection["outer_selections"][final_outer]["selection"]
    spec = (
        specs_by_id[int(selected["trial_id"])].to_dict()
        if selected["trial_id"] is not None
        else None
    )
    core = {
        "schema_version": "1",
        "study_name": configuration.optimization.study_name,
        "development_cutoff_timestamp": (
            configuration.optimization.development_cutoff_timestamp.isoformat()
        ),
        "dataset_fingerprint": dataset_hash,
        "configuration_fingerprint": configuration_hash,
        "selection_source": "outer_4_inner_results_only",
        "selection": selected,
        "trial_spec": spec,
    }
    path = study_directory / "locked-finalist.json"
    if path.exists():
        existing = _read_json(path)
        existing_core = {key: existing.get(key) for key in core}
        if existing_core != core:
            raise ValueError("Locked finalist differs from the current inner-only selection")
        return existing
    locked = {
        **core,
        "locked_at": datetime.now(UTC).isoformat(),
        "lock_fingerprint": _fingerprint(core),
    }
    _write_json_atomic(path, locked)
    return locked


def _evaluate_selected_outer_folds(
    database_path: Path,
    configuration: OptimizationConfiguration,
    frame: pd.DataFrame,
    selection: dict[str, Any],
    specs_by_id: dict[int, TrialSpec],
) -> tuple[dict[str, Any], dict[str, Any]]:
    folds = build_nested_folds(
        frame,
        configuration.optimization.nested_validation,
    )
    selected_reports: dict[str, Any] = {}
    searched_specs: list[TrialSpec] = []
    for fold in folds:
        selected = selection["outer_selections"][str(fold.fold_number)]["selection"]
        if selected["selection_kind"] == "searched_trial":
            searched_specs.append(specs_by_id[int(selected["trial_id"])])
        else:
            selected_reports[str(fold.fold_number)] = evaluate_outer_control(
                str(selected["fallback_candidate"]),
                fold,
                configuration,
            )

    if searched_specs:
        if configuration.optimization.worker_count == 1:
            envelopes = [
                execute_outer_worker(
                    str(database_path),
                    str(configuration.optimization_path),
                    spec.to_dict(),
                )
                for spec in searched_specs
            ]
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=min(
                    configuration.optimization.worker_count,
                    len(searched_specs),
                ),
                mp_context=context,
            ) as executor:
                future_specs = {
                    executor.submit(
                        execute_outer_worker,
                        str(database_path),
                        str(configuration.optimization_path),
                        spec.to_dict(),
                    ): spec
                    for spec in searched_specs
                }
                envelopes = [future.result() for future in as_completed(future_specs)]
        for envelope in envelopes:
            result = envelope["result"]
            selected_reports[str(result["outer_fold"])] = result

    control_reports: dict[str, Any] = {}
    print("Evaluating locked outer-fold selections and controls...", file=sys.stderr, flush=True)
    with threadpool_limits(limits=1):
        for fold in folds:
            control_reports[str(fold.fold_number)] = {
                name: evaluate_outer_control(name, fold, configuration) for name in CONTROL_NAMES
            }
    return selected_reports, control_reports


def _pooled_outer_report(
    frame: pd.DataFrame,
    selected_reports: dict[str, Any],
    control_reports: dict[str, Any],
    configuration: OptimizationConfiguration,
) -> dict[str, Any]:
    ordered_selected = [
        selected_reports[str(fold_number)]
        for fold_number in range(
            1,
            configuration.optimization.nested_validation.outer_fold_count + 1,
        )
    ]
    match_ids = [match_id for report in ordered_selected for match_id in report["match_ids"]]
    if len(match_ids) != len(set(match_ids)):
        raise ValueError("Outer score intervals contain duplicate match IDs")
    indexed = frame.set_index(frame["match_id"].astype(str), drop=False)
    pooled_frame = indexed.loc[match_ids].reset_index(drop=True)
    selected_probabilities = np.asarray(
        [probability for report in ordered_selected for probability in report["probabilities"]],
        dtype=float,
    )
    if pooled_frame["match_id"].astype(str).tolist() != match_ids:
        raise ValueError("Outer prediction rows do not align with the bounded feature frame")
    labels = pooled_frame["blue_win"].to_numpy(dtype=int)
    clusters = pooled_frame["series_id"].fillna(pooled_frame["match_id"]).to_numpy(dtype=str)
    iterations = configuration.experiment.backtest.bootstrap_iterations
    seed = configuration.optimization.random_seed
    selected_metrics = evaluate_probabilities(
        pooled_frame,
        selected_probabilities,
        include_breakdowns=True,
    )
    selected_metrics["series_clustered_log_loss_interval"] = clustered_log_loss_interval(
        labels,
        selected_probabilities,
        clusters,
        iterations=iterations,
        random_seed=seed,
    )

    controls: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for control_index, name in enumerate(CONTROL_NAMES):
        probabilities: list[float] = []
        control_match_ids: list[str] = []
        for fold_number in range(
            1,
            configuration.optimization.nested_validation.outer_fold_count + 1,
        ):
            report = control_reports[str(fold_number)][name]
            control_match_ids.extend(report["match_ids"])
            probabilities.extend(report["probabilities"])
        if control_match_ids != match_ids:
            raise ValueError(f"Outer control {name} does not align with selected predictions")
        control_probabilities = np.asarray(probabilities, dtype=float)
        metrics = evaluate_probabilities(
            pooled_frame,
            control_probabilities,
            include_breakdowns=True,
        )
        metrics["series_clustered_log_loss_interval"] = clustered_log_loss_interval(
            labels,
            control_probabilities,
            clusters,
            iterations=iterations,
            random_seed=seed + control_index + 1,
        )
        controls[name] = metrics
        paired[name] = clustered_paired_log_loss_difference_interval(
            labels,
            selected_probabilities,
            control_probabilities,
            clusters,
            iterations=iterations,
            random_seed=seed + 100 + control_index,
        )

    return {
        "score_sample_count": len(pooled_frame),
        "score_start_timestamp": pd.to_datetime(
            pooled_frame["match_timestamp"],
            utc=True,
        )
        .min()
        .isoformat(),
        "score_end_timestamp": pd.to_datetime(
            pooled_frame["match_timestamp"],
            utc=True,
        )
        .max()
        .isoformat(),
        "selected_policy": selected_metrics,
        "controls": controls,
        "paired_selected_minus_control_log_loss_intervals": paired,
        "outer_fold_selections": {
            fold_number: {
                key: value
                for key, value in report.items()
                if key not in {"match_ids", "probabilities"}
            }
            for fold_number, report in selected_reports.items()
        },
    }


def run_optimization(
    database_path: Path,
    optimization_config_path: Path,
    study_directory: Path,
) -> dict[str, Any]:
    """Run or resume the frozen nested study and write its locked report."""
    database_path = database_path.resolve()
    study_directory = study_directory.resolve()
    configuration = load_optimization_configuration(optimization_config_path)
    specs = generate_trial_schedule(configuration.optimization)
    specs_by_id = {spec.trial_id: spec for spec in specs}
    configuration_hash = _configuration_fingerprint(configuration)
    schedule_hash = schedule_fingerprint(specs)
    code_hash = _code_fingerprint()
    cutoff = configuration.optimization.development_cutoff_timestamp
    dataset_hash = dataset_fingerprint(
        database_path,
        before_timestamp=cutoff,
    )
    frame, population = load_cached_development_frame(
        database_path,
        configuration.experiment,
        cutoff=cutoff,
    )
    folds = build_nested_folds(
        frame,
        configuration.optimization.nested_validation,
    )
    contract = {
        "study_name": configuration.optimization.study_name,
        "development_cutoff_timestamp": cutoff.isoformat(),
        "dataset_fingerprint": dataset_hash,
        "configuration_fingerprint": configuration_hash,
        "schedule_fingerprint": schedule_hash,
        "code_fingerprint": code_hash,
        "trial_count": configuration.optimization.trial_count,
    }

    with StudyRegistry(study_directory) as registry, registry.exclusive_lock():
        registry.initialize(contract=contract, specs=specs)
        reset_trial_count = registry.reset_interrupted_trials()
        resolved = {
            "schema_version": "1",
            "contract": contract,
            "configuration": configuration.resolved(),
            "population": population,
            "nested_folds": [fold.summary() for fold in folds],
        }
        _write_json_atomic(study_directory / "resolved-study.json", resolved)

        controls_started = time.monotonic()
        controls = _ensure_controls(
            study_directory,
            frame,
            configuration,
            registry,
            dataset_hash=dataset_hash,
            configuration_hash=configuration_hash,
        )
        registry.add_active_elapsed(time.monotonic() - controls_started)
        _write_json_atomic(
            study_directory / "status.json",
            _status_payload(
                registry,
                configuration,
                reset_trial_count=reset_trial_count,
            ),
        )
        _run_pending_trials(
            database_path,
            configuration,
            registry,
            study_directory,
            reset_trial_count=reset_trial_count,
        )

        records = registry.trial_records()
        selection = select_outer_winners(
            records,
            controls,
            configuration,
        )
        selection_hash = _fingerprint(selection)
        registry.ensure_state_value("inner_only_selection_fingerprint", selection_hash)
        locked = _locked_finalist(
            study_directory,
            selection,
            specs_by_id,
            configuration,
            dataset_hash=dataset_hash,
            configuration_hash=configuration_hash,
        )
        selected_outer, control_outer = _evaluate_selected_outer_folds(
            database_path,
            configuration,
            frame,
            selection,
            specs_by_id,
        )
        outer_report = _pooled_outer_report(
            frame,
            selected_outer,
            control_outer,
            configuration,
        )
        status = _status_payload(
            registry,
            configuration,
            reset_trial_count=reset_trial_count,
        )
        report = {
            "schema_version": OPTIMIZATION_REPORT_SCHEMA_VERSION,
            "study_name": configuration.optimization.study_name,
            "generated_at": datetime.now(UTC).isoformat(),
            "development_cutoff_timestamp": cutoff.isoformat(),
            "contract": contract,
            "population": population,
            "nested_folds": [fold.summary() for fold in folds],
            "budget": {
                "trial_count": configuration.optimization.trial_count,
                "timeout_hours": configuration.optimization.timeout_hours,
                "worker_count": configuration.optimization.worker_count,
                "status_counts": status["status_counts"],
                "active_elapsed_seconds": status["active_elapsed_seconds"],
                "remaining_budget_seconds": status["remaining_budget_seconds"],
            },
            "selection": selection,
            "locked_finalist": locked,
            "outer_policy_evaluation": outer_report,
            "failed_trials": [
                {
                    "trial_id": record["trial_id"],
                    "family": record["family"],
                    "feature_mode": record["feature_mode"],
                    "error": record["error"],
                }
                for record in records
                if record["status"] == "failed"
            ],
            "interpretation": {
                "primary_metric": "log_loss",
                "outer_outcomes_used_for_selection": False,
                "outcomes_at_or_after_cutoff_loaded": False,
                "promotion_status": "development_only",
            },
        }
        report["report_fingerprint"] = _fingerprint(report)
        _write_json_atomic(study_directory / "nested-report.json", report)
        _write_json_atomic(study_directory / "status.json", status)
        return {
            "study_directory": str(study_directory),
            "registry": str(registry.database_path),
            "report": str(study_directory / "nested-report.json"),
            "locked_finalist": str(study_directory / "locked-finalist.json"),
            "development_cutoff_timestamp": cutoff.isoformat(),
            "status_counts": status["status_counts"],
            "selected_policy_log_loss": outer_report["selected_policy"]["log_loss"],
            "selected_policy_brier_score": outer_report["selected_policy"]["brier_score"],
            "selected_policy_expected_calibration_error": outer_report["selected_policy"][
                "expected_calibration_error"
            ],
        }


def optimization_status(study_directory: Path) -> dict[str, Any]:
    """Read a study status without changing its registry."""
    status_path = study_directory.resolve() / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"Optimization status does not exist: {status_path}")
    return _read_json(status_path)
