"""Feature materialization, baseline training, selection, and holdout evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lolpredictor.artifacts import load_artifact, save_artifact
from lolpredictor.evaluation import (
    clustered_log_loss_interval,
    clustered_paired_log_loss_difference_interval,
    evaluate_candidate,
    evaluate_probabilities,
)
from lolpredictor.features import (
    build_state_until,
    generate_historical_features,
    validate_feature_frame,
)
from lolpredictor.models import (
    CANDIDATE_NAMES,
    SIMPLE_CONTROL_NAMES,
    CalibratedCandidate,
    candidate_requires_calibration,
    configured_candidate_names,
    fit_all_candidates,
    fit_candidate,
)
from lolpredictor.settings import ExperimentSettings
from lolpredictor.splits import chronological_split, rolling_origin_splits
from lolpredictor.storage import (
    get_metadata,
    load_feature_table,
    load_matches,
    write_feature_table,
)

TRAINING_REPORT_SCHEMA_VERSION = "2"


def filter_modeling_population(
    frame: pd.DataFrame,
    settings: ExperimentSettings,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply only structural, pregame population rules to supervised rows."""
    selected = pd.Series(True, index=frame.index)
    if settings.modeling_population.require_official:
        selected &= frame["is_official"].eq(True)
    configured_levels = settings.modeling_population.tournament_levels
    if configured_levels:
        selected &= frame["tournament_level"].isin(configured_levels)
    filtered = (
        frame.loc[selected].sort_values(["match_timestamp", "match_id"]).reset_index(drop=True)
    )
    if filtered.empty:
        raise ValueError("Modeling-population rules excluded every feature row")
    return filtered, {
        "source_sample_count": len(frame),
        "selected_sample_count": len(filtered),
        "excluded_sample_count": len(frame) - len(filtered),
        "require_official": settings.modeling_population.require_official,
        "tournament_levels": list(configured_levels),
    }


def dataset_fingerprint(
    database_path: Path,
    *,
    before_timestamp: datetime | None = None,
) -> str:
    """Fingerprint canonical matches, optionally bounded by an exclusive cutoff."""
    matches = load_matches(database_path, before_timestamp=before_timestamp)
    digest = hashlib.sha256()
    for match in matches:
        payload = json.dumps(
            match.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(payload)
        digest.update(b"\n")
    return digest.hexdigest()


def materialize_historical_features(
    database_path: Path,
    settings: ExperimentSettings,
) -> dict[str, Any]:
    matches = load_matches(database_path)
    frame, state = generate_historical_features(matches, settings.features)
    write_feature_table(
        database_path,
        frame,
        feature_schema_version=settings.features.feature_schema_version,
    )
    return {
        "database": str(database_path.resolve()),
        "feature_schema_version": settings.features.feature_schema_version,
        "feature_row_count": len(frame),
        "history_cutoff_timestamp": (
            state.data_cutoff_timestamp.isoformat()
            if state.data_cutoff_timestamp is not None
            else None
        ),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _evaluate_with_cluster_interval(
    candidate: CalibratedCandidate,
    frame: pd.DataFrame,
    settings: ExperimentSettings,
    *,
    include_breakdowns: bool,
) -> dict[str, Any]:
    metrics = evaluate_candidate(
        candidate,
        frame,
        include_breakdowns=include_breakdowns,
    )
    probabilities = candidate.predict_probability(frame)
    clusters = frame["series_id"] if "series_id" in frame else frame["match_id"]
    metrics["series_clustered_log_loss_interval"] = clustered_log_loss_interval(
        frame["blue_win"].to_numpy(dtype=int),
        probabilities,
        clusters.fillna(frame["match_id"]).to_numpy(dtype=str),
        iterations=settings.backtest.bootstrap_iterations,
        random_seed=settings.random_seed,
    )
    return metrics


def _trailing_calibration_split(
    frame: pd.DataFrame,
    *,
    calibration_fraction: float,
    minimum_fit_timestamp_groups: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(["match_timestamp", "match_id"]).reset_index(drop=True)
    timestamp_series = pd.to_datetime(ordered["match_timestamp"], utc=True)
    unique_timestamps = sorted(timestamp_series.unique().tolist())
    calibration_count = max(
        2,
        math.ceil(len(unique_timestamps) * calibration_fraction),
    )
    if len(unique_timestamps) - calibration_count < minimum_fit_timestamp_groups:
        raise ValueError(
            f"Trailing calibration split needs at least {minimum_fit_timestamp_groups} "
            "fit timestamp groups"
        )
    calibration_timestamps = unique_timestamps[-calibration_count:]
    fit_timestamps = unique_timestamps[:-calibration_count]
    fit = ordered[timestamp_series.isin(fit_timestamps)].reset_index(drop=True)
    calibration = ordered[timestamp_series.isin(calibration_timestamps)].reset_index(drop=True)
    return fit, calibration


def _refit_partitions(
    frame: pd.DataFrame,
    candidate_name: str,
    settings: ExperimentSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(["match_timestamp", "match_id"]).reset_index(drop=True)
    if not candidate_requires_calibration(candidate_name):
        return ordered, ordered.iloc[0:0].copy()
    return _trailing_calibration_split(
        ordered,
        calibration_fraction=settings.models.production_calibration_fraction,
        minimum_fit_timestamp_groups=10,
    )


def _paired_log_loss_interval(
    frame: pd.DataFrame,
    candidate_probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
    settings: ExperimentSettings,
) -> dict[str, float | int] | None:
    clusters = frame["series_id"] if "series_id" in frame else frame["match_id"]
    return clustered_paired_log_loss_difference_interval(
        frame["blue_win"].to_numpy(dtype=int),
        candidate_probabilities,
        reference_probabilities,
        clusters.fillna(frame["match_id"]).to_numpy(dtype=str),
        iterations=settings.backtest.bootstrap_iterations,
        random_seed=settings.random_seed,
    )


def _development_candidate_eligibility(
    candidate_name: str,
    *,
    aggregate_candidates: dict[str, dict[str, Any]],
    fold_reports: list[dict[str, Any]],
    best_simple_control: str,
    settings: ExperimentSettings,
) -> dict[str, Any]:
    candidate = aggregate_candidates[candidate_name]
    elo = aggregate_candidates["elo_only"]
    paired_interval = candidate["paired_log_loss_difference_vs_elo_interval"]
    fold_differences = [
        {
            "fold_number": fold["fold_number"],
            "log_loss_difference_vs_elo": (
                fold["candidates"][candidate_name]["test"]["log_loss"]
                - fold["candidates"]["elo_only"]["test"]["log_loss"]
            ),
        }
        for fold in fold_reports
    ]
    candidate_leagues = candidate["breakdowns"]["league"]
    elo_leagues = elo["breakdowns"]["league"]
    major_league_differences: dict[str, dict[str, float | int]] = {}
    for league in settings.release_gate.major_leagues:
        candidate_metrics = candidate_leagues.get(league)
        elo_metrics = elo_leagues.get(league)
        if candidate_metrics is None or elo_metrics is None:
            continue
        sample_count = int(candidate_metrics["sample_count"])
        if sample_count < settings.backtest.minimum_breakdown_sample_count:
            continue
        major_league_differences[league] = {
            "sample_count": sample_count,
            "log_loss_difference_vs_elo": (candidate_metrics["log_loss"] - elo_metrics["log_loss"]),
        }

    checks = {
        "beats_elo": candidate["log_loss"] < elo["log_loss"],
        "beats_best_simple_control": (
            candidate["log_loss"] < aggregate_candidates[best_simple_control]["log_loss"]
        ),
        "series_clustered_interval_beats_elo": (
            paired_interval is not None and paired_interval["upper"] < 0.0
        ),
        "improves_or_matches_elo_in_every_fold": all(
            fold["log_loss_difference_vs_elo"] <= 0.0 for fold in fold_differences
        ),
        "calibration_within_limit": (
            candidate["expected_calibration_error"]
            <= settings.release_gate.maximum_expected_calibration_error
        ),
        "major_league_regressions_within_limit": all(
            metrics["log_loss_difference_vs_elo"]
            <= settings.release_gate.maximum_major_league_log_loss_regression
            for metrics in major_league_differences.values()
        ),
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "fold_log_loss_differences_vs_elo": fold_differences,
        "major_league_log_loss_differences_vs_elo": major_league_differences,
        "minimum_breakdown_sample_count": (settings.backtest.minimum_breakdown_sample_count),
    }


def train_baselines(
    database_path: Path,
    settings: ExperimentSettings,
    *,
    registry_directory: Path,
    reports_directory: Path,
) -> dict[str, Any]:
    metadata = get_metadata(database_path)
    if metadata.get("feature_schema_version") != settings.features.feature_schema_version:
        raise ValueError(
            "Feature table is missing or was generated with a different schema version"
        )
    source_frame = load_feature_table(database_path)
    validate_feature_frame(source_frame)
    frame, population_summary = filter_modeling_population(source_frame, settings)
    splits = chronological_split(frame, settings.splits)
    selection_candidates = fit_all_candidates(splits.fit, splits.calibration, settings)
    validation_metrics = {
        name: evaluate_candidate(
            candidate,
            splits.validation,
            include_breakdowns=False,
        )
        for name, candidate in selection_candidates.items()
    }
    selected_name = min(
        validation_metrics,
        key=lambda name: validation_metrics[name]["log_loss"],
    )
    selected_validation_log_loss = float(validation_metrics[selected_name]["log_loss"])
    selection_split_summary = splits.summary()

    if settings.splits.refit_before_test:
        pretest = pd.concat(
            [splits.training, splits.validation],
            ignore_index=True,
        ).sort_values(["match_timestamp", "match_id"])
        candidate_names = configured_candidate_names(settings)
        refit_partitions = {
            name: _refit_partitions(pretest, name, settings) for name in candidate_names
        }
        test_candidates = {
            name: fit_candidate(
                name,
                refit_partitions[name][0],
                refit_partitions[name][1],
                settings,
            )
            for name in candidate_names
        }
        release_fit, release_calibration = refit_partitions[selected_name]
        split_summary = {
            "fit": _production_frame_summary(release_fit),
            "calibration": _production_frame_summary(release_calibration),
            "train": _production_frame_summary(pretest),
            "validation": selection_split_summary["validation"],
            "test": selection_split_summary["test"],
        }
        training_frame = pretest
    else:
        test_candidates = selection_candidates
        split_summary = selection_split_summary
        training_frame = splits.training

    selected = test_candidates[selected_name]
    test_probabilities = {
        name: candidate.predict_probability(splits.test)
        for name, candidate in test_candidates.items()
    }
    elo_test_probabilities = test_probabilities["elo_only"]
    candidate_metrics = {
        name: {
            "validation": validation_metrics[name],
            "test": _evaluate_with_cluster_interval(
                candidate,
                splits.test,
                settings,
                include_breakdowns=True,
            ),
        }
        for name, candidate in test_candidates.items()
    }
    for name, candidate_report in candidate_metrics.items():
        candidate_report["test"]["paired_log_loss_difference_vs_elo_interval"] = (
            _paired_log_loss_interval(
                splits.test,
                test_probabilities[name],
                elo_test_probabilities,
                settings,
            )
        )

    train_timestamps = pd.to_datetime(training_frame["match_timestamp"], utc=True)
    data_cutoff = train_timestamps.max().to_pydatetime()
    matches = load_matches(database_path)
    feature_state = build_state_until(matches, settings.features, data_cutoff)
    fingerprint = dataset_fingerprint(database_path)
    metrics = {
        "report_schema_version": TRAINING_REPORT_SCHEMA_VERSION,
        "experiment_name": settings.experiment_name,
        "selection": {
            "candidate": selected_name,
            "metric": "validation_log_loss",
            "value": selected_validation_log_loss,
        },
        "data_cutoff_timestamp": data_cutoff.isoformat(),
        "dataset_fingerprint": fingerprint,
        "feature_schema_version": settings.features.feature_schema_version,
        "modeling_population": population_summary,
        "release_refit_before_test": settings.splits.refit_before_test,
        "selection_split_summary": selection_split_summary,
        "split_summary": split_summary,
        "candidates": candidate_metrics,
    }
    artifact_directory = save_artifact(
        registry_directory,
        candidate=selected,
        feature_state=feature_state,
        metrics=metrics,
        settings=settings,
        split_summary=split_summary,
        dataset_fingerprint=fingerprint,
        selected_validation_log_loss=selected_validation_log_loss,
        artifact_purpose="evaluation",
    )
    report = {
        **metrics,
        "artifact_directory": str(artifact_directory),
    }
    _write_json(reports_directory / "training-report.json", report)
    return report


def backtest_baselines(
    database_path: Path,
    settings: ExperimentSettings,
    *,
    reports_directory: Path,
) -> dict[str, Any]:
    """Run repeated expanding-window evaluation without opening the final holdout."""
    metadata = get_metadata(database_path)
    if metadata.get("feature_schema_version") != settings.features.feature_schema_version:
        raise ValueError(
            "Feature table is missing or was generated with a different schema version"
        )
    source_frame = load_feature_table(database_path)
    validate_feature_frame(source_frame)
    frame, population_summary = filter_modeling_population(source_frame, settings)
    release_split = chronological_split(frame, settings.splits)
    if settings.splits.uses_timestamp_boundaries:
        development = release_split.training.sort_values(["match_timestamp", "match_id"])
    else:
        development = pd.concat(
            [
                release_split.fit,
                release_split.calibration,
                release_split.validation,
            ],
            ignore_index=True,
        ).sort_values(["match_timestamp", "match_id"])
    folds = rolling_origin_splits(
        development,
        settings.splits,
        settings.backtest,
    )

    candidate_frames: dict[str, list[pd.DataFrame]] = {}
    candidate_probabilities: dict[str, list[np.ndarray]] = {}
    selected_frames: list[pd.DataFrame] = []
    selected_probabilities: list[np.ndarray] = []
    selected_names: list[str] = []
    fold_reports: list[dict[str, Any]] = []

    for fold in folds:
        splits = fold.splits
        candidates = fit_all_candidates(splits.fit, splits.calibration, settings)
        validation_metrics = {
            name: evaluate_candidate(
                candidate,
                splits.validation,
                include_breakdowns=False,
            )
            for name, candidate in candidates.items()
        }
        selected_name = min(
            validation_metrics,
            key=lambda name: validation_metrics[name]["log_loss"],
        )
        selected_names.append(selected_name)

        test_metrics: dict[str, dict[str, Any]] = {}
        for name, candidate in candidates.items():
            probabilities = candidate.predict_probability(splits.test)
            candidate_frames.setdefault(name, []).append(splits.test.copy())
            candidate_probabilities.setdefault(name, []).append(probabilities)
            test_metrics[name] = evaluate_probabilities(
                splits.test,
                probabilities,
                include_breakdowns=False,
            )

        selected_frames.append(splits.test.copy())
        selected_probabilities.append(candidates[selected_name].predict_probability(splits.test))
        training_timestamps = pd.to_datetime(splits.training["match_timestamp"], utc=True)
        fold_reports.append(
            {
                "fold_number": fold.fold_number,
                "data_cutoff_timestamp": training_timestamps.max().isoformat(),
                "split_summary": splits.summary(),
                "selection": {
                    "candidate": selected_name,
                    "metric": "validation_log_loss",
                    "value": validation_metrics[selected_name]["log_loss"],
                },
                "candidates": {
                    name: {
                        "validation": validation_metrics[name],
                        "test": test_metrics[name],
                    }
                    for name in candidates
                },
            }
        )

    aggregate_candidates: dict[str, dict[str, Any]] = {}
    aggregate_elo_probabilities = np.concatenate(candidate_probabilities["elo_only"])
    for candidate_name in sorted(candidate_frames):
        aggregate_frame = pd.concat(
            candidate_frames[candidate_name],
            ignore_index=True,
        )
        probabilities = np.concatenate(candidate_probabilities[candidate_name])
        metrics = evaluate_probabilities(
            aggregate_frame,
            probabilities,
            include_breakdowns=True,
        )
        clusters = (
            aggregate_frame["series_id"]
            if "series_id" in aggregate_frame
            else aggregate_frame["match_id"]
        )
        metrics["series_clustered_log_loss_interval"] = clustered_log_loss_interval(
            aggregate_frame["blue_win"].to_numpy(dtype=int),
            probabilities,
            clusters.fillna(aggregate_frame["match_id"]).to_numpy(dtype=str),
            iterations=settings.backtest.bootstrap_iterations,
            random_seed=settings.random_seed,
        )
        metrics["paired_log_loss_difference_vs_elo_interval"] = _paired_log_loss_interval(
            aggregate_frame,
            probabilities,
            aggregate_elo_probabilities,
            settings,
        )
        aggregate_candidates[candidate_name] = metrics

    configured_simple_controls = tuple(
        name for name in SIMPLE_CONTROL_NAMES if name in aggregate_candidates
    )
    best_simple_control = min(
        configured_simple_controls,
        key=lambda name: aggregate_candidates[name]["log_loss"],
    )
    simple_control_probabilities = np.concatenate(candidate_probabilities[best_simple_control])
    for candidate_name, metrics in aggregate_candidates.items():
        aggregate_frame = pd.concat(
            candidate_frames[candidate_name],
            ignore_index=True,
        )
        probabilities = np.concatenate(candidate_probabilities[candidate_name])
        metrics["paired_log_loss_difference_vs_best_simple_control_interval"] = (
            _paired_log_loss_interval(
                aggregate_frame,
                probabilities,
                simple_control_probabilities,
                settings,
            )
        )

    raw_metric_winner = min(
        aggregate_candidates,
        key=lambda name: aggregate_candidates[name]["log_loss"],
    )
    development_eligibility = {
        candidate_name: _development_candidate_eligibility(
            candidate_name,
            aggregate_candidates=aggregate_candidates,
            fold_reports=fold_reports,
            best_simple_control=best_simple_control,
            settings=settings,
        )
        for candidate_name in aggregate_candidates
    }
    eligible_candidates = [
        name for name, eligibility in development_eligibility.items() if eligibility["eligible"]
    ]
    development_selection = (
        min(
            eligible_candidates,
            key=lambda name: aggregate_candidates[name]["log_loss"],
        )
        if eligible_candidates
        else "elo_only"
    )

    selection_frame = pd.concat(selected_frames, ignore_index=True)
    selection_probability = np.concatenate(selected_probabilities)
    selection_metrics = evaluate_probabilities(
        selection_frame,
        selection_probability,
        include_breakdowns=True,
    )
    selection_clusters = (
        selection_frame["series_id"]
        if "series_id" in selection_frame
        else selection_frame["match_id"]
    )
    selection_metrics["series_clustered_log_loss_interval"] = clustered_log_loss_interval(
        selection_frame["blue_win"].to_numpy(dtype=int),
        selection_probability,
        selection_clusters.fillna(selection_frame["match_id"]).to_numpy(dtype=str),
        iterations=settings.backtest.bootstrap_iterations,
        random_seed=settings.random_seed,
    )

    report = {
        "report_schema_version": TRAINING_REPORT_SCHEMA_VERSION,
        "experiment_name": settings.experiment_name,
        "dataset_fingerprint": dataset_fingerprint(database_path),
        "feature_schema_version": settings.features.feature_schema_version,
        "modeling_population": population_summary,
        "fold_count": len(folds),
        "folds": fold_reports,
        "aggregate_candidates": aggregate_candidates,
        "development_eligibility": development_eligibility,
        "aggregate_selection": {
            "rule": "lowest_rolling_test_log_loss_among_development_eligible_candidates",
            "candidate": development_selection,
            "value": aggregate_candidates[development_selection]["log_loss"],
            "raw_metric_winner": raw_metric_winner,
            "raw_metric_winner_value": aggregate_candidates[raw_metric_winner]["log_loss"],
            "best_simple_control": best_simple_control,
            "best_simple_control_value": aggregate_candidates[best_simple_control]["log_loss"],
            "paired_log_loss_difference_vs_best_simple_control_interval": (
                aggregate_candidates[development_selection][
                    "paired_log_loss_difference_vs_best_simple_control_interval"
                ]
            ),
        },
        "selection_policy": {
            "rule": "lowest_validation_log_loss_per_fold",
            "selected_candidate_counts": {
                name: selected_names.count(name) for name in sorted(set(selected_names))
            },
            "test": selection_metrics,
        },
        "reserved_model_selection_validation": (
            release_split.summary()["validation"]
            if settings.splits.uses_timestamp_boundaries
            else None
        ),
        "reserved_final_holdout": release_split.summary()["test"],
        "reserved_final_holdout_evaluated": False,
    }
    _write_json(reports_directory / "backtest-report.json", report)
    return report


def confirm_backtest_selection(
    database_path: Path,
    settings: ExperimentSettings,
    *,
    backtest_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Evaluate one locked development selection without touching the final holdout."""
    backtest_report = json.loads(backtest_report_path.read_text(encoding="utf-8"))
    current_fingerprint = dataset_fingerprint(database_path)
    if backtest_report.get("dataset_fingerprint") != current_fingerprint:
        raise ValueError("Backtest report does not match the confirmation database")
    if backtest_report.get("feature_schema_version") != (settings.features.feature_schema_version):
        raise ValueError("Backtest report feature schema does not match the config")
    if backtest_report.get("experiment_name") != settings.experiment_name:
        raise ValueError("Backtest report experiment does not match the config")
    if not settings.splits.uses_timestamp_boundaries:
        raise ValueError("Confirmation requires a timestamp-boundary validation interval")
    selected_name = str(backtest_report["aggregate_selection"]["candidate"])
    if selected_name not in CANDIDATE_NAMES:
        raise ValueError(f"Backtest report selected an unknown candidate: {selected_name}")

    source_frame = load_feature_table(database_path)
    validate_feature_frame(source_frame)
    frame, population_summary = filter_modeling_population(source_frame, settings)
    release_split = chronological_split(frame, settings.splits)
    split_summary = release_split.summary()
    if split_summary["validation"] != backtest_report.get("reserved_model_selection_validation"):
        raise ValueError("Backtest report validation interval does not match the config")
    if split_summary["test"] != backtest_report.get("reserved_final_holdout"):
        raise ValueError("Backtest report final holdout does not match the config")

    candidate_fit, candidate_calibration = _refit_partitions(
        release_split.training,
        selected_name,
        settings,
    )
    candidate = fit_candidate(
        selected_name,
        candidate_fit,
        candidate_calibration,
        settings,
    )
    elo_fit, elo_calibration = _refit_partitions(
        release_split.training,
        "elo_only",
        settings,
    )
    elo = fit_candidate(
        "elo_only",
        elo_fit,
        elo_calibration,
        settings,
    )
    validation = release_split.validation
    candidate_metrics = _evaluate_with_cluster_interval(
        candidate,
        validation,
        settings,
        include_breakdowns=True,
    )
    elo_metrics = _evaluate_with_cluster_interval(
        elo,
        validation,
        settings,
        include_breakdowns=True,
    )
    candidate_probabilities = candidate.predict_probability(validation)
    elo_probabilities = elo.predict_probability(validation)
    candidate_metrics["paired_log_loss_difference_vs_elo_interval"] = _paired_log_loss_interval(
        validation,
        candidate_probabilities,
        elo_probabilities,
        settings,
    )
    training_timestamps = pd.to_datetime(
        release_split.training["match_timestamp"],
        utc=True,
    )
    report = {
        "report_schema_version": TRAINING_REPORT_SCHEMA_VERSION,
        "experiment_name": settings.experiment_name,
        "dataset_fingerprint": current_fingerprint,
        "feature_schema_version": settings.features.feature_schema_version,
        "modeling_population": population_summary,
        "selection": {
            "candidate": selected_name,
            "source": str(backtest_report_path.resolve()),
            "locked_before_confirmation": True,
        },
        "training_data_cutoff_timestamp": training_timestamps.max().isoformat(),
        "candidate_refit_split": {
            "fit": _production_frame_summary(candidate_fit),
            "calibration": _production_frame_summary(candidate_calibration),
            "train": _production_frame_summary(release_split.training),
        },
        "elo_refit_split": {
            "fit": _production_frame_summary(elo_fit),
            "calibration": _production_frame_summary(elo_calibration),
            "train": _production_frame_summary(release_split.training),
        },
        "confirmation_status": "opened_diagnostic_not_promotion_evidence",
        "validation_split": split_summary["validation"],
        "candidate_validation": candidate_metrics,
        "elo_validation": elo_metrics,
        "reserved_final_holdout": split_summary["test"],
        "reserved_final_holdout_evaluated": False,
    }
    _write_json(output_path, report)
    return report


def refit_development_artifact(
    database_path: Path,
    settings: ExperimentSettings,
    *,
    backtest_report_path: Path,
    confirmation_report_path: Path,
    registry_directory: Path,
    reports_directory: Path,
) -> dict[str, Any]:
    """Build a clearly labeled shadow artifact from the locked candidate."""
    backtest_report = json.loads(backtest_report_path.read_text(encoding="utf-8"))
    confirmation_report = json.loads(confirmation_report_path.read_text(encoding="utf-8"))
    current_fingerprint = dataset_fingerprint(database_path)
    for report_name, report in (
        ("Backtest", backtest_report),
        ("Confirmation", confirmation_report),
    ):
        if report.get("dataset_fingerprint") != current_fingerprint:
            raise ValueError(f"{report_name} report does not match the development database")
        if report.get("experiment_name") != settings.experiment_name:
            raise ValueError(f"{report_name} report experiment does not match the config")
        if report.get("feature_schema_version") != (settings.features.feature_schema_version):
            raise ValueError(f"{report_name} report feature schema does not match the config")

    selected_name = str(backtest_report["aggregate_selection"]["candidate"])
    if selected_name != confirmation_report.get("selection", {}).get("candidate"):
        raise ValueError("Confirmation report does not evaluate the backtest selection")
    if confirmation_report.get("selection", {}).get("locked_before_confirmation") is not True:
        raise ValueError("Confirmation report does not record a locked selection")
    if selected_name not in CANDIDATE_NAMES:
        raise ValueError(f"Backtest report selected an unknown candidate: {selected_name}")

    source_frame = load_feature_table(database_path)
    validate_feature_frame(source_frame)
    frame, population_summary = filter_modeling_population(source_frame, settings)
    ordered = frame.sort_values(["match_timestamp", "match_id"]).reset_index(drop=True)
    fit, calibration = _refit_partitions(
        ordered,
        selected_name,
        settings,
    )
    candidate = fit_candidate(
        selected_name,
        fit,
        calibration,
        settings,
    )

    source_timestamps = pd.to_datetime(source_frame["match_timestamp"], utc=True)
    cutoff = source_timestamps.max().to_pydatetime()
    feature_state = build_state_until(
        load_matches(database_path),
        settings.features,
        cutoff,
    )
    split_summary = {
        "fit": _production_frame_summary(fit),
        "calibration": _production_frame_summary(calibration),
        "train": _production_frame_summary(ordered),
        "validation": None,
        "test": None,
    }
    selected_log_loss = float(backtest_report["aggregate_candidates"][selected_name]["log_loss"])
    metrics = {
        "report_schema_version": TRAINING_REPORT_SCHEMA_VERSION,
        "artifact_purpose": "development",
        "experiment_name": settings.experiment_name,
        "model_kind": selected_name,
        "selection_source": {
            "metric": "rolling_test_log_loss",
            "value": selected_log_loss,
            "backtest_report": str(backtest_report_path.resolve()),
            "confirmation_report": str(confirmation_report_path.resolve()),
            "confirmation_status": confirmation_report["confirmation_status"],
        },
        "data_cutoff_timestamp": cutoff.isoformat(),
        "dataset_fingerprint": current_fingerprint,
        "feature_schema_version": settings.features.feature_schema_version,
        "modeling_population": population_summary,
        "split_summary": split_summary,
        "prospective_promotion_passed": False,
    }
    artifact_directory = save_artifact(
        registry_directory,
        candidate=candidate,
        feature_state=feature_state,
        metrics=metrics,
        settings=settings,
        split_summary=split_summary,
        dataset_fingerprint=current_fingerprint,
        selected_validation_log_loss=selected_log_loss,
        artifact_purpose="development",
        selection_metric="rolling_test_log_loss",
    )
    report = {
        **metrics,
        "artifact_directory": str(artifact_directory),
    }
    _write_json(reports_directory / "development-refit-report.json", report)
    return report


def _production_frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "sample_count": 0,
            "timestamp_count": 0,
            "start_timestamp": None,
            "end_timestamp": None,
        }
    timestamps = pd.to_datetime(frame["match_timestamp"], utc=True)
    return {
        "sample_count": len(frame),
        "timestamp_count": int(timestamps.nunique()),
        "start_timestamp": timestamps.min().isoformat(),
        "end_timestamp": timestamps.max().isoformat(),
    }


def refit_selected_artifact(
    database_path: Path,
    settings: ExperimentSettings,
    *,
    training_report_path: Path,
    registry_directory: Path,
    reports_directory: Path,
    promotion_report_path: Path | None = None,
) -> dict[str, Any]:
    """Refit an already evaluated candidate and snapshot state through all data."""
    evaluation_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    current_fingerprint = dataset_fingerprint(database_path)
    if evaluation_report.get("dataset_fingerprint") != current_fingerprint:
        raise ValueError("Training report does not match the production database")
    if evaluation_report.get("feature_schema_version") != settings.features.feature_schema_version:
        raise ValueError("Training report feature schema does not match the config")
    if evaluation_report.get("experiment_name") != settings.experiment_name:
        raise ValueError("Training report experiment does not match the config")

    selected_name = str(evaluation_report["selection"]["candidate"])
    promotion_source: dict[str, object] | None = None
    if promotion_report_path is not None:
        promotion_report = json.loads(promotion_report_path.read_text(encoding="utf-8"))
        if not isinstance(promotion_report, dict):
            raise ValueError("Promotion report must contain a JSON object")
        if promotion_report.get("dataset_fingerprint") != current_fingerprint:
            raise ValueError("Promotion report does not match the production database")
        if promotion_report.get("experiment_name") != settings.experiment_name:
            raise ValueError("Promotion report experiment does not match the config")
        recommended_candidate = promotion_report.get("recommended_candidate")
        if not isinstance(recommended_candidate, str):
            raise ValueError("Promotion report has no recommended candidate")
        selected_name = recommended_candidate
        promotion_source = {
            "report": str(promotion_report_path.resolve()),
            "promotion_passed": bool(promotion_report.get("promotion_passed")),
            "recommended_candidate": recommended_candidate,
        }
    if selected_name not in CANDIDATE_NAMES:
        raise ValueError(f"Training report selected an unknown candidate: {selected_name}")
    try:
        selected_validation_log_loss = float(
            evaluation_report["candidates"][selected_name]["validation"]["log_loss"]
        )
    except KeyError as error:
        raise ValueError(
            f"Training report has no validation metrics for candidate {selected_name}"
        ) from error

    source_frame = load_feature_table(database_path)
    validate_feature_frame(source_frame)
    frame, population_summary = filter_modeling_population(source_frame, settings)
    ordered = frame.sort_values(["match_timestamp", "match_id"]).reset_index(drop=True)
    fit, calibration = _refit_partitions(
        ordered,
        selected_name,
        settings,
    )
    candidate = fit_candidate(
        selected_name,
        fit,
        calibration,
        settings,
    )

    source_timestamps = pd.to_datetime(source_frame["match_timestamp"], utc=True)
    cutoff = source_timestamps.max().to_pydatetime()
    matches = load_matches(database_path)
    feature_state = build_state_until(matches, settings.features, cutoff)
    split_summary = {
        "fit": _production_frame_summary(fit),
        "calibration": _production_frame_summary(calibration),
        "train": _production_frame_summary(ordered),
        "validation": None,
        "test": None,
    }
    metrics = {
        "report_schema_version": TRAINING_REPORT_SCHEMA_VERSION,
        "artifact_purpose": "production",
        "experiment_name": settings.experiment_name,
        "model_kind": selected_name,
        "selection_source": {
            "metric": "validation_log_loss",
            "value": selected_validation_log_loss,
            "evaluation_artifact_directory": evaluation_report["artifact_directory"],
            "promotion": promotion_source,
        },
        "data_cutoff_timestamp": cutoff.isoformat(),
        "dataset_fingerprint": current_fingerprint,
        "feature_schema_version": settings.features.feature_schema_version,
        "modeling_population": population_summary,
        "split_summary": split_summary,
    }
    artifact_directory = save_artifact(
        registry_directory,
        candidate=candidate,
        feature_state=feature_state,
        metrics=metrics,
        settings=settings,
        split_summary=split_summary,
        dataset_fingerprint=current_fingerprint,
        selected_validation_log_loss=selected_validation_log_loss,
        artifact_purpose="production",
    )
    report = {
        **metrics,
        "artifact_directory": str(artifact_directory),
    }
    _write_json(reports_directory / "production-refit-report.json", report)
    return report


def evaluate_saved_artifact(
    database_path: Path,
    artifact_directory: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    artifact = load_artifact(artifact_directory)
    if artifact.manifest["artifact_purpose"] != "evaluation":
        raise ValueError("Non-evaluation artifacts do not have an untouched evaluation holdout")
    current_fingerprint = dataset_fingerprint(database_path)
    if current_fingerprint != artifact.manifest["dataset_fingerprint"]:
        raise ValueError("Evaluation database does not match the artifact dataset")

    source_frame = load_feature_table(database_path)
    validate_feature_frame(source_frame)
    artifact_settings = ExperimentSettings.model_validate(artifact.training_config)
    frame, _ = filter_modeling_population(source_frame, artifact_settings)
    test_summary = artifact.manifest["split_summary"]["test"]
    test_start = pd.Timestamp(test_summary["start_timestamp"])
    test_end = pd.Timestamp(test_summary["end_timestamp"])
    timestamps = pd.to_datetime(frame["match_timestamp"], utc=True)
    test = frame[(timestamps >= test_start) & (timestamps <= test_end)].copy()
    if len(test) != int(test_summary["sample_count"]):
        raise ValueError("Evaluation holdout does not match the artifact split manifest")

    report = {
        "report_schema_version": TRAINING_REPORT_SCHEMA_VERSION,
        "model_version": artifact.manifest["model_version"],
        "model_kind": artifact.manifest["model_kind"],
        "data_cutoff_timestamp": artifact.manifest["data_cutoff_timestamp"],
        "dataset_fingerprint": current_fingerprint,
        "test": evaluate_candidate(
            artifact.model,
            test,
            include_breakdowns=True,
        ),
    }
    clusters = test["series_id"] if "series_id" in test else test["match_id"]
    probabilities = artifact.model.predict_probability(test)
    report["test"]["series_clustered_log_loss_interval"] = clustered_log_loss_interval(
        test["blue_win"].to_numpy(dtype=int),
        probabilities,
        clusters.fillna(test["match_id"]).to_numpy(dtype=str),
        iterations=artifact.training_config["backtest"]["bootstrap_iterations"],
        random_seed=artifact.training_config["random_seed"],
    )
    if output_path is not None:
        _write_json(output_path, report)
    return report


def parse_iso_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)
