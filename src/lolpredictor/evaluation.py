"""Probabilistic metrics and chronological slice reports."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from lolpredictor.models import CalibratedCandidate


def probability_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    calibration_bin_count: int = 10,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) != len(probabilities) or len(labels) == 0:
        raise ValueError("Metrics require equal, non-empty label and probability arrays")
    if not np.isfinite(probabilities).all():
        raise ValueError("Metrics require finite probabilities")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Metrics require probabilities between zero and one")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Metrics require binary labels")
    probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)

    bin_indexes = np.minimum(
        (probabilities * calibration_bin_count).astype(int),
        calibration_bin_count - 1,
    )
    calibration_bins: list[dict[str, float | int]] = []
    expected_calibration_error = 0.0
    for bin_index in range(calibration_bin_count):
        selected = bin_indexes == bin_index
        count = int(selected.sum())
        if not count:
            continue
        mean_probability = float(probabilities[selected].mean())
        observed_rate = float(labels[selected].mean())
        expected_calibration_error += count / len(labels) * abs(mean_probability - observed_rate)
        calibration_bins.append(
            {
                "lower": bin_index / calibration_bin_count,
                "upper": (bin_index + 1) / calibration_bin_count,
                "sample_count": count,
                "mean_probability": mean_probability,
                "observed_blue_win_rate": observed_rate,
            }
        )

    roc_auc = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None
    return {
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "expected_calibration_error": float(expected_calibration_error),
        "roc_auc": roc_auc,
        "accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
        "sample_count": len(labels),
        "blue_win_rate": float(labels.mean()),
        "mean_blue_probability": float(probabilities.mean()),
        "probability_contract_valid": True,
        "calibration_bins": calibration_bins,
    }


def clustered_log_loss_interval(
    labels: np.ndarray,
    probabilities: np.ndarray,
    clusters: np.ndarray,
    *,
    iterations: int,
    random_seed: int,
) -> dict[str, float | int] | None:
    """Bootstrap log loss while resampling complete series clusters."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    clusters = np.asarray(clusters, dtype=str)
    if not (len(labels) == len(probabilities) == len(clusters)):
        raise ValueError("Clustered intervals require equal-length arrays")
    unique_clusters = np.unique(clusters)
    if len(unique_clusters) < 2:
        return None

    cluster_indexes = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique_clusters}
    random_generator = np.random.default_rng(random_seed)
    losses = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled_clusters = random_generator.choice(
            unique_clusters,
            size=len(unique_clusters),
            replace=True,
        )
        sampled_indexes = np.concatenate([cluster_indexes[cluster] for cluster in sampled_clusters])
        losses[iteration] = log_loss(
            labels[sampled_indexes],
            probabilities[sampled_indexes],
            labels=[0, 1],
        )
    lower, upper = np.quantile(losses, [0.025, 0.975])
    return {
        "confidence_level": 0.95,
        "lower": float(lower),
        "upper": float(upper),
        "iterations": iterations,
        "cluster_count": len(unique_clusters),
    }


def clustered_paired_log_loss_difference_interval(
    labels: np.ndarray,
    candidate_probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
    clusters: np.ndarray,
    *,
    iterations: int,
    random_seed: int,
) -> dict[str, float | int] | None:
    """Bootstrap candidate-minus-reference log loss by complete series."""
    labels = np.asarray(labels, dtype=int)
    candidate_probabilities = np.clip(
        np.asarray(candidate_probabilities, dtype=float),
        1e-6,
        1.0 - 1e-6,
    )
    reference_probabilities = np.clip(
        np.asarray(reference_probabilities, dtype=float),
        1e-6,
        1.0 - 1e-6,
    )
    clusters = np.asarray(clusters, dtype=str)
    lengths = {
        len(labels),
        len(candidate_probabilities),
        len(reference_probabilities),
        len(clusters),
    }
    if len(lengths) != 1 or not labels.size:
        raise ValueError("Paired intervals require equal, non-empty arrays")
    unique_clusters = np.unique(clusters)
    if len(unique_clusters) < 2:
        return None

    candidate_losses = -(
        labels * np.log(candidate_probabilities)
        + (1 - labels) * np.log(1.0 - candidate_probabilities)
    )
    reference_losses = -(
        labels * np.log(reference_probabilities)
        + (1 - labels) * np.log(1.0 - reference_probabilities)
    )
    loss_differences = candidate_losses - reference_losses
    cluster_indexes = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique_clusters}
    random_generator = np.random.default_rng(random_seed)
    bootstrap_differences = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled_clusters = random_generator.choice(
            unique_clusters,
            size=len(unique_clusters),
            replace=True,
        )
        sampled_indexes = np.concatenate([cluster_indexes[cluster] for cluster in sampled_clusters])
        bootstrap_differences[iteration] = float(loss_differences[sampled_indexes].mean())
    lower, upper = np.quantile(bootstrap_differences, [0.025, 0.975])
    return {
        "confidence_level": 0.95,
        "point_estimate": float(loss_differences.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "iterations": iterations,
        "cluster_count": len(unique_clusters),
    }


def _group_metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    group_values: pd.Series,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for group_name in sorted(group_values.astype(str).unique()):
        selected = group_values.astype(str) == group_name
        results[group_name] = probability_metrics(
            frame.loc[selected, "blue_win"].to_numpy(dtype=int),
            probabilities[selected.to_numpy()],
        )
    return results


def evaluate_candidate(
    candidate: CalibratedCandidate,
    frame: pd.DataFrame,
    *,
    include_breakdowns: bool,
) -> dict[str, Any]:
    probabilities = candidate.predict_probability(frame)
    result = evaluate_probabilities(
        frame,
        probabilities,
        include_breakdowns=include_breakdowns,
    )
    result["calibration_applied"] = candidate.probability_calibration_applied
    return result


def evaluate_probabilities(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    include_breakdowns: bool,
) -> dict[str, Any]:
    result = probability_metrics(
        frame["blue_win"].to_numpy(dtype=int),
        probabilities,
    )
    if include_breakdowns:
        periods = pd.to_datetime(frame["match_timestamp"], utc=True).dt.strftime("%Y-%m")
        breakdowns = {
            "time_period": _group_metrics(frame, probabilities, periods),
            "patch": _group_metrics(frame, probabilities, frame["patch"]),
            "league": _group_metrics(frame, probabilities, frame["league"]),
        }
        for column in ("region", "tournament_level", "official_status"):
            if column in frame:
                breakdowns[column] = _group_metrics(
                    frame,
                    probabilities,
                    frame[column],
                )
        result["breakdowns"] = breakdowns
    return result
