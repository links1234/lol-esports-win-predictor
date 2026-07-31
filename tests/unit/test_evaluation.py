import numpy as np

from lolpredictor.evaluation import (
    clustered_paired_log_loss_difference_interval,
)


def test_paired_cluster_interval_preserves_series_and_difference_direction() -> None:
    labels = np.array([1, 0, 1, 0, 1, 0])
    candidate = np.array([0.8, 0.2, 0.85, 0.15, 0.75, 0.25])
    reference = np.array([0.6, 0.4, 0.6, 0.4, 0.6, 0.4])
    clusters = np.array(["series-1", "series-1", "series-2", "series-2", "series-3", "series-3"])

    interval = clustered_paired_log_loss_difference_interval(
        labels,
        candidate,
        reference,
        clusters,
        iterations=500,
        random_seed=20260730,
    )

    assert interval is not None
    assert interval["point_estimate"] < 0
    assert interval["upper"] < 0
    assert interval["cluster_count"] == 3


def test_paired_cluster_interval_requires_multiple_series() -> None:
    interval = clustered_paired_log_loss_difference_interval(
        np.array([1, 0]),
        np.array([0.7, 0.3]),
        np.array([0.6, 0.4]),
        np.array(["series-1", "series-1"]),
        iterations=100,
        random_seed=7,
    )

    assert interval is None
