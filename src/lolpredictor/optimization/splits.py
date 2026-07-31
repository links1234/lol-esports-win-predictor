"""Nested expanding-window partitions for model optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from lolpredictor.optimization.settings import NestedValidationSettings
from lolpredictor.settings import BacktestSettings, SplitSettings
from lolpredictor.splits import rolling_origin_splits


@dataclass(frozen=True)
class OptimizationInnerFold:
    fold_number: int
    history: pd.DataFrame
    score: pd.DataFrame

    def summary(self) -> dict[str, Any]:
        return {
            "fold_number": self.fold_number,
            "history": frame_summary(self.history),
            "score": frame_summary(self.score),
        }


@dataclass(frozen=True)
class OptimizationOuterFold:
    fold_number: int
    history: pd.DataFrame
    score: pd.DataFrame
    inner_folds: tuple[OptimizationInnerFold, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "fold_number": self.fold_number,
            "history": frame_summary(self.history),
            "score": frame_summary(self.score),
            "inner_folds": [fold.summary() for fold in self.inner_folds],
        }


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["match_timestamp", "match_id"]).reset_index(drop=True)


def frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    timestamps = pd.to_datetime(frame["match_timestamp"], utc=True)
    return {
        "sample_count": len(frame),
        "timestamp_count": int(timestamps.nunique()),
        "start_timestamp": timestamps.min().isoformat(),
        "end_timestamp": timestamps.max().isoformat(),
    }


def _rolling_settings(
    nested: NestedValidationSettings,
    *,
    fold_count: int,
) -> tuple[SplitSettings, BacktestSettings]:
    split_settings = SplitSettings(
        train_fraction=0.70,
        validation_fraction=0.10,
        calibration_fraction_within_train=nested.calibration_fraction_within_train,
    )
    backtest_settings = BacktestSettings(
        fold_count=fold_count,
        validation_fraction_per_fold=nested.validation_fraction_per_fold,
        test_fraction_per_fold=nested.test_fraction_per_fold,
        bootstrap_iterations=100,
        minimum_breakdown_sample_count=1,
    )
    return split_settings, backtest_settings


def build_nested_folds(
    frame: pd.DataFrame,
    settings: NestedValidationSettings,
) -> tuple[OptimizationOuterFold, ...]:
    """Build outer and inner folds with all timestamp groups kept indivisible."""
    ordered = _ordered(frame)
    split_settings, outer_settings = _rolling_settings(
        settings,
        fold_count=settings.outer_fold_count,
    )
    source_outer_folds = rolling_origin_splits(
        ordered,
        split_settings,
        outer_settings,
    )
    result: list[OptimizationOuterFold] = []
    for source_outer in source_outer_folds:
        outer_history = _ordered(
            pd.concat(
                [
                    source_outer.splits.training,
                    source_outer.splits.validation,
                ],
                ignore_index=True,
            )
        )
        _, inner_settings = _rolling_settings(
            settings,
            fold_count=settings.inner_fold_count,
        )
        source_inner_folds = rolling_origin_splits(
            outer_history,
            split_settings,
            inner_settings,
        )
        inner_folds = tuple(
            OptimizationInnerFold(
                fold_number=source_inner.fold_number,
                history=_ordered(
                    pd.concat(
                        [
                            source_inner.splits.training,
                            source_inner.splits.validation,
                        ],
                        ignore_index=True,
                    )
                ),
                score=_ordered(source_inner.splits.test),
            )
            for source_inner in source_inner_folds
        )
        outer = OptimizationOuterFold(
            fold_number=source_outer.fold_number,
            history=outer_history,
            score=_ordered(source_outer.splits.test),
            inner_folds=inner_folds,
        )
        _validate_outer_fold(outer)
        result.append(outer)
    return tuple(result)


def trailing_fit_calibration_split(
    history: pd.DataFrame,
    *,
    requires_calibration: bool,
    calibration_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve a trailing mapper interval without exposing the scoring interval."""
    ordered = _ordered(history)
    if not requires_calibration:
        return ordered, ordered.iloc[0:0].copy()
    timestamps = pd.to_datetime(ordered["match_timestamp"], utc=True)
    unique_timestamps = sorted(timestamps.unique().tolist())
    calibration_count = max(2, math.ceil(len(unique_timestamps) * calibration_fraction))
    if len(unique_timestamps) - calibration_count < 4:
        raise ValueError("Optimization fit interval needs at least four timestamp groups")
    calibration_timestamps = unique_timestamps[-calibration_count:]
    fit = ordered[~timestamps.isin(calibration_timestamps)].reset_index(drop=True)
    calibration = ordered[timestamps.isin(calibration_timestamps)].reset_index(drop=True)
    if (
        pd.to_datetime(fit["match_timestamp"], utc=True).max()
        >= pd.to_datetime(calibration["match_timestamp"], utc=True).min()
    ):
        raise ValueError("Optimization fit and calibration intervals overlap")
    return fit, calibration


def _validate_outer_fold(fold: OptimizationOuterFold) -> None:
    outer_history_end = pd.to_datetime(fold.history["match_timestamp"], utc=True).max()
    outer_score_start = pd.to_datetime(fold.score["match_timestamp"], utc=True).min()
    if outer_history_end >= outer_score_start:
        raise ValueError("Outer optimization history overlaps its score interval")
    for inner in fold.inner_folds:
        history_end = pd.to_datetime(inner.history["match_timestamp"], utc=True).max()
        score_start = pd.to_datetime(inner.score["match_timestamp"], utc=True).min()
        if history_end >= score_start:
            raise ValueError("Inner optimization history overlaps its score interval")
        if score_start >= outer_score_start:
            raise ValueError("Inner scoring must occur before the outer score interval")
