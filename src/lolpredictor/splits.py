"""Chronological dataset splitting with indivisible timestamp groups."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import pandas as pd

from lolpredictor.settings import BacktestSettings, SplitSettings


@dataclass(frozen=True)
class ChronologicalSplits:
    fit: pd.DataFrame
    calibration: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def training(self) -> pd.DataFrame:
        return pd.concat([self.fit, self.calibration], ignore_index=True)

    def summary(self) -> dict[str, Any]:
        return {
            "fit": _frame_summary(self.fit),
            "calibration": _frame_summary(self.calibration),
            "train": _frame_summary(self.training),
            "validation": _frame_summary(self.validation),
            "test": _frame_summary(self.test),
        }


@dataclass(frozen=True)
class RollingOriginFold:
    fold_number: int
    splits: ChronologicalSplits

    def summary(self) -> dict[str, Any]:
        return {
            "fold_number": self.fold_number,
            **self.splits.summary(),
        }


def _frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    timestamps = pd.to_datetime(frame["match_timestamp"], utc=True)
    return {
        "sample_count": len(frame),
        "timestamp_count": int(timestamps.nunique()),
        "start_timestamp": timestamps.min().isoformat(),
        "end_timestamp": timestamps.max().isoformat(),
    }


def _take_timestamp_range(
    frame: pd.DataFrame,
    timestamps: list[pd.Timestamp],
) -> pd.DataFrame:
    selected = frame[pd.to_datetime(frame["match_timestamp"], utc=True).isin(timestamps)]
    return selected.sort_values(["match_timestamp", "match_id"]).reset_index(drop=True)


def _validate_boundaries(result: ChronologicalSplits) -> None:
    boundaries = [
        pd.to_datetime(result.fit["match_timestamp"], utc=True).max(),
        pd.to_datetime(result.calibration["match_timestamp"], utc=True).min(),
        pd.to_datetime(result.calibration["match_timestamp"], utc=True).max(),
        pd.to_datetime(result.validation["match_timestamp"], utc=True).min(),
        pd.to_datetime(result.validation["match_timestamp"], utc=True).max(),
        pd.to_datetime(result.test["match_timestamp"], utc=True).min(),
    ]
    if not all(left < right for left, right in pairwise(boundaries)):
        raise ValueError("Chronological split boundaries overlap")


def _split_training_timestamps(
    train_timestamps: list[pd.Timestamp],
    settings: SplitSettings,
) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    calibration_count = max(
        1,
        math.ceil(len(train_timestamps) * settings.calibration_fraction_within_train),
    )
    if len(train_timestamps) - calibration_count < 2:
        raise ValueError("The fit interval needs at least two timestamp groups")
    return train_timestamps[:-calibration_count], train_timestamps[-calibration_count:]


def chronological_split(
    frame: pd.DataFrame,
    settings: SplitSettings,
) -> ChronologicalSplits:
    """Create fit, calibration, validation, and test intervals without randomization."""
    ordered = frame.sort_values(["match_timestamp", "match_id"]).reset_index(drop=True)
    unique_timestamps = sorted(
        pd.to_datetime(ordered["match_timestamp"], utc=True).unique().tolist()
    )
    if len(unique_timestamps) < 10:
        raise ValueError("At least 10 unique timestamps are required for chronological splits")

    if settings.uses_timestamp_boundaries:
        if settings.validation_start_timestamp is None or settings.test_start_timestamp is None:
            raise AssertionError("Validated timestamp boundaries are missing")
        validation_start = pd.Timestamp(settings.validation_start_timestamp)
        test_start = pd.Timestamp(settings.test_start_timestamp)
        train_timestamps = [
            timestamp for timestamp in unique_timestamps if timestamp < validation_start
        ]
        validation_timestamps = [
            timestamp
            for timestamp in unique_timestamps
            if validation_start <= timestamp < test_start
        ]
        test_timestamps = [timestamp for timestamp in unique_timestamps if timestamp >= test_start]
    else:
        if settings.train_fraction is None or settings.validation_fraction is None:
            raise AssertionError("Validated split fractions are missing")
        train_count = math.floor(len(unique_timestamps) * settings.train_fraction)
        validation_count = math.floor(len(unique_timestamps) * settings.validation_fraction)
        train_timestamps = unique_timestamps[:train_count]
        validation_timestamps = unique_timestamps[train_count : train_count + validation_count]
        test_timestamps = unique_timestamps[train_count + validation_count :]

    if min(len(train_timestamps), len(validation_timestamps), len(test_timestamps)) < 2:
        raise ValueError("Each chronological partition needs at least two timestamp groups")

    fit_timestamps, calibration_timestamps = _split_training_timestamps(
        train_timestamps,
        settings,
    )

    result = ChronologicalSplits(
        fit=_take_timestamp_range(ordered, fit_timestamps),
        calibration=_take_timestamp_range(ordered, calibration_timestamps),
        validation=_take_timestamp_range(ordered, validation_timestamps),
        test=_take_timestamp_range(ordered, test_timestamps),
    )
    _validate_boundaries(result)
    return result


def rolling_origin_splits(
    frame: pd.DataFrame,
    split_settings: SplitSettings,
    backtest_settings: BacktestSettings,
) -> list[RollingOriginFold]:
    """Create expanding-window folds ending at the end of a development interval."""
    ordered = frame.sort_values(["match_timestamp", "match_id"]).reset_index(drop=True)
    unique_timestamps = sorted(
        pd.to_datetime(ordered["match_timestamp"], utc=True).unique().tolist()
    )
    validation_count = max(
        2,
        math.floor(len(unique_timestamps) * backtest_settings.validation_fraction_per_fold),
    )
    test_count = max(
        2,
        math.floor(len(unique_timestamps) * backtest_settings.test_fraction_per_fold),
    )
    first_train_count = (
        len(unique_timestamps) - validation_count - backtest_settings.fold_count * test_count
    )
    if first_train_count < 4:
        raise ValueError("Not enough timestamp groups for the configured rolling-origin folds")

    folds: list[RollingOriginFold] = []
    for fold_index in range(backtest_settings.fold_count):
        train_end = first_train_count + fold_index * test_count
        validation_end = train_end + validation_count
        test_end = validation_end + test_count
        train_timestamps = unique_timestamps[:train_end]
        validation_timestamps = unique_timestamps[train_end:validation_end]
        test_timestamps = unique_timestamps[validation_end:test_end]
        fit_timestamps, calibration_timestamps = _split_training_timestamps(
            train_timestamps,
            split_settings,
        )
        splits = ChronologicalSplits(
            fit=_take_timestamp_range(ordered, fit_timestamps),
            calibration=_take_timestamp_range(ordered, calibration_timestamps),
            validation=_take_timestamp_range(ordered, validation_timestamps),
            test=_take_timestamp_range(ordered, test_timestamps),
        )
        _validate_boundaries(splits)
        folds.append(
            RollingOriginFold(
                fold_number=fold_index + 1,
                splits=splits,
            )
        )

    if folds[-1].splits.test["match_timestamp"].max() != ordered["match_timestamp"].max():
        raise ValueError("The final rolling-origin fold must end at the development cutoff")
    return folds
