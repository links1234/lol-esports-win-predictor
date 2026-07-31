"""Validated configuration for the nested chronological optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lolpredictor.settings import ExperimentSettings, load_settings

ModelFamily = Literal[
    "logistic_regression",
    "histogram_gradient_boosting",
    "catboost",
    "elo_catboost_blend",
    "dynamic_bradley_terry",
]
FeatureMode = Literal["cached", "replay"]
OutputMapping = Literal["native", "platt", "beta", "isotonic"]

MODEL_FAMILIES: tuple[ModelFamily, ...] = (
    "logistic_regression",
    "histogram_gradient_boosting",
    "catboost",
    "elo_catboost_blend",
    "dynamic_bradley_terry",
)
OUTPUT_MAPPINGS: tuple[OutputMapping, ...] = (
    "native",
    "platt",
    "beta",
    "isotonic",
)


class NestedValidationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outer_fold_count: int = Field(ge=2, le=10)
    inner_fold_count: int = Field(ge=2, le=10)
    validation_fraction_per_fold: float = Field(gt=0.0, lt=0.3)
    test_fraction_per_fold: float = Field(gt=0.0, lt=0.3)
    calibration_fraction_within_train: float = Field(gt=0.0, lt=0.5)


class TrialAllocationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cached_feature_trials_per_outer: int = Field(ge=1)
    replay_feature_trials_per_outer: int = Field(ge=1)
    families: tuple[ModelFamily, ...]

    @model_validator(mode="after")
    def validate_balanced_allocation(self) -> TrialAllocationSettings:
        if tuple(self.families) != MODEL_FAMILIES:
            raise ValueError(
                "Optimization families must contain the frozen five-family order exactly"
            )
        family_count = len(self.families)
        for field_name, count in (
            ("cached_feature_trials_per_outer", self.cached_feature_trials_per_outer),
            ("replay_feature_trials_per_outer", self.replay_feature_trials_per_outer),
        ):
            if count % family_count:
                raise ValueError(f"{field_name} must be divisible by the family count")
        return self


class OptimizationSelectionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    maximum_expected_calibration_error: float = Field(ge=0.0, le=1.0)
    maximum_major_league_log_loss_regression: float = Field(ge=0.0)
    minimum_breakdown_sample_count: int = Field(ge=1)
    require_no_inner_fold_regression_vs_elo: bool
    fallback_candidate: Literal["elo_catboost_regional_raw_blend_50"]


class FeatureReplaySpace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    elo_k_factor: tuple[float, ...]
    player_elo_k_factor: tuple[float, ...]
    region_elo_k_factor: tuple[float, ...]
    form_window: tuple[int, ...]
    rate_prior_strength: tuple[float, ...]
    player_champion_prior_strength: tuple[float, ...]
    synergy_prior_strength: tuple[float, ...]
    matchup_prior_strength: tuple[float, ...]
    rating_half_life_days: tuple[float, ...]

    @model_validator(mode="after")
    def validate_choices(self) -> FeatureReplaySpace:
        for field_name, values in self:
            if not values:
                raise ValueError(f"{field_name} must contain at least one choice")
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} choices must be unique")
            if any(value <= 0 for value in values):
                raise ValueError(f"{field_name} choices must be positive")
        return self


class ModelSearchSpace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    logistic_c: tuple[float, float]
    histogram_learning_rate: tuple[float, float]
    histogram_iterations: tuple[int, int]
    histogram_max_leaf_nodes: tuple[int, int]
    histogram_l2_regularization: tuple[float, float]
    catboost_iterations: tuple[int, int]
    catboost_depth: tuple[int, int]
    catboost_learning_rate: tuple[float, float]
    catboost_l2_leaf_regularization: tuple[float, float]
    catboost_random_strength: tuple[float, float]
    catboost_rsm: tuple[float, float]
    elo_blend_weight: tuple[float, float]
    bradley_terry_c: tuple[float, float]
    bradley_terry_recency_half_life_days: tuple[float, float]
    bradley_terry_player_weight: tuple[float, float]
    bradley_terry_champion_weight: tuple[float, float]

    @model_validator(mode="after")
    def validate_ranges(self) -> ModelSearchSpace:
        for field_name, values in self:
            low, high = values
            if low >= high:
                raise ValueError(f"{field_name} lower bound must be below its upper bound")
            if low < 0:
                raise ValueError(f"{field_name} bounds cannot be negative")
        return self


class OptimizationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    study_name: str = Field(min_length=1)
    base_config: Path
    development_cutoff_timestamp: datetime
    random_seed: int
    trial_count: int = Field(ge=1)
    timeout_hours: float = Field(gt=0.0)
    worker_count: int = Field(ge=1, le=32)
    nested_validation: NestedValidationSettings
    allocation: TrialAllocationSettings
    selection: OptimizationSelectionSettings
    feature_replay_space: FeatureReplaySpace
    search_space: ModelSearchSpace
    calibration_methods: dict[ModelFamily, tuple[OutputMapping, ...]]

    @field_validator("development_cutoff_timestamp")
    @classmethod
    def normalize_cutoff(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("development_cutoff_timestamp must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_study_contract(self) -> OptimizationSettings:
        per_outer = (
            self.allocation.cached_feature_trials_per_outer
            + self.allocation.replay_feature_trials_per_outer
        )
        expected_trials = self.nested_validation.outer_fold_count * per_outer
        if self.trial_count != expected_trials:
            raise ValueError(
                f"trial_count must equal the balanced outer allocation ({expected_trials})"
            )
        if set(self.calibration_methods) != set(MODEL_FAMILIES):
            raise ValueError("Calibration methods must be configured for every model family")
        for family, methods in self.calibration_methods.items():
            if not methods:
                raise ValueError(f"{family} requires at least one output mapping")
            if len(set(methods)) != len(methods):
                raise ValueError(f"{family} output mappings must be unique")
            if any(method not in OUTPUT_MAPPINGS for method in methods):
                raise ValueError(f"{family} contains an unsupported output mapping")
        return self

    def resolved(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


@dataclass(frozen=True)
class OptimizationConfiguration:
    optimization: OptimizationSettings
    experiment: ExperimentSettings
    optimization_path: Path
    experiment_path: Path

    def resolved(self) -> dict[str, Any]:
        return {
            "optimization": self.optimization.resolved(),
            "experiment": self.experiment.resolved(),
            "optimization_path": str(self.optimization_path),
            "experiment_path": str(self.experiment_path),
        }


def load_optimization_configuration(path: Path) -> OptimizationConfiguration:
    """Load the optimizer and its referenced experiment without executing content."""
    optimization_path = path.resolve()
    raw = yaml.safe_load(optimization_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Optimization config must contain a mapping: {path}")
    optimization = OptimizationSettings.model_validate(raw)
    base_path = optimization.base_config
    experiment_path = (
        base_path.resolve()
        if base_path.is_absolute()
        else (optimization_path.parent / base_path).resolve()
    )
    experiment = load_settings(experiment_path)
    split_cutoff = experiment.splits.validation_start_timestamp
    if split_cutoff != optimization.development_cutoff_timestamp:
        raise ValueError("Optimization cutoff must equal the base experiment validation boundary")
    if optimization.random_seed != experiment.random_seed:
        raise ValueError("Optimization and base experiment random seeds must match")
    return OptimizationConfiguration(
        optimization=optimization,
        experiment=experiment,
        optimization_path=optimization_path,
        experiment_path=experiment_path,
    )
