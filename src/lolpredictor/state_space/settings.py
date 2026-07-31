"""Validated and frozen configuration for the v6 state-space experiment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lolpredictor.optimization.settings import NestedValidationSettings
from lolpredictor.settings import (
    ExperimentSettings,
    StateSpaceFeatureSettings,
    load_settings,
)

V6CandidateName = Literal[
    "state_space_native",
    "state_space_platt",
    "state_space_augmented_logistic",
    "state_space_v4_blend_15",
    "state_space_v4_blend_30",
]
V6_CANDIDATE_NAMES: tuple[V6CandidateName, ...] = (
    "state_space_native",
    "state_space_platt",
    "state_space_augmented_logistic",
    "state_space_v4_blend_15",
    "state_space_v4_blend_30",
)
V4_CONTROL_NAME = "elo_catboost_regional_raw_blend_50"
V6_MAJOR_LEAGUES = (
    "Tencent LoL Pro League",
    "LoL Champions Korea",
    "LoL EMEA Championship",
    "League of Legends Championship Pacific",
    "League of Legends Championship Series",
    "Circuit Brazilian League of Legends",
)


class V6CandidateSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    names: tuple[V6CandidateName, ...]
    augmented_logistic_c: float = Field(gt=0.0)
    state_calibration_fraction: float = Field(gt=0.0, lt=0.5)
    blend_15_state_weight: float = Field(gt=0.0, lt=1.0)
    blend_30_state_weight: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_frozen_candidates(self) -> V6CandidateSettings:
        if self.names != V6_CANDIDATE_NAMES:
            raise ValueError("V6 candidate names and ordering are frozen")
        frozen_values = {
            "augmented_logistic_c": 0.10,
            "state_calibration_fraction": 0.10,
            "blend_15_state_weight": 0.15,
            "blend_30_state_weight": 0.30,
        }
        for field_name, expected in frozen_values.items():
            if float(getattr(self, field_name)) != expected:
                raise ValueError(f"V6 {field_name} is frozen at {expected}")
        return self


class V6SelectionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    maximum_expected_calibration_error: float = Field(ge=0.0, le=1.0)
    maximum_inner_fold_log_loss_regression_vs_v4: float = Field(ge=0.0)
    maximum_major_league_log_loss_regression_vs_v4: float = Field(ge=0.0)
    minimum_breakdown_sample_count: int = Field(ge=1)
    fallback_candidate: Literal["elo_catboost_regional_raw_blend_50"]

    @model_validator(mode="after")
    def validate_frozen_selection(self) -> V6SelectionSettings:
        frozen_values: dict[str, float | int] = {
            "maximum_expected_calibration_error": 0.04,
            "maximum_inner_fold_log_loss_regression_vs_v4": 0.005,
            "maximum_major_league_log_loss_regression_vs_v4": 0.01,
            "minimum_breakdown_sample_count": 100,
        }
        for field_name, expected in frozen_values.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"V6 {field_name} is frozen at {expected}")
        return self


class V6DecisionGateSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_outer_games: int = Field(ge=1)
    minimum_outer_series: int = Field(ge=2)
    minimum_log_loss_improvement_vs_v4: float = Field(ge=0.0)
    maximum_expected_calibration_error: float = Field(ge=0.0, le=1.0)
    maximum_outer_fold_log_loss_regression_vs_v4: float = Field(ge=0.0)
    maximum_major_league_log_loss_regression_vs_v4: float = Field(ge=0.0)
    major_leagues: tuple[str, ...]

    @model_validator(mode="after")
    def validate_major_leagues(self) -> V6DecisionGateSettings:
        if not self.major_leagues:
            raise ValueError("The v6 decision gate requires major leagues")
        if len(set(self.major_leagues)) != len(self.major_leagues):
            raise ValueError("V6 decision-gate major leagues must be unique")
        frozen_values: dict[str, float | int] = {
            "minimum_outer_games": 750,
            "minimum_outer_series": 300,
            "minimum_log_loss_improvement_vs_v4": 0.003,
            "maximum_expected_calibration_error": 0.04,
            "maximum_outer_fold_log_loss_regression_vs_v4": 0.01,
            "maximum_major_league_log_loss_regression_vs_v4": 0.01,
        }
        for field_name, expected in frozen_values.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"V6 {field_name} is frozen at {expected}")
        if self.major_leagues != V6_MAJOR_LEAGUES:
            raise ValueError("V6 decision-gate major leagues and ordering are frozen")
        return self


class V6StudySettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    study_name: str = Field(min_length=1)
    base_config: Path
    development_cutoff_timestamp: datetime
    random_seed: int
    bootstrap_iterations: int = Field(ge=100, le=10_000)
    nested_validation: NestedValidationSettings
    state_space: StateSpaceFeatureSettings
    candidates: V6CandidateSettings
    selection: V6SelectionSettings
    decision_gate: V6DecisionGateSettings

    @field_validator("development_cutoff_timestamp")
    @classmethod
    def normalize_cutoff(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("development_cutoff_timestamp must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_frozen_filter(self) -> V6StudySettings:
        if not self.state_space.enabled:
            raise ValueError("The v6 state-space filter must be enabled")
        frozen = StateSpaceFeatureSettings(
            enabled=True,
            update_requires_official=True,
            update_tournament_levels=("Primary", "Premier"),
            mean_half_life_days=540.0,
            variance_reversion_half_life_days=180.0,
            minimum_variance=0.0001,
            global_side_prior_variance=0.04,
            league_side_prior_variance=0.04,
            region_prior_variance=0.36,
            team_prior_variance=0.64,
            player_prior_variance=0.64,
            roster_prior_variance=0.25,
            champion_prior_variance=0.16,
            patch_champion_prior_variance=0.09,
            matchup_prior_variance=0.16,
            region_weight=0.5,
            player_weight=0.6,
            roster_weight=0.35,
            champion_weight=0.5,
            patch_champion_weight=0.35,
            matchup_weight=0.4,
            uncertainty_warning_standard_deviation=1.5,
            coverage_warning_fraction=0.6,
        )
        if self.state_space != frozen:
            raise ValueError("V6 state-space parameters differ from the frozen protocol")
        if self.study_name == "leaguepedia-pre2026-v6-state-space":
            expected_nested = NestedValidationSettings(
                outer_fold_count=4,
                inner_fold_count=3,
                validation_fraction_per_fold=0.10,
                test_fraction_per_fold=0.10,
                calibration_fraction_within_train=0.10,
            )
            if self.nested_validation != expected_nested:
                raise ValueError("The real v6 nested-validation design is frozen")
            if self.bootstrap_iterations != 1000:
                raise ValueError("The real v6 bootstrap iteration count is frozen at 1000")
            if self.development_cutoff_timestamp != datetime(
                2026,
                1,
                1,
                tzinfo=UTC,
            ):
                raise ValueError("The real v6 exclusive cutoff is frozen at 2026-01-01")
        return self

    def resolved(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


@dataclass(frozen=True)
class V6Configuration:
    study: V6StudySettings
    experiment: ExperimentSettings
    study_path: Path
    experiment_path: Path

    def resolved(self) -> dict[str, Any]:
        return {
            "study": self.study.resolved(),
            "experiment": self.experiment.resolved(),
            "study_path": str(self.study_path),
            "experiment_path": str(self.experiment_path),
        }


def load_v6_configuration(path: Path) -> V6Configuration:
    """Load a v6 study and derive its state-space-enabled experiment."""
    study_path = path.resolve()
    raw = yaml.safe_load(study_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"V6 config must contain a mapping: {path}")
    study = V6StudySettings.model_validate(raw)
    base_path = study.base_config
    experiment_path = (
        base_path.resolve()
        if base_path.is_absolute()
        else (study_path.parent / base_path).resolve()
    )
    base = load_settings(experiment_path)
    if base.splits.validation_start_timestamp != study.development_cutoff_timestamp:
        raise ValueError("V6 cutoff must equal the base experiment validation boundary")
    if base.random_seed != study.random_seed:
        raise ValueError("V6 and base experiment random seeds must match")
    experiment_payload = base.model_dump(mode="python")
    experiment_payload.update(
        {
            "experiment_name": study.study_name,
            "model_version": "v6-state-space-locked",
            "features": {
                **base.features.model_dump(mode="python"),
                "feature_schema_version": "point-in-time-v6-state-space",
                "state_space": study.state_space.model_dump(mode="python"),
            },
        }
    )
    experiment = ExperimentSettings.model_validate(experiment_payload)
    return V6Configuration(
        study=study,
        experiment=experiment,
        study_path=study_path,
        experiment_path=experiment_path,
    )
