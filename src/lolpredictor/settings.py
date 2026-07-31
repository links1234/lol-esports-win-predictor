"""Validated experiment settings."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StateSpaceFeatureSettings(BaseModel):
    """Frozen parameters for the optional uncertainty-aware feature filter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    update_requires_official: bool = True
    update_tournament_levels: tuple[str, ...] = ("Primary", "Premier")
    mean_half_life_days: float = Field(default=540.0, gt=0.0)
    variance_reversion_half_life_days: float = Field(default=180.0, gt=0.0)
    minimum_variance: float = Field(default=0.0001, gt=0.0)
    global_side_prior_variance: float = Field(default=0.04, gt=0.0)
    league_side_prior_variance: float = Field(default=0.04, gt=0.0)
    region_prior_variance: float = Field(default=0.36, gt=0.0)
    team_prior_variance: float = Field(default=0.64, gt=0.0)
    player_prior_variance: float = Field(default=0.64, gt=0.0)
    roster_prior_variance: float = Field(default=0.25, gt=0.0)
    champion_prior_variance: float = Field(default=0.16, gt=0.0)
    patch_champion_prior_variance: float = Field(default=0.09, gt=0.0)
    matchup_prior_variance: float = Field(default=0.16, gt=0.0)
    region_weight: float = Field(default=0.5, gt=0.0)
    player_weight: float = Field(default=0.6, gt=0.0)
    roster_weight: float = Field(default=0.35, gt=0.0)
    champion_weight: float = Field(default=0.5, gt=0.0)
    patch_champion_weight: float = Field(default=0.35, gt=0.0)
    matchup_weight: float = Field(default=0.4, gt=0.0)
    uncertainty_warning_standard_deviation: float = Field(default=1.5, gt=0.0)
    coverage_warning_fraction: float = Field(default=0.6, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_filter_contract(self) -> "StateSpaceFeatureSettings":
        if not self.update_tournament_levels:
            raise ValueError("State-space updates require at least one tournament level")
        if len(set(self.update_tournament_levels)) != len(self.update_tournament_levels):
            raise ValueError("State-space tournament levels must be unique")
        prior_variances = (
            self.global_side_prior_variance,
            self.league_side_prior_variance,
            self.region_prior_variance,
            self.team_prior_variance,
            self.player_prior_variance,
            self.roster_prior_variance,
            self.champion_prior_variance,
            self.patch_champion_prior_variance,
            self.matchup_prior_variance,
        )
        if self.minimum_variance >= min(prior_variances):
            raise ValueError("State-space minimum variance must be below every prior variance")
        return self


class FeatureSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_schema_version: str
    elo_initial_rating: float = 1500.0
    elo_k_factor: float = Field(default=24.0, gt=0.0)
    player_elo_k_factor: float = Field(default=16.0, gt=0.0)
    form_window: int = Field(default=8, ge=1)
    rate_prior_strength: float = Field(default=6.0, gt=0.0)
    player_champion_prior_strength: float = Field(default=3.0, gt=0.0)
    synergy_prior_strength: float = Field(default=16.0, gt=0.0)
    matchup_prior_strength: float = Field(default=12.0, gt=0.0)
    rating_half_life_days: float = Field(default=365.0, gt=0.0)
    region_elo_k_factor: float = Field(default=12.0, gt=0.0)
    regional_rating_levels: tuple[str, ...] = ("Primary", "Premier")
    regional_rating_excluded_regions: tuple[str, ...] = ("International",)
    state_space: StateSpaceFeatureSettings = Field(default_factory=StateSpaceFeatureSettings)

    @model_validator(mode="after")
    def validate_regional_rating_contract(self) -> "FeatureSettings":
        if not self.regional_rating_levels:
            raise ValueError("Regional ratings require at least one tournament level")
        if len(set(self.regional_rating_levels)) != len(self.regional_rating_levels):
            raise ValueError("Regional rating tournament levels must be unique")
        if len(set(self.regional_rating_excluded_regions)) != len(
            self.regional_rating_excluded_regions
        ):
            raise ValueError("Regional rating excluded regions must be unique")
        return self


class SplitSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_fraction: float | None = Field(default=None, gt=0.0, lt=1.0)
    validation_fraction: float | None = Field(default=None, gt=0.0, lt=1.0)
    validation_start_timestamp: datetime | None = None
    test_start_timestamp: datetime | None = None
    calibration_fraction_within_train: float = Field(gt=0.0, lt=0.5)
    refit_before_test: bool = False

    @model_validator(mode="after")
    def validate_split_mode(self) -> "SplitSettings":
        fraction_mode = self.train_fraction is not None or self.validation_fraction is not None
        timestamp_mode = (
            self.validation_start_timestamp is not None or self.test_start_timestamp is not None
        )
        if fraction_mode == timestamp_mode:
            raise ValueError("Configure exactly one split mode: fractions or timestamp boundaries")
        if fraction_mode:
            if self.train_fraction is None or self.validation_fraction is None:
                raise ValueError("Fraction split mode requires both split fractions")
            if self.train_fraction + self.validation_fraction >= 1.0:
                raise ValueError("train_fraction + validation_fraction must leave a test interval")
            return self

        validation_start = self.validation_start_timestamp
        test_start = self.test_start_timestamp
        if validation_start is None or test_start is None:
            raise ValueError("Timestamp split mode requires both timestamp boundaries")
        for field_name, value in (
            ("validation_start_timestamp", validation_start),
            ("test_start_timestamp", test_start),
        ):
            if value is None or value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{field_name} must include a timezone")
        normalized_validation_start = validation_start.astimezone(UTC)
        normalized_test_start = test_start.astimezone(UTC)
        self.validation_start_timestamp = normalized_validation_start
        self.test_start_timestamp = normalized_test_start
        if normalized_validation_start >= normalized_test_start:
            raise ValueError("Validation must begin before the test interval")
        return self

    @property
    def uses_timestamp_boundaries(self) -> bool:
        return self.validation_start_timestamp is not None


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_names: tuple[str, ...] | None = None
    logistic_c: float = Field(default=0.5, gt=0.0)
    gradient_boosting_learning_rate: float = Field(default=0.05, gt=0.0)
    gradient_boosting_max_iter: int = Field(default=120, ge=10)
    gradient_boosting_max_leaf_nodes: int = Field(default=15, ge=2)
    gradient_boosting_l2_regularization: float = Field(default=1.0, ge=0.0)
    recency_half_life_days: float = Field(default=365.0, gt=0.0)
    catboost_iterations: int = Field(default=250, ge=10)
    catboost_depth: int = Field(default=6, ge=2, le=12)
    catboost_learning_rate: float = Field(default=0.04, gt=0.0)
    catboost_l2_leaf_regularization: float = Field(default=8.0, ge=0.0)
    catboost_random_strength: float = Field(default=0.5, ge=0.0)
    elo_logistic_blend_elo_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    production_calibration_fraction: float = Field(default=0.08, gt=0.0, lt=0.25)

    @model_validator(mode="after")
    def validate_candidate_names(self) -> "ModelSettings":
        if self.candidate_names is None:
            return self
        if not self.candidate_names:
            raise ValueError("candidate_names must contain at least one model")
        if len(set(self.candidate_names)) != len(self.candidate_names):
            raise ValueError("candidate_names must be unique")
        return self


class ModelingPopulationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require_official: bool = False
    tournament_levels: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_population(self) -> "ModelingPopulationSettings":
        if len(set(self.tournament_levels)) != len(self.tournament_levels):
            raise ValueError("Modeling-population tournament levels must be unique")
        return self


class BacktestSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fold_count: int = Field(default=3, ge=2, le=10)
    validation_fraction_per_fold: float = Field(default=0.1, gt=0.0, lt=0.3)
    test_fraction_per_fold: float = Field(default=0.1, gt=0.0, lt=0.3)
    bootstrap_iterations: int = Field(default=500, ge=100, le=10_000)
    minimum_breakdown_sample_count: int = Field(default=100, ge=1)


class PredictionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stale_after_days: int = Field(default=45, ge=1)


class ReleaseGateSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    minimum_holdout_games: int = Field(default=750, ge=1)
    minimum_holdout_series: int = Field(default=300, ge=2)
    minimum_log_loss_improvement: float = Field(default=0.003, ge=0.0)
    maximum_expected_calibration_error: float = Field(default=0.04, ge=0.0, le=1.0)
    maximum_major_league_log_loss_regression: float = Field(default=0.01, ge=0.0)
    major_leagues: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_enabled_gate(self) -> "ReleaseGateSettings":
        if self.enabled and not self.major_leagues:
            raise ValueError("An enabled release gate requires canonical major-league labels")
        if len(set(self.major_leagues)) != len(self.major_leagues):
            raise ValueError("Release-gate major-league labels must be unique")
        return self


class ExperimentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    random_seed: int
    features: FeatureSettings
    splits: SplitSettings
    models: ModelSettings
    modeling_population: ModelingPopulationSettings = Field(
        default_factory=ModelingPopulationSettings
    )
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    prediction: PredictionSettings
    release_gate: ReleaseGateSettings = Field(default_factory=ReleaseGateSettings)

    def resolved(self) -> dict[str, Any]:
        resolved = self.model_dump(mode="json")
        if not self.features.state_space.enabled:
            resolved["features"].pop("state_space", None)
        if self.models.candidate_names is None:
            resolved["models"].pop("candidate_names", None)
        return resolved


def load_settings(path: Path) -> ExperimentSettings:
    """Load and validate a YAML experiment file without executing content."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Experiment config must contain a mapping: {path}")
    return ExperimentSettings.model_validate(raw)
