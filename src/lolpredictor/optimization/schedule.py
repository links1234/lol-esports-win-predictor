"""Deterministic, balanced optimization trial schedules."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Literal

from lolpredictor.optimization.settings import (
    FeatureMode,
    ModelFamily,
    OptimizationSettings,
    OutputMapping,
)

FeatureGroup = Literal[
    "core",
    "team_strength",
    "roster_form",
    "champion_meta",
    "player_champion",
    "draft_interactions",
    "regional",
    "categorical_context",
]

FEATURE_GROUP_ORDER: tuple[FeatureGroup, ...] = (
    "core",
    "team_strength",
    "roster_form",
    "champion_meta",
    "player_champion",
    "draft_interactions",
    "regional",
    "categorical_context",
)
OPTIONAL_NUMERIC_GROUPS: tuple[FeatureGroup, ...] = (
    "roster_form",
    "champion_meta",
    "player_champion",
    "draft_interactions",
    "regional",
)
TRIAL_SPEC_SCHEMA_VERSION = "1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _stable_seed(*values: object) -> int:
    encoded = "\x1f".join(str(value) for value in values).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _bounded_float(value: float) -> float:
    return float(f"{value:.12g}")


def _uniform(generator: random.Random, bounds: tuple[float, float]) -> float:
    return _bounded_float(generator.uniform(*bounds))


def _log_uniform(generator: random.Random, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return _bounded_float(math.exp(generator.uniform(math.log(low), math.log(high))))


def _integer(generator: random.Random, bounds: tuple[int, int]) -> int:
    return generator.randint(bounds[0], bounds[1])


@dataclass(frozen=True)
class TrialSpec:
    trial_id: int
    outer_fold: int
    family: ModelFamily
    feature_mode: FeatureMode
    family_trial_index: int
    seed: int
    output_mapping: OutputMapping
    feature_groups: tuple[FeatureGroup, ...]
    feature_parameters: dict[str, float | int]
    model_parameters: dict[str, float | int]
    spec_hash: str

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": TRIAL_SPEC_SCHEMA_VERSION,
            "trial_id": self.trial_id,
            "outer_fold": self.outer_fold,
            "family": self.family,
            "feature_mode": self.feature_mode,
            "family_trial_index": self.family_trial_index,
            "seed": self.seed,
            "output_mapping": self.output_mapping,
            "feature_groups": list(self.feature_groups),
            "feature_parameters": self.feature_parameters,
            "model_parameters": self.model_parameters,
        }
        if include_hash:
            value["spec_hash"] = self.spec_hash
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrialSpec:
        if value.get("schema_version") != TRIAL_SPEC_SCHEMA_VERSION:
            raise ValueError("Unsupported trial specification schema")
        without_hash = {key: item for key, item in value.items() if key != "spec_hash"}
        actual_hash = hashlib.sha256(_canonical_json(without_hash).encode()).hexdigest()
        if actual_hash != value.get("spec_hash"):
            raise ValueError("Trial specification hash mismatch")
        return cls(
            trial_id=int(value["trial_id"]),
            outer_fold=int(value["outer_fold"]),
            family=value["family"],
            feature_mode=value["feature_mode"],
            family_trial_index=int(value["family_trial_index"]),
            seed=int(value["seed"]),
            output_mapping=value["output_mapping"],
            feature_groups=tuple(value["feature_groups"]),
            feature_parameters={
                str(key): item for key, item in value["feature_parameters"].items()
            },
            model_parameters={str(key): item for key, item in value["model_parameters"].items()},
            spec_hash=str(value["spec_hash"]),
        )


def _sample_feature_groups(
    generator: random.Random,
    family: ModelFamily,
) -> tuple[FeatureGroup, ...]:
    included: set[FeatureGroup] = {"core", "team_strength"}
    probabilities: dict[FeatureGroup, float] = {
        "roster_form": 0.75,
        "champion_meta": 0.70,
        "player_champion": 0.55,
        "draft_interactions": 0.65,
        "regional": 0.75,
    }
    for group in OPTIONAL_NUMERIC_GROUPS:
        if generator.random() < probabilities[group]:
            included.add(group)
    if family in {"catboost", "elo_catboost_blend"} and generator.random() < 0.5:
        included.add("categorical_context")
    return tuple(group for group in FEATURE_GROUP_ORDER if group in included)


def _sample_feature_parameters(
    generator: random.Random,
    settings: OptimizationSettings,
    feature_mode: FeatureMode,
) -> dict[str, float | int]:
    if feature_mode == "cached":
        return {}
    space = settings.feature_replay_space
    return {
        "elo_k_factor": generator.choice(space.elo_k_factor),
        "player_elo_k_factor": generator.choice(space.player_elo_k_factor),
        "region_elo_k_factor": generator.choice(space.region_elo_k_factor),
        "form_window": generator.choice(space.form_window),
        "rate_prior_strength": generator.choice(space.rate_prior_strength),
        "player_champion_prior_strength": generator.choice(space.player_champion_prior_strength),
        "synergy_prior_strength": generator.choice(space.synergy_prior_strength),
        "matchup_prior_strength": generator.choice(space.matchup_prior_strength),
        "rating_half_life_days": generator.choice(space.rating_half_life_days),
    }


def _sample_model_parameters(
    generator: random.Random,
    settings: OptimizationSettings,
    family: ModelFamily,
) -> dict[str, float | int]:
    space = settings.search_space
    if family == "logistic_regression":
        return {"c": _log_uniform(generator, space.logistic_c)}
    if family == "histogram_gradient_boosting":
        return {
            "learning_rate": _log_uniform(generator, space.histogram_learning_rate),
            "iterations": _integer(generator, space.histogram_iterations),
            "max_leaf_nodes": _integer(generator, space.histogram_max_leaf_nodes),
            "l2_regularization": _log_uniform(
                generator,
                space.histogram_l2_regularization,
            ),
        }
    if family in {"catboost", "elo_catboost_blend"}:
        parameters: dict[str, float | int] = {
            "iterations": _integer(generator, space.catboost_iterations),
            "depth": _integer(generator, space.catboost_depth),
            "learning_rate": _log_uniform(generator, space.catboost_learning_rate),
            "l2_leaf_regularization": _log_uniform(
                generator,
                space.catboost_l2_leaf_regularization,
            ),
            "random_strength": _uniform(generator, space.catboost_random_strength),
            "rsm": _uniform(generator, space.catboost_rsm),
        }
        if family == "elo_catboost_blend":
            parameters["elo_weight"] = _uniform(generator, space.elo_blend_weight)
        return parameters
    if family == "dynamic_bradley_terry":
        return {
            "c": _log_uniform(generator, space.bradley_terry_c),
            "recency_half_life_days": _log_uniform(
                generator,
                space.bradley_terry_recency_half_life_days,
            ),
            "player_weight": _uniform(generator, space.bradley_terry_player_weight),
            "champion_weight": _uniform(generator, space.bradley_terry_champion_weight),
        }
    raise AssertionError(f"Unhandled model family: {family}")


def _trial_spec(
    *,
    trial_id: int,
    outer_fold: int,
    family: ModelFamily,
    feature_mode: FeatureMode,
    family_trial_index: int,
    settings: OptimizationSettings,
) -> TrialSpec:
    seed = _stable_seed(
        settings.random_seed,
        outer_fold,
        family,
        feature_mode,
        family_trial_index,
    )
    generator = random.Random(seed)
    output_mapping = generator.choice(settings.calibration_methods[family])
    value_without_hash = {
        "schema_version": TRIAL_SPEC_SCHEMA_VERSION,
        "trial_id": trial_id,
        "outer_fold": outer_fold,
        "family": family,
        "feature_mode": feature_mode,
        "family_trial_index": family_trial_index,
        "seed": seed,
        "output_mapping": output_mapping,
        "feature_groups": list(_sample_feature_groups(generator, family)),
        "feature_parameters": _sample_feature_parameters(
            generator,
            settings,
            feature_mode,
        ),
        "model_parameters": _sample_model_parameters(generator, settings, family),
    }
    spec_hash = hashlib.sha256(_canonical_json(value_without_hash).encode()).hexdigest()
    return TrialSpec.from_dict({**value_without_hash, "spec_hash": spec_hash})


def generate_trial_schedule(settings: OptimizationSettings) -> list[TrialSpec]:
    """Generate a family- and outer-balanced deterministic trial order."""
    family_count = len(settings.allocation.families)
    cached_per_family = settings.allocation.cached_feature_trials_per_outer // family_count
    replay_per_family = settings.allocation.replay_feature_trials_per_outer // family_count
    slots_per_family = cached_per_family + replay_per_family
    if cached_per_family != replay_per_family * 2:
        raise ValueError("The frozen schedule requires two cached trials per replay trial")

    specs: list[TrialSpec] = []
    trial_id = 1
    for slot in range(slots_per_family):
        feature_mode: FeatureMode = "replay" if slot % 3 == 2 else "cached"
        for combination_index in range(settings.nested_validation.outer_fold_count * family_count):
            outer_index = combination_index % settings.nested_validation.outer_fold_count
            family_index = (
                combination_index // settings.nested_validation.outer_fold_count
                + outer_index
                + slot
            ) % family_count
            outer_fold = outer_index + 1
            family = settings.allocation.families[family_index]
            specs.append(
                _trial_spec(
                    trial_id=trial_id,
                    outer_fold=outer_fold,
                    family=family,
                    feature_mode=feature_mode,
                    family_trial_index=slot + 1,
                    settings=settings,
                )
            )
            trial_id += 1

    if len(specs) != settings.trial_count:
        raise AssertionError("Generated trial schedule does not match the configured budget")
    return specs


def schedule_fingerprint(specs: list[TrialSpec]) -> str:
    encoded = _canonical_json([spec.to_dict() for spec in specs]).encode()
    return hashlib.sha256(encoded).hexdigest()
