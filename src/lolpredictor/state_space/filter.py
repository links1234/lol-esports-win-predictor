"""Deterministic diagonal-Gaussian filtering for paired team outcomes."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np

from lolpredictor.settings import StateSpaceFeatureSettings

SkillKind = Literal[
    "global_side",
    "league_side",
    "region",
    "team",
    "player",
    "roster",
    "champion",
    "patch_champion",
    "matchup",
]

SKILL_KINDS: tuple[SkillKind, ...] = (
    "global_side",
    "league_side",
    "region",
    "team",
    "player",
    "roster",
    "champion",
    "patch_champion",
    "matchup",
)
_KEY_SEPARATOR = "\x1f"


def _normalize_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("State-space timestamps must include a timezone")
    return value.astimezone(UTC)


@dataclass(frozen=True)
class GaussianSkill:
    """One diagonal Gaussian latent component."""

    mean: float
    variance: float
    last_seen: datetime | None
    games: int

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "mean": self.mean,
            "variance": self.variance,
            "last_seen": self.last_seen.astimezone(UTC).isoformat()
            if self.last_seen is not None
            else None,
            "games": self.games,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GaussianSkill:
        timestamp_text = value.get("last_seen")
        timestamp = (
            _normalize_timestamp(datetime.fromisoformat(str(timestamp_text)))
            if timestamp_text is not None
            else None
        )
        skill = cls(
            mean=float(value["mean"]),
            variance=float(value["variance"]),
            last_seen=timestamp,
            games=int(value["games"]),
        )
        if not math.isfinite(skill.mean):
            raise ValueError("State-space skill mean must be finite")
        if not math.isfinite(skill.variance) or skill.variance <= 0.0:
            raise ValueError("State-space skill variance must be finite and positive")
        if skill.games < 0:
            raise ValueError("State-space skill games cannot be negative")
        return skill


@dataclass(frozen=True)
class StateSpacePrediction:
    """Uncertainty-integrated probability and its projected design state."""

    probability: float
    linear_mean: float
    linear_variance: float
    projected_skills: dict[str, GaussianSkill]
    contributions: dict[SkillKind, float]

    @property
    def linear_standard_deviation(self) -> float:
        return math.sqrt(self.linear_variance)


@dataclass(frozen=True)
class GaussianObservation:
    """One binary result with a stable identifier and signed latent design."""

    observation_id: str
    design: Mapping[str, float]
    outcome: bool


def skill_key(kind: SkillKind, *parts: str) -> str:
    """Encode a collision-safe, stable latent identity."""
    if kind not in SKILL_KINDS:
        raise ValueError(f"Unsupported state-space skill kind: {kind}")
    if not parts or any(not part for part in parts):
        raise ValueError("State-space skill keys require nonempty identity parts")
    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":"))
    return f"{kind}{_KEY_SEPARATOR}{payload}"


def skill_kind(key: str) -> SkillKind:
    raw_kind, separator, _ = key.partition(_KEY_SEPARATOR)
    if not separator or raw_kind not in SKILL_KINDS:
        raise ValueError("Malformed state-space skill key")
    return raw_kind


def prior_variance(key: str, settings: StateSpaceFeatureSettings) -> float:
    kind = skill_kind(key)
    return {
        "global_side": settings.global_side_prior_variance,
        "league_side": settings.league_side_prior_variance,
        "region": settings.region_prior_variance,
        "team": settings.team_prior_variance,
        "player": settings.player_prior_variance,
        "roster": settings.roster_prior_variance,
        "champion": settings.champion_prior_variance,
        "patch_champion": settings.patch_champion_prior_variance,
        "matchup": settings.matchup_prior_variance,
    }[kind]


def project_gaussian_skill(
    skill: GaussianSkill | None,
    key: str,
    timestamp: datetime,
    settings: StateSpaceFeatureSettings,
) -> GaussianSkill:
    """Project a latent component to a later timestamp without mutating state."""
    target = _normalize_timestamp(timestamp)
    prior = prior_variance(key, settings)
    if skill is None:
        return GaussianSkill(
            mean=0.0,
            variance=prior,
            last_seen=None,
            games=0,
        )
    if skill.last_seen is None:
        raise ValueError("An observed state-space skill must have a last_seen timestamp")
    if target <= skill.last_seen:
        raise ValueError("State-space projection target must be strictly after its evidence")
    age_days = (target - skill.last_seen).total_seconds() / 86400.0
    mean_retention = math.pow(0.5, age_days / settings.mean_half_life_days)
    variance_retention = math.pow(
        0.5,
        age_days / settings.variance_reversion_half_life_days,
    )
    bounded_variance = min(prior, max(settings.minimum_variance, skill.variance))
    variance = prior - (prior - bounded_variance) * variance_retention
    return GaussianSkill(
        mean=skill.mean * mean_retention,
        variance=max(settings.minimum_variance, min(prior, variance)),
        last_seen=skill.last_seen,
        games=skill.games,
    )


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(max(value, -700.0))
    return exponent / (1.0 + exponent)


def _normalized_design(design: Mapping[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, raw_coefficient in design.items():
        skill_kind(key)
        coefficient = float(raw_coefficient)
        if not math.isfinite(coefficient):
            raise ValueError("State-space design coefficients must be finite")
        if abs(coefficient) > 1e-15:
            normalized[key] = coefficient
    if not normalized:
        raise ValueError("State-space designs cannot be empty")
    return normalized


def predict_gaussian_design(
    skills: Mapping[str, GaussianSkill],
    design: Mapping[str, float],
    timestamp: datetime,
    settings: StateSpaceFeatureSettings,
) -> StateSpacePrediction:
    """Predict one outcome from state available strictly before its timestamp."""
    normalized = _normalized_design(design)
    projected = {
        key: project_gaussian_skill(skills.get(key), key, timestamp, settings)
        for key in sorted(normalized)
    }
    linear_mean = math.fsum(normalized[key] * projected[key].mean for key in sorted(normalized))
    linear_variance = math.fsum(
        normalized[key] ** 2 * projected[key].variance for key in sorted(normalized)
    )
    uncertainty_scale = math.sqrt(1.0 + math.pi * linear_variance / 8.0)
    probability = float(
        np.clip(
            _sigmoid(linear_mean / uncertainty_scale),
            1e-6,
            1.0 - 1e-6,
        )
    )
    contributions: dict[SkillKind, float] = {kind: 0.0 for kind in SKILL_KINDS}
    for key in sorted(normalized):
        kind = skill_kind(key)
        contributions[kind] += normalized[key] * projected[key].mean
    return StateSpacePrediction(
        probability=probability,
        linear_mean=linear_mean,
        linear_variance=linear_variance,
        projected_skills=projected,
        contributions=contributions,
    )


def update_gaussian_skills(
    skills: dict[str, GaussianSkill],
    observations: Sequence[GaussianObservation],
    timestamp: datetime,
    settings: StateSpaceFeatureSettings,
) -> dict[str, float]:
    """Apply one complete timestamp group with an order-independent update."""
    target = _normalize_timestamp(timestamp)
    if not observations:
        return {}
    ordered = sorted(observations, key=lambda item: item.observation_id)
    identifiers = [item.observation_id for item in ordered]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("State-space observation identifiers must be unique in a batch")

    normalized_designs = {item.observation_id: _normalized_design(item.design) for item in ordered}
    keys = sorted({key for item in ordered for key in normalized_designs[item.observation_id]})
    projected = {
        key: project_gaussian_skill(skills.get(key), key, target, settings) for key in keys
    }
    gradients: defaultdict[str, list[float]] = defaultdict(list)
    precisions: defaultdict[str, list[float]] = defaultdict(list)
    appearances: defaultdict[str, int] = defaultdict(int)
    probabilities: dict[str, float] = {}

    for item in ordered:
        design = normalized_designs[item.observation_id]
        linear_mean = math.fsum(design[key] * projected[key].mean for key in sorted(design))
        linear_variance = math.fsum(
            design[key] ** 2 * projected[key].variance for key in sorted(design)
        )
        uncertainty_scale = math.sqrt(1.0 + math.pi * linear_variance / 8.0)
        probability = float(
            np.clip(
                _sigmoid(linear_mean / uncertainty_scale),
                1e-6,
                1.0 - 1e-6,
            )
        )
        probabilities[item.observation_id] = probability
        residual = float(item.outcome) - probability
        curvature = probability * (1.0 - probability)
        for key in sorted(design):
            effective_coefficient = design[key] / uncertainty_scale
            gradients[key].append(effective_coefficient * residual)
            precisions[key].append(effective_coefficient**2 * curvature)
            appearances[key] += 1

    for key in keys:
        previous = projected[key]
        posterior_variance = 1.0 / (1.0 / previous.variance + math.fsum(precisions[key]))
        posterior_variance = max(settings.minimum_variance, posterior_variance)
        posterior_mean = previous.mean + posterior_variance * math.fsum(gradients[key])
        if not math.isfinite(posterior_mean) or not math.isfinite(posterior_variance):
            raise ValueError("State-space update produced a nonfinite posterior")
        skills[key] = GaussianSkill(
            mean=posterior_mean,
            variance=min(prior_variance(key, settings), posterior_variance),
            last_seen=target,
            games=previous.games + appearances[key],
        )
    return probabilities
