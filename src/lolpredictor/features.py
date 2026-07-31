"""Strict point-in-time feature generation shared by training and prediction."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import combinations, groupby
from typing import Any

import numpy as np
import pandas as pd

from lolpredictor.schemas import DraftRequest, HistoricalMatch
from lolpredictor.settings import FeatureSettings
from lolpredictor.state_space.filter import (
    GaussianObservation,
    GaussianSkill,
    StateSpacePrediction,
    predict_gaussian_design,
    skill_key,
    update_gaussian_skills,
)

LEGACY_FEATURE_NAMES = (
    "blue_side_prior",
    "league_blue_side_prior",
    "blue_first_pick",
    "first_pick_known",
    "blue_first_pick_prior",
    "blue_elo",
    "red_elo",
    "elo_diff",
    "elo_blue_win_probability",
    "blue_player_elo",
    "red_player_elo",
    "player_elo_diff",
    "blue_team_form",
    "red_team_form",
    "team_form_diff",
    "blue_team_games",
    "red_team_games",
    "team_games_log_diff",
    "blue_champion_strength",
    "red_champion_strength",
    "champion_strength_diff",
    "blue_player_champion_strength",
    "red_player_champion_strength",
    "player_champion_strength_diff",
    "blue_roster_experience",
    "red_roster_experience",
    "roster_experience_diff",
    "blue_roster_continuity",
    "red_roster_continuity",
    "roster_continuity_diff",
    "blue_team_roster_experience",
    "red_team_roster_experience",
    "team_roster_experience_diff",
    "blue_draft_known_fraction",
    "red_draft_known_fraction",
    "blue_ban_strength",
    "red_ban_strength",
    "ban_strength_diff",
    "blue_patch_champion_strength",
    "red_patch_champion_strength",
    "patch_champion_strength_diff",
    "blue_synergy_strength",
    "red_synergy_strength",
    "synergy_strength_diff",
    "blue_role_matchup_strength",
    "role_matchup_advantage",
    "head_to_head_blue_rate",
    "head_to_head_games",
    "game_number",
    "series_score_diff",
    "series_game_progress",
    "fearless_ban_count",
    "patch_major",
    "patch_minor",
)

ROLE_NAMES = ("top", "jungle", "mid", "bottom", "support")
INACTIVITY_FEATURE_NAMES = (
    "blue_team_inactivity_log_days",
    "red_team_inactivity_log_days",
    "team_inactivity_log_days_diff",
    "blue_adjusted_elo",
    "red_adjusted_elo",
    "adjusted_elo_diff",
    "blue_roster_min_games_log",
    "red_roster_min_games_log",
    "roster_min_games_log_diff",
)
ROLE_FEATURE_NAMES = tuple(
    feature_name
    for role in ROLE_NAMES
    for feature_name in (
        f"{role}_player_elo_diff",
        f"{role}_adjusted_player_elo_diff",
        f"blue_{role}_champion_strength",
        f"red_{role}_champion_strength",
        f"{role}_champion_strength_diff",
        f"blue_{role}_champion_games_log",
        f"red_{role}_champion_games_log",
        f"{role}_champion_games_log_diff",
        f"blue_{role}_player_champion_strength",
        f"red_{role}_player_champion_strength",
        f"{role}_player_champion_strength_diff",
        f"blue_{role}_player_champion_games_log",
        f"red_{role}_player_champion_games_log",
        f"{role}_player_champion_games_log_diff",
        f"{role}_matchup_strength",
        f"{role}_matchup_games_log",
    )
)
PRE_REGIONAL_FEATURE_NAMES = (
    *LEGACY_FEATURE_NAMES,
    *INACTIVITY_FEATURE_NAMES,
    *ROLE_FEATURE_NAMES,
)
REGIONAL_FEATURE_NAMES = (
    "blue_region_elo",
    "red_region_elo",
    "region_elo_diff",
    "region_elo_blue_win_probability",
    "blue_region_cross_region_games_log",
    "red_region_cross_region_games_log",
    "region_cross_region_games_log_diff",
    "blue_home_region_known",
    "red_home_region_known",
    "cross_region_match",
    "pooled_elo_diff",
    "pooled_elo_blue_win_probability",
)
V4_FEATURE_NAMES = (*PRE_REGIONAL_FEATURE_NAMES, *REGIONAL_FEATURE_NAMES)
STATE_SPACE_FEATURE_NAMES = (
    "state_space_blue_win_probability",
    "state_space_log_odds_mean",
    "state_space_log_odds_standard_deviation",
    "state_space_side_strength",
    "state_space_region_strength_diff",
    "state_space_team_strength_diff",
    "state_space_player_strength_diff",
    "state_space_roster_strength_diff",
    "state_space_champion_strength_diff",
    "state_space_patch_champion_strength_diff",
    "state_space_matchup_strength",
    "state_space_blue_team_standard_deviation",
    "state_space_red_team_standard_deviation",
    "state_space_blue_roster_standard_deviation",
    "state_space_red_roster_standard_deviation",
    "state_space_roster_uncertainty_difference",
    "state_space_observed_component_fraction",
)
FEATURE_NAMES = (*V4_FEATURE_NAMES, *STATE_SPACE_FEATURE_NAMES)

MODEL_CONTEXT_NAMES = (
    "league",
    "patch",
    "region",
    "tournament_level",
    "official_status",
    "blue_team_key",
    "red_team_key",
    *tuple(
        feature_name
        for side in ("blue", "red")
        for role in ROLE_NAMES
        for feature_name in (
            f"{side}_{role}_player",
            f"{side}_{role}_champion",
            f"{side}_{role}_player_champion",
        )
    ),
    *tuple(f"{side}_ban_{ban_index}" for side in ("blue", "red") for ban_index in range(5)),
)

PROVENANCE_COLUMNS = (
    "feature_history_max_timestamp",
    "elo_history_max_timestamp",
    "form_history_max_timestamp",
    "champion_history_max_timestamp",
    "player_champion_history_max_timestamp",
    "regional_elo_history_max_timestamp",
)


class TemporalLeakageError(ValueError):
    """Raised when a feature state would include information from the target or future."""


@dataclass
class OutcomeStat:
    wins: float = 0.0
    games: int = 0

    def add(self, won: bool) -> None:
        self.wins += float(won)
        self.games += 1

    def as_dict(self) -> dict[str, float | int]:
        return {"wins": self.wins, "games": self.games}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OutcomeStat:
        return cls(wins=float(value["wins"]), games=int(value["games"]))


def _timestamp_to_text(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value else None


def _timestamp_from_text(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value).astimezone(UTC) if value else None


def _stats_to_dict(values: dict[str, OutcomeStat]) -> dict[str, dict[str, float | int]]:
    return {key: stat.as_dict() for key, stat in sorted(values.items())}


def _nested_stats_to_dict(
    values: dict[str, dict[str, OutcomeStat]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    return {
        outer_key: _stats_to_dict(inner_values)
        for outer_key, inner_values in sorted(values.items())
    }


def _timestamps_to_dict(values: dict[str, datetime]) -> dict[str, str]:
    return {key: value.astimezone(UTC).isoformat() for key, value in sorted(values.items())}


def _nested_timestamps_to_dict(
    values: dict[str, dict[str, datetime]],
) -> dict[str, dict[str, str]]:
    return {
        outer_key: _timestamps_to_dict(inner_values)
        for outer_key, inner_values in sorted(values.items())
    }


@dataclass
class FeatureState:
    """Mutable state containing only outcomes already observed."""

    settings: FeatureSettings
    team_elo: dict[str, float] = field(default_factory=dict)
    region_elo: dict[str, float] = field(default_factory=dict)
    region_cross_region_games: dict[str, int] = field(default_factory=dict)
    team_home_region: dict[str, str] = field(default_factory=dict)
    player_elo: dict[str, float] = field(default_factory=dict)
    team_form: dict[str, list[int]] = field(default_factory=dict)
    team_outcomes: dict[str, OutcomeStat] = field(default_factory=dict)
    champion_outcomes: dict[str, OutcomeStat] = field(default_factory=dict)
    role_champion_outcomes: dict[str, OutcomeStat] = field(default_factory=dict)
    patch_champion_outcomes: dict[str, dict[str, OutcomeStat]] = field(default_factory=dict)
    champion_pair_outcomes: dict[str, OutcomeStat] = field(default_factory=dict)
    role_matchup_outcomes: dict[str, OutcomeStat] = field(default_factory=dict)
    head_to_head_outcomes: dict[str, OutcomeStat] = field(default_factory=dict)
    player_champion_outcomes: dict[str, dict[str, OutcomeStat]] = field(default_factory=dict)
    player_games: dict[str, int] = field(default_factory=dict)
    team_player_games: dict[str, dict[str, int]] = field(default_factory=dict)
    team_last_roster: dict[str, tuple[str, ...]] = field(default_factory=dict)
    league_blue_outcomes: dict[str, OutcomeStat] = field(default_factory=dict)
    league_regions: dict[str, str] = field(default_factory=dict)
    league_tournament_levels: dict[str, str] = field(default_factory=dict)
    league_official_status: dict[str, bool] = field(default_factory=dict)
    global_blue_outcomes: OutcomeStat = field(default_factory=OutcomeStat)
    first_pick_outcomes: OutcomeStat = field(default_factory=OutcomeStat)
    team_aliases: dict[str, str] = field(default_factory=dict)
    player_aliases: dict[str, str] = field(default_factory=dict)
    ambiguous_team_aliases: set[str] = field(default_factory=set)
    ambiguous_player_aliases: set[str] = field(default_factory=set)
    known_teams: set[str] = field(default_factory=set)
    known_players: set[str] = field(default_factory=set)
    known_champions: set[str] = field(default_factory=set)
    known_leagues: set[str] = field(default_factory=set)
    known_patches: set[str] = field(default_factory=set)
    team_last_seen: dict[str, datetime] = field(default_factory=dict)
    region_last_seen: dict[str, datetime] = field(default_factory=dict)
    champion_last_seen: dict[str, datetime] = field(default_factory=dict)
    player_last_seen: dict[str, datetime] = field(default_factory=dict)
    player_champion_last_seen: dict[str, dict[str, datetime]] = field(default_factory=dict)
    league_last_seen: dict[str, datetime] = field(default_factory=dict)
    state_space_skills: dict[str, GaussianSkill] = field(default_factory=dict)
    data_cutoff_timestamp: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_schema_version": self.settings.feature_schema_version,
            "settings": self.settings.model_dump(mode="json"),
            "team_elo": dict(sorted(self.team_elo.items())),
            "region_elo": dict(sorted(self.region_elo.items())),
            "region_cross_region_games": dict(sorted(self.region_cross_region_games.items())),
            "team_home_region": dict(sorted(self.team_home_region.items())),
            "player_elo": dict(sorted(self.player_elo.items())),
            "team_form": {key: value for key, value in sorted(self.team_form.items())},
            "team_outcomes": _stats_to_dict(self.team_outcomes),
            "champion_outcomes": _stats_to_dict(self.champion_outcomes),
            "role_champion_outcomes": _stats_to_dict(self.role_champion_outcomes),
            "patch_champion_outcomes": _nested_stats_to_dict(self.patch_champion_outcomes),
            "champion_pair_outcomes": _stats_to_dict(self.champion_pair_outcomes),
            "role_matchup_outcomes": _stats_to_dict(self.role_matchup_outcomes),
            "head_to_head_outcomes": _stats_to_dict(self.head_to_head_outcomes),
            "player_champion_outcomes": _nested_stats_to_dict(self.player_champion_outcomes),
            "player_games": dict(sorted(self.player_games.items())),
            "team_player_games": {
                team: dict(sorted(players.items()))
                for team, players in sorted(self.team_player_games.items())
            },
            "team_last_roster": {
                team: list(players) for team, players in sorted(self.team_last_roster.items())
            },
            "league_blue_outcomes": _stats_to_dict(self.league_blue_outcomes),
            "league_regions": dict(sorted(self.league_regions.items())),
            "league_tournament_levels": dict(sorted(self.league_tournament_levels.items())),
            "league_official_status": dict(sorted(self.league_official_status.items())),
            "global_blue_outcomes": self.global_blue_outcomes.as_dict(),
            "first_pick_outcomes": self.first_pick_outcomes.as_dict(),
            "team_aliases": dict(sorted(self.team_aliases.items())),
            "player_aliases": dict(sorted(self.player_aliases.items())),
            "ambiguous_team_aliases": sorted(self.ambiguous_team_aliases),
            "ambiguous_player_aliases": sorted(self.ambiguous_player_aliases),
            "known_teams": sorted(self.known_teams),
            "known_players": sorted(self.known_players),
            "known_champions": sorted(self.known_champions),
            "known_leagues": sorted(self.known_leagues),
            "known_patches": sorted(self.known_patches),
            "team_last_seen": _timestamps_to_dict(self.team_last_seen),
            "region_last_seen": _timestamps_to_dict(self.region_last_seen),
            "champion_last_seen": _timestamps_to_dict(self.champion_last_seen),
            "player_last_seen": _timestamps_to_dict(self.player_last_seen),
            "player_champion_last_seen": _nested_timestamps_to_dict(self.player_champion_last_seen),
            "league_last_seen": _timestamps_to_dict(self.league_last_seen),
            "state_space_skills": {
                key: skill.to_dict() for key, skill in sorted(self.state_space_skills.items())
            },
            "data_cutoff_timestamp": _timestamp_to_text(self.data_cutoff_timestamp),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FeatureState:
        state = cls(settings=FeatureSettings.model_validate(value["settings"]))
        state.team_elo = {key: float(item) for key, item in value["team_elo"].items()}
        state.region_elo = {
            str(key): float(item) for key, item in value.get("region_elo", {}).items()
        }
        state.region_cross_region_games = {
            str(key): int(item) for key, item in value.get("region_cross_region_games", {}).items()
        }
        state.team_home_region = {
            str(key): str(item) for key, item in value.get("team_home_region", {}).items()
        }
        state.player_elo = {key: float(item) for key, item in value.get("player_elo", {}).items()}
        state.team_form = {
            key: [int(item) for item in items] for key, items in value["team_form"].items()
        }
        state.team_outcomes = {
            key: OutcomeStat.from_dict(item) for key, item in value["team_outcomes"].items()
        }
        state.champion_outcomes = {
            key: OutcomeStat.from_dict(item) for key, item in value["champion_outcomes"].items()
        }
        state.role_champion_outcomes = {
            key: OutcomeStat.from_dict(item)
            for key, item in value.get("role_champion_outcomes", {}).items()
        }
        state.patch_champion_outcomes = {
            patch: {
                champion: OutcomeStat.from_dict(stat) for champion, stat in champion_values.items()
            }
            for patch, champion_values in value.get("patch_champion_outcomes", {}).items()
        }
        state.champion_pair_outcomes = {
            key: OutcomeStat.from_dict(item)
            for key, item in value.get("champion_pair_outcomes", {}).items()
        }
        state.role_matchup_outcomes = {
            key: OutcomeStat.from_dict(item)
            for key, item in value.get("role_matchup_outcomes", {}).items()
        }
        state.head_to_head_outcomes = {
            key: OutcomeStat.from_dict(item)
            for key, item in value.get("head_to_head_outcomes", {}).items()
        }
        state.player_champion_outcomes = {
            player: {
                champion: OutcomeStat.from_dict(stat) for champion, stat in champion_values.items()
            }
            for player, champion_values in value["player_champion_outcomes"].items()
        }
        state.player_games = {key: int(item) for key, item in value["player_games"].items()}
        state.team_player_games = {
            team: {player: int(games) for player, games in players.items()}
            for team, players in value.get("team_player_games", {}).items()
        }
        state.team_last_roster = {
            team: tuple(str(player) for player in players)
            for team, players in value.get("team_last_roster", {}).items()
        }
        state.league_blue_outcomes = {
            key: OutcomeStat.from_dict(item) for key, item in value["league_blue_outcomes"].items()
        }
        state.league_regions = {
            str(key): str(item) for key, item in value.get("league_regions", {}).items()
        }
        state.league_tournament_levels = {
            str(key): str(item) for key, item in value.get("league_tournament_levels", {}).items()
        }
        state.league_official_status = {
            str(key): bool(item) for key, item in value.get("league_official_status", {}).items()
        }
        state.global_blue_outcomes = OutcomeStat.from_dict(value["global_blue_outcomes"])
        state.first_pick_outcomes = OutcomeStat.from_dict(
            value.get("first_pick_outcomes", {"wins": 0.0, "games": 0})
        )
        state.team_aliases = {
            str(key): str(item) for key, item in value.get("team_aliases", {}).items()
        }
        state.player_aliases = {
            str(key): str(item) for key, item in value.get("player_aliases", {}).items()
        }
        state.ambiguous_team_aliases = set(value.get("ambiguous_team_aliases", []))
        state.ambiguous_player_aliases = set(value.get("ambiguous_player_aliases", []))
        state.known_teams = set(value["known_teams"])
        state.known_players = set(value["known_players"])
        state.known_champions = set(value["known_champions"])
        state.known_leagues = set(value["known_leagues"])
        state.known_patches = set(value["known_patches"])
        state.team_last_seen = {
            key: parsed
            for key, item in value["team_last_seen"].items()
            if (parsed := _timestamp_from_text(item)) is not None
        }
        state.region_last_seen = {
            key: parsed
            for key, item in value.get("region_last_seen", {}).items()
            if (parsed := _timestamp_from_text(item)) is not None
        }
        state.champion_last_seen = {
            key: parsed
            for key, item in value["champion_last_seen"].items()
            if (parsed := _timestamp_from_text(item)) is not None
        }
        state.player_last_seen = {
            key: parsed
            for key, item in value["player_last_seen"].items()
            if (parsed := _timestamp_from_text(item)) is not None
        }
        state.player_champion_last_seen = {
            player: {
                champion: parsed
                for champion, item in champion_values.items()
                if (parsed := _timestamp_from_text(item)) is not None
            }
            for player, champion_values in value["player_champion_last_seen"].items()
        }
        state.league_last_seen = {
            key: parsed
            for key, item in value["league_last_seen"].items()
            if (parsed := _timestamp_from_text(item)) is not None
        }
        state.state_space_skills = {
            str(key): GaussianSkill.from_dict(item)
            for key, item in value.get("state_space_skills", {}).items()
        }
        state.data_cutoff_timestamp = _timestamp_from_text(value["data_cutoff_timestamp"])
        return state


@dataclass(frozen=True)
class FeatureComputation:
    values: dict[str, float]
    provenance: dict[str, datetime | None]


def _smoothed_rate(stat: OutcomeStat | None, prior: float, strength: float) -> float:
    if stat is None:
        return prior
    return (stat.wins + prior * strength) / (stat.games + strength)


def _team_form(state: FeatureState, team: str) -> float:
    outcomes = state.team_form.get(team, [])
    return float(sum(outcomes) / len(outcomes)) if outcomes else 0.5


def _champion_rate(state: FeatureState, champion: str) -> float:
    return _smoothed_rate(
        state.champion_outcomes.get(champion),
        0.5,
        state.settings.rate_prior_strength,
    )


def _role_champion_key(role_index: int, champion: str) -> str:
    return f"{role_index}\x1f{champion}"


def _role_champion_rate(
    state: FeatureState,
    role_index: int,
    champion: str,
) -> float:
    return _smoothed_rate(
        state.role_champion_outcomes.get(_role_champion_key(role_index, champion)),
        _champion_rate(state, champion),
        state.settings.rate_prior_strength,
    )


def _stat_games_log(stat: OutcomeStat | None) -> float:
    return math.log1p(stat.games if stat is not None else 0)


def _inactivity_days(last_seen: datetime | None, target: datetime) -> float:
    if last_seen is None:
        return 0.0
    return max(0.0, (target - last_seen).total_seconds() / 86400.0)


def _adjusted_rating(
    rating: float,
    last_seen: datetime | None,
    target: datetime,
    settings: FeatureSettings,
) -> float:
    inactivity_days = _inactivity_days(last_seen, target)
    retained_fraction = math.pow(
        0.5,
        inactivity_days / settings.rating_half_life_days,
    )
    return settings.elo_initial_rating + (rating - settings.elo_initial_rating) * retained_fraction


def _champion_average(state: FeatureState, champions: Sequence[str]) -> float:
    if not champions:
        return 0.5
    return float(np.mean([_champion_rate(state, champion) for champion in champions]))


def _patch_champion_rate(
    state: FeatureState,
    patch: str,
    champion: str,
) -> float:
    return _smoothed_rate(
        state.patch_champion_outcomes.get(patch, {}).get(champion),
        _champion_rate(state, champion),
        state.settings.rate_prior_strength,
    )


def _patch_champion_average(
    state: FeatureState,
    patch: str,
    champions: Sequence[str],
) -> float:
    if not champions:
        return 0.5
    return float(np.mean([_patch_champion_rate(state, patch, champion) for champion in champions]))


def _champion_pair_key(left: str, right: str) -> str:
    first, second = sorted((left, right))
    return f"{first}\x1f{second}"


def _role_matchup_key(role_index: int, champion: str, opponent: str) -> str:
    return f"{role_index}\x1f{champion}\x1f{opponent}"


def _head_to_head_key(team: str, opponent: str) -> str:
    return f"{team}\x1f{opponent}"


def _synergy_average(
    state: FeatureState,
    champions: Sequence[str],
) -> float:
    rates = [
        _smoothed_rate(
            state.champion_pair_outcomes.get(_champion_pair_key(left, right)),
            (_champion_rate(state, left) + _champion_rate(state, right)) / 2.0,
            state.settings.synergy_prior_strength,
        )
        for left, right in combinations(champions, 2)
    ]
    return float(np.mean(rates)) if rates else 0.5


def _role_matchup_average(
    state: FeatureState,
    blue_champions: Sequence[str],
    red_champions: Sequence[str],
) -> float:
    rates = [
        _smoothed_rate(
            state.role_matchup_outcomes.get(
                _role_matchup_key(role_index, blue_champion, red_champion)
            ),
            0.5,
            state.settings.matchup_prior_strength,
        )
        for role_index, (blue_champion, red_champion) in enumerate(
            zip(blue_champions, red_champions, strict=True)
        )
    ]
    return float(np.mean(rates))


def _player_champion_rate(state: FeatureState, player: str, champion: str) -> float:
    player_stats = state.player_champion_outcomes.get(player, {})
    return _smoothed_rate(
        player_stats.get(champion),
        _champion_rate(state, champion),
        state.settings.player_champion_prior_strength,
    )


def _player_champion_average(
    state: FeatureState,
    players: Sequence[str],
    champions: Sequence[str],
) -> float:
    return float(
        np.mean(
            [
                _player_champion_rate(state, player, champion)
                for player, champion in zip(players, champions, strict=True)
            ]
        )
    )


def _roster_continuity(
    state: FeatureState,
    team: str,
    players: Sequence[str],
) -> float:
    previous = set(state.team_last_roster.get(team, ()))
    return len(previous & set(players)) / 5.0 if previous else 0.0


def _team_roster_experience(
    state: FeatureState,
    team: str,
    players: Sequence[str],
) -> float:
    appearances = state.team_player_games.get(team, {})
    return float(np.mean([math.log1p(appearances.get(player, 0)) for player in players]))


def _max_timestamp(values: Iterable[datetime | None]) -> datetime | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _patch_numbers(patch: str) -> tuple[float, float]:
    parts = patch.split(".")
    if len(parts) == 2:
        return float(int(parts[0])), float(int(parts[1]))
    return float(int(parts[0])), float(int(parts[2]))


def _identity_alias(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _identity_key(
    kind: str,
    name: str,
    explicit_id: str | None,
    aliases: dict[str, str],
) -> str:
    if explicit_id is not None:
        return explicit_id
    alias = _identity_alias(name)
    return aliases.get(alias, f"{kind}:name:{alias}")


def _register_identity(
    *,
    kind: str,
    name: str,
    explicit_id: str | None,
    aliases: dict[str, str],
    ambiguous_aliases: set[str],
) -> str:
    key = explicit_id or f"{kind}:name:{_identity_alias(name)}"
    alias = _identity_alias(name)
    if alias not in ambiguous_aliases:
        previous = aliases.get(alias)
        if previous is None or previous == key:
            aliases[alias] = key
        else:
            aliases.pop(alias)
            ambiguous_aliases.add(alias)
    return key


def _request_identity_keys(
    request: DraftRequest,
    state: FeatureState,
) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    blue_team = _identity_key(
        "team",
        request.blue_team,
        request.blue_team_id,
        state.team_aliases,
    )
    red_team = _identity_key(
        "team",
        request.red_team,
        request.red_team_id,
        state.team_aliases,
    )
    blue_ids = request.blue_player_ids or (None,) * 5
    red_ids = request.red_player_ids or (None,) * 5
    blue_players = tuple(
        _identity_key("player", name, player_id, state.player_aliases)
        for name, player_id in zip(request.blue_players, blue_ids, strict=True)
    )
    red_players = tuple(
        _identity_key("player", name, player_id, state.player_aliases)
        for name, player_id in zip(request.red_players, red_ids, strict=True)
    )
    return blue_team, red_team, blue_players, red_players


def _request_metadata(
    request: DraftRequest,
    state: FeatureState,
) -> tuple[str | None, str | None, bool | None]:
    region = getattr(request, "region", None)
    tournament_level = getattr(request, "tournament_level", None)
    is_official = getattr(request, "is_official", None)
    if region is None:
        region = state.league_regions.get(request.league)
    if tournament_level is None:
        tournament_level = state.league_tournament_levels.get(request.league)
    if is_official is None:
        is_official = state.league_official_status.get(request.league)
    return region, tournament_level, is_official


def _request_home_regions(
    request: DraftRequest,
    state: FeatureState,
    blue_team: str,
    red_team: str,
) -> tuple[str | None, str | None]:
    region, _, _ = _request_metadata(request, state)
    if region is not None and region not in state.settings.regional_rating_excluded_regions:
        return region, region
    return (
        state.team_home_region.get(blue_team),
        state.team_home_region.get(red_team),
    )


def _add_design_coefficient(
    design: dict[str, float],
    key: str,
    coefficient: float,
) -> None:
    updated = design.get(key, 0.0) + coefficient
    if abs(updated) <= 1e-15:
        design.pop(key, None)
    else:
        design[key] = updated


def _state_space_design(
    request: DraftRequest,
    state: FeatureState,
    *,
    blue_team: str,
    red_team: str,
    blue_players: tuple[str, ...],
    red_players: tuple[str, ...],
    blue_home_region: str | None,
    red_home_region: str | None,
) -> dict[str, float]:
    settings = state.settings.state_space
    design: dict[str, float] = {}
    _add_design_coefficient(
        design,
        skill_key("global_side", "blue"),
        1.0,
    )
    _add_design_coefficient(
        design,
        skill_key("league_side", request.league),
        1.0,
    )
    if blue_home_region is not None and red_home_region is not None:
        _add_design_coefficient(
            design,
            skill_key("region", blue_home_region),
            settings.region_weight,
        )
        _add_design_coefficient(
            design,
            skill_key("region", red_home_region),
            -settings.region_weight,
        )
    _add_design_coefficient(design, skill_key("team", blue_team), 1.0)
    _add_design_coefficient(design, skill_key("team", red_team), -1.0)

    player_coefficient = settings.player_weight / len(ROLE_NAMES)
    for player in blue_players:
        _add_design_coefficient(
            design,
            skill_key("player", player),
            player_coefficient,
        )
    for player in red_players:
        _add_design_coefficient(
            design,
            skill_key("player", player),
            -player_coefficient,
        )
    _add_design_coefficient(
        design,
        skill_key("roster", blue_team, *blue_players),
        settings.roster_weight,
    )
    _add_design_coefficient(
        design,
        skill_key("roster", red_team, *red_players),
        -settings.roster_weight,
    )

    champion_coefficient = settings.champion_weight / len(ROLE_NAMES)
    patch_champion_coefficient = settings.patch_champion_weight / len(ROLE_NAMES)
    matchup_coefficient = settings.matchup_weight / len(ROLE_NAMES)
    for role, blue_champion, red_champion in zip(
        ROLE_NAMES,
        request.blue_picks,
        request.red_picks,
        strict=True,
    ):
        _add_design_coefficient(
            design,
            skill_key("champion", role, blue_champion),
            champion_coefficient,
        )
        _add_design_coefficient(
            design,
            skill_key("champion", role, red_champion),
            -champion_coefficient,
        )
        _add_design_coefficient(
            design,
            skill_key("patch_champion", request.patch, role, blue_champion),
            patch_champion_coefficient,
        )
        _add_design_coefficient(
            design,
            skill_key("patch_champion", request.patch, role, red_champion),
            -patch_champion_coefficient,
        )
        first, second = sorted((blue_champion, red_champion))
        orientation = 1.0 if blue_champion == first else -1.0
        _add_design_coefficient(
            design,
            skill_key("matchup", role, first, second),
            orientation * matchup_coefficient,
        )
    return design


def _weighted_standard_deviation(
    prediction: StateSpacePrediction,
    weighted_keys: Sequence[tuple[str, float]],
) -> float:
    return math.sqrt(
        math.fsum(
            coefficient**2 * prediction.projected_skills[key].variance
            for key, coefficient in weighted_keys
        )
    )


def _disabled_state_space_values() -> dict[str, float]:
    return {
        "state_space_blue_win_probability": 0.5,
        "state_space_log_odds_mean": 0.0,
        "state_space_log_odds_standard_deviation": 0.0,
        "state_space_side_strength": 0.0,
        "state_space_region_strength_diff": 0.0,
        "state_space_team_strength_diff": 0.0,
        "state_space_player_strength_diff": 0.0,
        "state_space_roster_strength_diff": 0.0,
        "state_space_champion_strength_diff": 0.0,
        "state_space_patch_champion_strength_diff": 0.0,
        "state_space_matchup_strength": 0.0,
        "state_space_blue_team_standard_deviation": 0.0,
        "state_space_red_team_standard_deviation": 0.0,
        "state_space_blue_roster_standard_deviation": 0.0,
        "state_space_red_roster_standard_deviation": 0.0,
        "state_space_roster_uncertainty_difference": 0.0,
        "state_space_observed_component_fraction": 0.0,
    }


def _state_space_values(
    request: DraftRequest,
    state: FeatureState,
    *,
    blue_team: str,
    red_team: str,
    blue_players: tuple[str, ...],
    red_players: tuple[str, ...],
    blue_home_region: str | None,
    red_home_region: str | None,
) -> dict[str, float]:
    settings = state.settings.state_space
    if not settings.enabled:
        return _disabled_state_space_values()
    design = _state_space_design(
        request,
        state,
        blue_team=blue_team,
        red_team=red_team,
        blue_players=blue_players,
        red_players=red_players,
        blue_home_region=blue_home_region,
        red_home_region=red_home_region,
    )
    prediction = predict_gaussian_design(
        state.state_space_skills,
        design,
        request.match_timestamp,
        settings,
    )
    blue_team_key = skill_key("team", blue_team)
    red_team_key = skill_key("team", red_team)
    blue_roster_key = skill_key("roster", blue_team, *blue_players)
    red_roster_key = skill_key("roster", red_team, *red_players)
    player_coefficient = settings.player_weight / len(ROLE_NAMES)
    blue_roster_uncertainty = _weighted_standard_deviation(
        prediction,
        [
            (blue_roster_key, settings.roster_weight),
            *[(skill_key("player", player), player_coefficient) for player in blue_players],
        ],
    )
    red_roster_uncertainty = _weighted_standard_deviation(
        prediction,
        [
            (red_roster_key, settings.roster_weight),
            *[(skill_key("player", player), player_coefficient) for player in red_players],
        ],
    )
    total_design_weight = math.fsum(abs(value) for value in design.values())
    observed_design_weight = math.fsum(
        abs(coefficient)
        for key, coefficient in design.items()
        if prediction.projected_skills[key].games > 0
    )
    contributions = prediction.contributions
    return {
        "state_space_blue_win_probability": prediction.probability,
        "state_space_log_odds_mean": prediction.linear_mean,
        "state_space_log_odds_standard_deviation": (prediction.linear_standard_deviation),
        "state_space_side_strength": (contributions["global_side"] + contributions["league_side"]),
        "state_space_region_strength_diff": contributions["region"],
        "state_space_team_strength_diff": contributions["team"],
        "state_space_player_strength_diff": contributions["player"],
        "state_space_roster_strength_diff": contributions["roster"],
        "state_space_champion_strength_diff": contributions["champion"],
        "state_space_patch_champion_strength_diff": (contributions["patch_champion"]),
        "state_space_matchup_strength": contributions["matchup"],
        "state_space_blue_team_standard_deviation": math.sqrt(
            prediction.projected_skills[blue_team_key].variance
        ),
        "state_space_red_team_standard_deviation": math.sqrt(
            prediction.projected_skills[red_team_key].variance
        ),
        "state_space_blue_roster_standard_deviation": blue_roster_uncertainty,
        "state_space_red_roster_standard_deviation": red_roster_uncertainty,
        "state_space_roster_uncertainty_difference": (
            blue_roster_uncertainty - red_roster_uncertainty
        ),
        "state_space_observed_component_fraction": (observed_design_weight / total_design_weight),
    }


def build_model_context(
    request: DraftRequest,
    state: FeatureState,
) -> dict[str, str]:
    """Build stable categorical context shared by training and artifact prediction."""
    blue_team, red_team, blue_players, red_players = _request_identity_keys(request, state)
    region, tournament_level, is_official = _request_metadata(request, state)
    context: dict[str, str] = {
        "league": request.league,
        "patch": request.patch,
        "region": region or "<unknown>",
        "tournament_level": tournament_level or "<unknown>",
        "official_status": (
            "official"
            if is_official is True
            else "unofficial"
            if is_official is False
            else "<unknown>"
        ),
        "blue_team_key": blue_team,
        "red_team_key": red_team,
    }
    for side, players, champions in (
        ("blue", blue_players, request.blue_picks),
        ("red", red_players, request.red_picks),
    ):
        for role, player, champion in zip(
            ROLE_NAMES,
            players,
            champions,
            strict=True,
        ):
            context[f"{side}_{role}_player"] = player
            context[f"{side}_{role}_champion"] = champion
            context[f"{side}_{role}_player_champion"] = f"{player}\x1f{champion}"
    for side, bans in (("blue", request.blue_bans), ("red", request.red_bans)):
        for ban_index in range(5):
            context[f"{side}_ban_{ban_index}"] = (
                bans[ban_index] if ban_index < len(bans) else "<none>"
            )
    if set(context) != set(MODEL_CONTEXT_NAMES):
        raise AssertionError("Model context does not match its declared feature contract")
    return context


def build_model_row(
    request: DraftRequest,
    state: FeatureState,
    computation: FeatureComputation | None = None,
) -> dict[str, float | str]:
    resolved_computation = computation or compute_features(request, state)
    return {
        **resolved_computation.values,
        **build_model_context(request, state),
    }


def compute_features(request: DraftRequest, state: FeatureState) -> FeatureComputation:
    blue_team, red_team, blue_players, red_players = _request_identity_keys(request, state)
    blue_elo = state.team_elo.get(blue_team, state.settings.elo_initial_rating)
    red_elo = state.team_elo.get(red_team, state.settings.elo_initial_rating)
    elo_probability = 1.0 / (1.0 + 10.0 ** ((red_elo - blue_elo) / 400.0))
    blue_player_elo = float(
        np.mean(
            [
                state.player_elo.get(player, state.settings.elo_initial_rating)
                for player in blue_players
            ]
        )
    )
    red_player_elo = float(
        np.mean(
            [
                state.player_elo.get(player, state.settings.elo_initial_rating)
                for player in red_players
            ]
        )
    )
    blue_team_inactivity_days = _inactivity_days(
        state.team_last_seen.get(blue_team),
        request.match_timestamp,
    )
    red_team_inactivity_days = _inactivity_days(
        state.team_last_seen.get(red_team),
        request.match_timestamp,
    )
    blue_adjusted_elo = _adjusted_rating(
        blue_elo,
        state.team_last_seen.get(blue_team),
        request.match_timestamp,
        state.settings,
    )
    red_adjusted_elo = _adjusted_rating(
        red_elo,
        state.team_last_seen.get(red_team),
        request.match_timestamp,
        state.settings,
    )
    blue_home_region, red_home_region = _request_home_regions(
        request,
        state,
        blue_team,
        red_team,
    )
    blue_home_region_known = blue_home_region is not None
    red_home_region_known = red_home_region is not None
    both_home_regions_known = blue_home_region_known and red_home_region_known
    if both_home_regions_known:
        assert blue_home_region is not None
        assert red_home_region is not None
        blue_region_elo = state.region_elo.get(
            blue_home_region,
            state.settings.elo_initial_rating,
        )
        red_region_elo = state.region_elo.get(
            red_home_region,
            state.settings.elo_initial_rating,
        )
        blue_region_games = state.region_cross_region_games.get(blue_home_region, 0)
        red_region_games = state.region_cross_region_games.get(red_home_region, 0)
    else:
        blue_region_elo = state.settings.elo_initial_rating
        red_region_elo = state.settings.elo_initial_rating
        blue_region_games = 0
        red_region_games = 0
    region_elo_diff = blue_region_elo - red_region_elo
    region_elo_probability = 1.0 / (1.0 + 10.0 ** ((red_region_elo - blue_region_elo) / 400.0))
    cross_region_match = both_home_regions_known and blue_home_region != red_home_region
    pooled_elo_diff = blue_adjusted_elo - red_adjusted_elo + region_elo_diff
    pooled_elo_probability = 1.0 / (1.0 + 10.0 ** (-pooled_elo_diff / 400.0))

    global_prior = _smoothed_rate(
        state.global_blue_outcomes,
        0.5,
        state.settings.rate_prior_strength,
    )
    league_prior = _smoothed_rate(
        state.league_blue_outcomes.get(request.league),
        global_prior,
        state.settings.rate_prior_strength,
    )
    blue_form = _team_form(state, blue_team)
    red_form = _team_form(state, red_team)
    blue_games = state.team_outcomes.get(blue_team, OutcomeStat()).games
    red_games = state.team_outcomes.get(red_team, OutcomeStat()).games
    blue_champion = _champion_average(state, request.blue_picks)
    red_champion = _champion_average(state, request.red_picks)
    blue_player_champion = _player_champion_average(state, blue_players, request.blue_picks)
    red_player_champion = _player_champion_average(state, red_players, request.red_picks)
    blue_experience = float(
        np.mean([math.log1p(state.player_games.get(player, 0)) for player in blue_players])
    )
    red_experience = float(
        np.mean([math.log1p(state.player_games.get(player, 0)) for player in red_players])
    )
    blue_roster_min_games = min(state.player_games.get(player, 0) for player in blue_players)
    red_roster_min_games = min(state.player_games.get(player, 0) for player in red_players)
    blue_known = (
        sum(
            state.champion_outcomes.get(champion, OutcomeStat()).games > 0
            for champion in request.blue_picks
        )
        / 5.0
    )
    red_known = (
        sum(
            state.champion_outcomes.get(champion, OutcomeStat()).games > 0
            for champion in request.red_picks
        )
        / 5.0
    )
    blue_ban = _champion_average(state, request.blue_bans)
    red_ban = _champion_average(state, request.red_bans)
    blue_patch_champion = _patch_champion_average(
        state,
        request.patch,
        request.blue_picks,
    )
    red_patch_champion = _patch_champion_average(
        state,
        request.patch,
        request.red_picks,
    )
    blue_synergy = _synergy_average(state, request.blue_picks)
    red_synergy = _synergy_average(state, request.red_picks)
    role_matchup = _role_matchup_average(
        state,
        request.blue_picks,
        request.red_picks,
    )
    blue_continuity = _roster_continuity(state, blue_team, blue_players)
    red_continuity = _roster_continuity(state, red_team, red_players)
    blue_team_roster_experience = _team_roster_experience(
        state,
        blue_team,
        blue_players,
    )
    red_team_roster_experience = _team_roster_experience(
        state,
        red_team,
        red_players,
    )
    head_to_head = state.head_to_head_outcomes.get(_head_to_head_key(blue_team, red_team))
    head_to_head_rate = _smoothed_rate(
        head_to_head,
        elo_probability,
        state.settings.rate_prior_strength,
    )
    patch_major, patch_minor = _patch_numbers(request.patch)
    first_pick_rate = _smoothed_rate(
        state.first_pick_outcomes,
        0.5,
        state.settings.rate_prior_strength,
    )
    if request.first_pick_side == "blue":
        blue_first_pick = 1.0
        blue_first_pick_prior = first_pick_rate
    elif request.first_pick_side == "red":
        blue_first_pick = 0.0
        blue_first_pick_prior = 1.0 - first_pick_rate
    else:
        blue_first_pick = 0.5
        blue_first_pick_prior = 0.5

    values = {
        "blue_side_prior": global_prior,
        "league_blue_side_prior": league_prior,
        "blue_first_pick": blue_first_pick,
        "first_pick_known": float(request.first_pick_side is not None),
        "blue_first_pick_prior": blue_first_pick_prior,
        "blue_elo": blue_elo,
        "red_elo": red_elo,
        "elo_diff": blue_elo - red_elo,
        "elo_blue_win_probability": elo_probability,
        "blue_player_elo": blue_player_elo,
        "red_player_elo": red_player_elo,
        "player_elo_diff": blue_player_elo - red_player_elo,
        "blue_team_form": blue_form,
        "red_team_form": red_form,
        "team_form_diff": blue_form - red_form,
        "blue_team_games": float(blue_games),
        "red_team_games": float(red_games),
        "team_games_log_diff": math.log1p(blue_games) - math.log1p(red_games),
        "blue_champion_strength": blue_champion,
        "red_champion_strength": red_champion,
        "champion_strength_diff": blue_champion - red_champion,
        "blue_player_champion_strength": blue_player_champion,
        "red_player_champion_strength": red_player_champion,
        "player_champion_strength_diff": blue_player_champion - red_player_champion,
        "blue_roster_experience": blue_experience,
        "red_roster_experience": red_experience,
        "roster_experience_diff": blue_experience - red_experience,
        "blue_roster_continuity": blue_continuity,
        "red_roster_continuity": red_continuity,
        "roster_continuity_diff": blue_continuity - red_continuity,
        "blue_team_roster_experience": blue_team_roster_experience,
        "red_team_roster_experience": red_team_roster_experience,
        "team_roster_experience_diff": (blue_team_roster_experience - red_team_roster_experience),
        "blue_draft_known_fraction": blue_known,
        "red_draft_known_fraction": red_known,
        "blue_ban_strength": blue_ban,
        "red_ban_strength": red_ban,
        "ban_strength_diff": blue_ban - red_ban,
        "blue_patch_champion_strength": blue_patch_champion,
        "red_patch_champion_strength": red_patch_champion,
        "patch_champion_strength_diff": (blue_patch_champion - red_patch_champion),
        "blue_synergy_strength": blue_synergy,
        "red_synergy_strength": red_synergy,
        "synergy_strength_diff": blue_synergy - red_synergy,
        "blue_role_matchup_strength": role_matchup,
        "role_matchup_advantage": role_matchup - 0.5,
        "head_to_head_blue_rate": head_to_head_rate,
        "head_to_head_games": float(head_to_head.games if head_to_head else 0),
        "game_number": float(request.game_number),
        "series_score_diff": float(
            request.blue_series_wins_before - request.red_series_wins_before
        ),
        "series_game_progress": (request.game_number - 1.0) / 4.0,
        "fearless_ban_count": float(len(request.fearless_bans)),
        "patch_major": patch_major,
        "patch_minor": patch_minor,
        "blue_team_inactivity_log_days": math.log1p(blue_team_inactivity_days),
        "red_team_inactivity_log_days": math.log1p(red_team_inactivity_days),
        "team_inactivity_log_days_diff": (
            math.log1p(blue_team_inactivity_days) - math.log1p(red_team_inactivity_days)
        ),
        "blue_adjusted_elo": blue_adjusted_elo,
        "red_adjusted_elo": red_adjusted_elo,
        "adjusted_elo_diff": blue_adjusted_elo - red_adjusted_elo,
        "blue_roster_min_games_log": math.log1p(blue_roster_min_games),
        "red_roster_min_games_log": math.log1p(red_roster_min_games),
        "roster_min_games_log_diff": (
            math.log1p(blue_roster_min_games) - math.log1p(red_roster_min_games)
        ),
        "blue_region_elo": blue_region_elo,
        "red_region_elo": red_region_elo,
        "region_elo_diff": region_elo_diff,
        "region_elo_blue_win_probability": region_elo_probability,
        "blue_region_cross_region_games_log": math.log1p(blue_region_games),
        "red_region_cross_region_games_log": math.log1p(red_region_games),
        "region_cross_region_games_log_diff": (
            math.log1p(blue_region_games) - math.log1p(red_region_games)
        ),
        "blue_home_region_known": float(blue_home_region_known),
        "red_home_region_known": float(red_home_region_known),
        "cross_region_match": float(cross_region_match),
        "pooled_elo_diff": pooled_elo_diff,
        "pooled_elo_blue_win_probability": pooled_elo_probability,
    }
    for role_index, role in enumerate(ROLE_NAMES):
        blue_player = blue_players[role_index]
        red_player = red_players[role_index]
        blue_role_champion = request.blue_picks[role_index]
        red_role_champion = request.red_picks[role_index]
        blue_role_stat = state.role_champion_outcomes.get(
            _role_champion_key(role_index, blue_role_champion)
        )
        red_role_stat = state.role_champion_outcomes.get(
            _role_champion_key(role_index, red_role_champion)
        )
        blue_role_rate = _role_champion_rate(
            state,
            role_index,
            blue_role_champion,
        )
        red_role_rate = _role_champion_rate(
            state,
            role_index,
            red_role_champion,
        )
        blue_player_champion_stat = state.player_champion_outcomes.get(
            blue_player,
            {},
        ).get(blue_role_champion)
        red_player_champion_stat = state.player_champion_outcomes.get(
            red_player,
            {},
        ).get(red_role_champion)
        blue_player_champion_rate = _player_champion_rate(
            state,
            blue_player,
            blue_role_champion,
        )
        red_player_champion_rate = _player_champion_rate(
            state,
            red_player,
            red_role_champion,
        )
        matchup_stat = state.role_matchup_outcomes.get(
            _role_matchup_key(
                role_index,
                blue_role_champion,
                red_role_champion,
            )
        )
        matchup_rate = _smoothed_rate(
            matchup_stat,
            0.5,
            state.settings.matchup_prior_strength,
        )
        blue_role_player_elo = state.player_elo.get(
            blue_player,
            state.settings.elo_initial_rating,
        )
        red_role_player_elo = state.player_elo.get(
            red_player,
            state.settings.elo_initial_rating,
        )
        blue_adjusted_player_elo = _adjusted_rating(
            blue_role_player_elo,
            state.player_last_seen.get(blue_player),
            request.match_timestamp,
            state.settings,
        )
        red_adjusted_player_elo = _adjusted_rating(
            red_role_player_elo,
            state.player_last_seen.get(red_player),
            request.match_timestamp,
            state.settings,
        )
        values.update(
            {
                f"{role}_player_elo_diff": (blue_role_player_elo - red_role_player_elo),
                f"{role}_adjusted_player_elo_diff": (
                    blue_adjusted_player_elo - red_adjusted_player_elo
                ),
                f"blue_{role}_champion_strength": blue_role_rate,
                f"red_{role}_champion_strength": red_role_rate,
                f"{role}_champion_strength_diff": (blue_role_rate - red_role_rate),
                f"blue_{role}_champion_games_log": _stat_games_log(blue_role_stat),
                f"red_{role}_champion_games_log": _stat_games_log(red_role_stat),
                f"{role}_champion_games_log_diff": (
                    _stat_games_log(blue_role_stat) - _stat_games_log(red_role_stat)
                ),
                f"blue_{role}_player_champion_strength": (blue_player_champion_rate),
                f"red_{role}_player_champion_strength": (red_player_champion_rate),
                f"{role}_player_champion_strength_diff": (
                    blue_player_champion_rate - red_player_champion_rate
                ),
                f"blue_{role}_player_champion_games_log": _stat_games_log(
                    blue_player_champion_stat
                ),
                f"red_{role}_player_champion_games_log": _stat_games_log(red_player_champion_stat),
                f"{role}_player_champion_games_log_diff": (
                    _stat_games_log(blue_player_champion_stat)
                    - _stat_games_log(red_player_champion_stat)
                ),
                f"{role}_matchup_strength": matchup_rate,
                f"{role}_matchup_games_log": _stat_games_log(matchup_stat),
            }
        )
    values.update(
        _state_space_values(
            request,
            state,
            blue_team=blue_team,
            red_team=red_team,
            blue_players=blue_players,
            red_players=red_players,
            blue_home_region=blue_home_region,
            red_home_region=red_home_region,
        )
    )
    if set(values) != set(FEATURE_NAMES):
        raise AssertionError("Computed numeric features do not match the declared contract")

    elo_history = _max_timestamp(
        [state.team_last_seen.get(blue_team), state.team_last_seen.get(red_team)]
    )
    form_history = elo_history
    champion_history = _max_timestamp(
        state.champion_last_seen.get(champion)
        for champion in (
            request.blue_picks + request.red_picks + request.blue_bans + request.red_bans
        )
    )
    player_champion_history = _max_timestamp(
        state.player_champion_last_seen.get(player, {}).get(champion)
        for player, champion in zip(
            blue_players + red_players,
            request.blue_picks + request.red_picks,
            strict=True,
        )
    )
    regional_elo_history = _max_timestamp(
        state.region_last_seen.get(region)
        for region in (blue_home_region, red_home_region)
        if region is not None
    )
    feature_history = _max_timestamp(
        [
            state.data_cutoff_timestamp,
            state.league_last_seen.get(request.league),
            elo_history,
            champion_history,
            player_champion_history,
            regional_elo_history,
        ]
    )
    provenance = {
        "feature_history_max_timestamp": feature_history,
        "elo_history_max_timestamp": elo_history,
        "form_history_max_timestamp": form_history,
        "champion_history_max_timestamp": champion_history,
        "player_champion_history_max_timestamp": player_champion_history,
        "regional_elo_history_max_timestamp": regional_elo_history,
    }
    return FeatureComputation(values=values, provenance=provenance)


def _get_stat(values: dict[str, OutcomeStat], key: str) -> OutcomeStat:
    if key not in values:
        values[key] = OutcomeStat()
    return values[key]


def _get_nested_stat(
    values: dict[str, dict[str, OutcomeStat]],
    outer_key: str,
    inner_key: str,
) -> OutcomeStat:
    if outer_key not in values:
        values[outer_key] = {}
    return _get_stat(values[outer_key], inner_key)


def update_state_batch(state: FeatureState, matches: Sequence[HistoricalMatch]) -> None:
    """Apply one timestamp group after every match in it has been featurized."""
    if not matches:
        return
    timestamps = {match.match_timestamp for match in matches}
    if len(timestamps) != 1:
        raise ValueError("A state update batch must contain exactly one timestamp")
    timestamp = next(iter(timestamps))
    if state.data_cutoff_timestamp is not None and timestamp <= state.data_cutoff_timestamp:
        raise TemporalLeakageError("State updates must be strictly chronological")

    identity_keys: dict[str, tuple[str, str, tuple[str, ...], tuple[str, ...]]] = {}
    home_region_updates: defaultdict[str, set[str]] = defaultdict(set)
    elo_deltas: defaultdict[str, float] = defaultdict(float)
    player_elo_deltas: defaultdict[str, float] = defaultdict(float)
    region_elo_deltas: defaultdict[str, float] = defaultdict(float)
    region_game_deltas: defaultdict[str, int] = defaultdict(int)
    state_space_observations: list[GaussianObservation] = []
    for match in matches:
        blue_team = _register_identity(
            kind="team",
            name=match.blue_team,
            explicit_id=match.blue_team_id,
            aliases=state.team_aliases,
            ambiguous_aliases=state.ambiguous_team_aliases,
        )
        red_team = _register_identity(
            kind="team",
            name=match.red_team,
            explicit_id=match.red_team_id,
            aliases=state.team_aliases,
            ambiguous_aliases=state.ambiguous_team_aliases,
        )
        blue_ids = match.blue_player_ids or (None,) * 5
        red_ids = match.red_player_ids or (None,) * 5
        blue_players = tuple(
            _register_identity(
                kind="player",
                name=name,
                explicit_id=player_id,
                aliases=state.player_aliases,
                ambiguous_aliases=state.ambiguous_player_aliases,
            )
            for name, player_id in zip(match.blue_players, blue_ids, strict=True)
        )
        red_players = tuple(
            _register_identity(
                kind="player",
                name=name,
                explicit_id=player_id,
                aliases=state.player_aliases,
                ambiguous_aliases=state.ambiguous_player_aliases,
            )
            for name, player_id in zip(match.red_players, red_ids, strict=True)
        )
        identity_keys[match.match_id] = (
            blue_team,
            red_team,
            blue_players,
            red_players,
        )
        blue_home_region, red_home_region = _request_home_regions(
            match,
            state,
            blue_team,
            red_team,
        )
        event_region, tournament_level, is_official = _request_metadata(match, state)
        if (
            event_region is not None
            and event_region not in state.settings.regional_rating_excluded_regions
        ):
            home_region_updates[blue_team].add(event_region)
            home_region_updates[red_team].add(event_region)

        state_space_settings = state.settings.state_space
        state_space_eligible = (
            state_space_settings.enabled
            and (not state_space_settings.update_requires_official or is_official is True)
            and tournament_level in state_space_settings.update_tournament_levels
        )
        if state_space_eligible:
            state_space_observations.append(
                GaussianObservation(
                    observation_id=match.match_id,
                    design=_state_space_design(
                        match,
                        state,
                        blue_team=blue_team,
                        red_team=red_team,
                        blue_players=blue_players,
                        red_players=red_players,
                        blue_home_region=blue_home_region,
                        red_home_region=red_home_region,
                    ),
                    outcome=match.blue_win,
                )
            )

        regional_rating_eligible = (
            is_official is True
            and tournament_level in state.settings.regional_rating_levels
            and blue_home_region is not None
            and red_home_region is not None
            and blue_home_region != red_home_region
        )
        if regional_rating_eligible:
            assert blue_home_region is not None
            assert red_home_region is not None
            blue_region_rating = state.region_elo.get(
                blue_home_region,
                state.settings.elo_initial_rating,
            )
            red_region_rating = state.region_elo.get(
                red_home_region,
                state.settings.elo_initial_rating,
            )
            expected_blue_region = 1.0 / (
                1.0 + 10.0 ** ((red_region_rating - blue_region_rating) / 400.0)
            )
            region_delta = state.settings.region_elo_k_factor * (
                float(match.blue_win) - expected_blue_region
            )
            region_elo_deltas[blue_home_region] += region_delta
            region_elo_deltas[red_home_region] -= region_delta
            region_game_deltas[blue_home_region] += 1
            region_game_deltas[red_home_region] += 1

        blue_rating = state.team_elo.get(blue_team, state.settings.elo_initial_rating)
        red_rating = state.team_elo.get(red_team, state.settings.elo_initial_rating)
        expected_blue = 1.0 / (1.0 + 10.0 ** ((red_rating - blue_rating) / 400.0))
        delta = state.settings.elo_k_factor * (float(match.blue_win) - expected_blue)
        elo_deltas[blue_team] += delta
        elo_deltas[red_team] -= delta

        blue_player_rating = float(
            np.mean(
                [
                    state.player_elo.get(
                        player,
                        state.settings.elo_initial_rating,
                    )
                    for player in blue_players
                ]
            )
        )
        red_player_rating = float(
            np.mean(
                [
                    state.player_elo.get(
                        player,
                        state.settings.elo_initial_rating,
                    )
                    for player in red_players
                ]
            )
        )
        expected_blue_players = 1.0 / (
            1.0 + 10.0 ** ((red_player_rating - blue_player_rating) / 400.0)
        )
        player_delta = state.settings.player_elo_k_factor * (
            float(match.blue_win) - expected_blue_players
        )
        for player in blue_players:
            player_elo_deltas[player] += player_delta
        for player in red_players:
            player_elo_deltas[player] -= player_delta

    conflicting_home_regions = {
        team: sorted(regions) for team, regions in home_region_updates.items() if len(regions) > 1
    }
    if conflicting_home_regions:
        raise ValueError(
            "A team cannot have multiple domestic home regions at one timestamp: "
            f"{conflicting_home_regions}"
        )

    if state_space_observations:
        update_gaussian_skills(
            state.state_space_skills,
            state_space_observations,
            timestamp,
            state.settings.state_space,
        )

    for team, delta in elo_deltas.items():
        current = state.team_elo.get(team, state.settings.elo_initial_rating)
        state.team_elo[team] = current + delta
    for player, delta in player_elo_deltas.items():
        current = state.player_elo.get(player, state.settings.elo_initial_rating)
        state.player_elo[player] = current + delta
    for region, delta in region_elo_deltas.items():
        current = state.region_elo.get(region, state.settings.elo_initial_rating)
        state.region_elo[region] = current + delta
    for region, games in region_game_deltas.items():
        state.region_cross_region_games[region] = (
            state.region_cross_region_games.get(region, 0) + games
        )
        state.region_last_seen[region] = timestamp

    for match in sorted(matches, key=lambda item: item.match_id):
        blue_team, red_team, blue_players, red_players = identity_keys[match.match_id]
        blue_outcome = _get_stat(state.team_outcomes, blue_team)
        red_outcome = _get_stat(state.team_outcomes, red_team)
        blue_outcome.add(match.blue_win)
        red_outcome.add(not match.blue_win)

        for team, won in (
            (blue_team, match.blue_win),
            (red_team, not match.blue_win),
        ):
            history = state.team_form.setdefault(team, [])
            history.append(int(won))
            del history[: -state.settings.form_window]
            state.team_last_seen[team] = timestamp
            state.known_teams.add(team)

        _get_stat(state.league_blue_outcomes, match.league).add(match.blue_win)
        state.global_blue_outcomes.add(match.blue_win)
        _get_stat(
            state.head_to_head_outcomes,
            _head_to_head_key(blue_team, red_team),
        ).add(match.blue_win)
        _get_stat(
            state.head_to_head_outcomes,
            _head_to_head_key(red_team, blue_team),
        ).add(not match.blue_win)
        if match.first_pick_side is not None:
            first_pick_won = (
                match.blue_win if match.first_pick_side == "blue" else not match.blue_win
            )
            state.first_pick_outcomes.add(first_pick_won)
        state.league_last_seen[match.league] = timestamp
        state.known_leagues.add(match.league)
        state.known_patches.add(match.patch)
        if match.region is not None:
            state.league_regions[match.league] = match.region
        if match.tournament_level is not None:
            state.league_tournament_levels[match.league] = match.tournament_level
        if match.is_official is not None:
            state.league_official_status[match.league] = match.is_official

        for players, champions, won in (
            (blue_players, match.blue_picks, match.blue_win),
            (red_players, match.red_picks, not match.blue_win),
        ):
            for role_index, (player, champion) in enumerate(zip(players, champions, strict=True)):
                _get_stat(state.champion_outcomes, champion).add(won)
                _get_stat(
                    state.role_champion_outcomes,
                    _role_champion_key(role_index, champion),
                ).add(won)
                _get_nested_stat(
                    state.patch_champion_outcomes,
                    match.patch,
                    champion,
                ).add(won)
                _get_nested_stat(state.player_champion_outcomes, player, champion).add(won)
                state.player_games[player] = state.player_games.get(player, 0) + 1
                state.champion_last_seen[champion] = timestamp
                state.player_last_seen[player] = timestamp
                state.player_champion_last_seen.setdefault(player, {})[champion] = timestamp
                state.known_champions.add(champion)
                state.known_players.add(player)

        for team, players, champions, won in (
            (blue_team, blue_players, match.blue_picks, match.blue_win),
            (red_team, red_players, match.red_picks, not match.blue_win),
        ):
            appearances = state.team_player_games.setdefault(team, {})
            for player in players:
                appearances[player] = appearances.get(player, 0) + 1
            state.team_last_roster[team] = players
            for left, right in combinations(champions, 2):
                _get_stat(
                    state.champion_pair_outcomes,
                    _champion_pair_key(left, right),
                ).add(won)

        for role_index, (blue_champion, red_champion) in enumerate(
            zip(match.blue_picks, match.red_picks, strict=True)
        ):
            _get_stat(
                state.role_matchup_outcomes,
                _role_matchup_key(role_index, blue_champion, red_champion),
            ).add(match.blue_win)
            _get_stat(
                state.role_matchup_outcomes,
                _role_matchup_key(role_index, red_champion, blue_champion),
            ).add(not match.blue_win)

        for champion in match.blue_bans + match.red_bans + match.fearless_bans:
            state.known_champions.add(champion)

    for team, regions in home_region_updates.items():
        state.team_home_region[team] = next(iter(regions))

    state.data_cutoff_timestamp = timestamp


def generate_historical_features(
    matches: Sequence[HistoricalMatch],
    settings: FeatureSettings,
) -> tuple[pd.DataFrame, FeatureState]:
    """Replay history and capture every feature before applying its target outcome."""
    ordered = sorted(matches, key=lambda match: (match.match_timestamp, match.match_id))
    state = FeatureState(settings=settings)
    rows: list[dict[str, object]] = []

    for _, grouped_matches in groupby(ordered, key=lambda match: match.match_timestamp):
        timestamp_matches = list(grouped_matches)
        for match in timestamp_matches:
            computation = compute_features(match, state)
            _, _, is_official = _request_metadata(match, state)
            model_context = build_model_context(match, state)
            for source_timestamp in computation.provenance.values():
                if source_timestamp is not None and source_timestamp >= match.match_timestamp:
                    raise TemporalLeakageError(
                        f"Feature source {source_timestamp.isoformat()} is not before "
                        f"{match.match_id} at {match.match_timestamp.isoformat()}"
                    )
            rows.append(
                {
                    "match_id": match.match_id,
                    "match_timestamp": match.match_timestamp,
                    "league": match.league,
                    "tournament": match.tournament,
                    "patch": match.patch,
                    "series_id": match.series_id or match.match_id,
                    "blue_team": match.blue_team,
                    "red_team": match.red_team,
                    "is_official": is_official,
                    "blue_win": int(match.blue_win),
                    **computation.values,
                    **model_context,
                    **computation.provenance,
                }
            )
        update_state_batch(state, timestamp_matches)

    frame = pd.DataFrame(rows)
    validate_feature_frame(frame)
    return frame, state


def build_state_until(
    matches: Sequence[HistoricalMatch],
    settings: FeatureSettings,
    cutoff_timestamp: datetime,
) -> FeatureState:
    state = FeatureState(settings=settings)
    eligible = sorted(
        (match for match in matches if match.match_timestamp <= cutoff_timestamp),
        key=lambda match: (match.match_timestamp, match.match_id),
    )
    for _, grouped_matches in groupby(eligible, key=lambda match: match.match_timestamp):
        update_state_batch(state, list(grouped_matches))
    if state.data_cutoff_timestamp != cutoff_timestamp:
        raise ValueError("Cutoff must be the timestamp of a complete match group")
    return state


def validate_feature_frame(frame: pd.DataFrame) -> None:
    state_space_columns = set(STATE_SPACE_FEATURE_NAMES)
    present_state_space_columns = state_space_columns & set(frame.columns)
    if present_state_space_columns and present_state_space_columns != state_space_columns:
        missing_state_space = sorted(state_space_columns - present_state_space_columns)
        raise ValueError(
            f"Historical feature table has a partial state-space contract: {missing_state_space}"
        )
    numeric_feature_names = FEATURE_NAMES if present_state_space_columns else V4_FEATURE_NAMES
    required = {
        "match_id",
        "match_timestamp",
        "region",
        "tournament_level",
        "is_official",
        "blue_win",
        *numeric_feature_names,
        *MODEL_CONTEXT_NAMES,
        *PROVENANCE_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Historical feature table is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Historical feature table is empty")

    timestamps = pd.to_datetime(frame["match_timestamp"], utc=True)
    for column in PROVENANCE_COLUMNS:
        source_timestamps = pd.to_datetime(frame[column], utc=True)
        invalid = source_timestamps.notna() & (source_timestamps >= timestamps)
        if bool(invalid.any()):
            match_ids = frame.loc[invalid, "match_id"].tolist()
            raise TemporalLeakageError(
                f"{column} is not strictly before target matches: {match_ids}"
            )

    numeric = frame.loc[:, numeric_feature_names].to_numpy(dtype=float)
    if not bool(np.isfinite(numeric).all()):
        raise ValueError("Historical features contain NaN or infinite values")


def prediction_warnings(
    request: DraftRequest,
    state: FeatureState,
    *,
    stale_after_days: int,
) -> list[str]:
    cutoff = state.data_cutoff_timestamp
    if cutoff is None:
        raise ValueError("Feature state has no historical data")
    if request.match_timestamp <= cutoff:
        raise TemporalLeakageError(
            "Prediction timestamp must be strictly later than the artifact data cutoff"
        )

    warnings: list[str] = []
    blue_team, red_team, blue_players, red_players = _request_identity_keys(request, state)
    unknown_teams = sorted(
        {
            name
            for name, key in (
                (request.blue_team, blue_team),
                (request.red_team, red_team),
            )
            if key not in state.known_teams
        }
    )
    unknown_players = sorted(
        {
            name
            for name, key in zip(
                request.blue_players + request.red_players,
                blue_players + red_players,
                strict=True,
            )
            if key not in state.known_players
        }
    )
    unknown_champions = sorted(
        set(
            request.blue_picks
            + request.red_picks
            + request.blue_bans
            + request.red_bans
            + request.fearless_bans
        )
        - state.known_champions
    )
    if unknown_teams:
        warnings.append(f"Unknown teams: {', '.join(unknown_teams)}")
    if unknown_players:
        warnings.append(f"Unknown players: {', '.join(unknown_players)}")
    if unknown_champions:
        warnings.append(f"Unknown champions: {', '.join(unknown_champions)}")
    blue_home_region, red_home_region = _request_home_regions(
        request,
        state,
        blue_team,
        red_team,
    )
    unknown_home_regions = [
        name
        for name, region in (
            (request.blue_team, blue_home_region),
            (request.red_team, red_home_region),
        )
        if region is None
    ]
    if unknown_home_regions:
        warnings.append(
            "Unknown home regions: "
            f"{', '.join(unknown_home_regions)}; using neutral regional priors"
        )
    state_space_settings = state.settings.state_space
    if state_space_settings.enabled:
        state_space_values = _state_space_values(
            request,
            state,
            blue_team=blue_team,
            red_team=red_team,
            blue_players=blue_players,
            red_players=red_players,
            blue_home_region=blue_home_region,
            red_home_region=red_home_region,
        )
        uncertainty = state_space_values["state_space_log_odds_standard_deviation"]
        coverage = state_space_values["state_space_observed_component_fraction"]
        if uncertainty >= state_space_settings.uncertainty_warning_standard_deviation:
            warnings.append(
                "High model uncertainty: latent pregame log-odds standard "
                f"deviation is {uncertainty:.2f}"
            )
        if coverage < state_space_settings.coverage_warning_fraction:
            warnings.append(
                "Sparse state-space history: "
                f"{coverage:.1%} of weighted draft components have prior observations"
            )
    if request.league not in state.known_leagues:
        warnings.append(f"Unknown league: {request.league}")
    if request.patch not in state.known_patches:
        warnings.append(f"Unknown patch: {request.patch}")
    if request.first_pick_side is None:
        warnings.append("Unknown first-pick side: using a neutral prior")
    if request.game_number > 1 and request.series_id is None:
        warnings.append("Missing series ID for a multi-game series context")

    age_days = (request.match_timestamp - cutoff).total_seconds() / 86400.0
    if age_days > stale_after_days:
        warnings.append(f"Stale data: artifact cutoff is {age_days:.1f} days before the match")
    return warnings
