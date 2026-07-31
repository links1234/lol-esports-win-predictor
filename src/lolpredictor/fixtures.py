"""Deterministic synthetic data that exercises the complete pipeline."""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from lolpredictor.schemas import DraftPicks, HistoricalMatch, Roster
from lolpredictor.storage import (
    database_summary,
    initialize_database,
    insert_matches,
    set_metadata,
)

FIXTURE_SEED = 20260730
ROLES = ("top", "jungle", "mid", "bottom", "support")
TEAMS = (
    "SYN_ALPHA",
    "SYN_BRAVO",
    "SYN_COMET",
    "SYN_DELTA",
    "SYN_ECHO",
    "SYN_FLARE",
    "SYN_GAMMA",
    "SYN_HORIZON",
)
TEAM_STRENGTH = {
    "SYN_ALPHA": 0.90,
    "SYN_BRAVO": 0.58,
    "SYN_COMET": 0.34,
    "SYN_DELTA": 0.08,
    "SYN_ECHO": -0.10,
    "SYN_FLARE": -0.32,
    "SYN_GAMMA": -0.56,
    "SYN_HORIZON": -0.82,
}
ROLE_CHAMPIONS = (
    ("Aegis", "Atlas", "Fable", "Kite", "Prism", "Vale"),
    ("Bramble", "Brook", "Gale", "Lumen", "Quill", "Wisp"),
    ("Cipher", "Cinder", "Haven", "Morrow", "Rune", "Xenon"),
    ("Dawn", "Drift", "Ion", "Nova", "Spark", "Yarrow"),
    ("Ember", "Echo", "Jade", "Onyx", "Thorn", "Zephyr"),
)
ALL_CHAMPIONS = tuple(champion for pool in ROLE_CHAMPIONS for champion in pool)
CHAMPION_STRENGTH = {
    champion: ((index % 6) - 2.5) * 0.08 for index, champion in enumerate(ALL_CHAMPIONS)
}


def _round_pairings(round_number: int) -> list[tuple[int, int]]:
    rotating = list(range(1, len(TEAMS)))
    shift = round_number % len(rotating)
    rotating = rotating[shift:] + rotating[:shift]
    arrangement = [0, *rotating]
    return [(arrangement[index], arrangement[-index - 1]) for index in range(len(arrangement) // 2)]


def _draft(team_index: int, round_number: int, offset: int) -> DraftPicks:
    picks: list[str] = []
    for role_index, pool in enumerate(ROLE_CHAMPIONS):
        pick_index = (team_index + round_number * (offset + 1) + role_index * (offset + 2)) % len(
            pool
        )
        picks.append(pool[pick_index])
    return cast(DraftPicks, tuple(picks))


def _bans(
    round_number: int,
    pairing_index: int,
    excluded: set[str],
    offset: int,
) -> tuple[str, ...]:
    available = [champion for champion in ALL_CHAMPIONS if champion not in excluded]
    start = (round_number * 3 + pairing_index * 5 + offset) % len(available)
    return tuple(available[(start + index * 4) % len(available)] for index in range(5))


def _roster(team: str, round_number: int) -> Roster:
    roster = [f"{team}_{role}" for role in ROLES]
    if team == "SYN_HORIZON" and round_number >= 22:
        roster[-1] = f"{team}_support_sub"
    return cast(Roster, tuple(roster))


def generate_synthetic_matches(
    *,
    seed: int = FIXTURE_SEED,
    rounds: int = 30,
) -> list[HistoricalMatch]:
    """Generate 120 matches in 30 simultaneous four-match timestamp groups."""
    random_generator = random.Random(seed)
    first_timestamp = datetime(2024, 1, 1, 12, tzinfo=UTC)
    matches: list[HistoricalMatch] = []

    for round_number in range(rounds):
        timestamp = first_timestamp + timedelta(days=3 * round_number)
        patch = f"14.{round_number // 5 + 1}"
        for pairing_index, (left_index, right_index) in enumerate(_round_pairings(round_number)):
            if (round_number + pairing_index) % 2:
                blue_index, red_index = right_index, left_index
            else:
                blue_index, red_index = left_index, right_index

            blue_team = TEAMS[blue_index]
            red_team = TEAMS[red_index]
            blue_picks = _draft(blue_index, round_number, 0)
            red_picks_list = list(_draft(red_index, round_number, 1))
            for role_index, champion in enumerate(red_picks_list):
                if champion == blue_picks[role_index]:
                    pool = ROLE_CHAMPIONS[role_index]
                    red_picks_list[role_index] = pool[(pool.index(champion) + 1) % len(pool)]
            red_picks = cast(DraftPicks, tuple(red_picks_list))

            excluded = set(blue_picks) | set(red_picks)
            blue_bans = _bans(round_number, pairing_index, excluded, 1)
            red_bans = _bans(
                round_number,
                pairing_index,
                excluded | set(blue_bans),
                9,
            )

            same_region = (blue_index < 4) == (red_index < 4)
            if same_region:
                league = "LCK" if blue_index < 4 else "LEC"
                tournament = f"Synthetic {league} Spring"
                region = "Korea" if blue_index < 4 else "Europe"
            else:
                league = "INTL"
                tournament = "Synthetic Invitational"
                region = "International"

            champion_delta = sum(CHAMPION_STRENGTH[pick] for pick in blue_picks) / 5
            champion_delta -= sum(CHAMPION_STRENGTH[pick] for pick in red_picks) / 5
            latent = (
                0.18
                + TEAM_STRENGTH[blue_team]
                - TEAM_STRENGTH[red_team]
                + 0.8 * champion_delta
                + 0.05 * math.sin(round_number + pairing_index)
            )
            blue_probability = 1.0 / (1.0 + math.exp(-latent))
            blue_win = random_generator.random() < blue_probability

            matches.append(
                HistoricalMatch(
                    match_id=f"synthetic-{round_number:02d}-{pairing_index}",
                    match_timestamp=timestamp,
                    league=league,
                    tournament=tournament,
                    region=region,
                    tournament_level="Primary",
                    is_official=True,
                    patch=patch,
                    blue_team=blue_team,
                    red_team=red_team,
                    blue_players=_roster(blue_team, round_number),
                    red_players=_roster(red_team, round_number),
                    blue_picks=blue_picks,
                    red_picks=red_picks,
                    blue_bans=blue_bans,
                    red_bans=red_bans,
                    first_pick_side="blue",
                    blue_win=blue_win,
                )
            )
    return matches


def build_fixture_database(database_path: Path, *, force: bool = False) -> dict[str, object]:
    matches = generate_synthetic_matches()
    initialize_database(database_path, reset=force)
    inserted = insert_matches(database_path, matches, source="synthetic-fixture-v1")

    import duckdb

    with duckdb.connect(str(database_path)) as connection:
        set_metadata(connection, "fixture_name", "synthetic-professional-drafts-v1")
        set_metadata(connection, "fixture_seed", str(FIXTURE_SEED))
        set_metadata(connection, "fixture_match_count", str(inserted))
    return database_summary(database_path)
