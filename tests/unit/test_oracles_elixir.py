from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from lolpredictor.oracles_elixir import (
    ORACLES_ELIXIR_LABEL_COLUMNS,
    ORACLES_ELIXIR_PREGAME_COLUMNS,
    ORACLES_ELIXIR_QUALITY_COLUMNS,
    ORACLES_ELIXIR_READ_COLUMNS,
    ingest_oracles_elixir_source,
    parse_oracles_elixir_source,
)
from lolpredictor.storage import load_matches

ROLES = ("top", "jng", "mid", "bot", "sup")


def _source_game_rows(
    *,
    game_id: str,
    timestamp: str,
    game_number: int,
    blue_team: str,
    blue_team_id: str,
    red_team: str,
    red_team_id: str,
    blue_win: bool,
    first_pick_side: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    side_values = (
        (
            "Blue",
            blue_team,
            blue_team_id,
            blue_win,
            first_pick_side == "blue",
            "BlueChampion",
            "BlueBan",
            100,
        ),
        (
            "Red",
            red_team,
            red_team_id,
            not blue_win,
            first_pick_side == "red",
            "RedChampion",
            "RedBan",
            200,
        ),
    )
    for side, team, team_id, won, first_pick, champion_prefix, ban_prefix, team_pid in side_values:
        common = {
            "gameid": game_id,
            "datacompleteness": "complete",
            "league": "LCK",
            "year": "2026",
            "split": "Spring",
            "playoffs": "0",
            "date": timestamp,
            "game": str(game_number),
            "patch": "16.01",
            "side": side,
            "teamname": team,
            "teamid": team_id,
            "firstPick": str(int(first_pick)),
            "result": str(int(won)),
            "goldat15": "999999",
            "kills": "99",
        }
        for role_index, role in enumerate(ROLES, start=1):
            rows.append(
                {
                    **common,
                    "participantid": str(role_index if side == "Blue" else role_index + 5),
                    "position": role,
                    "playername": f"{team}_{role}",
                    "playerid": f"oe:player:{team_id}:{role}",
                    "champion": f"{champion_prefix}{role_index}",
                    **{f"ban{index}": f"{ban_prefix}{index}" for index in range(1, 6)},
                }
            )
        rows.append(
            {
                **common,
                "participantid": str(team_pid),
                "position": "team",
                "playername": "",
                "playerid": "",
                "champion": "",
                **{f"ban{index}": f"{ban_prefix}{index}" for index in range(1, 6)},
            }
        )
    return rows


def _write_source(path: Path, *, corrupt_role: bool = False) -> None:
    rows = [
        *_source_game_rows(
            game_id="game-1",
            timestamp="2026-01-01 12:00:00",
            game_number=1,
            blue_team="Team A",
            blue_team_id="oe:team:a",
            red_team="Team B",
            red_team_id="oe:team:b",
            blue_win=True,
            first_pick_side="red",
        ),
        *_source_game_rows(
            game_id="game-2",
            timestamp="2026-01-01 13:00:00",
            game_number=2,
            blue_team="Team B",
            blue_team_id="oe:team:b",
            red_team="Team A",
            red_team_id="oe:team:a",
            blue_win=True,
            first_pick_side="blue",
        ),
    ]
    if corrupt_role:
        rows[0]["position"] = "coach"
    columns = [*ORACLES_ELIXIR_READ_COLUMNS, "goldat15", "kills"]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def test_adapter_uses_an_explicit_pregame_allowlist() -> None:
    assert set(ORACLES_ELIXIR_READ_COLUMNS) == {
        *ORACLES_ELIXIR_PREGAME_COLUMNS,
        *ORACLES_ELIXIR_QUALITY_COLUMNS,
        *ORACLES_ELIXIR_LABEL_COLUMNS,
    }
    assert "goldat15" not in ORACLES_ELIXIR_READ_COLUMNS
    assert "kills" not in ORACLES_ELIXIR_READ_COLUMNS
    assert "gamelength" not in ORACLES_ELIXIR_READ_COLUMNS


def test_parse_reconstructs_pregame_series_context(tmp_path: Path) -> None:
    source = tmp_path / "oe.csv"
    _write_source(source)
    result = parse_oracles_elixir_source(source)
    assert result.source_row_count == 24
    assert result.source_game_count == 2
    assert not result.issues
    assert len(result.matches) == 2

    first, second = result.matches
    assert first.patch == "16.1"
    assert first.first_pick_side == "red"
    assert first.game_number == 1
    assert second.game_number == 2
    assert second.blue_series_wins_before == 0
    assert second.red_series_wins_before == 1
    assert first.series_id == second.series_id
    assert first.blue_team_id == "oe:team:a"
    assert first.blue_player_ids is not None


def test_postgame_source_mutation_cannot_change_canonical_drafts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "oe.csv"
    _write_source(source)
    original = parse_oracles_elixir_source(source)

    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    frame.loc[:, "goldat15"] = "-123456"
    frame.loc[:, "kills"] = "0"
    mutated = tmp_path / "oe-mutated.csv"
    frame.to_csv(mutated, index=False)
    changed = parse_oracles_elixir_source(mutated)

    assert [match.model_dump() for match in original.matches] == [
        match.model_dump() for match in changed.matches
    ]


def test_legacy_first_pick_assumption_is_explicit_and_year_bounded(
    tmp_path: Path,
) -> None:
    source = tmp_path / "oe.csv"
    _write_source(source)
    frame = pd.read_csv(source, dtype=str, keep_default_na=False).drop(columns=["firstPick"])
    frame.loc[:, "year"] = "2025"
    legacy = tmp_path / "oe-legacy.csv"
    frame.to_csv(legacy, index=False)

    with pytest.raises(ValueError, match="firstPick"):
        parse_oracles_elixir_source(legacy)
    result = parse_oracles_elixir_source(
        legacy,
        assume_legacy_blue_first_pick=True,
    )
    assert result.matches[0].first_pick_side == "blue"

    frame.loc[:, "year"] = "2026"
    future = tmp_path / "oe-future.csv"
    frame.to_csv(future, index=False)
    with pytest.raises(ValueError, match="only allowed through 2025"):
        parse_oracles_elixir_source(
            future,
            assume_legacy_blue_first_pick=True,
        )


def test_invalid_games_are_quarantined_with_reason_codes(tmp_path: Path) -> None:
    source = tmp_path / "oe-invalid.csv"
    _write_source(source, corrupt_role=True)
    result = parse_oracles_elixir_source(source)
    assert any(issue.reason_code == "invalid_role_roster" for issue in result.issues)
    assert not result.matches


def test_ingestion_records_source_provenance(tmp_path: Path) -> None:
    source = tmp_path / "oe.csv"
    _write_source(source)
    database = tmp_path / "matches.duckdb"
    report = ingest_oracles_elixir_source(
        database,
        source,
        source_url="https://example.invalid/public-source.csv",
        retrieved_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert report["accepted_match_count"] == 2
    assert report["quarantined_game_count"] == 0
    assert len(str(report["source_sha256"])) == 64
    assert len(load_matches(database)) == 2

    with duckdb.connect(str(database), read_only=True) as connection:
        run = connection.execute(
            """
            SELECT source_schema_version, source_sha256, accepted_match_count
            FROM ingestion_runs
            """
        ).fetchone()
    assert run is not None
    assert run[0] == "oe-tabular-pregame-v2"
    assert run[1] == report["source_sha256"]
    assert run[2] == 2


def test_xlsx_and_csv_sources_produce_identical_canonical_records(
    tmp_path: Path,
) -> None:
    csv_source = tmp_path / "oe.csv"
    _write_source(csv_source)
    frame = pd.read_csv(csv_source, dtype=str, keep_default_na=False)
    xlsx_source = tmp_path / "oe.xlsx"
    frame.to_excel(xlsx_source, index=False, engine="openpyxl")

    csv_result = parse_oracles_elixir_source(csv_source)
    xlsx_result = parse_oracles_elixir_source(xlsx_source)

    assert xlsx_result.source_row_count == csv_result.source_row_count
    assert [match.model_dump() for match in xlsx_result.matches] == [
        match.model_dump() for match in csv_result.matches
    ]
