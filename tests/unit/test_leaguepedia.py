from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import duckdb
import pytest

import lolpredictor.leaguepedia as leaguepedia
from lolpredictor.leaguepedia import (
    LEAGUEPEDIA_ENDPOINT,
    LEAGUEPEDIA_INGESTION_SCHEMA_VERSION,
    LEAGUEPEDIA_QUERY_FIELDS,
    LEAGUEPEDIA_SNAPSHOT_SCHEMA_VERSION,
    LEAGUEPEDIA_SOURCE_NAME,
    fetch_leaguepedia_snapshot,
    ingest_leaguepedia_snapshot,
    parse_leaguepedia_snapshot,
)
from lolpredictor.storage import load_matches


def _source_row(
    *,
    game_id: str,
    game_number: int,
    timestamp: str,
    blue: str,
    red: str,
    winner: int,
    first_pick: str | None,
) -> dict[str, object]:
    team1 = "Team A"
    team2 = "Team B"
    return {
        "game_id": game_id,
        "match_id": "series-1",
        "overview_page": "LCK/2026 Season/Cup",
        "league": "League of Legends Champions Korea",
        "tournament": "LCK 2026 Cup",
        "region": "Korea",
        "tournament_level": "Primary",
        "is_official": 1,
        "timestamp": timestamp,
        "timestamp__precision": 0,
        "patch": "16.1",
        "team1": team1,
        "team2": team2,
        "team1_picks": ["Aatrox", "Lee Sin", "Ahri", "Ashe", "Braum"],
        "team2_picks": ["Gnar", "Vi", "Orianna", "Jinx", "Nautilus"],
        "team1_bans": ["Renekton", "Xin Zhao", "Syndra", "Caitlyn", "Leona"],
        "team2_bans": ["Kennen", "Wukong", "Azir", "Ezreal", "Rakan"],
        "team1_players": ["A Top", "A Jungle", "A Mid", "A Bot", "A Support"],
        "team2_players": ["B Top", "B Jungle", "B Mid", "B Bot", "B Support"],
        "winner": str(winner),
        "game_number": str(game_number),
        "blue": blue,
        "red": red,
        "first_pick": first_pick,
        "first_selection": first_pick,
        "selection": "Side",
        "pick_selection": "Standard",
    }


def _snapshot(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "snapshot_schema_version": LEAGUEPEDIA_SNAPSHOT_SCHEMA_VERSION,
        "source_name": LEAGUEPEDIA_SOURCE_NAME,
        "source_endpoint": LEAGUEPEDIA_ENDPOINT,
        "retrieved_at": "2026-07-30T20:00:00+00:00",
        "interval": {
            "start_inclusive": "2026-01-01T00:00:00+00:00",
            "end_exclusive": "2026-07-30T19:00:00+00:00",
        },
        "query_fields": LEAGUEPEDIA_QUERY_FIELDS,
        "row_count": len(rows),
        "rows": rows,
    }


def _write_snapshot(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(_snapshot(rows)), encoding="utf-8")


def test_parse_maps_sides_and_reconstructs_only_prior_series_results(
    tmp_path: Path,
) -> None:
    path = tmp_path / "snapshot.json"
    _write_snapshot(
        path,
        [
            _source_row(
                game_id="game-1",
                game_number=1,
                timestamp="2026-01-01 12:00:00",
                blue="Team A",
                red="Team B",
                winner=1,
                first_pick="Team B",
            ),
            _source_row(
                game_id="game-2",
                game_number=2,
                timestamp="2026-01-01 13:00:00",
                blue="Team B",
                red="Team A",
                winner=1,
                first_pick="Team B",
            ),
        ],
    )

    result = parse_leaguepedia_snapshot(path)

    assert not result.issues
    assert result.source_game_count == 2
    first, second = result.matches
    assert first.first_pick_side == "red"
    assert first.blue_win
    assert second.first_pick_side == "blue"
    assert not second.blue_win
    assert second.blue_team == "Team B"
    assert second.blue_players[0] == "B Top"
    assert second.blue_picks[0] == "Gnar"
    assert second.blue_series_wins_before == 0
    assert second.red_series_wins_before == 1
    assert first.blue_team_id == second.red_team_id
    assert first.blue_player_ids is not None
    assert first.blue_player_ids[0] == second.red_player_ids[0]


def test_missing_first_pick_is_preserved_as_unknown(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    _write_snapshot(
        path,
        [
            _source_row(
                game_id="game-1",
                game_number=1,
                timestamp="2026-01-01 12:00:00",
                blue="Team A",
                red="Team B",
                winner=1,
                first_pick=None,
            )
        ],
    )

    result = parse_leaguepedia_snapshot(path)

    assert not result.issues
    assert result.matches[0].first_pick_side is None


def test_source_patch_variants_and_missing_ban_slots_are_normalized(
    tmp_path: Path,
) -> None:
    path = tmp_path / "snapshot.json"
    row = _source_row(
        game_id="game-1",
        game_number=1,
        timestamp="2026-01-01 12:00:00",
        blue="Team A",
        red="Team B",
        winner=1,
        first_pick="Team A",
    )
    row["patch"] = "25.S1.02"
    row["team1_bans"] = ["Renekton", "None", "None", "None", "None"]
    row["team2_bans"] = ["Missing Data", "None", "None", "None", "None"]
    _write_snapshot(path, [row])

    result = parse_leaguepedia_snapshot(path)

    assert not result.issues
    assert result.matches[0].patch == "25.S1.2"
    assert result.matches[0].blue_bans == ("Renekton",)
    assert result.matches[0].red_bans == ()

    row["patch"] = "10.25b"
    _write_snapshot(path, [row])
    assert parse_leaguepedia_snapshot(path).matches[0].patch == "10.25"

    row["patch"] = "v25.10"
    _write_snapshot(path, [row])
    assert parse_leaguepedia_snapshot(path).matches[0].patch == "25.10"


def test_postgame_or_uncontracted_fields_are_quarantined(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    row = _source_row(
        game_id="game-1",
        game_number=1,
        timestamp="2026-01-01 12:00:00",
        blue="Team A",
        red="Team B",
        winner=1,
        first_pick="Team A",
    )
    row["kills"] = 42
    _write_snapshot(path, [row])

    result = parse_leaguepedia_snapshot(path)

    assert not result.matches
    assert [issue.reason_code for issue in result.issues] == ["unexpected_source_field"]


def test_series_level_failure_accounts_for_every_source_game(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    _write_snapshot(
        path,
        [
            _source_row(
                game_id="game-1",
                game_number=1,
                timestamp="2026-01-01 12:00:00",
                blue="Team A",
                red="Team B",
                winner=1,
                first_pick="Team A",
            ),
            _source_row(
                game_id="game-2",
                game_number=2,
                timestamp="2026-01-01 12:00:00",
                blue="Team B",
                red="Team A",
                winner=2,
                first_pick="Team B",
            ),
        ],
    )

    result = parse_leaguepedia_snapshot(path)

    assert not result.matches
    assert {issue.source_game_id for issue in result.issues} == {"game-1", "game-2"}
    assert {issue.reason_code for issue in result.issues} == {"invalid_series_timestamps"}


def test_snapshot_metadata_and_interval_are_enforced(tmp_path: Path) -> None:
    row = _source_row(
        game_id="game-1",
        game_number=1,
        timestamp="2025-12-31 23:59:59",
        blue="Team A",
        red="Team B",
        winner=1,
        first_pick="Team A",
    )
    path = tmp_path / "snapshot.json"
    _write_snapshot(path, [row])

    result = parse_leaguepedia_snapshot(path)
    assert [issue.reason_code for issue in result.issues] == ["timestamp_outside_snapshot_interval"]

    payload = _snapshot([row])
    payload["source_endpoint"] = "https://example.invalid/"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="endpoint"):
        parse_leaguepedia_snapshot(path)


def test_ingestion_records_provenance_and_first_pick_coverage(tmp_path: Path) -> None:
    source = tmp_path / "snapshot.json"
    _write_snapshot(
        source,
        [
            _source_row(
                game_id="game-1",
                game_number=1,
                timestamp="2026-01-01 12:00:00",
                blue="Team A",
                red="Team B",
                winner=1,
                first_pick="Team A",
            )
        ],
    )
    database = tmp_path / "matches.duckdb"

    report = ingest_leaguepedia_snapshot(database, source)

    assert report["accepted_match_count"] == 1
    assert report["accepted_2026_match_count"] == 1
    assert report["first_pick_coverage_2026"] == 1.0
    assert len(load_matches(database)) == 1
    with duckdb.connect(str(database), read_only=True) as connection:
        run = connection.execute(
            """
            SELECT source_schema_version, source_url, accepted_match_count
            FROM ingestion_runs
            """
        ).fetchone()
    assert run == (LEAGUEPEDIA_INGESTION_SCHEMA_VERSION, LEAGUEPEDIA_ENDPOINT, 1)


def test_fetch_paginates_and_refuses_to_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_offsets: list[int] = []

    def fake_fetch(
        url: str,
        *,
        timeout_seconds: float,
        retry_count: int,
    ) -> list[dict[str, object]]:
        del timeout_seconds, retry_count
        query = parse_qs(urlparse(url).query)
        assert "CONCAT('v',SG.Patch)=patch" in query["fields"][0]
        offset = int(query["offset"][0])
        requested_offsets.append(offset)
        count = 100 if offset == 0 else 1
        return [{"game_id": f"game-{offset + index}"} for index in range(count)]

    monkeypatch.setattr(leaguepedia, "_fetch_page", fake_fetch)
    output = tmp_path / "snapshot.json"
    report = fetch_leaguepedia_snapshot(
        output,
        start_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        end_timestamp=datetime(2025, 2, 1, tzinfo=UTC),
        retrieved_at=datetime(2025, 2, 2, tzinfo=UTC),
        page_size=100,
    )

    assert requested_offsets == [0, 100]
    assert report["row_count"] == 101
    assert len(str(report["sha256"])) == 64
    assert not (tmp_path / ".snapshot.json.partial").exists()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        fetch_leaguepedia_snapshot(
            output,
            start_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            end_timestamp=datetime(2025, 2, 1, tzinfo=UTC),
            retrieved_at=datetime(2025, 2, 2, tzinfo=UTC),
            page_size=100,
        )
    assert requested_offsets == [0, 100]


def test_fetch_requires_timezone_aware_boundaries(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="start must include a timezone"):
        fetch_leaguepedia_snapshot(
            tmp_path / "snapshot.json",
            start_timestamp=datetime(2025, 1, 1),
            end_timestamp=datetime(2025, 2, 1, tzinfo=UTC),
            retrieved_at=datetime(2025, 2, 2, tzinfo=UTC),
        )
