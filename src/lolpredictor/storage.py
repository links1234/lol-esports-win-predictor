"""DuckDB storage for canonical matches and derived historical features."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from lolpredictor.schemas import HistoricalMatch

DATABASE_SCHEMA_VERSION = "3"


def initialize_database(database_path: Path, *, reset: bool = False) -> None:
    """Create an empty canonical database.

    Reset is explicit and is only intended for generated fixture databases.
    """
    database_path = database_path.resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if reset and database_path.exists():
        if not database_path.is_file():
            raise ValueError(f"Refusing to replace non-file database path: {database_path}")
        database_path.unlink()

    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key VARCHAR PRIMARY KEY,
                value VARCHAR NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS matches (
                match_id VARCHAR PRIMARY KEY,
                request_schema_version VARCHAR NOT NULL,
                match_timestamp TIMESTAMPTZ NOT NULL,
                league VARCHAR NOT NULL,
                tournament VARCHAR NOT NULL,
                region VARCHAR,
                tournament_level VARCHAR,
                is_official BOOLEAN,
                patch VARCHAR NOT NULL,
                series_id VARCHAR,
                game_number INTEGER NOT NULL,
                blue_series_wins_before INTEGER NOT NULL,
                red_series_wins_before INTEGER NOT NULL,
                blue_team VARCHAR NOT NULL,
                red_team VARCHAR NOT NULL,
                blue_team_id VARCHAR,
                red_team_id VARCHAR,
                blue_players_json VARCHAR NOT NULL,
                red_players_json VARCHAR NOT NULL,
                blue_player_ids_json VARCHAR,
                red_player_ids_json VARCHAR,
                blue_picks_json VARCHAR NOT NULL,
                red_picks_json VARCHAR NOT NULL,
                blue_bans_json VARCHAR NOT NULL,
                red_bans_json VARCHAR NOT NULL,
                first_pick_side VARCHAR,
                fearless_bans_json VARCHAR NOT NULL,
                blue_win BOOLEAN NOT NULL,
                source VARCHAR NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_runs (
                run_id VARCHAR PRIMARY KEY,
                source_name VARCHAR NOT NULL,
                source_schema_version VARCHAR NOT NULL,
                source_filename VARCHAR NOT NULL,
                source_url VARCHAR,
                source_sha256 VARCHAR NOT NULL,
                retrieved_at TIMESTAMPTZ,
                ingested_at TIMESTAMPTZ NOT NULL,
                license_note VARCHAR NOT NULL,
                source_row_count BIGINT NOT NULL,
                source_game_count BIGINT NOT NULL,
                accepted_match_count BIGINT NOT NULL,
                quarantined_game_count BIGINT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_quarantine (
                run_id VARCHAR NOT NULL,
                source_game_id VARCHAR NOT NULL,
                reason_code VARCHAR NOT NULL,
                details_json VARCHAR NOT NULL,
                PRIMARY KEY (run_id, source_game_id, reason_code)
            )
            """
        )
        migrations = (
            "ADD COLUMN IF NOT EXISTS request_schema_version VARCHAR DEFAULT '2'",
            "ADD COLUMN IF NOT EXISTS series_id VARCHAR",
            "ADD COLUMN IF NOT EXISTS game_number INTEGER DEFAULT 1",
            "ADD COLUMN IF NOT EXISTS blue_series_wins_before INTEGER DEFAULT 0",
            "ADD COLUMN IF NOT EXISTS red_series_wins_before INTEGER DEFAULT 0",
            "ADD COLUMN IF NOT EXISTS blue_team_id VARCHAR",
            "ADD COLUMN IF NOT EXISTS red_team_id VARCHAR",
            "ADD COLUMN IF NOT EXISTS blue_player_ids_json VARCHAR",
            "ADD COLUMN IF NOT EXISTS red_player_ids_json VARCHAR",
            "ADD COLUMN IF NOT EXISTS first_pick_side VARCHAR",
            "ADD COLUMN IF NOT EXISTS fearless_bans_json VARCHAR DEFAULT '[]'",
            "ADD COLUMN IF NOT EXISTS region VARCHAR",
            "ADD COLUMN IF NOT EXISTS tournament_level VARCHAR",
            "ADD COLUMN IF NOT EXISTS is_official BOOLEAN",
        )
        for migration in migrations:
            connection.execute(f"ALTER TABLE matches {migration}")
        set_metadata(connection, "database_schema_version", DATABASE_SCHEMA_VERSION)


def set_metadata(connection: duckdb.DuckDBPyConnection, key: str, value: str) -> None:
    connection.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)", [key, value])


def get_metadata(database_path: Path) -> dict[str, str]:
    with duckdb.connect(str(database_path), read_only=True) as connection:
        return dict(connection.execute("SELECT key, value FROM metadata ORDER BY key").fetchall())


def insert_matches(
    database_path: Path,
    matches: Iterable[HistoricalMatch],
    *,
    source: str,
) -> int:
    rows = [
        (
            match.match_id,
            match.request_schema_version,
            match.match_timestamp,
            match.league,
            match.tournament,
            match.region,
            match.tournament_level,
            match.is_official,
            match.patch,
            match.series_id,
            match.game_number,
            match.blue_series_wins_before,
            match.red_series_wins_before,
            match.blue_team,
            match.red_team,
            match.blue_team_id,
            match.red_team_id,
            json.dumps(match.blue_players),
            json.dumps(match.red_players),
            json.dumps(match.blue_player_ids) if match.blue_player_ids is not None else None,
            json.dumps(match.red_player_ids) if match.red_player_ids is not None else None,
            json.dumps(match.blue_picks),
            json.dumps(match.red_picks),
            json.dumps(match.blue_bans),
            json.dumps(match.red_bans),
            match.first_pick_side,
            json.dumps(match.fearless_bans),
            match.blue_win,
            source,
        )
        for match in matches
    ]
    if not rows:
        return 0

    with duckdb.connect(str(database_path)) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.executemany(
                """
                INSERT INTO matches (
                    match_id,
                    request_schema_version,
                    match_timestamp,
                    league,
                    tournament,
                    region,
                    tournament_level,
                    is_official,
                    patch,
                    series_id,
                    game_number,
                    blue_series_wins_before,
                    red_series_wins_before,
                    blue_team,
                    red_team,
                    blue_team_id,
                    red_team_id,
                    blue_players_json,
                    red_players_json,
                    blue_player_ids_json,
                    red_player_ids_json,
                    blue_picks_json,
                    red_picks_json,
                    blue_bans_json,
                    red_bans_json,
                    first_pick_side,
                    fearless_bans_json,
                    blue_win,
                    source
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?
                )
                """,
                rows,
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    return len(rows)


def record_ingestion_run(
    database_path: Path,
    *,
    run: dict[str, object],
    issues: Iterable[dict[str, object]],
) -> None:
    issue_rows = [
        (
            run["run_id"],
            issue["source_game_id"],
            issue["reason_code"],
            json.dumps(issue["details"], sort_keys=True),
        )
        for issue in issues
    ]
    with duckdb.connect(str(database_path)) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(
                """
                INSERT INTO ingestion_runs VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    run["run_id"],
                    run["source_name"],
                    run["source_schema_version"],
                    run["source_filename"],
                    run["source_url"],
                    run["source_sha256"],
                    run["retrieved_at"],
                    run["ingested_at"],
                    run["license_note"],
                    run["source_row_count"],
                    run["source_game_count"],
                    run["accepted_match_count"],
                    run["quarantined_game_count"],
                ],
            )
            if issue_rows:
                connection.executemany(
                    "INSERT INTO ingestion_quarantine VALUES (?, ?, ?, ?)",
                    issue_rows,
                )
            set_metadata(
                connection,
                f"ingestion_run:{run['run_id']}:sha256",
                str(run["source_sha256"]),
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise


def _normalized_exclusive_cutoff(before_timestamp: datetime | None) -> datetime | None:
    if before_timestamp is None:
        return None
    if before_timestamp.tzinfo is None or before_timestamp.utcoffset() is None:
        raise ValueError("Exclusive storage cutoff must include a timezone")
    return before_timestamp.astimezone(UTC)


def load_matches(
    database_path: Path,
    *,
    before_timestamp: datetime | None = None,
) -> list[HistoricalMatch]:
    """Load ordered matches, optionally enforcing an exclusive timestamp cutoff in SQL."""
    cutoff = _normalized_exclusive_cutoff(before_timestamp)
    where_clause = "WHERE match_timestamp < ?" if cutoff is not None else ""
    parameters = [cutoff] if cutoff is not None else []
    with duckdb.connect(str(database_path), read_only=True) as connection:
        rows = connection.execute(
            f"""
            SELECT
                match_id,
                request_schema_version,
                match_timestamp,
                league,
                tournament,
                region,
                tournament_level,
                is_official,
                patch,
                series_id,
                game_number,
                blue_series_wins_before,
                red_series_wins_before,
                blue_team,
                red_team,
                blue_team_id,
                red_team_id,
                blue_players_json,
                red_players_json,
                blue_player_ids_json,
                red_player_ids_json,
                blue_picks_json,
                red_picks_json,
                blue_bans_json,
                red_bans_json,
                first_pick_side,
                fearless_bans_json,
                blue_win
            FROM matches
            {where_clause}
            ORDER BY match_timestamp, match_id
            """,
            parameters,
        ).fetchall()

    return [
        HistoricalMatch(
            match_id=row[0],
            request_schema_version=row[1],
            match_timestamp=row[2],
            league=row[3],
            tournament=row[4],
            region=row[5],
            tournament_level=row[6],
            is_official=bool(row[7]) if row[7] is not None else None,
            patch=row[8],
            series_id=row[9],
            game_number=row[10],
            blue_series_wins_before=row[11],
            red_series_wins_before=row[12],
            blue_team=row[13],
            red_team=row[14],
            blue_team_id=row[15],
            red_team_id=row[16],
            blue_players=tuple(json.loads(row[17])),
            red_players=tuple(json.loads(row[18])),
            blue_player_ids=tuple(json.loads(row[19])) if row[19] else None,
            red_player_ids=tuple(json.loads(row[20])) if row[20] else None,
            blue_picks=tuple(json.loads(row[21])),
            red_picks=tuple(json.loads(row[22])),
            blue_bans=tuple(json.loads(row[23])),
            red_bans=tuple(json.loads(row[24])),
            first_pick_side=row[25],
            fearless_bans=tuple(json.loads(row[26])),
            blue_win=bool(row[27]),
        )
        for row in rows
    ]


def write_feature_table(
    database_path: Path,
    features: pd.DataFrame,
    *,
    feature_schema_version: str,
) -> None:
    with duckdb.connect(str(database_path)) as connection:
        connection.register("generated_features", features)
        connection.execute(
            "CREATE OR REPLACE TABLE historical_features AS SELECT * FROM generated_features"
        )
        connection.unregister("generated_features")
        set_metadata(connection, "feature_schema_version", feature_schema_version)
        set_metadata(connection, "feature_row_count", str(len(features)))


def load_feature_table(
    database_path: Path,
    *,
    before_timestamp: datetime | None = None,
) -> pd.DataFrame:
    """Load ordered feature rows with an optional cutoff applied inside DuckDB."""
    cutoff = _normalized_exclusive_cutoff(before_timestamp)
    where_clause = "WHERE match_timestamp < ?" if cutoff is not None else ""
    parameters = [cutoff] if cutoff is not None else []
    with duckdb.connect(str(database_path), read_only=True) as connection:
        return connection.execute(
            f"""
            SELECT *
            FROM historical_features
            {where_clause}
            ORDER BY match_timestamp, match_id
            """,
            parameters,
        ).df()


def database_summary(database_path: Path) -> dict[str, object]:
    with duckdb.connect(str(database_path), read_only=True) as connection:
        row = connection.execute(
            "SELECT COUNT(*), MIN(match_timestamp), MAX(match_timestamp) FROM matches"
        ).fetchone()
    if row is None:
        raise ValueError("Database summary query returned no row")
    count, first_timestamp, last_timestamp = row
    return {
        "database": str(database_path.resolve()),
        "match_count": int(count),
        "first_match_timestamp": (
            first_timestamp.astimezone(UTC).isoformat() if first_timestamp else None
        ),
        "last_match_timestamp": (
            last_timestamp.astimezone(UTC).isoformat() if last_timestamp else None
        ),
    }
