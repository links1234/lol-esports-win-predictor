"""Versioned Oracle's Elixir tabular ingestion with a strict pre-game allowlist."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
from pydantic import ValidationError

from lolpredictor.schemas import DraftPicks, EntityIds, HistoricalMatch, Roster, TeamSide
from lolpredictor.storage import (
    initialize_database,
    insert_matches,
    record_ingestion_run,
)

ORACLES_ELIXIR_SCHEMA_VERSION = "oe-tabular-pregame-v2"
ORACLES_ELIXIR_SOURCE_NAME = "Oracle's Elixir"
ORACLES_ELIXIR_LICENSE_NOTE = (
    "Public Oracle's Elixir download; verify current source terms before redistribution."
)

ORACLES_ELIXIR_PREGAME_COLUMNS = (
    "gameid",
    "league",
    "year",
    "split",
    "playoffs",
    "date",
    "game",
    "patch",
    "participantid",
    "side",
    "position",
    "playername",
    "playerid",
    "teamname",
    "teamid",
    "firstPick",
    "champion",
    "ban1",
    "ban2",
    "ban3",
    "ban4",
    "ban5",
)
ORACLES_ELIXIR_QUALITY_COLUMNS = ("datacompleteness",)
ORACLES_ELIXIR_LABEL_COLUMNS = ("result",)
ORACLES_ELIXIR_READ_COLUMNS = (
    *ORACLES_ELIXIR_PREGAME_COLUMNS,
    *ORACLES_ELIXIR_QUALITY_COLUMNS,
    *ORACLES_ELIXIR_LABEL_COLUMNS,
)

ROLE_ORDER = ("top", "jng", "mid", "bot", "sup")
ROLE_ALIASES = {
    "top": "top",
    "jng": "jng",
    "jungle": "jng",
    "mid": "mid",
    "middle": "mid",
    "bot": "bot",
    "bottom": "bot",
    "adc": "bot",
    "sup": "sup",
    "support": "sup",
}


class OracleElixirRecordError(ValueError):
    """A quarantinable source-game error."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class OracleElixirIssue:
    source_game_id: str
    reason_code: str
    details: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "source_game_id": self.source_game_id,
            "reason_code": self.reason_code,
            "details": self.details,
        }


@dataclass(frozen=True)
class ParsedOracleElixirGame:
    match: HistoricalMatch
    source_game_number: int


@dataclass(frozen=True)
class OracleElixirParseResult:
    matches: list[HistoricalMatch]
    issues: list[OracleElixirIssue]
    source_row_count: int
    source_game_count: int


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: object) -> str:
    return str(value).strip()


def _unique_value(
    frame: pd.DataFrame,
    column: str,
    *,
    required: bool = True,
) -> str | None:
    values = {_clean(value) for value in frame[column].tolist() if _clean(value)}
    if not values:
        if required:
            raise OracleElixirRecordError(
                "missing_required_value",
                f"Missing required value in {column}",
            )
        return None
    if len(values) != 1:
        raise OracleElixirRecordError(
            "inconsistent_source_values",
            f"Expected one value in {column}, found {len(values)}",
        )
    return next(iter(values))


def _parse_binary(value: str, *, column: str) -> bool:
    normalized = value.casefold()
    if normalized in {"1", "1.0", "true"}:
        return True
    if normalized in {"0", "0.0", "false"}:
        return False
    raise OracleElixirRecordError(
        "invalid_binary_value",
        f"{column} must be a binary value",
    )


def _parse_timestamp(value: str) -> datetime:
    try:
        timestamp = pd.Timestamp(value)
    except ValueError as error:
        raise OracleElixirRecordError(
            "invalid_timestamp",
            "date is not a valid timestamp",
        ) from error
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(UTC)
    else:
        timestamp = timestamp.tz_convert(UTC)
    return timestamp.to_pydatetime()


def _normalize_side(frame: pd.DataFrame) -> pd.Series:
    return frame["side"].astype(str).str.strip().str.casefold()


def _normalize_position(frame: pd.DataFrame) -> pd.Series:
    raw = frame["position"].astype(str).str.strip().str.casefold()
    return raw.map(ROLE_ALIASES).fillna(raw)


def _team_rows(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    selected = frame[(_normalize_side(frame) == side) & (_normalize_position(frame) == "team")]
    if len(selected) != 1:
        raise OracleElixirRecordError(
            "invalid_team_rows",
            f"Expected exactly one {side} team row",
        )
    return selected


def _player_rows(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    selected = frame[
        (_normalize_side(frame) == side) & (_normalize_position(frame).isin(ROLE_ORDER))
    ].copy()
    selected["_normalized_position"] = _normalize_position(selected)
    counts = selected["_normalized_position"].value_counts().to_dict()
    if len(selected) != 5 or counts != {role: 1 for role in ROLE_ORDER}:
        raise OracleElixirRecordError(
            "invalid_role_roster",
            f"Expected one {side} player for each canonical role",
        )
    return selected.set_index("_normalized_position").loc[list(ROLE_ORDER)]


def _optional_entity_id(frame: pd.DataFrame, column: str) -> str | None:
    return _unique_value(frame, column, required=False)


def _player_ids(frame: pd.DataFrame) -> EntityIds | None:
    values = tuple(_clean(value) for value in frame["playerid"].tolist())
    if not any(values):
        return None
    if not all(values):
        return None
    return cast(EntityIds, values)


def _bans(frame: pd.DataFrame) -> tuple[str, ...]:
    values: list[str] = []
    for column in ("ban1", "ban2", "ban3", "ban4", "ban5"):
        value = _unique_value(frame, column, required=False)
        if value is not None:
            values.append(value)
    return tuple(values)


def _first_pick_side(
    blue_team: pd.DataFrame,
    red_team: pd.DataFrame,
) -> TeamSide | None:
    blue_value = _unique_value(blue_team, "firstPick", required=False)
    red_value = _unique_value(red_team, "firstPick", required=False)
    if blue_value is None and red_value is None:
        return None
    if blue_value is None or red_value is None:
        raise OracleElixirRecordError(
            "inconsistent_first_pick",
            "firstPick must be present for both sides or neither side",
        )
    blue_first = _parse_binary(blue_value, column="firstPick")
    red_first = _parse_binary(red_value, column="firstPick")
    if blue_first == red_first:
        raise OracleElixirRecordError(
            "inconsistent_first_pick",
            "Exactly one side must have firstPick",
        )
    return "blue" if blue_first else "red"


def _tournament_name(frame: pd.DataFrame) -> str:
    league = _unique_value(frame, "league")
    year = _unique_value(frame, "year")
    assert league is not None
    assert year is not None
    split = _unique_value(frame, "split", required=False)
    playoffs_value = _unique_value(frame, "playoffs", required=False)
    parts: list[str] = [league, year]
    if split is not None:
        parts.append(split)
    if playoffs_value is not None and _parse_binary(playoffs_value, column="playoffs"):
        parts.append("Playoffs")
    return " ".join(parts)


def _parse_source_game(source_game_id: str, frame: pd.DataFrame) -> ParsedOracleElixirGame:
    blue_team_row = _team_rows(frame, "blue")
    red_team_row = _team_rows(frame, "red")
    blue_players = _player_rows(frame, "blue")
    red_players = _player_rows(frame, "red")

    blue_result_value = _unique_value(blue_team_row, "result")
    red_result_value = _unique_value(red_team_row, "result")
    assert blue_result_value is not None
    assert red_result_value is not None
    blue_win = _parse_binary(blue_result_value, column="result")
    red_win = _parse_binary(red_result_value, column="result")
    if blue_win == red_win:
        raise OracleElixirRecordError(
            "inconsistent_result",
            "Blue and red results must be complementary",
        )

    game_value = _unique_value(frame, "game")
    assert game_value is not None
    try:
        source_game_number = int(float(game_value))
    except ValueError as error:
        raise OracleElixirRecordError(
            "invalid_game_number",
            "game must be an integer",
        ) from error
    if source_game_number < 1:
        raise OracleElixirRecordError(
            "invalid_game_number",
            "game must be positive",
        )

    blue_player_names = cast(
        Roster,
        tuple(_clean(value) for value in blue_players["playername"].tolist()),
    )
    red_player_names = cast(
        Roster,
        tuple(_clean(value) for value in red_players["playername"].tolist()),
    )
    blue_champions = cast(
        DraftPicks,
        tuple(_clean(value) for value in blue_players["champion"].tolist()),
    )
    red_champions = cast(
        DraftPicks,
        tuple(_clean(value) for value in red_players["champion"].tolist()),
    )

    try:
        match = HistoricalMatch(
            match_id=source_game_id,
            match_timestamp=_parse_timestamp(str(_unique_value(frame, "date"))),
            league=str(_unique_value(frame, "league")),
            tournament=_tournament_name(frame),
            patch=str(_unique_value(frame, "patch")),
            blue_team=str(_unique_value(blue_team_row, "teamname")),
            red_team=str(_unique_value(red_team_row, "teamname")),
            blue_team_id=_optional_entity_id(blue_team_row, "teamid"),
            red_team_id=_optional_entity_id(red_team_row, "teamid"),
            blue_players=blue_player_names,
            red_players=red_player_names,
            blue_player_ids=_player_ids(blue_players),
            red_player_ids=_player_ids(red_players),
            blue_picks=blue_champions,
            red_picks=red_champions,
            blue_bans=_bans(blue_team_row),
            red_bans=_bans(red_team_row),
            first_pick_side=_first_pick_side(blue_team_row, red_team_row),
            blue_win=blue_win,
        )
    except ValidationError as error:
        raise OracleElixirRecordError(
            "invalid_canonical_draft",
            str(error),
        ) from error
    return ParsedOracleElixirGame(
        match=match,
        source_game_number=source_game_number,
    )


def _team_key(match: HistoricalMatch, side: str) -> str:
    if side == "blue":
        return match.blue_team_id or f"name:{match.blue_team.casefold()}"
    return match.red_team_id or f"name:{match.red_team.casefold()}"


def _series_group_key(parsed: ParsedOracleElixirGame) -> tuple[object, ...]:
    match = parsed.match
    teams = sorted((_team_key(match, "blue"), _team_key(match, "red")))
    return (
        match.match_timestamp.date().isoformat(),
        match.league,
        match.tournament,
        *teams,
    )


def _series_id(
    group_key: tuple[object, ...],
    run_number: int,
) -> str:
    encoded = "|".join(str(value) for value in (*group_key, run_number)).encode()
    return f"oe-series:{hashlib.sha256(encoded).hexdigest()[:20]}"


def _finalize_series_run(
    run: list[ParsedOracleElixirGame],
    *,
    group_key: tuple[object, ...],
    run_number: int,
) -> tuple[list[HistoricalMatch], list[OracleElixirIssue]]:
    expected_numbers = list(range(1, len(run) + 1))
    actual_numbers = [item.source_game_number for item in run]
    if actual_numbers != expected_numbers:
        issues = [
            OracleElixirIssue(
                source_game_id=item.match.match_id,
                reason_code="invalid_series_sequence",
                details={
                    "message": "Source game numbers are not a contiguous series starting at one",
                    "source_game_number": item.source_game_number,
                },
            )
            for item in run
        ]
        return [], issues

    identifier = _series_id(group_key, run_number)
    wins: defaultdict[str, int] = defaultdict(int)
    matches: list[HistoricalMatch] = []
    previous_timestamp: datetime | None = None
    for item in run:
        match = item.match
        if previous_timestamp is not None and match.match_timestamp <= previous_timestamp:
            issue = OracleElixirIssue(
                source_game_id=match.match_id,
                reason_code="invalid_series_timestamps",
                details={"message": "Series timestamps must be strictly increasing"},
            )
            return [], [issue]
        blue_key = _team_key(match, "blue")
        red_key = _team_key(match, "red")
        try:
            finalized = HistoricalMatch.model_validate(
                {
                    **match.model_dump(),
                    "series_id": identifier,
                    "game_number": item.source_game_number,
                    "blue_series_wins_before": wins[blue_key],
                    "red_series_wins_before": wins[red_key],
                }
            )
        except ValidationError as error:
            issue = OracleElixirIssue(
                source_game_id=match.match_id,
                reason_code="invalid_series_context",
                details={"message": str(error)},
            )
            return [], [issue]
        matches.append(finalized)
        winner_key = blue_key if match.blue_win else red_key
        wins[winner_key] += 1
        previous_timestamp = match.match_timestamp
    return matches, []


def _apply_series_context(
    parsed_games: list[ParsedOracleElixirGame],
) -> tuple[list[HistoricalMatch], list[OracleElixirIssue]]:
    grouped: defaultdict[tuple[object, ...], list[ParsedOracleElixirGame]] = defaultdict(list)
    for parsed in parsed_games:
        grouped[_series_group_key(parsed)].append(parsed)

    matches: list[HistoricalMatch] = []
    issues: list[OracleElixirIssue] = []
    for group_key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        ordered = sorted(
            grouped[group_key],
            key=lambda item: (item.match.match_timestamp, item.match.match_id),
        )
        runs: list[list[ParsedOracleElixirGame]] = []
        current: list[ParsedOracleElixirGame] = []
        for item in ordered:
            if item.source_game_number == 1:
                if current:
                    runs.append(current)
                current = [item]
            elif current:
                current.append(item)
            else:
                issues.append(
                    OracleElixirIssue(
                        source_game_id=item.match.match_id,
                        reason_code="series_missing_game_one",
                        details={
                            "message": "A series game appeared before game one",
                            "source_game_number": item.source_game_number,
                        },
                    )
                )
        if current:
            runs.append(current)

        for run_number, run in enumerate(runs, start=1):
            finalized, run_issues = _finalize_series_run(
                run,
                group_key=group_key,
                run_number=run_number,
            )
            matches.extend(finalized)
            issues.extend(run_issues)
    return matches, issues


def _read_source_frame(
    input_path: Path,
    *,
    assume_legacy_blue_first_pick: bool,
) -> pd.DataFrame:
    suffix = input_path.suffix.casefold()
    if suffix == ".csv":
        header = pd.read_csv(input_path, nrows=0).columns.tolist()
    elif suffix == ".xlsx":
        header = pd.read_excel(
            input_path,
            nrows=0,
            engine="openpyxl",
        ).columns.tolist()
    else:
        raise ValueError("Oracle's Elixir source must be a .csv or .xlsx file")

    missing = set(ORACLES_ELIXIR_READ_COLUMNS) - set(header)
    if assume_legacy_blue_first_pick:
        missing.discard("firstPick")
    if missing:
        raise ValueError(f"Oracle's Elixir source is missing required columns: {sorted(missing)}")

    read_columns = [column for column in ORACLES_ELIXIR_READ_COLUMNS if column in header]
    if suffix == ".csv":
        return pd.read_csv(
            input_path,
            usecols=read_columns,
            dtype=str,
            keep_default_na=False,
        )
    return pd.read_excel(
        input_path,
        usecols=read_columns,
        dtype=str,
        keep_default_na=False,
        engine="openpyxl",
    )


def parse_oracles_elixir_source(
    input_path: Path,
    *,
    assume_legacy_blue_first_pick: bool = False,
) -> OracleElixirParseResult:
    """Parse one source file without reading any post-game feature columns."""
    frame = _read_source_frame(
        input_path,
        assume_legacy_blue_first_pick=assume_legacy_blue_first_pick,
    )
    if frame.empty:
        raise ValueError("Oracle's Elixir source is empty")
    if "firstPick" not in frame:
        if not assume_legacy_blue_first_pick:
            raise ValueError("Oracle's Elixir source is missing firstPick")
        try:
            years = {int(float(value)) for value in frame["year"].unique()}
        except ValueError as error:
            raise ValueError(
                "Cannot validate years for the legacy first-pick assumption"
            ) from error
        if not years or max(years) > 2025:
            raise ValueError("The blue-side first-pick assumption is only allowed through 2025")
        frame["firstPick"] = (
            (frame["side"].astype(str).str.strip().str.casefold() == "blue").astype(int).astype(str)
        )

    parsed_games: list[ParsedOracleElixirGame] = []
    issues: list[OracleElixirIssue] = []
    for source_game_id, game_frame in frame.groupby("gameid", sort=False, dropna=False):
        normalized_game_id = _clean(source_game_id)
        if not normalized_game_id:
            normalized_game_id = f"<missing-game-id-row-{int(game_frame.index.min())}>"
        try:
            parsed_games.append(_parse_source_game(normalized_game_id, game_frame))
        except OracleElixirRecordError as error:
            issues.append(
                OracleElixirIssue(
                    source_game_id=normalized_game_id,
                    reason_code=error.reason_code,
                    details={
                        "message": str(error),
                        "source_row_count": len(game_frame),
                    },
                )
            )

    matches, series_issues = _apply_series_context(parsed_games)
    issues.extend(series_issues)
    represented_game_ids = {match.match_id for match in matches} | {
        issue.source_game_id for issue in issues
    }
    source_game_count = int(frame["gameid"].nunique(dropna=False))
    if len(represented_game_ids) != source_game_count:
        raise RuntimeError("Oracle's Elixir parser did not account for every source game")
    return OracleElixirParseResult(
        matches=sorted(matches, key=lambda item: (item.match_timestamp, item.match_id)),
        issues=issues,
        source_row_count=len(frame),
        source_game_count=source_game_count,
    )


def ingest_oracles_elixir_source(
    database_path: Path,
    input_path: Path,
    *,
    source_url: str | None = None,
    retrieved_at: datetime | None = None,
    license_note: str = ORACLES_ELIXIR_LICENSE_NOTE,
    assume_legacy_blue_first_pick: bool = False,
) -> dict[str, Any]:
    if retrieved_at is not None:
        if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
            raise ValueError("retrieved_at must include a timezone")
        retrieved_at = retrieved_at.astimezone(UTC)

    checksum = _file_sha256(input_path)
    parsed = parse_oracles_elixir_source(
        input_path,
        assume_legacy_blue_first_pick=assume_legacy_blue_first_pick,
    )
    initialize_database(database_path)
    source_schema_version = ORACLES_ELIXIR_SCHEMA_VERSION
    if assume_legacy_blue_first_pick:
        source_schema_version += "+legacy-blue-first-pick"
    run_id = f"oe-{checksum[:20]}-{source_schema_version}"
    inserted = insert_matches(
        database_path,
        parsed.matches,
        source=run_id,
    )
    run: dict[str, object] = {
        "run_id": run_id,
        "source_name": ORACLES_ELIXIR_SOURCE_NAME,
        "source_schema_version": source_schema_version,
        "source_filename": input_path.name,
        "source_url": source_url,
        "source_sha256": checksum,
        "retrieved_at": retrieved_at,
        "ingested_at": datetime.now(UTC),
        "license_note": license_note,
        "source_row_count": parsed.source_row_count,
        "source_game_count": parsed.source_game_count,
        "accepted_match_count": inserted,
        "quarantined_game_count": len({issue.source_game_id for issue in parsed.issues}),
    }
    record_ingestion_run(
        database_path,
        run=run,
        issues=(issue.as_dict() for issue in parsed.issues),
    )
    return {
        **{
            key: (value.astimezone(UTC).isoformat() if isinstance(value, datetime) else value)
            for key, value in run.items()
        },
        "database": str(database_path.resolve()),
        "quarantine_reason_counts": {
            reason: sum(issue.reason_code == reason for issue in parsed.issues)
            for reason in sorted({issue.reason_code for issue in parsed.issues})
        },
        "legacy_blue_first_pick_assumption": assume_legacy_blue_first_pick,
    }
