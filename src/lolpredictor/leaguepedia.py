"""Current professional-draft ingestion from a pinned Leaguepedia Cargo snapshot."""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import ValidationError

from lolpredictor.schemas import DraftPicks, EntityIds, HistoricalMatch, Roster, TeamSide
from lolpredictor.storage import initialize_database, insert_matches, record_ingestion_run

LEAGUEPEDIA_ENDPOINT = "https://lol.fandom.com/wiki/Special:CargoExport"
LEAGUEPEDIA_SOURCE_NAME = "Leaguepedia"
LEAGUEPEDIA_SNAPSHOT_SCHEMA_VERSION = "leaguepedia-cargo-pregame-v2"
LEAGUEPEDIA_INGESTION_SCHEMA_VERSION = "leaguepedia-canonical-drafts-v3"
LEAGUEPEDIA_LICENSE_NOTE = (
    "Leaguepedia content is available under CC BY-SA 3.0 unless otherwise noted; "
    "retain attribution and verify current Leaguepedia and Fandom terms before redistribution."
)
LEAGUEPEDIA_USER_AGENT = (
    "lol-draft-predictor/0.2 (https://github.com/links1234/lol-draft-predictor; pregame research)"
)

LEAGUEPEDIA_QUERY_FIELDS = {
    "game_id": "SG.GameId",
    "match_id": "SG.MatchId",
    "overview_page": "SG.OverviewPage",
    "league": "T.League",
    "tournament": "T.Name",
    "region": "T.Region",
    "tournament_level": "T.TournamentLevel",
    "is_official": "T.IsOfficial",
    "timestamp": "SG.DateTime_UTC",
    # CargoExport serializes numeric-looking strings as JSON numbers, which loses
    # the trailing zero in patches such as 25.10. The sentinel forces a JSON
    # string and is removed by _normalize_source_patch.
    "patch": "CONCAT('v',SG.Patch)",
    "team1": "SG.Team1",
    "team2": "SG.Team2",
    "team1_picks": "SG.Team1Picks",
    "team2_picks": "SG.Team2Picks",
    "team1_bans": "SG.Team1Bans",
    "team2_bans": "SG.Team2Bans",
    "team1_players": "SG.Team1Players",
    "team2_players": "SG.Team2Players",
    "winner": "SG.Winner",
    "game_number": "SG.N_GameInMatch",
    "blue": "MSG.Blue",
    "red": "MSG.Red",
    "first_pick": "MSG.FirstPick",
    "first_selection": "MSG.FirstSelection",
    "selection": "MSG.Selection",
    "pick_selection": "MSG.PickSelection",
}
LEAGUEPEDIA_ROW_FIELDS = frozenset(LEAGUEPEDIA_QUERY_FIELDS)
LEAGUEPEDIA_AUTOMATIC_FIELDS = frozenset({"timestamp__precision"})
LEAGUEPEDIA_MISSING_BAN_VALUES = frozenset({"none", "missing data"})
LEAGUEPEDIA_LEGACY_QUERY_FIELDS = {
    alias: ("SG.Patch" if alias == "patch" else expression)
    for alias, expression in LEAGUEPEDIA_QUERY_FIELDS.items()
    if alias not in {"region", "tournament_level", "is_official"}
}
LEAGUEPEDIA_SUPPORTED_SNAPSHOT_CONTRACTS = {
    "leaguepedia-cargo-pregame-v1": LEAGUEPEDIA_LEGACY_QUERY_FIELDS,
    LEAGUEPEDIA_SNAPSHOT_SCHEMA_VERSION: LEAGUEPEDIA_QUERY_FIELDS,
}


class LeaguepediaRecordError(ValueError):
    """A source-game error that can be quarantined."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class LeaguepediaIssue:
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
class ParsedLeaguepediaGame:
    source_game_id: str
    source_match_id: str
    game_number: int
    timestamp: datetime
    league: str
    tournament: str
    region: str | None
    tournament_level: str | None
    is_official: bool | None
    patch: str
    blue_team: str
    red_team: str
    blue_team_id: str
    red_team_id: str
    blue_players: tuple[str, str, str, str, str]
    red_players: tuple[str, str, str, str, str]
    blue_player_ids: tuple[str, str, str, str, str]
    red_player_ids: tuple[str, str, str, str, str]
    blue_picks: tuple[str, str, str, str, str]
    red_picks: tuple[str, str, str, str, str]
    blue_bans: tuple[str, ...]
    red_bans: tuple[str, ...]
    first_pick_side: TeamSide | None
    blue_win: bool


@dataclass(frozen=True)
class LeaguepediaParseResult:
    matches: list[HistoricalMatch]
    issues: list[LeaguepediaIssue]
    source_row_count: int
    source_game_count: int


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_cargo_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Leaguepedia query timestamps must include a timezone")
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _cargo_url(
    *,
    start_timestamp: datetime,
    end_timestamp: datetime,
    limit: int,
    offset: int,
) -> str:
    start = _format_cargo_timestamp(start_timestamp)
    end = _format_cargo_timestamp(end_timestamp)
    fields = ",".join(
        f"{expression}={alias}" for alias, expression in LEAGUEPEDIA_QUERY_FIELDS.items()
    )
    parameters = {
        "tables": "ScoreboardGames=SG,MatchScheduleGame=MSG,Tournaments=T",
        "join_on": "SG.GameId=MSG.GameId,SG.OverviewPage=T.OverviewPage",
        "fields": fields,
        "where": f"SG.DateTime_UTC >= '{start}' AND SG.DateTime_UTC < '{end}'",
        "order_by": "SG.DateTime_UTC ASC,SG.GameId ASC",
        "limit": str(limit),
        "offset": str(offset),
        "format": "json",
    }
    return f"{LEAGUEPEDIA_ENDPOINT}?{urlencode(parameters)}"


def _fetch_page(
    url: str,
    *,
    timeout_seconds: float,
    retry_count: int,
) -> list[dict[str, object]]:
    last_error: Exception | None = None
    for attempt in range(retry_count + 1):
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": LEAGUEPEDIA_USER_AGENT,
            },
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = json.load(response)
            if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
                raise ValueError("Leaguepedia Cargo export returned an unexpected payload")
            return payload
        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            last_error = error
            if attempt == retry_count:
                break
            time.sleep(min(2**attempt, 8))
    raise RuntimeError("Leaguepedia Cargo export failed after retries") from last_error


def _write_json_atomic(path: Path, payload: object) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Leaguepedia snapshot: {path}")
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite partial snapshot: {temporary}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.rename(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def fetch_leaguepedia_snapshot(
    output_path: Path,
    *,
    start_timestamp: datetime,
    end_timestamp: datetime,
    retrieved_at: datetime | None = None,
    page_size: int = 5000,
    timeout_seconds: float = 90.0,
    retry_count: int = 4,
) -> dict[str, object]:
    """Fetch a reproducible game-level snapshot containing only approved fields."""
    resolved_output = output_path.resolve()
    partial_output = resolved_output.with_name(f".{resolved_output.name}.partial")
    if resolved_output.exists():
        raise FileExistsError(f"Refusing to overwrite Leaguepedia snapshot: {resolved_output}")
    if partial_output.exists():
        raise FileExistsError(f"Refusing to overwrite partial snapshot: {partial_output}")
    if page_size < 100 or page_size > 5000:
        raise ValueError("page_size must be between 100 and 5000")
    if start_timestamp.tzinfo is None or start_timestamp.utcoffset() is None:
        raise ValueError("Leaguepedia snapshot start must include a timezone")
    if end_timestamp.tzinfo is None or end_timestamp.utcoffset() is None:
        raise ValueError("Leaguepedia snapshot end must include a timezone")
    start = start_timestamp.astimezone(UTC)
    end = end_timestamp.astimezone(UTC)
    if start >= end:
        raise ValueError("Leaguepedia snapshot start must be before end")
    retrieval_value = retrieved_at or datetime.now(UTC)
    if retrieval_value.tzinfo is None or retrieval_value.utcoffset() is None:
        raise ValueError("Leaguepedia snapshot retrieval time must include a timezone")
    retrieval = retrieval_value.astimezone(UTC)
    if end > retrieval:
        raise ValueError("Leaguepedia snapshot end cannot be later than retrieval time")

    rows: list[dict[str, object]] = []
    offset = 0
    while True:
        page = _fetch_page(
            _cargo_url(
                start_timestamp=start,
                end_timestamp=end,
                limit=page_size,
                offset=offset,
            ),
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
        )
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += len(page)

    game_ids = [str(row.get("game_id") or "").strip() for row in rows]
    if not rows:
        raise ValueError("Leaguepedia snapshot query returned no games")
    if any(not game_id for game_id in game_ids):
        raise ValueError("Leaguepedia snapshot contains an empty game ID")
    duplicates = [game_id for game_id, count in Counter(game_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Leaguepedia snapshot contains duplicate game IDs: {duplicates[:5]}")

    snapshot = {
        "snapshot_schema_version": LEAGUEPEDIA_SNAPSHOT_SCHEMA_VERSION,
        "source_name": LEAGUEPEDIA_SOURCE_NAME,
        "source_endpoint": LEAGUEPEDIA_ENDPOINT,
        "retrieved_at": retrieval.isoformat(),
        "interval": {
            "start_inclusive": start.isoformat(),
            "end_exclusive": end.isoformat(),
        },
        "query_fields": LEAGUEPEDIA_QUERY_FIELDS,
        "row_count": len(rows),
        "rows": rows,
    }
    _write_json_atomic(output_path, snapshot)
    return {
        "snapshot": str(output_path.resolve()),
        "snapshot_schema_version": LEAGUEPEDIA_SNAPSHOT_SCHEMA_VERSION,
        "source_endpoint": LEAGUEPEDIA_ENDPOINT,
        "retrieved_at": retrieval.isoformat(),
        "start_inclusive": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "row_count": len(rows),
        "sha256": _file_sha256(output_path),
    }


def _clean_text(value: object, *, field: str) -> str:
    if value is None:
        raise LeaguepediaRecordError("missing_required_value", f"{field} is missing")
    cleaned = str(value).strip()
    if not cleaned:
        raise LeaguepediaRecordError("missing_required_value", f"{field} is empty")
    return cleaned


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _optional_bool(value: object, *, field: str) -> bool | None:
    if value is None or str(value).strip() == "":
        return None
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise LeaguepediaRecordError(
        "invalid_boolean_value",
        f"{field} must be a recognized boolean",
    )


def _text_list(
    value: object,
    *,
    field: str,
    expected_length: int | None = None,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise LeaguepediaRecordError("invalid_source_list", f"{field} must be a list")
    result = tuple(_clean_text(item, field=field) for item in value)
    if expected_length is not None and len(result) != expected_length:
        raise LeaguepediaRecordError(
            "invalid_source_list",
            f"{field} must contain exactly {expected_length} values",
        )
    return result


def _ban_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise LeaguepediaRecordError("invalid_source_list", f"{field} must be a list")
    if len(value) > 5:
        raise LeaguepediaRecordError(
            "invalid_source_list",
            f"{field} may contain at most 5 values",
        )
    result: list[str] = []
    for item in value:
        cleaned = _optional_text(item)
        if cleaned is None or _identity_text(cleaned) in LEAGUEPEDIA_MISSING_BAN_VALUES:
            continue
        result.append(cleaned)
    return tuple(result)


def _parse_timestamp(value: object) -> datetime:
    raw = _clean_text(value, field="timestamp")
    try:
        parsed = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
    except ValueError as error:
        raise LeaguepediaRecordError(
            "invalid_timestamp",
            "timestamp must use Leaguepedia's UTC datetime format",
        ) from error
    return parsed.replace(tzinfo=UTC)


def _identity_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _stable_id(kind: str, value: str) -> str:
    digest = hashlib.sha256(_identity_text(value).encode()).hexdigest()[:24]
    return f"leaguepedia:{kind}:{digest}"


def _normalize_source_patch(value: object) -> str:
    patch = _clean_text(value, field="patch")
    if patch.startswith("v"):
        patch = patch.removeprefix("v")
    legacy_suffix = re.fullmatch(r"(\d+)\.(\d+)[a-zA-Z]", patch)
    if legacy_suffix is not None:
        return f"{int(legacy_suffix.group(1))}.{int(legacy_suffix.group(2))}"
    return patch


def _parse_game(row: dict[str, object]) -> ParsedLeaguepediaGame:
    unexpected = set(row) - LEAGUEPEDIA_ROW_FIELDS - LEAGUEPEDIA_AUTOMATIC_FIELDS
    missing = LEAGUEPEDIA_ROW_FIELDS - set(row)
    if unexpected:
        raise LeaguepediaRecordError(
            "unexpected_source_field",
            f"Snapshot row contains fields outside the pre-game allowlist: {sorted(unexpected)}",
        )
    if missing:
        raise LeaguepediaRecordError(
            "missing_source_field",
            f"Snapshot row is missing fields: {sorted(missing)}",
        )

    source_game_id = _clean_text(row["game_id"], field="game_id")
    source_match_id = _clean_text(row["match_id"], field="match_id")
    team1 = _clean_text(row["team1"], field="team1")
    team2 = _clean_text(row["team2"], field="team2")
    blue = _clean_text(row["blue"], field="blue")
    red = _clean_text(row["red"], field="red")
    team1_key = _identity_text(team1)
    team2_key = _identity_text(team2)
    blue_key = _identity_text(blue)
    red_key = _identity_text(red)
    if {team1_key, team2_key} != {blue_key, red_key} or blue_key == red_key:
        raise LeaguepediaRecordError(
            "inconsistent_side_assignment",
            "Blue and red teams must match team1 and team2",
        )
    blue_is_team1 = blue_key == team1_key

    team1_players = _text_list(row["team1_players"], field="team1_players", expected_length=5)
    team2_players = _text_list(row["team2_players"], field="team2_players", expected_length=5)
    team1_picks = _text_list(row["team1_picks"], field="team1_picks", expected_length=5)
    team2_picks = _text_list(row["team2_picks"], field="team2_picks", expected_length=5)
    team1_bans = _ban_list(row["team1_bans"], field="team1_bans")
    team2_bans = _ban_list(row["team2_bans"], field="team2_bans")

    try:
        winner = int(str(row["winner"]))
        game_number = int(str(row["game_number"]))
    except (TypeError, ValueError) as error:
        raise LeaguepediaRecordError(
            "invalid_integer_value",
            "winner and game_number must be integers",
        ) from error
    if winner not in {1, 2}:
        raise LeaguepediaRecordError("invalid_winner", "winner must identify team1 or team2")
    if game_number < 1:
        raise LeaguepediaRecordError(
            "invalid_game_number",
            "game_number must be positive",
        )

    first_pick = _optional_text(row["first_pick"])
    first_pick_side: TeamSide | None = None
    if first_pick is not None:
        first_pick_key = _identity_text(first_pick)
        if first_pick_key == blue_key:
            first_pick_side = "blue"
        elif first_pick_key == red_key:
            first_pick_side = "red"
        else:
            raise LeaguepediaRecordError(
                "inconsistent_first_pick",
                "First-pick team must match blue or red team",
            )

    _clean_text(row["overview_page"], field="overview_page")
    league = _clean_text(row["league"], field="league")
    tournament = _clean_text(row["tournament"], field="tournament")
    region = _optional_text(row["region"])
    tournament_level = _optional_text(row["tournament_level"])
    is_official = _optional_bool(row["is_official"], field="is_official")
    blue_players = cast(Roster, team1_players if blue_is_team1 else team2_players)
    red_players = cast(Roster, team2_players if blue_is_team1 else team1_players)
    blue_picks = cast(DraftPicks, team1_picks if blue_is_team1 else team2_picks)
    red_picks = cast(DraftPicks, team2_picks if blue_is_team1 else team1_picks)
    blue_bans = team1_bans if blue_is_team1 else team2_bans
    red_bans = team2_bans if blue_is_team1 else team1_bans
    team1_won = winner == 1
    blue_win = team1_won if blue_is_team1 else not team1_won

    return ParsedLeaguepediaGame(
        source_game_id=source_game_id,
        source_match_id=source_match_id,
        game_number=game_number,
        timestamp=_parse_timestamp(row["timestamp"]),
        league=league,
        tournament=tournament,
        region=region,
        tournament_level=tournament_level,
        is_official=is_official,
        patch=_normalize_source_patch(row["patch"]),
        blue_team=blue,
        red_team=red,
        blue_team_id=_stable_id("team", blue),
        red_team_id=_stable_id("team", red),
        blue_players=blue_players,
        red_players=red_players,
        blue_player_ids=cast(
            EntityIds,
            tuple(_stable_id("player", player) for player in blue_players),
        ),
        red_player_ids=cast(
            EntityIds,
            tuple(_stable_id("player", player) for player in red_players),
        ),
        blue_picks=blue_picks,
        red_picks=red_picks,
        blue_bans=blue_bans,
        red_bans=red_bans,
        first_pick_side=first_pick_side,
        blue_win=blue_win,
    )


def _finalize_series(
    games: list[ParsedLeaguepediaGame],
) -> tuple[list[HistoricalMatch], list[LeaguepediaIssue]]:
    ordered = sorted(games, key=lambda game: (game.game_number, game.timestamp))
    actual_numbers = [game.game_number for game in ordered]
    expected_numbers = list(range(1, len(ordered) + 1))
    if actual_numbers != expected_numbers:
        return [], [
            LeaguepediaIssue(
                source_game_id=game.source_game_id,
                reason_code="invalid_series_sequence",
                details={
                    "message": "Series games must be contiguous and start at one",
                    "source_game_number": game.game_number,
                },
            )
            for game in ordered
        ]

    series_id = f"leaguepedia:{ordered[0].source_match_id}"
    wins: defaultdict[str, int] = defaultdict(int)
    matches: list[HistoricalMatch] = []
    previous_timestamp: datetime | None = None
    for game in ordered:
        if previous_timestamp is not None and game.timestamp <= previous_timestamp:
            return [], [
                LeaguepediaIssue(
                    source_game_id=series_game.source_game_id,
                    reason_code="invalid_series_timestamps",
                    details={
                        "message": "Series timestamps must be strictly increasing",
                        "invalid_source_game_id": game.source_game_id,
                    },
                )
                for series_game in ordered
            ]
        try:
            match = HistoricalMatch(
                match_id=f"leaguepedia:{game.source_game_id}",
                match_timestamp=game.timestamp,
                league=game.league,
                tournament=game.tournament,
                region=game.region,
                tournament_level=game.tournament_level,
                is_official=game.is_official,
                patch=game.patch,
                series_id=series_id,
                game_number=game.game_number,
                blue_series_wins_before=wins[game.blue_team_id],
                red_series_wins_before=wins[game.red_team_id],
                blue_team=game.blue_team,
                red_team=game.red_team,
                blue_team_id=game.blue_team_id,
                red_team_id=game.red_team_id,
                blue_players=game.blue_players,
                red_players=game.red_players,
                blue_player_ids=game.blue_player_ids,
                red_player_ids=game.red_player_ids,
                blue_picks=game.blue_picks,
                red_picks=game.red_picks,
                blue_bans=game.blue_bans,
                red_bans=game.red_bans,
                first_pick_side=game.first_pick_side,
                fearless_bans=(),
                blue_win=game.blue_win,
            )
        except ValidationError as error:
            return [], [
                LeaguepediaIssue(
                    source_game_id=series_game.source_game_id,
                    reason_code="invalid_canonical_draft",
                    details={
                        "message": str(error),
                        "invalid_source_game_id": game.source_game_id,
                    },
                )
                for series_game in ordered
            ]
        matches.append(match)
        winner_id = game.blue_team_id if game.blue_win else game.red_team_id
        wins[winner_id] += 1
        previous_timestamp = game.timestamp
    return matches, []


def _snapshot_metadata(
    payload: object,
) -> tuple[dict[str, object], list[dict[str, object]], datetime, datetime, datetime]:
    if not isinstance(payload, dict):
        raise ValueError("Leaguepedia snapshot must contain a JSON object")
    typed_payload = cast(dict[str, object], payload)
    snapshot_schema_version = typed_payload.get("snapshot_schema_version")
    expected_query_fields = LEAGUEPEDIA_SUPPORTED_SNAPSHOT_CONTRACTS.get(
        str(snapshot_schema_version)
    )
    if expected_query_fields is None:
        raise ValueError("Unsupported Leaguepedia snapshot schema version")
    if typed_payload.get("source_name") != LEAGUEPEDIA_SOURCE_NAME:
        raise ValueError("Leaguepedia snapshot source name does not match its contract")
    if typed_payload.get("source_endpoint") != LEAGUEPEDIA_ENDPOINT:
        raise ValueError("Leaguepedia snapshot endpoint does not match its contract")
    if typed_payload.get("query_fields") != expected_query_fields:
        raise ValueError("Leaguepedia snapshot query fields do not match the pre-game contract")

    rows_value = typed_payload.get("rows")
    if not isinstance(rows_value, list) or any(not isinstance(row, dict) for row in rows_value):
        raise ValueError("Leaguepedia snapshot rows must be a list of objects")
    rows = cast(list[dict[str, object]], rows_value)
    if typed_payload.get("row_count") != len(rows):
        raise ValueError("Leaguepedia snapshot row count does not match its payload")
    if snapshot_schema_version == "leaguepedia-cargo-pregame-v1":
        rows = [
            {
                **row,
                "region": None,
                "tournament_level": None,
                "is_official": None,
            }
            for row in rows
        ]

    interval = typed_payload.get("interval")
    if not isinstance(interval, dict):
        raise ValueError("Leaguepedia snapshot interval must be an object")
    try:
        start = datetime.fromisoformat(str(interval["start_inclusive"]))
        end = datetime.fromisoformat(str(interval["end_exclusive"]))
        retrieval = datetime.fromisoformat(str(typed_payload["retrieved_at"]))
    except (KeyError, ValueError) as error:
        raise ValueError("Leaguepedia snapshot timestamps are invalid") from error
    for label, value in (("start", start), ("end", end), ("retrieval", retrieval)):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"Leaguepedia snapshot {label} must include a timezone")
    start = start.astimezone(UTC)
    end = end.astimezone(UTC)
    retrieval = retrieval.astimezone(UTC)
    if not start < end <= retrieval:
        raise ValueError("Leaguepedia snapshot interval and retrieval time are inconsistent")
    return typed_payload, rows, start, end, retrieval


def parse_leaguepedia_snapshot(input_path: Path) -> LeaguepediaParseResult:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    _, rows, start, end, _ = _snapshot_metadata(payload)

    parsed_games: list[ParsedLeaguepediaGame] = []
    issues: list[LeaguepediaIssue] = []
    seen_game_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        source_game_id = str(row.get("game_id") or f"<missing-game-id-row-{row_index}>").strip()
        if source_game_id in seen_game_ids:
            issues.append(
                LeaguepediaIssue(
                    source_game_id=source_game_id,
                    reason_code="duplicate_source_game",
                    details={"message": "Snapshot contains a duplicate game ID"},
                )
            )
            continue
        seen_game_ids.add(source_game_id)
        try:
            parsed_game = _parse_game(row)
            if not start <= parsed_game.timestamp < end:
                raise LeaguepediaRecordError(
                    "timestamp_outside_snapshot_interval",
                    "Game timestamp is outside the pinned snapshot interval",
                )
            parsed_games.append(parsed_game)
        except LeaguepediaRecordError as error:
            issues.append(
                LeaguepediaIssue(
                    source_game_id=source_game_id,
                    reason_code=error.reason_code,
                    details={"message": str(error)},
                )
            )

    grouped: defaultdict[str, list[ParsedLeaguepediaGame]] = defaultdict(list)
    for game in parsed_games:
        grouped[game.source_match_id].append(game)
    matches: list[HistoricalMatch] = []
    for source_match_id in sorted(grouped):
        finalized, series_issues = _finalize_series(grouped[source_match_id])
        matches.extend(finalized)
        issues.extend(series_issues)

    represented = {match.match_id.removeprefix("leaguepedia:") for match in matches} | {
        issue.source_game_id for issue in issues
    }
    source_ids = {
        str(row.get("game_id") or f"<missing-game-id-row-{index}>").strip()
        for index, row in enumerate(rows)
    }
    if represented != source_ids:
        raise RuntimeError("Leaguepedia parser did not account for every source game")
    return LeaguepediaParseResult(
        matches=sorted(matches, key=lambda match: (match.match_timestamp, match.match_id)),
        issues=issues,
        source_row_count=len(rows),
        source_game_count=len(source_ids),
    )


def ingest_leaguepedia_snapshot(
    database_path: Path,
    input_path: Path,
    *,
    license_note: str = LEAGUEPEDIA_LICENSE_NOTE,
) -> dict[str, Any]:
    checksum = _file_sha256(input_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    typed_payload, _, _, _, retrieved_at = _snapshot_metadata(payload)
    parsed = parse_leaguepedia_snapshot(input_path)
    initialize_database(database_path)
    run_id = f"leaguepedia-{checksum[:20]}-{LEAGUEPEDIA_INGESTION_SCHEMA_VERSION}"
    inserted = insert_matches(database_path, parsed.matches, source=run_id)
    run: dict[str, object] = {
        "run_id": run_id,
        "source_name": LEAGUEPEDIA_SOURCE_NAME,
        "source_schema_version": LEAGUEPEDIA_INGESTION_SCHEMA_VERSION,
        "source_filename": input_path.name,
        "source_url": str(typed_payload["source_endpoint"]),
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
    accepted_2026 = [match for match in parsed.matches if match.match_timestamp.year == 2026]
    known_first_pick_2026 = sum(match.first_pick_side is not None for match in accepted_2026)
    league_counts = Counter(match.league for match in parsed.matches)
    return {
        **{
            key: (value.isoformat() if isinstance(value, datetime) else value)
            for key, value in run.items()
        },
        "database": str(database_path.resolve()),
        "quarantine_reason_counts": {
            reason: sum(issue.reason_code == reason for issue in parsed.issues)
            for reason in sorted({issue.reason_code for issue in parsed.issues})
        },
        "league_counts": dict(sorted(league_counts.items())),
        "first_pick_coverage_2026": (
            known_first_pick_2026 / len(accepted_2026) if accepted_2026 else None
        ),
        "accepted_2026_match_count": len(accepted_2026),
    }
