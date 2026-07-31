"""Canonical validated data and prediction contracts."""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Roster = tuple[str, str, str, str, str]
DraftPicks = tuple[str, str, str, str, str]
EntityIds = tuple[str, str, str, str, str]
TeamSide = Literal["blue", "red"]
PATCH_IDENTIFIER_PATTERN = r"^\d+\.(?:\d+|S\d+\.\d+)$"


def normalize_patch_identifier(value: str) -> str:
    parts = value.split(".")
    major = str(int(parts[0]))
    if len(parts) == 2:
        return f"{major}.{int(parts[1])}"
    season = parts[1]
    return f"{major}.S{int(season.removeprefix('S'))}.{int(parts[2])}"


class DraftRequest(BaseModel):
    """Draft information known before a match begins."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    request_schema_version: Literal["2"] = "2"
    match_timestamp: datetime
    league: str = Field(min_length=1)
    tournament: str = Field(min_length=1)
    region: str | None = Field(default=None, min_length=1)
    tournament_level: str | None = Field(default=None, min_length=1)
    is_official: bool | None = None
    patch: str = Field(pattern=PATCH_IDENTIFIER_PATTERN)
    series_id: str | None = Field(default=None, min_length=1)
    game_number: int = Field(default=1, ge=1, le=9)
    blue_series_wins_before: int = Field(default=0, ge=0, le=4)
    red_series_wins_before: int = Field(default=0, ge=0, le=4)
    blue_team: str = Field(min_length=1)
    red_team: str = Field(min_length=1)
    blue_team_id: str | None = Field(default=None, min_length=1)
    red_team_id: str | None = Field(default=None, min_length=1)
    blue_players: Roster
    red_players: Roster
    blue_player_ids: EntityIds | None = None
    red_player_ids: EntityIds | None = None
    blue_picks: DraftPicks
    red_picks: DraftPicks
    blue_bans: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    red_bans: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    first_pick_side: TeamSide | None = None
    fearless_bans: tuple[str, ...] = Field(default_factory=tuple, max_length=40)

    @field_validator("match_timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("match_timestamp must include a timezone")
        return value.astimezone(UTC)

    @field_validator("patch")
    @classmethod
    def normalize_patch(cls, value: str) -> str:
        return normalize_patch_identifier(value)

    @field_validator(
        "blue_players",
        "red_players",
        "blue_picks",
        "red_picks",
        "blue_bans",
        "red_bans",
        "fearless_bans",
    )
    @classmethod
    def validate_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("draft lists may not contain empty values")
        return tuple(item.strip() for item in value)

    @field_validator("blue_player_ids", "red_player_ids")
    @classmethod
    def validate_optional_ids(cls, value: EntityIds | None) -> EntityIds | None:
        if value is not None and any(not item.strip() for item in value):
            raise ValueError("player ID lists may not contain empty values")
        return value

    @model_validator(mode="after")
    def validate_draft(self) -> "DraftRequest":
        if self.blue_team == self.red_team:
            raise ValueError("blue_team and red_team must differ")
        if (
            self.blue_team_id is not None
            and self.red_team_id is not None
            and self.blue_team_id == self.red_team_id
        ):
            raise ValueError("blue_team_id and red_team_id must differ")
        expected_game_number = self.blue_series_wins_before + self.red_series_wins_before + 1
        if self.game_number != expected_game_number:
            raise ValueError("game_number must equal prior blue wins plus prior red wins plus one")
        all_picks = self.blue_picks + self.red_picks
        if len(set(all_picks)) != len(all_picks):
            raise ValueError("a champion may be picked only once in a draft")
        standard_bans = self.blue_bans + self.red_bans
        if len(set(standard_bans)) != len(standard_bans):
            raise ValueError("a champion may be standard-banned only once in a draft")
        if len(set(self.fearless_bans)) != len(self.fearless_bans):
            raise ValueError("fearless_bans may not contain duplicate champions")
        unavailable = set(standard_bans) | set(self.fearless_bans)
        if set(all_picks) & unavailable:
            raise ValueError("a banned champion may not also be picked")
        if set(standard_bans) & set(self.fearless_bans):
            raise ValueError("a standard ban may not repeat a Fearless Draft exclusion")
        return self


class HistoricalMatch(DraftRequest):
    """A completed match used for historical state replay."""

    match_id: str = Field(min_length=1)
    blue_win: bool


class PredictionResponse(BaseModel):
    """Stable prediction response."""

    model_config = ConfigDict(extra="forbid")

    estimate_type: Literal["post_draft_pregame"] = "post_draft_pregame"
    blue_win_probability: float = Field(ge=0.0, le=1.0)
    red_win_probability: float = Field(ge=0.0, le=1.0)
    model_version: str
    data_cutoff_timestamp: datetime
    warnings: list[str]

    @model_validator(mode="after")
    def validate_probability_sum(self) -> "PredictionResponse":
        if abs(self.blue_win_probability + self.red_win_probability - 1.0) > 1e-12:
            raise ValueError("blue and red probabilities must sum to one")
        return self


def parse_historical_match(value: dict[str, Any]) -> HistoricalMatch:
    return HistoricalMatch.model_validate(value)


def parse_draft_request(value: dict[str, Any]) -> DraftRequest:
    return DraftRequest.model_validate(value)
