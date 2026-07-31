"""Validated screenshot profile, catalog, candidate, and confirmation contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lolpredictor.schemas import (
    PATCH_IDENTIFIER_PATTERN,
    DraftPicks,
    EntityIds,
    Roster,
    TeamSide,
    normalize_patch_identifier,
)


class NormalizedBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=0.0, lt=1.0)
    y: float = Field(ge=0.0, lt=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> NormalizedBox:
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ValueError("normalized crop rectangle extends beyond the image")
        return self


class OverlayProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_schema_version: Literal["1"] = "1"
    profile_id: str = Field(min_length=1)
    expected_aspect_ratio: float = Field(gt=0.0)
    aspect_ratio_tolerance: float = Field(default=0.03, gt=0.0, le=0.25)
    minimum_similarity: float = Field(default=0.96, ge=0.0, le=1.0)
    minimum_margin: float = Field(default=0.04, ge=0.0, le=1.0)
    blue_team: NormalizedBox
    red_team: NormalizedBox
    blue_picks: tuple[
        NormalizedBox,
        NormalizedBox,
        NormalizedBox,
        NormalizedBox,
        NormalizedBox,
    ]
    red_picks: tuple[
        NormalizedBox,
        NormalizedBox,
        NormalizedBox,
        NormalizedBox,
        NormalizedBox,
    ]
    blue_bans: tuple[NormalizedBox, ...] = Field(default_factory=tuple, max_length=5)
    red_bans: tuple[NormalizedBox, ...] = Field(default_factory=tuple, max_length=5)


class TemplateEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["team", "champion"]
    value: str = Field(min_length=1)
    variant_id: str = Field(default="default", min_length=1)
    image: str = Field(min_length=1)


class TemplateCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    catalog_schema_version: Literal["1"] = "1"
    catalog_id: str = Field(min_length=1)
    templates: list[TemplateEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_variants(self) -> TemplateCatalog:
        identities = [
            (entry.kind, entry.value.casefold(), entry.variant_id.casefold())
            for entry in self.templates
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("template variants must be unique within each kind and value")
        if {entry.kind for entry in self.templates} != {"team", "champion"}:
            raise ValueError("template catalog must contain team and champion entries")
        return self


class ParsedField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_name: str
    value: str
    similarity: float = Field(ge=0.0, le=1.0)
    runner_up_value: str | None
    runner_up_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    margin: float = Field(ge=0.0, le=1.0)
    disposition: Literal["accepted", "review"]


class ScreenshotDraftCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parser_version: str
    screenshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_id: str
    catalog_id: str
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    blue_team: ParsedField
    red_team: ParsedField
    blue_picks: tuple[
        ParsedField,
        ParsedField,
        ParsedField,
        ParsedField,
        ParsedField,
    ]
    red_picks: tuple[
        ParsedField,
        ParsedField,
        ParsedField,
        ParsedField,
        ParsedField,
    ]
    blue_bans: tuple[ParsedField, ...]
    red_bans: tuple[ParsedField, ...]
    requires_confirmation: bool
    warnings: list[str]


class ScreenshotPredictionContext(BaseModel):
    """Pregame context not reliably available from champion portrait crops."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    context_schema_version: Literal["1"] = "1"
    match_timestamp: datetime
    league: str = Field(min_length=1)
    tournament: str = Field(min_length=1)
    patch: str = Field(pattern=PATCH_IDENTIFIER_PATTERN)
    series_id: str | None = Field(default=None, min_length=1)
    game_number: int = Field(default=1, ge=1, le=9)
    blue_series_wins_before: int = Field(default=0, ge=0, le=4)
    red_series_wins_before: int = Field(default=0, ge=0, le=4)
    blue_players: Roster
    red_players: Roster
    blue_player_ids: EntityIds | None = None
    red_player_ids: EntityIds | None = None
    blue_team_id: str | None = Field(default=None, min_length=1)
    red_team_id: str | None = Field(default=None, min_length=1)
    first_pick_side: TeamSide | None = None
    fearless_bans: tuple[str, ...] = Field(default_factory=tuple, max_length=40)
    blue_bans: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    red_bans: tuple[str, ...] = Field(default_factory=tuple, max_length=5)

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

    @model_validator(mode="after")
    def validate_series_score(self) -> ScreenshotPredictionContext:
        if self.game_number != (self.blue_series_wins_before + self.red_series_wins_before + 1):
            raise ValueError("game_number must equal prior blue wins plus prior red wins plus one")
        return self


class ScreenshotCorrections(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    correction_schema_version: Literal["1"] = "1"
    blue_team: str | None = Field(default=None, min_length=1)
    red_team: str | None = Field(default=None, min_length=1)
    blue_picks: tuple[str, str, str, str, str] | None = None
    red_picks: tuple[str, str, str, str, str] | None = None
    blue_bans: tuple[str, ...] | None = Field(default=None, max_length=5)
    red_bans: tuple[str, ...] | None = Field(default=None, max_length=5)


class ConfirmedScreenshotDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmation_mode: Literal["automatic", "corrected"]
    screenshot_sha256: str
    parser_version: str
    profile_id: str
    catalog_id: str
    catalog_sha256: str


class OverlayCorpusFrame(BaseModel):
    """One rights-aware, verified frame in a real-overlay benchmark."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    frame_id: str = Field(min_length=1)
    image: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_url: str = Field(min_length=1)
    source_license: str = Field(min_length=1)
    redistribution_rights: Literal["cleared", "not_cleared", "unknown"]
    recording_group: str = Field(min_length=1)
    match_group: str = Field(min_length=1)
    overlay_profile_id: str = Field(min_length=1)
    profile: str = Field(min_length=1)
    catalog: str = Field(min_length=1)
    frame_timestamp_seconds: float = Field(ge=0.0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    partition: Literal["development", "holdout"]
    expected_supported: bool = True
    label_status: Literal["verified"]
    verified_by: str = Field(min_length=1)
    blue_team: str = Field(min_length=1)
    red_team: str = Field(min_length=1)
    blue_picks: DraftPicks
    red_picks: DraftPicks

    @model_validator(mode="after")
    def validate_verified_draft(self) -> OverlayCorpusFrame:
        if self.blue_team == self.red_team:
            raise ValueError("verified blue and red teams must differ")
        if len(set(self.blue_picks + self.red_picks)) != 10:
            raise ValueError("verified champion picks must contain ten unique champions")
        return self


class OverlayCorpusManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    corpus_schema_version: Literal["1"] = "1"
    corpus_id: str = Field(min_length=1)
    frames: list[OverlayCorpusFrame] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_grouped_partitions(self) -> OverlayCorpusManifest:
        frame_ids = [frame.frame_id for frame in self.frames]
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("overlay corpus frame IDs must be unique")
        for field_name in ("recording_group", "match_group"):
            partitions_by_group: dict[str, set[str]] = {}
            for frame in self.frames:
                group = getattr(frame, field_name)
                partitions_by_group.setdefault(group, set()).add(frame.partition)
            crossing = [
                group for group, partitions in partitions_by_group.items() if len(partitions) > 1
            ]
            if crossing:
                raise ValueError(f"{field_name} values may not cross corpus partitions: {crossing}")
        return self
