"""Template-based screenshot parsing with an explicit confirmation gate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from PIL import Image, ImageOps

from lolpredictor.artifacts import load_artifact
from lolpredictor.prediction import predict_with_artifact
from lolpredictor.schemas import DraftPicks, DraftRequest
from lolpredictor.vision.schemas import (
    ConfirmedScreenshotDraft,
    NormalizedBox,
    OverlayProfile,
    ParsedField,
    ScreenshotCorrections,
    ScreenshotDraftCandidate,
    ScreenshotPredictionContext,
    TemplateCatalog,
)

PARSER_VERSION = "template-parser-v2"
STANDARDIZED_TEMPLATE_SIZE = (32, 32)


class DraftConfirmationRequiredError(ValueError):
    """Raised when screenshot fields require correction before prediction."""

    def __init__(self, fields: list[str]) -> None:
        self.fields = fields
        super().__init__("Screenshot confirmation is required for: " + ", ".join(fields))


@dataclass(frozen=True)
class _Template:
    value: str
    pixels: np.ndarray


@dataclass(frozen=True)
class _LoadedCatalog:
    catalog: TemplateCatalog
    checksum: str
    teams: tuple[_Template, ...]
    champions: tuple[_Template, ...]


def load_overlay_profile(path: Path) -> OverlayProfile:
    return OverlayProfile.model_validate_json(path.read_text(encoding="utf-8"))


def _standardize(image: Image.Image) -> np.ndarray:
    fitted = ImageOps.fit(
        image.convert("RGB"),
        STANDARDIZED_TEMPLATE_SIZE,
        method=Image.Resampling.BILINEAR,
    )
    return np.asarray(fitted, dtype=np.float32) / 255.0


def _catalog_checksum(
    manifest_path: Path,
    entries: list[tuple[str, str, Path]],
) -> str:
    digest = hashlib.sha256()
    digest.update(manifest_path.read_bytes())
    for kind, value, image_path in sorted(entries):
        digest.update(kind.encode())
        digest.update(b"\0")
        digest.update(value.encode())
        digest.update(b"\0")
        digest.update(image_path.read_bytes())
    return digest.hexdigest()


def _load_template_catalog(path: Path) -> _LoadedCatalog:
    catalog = TemplateCatalog.model_validate_json(path.read_text(encoding="utf-8"))
    root = path.resolve().parent
    entries: list[tuple[str, str, Path]] = []
    loaded: dict[str, list[_Template]] = {"team": [], "champion": []}
    for entry in catalog.templates:
        image_path = (root / entry.image).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError(f"Template path escapes the catalog directory: {entry.image}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Template image does not exist: {image_path}")
        with Image.open(image_path) as image:
            pixels = _standardize(image)
        loaded[entry.kind].append(_Template(value=entry.value, pixels=pixels))
        entries.append((entry.kind, entry.value, image_path))
    return _LoadedCatalog(
        catalog=catalog,
        checksum=_catalog_checksum(path, entries),
        teams=tuple(loaded["team"]),
        champions=tuple(loaded["champion"]),
    )


def _pixel_box(box: NormalizedBox, image: Image.Image) -> tuple[int, int, int, int]:
    width, height = image.size
    left = round(box.x * width)
    top = round(box.y * height)
    right = round((box.x + box.width) * width)
    bottom = round((box.y + box.height) * height)
    if right <= left or bottom <= top:
        raise ValueError("Overlay crop resolved to an empty pixel rectangle")
    return left, top, right, bottom


def _match_field(
    image: Image.Image,
    box: NormalizedBox,
    templates: tuple[_Template, ...],
    *,
    field_name: str,
    profile: OverlayProfile,
) -> ParsedField:
    if not templates:
        raise ValueError(f"No templates are available for {field_name}")
    crop = image.crop(_pixel_box(box, image))
    query = _standardize(crop)
    best_by_value: dict[str, tuple[float, str]] = {}
    for template in templates:
        similarity = float(
            np.clip(
                1.0 - np.mean(np.abs(query - template.pixels)),
                0.0,
                1.0,
            )
        )
        identity = template.value.casefold()
        current = best_by_value.get(identity)
        if current is None or similarity > current[0]:
            best_by_value[identity] = (similarity, template.value)
    scores = sorted(best_by_value.values(), reverse=True)
    best_similarity, best_value = scores[0]
    if len(scores) > 1:
        runner_up_similarity, runner_up_value = scores[1]
    else:
        runner_up_similarity, runner_up_value = 0.0, None
    margin = best_similarity - runner_up_similarity
    disposition: Literal["accepted", "review"] = (
        "accepted"
        if best_similarity >= profile.minimum_similarity and margin >= profile.minimum_margin
        else "review"
    )
    return ParsedField(
        field_name=field_name,
        value=best_value,
        similarity=best_similarity,
        runner_up_value=runner_up_value,
        runner_up_similarity=(runner_up_similarity if runner_up_value is not None else None),
        margin=margin,
        disposition=disposition,
    )


def _match_slots(
    image: Image.Image,
    boxes: tuple[NormalizedBox, ...],
    templates: tuple[_Template, ...],
    *,
    prefix: str,
    profile: OverlayProfile,
) -> tuple[ParsedField, ...]:
    return tuple(
        _match_field(
            image,
            box,
            templates,
            field_name=f"{prefix}_{index}",
            profile=profile,
        )
        for index, box in enumerate(boxes, start=1)
    )


def parse_screenshot(
    screenshot_path: Path,
    profile_path: Path,
    catalog_path: Path,
) -> ScreenshotDraftCandidate:
    profile = load_overlay_profile(profile_path)
    catalog = _load_template_catalog(catalog_path)
    screenshot_bytes = screenshot_path.read_bytes()
    screenshot_sha256 = hashlib.sha256(screenshot_bytes).hexdigest()
    with Image.open(screenshot_path) as source:
        image = source.convert("RGB")

    aspect_ratio = image.width / image.height
    relative_error = abs(aspect_ratio - profile.expected_aspect_ratio) / (
        profile.expected_aspect_ratio
    )
    if relative_error > profile.aspect_ratio_tolerance:
        raise ValueError(
            f"Screenshot aspect ratio {aspect_ratio:.4f} is not supported by "
            f"profile {profile.profile_id}"
        )

    blue_team = _match_field(
        image,
        profile.blue_team,
        catalog.teams,
        field_name="blue_team",
        profile=profile,
    )
    red_team = _match_field(
        image,
        profile.red_team,
        catalog.teams,
        field_name="red_team",
        profile=profile,
    )
    blue_picks = _match_slots(
        image,
        profile.blue_picks,
        catalog.champions,
        prefix="blue_pick",
        profile=profile,
    )
    red_picks = _match_slots(
        image,
        profile.red_picks,
        catalog.champions,
        prefix="red_pick",
        profile=profile,
    )
    blue_bans = _match_slots(
        image,
        profile.blue_bans,
        catalog.champions,
        prefix="blue_ban",
        profile=profile,
    )
    red_bans = _match_slots(
        image,
        profile.red_bans,
        catalog.champions,
        prefix="red_ban",
        profile=profile,
    )
    fields = (
        blue_team,
        red_team,
        *blue_picks,
        *red_picks,
        *blue_bans,
        *red_bans,
    )
    warnings = [
        f"Review required for {field.field_name}"
        for field in fields
        if field.disposition == "review"
    ]
    pick_values = [field.value for field in (*blue_picks, *red_picks)]
    if len(set(pick_values)) != len(pick_values):
        warnings.append("Parsed champion picks contain duplicates")
    return ScreenshotDraftCandidate(
        parser_version=PARSER_VERSION,
        screenshot_sha256=screenshot_sha256,
        profile_id=profile.profile_id,
        catalog_id=catalog.catalog.catalog_id,
        catalog_sha256=catalog.checksum,
        blue_team=blue_team,
        red_team=red_team,
        blue_picks=cast(
            tuple[ParsedField, ParsedField, ParsedField, ParsedField, ParsedField], blue_picks
        ),
        red_picks=cast(
            tuple[ParsedField, ParsedField, ParsedField, ParsedField, ParsedField], red_picks
        ),
        blue_bans=blue_bans,
        red_bans=red_bans,
        requires_confirmation=bool(warnings),
        warnings=warnings,
    )


def _reviewed_fields(candidate: ScreenshotDraftCandidate) -> list[str]:
    fields = (
        candidate.blue_team,
        candidate.red_team,
        *candidate.blue_picks,
        *candidate.red_picks,
        *candidate.blue_bans,
        *candidate.red_bans,
    )
    return [field.field_name for field in fields if field.disposition == "review"]


def build_confirmed_draft(
    candidate: ScreenshotDraftCandidate,
    context: ScreenshotPredictionContext,
    corrections: ScreenshotCorrections | None = None,
) -> tuple[DraftRequest, ConfirmedScreenshotDraft]:
    corrections = corrections or ScreenshotCorrections()
    missing_confirmation: list[str] = []

    blue_team = corrections.blue_team or candidate.blue_team.value
    red_team = corrections.red_team or candidate.red_team.value
    if candidate.blue_team.disposition == "review" and corrections.blue_team is None:
        missing_confirmation.append("blue_team")
    if candidate.red_team.disposition == "review" and corrections.red_team is None:
        missing_confirmation.append("red_team")

    parsed_blue_picks = cast(
        DraftPicks,
        tuple(field.value for field in candidate.blue_picks),
    )
    parsed_red_picks = cast(
        DraftPicks,
        tuple(field.value for field in candidate.red_picks),
    )
    blue_picks = corrections.blue_picks or parsed_blue_picks
    red_picks = corrections.red_picks or parsed_red_picks
    if (
        any(field.disposition == "review" for field in candidate.blue_picks)
        and corrections.blue_picks is None
    ):
        missing_confirmation.append("blue_picks")
    if (
        any(field.disposition == "review" for field in candidate.red_picks)
        and corrections.red_picks is None
    ):
        missing_confirmation.append("red_picks")

    parsed_blue_bans = tuple(field.value for field in candidate.blue_bans)
    parsed_red_bans = tuple(field.value for field in candidate.red_bans)
    blue_bans = (
        corrections.blue_bans
        if corrections.blue_bans is not None
        else parsed_blue_bans or context.blue_bans
    )
    red_bans = (
        corrections.red_bans
        if corrections.red_bans is not None
        else parsed_red_bans or context.red_bans
    )
    if (
        any(field.disposition == "review" for field in candidate.blue_bans)
        and corrections.blue_bans is None
        and not context.blue_bans
    ):
        missing_confirmation.append("blue_bans")
    if (
        any(field.disposition == "review" for field in candidate.red_bans)
        and corrections.red_bans is None
        and not context.red_bans
    ):
        missing_confirmation.append("red_bans")

    duplicate_picks = len(set(blue_picks + red_picks)) != 10
    if duplicate_picks and (corrections.blue_picks is None or corrections.red_picks is None):
        missing_confirmation.append("duplicate_picks")
    if missing_confirmation:
        raise DraftConfirmationRequiredError(sorted(set(missing_confirmation)))

    request = DraftRequest(
        match_timestamp=context.match_timestamp,
        league=context.league,
        tournament=context.tournament,
        patch=context.patch,
        series_id=context.series_id,
        game_number=context.game_number,
        blue_series_wins_before=context.blue_series_wins_before,
        red_series_wins_before=context.red_series_wins_before,
        blue_team=blue_team,
        red_team=red_team,
        blue_team_id=context.blue_team_id,
        red_team_id=context.red_team_id,
        blue_players=context.blue_players,
        red_players=context.red_players,
        blue_player_ids=context.blue_player_ids,
        red_player_ids=context.red_player_ids,
        blue_picks=blue_picks,
        red_picks=red_picks,
        blue_bans=blue_bans,
        red_bans=red_bans,
        first_pick_side=context.first_pick_side,
        fearless_bans=context.fearless_bans,
    )
    corrected = corrections != ScreenshotCorrections() or bool(_reviewed_fields(candidate))
    provenance = ConfirmedScreenshotDraft(
        confirmation_mode="corrected" if corrected else "automatic",
        screenshot_sha256=candidate.screenshot_sha256,
        parser_version=candidate.parser_version,
        profile_id=candidate.profile_id,
        catalog_id=candidate.catalog_id,
        catalog_sha256=candidate.catalog_sha256,
    )
    return request, provenance


def predict_screenshot(
    artifact_directory: Path,
    screenshot_path: Path,
    profile_path: Path,
    catalog_path: Path,
    context: ScreenshotPredictionContext,
    corrections: ScreenshotCorrections | None = None,
) -> dict[str, object]:
    candidate = parse_screenshot(screenshot_path, profile_path, catalog_path)
    try:
        request, provenance = build_confirmed_draft(
            candidate,
            context,
            corrections,
        )
    except DraftConfirmationRequiredError as error:
        return {
            "status": "confirmation_required",
            "review_fields": error.fields,
            "parse": candidate.model_dump(mode="json"),
        }

    artifact = load_artifact(artifact_directory)
    prediction = predict_with_artifact(artifact, request)
    return {
        "status": "predicted",
        "parse": candidate.model_dump(mode="json"),
        "confirmation": provenance.model_dump(mode="json"),
        "draft_request": request.model_dump(mode="json"),
        "prediction": prediction.model_dump(mode="json"),
    }


def load_screenshot_context(path: Path) -> ScreenshotPredictionContext:
    return ScreenshotPredictionContext.model_validate_json(path.read_text(encoding="utf-8"))


def load_screenshot_corrections(path: Path) -> ScreenshotCorrections:
    return ScreenshotCorrections.model_validate_json(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
