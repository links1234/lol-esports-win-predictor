"""Deterministic image fixture for the screenshot-to-prediction workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from lolpredictor.prediction import load_draft_request
from lolpredictor.vision.schemas import (
    NormalizedBox,
    OverlayCorpusFrame,
    OverlayCorpusManifest,
    OverlayProfile,
    ScreenshotPredictionContext,
    TemplateCatalog,
    TemplateEntry,
)

FIXTURE_WIDTH = 1280
FIXTURE_HEIGHT = 720
TILE_SIZE = 64


def _pattern(value: str, kind: str) -> Image.Image:
    digest = hashlib.sha256(f"{kind}:{value}".encode()).digest()
    background = tuple(40 + byte % 176 for byte in digest[:3])
    foreground = tuple(255 - channel for channel in background)
    image = Image.new("RGB", (TILE_SIZE, TILE_SIZE), background)
    draw = ImageDraw.Draw(image)
    for index in range(4):
        inset = 4 + index * 7
        width = 2 + digest[3 + index] % 5
        draw.rectangle(
            (inset, inset, TILE_SIZE - inset - 1, TILE_SIZE - inset - 1),
            outline=foreground,
            width=width,
        )
    if digest[8] % 2:
        draw.line((0, 0, TILE_SIZE - 1, TILE_SIZE - 1), fill=foreground, width=5)
    else:
        draw.line((0, TILE_SIZE - 1, TILE_SIZE - 1, 0), fill=foreground, width=5)
    return image


def _box(x: int, y: int) -> NormalizedBox:
    return NormalizedBox(
        x=x / FIXTURE_WIDTH,
        y=y / FIXTURE_HEIGHT,
        width=TILE_SIZE / FIXTURE_WIDTH,
        height=TILE_SIZE / FIXTURE_HEIGHT,
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_screenshot_fixture(
    directory: Path,
    sample_draft_path: Path,
) -> dict[str, str]:
    """Build templates, profile, context, and one supported synthetic screenshot."""
    directory = directory.resolve()
    profile_path = directory / "profile.json"
    catalog_path = directory / "catalog.json"
    context_path = directory / "context.json"
    manifest_path = directory / "corpus.json"
    screenshot_path = directory / "draft.png"
    for path in (
        profile_path,
        catalog_path,
        context_path,
        manifest_path,
        screenshot_path,
    ):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite screenshot fixture file: {path}")

    directory.mkdir(parents=True, exist_ok=True)
    templates_directory = directory / "templates"
    templates_directory.mkdir()
    request = load_draft_request(sample_draft_path)

    team_values = (request.blue_team, request.red_team)
    champion_values = tuple(
        dict.fromkeys(request.blue_picks + request.red_picks + request.blue_bans + request.red_bans)
    )
    entries: list[TemplateEntry] = []
    images: dict[tuple[str, str], Image.Image] = {}
    for kind, values in (("team", team_values), ("champion", champion_values)):
        for index, value in enumerate(values):
            image = _pattern(value, kind)
            relative_path = Path("templates") / f"{kind}-{index:02d}.png"
            image.save(directory / relative_path)
            entries.append(
                TemplateEntry(
                    kind=kind,  # type: ignore[arg-type]
                    value=value,
                    image=relative_path.as_posix(),
                )
            )
            images[(kind, value)] = image

    blue_team_box = _box(32, 24)
    red_team_box = _box(1184, 24)
    blue_pick_boxes = tuple(_box(32, 112 + index * 100) for index in range(5))
    red_pick_boxes = tuple(_box(1184, 112 + index * 100) for index in range(5))
    blue_ban_boxes = tuple(_box(260 + index * 76, 640) for index in range(5))
    red_ban_boxes = tuple(_box(640 + index * 76, 640) for index in range(5))
    profile = OverlayProfile(
        profile_id="synthetic-overlay-v1",
        expected_aspect_ratio=FIXTURE_WIDTH / FIXTURE_HEIGHT,
        minimum_similarity=0.985,
        minimum_margin=0.04,
        blue_team=blue_team_box,
        red_team=red_team_box,
        blue_picks=blue_pick_boxes,  # type: ignore[arg-type]
        red_picks=red_pick_boxes,  # type: ignore[arg-type]
        blue_bans=blue_ban_boxes,
        red_bans=red_ban_boxes,
    )
    catalog = TemplateCatalog(
        catalog_id="synthetic-catalog-v1",
        templates=entries,
    )

    screenshot = Image.new("RGB", (FIXTURE_WIDTH, FIXTURE_HEIGHT), (12, 18, 28))
    draw = ImageDraw.Draw(screenshot)
    for stripe in range(12):
        color = (20 + stripe * 5, 25 + stripe * 3, 45 + stripe * 4)
        draw.rectangle(
            (160, stripe * 60, 1120, stripe * 60 + 30),
            fill=color,
        )
    placements = (
        (blue_team_box, "team", request.blue_team),
        (red_team_box, "team", request.red_team),
        *(
            (box, "champion", value)
            for box, value in zip(blue_pick_boxes, request.blue_picks, strict=True)
        ),
        *(
            (box, "champion", value)
            for box, value in zip(red_pick_boxes, request.red_picks, strict=True)
        ),
        *(
            (box, "champion", value)
            for box, value in zip(blue_ban_boxes, request.blue_bans, strict=True)
        ),
        *(
            (box, "champion", value)
            for box, value in zip(red_ban_boxes, request.red_bans, strict=True)
        ),
    )
    for box, kind, value in placements:
        x = round(box.x * FIXTURE_WIDTH)
        y = round(box.y * FIXTURE_HEIGHT)
        screenshot.paste(images[(kind, value)], (x, y))
    screenshot.save(screenshot_path)
    manifest = OverlayCorpusManifest(
        corpus_id="synthetic-overlay-corpus-v1",
        frames=[
            OverlayCorpusFrame(
                frame_id="synthetic-supported-frame-1",
                image=screenshot_path.name,
                image_sha256=hashlib.sha256(screenshot_path.read_bytes()).hexdigest(),
                source_url="https://example.invalid/synthetic-overlay-fixture",
                source_license="generated test fixture",
                redistribution_rights="cleared",
                recording_group="synthetic-recording-1",
                match_group="synthetic-match-1",
                overlay_profile_id=profile.profile_id,
                profile=profile_path.name,
                catalog=catalog_path.name,
                frame_timestamp_seconds=0.0,
                width=FIXTURE_WIDTH,
                height=FIXTURE_HEIGHT,
                partition="holdout",
                expected_supported=True,
                label_status="verified",
                verified_by="deterministic-fixture-generator",
                blue_team=request.blue_team,
                red_team=request.red_team,
                blue_picks=request.blue_picks,
                red_picks=request.red_picks,
            )
        ],
    )

    context = ScreenshotPredictionContext(
        match_timestamp=request.match_timestamp,
        league=request.league,
        tournament=request.tournament,
        patch=request.patch,
        series_id=request.series_id,
        game_number=request.game_number,
        blue_series_wins_before=request.blue_series_wins_before,
        red_series_wins_before=request.red_series_wins_before,
        blue_players=request.blue_players,
        red_players=request.red_players,
        blue_player_ids=request.blue_player_ids,
        red_player_ids=request.red_player_ids,
        blue_team_id=request.blue_team_id,
        red_team_id=request.red_team_id,
        first_pick_side=request.first_pick_side or "blue",
        fearless_bans=request.fearless_bans,
    )
    _write_json(profile_path, profile.model_dump(mode="json"))
    _write_json(catalog_path, catalog.model_dump(mode="json"))
    _write_json(context_path, context.model_dump(mode="json"))
    _write_json(manifest_path, manifest.model_dump(mode="json"))
    return {
        "directory": str(directory),
        "screenshot": str(screenshot_path),
        "profile": str(profile_path),
        "catalog": str(catalog_path),
        "context": str(context_path),
        "manifest": str(manifest_path),
    }
