from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

from lolpredictor.vision.benchmark import benchmark_overlay_corpus
from lolpredictor.vision.fixtures import build_screenshot_fixture
from lolpredictor.vision.parser import (
    DraftConfirmationRequiredError,
    build_confirmed_draft,
    load_screenshot_context,
    parse_screenshot,
)
from lolpredictor.vision.schemas import (
    OverlayCorpusManifest,
    ScreenshotPredictionContext,
    TemplateCatalog,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def vision_fixture(tmp_path: Path) -> dict[str, Path]:
    result = build_screenshot_fixture(
        tmp_path / "vision",
        REPOSITORY_ROOT / "examples" / "sample_draft.json",
    )
    return {key: Path(value) for key, value in result.items()}


def test_supported_overlay_extracts_exact_structured_draft(
    vision_fixture: dict[str, Path],
) -> None:
    candidate = parse_screenshot(
        vision_fixture["screenshot"],
        vision_fixture["profile"],
        vision_fixture["catalog"],
    )
    assert candidate.requires_confirmation is False
    assert candidate.blue_team.value == "SYN_ALPHA"
    assert candidate.red_team.value == "SYN_BRAVO"
    assert tuple(field.value for field in candidate.blue_picks) == (
        "Aegis",
        "Bramble",
        "Cipher",
        "Dawn",
        "Ember",
    )
    assert tuple(field.value for field in candidate.red_picks) == (
        "Atlas",
        "Brook",
        "Cinder",
        "Drift",
        "Echo",
    )


def test_catalog_consolidates_multiple_variants_before_computing_margin(
    vision_fixture: dict[str, Path],
) -> None:
    catalog_payload = json.loads(vision_fixture["catalog"].read_text(encoding="utf-8"))
    first_champion = next(
        entry for entry in catalog_payload["templates"] if entry["kind"] == "champion"
    )
    first_champion["variant_id"] = "original"
    alternate = {**first_champion, "variant_id": "alternate"}
    catalog_payload["templates"].append(alternate)
    variant_catalog = vision_fixture["directory"] / "variant-catalog.json"
    variant_catalog.write_text(json.dumps(catalog_payload), encoding="utf-8")

    candidate = parse_screenshot(
        vision_fixture["screenshot"],
        vision_fixture["profile"],
        variant_catalog,
    )

    first_pick = candidate.blue_picks[0]
    assert first_pick.runner_up_value != first_pick.value
    assert first_pick.margin > 0.0


def test_catalog_rejects_duplicate_variant_identity() -> None:
    with pytest.raises(ValidationError, match="template variants must be unique"):
        TemplateCatalog.model_validate(
            {
                "catalog_schema_version": "1",
                "catalog_id": "duplicate-variant",
                "templates": [
                    {
                        "kind": "team",
                        "value": "Alpha",
                        "variant_id": "default",
                        "image": "alpha.png",
                    },
                    {
                        "kind": "champion",
                        "value": "Aegis",
                        "variant_id": "same",
                        "image": "aegis-a.png",
                    },
                    {
                        "kind": "champion",
                        "value": "aegis",
                        "variant_id": "SAME",
                        "image": "aegis-b.png",
                    },
                ],
            }
        )


def test_pixels_outside_profile_regions_cannot_change_the_parse(
    vision_fixture: dict[str, Path],
) -> None:
    original = parse_screenshot(
        vision_fixture["screenshot"],
        vision_fixture["profile"],
        vision_fixture["catalog"],
    )
    changed_path = vision_fixture["directory"] / "changed-center.png"
    with Image.open(vision_fixture["screenshot"]) as source:
        changed = source.convert("RGB")
    draw = ImageDraw.Draw(changed)
    draw.rectangle((170, 80, 1110, 610), fill=(255, 0, 255))
    changed.save(changed_path)

    reparsed = parse_screenshot(
        changed_path,
        vision_fixture["profile"],
        vision_fixture["catalog"],
    )
    assert original.screenshot_sha256 != reparsed.screenshot_sha256
    assert original.model_dump(exclude={"screenshot_sha256"}) == reparsed.model_dump(
        exclude={"screenshot_sha256"}
    )


def test_uncertain_required_crop_abstains_until_correction(
    vision_fixture: dict[str, Path],
) -> None:
    changed_path = vision_fixture["directory"] / "uncertain-pick.png"
    with Image.open(vision_fixture["screenshot"]) as source:
        changed = source.convert("RGB")
    draw = ImageDraw.Draw(changed)
    draw.rectangle((32, 112, 95, 175), fill=(127, 127, 127))
    changed.save(changed_path)

    candidate = parse_screenshot(
        changed_path,
        vision_fixture["profile"],
        vision_fixture["catalog"],
    )
    assert candidate.requires_confirmation is True
    assert candidate.blue_picks[0].disposition == "review"
    context = load_screenshot_context(vision_fixture["context"])
    with pytest.raises(DraftConfirmationRequiredError, match="blue_picks"):
        build_confirmed_draft(candidate, context)


def test_screenshot_context_rejects_in_game_state(
    vision_fixture: dict[str, Path],
) -> None:
    payload = json.loads(vision_fixture["context"].read_text(encoding="utf-8"))
    payload["gold_difference"] = 1500
    with pytest.raises(ValidationError, match="Extra inputs"):
        ScreenshotPredictionContext.model_validate(payload)


def _corpus_frame(
    vision_fixture: dict[str, Path],
    *,
    frame_id: str = "frame-1",
    partition: str = "holdout",
    recording_group: str = "recording-1",
    match_group: str = "match-1",
) -> dict[str, object]:
    screenshot = vision_fixture["screenshot"]
    return {
        "frame_id": frame_id,
        "image": screenshot.name,
        "image_sha256": hashlib.sha256(screenshot.read_bytes()).hexdigest(),
        "source_url": "https://example.invalid/public-broadcast",
        "source_license": "synthetic-test-fixture",
        "redistribution_rights": "cleared",
        "recording_group": recording_group,
        "match_group": match_group,
        "overlay_profile_id": "synthetic-overlay-v1",
        "profile": vision_fixture["profile"].name,
        "catalog": vision_fixture["catalog"].name,
        "frame_timestamp_seconds": 10.0,
        "width": 1280,
        "height": 720,
        "partition": partition,
        "expected_supported": True,
        "label_status": "verified",
        "verified_by": "fixture-generator",
        "blue_team": "SYN_ALPHA",
        "red_team": "SYN_BRAVO",
        "blue_picks": ["Aegis", "Bramble", "Cipher", "Dawn", "Ember"],
        "red_picks": ["Atlas", "Brook", "Cinder", "Drift", "Echo"],
    }


def test_overlay_corpus_benchmark_reports_precision_without_passing_small_fixture(
    vision_fixture: dict[str, Path],
) -> None:
    manifest_path = vision_fixture["directory"] / "corpus.json"
    manifest_path.write_text(
        json.dumps(
            {
                "corpus_schema_version": "1",
                "corpus_id": "synthetic-corpus-v1",
                "frames": [_corpus_frame(vision_fixture)],
            }
        ),
        encoding="utf-8",
    )

    report = benchmark_overlay_corpus(manifest_path)

    holdout = report["partitions"]["holdout"]
    assert holdout["accepted_team_field_precision"] == 1.0
    assert holdout["accepted_champion_field_precision"] == 1.0
    assert holdout["exact_ten_champion_draft_accuracy"] == 1.0
    assert report["gates"]["holdout_size"]["passed"] is False
    assert report["release_gate_passed"] is False


def test_overlay_corpus_groups_cannot_cross_partitions(
    vision_fixture: dict[str, Path],
) -> None:
    first = _corpus_frame(
        vision_fixture,
        frame_id="development-frame",
        partition="development",
    )
    second = _corpus_frame(
        vision_fixture,
        frame_id="holdout-frame",
        partition="holdout",
    )

    with pytest.raises(ValidationError, match="recording_group"):
        OverlayCorpusManifest.model_validate(
            {
                "corpus_schema_version": "1",
                "corpus_id": "invalid-crossing-corpus",
                "frames": [first, second],
            }
        )


def test_overlay_corpus_rejects_image_integrity_mismatch(
    vision_fixture: dict[str, Path],
) -> None:
    frame = _corpus_frame(vision_fixture)
    frame["image_sha256"] = "0" * 64
    manifest_path = vision_fixture["directory"] / "corpus.json"
    manifest_path.write_text(
        json.dumps(
            {
                "corpus_schema_version": "1",
                "corpus_id": "tampered-corpus",
                "frames": [frame],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        benchmark_overlay_corpus(manifest_path)
