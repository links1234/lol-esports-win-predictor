"""Grouped, rights-aware benchmark for supported real broadcast overlays."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from lolpredictor.vision.parser import load_overlay_profile, parse_screenshot
from lolpredictor.vision.schemas import OverlayCorpusFrame, OverlayCorpusManifest

OVERLAY_BENCHMARK_SCHEMA_VERSION = "1"
MINIMUM_HOLDOUT_SUPPORTED_FRAMES = 100
MINIMUM_HOLDOUT_RECORDINGS = 3
MINIMUM_HOLDOUT_MATCHES = 10
MINIMUM_PROFILE_HOLDOUT_FRAMES = 25
MINIMUM_ACCEPTED_FIELD_PRECISION = 0.995
MINIMUM_EXACT_DRAFT_ACCURACY = 0.95


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_corpus_path(root: Path, value: str, *, field_name: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{field_name} must be relative to the corpus manifest")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{field_name} escapes the corpus directory")
    if not resolved.is_file():
        raise FileNotFoundError(f"{field_name} does not exist: {resolved}")
    return resolved


def load_overlay_corpus_manifest(path: Path) -> OverlayCorpusManifest:
    return OverlayCorpusManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _verified_frame_paths(
    root: Path,
    frame: OverlayCorpusFrame,
) -> tuple[Path, Path, Path]:
    image_path = _resolve_corpus_path(root, frame.image, field_name="image")
    profile_path = _resolve_corpus_path(root, frame.profile, field_name="profile")
    catalog_path = _resolve_corpus_path(root, frame.catalog, field_name="catalog")
    actual_checksum = _sha256_file(image_path)
    if actual_checksum != frame.image_sha256:
        raise ValueError(f"Image checksum mismatch for frame {frame.frame_id}")
    with Image.open(image_path) as image:
        actual_size = image.size
    if actual_size != (frame.width, frame.height):
        raise ValueError(
            f"Image resolution mismatch for frame {frame.frame_id}: "
            f"expected {(frame.width, frame.height)}, found {actual_size}"
        )
    profile = load_overlay_profile(profile_path)
    if profile.profile_id != frame.overlay_profile_id:
        raise ValueError(f"Overlay profile ID mismatch for frame {frame.frame_id}")
    return image_path, profile_path, catalog_path


def _evaluate_frame(
    root: Path,
    frame: OverlayCorpusFrame,
) -> dict[str, Any]:
    image_path, profile_path, catalog_path = _verified_frame_paths(root, frame)
    base: dict[str, Any] = {
        "frame_id": frame.frame_id,
        "partition": frame.partition,
        "recording_group": frame.recording_group,
        "match_group": frame.match_group,
        "profile_id": frame.overlay_profile_id,
        "expected_supported": frame.expected_supported,
    }
    try:
        candidate = parse_screenshot(
            image_path,
            profile_path,
            catalog_path,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        return {
            **base,
            "parser_status": "abstained",
            "error_type": type(error).__name__,
            "error": str(error),
            "required_auto_accepted": False,
            "accepted_team_fields": 0,
            "correct_accepted_team_fields": 0,
            "accepted_champion_fields": 0,
            "correct_accepted_champion_fields": 0,
            "exact_ten_champions": False,
        }

    if candidate.profile_id != frame.overlay_profile_id:
        raise ValueError(f"Parser profile mismatch for frame {frame.frame_id}")
    team_fields = (candidate.blue_team, candidate.red_team)
    expected_teams = (frame.blue_team, frame.red_team)
    champion_fields = (*candidate.blue_picks, *candidate.red_picks)
    expected_champions = (*frame.blue_picks, *frame.red_picks)
    accepted_team_fields = sum(field.disposition == "accepted" for field in team_fields)
    correct_accepted_team_fields = sum(
        field.disposition == "accepted" and field.value == expected
        for field, expected in zip(team_fields, expected_teams, strict=True)
    )
    accepted_champion_fields = sum(field.disposition == "accepted" for field in champion_fields)
    correct_accepted_champion_fields = sum(
        field.disposition == "accepted" and field.value == expected
        for field, expected in zip(champion_fields, expected_champions, strict=True)
    )
    required_auto_accepted = accepted_team_fields == 2 and accepted_champion_fields == 10
    exact_ten_champions = all(
        field.value == expected
        for field, expected in zip(champion_fields, expected_champions, strict=True)
    )
    return {
        **base,
        "parser_status": ("accepted" if required_auto_accepted else "confirmation_required"),
        "required_auto_accepted": required_auto_accepted,
        "accepted_team_fields": accepted_team_fields,
        "correct_accepted_team_fields": correct_accepted_team_fields,
        "accepted_champion_fields": accepted_champion_fields,
        "correct_accepted_champion_fields": correct_accepted_champion_fields,
        "exact_ten_champions": exact_ten_champions,
    }


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _summarize_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    supported = [frame for frame in frames if frame["expected_supported"]]
    unsupported = [frame for frame in frames if not frame["expected_supported"]]
    accepted_team_fields = sum(int(frame["accepted_team_fields"]) for frame in supported)
    correct_team_fields = sum(int(frame["correct_accepted_team_fields"]) for frame in supported)
    accepted_champion_fields = sum(int(frame["accepted_champion_fields"]) for frame in supported)
    correct_champion_fields = sum(
        int(frame["correct_accepted_champion_fields"]) for frame in supported
    )
    exact_drafts = sum(bool(frame["exact_ten_champions"]) for frame in supported)
    unsupported_false_accepts = sum(bool(frame["required_auto_accepted"]) for frame in unsupported)
    return {
        "frame_count": len(frames),
        "supported_frame_count": len(supported),
        "unsupported_frame_count": len(unsupported),
        "recording_group_count": len({frame["recording_group"] for frame in frames}),
        "match_group_count": len({frame["match_group"] for frame in frames}),
        "hard_abstention_count": sum(frame["parser_status"] == "abstained" for frame in supported),
        "confirmation_required_count": sum(
            frame["parser_status"] == "confirmation_required" for frame in supported
        ),
        "accepted_team_field_count": accepted_team_fields,
        "accepted_team_field_precision": _safe_ratio(
            correct_team_fields,
            accepted_team_fields,
        ),
        "accepted_team_field_coverage": _safe_ratio(
            accepted_team_fields,
            len(supported) * 2,
        ),
        "accepted_champion_field_count": accepted_champion_fields,
        "accepted_champion_field_precision": _safe_ratio(
            correct_champion_fields,
            accepted_champion_fields,
        ),
        "accepted_champion_field_coverage": _safe_ratio(
            accepted_champion_fields,
            len(supported) * 10,
        ),
        "exact_ten_champion_draft_count": exact_drafts,
        "exact_ten_champion_draft_accuracy": _safe_ratio(
            exact_drafts,
            len(supported),
        ),
        "unsupported_safe_abstention_count": len(unsupported) - unsupported_false_accepts,
        "unsupported_false_acceptance_count": unsupported_false_accepts,
    }


def _threshold_passed(value: object, threshold: float) -> bool:
    return isinstance(value, int | float) and float(value) >= threshold


def benchmark_overlay_corpus(manifest_path: Path) -> dict[str, Any]:
    """Run a deterministic grouped benchmark without fitting parser thresholds."""
    manifest_path = manifest_path.resolve()
    manifest = load_overlay_corpus_manifest(manifest_path)
    root = manifest_path.parent
    frame_results = [_evaluate_frame(root, frame) for frame in manifest.frames]
    partition_results = {
        partition: _summarize_frames(
            [frame for frame in frame_results if frame["partition"] == partition]
        )
        for partition in ("development", "holdout")
    }
    profile_frames: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frame_results:
        if frame["partition"] == "holdout":
            profile_frames[str(frame["profile_id"])].append(frame)
    holdout_profiles = {
        profile_id: _summarize_frames(frames)
        for profile_id, frames in sorted(profile_frames.items())
    }
    holdout = partition_results["holdout"]
    profile_gates: dict[str, dict[str, Any]] = {}
    for profile_id, metrics in holdout_profiles.items():
        supported_count = int(metrics["supported_frame_count"])
        passed = (
            supported_count >= MINIMUM_PROFILE_HOLDOUT_FRAMES
            and _threshold_passed(
                metrics["accepted_team_field_precision"],
                MINIMUM_ACCEPTED_FIELD_PRECISION,
            )
            and _threshold_passed(
                metrics["accepted_champion_field_precision"],
                MINIMUM_ACCEPTED_FIELD_PRECISION,
            )
            and _threshold_passed(
                metrics["exact_ten_champion_draft_accuracy"],
                MINIMUM_EXACT_DRAFT_ACCURACY,
            )
            and int(metrics["unsupported_false_acceptance_count"]) == 0
        )
        profile_gates[profile_id] = {
            "passed": passed,
            "metrics": metrics,
        }

    gates = {
        "holdout_size": {
            "passed": (
                int(holdout["supported_frame_count"]) >= MINIMUM_HOLDOUT_SUPPORTED_FRAMES
                and int(holdout["recording_group_count"]) >= MINIMUM_HOLDOUT_RECORDINGS
                and int(holdout["match_group_count"]) >= MINIMUM_HOLDOUT_MATCHES
            ),
            "observed": {
                "supported_frames": holdout["supported_frame_count"],
                "recordings": holdout["recording_group_count"],
                "matches": holdout["match_group_count"],
            },
        },
        "supported_profiles": {
            "passed": bool(profile_gates)
            and all(bool(value["passed"]) for value in profile_gates.values()),
            "profiles": profile_gates,
        },
        "unsupported_layouts": {
            "passed": int(holdout["unsupported_false_acceptance_count"]) == 0,
            "false_acceptance_count": holdout["unsupported_false_acceptance_count"],
        },
    }
    return {
        "benchmark_schema_version": OVERLAY_BENCHMARK_SCHEMA_VERSION,
        "corpus_id": manifest.corpus_id,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "thresholds": {
            "minimum_holdout_supported_frames": MINIMUM_HOLDOUT_SUPPORTED_FRAMES,
            "minimum_holdout_recordings": MINIMUM_HOLDOUT_RECORDINGS,
            "minimum_holdout_matches": MINIMUM_HOLDOUT_MATCHES,
            "minimum_profile_holdout_frames": MINIMUM_PROFILE_HOLDOUT_FRAMES,
            "minimum_accepted_field_precision": MINIMUM_ACCEPTED_FIELD_PRECISION,
            "minimum_exact_ten_champion_draft_accuracy": (MINIMUM_EXACT_DRAFT_ACCURACY),
        },
        "partitions": partition_results,
        "gates": gates,
        "release_gate_passed": all(bool(gate["passed"]) for gate in gates.values()),
        "frames": frame_results,
    }


def write_overlay_benchmark_report(
    manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    report = benchmark_overlay_corpus(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
