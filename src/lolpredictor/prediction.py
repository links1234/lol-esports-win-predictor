"""Artifact-backed single and batch draft prediction."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from lolpredictor.artifacts import LoadedArtifact, load_artifact
from lolpredictor.features import build_model_row, compute_features, prediction_warnings
from lolpredictor.schemas import DraftRequest, PredictionResponse


def predict_with_artifact(
    artifact: LoadedArtifact,
    request: DraftRequest,
) -> PredictionResponse:
    warnings = prediction_warnings(
        request,
        artifact.feature_state,
        stale_after_days=int(artifact.manifest["stale_after_days"]),
    )
    if artifact.manifest["artifact_purpose"] == "development":
        warnings.insert(
            0,
            "Development model: this artifact has not passed a prospective promotion gate.",
        )
    computation = compute_features(request, artifact.feature_state)
    feature_frame = pd.DataFrame([build_model_row(request, artifact.feature_state, computation)])
    blue_probability = float(artifact.model.predict_probability(feature_frame)[0])
    cutoff = artifact.feature_state.data_cutoff_timestamp
    if cutoff is None:
        raise ValueError("Artifact feature state has no data cutoff")
    return PredictionResponse(
        blue_win_probability=blue_probability,
        red_win_probability=1.0 - blue_probability,
        model_version=artifact.manifest["model_version"],
        data_cutoff_timestamp=cutoff,
        warnings=warnings,
    )


def load_draft_request(input_path: Path) -> DraftRequest:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    return DraftRequest.model_validate(payload)


def predict_file(artifact_directory: Path, input_path: Path) -> PredictionResponse:
    artifact = load_artifact(artifact_directory)
    return predict_with_artifact(artifact, load_draft_request(input_path))


def parse_jsonl_requests(lines: Iterable[str]) -> list[DraftRequest]:
    requests: list[DraftRequest] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            requests.append(DraftRequest.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"Invalid prediction request on line {line_number}: {error}"
            ) from error
    return requests


def predict_jsonl(
    artifact_directory: Path,
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    artifact = load_artifact(artifact_directory)
    with input_path.open("r", encoding="utf-8") as handle:
        requests = parse_jsonl_requests(handle)
    responses = [predict_with_artifact(artifact, request) for request in requests]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for response in responses:
            handle.write(response.model_dump_json())
            handle.write("\n")
    return {
        "model_version": artifact.manifest["model_version"],
        "prediction_count": len(responses),
        "output": str(output_path.resolve()),
    }
