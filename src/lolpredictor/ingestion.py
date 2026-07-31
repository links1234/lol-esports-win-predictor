"""Validated ingestion boundaries."""

from __future__ import annotations

import json
from pathlib import Path

from lolpredictor.schemas import HistoricalMatch
from lolpredictor.storage import initialize_database, insert_matches


def read_historical_jsonl(input_path: Path) -> list[HistoricalMatch]:
    matches: list[HistoricalMatch] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                matches.append(HistoricalMatch.model_validate(payload))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"Invalid record at {input_path}:{line_number}: {error}"
                ) from error
    return matches


def ingest_jsonl(database_path: Path, input_path: Path, *, source: str) -> int:
    initialize_database(database_path)
    matches = read_historical_jsonl(input_path)
    return insert_matches(database_path, matches, source=source)
