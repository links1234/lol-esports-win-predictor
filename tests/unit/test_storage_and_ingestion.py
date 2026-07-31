import json
from datetime import timedelta
from pathlib import Path

import pytest

from lolpredictor.features import generate_historical_features
from lolpredictor.ingestion import ingest_jsonl, read_historical_jsonl
from lolpredictor.storage import (
    database_summary,
    load_feature_table,
    load_matches,
    write_feature_table,
)


def test_jsonl_ingestion_validates_before_duckdb_insert(
    tmp_path: Path,
    synthetic_matches: list,
) -> None:
    input_path = tmp_path / "matches.jsonl"
    input_path.write_text(
        synthetic_matches[0].model_dump_json() + "\n",
        encoding="utf-8",
    )
    database = tmp_path / "matches.duckdb"
    assert ingest_jsonl(database, input_path, source="test") == 1
    assert len(load_matches(database)) == 1
    assert database_summary(database)["match_count"] == 1


def test_invalid_jsonl_reports_line_number(
    tmp_path: Path,
    synthetic_matches: list,
) -> None:
    input_path = tmp_path / "bad.jsonl"
    valid = json.loads(synthetic_matches[0].model_dump_json())
    input_path.write_text(
        json.dumps(valid) + "\n" + '{"match_id":\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        read_historical_jsonl(input_path)


def test_storage_cutoff_is_exclusive_and_applied_to_matches_and_features(
    tmp_path: Path,
    synthetic_matches: list,
    settings,
) -> None:
    input_path = tmp_path / "matches.jsonl"
    input_path.write_text(
        "\n".join(match.model_dump_json() for match in synthetic_matches[:12]) + "\n",
        encoding="utf-8",
    )
    database = tmp_path / "matches.duckdb"
    assert ingest_jsonl(database, input_path, source="test") == 12
    frame, _ = generate_historical_features(synthetic_matches[:12], settings.features)
    write_feature_table(
        database,
        frame,
        feature_schema_version=settings.features.feature_schema_version,
    )
    cutoff = synthetic_matches[4].match_timestamp

    bounded_matches = load_matches(database, before_timestamp=cutoff)
    bounded_features = load_feature_table(database, before_timestamp=cutoff)

    assert len(bounded_matches) == 4
    assert len(bounded_features) == 4
    assert all(match.match_timestamp < cutoff for match in bounded_matches)
    assert bounded_features["match_timestamp"].max() < cutoff
    with pytest.raises(ValueError, match="must include a timezone"):
        load_matches(
            database,
            before_timestamp=cutoff.replace(tzinfo=None) + timedelta(),
        )
