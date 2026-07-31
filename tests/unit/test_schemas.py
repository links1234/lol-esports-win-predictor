from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from lolpredictor.schemas import DraftRequest


def valid_request() -> dict[str, object]:
    return {
        "match_timestamp": datetime(2025, 1, 1, tzinfo=UTC),
        "league": "LCK",
        "tournament": "Spring",
        "patch": "15.1",
        "blue_team": "A",
        "red_team": "B",
        "blue_players": ["a1", "a2", "a3", "a4", "a5"],
        "red_players": ["b1", "b2", "b3", "b4", "b5"],
        "blue_picks": ["c1", "c2", "c3", "c4", "c5"],
        "red_picks": ["c6", "c7", "c8", "c9", "c10"],
        "blue_bans": ["c11"],
        "red_bans": ["c12"],
    }


def test_timestamp_must_be_timezone_aware() -> None:
    payload = valid_request()
    payload["match_timestamp"] = datetime(2025, 1, 1)
    with pytest.raises(ValidationError, match="timezone"):
        DraftRequest.model_validate(payload)


def test_duplicate_pick_is_rejected() -> None:
    payload = valid_request()
    payload["red_picks"] = ["c1", "c7", "c8", "c9", "c10"]
    with pytest.raises(ValidationError, match="picked only once"):
        DraftRequest.model_validate(payload)


def test_banned_pick_is_rejected() -> None:
    payload = valid_request()
    payload["blue_bans"] = ["c1"]
    with pytest.raises(ValidationError, match="banned champion"):
        DraftRequest.model_validate(payload)


def test_extra_fields_are_rejected() -> None:
    payload = valid_request()
    payload["gold_difference"] = 1200
    with pytest.raises(ValidationError, match="Extra inputs"):
        DraftRequest.model_validate(payload)


def test_series_score_must_describe_only_completed_earlier_games() -> None:
    payload = valid_request()
    payload.update(
        {
            "series_id": "series-1",
            "game_number": 3,
            "blue_series_wins_before": 1,
            "red_series_wins_before": 0,
        }
    )
    with pytest.raises(ValidationError, match="prior blue wins"):
        DraftRequest.model_validate(payload)


def test_side_and_first_pick_are_independent() -> None:
    payload = valid_request()
    payload["first_pick_side"] = "red"
    request = DraftRequest.model_validate(payload)
    assert request.blue_team == "A"
    assert request.first_pick_side == "red"


def test_patch_is_normalized_for_live_input() -> None:
    payload = valid_request()
    payload["patch"] = "16.01"
    assert DraftRequest.model_validate(payload).patch == "16.1"

    payload["patch"] = "25.S01.02"
    assert DraftRequest.model_validate(payload).patch == "25.S1.2"
