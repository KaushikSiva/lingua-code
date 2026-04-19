from __future__ import annotations

import json

import pytest

from app.schema import TranscriptLabel, validate_json_output


def test_valid_schema_round_trip() -> None:
    payload = {
        "lang": "hinglish",
        "raw_mixed": "kal meeting hai",
        "clean_native": "कल मीटिंग है",
        "clean_english": "There is a meeting tomorrow.",
    }
    parsed = validate_json_output(json.dumps(payload, ensure_ascii=False))
    assert isinstance(parsed, TranscriptLabel)
    assert parsed.lang.value == "hinglish"
    assert list(parsed.ordered_dict().keys()) == [
        "lang",
        "raw_mixed",
        "clean_native",
        "clean_english",
    ]


def test_invalid_json_rejected() -> None:
    with pytest.raises(ValueError):
        validate_json_output('{"lang":"hinglish","raw_mixed":"x"}')


def test_wrong_key_order_rejected() -> None:
    with pytest.raises(ValueError):
        validate_json_output(
            '{"raw_mixed":"x","lang":"hinglish","clean_native":"y","clean_english":"z"}'
        )

