from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class SupportedLang(str, Enum):
    HINGLISH = "hinglish"
    TANGLISH = "tanglish"
    BANGLISH = "banglish"


EXPECTED_KEY_ORDER = ["lang", "raw_mixed", "clean_native", "clean_english"]


class IntermediateSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio: str
    transcript: str


class TranscriptLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lang: SupportedLang = Field(...)
    raw_mixed: str = Field(min_length=1)
    clean_native: str = Field(min_length=1)
    clean_english: str = Field(min_length=1)

    def ordered_dict(self) -> dict[str, Any]:
        return {
            "lang": self.lang.value,
            "raw_mixed": self.raw_mixed,
            "clean_native": self.clean_native,
            "clean_english": self.clean_english,
        }

    def json_string(self) -> str:
        return json.dumps(
            self.ordered_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        )


class TrainingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio: str
    target: str


def parse_label_json(payload: str | dict[str, Any]) -> TranscriptLabel:
    if isinstance(payload, str):
        raw = json.loads(payload)
    else:
        raw = payload
    if list(raw.keys()) != EXPECTED_KEY_ORDER:
        raise ValueError(
            f"JSON keys must be exactly {EXPECTED_KEY_ORDER}, got {list(raw.keys())}"
        )
    return TranscriptLabel.model_validate(raw)


def validate_json_output(payload: str | dict[str, Any]) -> TranscriptLabel:
    try:
        return parse_label_json(payload)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ValueError(f"Invalid transcript label payload: {exc}") from exc

