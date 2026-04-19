from __future__ import annotations

from app.label_transcript import TranscriptLabeler


def test_hindi_labeling_returns_required_keys() -> None:
    labeler = TranscriptLabeler.from_defaults(translation_backend="rule", use_aksharantar=False)
    result = labeler.label_transcript("कल मीटिंग है")
    assert result.lang.value == "hinglish"
    assert result.clean_native == "कल मीटिंग है"
    assert result.raw_mixed
    assert result.clean_english == "There is a meeting tomorrow."


def test_tamil_labeling_returns_required_keys() -> None:
    labeler = TranscriptLabeler.from_defaults(translation_backend="rule", use_aksharantar=False)
    result = labeler.label_transcript("நாளைக்கு meeting இருக்கு")
    assert result.lang.value == "tanglish"
    assert result.clean_native == "நாளைக்கு meeting இருக்கு"
    assert result.raw_mixed
    assert result.clean_english == "There is a meeting tomorrow."


def test_bengali_labeling_returns_required_keys() -> None:
    labeler = TranscriptLabeler.from_defaults(translation_backend="rule", use_aksharantar=False)
    result = labeler.label_transcript("কাল meeting আছে")
    assert result.lang.value == "banglish"
    assert result.clean_native == "কাল meeting আছে"
    assert result.raw_mixed
    assert result.clean_english == "There is a meeting tomorrow."
