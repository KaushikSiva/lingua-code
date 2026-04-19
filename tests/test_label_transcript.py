from __future__ import annotations

import logging

from app.label_transcript import OpenAIJsonLabeler, TranscriptLabeler


def test_openai_batch_logging_messages(caplog, monkeypatch) -> None:
    caplog.set_level(logging.INFO)

    def fake_label_batch(
        self,
        transcripts,
        prompt_template,
        lang_hints=None,
        batch_start=None,
        batch_end=None,
        attempt=None,
        log_prefix=None,
    ):
        return [
            {
                "lang": "hinglish",
                "raw_mixed": text,
                "clean_native": text,
                "clean_english": text,
            }
            for text in transcripts
        ]

    monkeypatch.setattr(OpenAIJsonLabeler, "label_batch", fake_label_batch)
    labeler = TranscriptLabeler.from_defaults(provider="openai", translation_backend="rule")
    labeler.label_transcripts(["a", "b"], batch_start=11, batch_end=12, log_prefix="label[tamil]")
    messages = [record.message for record in caplog.records]
    assert any("label[tamil] batch start | rows 11-12 | size=2 | attempt=1" in message for message in messages)
    assert any("label[tamil] batch complete | rows 11-12 | size=2 | attempt=1" in message for message in messages)


def test_openai_batch_failure_logging(caplog, monkeypatch) -> None:
    caplog.set_level(logging.WARNING)

    def fake_label_batch(
        self,
        transcripts,
        prompt_template,
        lang_hints=None,
        batch_start=None,
        batch_end=None,
        attempt=None,
        log_prefix=None,
    ):
        raise ValueError("bad json")

    monkeypatch.setattr(OpenAIJsonLabeler, "label_batch", fake_label_batch)
    labeler = TranscriptLabeler.from_defaults(provider="openai", translation_backend="rule", max_retries=1)
    try:
        labeler.label_transcripts(["a"], batch_start=1, batch_end=1, log_prefix="label[hindi]")
    except ValueError:
        pass
    messages = [record.message for record in caplog.records]
    assert any("label[hindi] batch failed | rows 1-1 | size=1 | attempt=1" in message for message in messages)
