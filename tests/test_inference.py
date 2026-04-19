from __future__ import annotations

import json

from app.infer import BaselineInferenceEngine, save_inference_output
from app.schema import TranscriptLabel


class DummyTranscriber:
    def transcribe(self, audio_path: str) -> str:
        return "कल मीटिंग है"


class DummyLabeler:
    def label_transcript(self, transcript: str) -> TranscriptLabel:
        return TranscriptLabel(
            lang="hinglish",
            raw_mixed="kal meeting hai",
            clean_native="कल मीटिंग है",
            clean_english="There is a meeting tomorrow.",
        )


def test_baseline_inference_and_output_write(tmp_path) -> None:
    engine = BaselineInferenceEngine(transcriber=DummyTranscriber(), labeler=DummyLabeler())
    label = engine.run("sample.wav")
    output_path = save_inference_output(label, "sample.wav", output_dir=tmp_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["lang"] == "hinglish"
    assert output_path.exists()

