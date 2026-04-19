from __future__ import annotations

import json
import sys

from app.schema import TranscriptLabel


class FakeService:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def run(self, audio_path: str) -> TranscriptLabel:
        return TranscriptLabel(
            lang="hinglish",
            raw_mixed="kal meeting hai",
            clean_native="कल मीटिंग है",
            clean_english="There is a meeting tomorrow.",
        )


def test_cli_writes_output_file(tmp_path, monkeypatch) -> None:
    from app import main as main_module

    monkeypatch.setattr(main_module, "SpeechJsonInferenceService", FakeService)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python",
            "--audio",
            str(tmp_path / "clip.wav"),
            "--mode",
            "baseline",
            "--translation-backend",
            "rule",
            "--output-dir",
            str(tmp_path),
        ],
    )
    main_module.main()
    output_file = tmp_path / "clip.json"
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["clean_english"] == "There is a meeting tomorrow."

