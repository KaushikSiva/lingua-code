from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import pipeline

from app.utils import setup_logging

LOGGER = setup_logging("app.transcribe", "transcribe.log")


@dataclass
class WhisperTranscriber:
    model_name: str = "openai/whisper-small"
    chunk_length_s: int = 30
    batch_size: int = 4

    def __post_init__(self) -> None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            torch_dtype=dtype,
            device=device,
        )

    def transcribe(self, audio_path: str) -> str:
        result = self.pipe(
            audio_path,
            chunk_length_s=self.chunk_length_s,
            batch_size=self.batch_size,
            generate_kwargs={"task": "transcribe"},
        )
        text = result["text"].strip()
        LOGGER.info("Transcribed %s -> %s", audio_path, text)
        return text
