from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Iterable, Iterator

import librosa
import soundfile as sf
from dotenv import load_dotenv

from app.config import LOG_DIR, TRANSCRIPT_PROMPT_PATH, ensure_project_dirs


def bootstrap() -> None:
    load_dotenv()
    ensure_project_dirs()


def setup_logging(name: str, log_file: str | None = None) -> logging.Logger:
    bootstrap()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        file_handler = logging.FileHandler(LOG_DIR / log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_prompt_template(path: Path = TRANSCRIPT_PROMPT_PATH) -> str:
    return path.read_text(encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return float(value)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def slugify_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return cleaned.strip("._") or "sample"


def load_audio(
    audio_path: str | Path,
    sample_rate: int = 16000,
) -> tuple[list[float], int]:
    audio, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    return audio.tolist(), sr


def save_audio(
    audio_path: str | Path,
    audio: list[float] | tuple[float, ...],
    sample_rate: int = 16000,
) -> None:
    destination = Path(audio_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, audio, sample_rate)


def exact_json_from_text(text: str) -> str | None:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]
