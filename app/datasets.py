from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterable

from app.config import HF_LANGUAGE_NAMES, LANGUAGE_SPECS
from app.schema import IntermediateSample
from app.utils import load_audio, normalize_whitespace, save_audio, setup_logging

LOGGER = setup_logging("app.datasets", "datasets.log")

AUDIO_FIELD_CANDIDATES = ("audio_filepath", "audio", "audio_path", "path", "file")
TEXT_FIELD_CANDIDATES = (
    "normalized",
    "transcript",
    "text",
    "sentence",
    "normalized_text",
    "verbatim",
    "cleaned_text",
    "unsanitized_normalized",
    "unsanitized_verbatim",
)
LANG_FIELD_CANDIDATES = ("language", "lang", "locale")


def _resolve_first(row: dict, field_names: Iterable[str]):
    for field_name in field_names:
        if field_name in row and row[field_name] not in (None, ""):
            return row[field_name]
    return None


def _resolve_language_name(row: dict) -> str | None:
    raw_language = _resolve_first(row, LANG_FIELD_CANDIDATES)
    if raw_language is None:
        return None
    value = str(raw_language).strip()
    for spec in LANGUAGE_SPECS.values():
        if value.lower() == spec.hf_name.lower():
            return spec.hf_name
    return value


def _extract_audio_to_file(
    audio_value,
    destination: Path,
    sample_rate: int = 16000,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(audio_value, str):
        source = Path(audio_value)
        if source.exists():
            shutil.copy2(source, destination)
            return str(destination)
        raise FileNotFoundError(f"Audio path does not exist: {source}")

    if isinstance(audio_value, dict):
        path = audio_value.get("path")
        if path and Path(path).exists():
            shutil.copy2(path, destination)
            return str(destination)
        raw_bytes = audio_value.get("bytes")
        if raw_bytes:
            suffix = Path(path).suffix if path else ".bin"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(raw_bytes)
                temp_path = Path(handle.name)
            try:
                audio_array, detected_sr = load_audio(temp_path, sample_rate=sample_rate)
                save_audio(destination, audio=audio_array, sample_rate=detected_sr)
                return str(destination)
            finally:
                temp_path.unlink(missing_ok=True)
        array = audio_value.get("array")
        sr = audio_value.get("sampling_rate", sample_rate)
        if array is not None:
            save_audio(destination, audio=array, sample_rate=sr)
            return str(destination)

    raise ValueError("Unsupported audio value. Expected path string or datasets Audio dict.")


def load_indicvoices_dataset(
    language_name: str,
    split: str = "train",
    cache_dir: str | Path | None = None,
    token: str | None = None,
    streaming: bool = False,
):
    from datasets import Audio, load_dataset

    if language_name not in LANGUAGE_SPECS:
        raise KeyError(f"Unsupported IndicVoices language: {language_name}")

    dataset = load_dataset(
        "ai4bharat/IndicVoices",
        language_name,
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
        token=token,
        streaming=streaming,
    )
    if hasattr(dataset, "column_names"):
        if "audio_filepath" in dataset.column_names:
            dataset = dataset.cast_column("audio_filepath", Audio(decode=False))
        elif "audio" in dataset.column_names:
            dataset = dataset.cast_column("audio", Audio(decode=False))
    return dataset


def ingest_indicvoices(
    output_path: str | Path,
    audio_dir: str | Path,
    split: str = "train",
    max_samples_per_language: int | None = None,
    cache_dir: str | Path | None = None,
    token: str | None = None,
    streaming: bool = False,
    languages: list[str] | tuple[str, ...] | None = None,
) -> list[dict]:
    destination_root = Path(audio_dir)
    output_rows: list[dict] = []
    counts: dict[str, int] = {language: 0 for language in HF_LANGUAGE_NAMES}
    selected_languages = list(languages) if languages else list(LANGUAGE_SPECS.keys())
    for language_name in selected_languages:
        if language_name not in LANGUAGE_SPECS:
            raise KeyError(f"Unsupported IndicVoices language: {language_name}")
        spec = LANGUAGE_SPECS[language_name]
        dataset = load_indicvoices_dataset(
            language_name=language_name,
            split=split,
            cache_dir=cache_dir,
            token=token,
            streaming=streaming,
        )
        language = spec.hf_name

        for index, row in enumerate(dataset):
            if max_samples_per_language is not None and counts[language] >= max_samples_per_language:
                break

            transcript = _resolve_first(row, TEXT_FIELD_CANDIDATES)
            audio_value = _resolve_first(row, AUDIO_FIELD_CANDIDATES)
            if not transcript or audio_value is None:
                continue

            language_slug = language.lower()
            audio_path = destination_root / language_slug / f"{language_slug}_{counts[language]:06d}.wav"
            try:
                local_audio_path = _extract_audio_to_file(audio_value, audio_path)
            except Exception as exc:
                LOGGER.warning(
                    "Failed to materialize audio for %s row %s: %s",
                    language_name,
                    index,
                    exc,
                )
                continue

            sample = IntermediateSample(
                audio=local_audio_path,
                transcript=normalize_whitespace(str(transcript)),
            )
            output_rows.append(sample.model_dump())
            counts[language] += 1

    return output_rows


def load_bhasha_abhijnaanam(
    language_name: str,
    split: str = "train",
    cache_dir: str | Path | None = None,
):
    from datasets import load_dataset

    spec = LANGUAGE_SPECS[language_name]
    return load_dataset(
        "ai4bharat/Bhasha-Abhijnaanam",
        spec.bhasha_code,
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
