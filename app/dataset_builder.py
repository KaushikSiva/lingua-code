from __future__ import annotations

import json
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from app.config import (
    DEFAULT_HF_DATASET_REPO_NAME,
    DEFAULT_OPENAI_BATCH_SIZE,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_VALID_RATIO,
    INTERMEDIATE_DIR,
    LABEL_TO_LANGUAGE,
    LANGUAGE_SPECS,
)
from app.datasets import ingest_indicvoices, load_bhasha_abhijnaanam
from app.label_transcript import TranscriptLabeler
from app.schema import TrainingRecord, validate_json_output
from app.transliteration import guess_label
from app.utils import append_jsonl, env_bool, env_int, normalize_whitespace, read_jsonl, write_jsonl, setup_logging

LOGGER = setup_logging("app.dataset_builder", "dataset_builder.log")
QUALITY_REJECTION_LOG = "logs/quality_rejections.jsonl"
DEFAULT_PROGRESS_EVERY = 100
DEFAULT_RESUME_LABELING = env_bool("PREP_RESUME_LABELING", True)
INDIC_CHAR_RANGES = (
    ("\u0900", "\u097f"),
    ("\u0980", "\u09ff"),
    ("\u0b80", "\u0bff"),
)


def build_intermediate_dataset(
    output_path: str | Path,
    audio_dir: str | Path,
    split: str = "train",
    max_samples_per_language: int | None = None,
    cache_dir: str | Path | None = None,
    token: str | None = None,
    streaming: bool = False,
    language_name: str | None = None,
) -> list[dict]:
    stage_name = _stage_name("ingest", language_name)
    rows = ingest_indicvoices(
        output_path=output_path,
        audio_dir=audio_dir,
        split=split,
        max_samples_per_language=max_samples_per_language,
        cache_dir=cache_dir,
        token=token,
        streaming=streaming,
        languages=[language_name] if language_name else None,
    )
    write_jsonl(output_path, rows)
    LOGGER.info("%s wrote %s intermediate rows to %s", stage_name, len(rows), output_path)
    return rows


def label_intermediate_dataset(
    input_path: str | Path,
    output_path: str | Path,
    provider: str = "heuristic",
    translation_backend: str = "nllb",
    max_retries: int = 3,
    use_aksharantar: bool = True,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    openai_batch_size: int = DEFAULT_OPENAI_BATCH_SIZE,
    reject_log_path: str = QUALITY_REJECTION_LOG,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
    resume: bool = DEFAULT_RESUME_LABELING,
    language_name: str | None = None,
    start_row: int = 1,
    end_row: int | None = None,
) -> list[dict]:
    stage_name = _stage_name("label", language_name)
    labeler = TranscriptLabeler.from_defaults(
        provider=provider,
        translation_backend=translation_backend,
        max_retries=max_retries,
        openai_model=openai_model,
        use_aksharantar=use_aksharantar,
        openai_batch_size=openai_batch_size,
        openai_heartbeat_seconds=env_int("PREP_OPENAI_HEARTBEAT_SECONDS", 30) or 30,
    )
    all_rows = list(read_jsonl(input_path))
    total_available_rows = len(all_rows)
    if start_row < 1:
        raise ValueError("start_row must be >= 1")
    if end_row is not None and end_row < start_row:
        raise ValueError("end_row must be >= start_row")
    slice_start = start_row - 1
    slice_end = end_row if end_row is not None else total_available_rows
    rows = all_rows[slice_start:slice_end]
    total_rows = len(rows)
    existing_rows = list(read_jsonl(output_path)) if resume and Path(output_path).exists() else []
    completed_count = len(existing_rows)
    labeled_rows: list[dict] = list(existing_rows)
    rejected_rows = 0
    started_at = time.monotonic()
    if not resume and Path(output_path).exists():
        Path(output_path).unlink()
        labeled_rows = []
        completed_count = 0
    if completed_count > total_rows:
        raise ValueError(
            f"Labeled output has {completed_count} rows but input only has {total_rows}; cannot resume safely"
        )
    pending_rows = rows[completed_count:]
    LOGGER.info(
        "%s resume state | selected_rows=%s-%s | existing labeled rows=%s | remaining=%s | total=%s",
        stage_name,
        start_row,
        start_row + total_rows - 1 if total_rows else start_row - 1,
        completed_count,
        len(pending_rows),
        total_rows,
    )

    if provider == "openai":
        batch_size = max(1, openai_batch_size)
        for offset in range(0, len(pending_rows), batch_size):
            batch = pending_rows[offset : offset + batch_size]
            batch_start = start_row + completed_count + offset
            batch_end = batch_start + len(batch) - 1
            labels = labeler.label_transcripts(
                [row["transcript"] for row in batch],
                batch_start=batch_start,
                batch_end=batch_end,
                log_prefix=stage_name,
            )
            for row, label in zip(batch, labels):
                issues = quality_issues_for_label(label.ordered_dict())
                if issues:
                    rejected_rows += 1
                    append_jsonl(
                        reject_log_path,
                        {
                            "audio": row["audio"],
                            "transcript": row["transcript"],
                            "label": label.ordered_dict(),
                            "issues": issues,
                        },
                    )
                    continue
                accepted_row = {"audio": row["audio"], "label": label.ordered_dict()}
                labeled_rows.append(accepted_row)
                append_jsonl(output_path, accepted_row)
            processed = min(completed_count + offset + len(batch), total_rows)
            log_progress(
                stage="label",
                language_name=language_name,
                processed=processed,
                total=total_rows,
                started_at=started_at,
                progress_every=max(1, progress_every),
                extra=f"accepted={len(labeled_rows)} rejected={rejected_rows}",
                force=True,
            )
    else:
        for offset, row in enumerate(pending_rows, start=1):
            index = completed_count + offset
            label = labeler.label_transcript(row["transcript"])
            issues = quality_issues_for_label(label.ordered_dict())
            if issues:
                rejected_rows += 1
                append_jsonl(
                    reject_log_path,
                    {
                        "audio": row["audio"],
                        "transcript": row["transcript"],
                        "label": label.ordered_dict(),
                        "issues": issues,
                    },
                )
            else:
                accepted_row = {"audio": row["audio"], "label": label.ordered_dict()}
                labeled_rows.append(accepted_row)
                append_jsonl(output_path, accepted_row)
            log_progress(
                stage="label",
                language_name=language_name,
                processed=index,
                total=total_rows,
                started_at=started_at,
                progress_every=max(1, progress_every),
                extra=f"accepted={len(labeled_rows)} rejected={rejected_rows}",
                force=index == total_rows,
            )
    LOGGER.info("%s wrote %s labeled rows to %s", stage_name, len(labeled_rows), output_path)
    return labeled_rows


def validate_labeled_dataset(
    input_path: str | Path,
    reject_log_path: str = QUALITY_REJECTION_LOG,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
    language_name: str | None = None,
) -> list[dict]:
    rows = list(read_jsonl(input_path))
    validated_rows: list[dict] = []
    rejected_rows = 0
    started_at = time.monotonic()
    total_rows = len(rows)
    for index, row in enumerate(rows, start=1):
        label = validate_json_output(json.dumps(row["label"], ensure_ascii=False))
        issues = quality_issues_for_label(label.ordered_dict())
        if issues:
            rejected_rows += 1
            append_jsonl(
                reject_log_path,
                {
                    "audio": row["audio"],
                    "label": label.ordered_dict(),
                    "issues": issues,
                    "stage": "validate",
                },
            )
        else:
            validated_rows.append({"audio": row["audio"], "label": label.ordered_dict()})
        log_progress(
            stage="validate",
            language_name=language_name,
            processed=index,
            total=total_rows,
            started_at=started_at,
            progress_every=max(1, progress_every),
            extra=f"accepted={len(validated_rows)} rejected={rejected_rows}",
            force=index == total_rows,
        )
    return validated_rows


def export_training_jsonl(
    labeled_path: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
    train_path: str | Path,
    valid_path: str | Path,
    valid_ratio: float = DEFAULT_VALID_RATIO,
    seed: int = 42,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
) -> tuple[list[dict], list[dict]]:
    labeled_paths = resolve_labeled_paths(labeled_path)
    if not labeled_paths:
        raise FileNotFoundError("No labeled JSONL files found to export")
    rows = merge_labeled_datasets(labeled_paths, progress_every=progress_every)
    train_rows, valid_rows = stratified_split_rows(rows, valid_ratio=valid_ratio, seed=seed)

    def _convert(items: list[dict], split_name: str) -> list[dict]:
        converted: list[dict] = []
        started_at = time.monotonic()
        total_items = len(items)
        for index, item in enumerate(items, start=1):
            record = TrainingRecord(
                audio=item["audio"],
                target=json.dumps(item["label"], ensure_ascii=False, separators=(",", ":")),
            )
            converted.append(record.model_dump())
            log_progress(
                stage=f"export:{split_name}",
                language_name=None,
                processed=index,
                total=total_items,
                started_at=started_at,
                progress_every=max(1, progress_every),
                force=index == total_items,
            )
        return converted

    train_records = _convert(train_rows, "train")
    valid_records = _convert(valid_rows, "valid")
    write_jsonl(train_path, train_records)
    write_jsonl(valid_path, valid_records)
    LOGGER.info("Exported %s train and %s valid rows", len(train_records), len(valid_records))
    return train_records, valid_records


def merge_labeled_datasets(
    labeled_paths: list[str | Path] | tuple[str | Path, ...],
    progress_every: int = DEFAULT_PROGRESS_EVERY,
) -> list[dict]:
    merged_rows: list[dict] = []
    paths = [Path(path) for path in labeled_paths]
    for path in paths:
        rows = validate_labeled_dataset(path, progress_every=progress_every, language_name=_language_from_path(path))
        merged_rows.extend(rows)
    LOGGER.info("Merged %s labeled rows from %s files", len(merged_rows), len(paths))
    return merged_rows


def resolve_labeled_paths(
    labeled_path: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
) -> list[Path]:
    if labeled_path is None:
        return discover_language_labeled_paths()
    if isinstance(labeled_path, (list, tuple)):
        return [Path(path) for path in labeled_path]
    return [Path(labeled_path)]


def discover_language_labeled_paths(intermediate_dir: str | Path = INTERMEDIATE_DIR) -> list[Path]:
    root = Path(intermediate_dir)
    paths: list[Path] = []
    default_path = root / "indicvoices_labeled.jsonl"
    if default_path.exists():
        paths.append(default_path)
    for language_name in LANGUAGE_SPECS:
        if language_name == "hindi" and default_path.exists():
            continue
        candidate = root / f"{language_name}_labeled.jsonl"
        if candidate.exists():
            paths.append(candidate)
    return paths


def _language_from_path(path: str | Path) -> str | None:
    stem = Path(path).stem.lower()
    for language_name in LANGUAGE_SPECS:
        if language_name in stem:
            return language_name
    return None


def _stage_name(stage: str, language_name: str | None = None) -> str:
    return f"{stage}[{language_name}]" if language_name else stage


def stratified_split_rows(
    rows: list[dict],
    valid_ratio: float = DEFAULT_VALID_RATIO,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]["lang"]].append(row)

    rng = random.Random(seed)
    train_rows: list[dict] = []
    valid_rows: list[dict] = []
    for lang, items in grouped.items():
        items = list(items)
        rng.shuffle(items)
        if len(items) <= 1:
            valid_size = 0
        else:
            valid_size = max(1, int(round(len(items) * valid_ratio)))
            valid_size = min(valid_size, len(items) - 1)
        valid_rows.extend(items[:valid_size])
        train_rows.extend(items[valid_size:])
        LOGGER.info(
            "Split %s rows for %s into %s train / %s valid",
            len(items),
            lang,
            len(items) - valid_size,
            valid_size,
        )
    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)
    return train_rows, valid_rows


def quality_issues_for_label(label: dict[str, Any]) -> list[str]:
    raw_mixed = normalize_whitespace(str(label["raw_mixed"]))
    clean_english = normalize_whitespace(str(label["clean_english"]))
    issues: list[str] = []

    if any(start <= character <= end for character in raw_mixed for start, end in INDIC_CHAR_RANGES):
        issues.append("raw_mixed_contains_native_script")
    if any(ord(character) >= 128 for character in raw_mixed):
        issues.append("raw_mixed_contains_non_ascii")
    if not any(character.isalpha() for character in raw_mixed):
        issues.append("raw_mixed_missing_alpha")
    if clean_english.lower().startswith("english translation:"):
        issues.append("clean_english_placeholder_prefix")
    if any(start <= character <= end for character in clean_english for start, end in INDIC_CHAR_RANGES):
        issues.append("clean_english_contains_native_script")
    if not any("a" <= character.lower() <= "z" for character in clean_english):
        issues.append("clean_english_missing_ascii_alpha")
    return issues


def compute_split_stats(records: list[dict]) -> dict[str, Any]:
    lang_counts = Counter()
    for record in records:
        payload = json.loads(record["target"])
        lang_counts[payload["lang"]] += 1
    return {
        "rows": len(records),
        "language_counts": dict(sorted(lang_counts.items())),
    }


def stage_training_dataset_for_hub(
    train_path: str | Path,
    valid_path: str | Path,
    staging_dir: str | Path,
    repo_id: str | None = None,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
) -> dict[str, Any]:
    staging_root = Path(staging_dir)
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)

    train_records = stage_split_records_for_hub(
        train_path,
        staging_root=staging_root,
        split_name="train",
        progress_every=progress_every,
    )
    valid_records = stage_split_records_for_hub(
        valid_path,
        staging_root=staging_root,
        split_name="valid",
        progress_every=progress_every,
    )

    write_jsonl(staging_root / "train.jsonl", train_records)
    write_jsonl(staging_root / "valid.jsonl", valid_records)

    stats = {
        "repo_id": repo_id,
        "train": compute_split_stats(train_records),
        "valid": compute_split_stats(valid_records),
    }
    (staging_root / "README.md").write_text(
        build_dataset_card(repo_id=repo_id, stats=stats),
        encoding="utf-8",
    )
    LOGGER.info("Staged dataset for Hub at %s", staging_root)
    return stats


def stage_split_records_for_hub(
    jsonl_path: str | Path,
    staging_root: Path,
    split_name: str,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
) -> list[dict]:
    rows = list(read_jsonl(jsonl_path))
    staged_records: list[dict] = []
    started_at = time.monotonic()
    total_rows = len(rows)
    for index, row in enumerate(rows, start=1):
        record = TrainingRecord.model_validate(row)
        label = validate_json_output(json.loads(record.target))
        language_name = LABEL_TO_LANGUAGE[label.lang.value]
        source_audio = Path(record.audio)
        suffix = source_audio.suffix or ".wav"
        staged_audio = Path("audio") / split_name / language_name / f"{language_name}_{index - 1:06d}{suffix}"
        destination = staging_root / staged_audio
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_audio, destination)
        staged_records.append(
            TrainingRecord(
                audio=staged_audio.as_posix(),
                target=record.target,
            ).model_dump()
        )
        log_progress(
            stage=f"stage:{split_name}",
            language_name=None,
            processed=index,
            total=total_rows,
            started_at=started_at,
            progress_every=max(1, progress_every),
            force=index == total_rows,
        )
    return staged_records


def build_dataset_card(repo_id: str | None, stats: dict[str, Any]) -> str:
    repo_line = repo_id or "<owner>/linguacode-audio-json-v1"
    total_rows = stats["train"]["rows"] + stats["valid"]["rows"]
    if total_rows < 1_000:
        size_category = "n<1K"
    elif total_rows < 10_000:
        size_category = "1K<n<10K"
    elif total_rows < 100_000:
        size_category = "10K<n<100K"
    else:
        size_category = "100K<n<1M"
    return f"""---
language:
- hi
- ta
- bn
license: apache-2.0
task_categories:
- automatic-speech-recognition
- text-generation
pretty_name: Linguacode Audio JSON Dataset
size_categories:
- {size_category}
---

# Linguacode Audio JSON Dataset

This dataset is packaged for fine-tuning `Qwen/Qwen2-Audio-7B-Instruct` on Hindi/Hinglish, Tamil/Tanglish, and Bengali/Banglish audio-to-JSON generation.

## Repo

- Dataset repo: `{repo_line}`
- Format: local audio files plus `train.jsonl` and `valid.jsonl`
- Schema: `{{"audio":"audio/train/...wav","target":"{{\\"lang\\":...\\"clean_english\\":...}}"}}`

## Sources

- `ai4bharat/IndicVoices`
- `ai4bharat/Aksharantar`
- `ai4bharat/Bhasha-Abhijnaanam`

## Split Stats

### Train

- Rows: {stats["train"]["rows"]}
- Language counts: {json.dumps(stats["train"]["language_counts"], ensure_ascii=False)}

### Valid

- Rows: {stats["valid"]["rows"]}
- Language counts: {json.dumps(stats["valid"]["language_counts"], ensure_ascii=False)}
"""


def resolve_default_dataset_repo(token: str | None) -> str:
    api = HfApi(token=token)
    username = api.whoami()["name"]
    return f"{username}/{DEFAULT_HF_DATASET_REPO_NAME}"


def upload_staged_dataset_to_hub(
    staging_dir: str | Path,
    token: str | None,
    repo_id: str | None = None,
    private: bool = True,
    commit_message: str = "Upload Linguacode audio JSON dataset",
) -> str:
    api = HfApi(token=token)
    resolved_repo_id = repo_id or resolve_default_dataset_repo(token)
    api.create_repo(
        repo_id=resolved_repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=resolved_repo_id,
        repo_type="dataset",
        folder_path=str(staging_dir),
        commit_message=commit_message,
    )
    LOGGER.info("Uploaded staged dataset at %s to %s", staging_dir, resolved_repo_id)
    return resolved_repo_id


def log_progress(
    stage: str,
    language_name: str | None,
    processed: int,
    total: int,
    started_at: float,
    progress_every: int,
    extra: str | None = None,
    force: bool = False,
) -> None:
    if total <= 0:
        return
    if not force and processed % progress_every != 0:
        return
    elapsed = max(0.0, time.monotonic() - started_at)
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = total - processed
    eta_seconds = int(remaining / rate) if rate > 0 else 0
    percent = (processed / total) * 100
    stage_name = _stage_name(stage, language_name)
    message = (
        f"{stage_name} progress {processed}/{total} | {percent:.1f}% | "
        f"elapsed {format_duration(elapsed)} | eta {format_duration(eta_seconds)}"
    )
    if extra:
        message += f" | {extra}"
    LOGGER.info(message)


def format_duration(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def evaluate_language_id_with_bhasha(
    output_path: str | Path,
    max_rows_per_language: int = 100,
) -> dict[str, float]:
    results: dict[str, float] = {}
    rows_out: list[dict] = []
    for language_name in ("hindi", "tamil", "bengali"):
        dataset = load_bhasha_abhijnaanam(language_name)
        total = 0
        correct = 0
        for row in dataset:
            native = str(row.get("native sentence", "")).strip()
            romanized = str(row.get("romanized sentence", "")).strip()
            for sample_text in (native, romanized):
                if not sample_text:
                    continue
                predicted = guess_label(sample_text)
                total += 1
                correct += int(predicted == LANGUAGE_SPECS[language_name].label)
                rows_out.append(
                    {
                        "language": language_name,
                        "text": sample_text,
                        "predicted": predicted,
                        "expected": LANGUAGE_SPECS[language_name].label,
                    }
                )
                if total >= max_rows_per_language:
                    break
            if total >= max_rows_per_language:
                break
        results[language_name] = (correct / total) if total else 0.0

    write_jsonl(output_path, rows_out)
    return results
