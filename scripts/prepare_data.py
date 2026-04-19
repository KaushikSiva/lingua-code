from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import (
    DEFAULT_OPENAI_BATCH_SIZE,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_VALID_RATIO,
    EXPORT_DIR,
    INTERMEDIATE_DIR,
    SUPPORTED_LANGUAGES,
    ensure_project_dirs,
)
from app.dataset_builder import (
    build_intermediate_dataset,
    discover_language_labeled_paths,
    evaluate_language_id_with_bhasha,
    export_training_jsonl,
    label_intermediate_dataset,
    validate_labeled_dataset,
)
from app.utils import env_bool, env_float, env_int, setup_logging

LOGGER = setup_logging("scripts.prepare_data", "prepare_data.log")


def legacy_shared_intermediate_path() -> str:
    return str(INTERMEDIATE_DIR / "indicvoices_intermediate.jsonl")


def legacy_shared_labeled_path() -> str:
    return str(INTERMEDIATE_DIR / "indicvoices_labeled.jsonl")


def default_intermediate_path(language: str | None) -> str:
    legacy_path = Path(legacy_shared_intermediate_path())
    if language == "hindi" and legacy_path.exists():
        return str(legacy_path)
    if language:
        return str(INTERMEDIATE_DIR / f"{language}_intermediate.jsonl")
    return str(legacy_path)


def default_labeled_path(language: str | None) -> str:
    legacy_path = Path(legacy_shared_labeled_path())
    if language == "hindi" and legacy_path.exists():
        return str(legacy_path)
    if language:
        return str(INTERMEDIATE_DIR / f"{language}_labeled.jsonl")
    return str(legacy_path)


def resolve_default_path(current: str, default_path: str, replacement: str) -> str:
    return replacement if current == default_path else current


def build_parser() -> argparse.ArgumentParser:
    import os

    parser = argparse.ArgumentParser(description="Prepare Indic speech data for audio-to-JSON training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Load IndicVoices and save intermediate JSONL.")
    ingest.add_argument("--output", default=str(INTERMEDIATE_DIR / "indicvoices_intermediate.jsonl"))
    ingest.add_argument("--audio-dir", default=str(INTERMEDIATE_DIR / "audio"))
    ingest.add_argument("--language", choices=SUPPORTED_LANGUAGES, default=None)
    ingest.add_argument("--split", default=os.getenv("PREP_SPLIT", "train"))
    ingest.add_argument(
        "--max-samples-per-language",
        type=int,
        default=env_int("PREP_MAX_SAMPLES_PER_LANGUAGE"),
    )
    ingest.add_argument("--cache-dir", default=None)
    ingest.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    ingest.add_argument("--streaming", action="store_true", default=env_bool("PREP_STREAMING", False))

    label = subparsers.add_parser("label", help="Convert transcripts to strict JSON labels.")
    label.add_argument("--input", default=str(INTERMEDIATE_DIR / "indicvoices_intermediate.jsonl"))
    label.add_argument("--output", default=str(INTERMEDIATE_DIR / "indicvoices_labeled.jsonl"))
    label.add_argument("--language", choices=SUPPORTED_LANGUAGES, default=None)
    label.add_argument(
        "--provider",
        choices=["heuristic", "openai"],
        default=os.getenv("LABEL_PROVIDER", "heuristic"),
    )
    label.add_argument("--openai-model", default=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))
    label.add_argument(
        "--openai-batch-size",
        type=int,
        default=env_int("OPENAI_BATCH_SIZE", DEFAULT_OPENAI_BATCH_SIZE),
    )
    label.add_argument(
        "--translation-backend",
        choices=["nllb", "rule"],
        default=os.getenv("TRANSLATION_BACKEND", "nllb"),
    )
    label.add_argument("--max-retries", type=int, default=env_int("LABEL_MAX_RETRIES", 3))
    label.add_argument("--progress-every", type=int, default=env_int("PREP_PROGRESS_EVERY", 100))
    label.add_argument("--resume", action="store_true", default=env_bool("PREP_RESUME_LABELING", True))
    label.add_argument("--start-row", type=int, default=1)
    label.add_argument("--end-row", type=int, default=None)
    label.add_argument(
        "--skip-aksharantar",
        action="store_true",
        default=env_bool("SKIP_AKSHARANTAR", False),
    )

    validate = subparsers.add_parser("validate", help="Validate labeled JSON rows.")
    validate.add_argument("--input", default=str(INTERMEDIATE_DIR / "indicvoices_labeled.jsonl"))
    validate.add_argument("--language", choices=SUPPORTED_LANGUAGES, default=None)

    export = subparsers.add_parser("export", help="Export train/valid JSONL in training shape.")
    export.add_argument("--input", default=str(INTERMEDIATE_DIR / "indicvoices_labeled.jsonl"))
    export.add_argument("--language", choices=SUPPORTED_LANGUAGES, default=None)
    export.add_argument("--train", default=str(EXPORT_DIR / "train.jsonl"))
    export.add_argument("--valid", default=str(EXPORT_DIR / "valid.jsonl"))
    export.add_argument("--valid-ratio", type=float, default=env_float("VALID_RATIO", DEFAULT_VALID_RATIO))
    export.add_argument("--seed", type=int, default=env_int("PREP_SEED", 42))
    export.add_argument("--progress-every", type=int, default=env_int("PREP_PROGRESS_EVERY", 100))

    eval_parser = subparsers.add_parser("lid-eval", help="Optional language-id evaluation using Bhasha-Abhijnaanam.")
    eval_parser.add_argument("--output", default=str(INTERMEDIATE_DIR / "bhasha_lid_eval.jsonl"))
    eval_parser.add_argument("--max-rows-per-language", type=int, default=100)

    all_parser = subparsers.add_parser("all", help="Run ingest, label, validate, and export.")
    all_parser.add_argument("--intermediate", default=str(INTERMEDIATE_DIR / "indicvoices_intermediate.jsonl"))
    all_parser.add_argument("--audio-dir", default=str(INTERMEDIATE_DIR / "audio"))
    all_parser.add_argument("--labeled", default=str(INTERMEDIATE_DIR / "indicvoices_labeled.jsonl"))
    all_parser.add_argument("--language", choices=SUPPORTED_LANGUAGES, default=None)
    all_parser.add_argument("--train", default=str(EXPORT_DIR / "train.jsonl"))
    all_parser.add_argument("--valid", default=str(EXPORT_DIR / "valid.jsonl"))
    all_parser.add_argument("--split", default=os.getenv("PREP_SPLIT", "train"))
    all_parser.add_argument(
        "--max-samples-per-language",
        type=int,
        default=env_int("PREP_MAX_SAMPLES_PER_LANGUAGE"),
    )
    all_parser.add_argument("--cache-dir", default=None)
    all_parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    all_parser.add_argument("--streaming", action="store_true", default=env_bool("PREP_STREAMING", False))
    all_parser.add_argument(
        "--provider",
        choices=["heuristic", "openai"],
        default=os.getenv("LABEL_PROVIDER", "heuristic"),
    )
    all_parser.add_argument("--openai-model", default=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))
    all_parser.add_argument(
        "--openai-batch-size",
        type=int,
        default=env_int("OPENAI_BATCH_SIZE", DEFAULT_OPENAI_BATCH_SIZE),
    )
    all_parser.add_argument(
        "--translation-backend",
        choices=["nllb", "rule"],
        default=os.getenv("TRANSLATION_BACKEND", "nllb"),
    )
    all_parser.add_argument("--max-retries", type=int, default=env_int("LABEL_MAX_RETRIES", 3))
    all_parser.add_argument("--progress-every", type=int, default=env_int("PREP_PROGRESS_EVERY", 100))
    all_parser.add_argument("--resume", action="store_true", default=env_bool("PREP_RESUME_LABELING", True))
    all_parser.add_argument("--start-row", type=int, default=1)
    all_parser.add_argument("--end-row", type=int, default=None)
    all_parser.add_argument(
        "--skip-aksharantar",
        action="store_true",
        default=env_bool("SKIP_AKSHARANTAR", False),
    )
    all_parser.add_argument("--valid-ratio", type=float, default=env_float("VALID_RATIO", DEFAULT_VALID_RATIO))
    all_parser.add_argument("--seed", type=int, default=env_int("PREP_SEED", 42))
    all_parser.add_argument("--run-lid-eval", action="store_true", default=env_bool("RUN_LID_EVAL", False))

    return parser


def main() -> None:
    ensure_project_dirs()
    parser = build_parser()
    args = parser.parse_args()
    default_shared_intermediate = legacy_shared_intermediate_path()
    default_shared_labeled = legacy_shared_labeled_path()

    if hasattr(args, "language"):
        if hasattr(args, "output"):
            args.output = resolve_default_path(
                args.output,
                default_shared_intermediate if args.command == "ingest" else default_shared_labeled,
                default_intermediate_path(args.language) if args.command == "ingest" else default_labeled_path(args.language),
            )
        if hasattr(args, "input"):
            args.input = resolve_default_path(
                args.input,
                default_shared_labeled if args.command in {"validate", "export"} else default_shared_intermediate,
                default_labeled_path(args.language) if args.command in {"validate", "export"} else default_intermediate_path(args.language),
            )
        if hasattr(args, "intermediate"):
            args.intermediate = resolve_default_path(
                args.intermediate,
                default_shared_intermediate,
                default_intermediate_path(args.language),
            )
        if hasattr(args, "labeled"):
            args.labeled = resolve_default_path(
                args.labeled,
                default_shared_labeled,
                default_labeled_path(args.language),
            )

    if args.command == "ingest":
        build_intermediate_dataset(
            output_path=args.output,
            audio_dir=args.audio_dir,
            language_name=args.language,
            split=args.split,
            max_samples_per_language=args.max_samples_per_language,
            cache_dir=args.cache_dir,
            token=args.hf_token,
            streaming=args.streaming,
        )
        return

    if args.command == "label":
        label_intermediate_dataset(
            input_path=args.input,
            output_path=args.output,
            provider=args.provider,
            openai_model=args.openai_model,
            translation_backend=args.translation_backend,
            max_retries=args.max_retries,
            use_aksharantar=not args.skip_aksharantar,
            openai_batch_size=args.openai_batch_size,
            progress_every=args.progress_every,
            resume=args.resume,
            language_name=args.language,
            start_row=args.start_row,
            end_row=args.end_row,
        )
        return

    if args.command == "validate":
        rows = validate_labeled_dataset(args.input, language_name=args.language)
        print(json.dumps({"validated_rows": len(rows)}, indent=2))
        return

    if args.command == "export":
        export_input = None
        if args.language:
            export_input = args.input
        elif args.input != default_shared_labeled and Path(args.input).exists():
            export_input = args.input
        else:
            discovered = discover_language_labeled_paths()
            export_input = discovered if discovered else args.input
        export_training_jsonl(
            labeled_path=export_input,
            train_path=args.train,
            valid_path=args.valid,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
            progress_every=args.progress_every,
        )
        return

    if args.command == "lid-eval":
        result = evaluate_language_id_with_bhasha(
            output_path=args.output,
            max_rows_per_language=args.max_rows_per_language,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "all":
        build_intermediate_dataset(
            output_path=args.intermediate,
            audio_dir=args.audio_dir,
            language_name=args.language,
            split=args.split,
            max_samples_per_language=args.max_samples_per_language,
            cache_dir=args.cache_dir,
            token=args.hf_token,
            streaming=args.streaming,
        )
        label_intermediate_dataset(
            input_path=args.intermediate,
            output_path=args.labeled,
            provider=args.provider,
            openai_model=args.openai_model,
            translation_backend=args.translation_backend,
            max_retries=args.max_retries,
            use_aksharantar=not args.skip_aksharantar,
            openai_batch_size=args.openai_batch_size,
            progress_every=args.progress_every,
            resume=args.resume,
            language_name=args.language,
            start_row=args.start_row,
            end_row=args.end_row,
        )
        rows = validate_labeled_dataset(args.labeled, language_name=args.language)
        LOGGER.info("Validated %s labeled rows", len(rows))
        if args.language:
            LOGGER.info(
                "Completed per-language pipeline for %s. Run `python scripts/prepare_data.py export` after all language jobs finish.",
                args.language,
            )
            return
        export_input = args.labeled if args.language else (discover_language_labeled_paths() or args.labeled)
        export_training_jsonl(
            labeled_path=export_input,
            train_path=args.train,
            valid_path=args.valid,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
            progress_every=args.progress_every,
        )
        if args.run_lid_eval:
            scores = evaluate_language_id_with_bhasha(
                output_path=str(INTERMEDIATE_DIR / "bhasha_lid_eval.jsonl")
            )
            LOGGER.info("Bhasha-Abhijnaanam LID scores: %s", scores)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
