from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import EXPORT_DIR, STAGING_DIR, ensure_project_dirs
from app.dataset_builder import (
    resolve_default_dataset_repo,
    stage_training_dataset_for_hub,
    upload_staged_dataset_to_hub,
)
from app.utils import env_bool, setup_logging

LOGGER = setup_logging("scripts.upload_dataset", "upload_dataset.log")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage and upload the training dataset to Hugging Face.")
    parser.add_argument("--train", default=str(EXPORT_DIR / "train.jsonl"))
    parser.add_argument("--valid", default=str(EXPORT_DIR / "valid.jsonl"))
    parser.add_argument("--staging-dir", default=str(STAGING_DIR))
    parser.add_argument("--repo-id", default=os.getenv("HF_DATASET_REPO"))
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--revision", default=os.getenv("HF_DATASET_REVISION"))
    parser.add_argument("--private", action="store_true", default=env_bool("HF_DATASET_PRIVATE", True))
    parser.add_argument("--public", action="store_true", help="Override private upload and publish the dataset repo.")
    parser.add_argument("--commit-message", default="Upload Linguacode audio JSON dataset")
    parser.add_argument("--stage-only", action="store_true")
    return parser


def main() -> None:
    ensure_project_dirs()
    parser = build_parser()
    args = parser.parse_args()
    private = False if args.public else args.private
    repo_id = args.repo_id or resolve_default_dataset_repo(args.hf_token)

    stats = stage_training_dataset_for_hub(
        train_path=args.train,
        valid_path=args.valid,
        staging_dir=args.staging_dir,
        repo_id=repo_id,
    )
    LOGGER.info("Dataset stats: %s", stats)

    if args.stage_only:
        print(json.dumps({"repo_id": repo_id, "staging_dir": args.staging_dir, "stats": stats}, indent=2))
        return

    uploaded_repo = upload_staged_dataset_to_hub(
        staging_dir=args.staging_dir,
        token=args.hf_token,
        repo_id=repo_id,
        private=private,
        commit_message=args.commit_message,
    )
    print(json.dumps({"repo_id": uploaded_repo, "staging_dir": args.staging_dir, "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
