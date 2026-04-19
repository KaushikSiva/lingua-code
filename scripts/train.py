from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.train_utils import TrainingConfig, resolve_training_files, train, training_data_defaults
from app.utils import setup_logging

LOGGER = setup_logging("scripts.train", "train.log")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-Audio on audio->JSON data.")
    default_repo, default_revision, default_token = training_data_defaults()
    parser.add_argument("--train", default=None, help="Training JSONL file")
    parser.add_argument("--valid", default=None, help="Validation JSONL file")
    parser.add_argument("--hf-dataset-repo", default=default_repo, help="Hugging Face dataset repo id")
    parser.add_argument("--hf-dataset-revision", default=default_revision, help="Optional dataset revision")
    parser.add_argument("--hf-token", default=default_token or os.getenv("HF_TOKEN"), help="Hugging Face token")
    parser.add_argument("--config", default="configs/train.yaml", help="Training config YAML")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser


def load_config(config_path: str | Path) -> TrainingConfig:
    path = Path(config_path)
    if not path.exists():
        return TrainingConfig()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return TrainingConfig.from_dict(data)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.output_dir:
        config.output_dir = args.output_dir
    if args.model_name:
        config.model_name = args.model_name
    if args.quantization:
        config.quantization = args.quantization
    if args.epochs is not None:
        config.num_train_epochs = args.epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate

    train_file, valid_file = resolve_training_files(
        train_file=args.train,
        valid_file=args.valid,
        hf_dataset_repo=args.hf_dataset_repo,
        hf_dataset_revision=args.hf_dataset_revision,
        hf_token=args.hf_token,
    )
    LOGGER.info("Starting training with config: %s", config)
    train(train_file=train_file, valid_file=valid_file, config=config)


if __name__ == "__main__":
    main()
