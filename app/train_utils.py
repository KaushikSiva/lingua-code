from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2AudioForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from app.config import TRAINING_INSTRUCTION
from app.schema import TrainingRecord
from app.utils import load_audio, read_jsonl, setup_logging

LOGGER = setup_logging("app.train_utils", "train_utils.log")


@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen2-Audio-7B-Instruct"
    output_dir: str = "artifacts/qwen2_audio_json"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 2
    max_new_tokens: int = 256
    use_lora: bool = True
    quantization: str = "4bit"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 42
    gradient_checkpointing: bool = True
    max_eval_samples: int = 2
    sample_rate: int = 16000
    resume_from_checkpoint: str | None = None
    trust_remote_code: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingConfig":
        allowed = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in payload.items() if key in allowed}
        return cls(**filtered)


class AudioJsonTrainingDataset(Dataset):
    def __init__(self, jsonl_path: str | Path) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.base_dir = self.jsonl_path.parent
        self.records = [TrainingRecord.model_validate(row).model_dump() for row in read_jsonl(jsonl_path)]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, str]:
        record = dict(self.records[index])
        audio_path = Path(record["audio"])
        if not audio_path.is_absolute():
            audio_path = (self.base_dir / audio_path).resolve()
        record["audio"] = str(audio_path)
        return record


class AudioJsonCollator:
    def __init__(
        self,
        processor,
        instruction: str = TRAINING_INSTRUCTION,
        sample_rate: int = 16000,
    ) -> None:
        self.processor = processor
        self.instruction = instruction
        self.sample_rate = sample_rate

    @staticmethod
    def _prompt_conversation(audio_path: str, instruction: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": instruction},
                ],
            }
        ]

    @classmethod
    def _full_conversation(
        cls,
        audio_path: str,
        instruction: str,
        target: str,
    ) -> list[dict]:
        return cls._prompt_conversation(audio_path, instruction) + [
            {"role": "assistant", "content": target}
        ]

    def __call__(self, features: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        audios = []
        prompt_texts: list[str] = []
        full_texts: list[str] = []

        for feature in features:
            audio_values, _ = load_audio(feature["audio"], sample_rate=self.sample_rate)
            audios.append(audio_values)
            prompt_conversation = self._prompt_conversation(feature["audio"], self.instruction)
            full_conversation = self._full_conversation(
                feature["audio"],
                self.instruction,
                feature["target"],
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    prompt_conversation,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )
            full_texts.append(
                self.processor.apply_chat_template(
                    full_conversation,
                    add_generation_prompt=False,
                    tokenize=False,
                )
            )

        model_inputs = self.processor(
            text=full_texts,
            audio=audios,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        prompt_inputs = self.processor(
            text=prompt_texts,
            audio=audios,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )

        labels = model_inputs["input_ids"].clone()
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        for index, prompt_length in enumerate(prompt_lengths):
            labels[index, : int(prompt_length)] = -100
        labels[model_inputs["attention_mask"] == 0] = -100
        model_inputs["labels"] = labels
        return model_inputs


def _can_quantize(mode: str) -> bool:
    return mode in {"4bit", "8bit"} and torch.cuda.is_available()


def load_model_and_processor(
    config: TrainingConfig,
    adapter_path: str | None = None,
):
    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
    )

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if _can_quantize(config.quantization):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=config.quantization == "4bit",
            load_in_8bit=config.quantization == "8bit",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif config.quantization != "none":
        LOGGER.warning(
            "Requested quantization=%s but CUDA is unavailable; falling back to full precision",
            config.quantization,
        )

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        config.model_name,
        **model_kwargs,
    )
    model.config.use_cache = False

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        return model, processor

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if config.use_lora:
        if _can_quantize(config.quantization):
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, processor


class SampleGenerationCallback(TrainerCallback):
    def __init__(
        self,
        processor,
        eval_dataset: AudioJsonTrainingDataset,
        instruction: str = TRAINING_INSTRUCTION,
        max_new_tokens: int = 256,
        max_samples: int = 2,
        sample_rate: int = 16000,
    ) -> None:
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.eval_samples = eval_dataset.records[:max_samples]
        self.instruction = instruction
        self.max_new_tokens = max_new_tokens
        self.sample_rate = sample_rate

    def _resolve_audio_path(self, audio_path: str) -> str:
        path = Path(audio_path)
        if not path.is_absolute():
            path = (self.eval_dataset.base_dir / path).resolve()
        return str(path)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        model.eval()
        for sample in self.eval_samples:
            resolved_audio_path = self._resolve_audio_path(sample["audio"])
            audio_values, _ = load_audio(resolved_audio_path, sample_rate=self.sample_rate)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": resolved_audio_path},
                        {"type": "text", "text": self.instruction},
                    ],
                }
            ]
            prompt_text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = self.processor(
                text=prompt_text,
                audio=[audio_values],
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(model.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            generated = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            generated = generated[:, inputs["input_ids"].size(1) :]
            decoded = self.processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            LOGGER.info(
                "Eval sample | audio=%s | target=%s | prediction=%s",
                resolved_audio_path,
                sample["target"],
                decoded,
            )


def train(
    train_file: str | Path,
    valid_file: str | Path | None,
    config: TrainingConfig,
) -> Trainer:
    train_dataset = AudioJsonTrainingDataset(train_file)
    valid_dataset = AudioJsonTrainingDataset(valid_file) if valid_file else None
    model, processor = load_model_and_processor(config)
    collator = AudioJsonCollator(
        processor=processor,
        sample_rate=config.sample_rate,
    )

    evaluation_strategy = "steps" if valid_dataset and len(valid_dataset) > 0 else "no"
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps if evaluation_strategy == "steps" else None,
        eval_strategy=evaluation_strategy,
        save_strategy="steps",
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
        fp16=False,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=evaluation_strategy == "steps",
        metric_for_best_model="eval_loss" if evaluation_strategy == "steps" else None,
        report_to=[],
        seed=config.seed,
    )

    callbacks: list[TrainerCallback] = []
    if valid_dataset and len(valid_dataset) > 0:
        callbacks.append(
            SampleGenerationCallback(
                processor=processor,
                eval_dataset=valid_dataset,
                max_new_tokens=config.max_new_tokens,
                max_samples=config.max_eval_samples,
                sample_rate=config.sample_rate,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)
    LOGGER.info("Training complete. Saved model artifacts to %s", config.output_dir)
    return trainer


def resolve_training_files(
    train_file: str | Path | None = None,
    valid_file: str | Path | None = None,
    hf_dataset_repo: str | None = None,
    hf_dataset_revision: str | None = None,
    hf_token: str | None = None,
) -> tuple[str, str | None]:
    if hf_dataset_repo:
        local_root = Path(
            snapshot_download(
                repo_id=hf_dataset_repo,
                repo_type="dataset",
                revision=hf_dataset_revision,
                token=hf_token,
                allow_patterns=["train.jsonl", "valid.jsonl", "audio/**"],
            )
        )
        resolved_train = local_root / "train.jsonl"
        resolved_valid = local_root / "valid.jsonl"
        if not resolved_train.exists():
            raise FileNotFoundError(f"Downloaded dataset repo {hf_dataset_repo} is missing train.jsonl")
        LOGGER.info("Resolved training data from dataset repo %s into %s", hf_dataset_repo, local_root)
        return str(resolved_train), str(resolved_valid) if resolved_valid.exists() else None

    if train_file is None:
        raise ValueError("train_file is required unless hf_dataset_repo is provided")
    resolved_train = Path(train_file)
    resolved_valid = Path(valid_file) if valid_file else None
    if not resolved_train.exists():
        raise FileNotFoundError(f"Training file not found: {resolved_train}")
    if resolved_valid and not resolved_valid.exists():
        raise FileNotFoundError(f"Validation file not found: {resolved_valid}")
    return str(resolved_train), str(resolved_valid) if resolved_valid else None


def training_data_defaults() -> tuple[str | None, str | None, str | None]:
    return (
        os.getenv("HF_DATASET_REPO"),
        os.getenv("HF_DATASET_REVISION"),
        os.getenv("HF_TOKEN"),
    )
