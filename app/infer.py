from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from peft import PeftModel

from app.config import DIRECT_INFERENCE_INSTRUCTION, OUTPUT_DIR
from app.label_transcript import TranscriptLabeler
from app.schema import TranscriptLabel, validate_json_output
from app.train_utils import TrainingConfig, load_model_and_processor
from app.transcribe import WhisperTranscriber
from app.utils import exact_json_from_text, load_audio, setup_logging

LOGGER = setup_logging("app.infer", "infer.log")


@dataclass
class BaselineInferenceEngine:
    transcriber: WhisperTranscriber
    labeler: TranscriptLabeler

    def run(self, audio_path: str) -> TranscriptLabel:
        transcript = self.transcriber.transcribe(audio_path)
        return self.labeler.label_transcript(transcript)


@dataclass
class DirectAudioJsonEngine:
    model_name: str = "Qwen/Qwen2-Audio-7B-Instruct"
    adapter_path: str | None = None
    sample_rate: int = 16000
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        config = TrainingConfig(model_name=self.model_name, use_lora=False, quantization="none")
        self.model, self.processor = load_model_and_processor(config, adapter_path=self.adapter_path)

    def run(self, audio_path: str) -> TranscriptLabel:
        audio_values, _ = load_audio(audio_path, sample_rate=self.sample_rate)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": DIRECT_INFERENCE_INSTRUCTION},
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
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated = generated[:, inputs["input_ids"].size(1) :]
        decoded = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        json_text = exact_json_from_text(decoded)
        if not json_text:
            raise ValueError(f"Direct model returned non-JSON output: {decoded}")
        return validate_json_output(json_text)


@dataclass
class SpeechJsonInferenceService:
    mode: str = "auto"
    transcriber_model: str = "openai/whisper-small"
    label_provider: str = "heuristic"
    translation_backend: str = "nllb"
    direct_model_name: str = "Qwen/Qwen2-Audio-7B-Instruct"
    adapter_path: str | None = None

    def __post_init__(self) -> None:
        self._baseline = BaselineInferenceEngine(
            transcriber=WhisperTranscriber(model_name=self.transcriber_model),
            labeler=TranscriptLabeler.from_defaults(
                provider=self.label_provider,
                translation_backend=self.translation_backend,
            ),
        )
        self._direct = None
        if self.mode in {"auto", "direct"}:
            try:
                self._direct = DirectAudioJsonEngine(
                    model_name=self.direct_model_name,
                    adapter_path=self.adapter_path,
                )
            except Exception as exc:
                LOGGER.warning("Failed to initialize direct audio->JSON model: %s", exc)
                if self.mode == "direct":
                    raise

    def run(self, audio_path: str) -> TranscriptLabel:
        if self.mode == "baseline":
            return self._baseline.run(audio_path)
        if self.mode == "direct" and self._direct is not None:
            return self._direct.run(audio_path)
        if self.mode == "auto" and self._direct is not None:
            try:
                return self._direct.run(audio_path)
            except Exception as exc:
                LOGGER.warning("Direct inference failed for %s, falling back to baseline: %s", audio_path, exc)
        return self._baseline.run(audio_path)


def save_inference_output(
    label: TranscriptLabel,
    audio_path: str,
    output_dir: str | Path = OUTPUT_DIR,
) -> Path:
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(audio_path).stem
    output_path = destination_dir / f"{stem}.json"
    output_path.write_text(
        json.dumps(label.ordered_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
