from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from app.config import DEFAULT_OPENAI_BATCH_SIZE, DEFAULT_OPENAI_MODEL, LABEL_TO_LANGUAGE, LANGUAGE_SPECS, TECHNICAL_TOKENS
from app.schema import EXPECTED_KEY_ORDER, TranscriptLabel, validate_json_output
from app.transliteration import (
    AksharantarLexicon,
    guess_label,
    load_aksharantar_lexicon,
    nativeize_text,
    romanize_text,
)
from app.utils import append_jsonl, env_int, exact_json_from_text, load_prompt_template, normalize_whitespace, setup_logging

LOGGER = setup_logging("app.label_transcript", "label_transcript.log")


class Translator(Protocol):
    def translate(self, text: str, source_label: str) -> str:
        ...


class RuleBasedTranslator:
    PHRASE_MAP = {
        ("hinglish", "कल मीटिंग है"): "There is a meeting tomorrow.",
        (
            "hinglish",
            "लॉगिन के समय user का last_seen अपडेट करो",
        ): "Update the user's last_seen field during login.",
        ("tanglish", "நாளைக்கு meeting இருக்கு"): "There is a meeting tomorrow.",
        ("tanglish", "deploy பண்ணும் முன் test ஓட்டு"): "Run the tests before deploying.",
        ("banglish", "কাল meeting আছে"): "There is a meeting tomorrow.",
        ("banglish", "deploy এর আগে test চালাও"): "Run the tests before deploying.",
    }

    WORD_MAP = {
        "कल": "tomorrow",
        "मीटिंग": "meeting",
        "है": "there is",
        "நாளைக்கு": "tomorrow",
        "இருக்கு": "there is",
        "কাল": "tomorrow",
        "আছে": "there is",
    }

    def translate(self, text: str, source_label: str) -> str:
        normalized = normalize_whitespace(text)
        if (source_label, normalized) in self.PHRASE_MAP:
            return self.PHRASE_MAP[(source_label, normalized)]

        translated_tokens = [self.WORD_MAP.get(token, token) for token in normalized.split()]
        fallback = " ".join(translated_tokens)
        if fallback == normalized:
            return f"English translation: {normalized}"
        return fallback[:1].upper() + fallback[1:] + "."


class NLLBTranslator:
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str | int | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._pipeline = None
        self._fallback = RuleBasedTranslator()

    def _get_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "translation",
                model=self.model_name,
                device=self.device,
            )
        return self._pipeline

    def translate(self, text: str, source_label: str) -> str:
        source_language = LABEL_TO_LANGUAGE[source_label]
        source_code = LANGUAGE_SPECS[source_language].nllb_code
        try:
            pipe = self._get_pipeline()
            result = pipe(
                text,
                src_lang=source_code,
                tgt_lang="eng_Latn",
                max_new_tokens=128,
            )
            if result and result[0].get("translation_text"):
                return normalize_whitespace(str(result[0]["translation_text"]))
        except Exception as exc:
            LOGGER.warning("NLLB translation failed for %s: %s", source_label, exc)
        return self._fallback.translate(text, source_label)


class OpenAIJsonLabeler:
    def __init__(self, model: str = DEFAULT_OPENAI_MODEL, heartbeat_seconds: int = 30) -> None:
        self.model = model
        self.heartbeat_seconds = heartbeat_seconds
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required for provider=openai")
            self._client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
        return self._client

    def _create_response(self, prompt: str, log_context: str | None = None) -> str:
        client = self._get_client()
        stop_event = threading.Event()
        started_at = time.monotonic()

        def _heartbeat() -> None:
            while not stop_event.wait(self.heartbeat_seconds):
                elapsed = format_duration(time.monotonic() - started_at)
                message = f"OpenAI request waiting | elapsed {elapsed}"
                if log_context:
                    message += f" | {log_context}"
                LOGGER.info(message)

        heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat_thread.start()
        try:
            response = client.responses.create(
                model=self.model,
                input=prompt,
            )
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=0.1)
        return response.output_text

    def label(
        self,
        transcript: str,
        prompt_template: str,
        lang_hint: str | None,
    ) -> str:
        prompt = prompt_template.format(transcript=transcript, lang_hint=lang_hint or "auto")
        return self._create_response(prompt)

    def label_batch(
        self,
        transcripts: list[str],
        prompt_template: str,
        lang_hints: list[str | None] | None = None,
        batch_start: int | None = None,
        batch_end: int | None = None,
        attempt: int | None = None,
        log_prefix: str | None = None,
    ) -> list[dict]:
        lang_hints = lang_hints or [None] * len(transcripts)
        batch_items = [
            {
                "id": index,
                "lang_hint": lang_hint or "auto",
                "transcript": transcript,
            }
            for index, (transcript, lang_hint) in enumerate(zip(transcripts, lang_hints))
        ]
        prompt = (
            "You convert transcripts into strict JSON labels.\n"
            "Follow the same single-item schema and language rules below.\n\n"
            f"{prompt_template}\n\n"
            "Now process a batch.\n"
            "Return JSON only.\n"
            "Return a JSON array.\n"
            "Each array item must be an object with exactly two keys in this order:\n"
            "id, label\n"
            "The label value must itself be a JSON object with exactly these keys in this order:\n"
            f"{', '.join(EXPECTED_KEY_ORDER)}\n"
            "Preserve array order and use the numeric id values provided.\n"
            f"Batch input:\n{json.dumps(batch_items, ensure_ascii=False)}"
        )
        context_parts = []
        if batch_start is not None and batch_end is not None:
            context_parts.append(f"rows {batch_start}-{batch_end}")
        context_parts.append(f"size={len(transcripts)}")
        if attempt is not None:
            context_parts.append(f"attempt={attempt}")
        if log_prefix:
            context_parts.insert(0, log_prefix)
        log_context = " | ".join(context_parts) if context_parts else None
        raw_text = self._create_response(prompt, log_context=log_context)
        json_text = extract_top_level_json(raw_text, prefer_array=True)
        if not json_text:
            raise ValueError("No JSON array found in OpenAI batch response")
        payload = json.loads(json_text)
        if not isinstance(payload, list):
            raise ValueError("OpenAI batch response must be a JSON array")
        ordered: list[dict | None] = [None] * len(transcripts)
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("OpenAI batch items must be objects")
            if list(item.keys()) != ["id", "label"]:
                raise ValueError("OpenAI batch items must have keys ['id', 'label']")
            index = item["id"]
            if not isinstance(index, int) or not 0 <= index < len(transcripts):
                raise ValueError(f"Invalid batch item id: {index}")
            ordered[index] = validate_json_output(item["label"]).ordered_dict()
        if any(item is None for item in ordered):
            raise ValueError("OpenAI batch response omitted one or more transcripts")
        return [item for item in ordered if item is not None]


@dataclass
class TranscriptLabeler:
    prompt_template: str
    translator: Translator
    provider: str = "heuristic"
    max_retries: int = 3
    failure_log_path: str = "logs/label_failures.jsonl"
    openai_model: str = DEFAULT_OPENAI_MODEL
    use_aksharantar: bool = True
    openai_batch_size: int = DEFAULT_OPENAI_BATCH_SIZE
    openai_heartbeat_seconds: int = 30

    def __post_init__(self) -> None:
        self._openai_labeler = (
            OpenAIJsonLabeler(
                model=self.openai_model,
                heartbeat_seconds=self.openai_heartbeat_seconds,
            )
            if self.provider == "openai"
            else None
        )

    @classmethod
    def from_defaults(
        cls,
        provider: str = "heuristic",
        max_retries: int = 3,
        translation_backend: str = "nllb",
        openai_model: str = DEFAULT_OPENAI_MODEL,
        use_aksharantar: bool = True,
        openai_batch_size: int = DEFAULT_OPENAI_BATCH_SIZE,
        openai_heartbeat_seconds: int = env_int("PREP_OPENAI_HEARTBEAT_SECONDS", 30) or 30,
    ) -> "TranscriptLabeler":
        translator: Translator
        if translation_backend == "rule":
            translator = RuleBasedTranslator()
        else:
            translator = NLLBTranslator()
        return cls(
            prompt_template=load_prompt_template(),
            translator=translator,
            provider=provider,
            max_retries=max_retries,
            openai_model=openai_model,
            use_aksharantar=use_aksharantar,
            openai_batch_size=openai_batch_size,
            openai_heartbeat_seconds=openai_heartbeat_seconds,
        )

    def label_transcript(
        self,
        transcript: str,
        lang_hint: str | None = None,
    ) -> TranscriptLabel:
        errors: list[str] = []
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.provider == "openai":
                    if self._openai_labeler is None:
                        raise RuntimeError("OpenAI labeler was not initialized")
                    raw_response = self._openai_labeler.label(
                        transcript=transcript,
                        prompt_template=self.prompt_template,
                        lang_hint=lang_hint,
                    )
                    json_text = exact_json_from_text(raw_response)
                    if not json_text:
                        raise ValueError("No JSON object found in model response")
                    return validate_json_output(json_text)
                return self._heuristic_label(transcript=transcript, lang_hint=lang_hint)
            except Exception as exc:
                errors.append(str(exc))
                LOGGER.warning("Labeling attempt %s failed: %s", attempt, exc)

        append_jsonl(
            self.failure_log_path,
            {"transcript": transcript, "lang_hint": lang_hint, "errors": errors},
        )
        raise ValueError(f"Failed to label transcript after {self.max_retries} attempts: {errors}")

    def label_transcripts(
        self,
        transcripts: list[str],
        lang_hints: list[str | None] | None = None,
        batch_start: int | None = None,
        batch_end: int | None = None,
        log_prefix: str | None = None,
    ) -> list[TranscriptLabel]:
        if not transcripts:
            return []
        if self.provider != "openai":
            return [
                self.label_transcript(transcript=transcript, lang_hint=(lang_hints or [None] * len(transcripts))[index])
                for index, transcript in enumerate(transcripts)
            ]

        if self._openai_labeler is None:
            raise RuntimeError("OpenAI labeler was not initialized")

        lang_hints = lang_hints or [None] * len(transcripts)
        errors: list[str] = []
        for attempt in range(1, self.max_retries + 1):
            started_at = time.monotonic()
            if batch_start is not None and batch_end is not None:
                LOGGER.info(
                    "%s batch start | rows %s-%s | size=%s | attempt=%s",
                    log_prefix or "label",
                    batch_start,
                    batch_end,
                    len(transcripts),
                    attempt,
                )
            try:
                payloads = self._openai_labeler.label_batch(
                    transcripts=transcripts,
                    prompt_template=self.prompt_template,
                    lang_hints=lang_hints,
                    batch_start=batch_start,
                    batch_end=batch_end,
                    attempt=attempt,
                    log_prefix=log_prefix,
                )
                LOGGER.info(
                    "%s batch complete | rows %s-%s | size=%s | attempt=%s | elapsed %s",
                    log_prefix or "label",
                    batch_start,
                    batch_end,
                    len(transcripts),
                    attempt,
                    format_duration(time.monotonic() - started_at),
                )
                return [validate_json_output(payload) for payload in payloads]
            except Exception as exc:
                errors.append(str(exc))
                LOGGER.warning(
                    "%s batch failed | rows %s-%s | size=%s | attempt=%s | elapsed %s | error=%s",
                    log_prefix or "label",
                    batch_start,
                    batch_end,
                    len(transcripts),
                    attempt,
                    format_duration(time.monotonic() - started_at),
                    exc,
                )

        for transcript, lang_hint in zip(transcripts, lang_hints):
            append_jsonl(
                self.failure_log_path,
                {"transcript": transcript, "lang_hint": lang_hint, "errors": errors, "mode": "batch"},
            )
        raise ValueError(
            f"Failed to batch-label {len(transcripts)} transcripts after {self.max_retries} attempts: {errors}"
        )

    def _heuristic_label(
        self,
        transcript: str,
        lang_hint: str | None = None,
    ) -> TranscriptLabel:
        transcript = normalize_whitespace(transcript)
        label = guess_label(transcript, lang_hint=lang_hint)
        language_name = LABEL_TO_LANGUAGE[label]
        lexicon = (
            load_aksharantar_lexicon(language_name)
            if self.use_aksharantar
            else AksharantarLexicon.empty()
        )

        if any(token in transcript for token in TECHNICAL_TOKENS):
            lang_hint = lang_hint or language_name

        clean_native = transcript
        if all(ord(character) < 128 for character in transcript if character.strip()):
            clean_native = nativeize_text(transcript, language_name=language_name, lexicon=lexicon)

        raw_mixed = transcript
        if transcript == clean_native:
            raw_mixed = romanize_text(clean_native, label=label, lexicon=lexicon)
        else:
            raw_mixed = normalize_whitespace(transcript.lower())

        clean_english = self.translator.translate(clean_native, label)

        payload = {
            "lang": label,
            "raw_mixed": normalize_whitespace(raw_mixed),
            "clean_native": normalize_whitespace(clean_native),
            "clean_english": normalize_whitespace(clean_english),
        }
        return validate_json_output(json.dumps(payload, ensure_ascii=False))


def extract_top_level_json(text: str, prefer_array: bool = False) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()

    if prefer_array:
        array_text = _extract_balanced_block(stripped, "[", "]")
        if array_text:
            return array_text

    object_text = exact_json_from_text(stripped)
    if object_text:
        return object_text

    if not prefer_array:
        return _extract_balanced_block(stripped, "[", "]")
    return None


def _extract_balanced_block(text: str, opener: str, closer: str) -> str | None:
    start = text.find(opener)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def format_duration(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
