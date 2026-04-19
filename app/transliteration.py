from __future__ import annotations

import json
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from app.config import LANGUAGE_SPECS, TECHNICAL_TOKENS
from app.utils import normalize_whitespace, setup_logging

LOGGER = setup_logging("app.transliteration", "transliteration.log")

WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SCRIPT_RANGES = {
    "hinglish": ("\u0900", "\u097F"),
    "tanglish": ("\u0B80", "\u0BFF"),
    "banglish": ("\u0980", "\u09FF"),
}
SCRIPT_TO_SANSKRIT = {
    "hinglish": sanscript.DEVANAGARI,
    "tanglish": sanscript.TAMIL,
    "banglish": sanscript.BENGALI,
}


@dataclass
class AksharantarLexicon:
    native_to_roman: dict[str, str]
    roman_to_native: dict[str, str]

    @classmethod
    def empty(cls) -> "AksharantarLexicon":
        return cls(native_to_roman={}, roman_to_native={})


def is_english_or_technical(token: str) -> bool:
    lowered = token.lower()
    return (
        lowered in TECHNICAL_TOKENS
        or bool(re.fullmatch(r"[A-Za-z0-9_./-]+", token))
        or token.isascii()
    )


def normalize_roman_token(token: str) -> str:
    token = unicodedata.normalize("NFKD", token)
    token = "".join(character for character in token if not unicodedata.combining(character))
    token = token.lower().strip()
    token = re.sub(r"[^a-z0-9_./-]+", "", token)
    return token


def infer_label_from_script(text: str) -> str | None:
    for label, (start, end) in SCRIPT_RANGES.items():
        if any(start <= char <= end for char in text):
            return label
    return None


def infer_label_from_romanized(text: str) -> str:
    lowered = f" {text.lower()} "
    tamil_signals = (" irukku ", " illa ", " venum ", " pannu ", " appo ", " ippo ")
    bengali_signals = (" ami ", " tumi ", " ache ", " hobe ", " koro ", " kalke ")
    hindi_signals = (" hai ", " nahi ", " kya ", " kal ", " mera ", " karna ")
    if any(signal in lowered for signal in tamil_signals):
        return "tanglish"
    if any(signal in lowered for signal in bengali_signals):
        return "banglish"
    if any(signal in lowered for signal in hindi_signals):
        return "hinglish"
    return "hinglish"


def guess_label(text: str, lang_hint: str | None = None) -> str:
    if lang_hint:
        normalized = lang_hint.strip().lower()
        if normalized in LANGUAGE_SPECS:
            return LANGUAGE_SPECS[normalized].label
        if normalized in SCRIPT_RANGES:
            return normalized
    by_script = infer_label_from_script(text)
    if by_script:
        return by_script
    return infer_label_from_romanized(text)


@lru_cache(maxsize=8)
def load_aksharantar_lexicon(
    language_name: str,
    max_rows: int | None = 200_000,
    cache_dir: str | Path | None = None,
) -> AksharantarLexicon:
    if language_name not in LANGUAGE_SPECS:
        raise KeyError(f"Unsupported language for Aksharantar: {language_name}")

    spec = LANGUAGE_SPECS[language_name]
    native_to_roman: dict[str, str] = {}
    roman_to_native: dict[str, str] = {}

    try:
        archive_path = hf_hub_download(
            repo_id="ai4bharat/Aksharantar",
            filename=f"{spec.aksharantar_code}.zip",
            repo_type="dataset",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    except Exception as exc:
        LOGGER.warning("Failed to load Aksharantar data for %s: %s", spec.aksharantar_code, exc)
        return AksharantarLexicon.empty()

    with zipfile.ZipFile(archive_path) as archive:
        json_members = sorted(
            name
            for name in archive.namelist()
            if name.endswith(".json") and "train" in Path(name).name.lower()
        )
        if not json_members:
            json_members = sorted(name for name in archive.namelist() if name.endswith(".json"))

        for member in json_members:
            with archive.open(member) as handle:
                raw_text = handle.read().decode("utf-8")

            rows = _parse_aksharantar_rows(raw_text)

            for row in rows:
                if not isinstance(row, dict):
                    continue
                identifier = str(row.get("unique_identifier", ""))
                if identifier and not identifier.startswith(spec.aksharantar_code):
                    continue
                native = normalize_whitespace(str(row.get("native word", "")).strip())
                roman = normalize_whitespace(str(row.get("english word", "")).strip())
                if not native or not roman:
                    continue
                native_to_roman.setdefault(native, roman.lower())
                roman_to_native.setdefault(normalize_roman_token(roman), native)
                if max_rows is not None and len(native_to_roman) >= max_rows:
                    break

            if max_rows is not None and len(native_to_roman) >= max_rows:
                break

    return AksharantarLexicon(native_to_roman=native_to_roman, roman_to_native=roman_to_native)


def _parse_aksharantar_rows(raw_text: str) -> list[dict]:
    raw_text = raw_text.strip()
    if not raw_text:
        return []

    try:
        payload = json.loads(raw_text)
        if isinstance(payload, dict):
            return [row for row in payload.values() if isinstance(row, dict)]
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        return []
    except json.JSONDecodeError:
        rows: list[dict] = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows


def _fallback_transliterate_word(word: str, label: str) -> str:
    script = SCRIPT_TO_SANSKRIT.get(label)
    if not script:
        return word
    roman = transliterate(word, script, sanscript.ITRANS)
    roman = roman.replace("~N", "n").replace(".a", "a").replace("aa", "a")
    return normalize_roman_token(roman) or word.lower()


def romanize_tokens(
    tokens: Iterable[str],
    label: str,
    lexicon: AksharantarLexicon,
) -> list[str]:
    output: list[str] = []
    for token in tokens:
        if not token.strip():
            continue
        if is_english_or_technical(token) or not re.search(r"\w", token, re.UNICODE):
            output.append(token)
            continue
        roman = lexicon.native_to_roman.get(token)
        output.append((roman or _fallback_transliterate_word(token, label)).lower())
    return output


def nativeize_tokens(
    tokens: Iterable[str],
    language_name: str,
    lexicon: AksharantarLexicon,
) -> list[str]:
    output: list[str] = []
    label = LANGUAGE_SPECS[language_name].label
    for token in tokens:
        if not token.strip():
            continue
        if infer_label_from_script(token) == label or not re.search(r"\w", token, re.UNICODE):
            output.append(token)
            continue
        if is_english_or_technical(token):
            output.append(token)
            continue
        output.append(lexicon.roman_to_native.get(normalize_roman_token(token), token))
    return output


def _join_tokens(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    return normalize_whitespace(text)


def romanize_text(text: str, label: str, lexicon: AksharantarLexicon) -> str:
    tokens = WORD_RE.findall(text)
    return _join_tokens(romanize_tokens(tokens, label, lexicon))


def nativeize_text(text: str, language_name: str, lexicon: AksharantarLexicon) -> str:
    tokens = WORD_RE.findall(text)
    return _join_tokens(nativeize_tokens(tokens, language_name, lexicon))
