from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
EXPORT_DIR = DATA_DIR / "exports"
OUTPUT_DIR = DATA_DIR / "output"
STAGING_DIR = DATA_DIR / "hf_stage"
SAMPLES_DIR = DATA_DIR / "samples"
PROMPTS_DIR = ROOT_DIR / "prompts"
LOG_DIR = ROOT_DIR / "logs"
CONFIGS_DIR = ROOT_DIR / "configs"

TRANSCRIPT_PROMPT_PATH = PROMPTS_DIR / "transcript_to_json.txt"

SUPPORTED_LABELS = ("hinglish", "tanglish", "banglish")
TECHNICAL_TOKENS = {
    "login",
    "endpoint",
    "api",
    "retry",
    "async",
    "bug",
    "function",
    "file",
    "test",
    "deploy",
    "commit",
    "branch",
    "pr",
    "user",
    "last_seen",
}

TRAINING_INSTRUCTION = (
    "Listen to the audio and return strict JSON with exactly these keys in this "
    "order: lang, raw_mixed, clean_native, clean_english. Output JSON only."
)

DIRECT_INFERENCE_INSTRUCTION = TRAINING_INSTRUCTION
DEFAULT_HF_DATASET_REPO_NAME = "linguacode-audio-json-v1"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_OPENAI_BATCH_SIZE = 16
DEFAULT_SAMPLES_PER_LANGUAGE = 15_000
DEFAULT_VALID_RATIO = 0.1


@dataclass(frozen=True)
class LanguageSpec:
    label: str
    hf_name: str
    aksharantar_code: str
    bhasha_code: str
    nllb_code: str
    script_name: str


LANGUAGE_SPECS: dict[str, LanguageSpec] = {
    "hindi": LanguageSpec(
        label="hinglish",
        hf_name="Hindi",
        aksharantar_code="hin",
        bhasha_code="hin",
        nllb_code="hin_Deva",
        script_name="devanagari",
    ),
    "tamil": LanguageSpec(
        label="tanglish",
        hf_name="Tamil",
        aksharantar_code="tam",
        bhasha_code="tam",
        nllb_code="tam_Taml",
        script_name="tamil",
    ),
    "bengali": LanguageSpec(
        label="banglish",
        hf_name="Bengali",
        aksharantar_code="ben",
        bhasha_code="ben",
        nllb_code="ben_Beng",
        script_name="bengali",
    ),
}

LABEL_TO_LANGUAGE = {spec.label: name for name, spec in LANGUAGE_SPECS.items()}
HF_LANGUAGE_NAMES = {spec.hf_name for spec in LANGUAGE_SPECS.values()}
SUPPORTED_LANGUAGES = tuple(LANGUAGE_SPECS.keys())


def ensure_project_dirs() -> None:
    for directory in (
        DATA_DIR,
        INTERMEDIATE_DIR,
        EXPORT_DIR,
        OUTPUT_DIR,
        STAGING_DIR,
        SAMPLES_DIR,
        PROMPTS_DIR,
        LOG_DIR,
        CONFIGS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
