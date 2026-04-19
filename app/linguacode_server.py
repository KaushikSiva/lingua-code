from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from app.utils import bootstrap, env_bool, env_float, setup_logging

bootstrap()
LOGGER = setup_logging("app.linguacode_server", "linguacode_server.log")

ROOT = Path(__file__).resolve().parents[1]
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value


@dataclass
class ClientRunOptions:
    base_url: str = field(default_factory=lambda: _env_str("TEST_SERVER_BASE_URL", "http://127.0.0.1:8000"))
    endpoint: str = field(default_factory=lambda: _env_str("TEST_SERVER_ENDPOINT", "/infer"))
    audio_root: str = field(default_factory=lambda: _env_str("TEST_SERVER_AUDIO_ROOT", "data/hf_stage/audio"))
    audio: str | None = None
    seed: int | None = None
    postbin_url: str | None = field(default_factory=lambda: os.getenv("TEST_SERVER_POSTBIN_URL"))
    adapter_path: str | None = field(default_factory=lambda: os.getenv("TEST_SERVER_ADAPTER_PATH"))
    mode: str = field(default_factory=lambda: _env_str("TEST_SERVER_MODE", "direct"))
    timeout: float = field(default_factory=lambda: env_float("TEST_SERVER_TIMEOUT", 300.0))
    in_process: bool = field(default_factory=lambda: env_bool("TEST_SERVER_IN_PROCESS", False))
    run_codex: bool = field(default_factory=lambda: env_bool("TEST_SERVER_RUN_CODEX", False))
    codex_mode: str = field(default_factory=lambda: _env_str("TEST_SERVER_CODEX_MODE", "interactive"))
    mock_postbin: bool = field(default_factory=lambda: env_bool("TEST_SERVER_MOCK_POSTBIN", False))


@dataclass
class ClientRunResult:
    selected_audio: str
    status_code: int
    payload: dict[str, Any] | None
    raw_text: str
    codex_executed: bool = False


def find_audio_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)


def choose_audio_file(audio: str | None, audio_root: str, seed: int | None) -> Path:
    if audio:
        path = Path(audio).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return path

    root = Path(audio_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Audio root not found: {root}")
    files = find_audio_files(root)
    if not files:
        raise FileNotFoundError(f"No audio files found under: {root}")

    rng = random.Random(seed)
    return rng.choice(files)


def _request_in_process(options: ClientRunOptions, audio_path: Path):
    LOGGER.info("Calling in-process inference | audio=%s | mode=%s | endpoint=%s", audio_path, options.mode, options.endpoint)
    from fastapi.testclient import TestClient
    import app.server as server_module

    if options.mock_postbin:
        server_module._rewrite_to_codex_prompt = lambda clean_english, openai_model: clean_english

        def fake_post_result(postbin_url: str, payload: dict[str, object]) -> tuple[int, str]:
            body = json.dumps({"mock_postbin_url": postbin_url, "received": payload}, ensure_ascii=False)
            return 200, body

        server_module._post_result = fake_post_result

    data = {"mode": options.mode}
    if options.postbin_url:
        data["postbin_url"] = options.postbin_url
    if options.adapter_path:
        data["adapter_path"] = options.adapter_path

    with audio_path.open("rb") as handle:
        files = {"file": (audio_path.name, handle, "audio/wav")}
        client = TestClient(server_module.app)
        return client.post(
            options.endpoint,
            data=data,
            files=files,
        )


def _request_over_http(options: ClientRunOptions, audio_path: Path):
    url = options.base_url.rstrip("/") + options.endpoint
    LOGGER.info("Calling Brev inference | url=%s | audio=%s | mode=%s", url, audio_path, options.mode)
    data = {"mode": options.mode}
    if options.postbin_url:
        data["postbin_url"] = options.postbin_url
    if options.adapter_path:
        data["adapter_path"] = options.adapter_path

    with audio_path.open("rb") as handle:
        files = {"file": (audio_path.name, handle, "audio/wav")}
        return httpx.post(
            url,
            data=data,
            files=files,
            timeout=options.timeout,
            follow_redirects=True,
        )


def run_codex_prompt(payload: dict[str, Any], mode: str, *, allow_interactive_replace: bool) -> bool:
    codex_prompt = payload.get("codex_prompt")
    if not isinstance(codex_prompt, str) or not codex_prompt.strip():
        raise RuntimeError("Response did not include a usable codex_prompt")

    codex_bin = shutil.which("codex")
    if not codex_bin:
        raise RuntimeError("codex CLI was not found on PATH")

    if mode == "interactive":
        if not allow_interactive_replace:
            raise RuntimeError("Interactive Codex mode is not supported from the API server; use codex_mode=exec")
        LOGGER.info("Calling Codex CLI | mode=interactive")
        os.execvp(codex_bin, [codex_bin, codex_prompt])

    LOGGER.info("Calling Codex CLI | mode=exec")
    subprocess.run(
        [codex_bin, "exec", "-C", str(ROOT), codex_prompt],
        check=True,
    )
    return True


def execute_test_server(options: ClientRunOptions, *, allow_interactive_replace: bool = True) -> ClientRunResult:
    audio_path = choose_audio_file(options.audio, options.audio_root, options.seed)
    LOGGER.info("Prepared client run | audio=%s | mode=%s | run_codex=%s", audio_path, options.mode, options.run_codex)
    if options.in_process:
        response = _request_in_process(options, audio_path)
    else:
        response = _request_over_http(options, audio_path)

    try:
        payload = response.json()
        raw_text = json.dumps(payload, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        payload = None
        raw_text = response.text

    response.raise_for_status()
    LOGGER.info("Inference request completed | status=%s | audio=%s", response.status_code, audio_path)

    codex_executed = False
    if options.run_codex:
        if not isinstance(payload, dict):
            raise RuntimeError("Server response is not JSON; cannot run Codex prompt")
        codex_executed = run_codex_prompt(payload, options.codex_mode, allow_interactive_replace=allow_interactive_replace)
        LOGGER.info("Codex CLI completed | executed=%s | audio=%s", codex_executed, audio_path)

    return ClientRunResult(
        selected_audio=str(audio_path),
        status_code=response.status_code,
        payload=payload,
        raw_text=raw_text,
        codex_executed=codex_executed,
    )
