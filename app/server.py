from __future__ import annotations

import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from app.config import DEFAULT_OPENAI_MODEL
from app.schema import TranscriptLabel, validate_json_output
from app.utils import bootstrap, setup_logging, slugify_filename

bootstrap()
LOGGER = setup_logging("app.server", "server.log")

app = FastAPI(title="Linguacode Inference Server", version="0.1.0")


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str | None = None
    source_url: str | None = None
    s3_url: str | None = None
    postbin_url: str | None = None
    supabase_table: str | None = None
    supabase_id_column: str = "id"
    supabase_url_column: str = "url"
    mode: str = "direct"
    adapter_path: str | None = None
    direct_model_name: str | None = None
    transcriber_model: str = "openai/whisper-small"
    label_provider: str = "heuristic"
    translation_backend: str = "rule"
    openai_model: str = Field(default=DEFAULT_OPENAI_MODEL)


class InferenceResponse(BaseModel):
    audio_source: str
    transcript_label: dict[str, Any]
    codex_prompt: str
    postbin_url: str | None = None
    postbin_status: int | None = None
    postbin_response: str | None = None


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value


def _coerce_request_value(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


async def _parse_request(request: Request) -> tuple[InferenceRequest, UploadFile | None]:
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        if upload is not None and not hasattr(upload, "read"):
            raise HTTPException(status_code=400, detail="Invalid uploaded file")
        payload = {
            "source_id": form.get("source_id"),
            "source_url": form.get("source_url"),
            "s3_url": form.get("s3_url"),
            "postbin_url": form.get("postbin_url"),
            "supabase_table": form.get("supabase_table"),
            "supabase_id_column": form.get("supabase_id_column", "id"),
            "supabase_url_column": form.get("supabase_url_column", "url"),
            "mode": form.get("mode", "direct"),
            "adapter_path": form.get("adapter_path"),
            "direct_model_name": form.get("direct_model_name"),
            "transcriber_model": form.get("transcriber_model", "openai/whisper-small"),
            "label_provider": form.get("label_provider", "heuristic"),
            "translation_backend": form.get("translation_backend", "rule"),
            "openai_model": form.get("openai_model", DEFAULT_OPENAI_MODEL),
        }
        normalized = {key: _coerce_request_value(payload, key) for key in payload}
        return InferenceRequest.model_validate(normalized), upload

    if "application/json" in content_type:
        payload = await request.json()
        return InferenceRequest.model_validate(payload), None

    raw_body = (await request.body()).decode("utf-8").strip()
    if raw_body:
        parsed = parse_qs(raw_body)
        payload = {key: values[-1] for key, values in parsed.items()}
        normalized = {key: _coerce_request_value(payload, key) for key in payload}
        return InferenceRequest.model_validate(normalized), None
    raise HTTPException(status_code=400, detail="Request must be JSON, multipart form-data, or form-encoded")


def _supabase_headers(service_key: str) -> dict[str, str]:
    return {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
    }


def _resolve_supabase_audio_url(payload: InferenceRequest) -> str:
    supabase_url = _get_env("SUPABASE_URL")
    service_key = _get_env("SUPABASE_SERVICE_ROLE_KEY")
    table = payload.supabase_table or _get_env("SUPABASE_TABLE")
    if not supabase_url or not service_key or not table:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, and SUPABASE_TABLE are required for source_id lookups",
        )
    if not payload.source_id:
        raise HTTPException(status_code=400, detail="source_id is required for Supabase lookup")

    endpoint = f"{supabase_url.rstrip('/')}/rest/v1/{table}"
    params = {
        "select": payload.supabase_url_column,
        payload.supabase_id_column: f"eq.{payload.source_id}",
        "limit": "1",
    }
    response = httpx.get(endpoint, headers=_supabase_headers(service_key), params=params, timeout=30.0)
    response.raise_for_status()
    rows = response.json()
    if not rows:
        raise HTTPException(status_code=404, detail=f"No Supabase row found for {payload.supabase_id_column}={payload.source_id}")
    audio_url = rows[0].get(payload.supabase_url_column)
    if not audio_url:
        raise HTTPException(status_code=404, detail=f"Supabase row is missing {payload.supabase_url_column}")
    return str(audio_url)


def _download_audio(url: str) -> Path:
    suffix = Path(url.split("?")[0]).suffix or ".wav"
    destination = Path(tempfile.mkdtemp(prefix="linguacode-audio-")) / f"input{suffix}"
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    return destination


async def _save_uploaded_file(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "upload.wav").suffix or ".wav"
    filename = slugify_filename(Path(upload.filename or "upload.wav").stem) + suffix
    destination = Path(tempfile.mkdtemp(prefix="linguacode-upload-")) / filename
    with destination.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    await upload.close()
    return destination


def _cleanup_temp_path(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.exists():
            path.unlink()
        parent = path.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        LOGGER.warning("Failed to clean up temp path %s", path)


@lru_cache(maxsize=4)
def _get_inference_service(
    mode: str,
    adapter_path: str | None,
    direct_model_name: str,
    transcriber_model: str,
    label_provider: str,
    translation_backend: str,
) -> Any:
    from app.infer import SpeechJsonInferenceService

    LOGGER.info(
        "Initializing inference service | mode=%s | model=%s | adapter=%s",
        mode,
        direct_model_name,
        adapter_path,
    )
    return SpeechJsonInferenceService(
        mode=mode,
        transcriber_model=transcriber_model,
        label_provider=label_provider,
        translation_backend=translation_backend,
        direct_model_name=direct_model_name,
        adapter_path=adapter_path,
    )


def _rewrite_to_codex_prompt(clean_english: str, openai_model: str) -> str:
    api_key = _get_env("OPENAI_API_KEY")
    if not api_key:
        LOGGER.warning("OPENAI_API_KEY is not set; using clean_english as the codex prompt")
        return clean_english

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=openai_model,
            instructions=(
                "Rewrite the user's request into a concise, high-signal Codex-style coding prompt. "
                "Keep concrete implementation intent, constraints, and desired output. "
                "Return plain text only with no markdown fences."
            ),
            input=f"Original instruction:\n{clean_english}",
        )
    except Exception as exc:
        LOGGER.warning("OpenAI prompt rewrite failed; using clean_english as the codex prompt: %s", exc)
        return clean_english

    if not response.output_text:
        LOGGER.warning("OpenAI returned no prompt text; using clean_english as the codex prompt")
        return clean_english
    return response.output_text.strip()


def _post_result(postbin_url: str, payload: dict[str, Any]) -> tuple[int, str]:
    response = httpx.post(postbin_url, json=payload, timeout=30.0)
    response.raise_for_status()
    body = response.text
    if len(body) > 2000:
        body = body[:2000] + "...<truncated>"
    return response.status_code, body


def _fake_inference_label(audio_source: str) -> TranscriptLabel:
    slug = Path(audio_source.split("?")[0]).stem or "audio"
    payload = {
        "lang": "hinglish",
        "raw_mixed": f"{slug} ke liye codex prompt banao",
        "clean_native": f"{slug} के लिए Codex प्रॉम्प्ट बनाओ।",
        "clean_english": f"Create a Codex prompt for {slug}.",
    }
    return validate_json_output(payload)


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: Request) -> InferenceResponse:
    payload, upload = await _parse_request(request)
    temp_audio_path: Path | None = None

    try:
        resolved_url = payload.source_url or payload.s3_url
        if upload is not None:
            temp_audio_path = await _save_uploaded_file(upload)
            audio_source = f"upload:{upload.filename or temp_audio_path.name}"
        else:
            if payload.source_id:
                resolved_url = _resolve_supabase_audio_url(payload)
            if not resolved_url:
                raise HTTPException(status_code=400, detail="Provide file, source_url/s3_url, or source_id")
            temp_audio_path = _download_audio(resolved_url)
            audio_source = resolved_url

        adapter_path = payload.adapter_path or _get_env("HF_ADAPTER_REPO")
        direct_model_name = payload.direct_model_name or _get_env("DIRECT_MODEL_NAME", "Qwen/Qwen2-Audio-7B-Instruct")

        if _get_env("LINGUACODE_FAKE_INFERENCE") == "1":
            LOGGER.warning("LINGUACODE_FAKE_INFERENCE is enabled; returning a synthetic label")
            transcript_label = _fake_inference_label(audio_source)
        else:
            service = _get_inference_service(
                mode=payload.mode,
                adapter_path=adapter_path,
                direct_model_name=direct_model_name,
                transcriber_model=payload.transcriber_model,
                label_provider=payload.label_provider,
                translation_backend=payload.translation_backend,
            )
            transcript_label = service.run(str(temp_audio_path))
        codex_prompt = _rewrite_to_codex_prompt(transcript_label.clean_english, payload.openai_model)

        postbin_url = payload.postbin_url or _get_env("POSTBIN_URL")
        post_status: int | None = None
        post_response: str | None = None
        if postbin_url:
            webhook_payload = {
                "audio_source": audio_source,
                "transcript_label": transcript_label.ordered_dict(),
                "codex_prompt": codex_prompt,
            }
            post_status, post_response = _post_result(postbin_url, webhook_payload)

        return InferenceResponse(
            audio_source=audio_source,
            transcript_label=transcript_label.ordered_dict(),
            codex_prompt=codex_prompt,
            postbin_url=postbin_url,
            postbin_status=post_status,
            postbin_response=post_response,
        )
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(status_code=status, detail=detail) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        _cleanup_temp_path(temp_audio_path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
