from __future__ import annotations

import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.linguacode_server import ClientRunOptions, execute_test_server
from app.run_store import upsert_run
from app.utils import bootstrap, setup_logging, slugify_filename

bootstrap()
LOGGER = setup_logging("app.linguacode_server_api", "linguacode_server_api.log")

app = FastAPI(title="Linguacode Server API", version="0.2.0")
TASKS: dict[str, dict[str, Any]] = {}


def _form_bool(value: str | None, default: bool) -> bool:
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "upload.wav").suffix or ".wav"
    filename = slugify_filename(Path(upload.filename or "upload.wav").stem) + suffix
    destination = Path(tempfile.mkdtemp(prefix="linguacode-test-client-")) / filename
    with destination.open("wb") as handle:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return destination


def _cleanup_path(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
        parent = path.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        pass


def _run_task(task_id: str, options: ClientRunOptions, audio_path: Path) -> None:
    TASKS[task_id]["status"] = "running"
    upsert_run(
        task_id=task_id,
        status="running",
        selected_audio=str(audio_path),
        base_url=options.base_url,
        mode=options.mode,
        run_codex=options.run_codex,
        codex_mode=options.codex_mode,
        codex_subdir=options.codex_subdir,
    )
    LOGGER.info("Job running | task_id=%s | audio=%s | mode=%s | run_codex=%s", task_id, audio_path, options.mode, options.run_codex)
    try:
        result = execute_test_server(options, allow_interactive_replace=False)
    except Exception as exc:
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = str(exc)
        upsert_run(
            task_id=task_id,
            status="failed",
            selected_audio=str(audio_path),
            error=str(exc),
        )
        LOGGER.exception("Job failed | task_id=%s | audio=%s", task_id, audio_path)
    else:
        TASKS[task_id]["status"] = "completed"
        TASKS[task_id]["result"] = {
            "selected_audio": result.selected_audio,
            "status_code": result.status_code,
            "payload": result.payload,
            "raw_text": result.raw_text,
            "codex_executed": result.codex_executed,
        }
        upsert_run(
            task_id=task_id,
            status="completed",
            selected_audio=result.selected_audio,
            status_code=result.status_code,
            codex_executed=result.codex_executed,
            result_payload=result.payload,
        )
        LOGGER.info("Job completed | task_id=%s | audio=%s | codex_executed=%s", task_id, audio_path, result.codex_executed)
    finally:
        _cleanup_path(audio_path)


def _start_task(task_id: str, options: ClientRunOptions, audio_path: Path) -> None:
    LOGGER.info("Starting background job | task_id=%s | audio=%s", task_id, audio_path)
    thread = threading.Thread(target=_run_task, args=(task_id, options, audio_path), daemon=True)
    thread.start()


@app.post("/run")
async def run(
    file: UploadFile = File(...),
    base_url: str | None = Form(default=None),
    endpoint: str | None = Form(default=None),
    postbin_url: str | None = Form(default=None),
    adapter_path: str | None = Form(default=None),
    mode: str | None = Form(default=None),
    timeout: float | None = Form(default=None),
    run_codex: str | None = Form(default=None),
    codex_mode: str | None = Form(default=None),
    codex_subdir: str | None = Form(default=None),
    in_process: str | None = Form(default=None),
    mock_postbin: str | None = Form(default=None),
) -> dict[str, Any]:
    defaults = ClientRunOptions()
    resolved_codex_mode = codex_mode or defaults.codex_mode
    if _form_bool(run_codex, defaults.run_codex) and resolved_codex_mode == "interactive":
        raise HTTPException(status_code=400, detail="Interactive Codex mode is not supported from the API server; use codex_mode=exec")

    audio_path = _save_upload(file)
    options = ClientRunOptions(
        base_url=base_url or defaults.base_url,
        endpoint=endpoint or defaults.endpoint,
        audio_root=defaults.audio_root,
        audio=str(audio_path),
        seed=defaults.seed,
        postbin_url=postbin_url if postbin_url is not None else defaults.postbin_url,
        adapter_path=adapter_path if adapter_path is not None else defaults.adapter_path,
        mode=mode or defaults.mode,
        timeout=timeout if timeout is not None else defaults.timeout,
        in_process=_form_bool(in_process, defaults.in_process),
        run_codex=_form_bool(run_codex, defaults.run_codex),
        codex_mode=resolved_codex_mode,
        codex_subdir=codex_subdir or defaults.codex_subdir,
        mock_postbin=_form_bool(mock_postbin, defaults.mock_postbin),
    )

    task_id = uuid.uuid4().hex
    TASKS[task_id] = {
        "status": "submitted",
        "selected_audio": str(audio_path),
        "codex_requested": options.run_codex,
    }
    upsert_run(
        task_id=task_id,
        status="submitted",
        selected_audio=str(audio_path),
        base_url=options.base_url,
        mode=options.mode,
        run_codex=options.run_codex,
        codex_mode=options.codex_mode,
        codex_subdir=options.codex_subdir,
    )
    LOGGER.info("Job submitted | task_id=%s | audio=%s | run_codex=%s", task_id, audio_path, options.run_codex)
    _start_task(task_id, options, audio_path)

    return {
        "task_id": task_id,
        "status": "submitted",
        "message": "Codex task submitted." if options.run_codex else "Inference task submitted.",
        "selected_audio": str(audio_path),
        "codex_requested": options.run_codex,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
