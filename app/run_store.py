from __future__ import annotations

import os
from typing import Any

from app.utils import setup_logging

LOGGER = setup_logging("app.run_store", "run_store.log")


def _database_url() -> str | None:
    value = os.getenv("DATABASE_URL")
    if value in (None, ""):
        return None
    return value


def _connect():
    database_url = _database_url()
    if not database_url:
        return None

    import psycopg

    return psycopg.connect(database_url)


def _ensure_table(cursor) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS linguacode_runs (
            task_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            selected_audio TEXT,
            base_url TEXT,
            mode TEXT,
            run_codex BOOLEAN NOT NULL DEFAULT FALSE,
            codex_mode TEXT,
            codex_subdir TEXT,
            status_code INTEGER,
            codex_executed BOOLEAN,
            error TEXT,
            result_payload JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


def upsert_run(
    *,
    task_id: str,
    status: str,
    selected_audio: str | None = None,
    base_url: str | None = None,
    mode: str | None = None,
    run_codex: bool | None = None,
    codex_mode: str | None = None,
    codex_subdir: str | None = None,
    status_code: int | None = None,
    codex_executed: bool | None = None,
    error: str | None = None,
    result_payload: dict[str, Any] | None = None,
) -> None:
    conn = _connect()
    if conn is None:
        return

    try:
        with conn:
            with conn.cursor() as cur:
                _ensure_table(cur)
                cur.execute(
                    """
                    INSERT INTO linguacode_runs (
                        task_id,
                        status,
                        selected_audio,
                        base_url,
                        mode,
                        run_codex,
                        codex_mode,
                        codex_subdir,
                        status_code,
                        codex_executed,
                        error,
                        result_payload
                    ) VALUES (
                        %(task_id)s,
                        %(status)s,
                        %(selected_audio)s,
                        %(base_url)s,
                        %(mode)s,
                        COALESCE(%(run_codex)s, FALSE),
                        %(codex_mode)s,
                        %(codex_subdir)s,
                        %(status_code)s,
                        %(codex_executed)s,
                        %(error)s,
                        %(result_payload)s
                    )
                    ON CONFLICT (task_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        selected_audio = COALESCE(EXCLUDED.selected_audio, linguacode_runs.selected_audio),
                        base_url = COALESCE(EXCLUDED.base_url, linguacode_runs.base_url),
                        mode = COALESCE(EXCLUDED.mode, linguacode_runs.mode),
                        run_codex = COALESCE(EXCLUDED.run_codex, linguacode_runs.run_codex),
                        codex_mode = COALESCE(EXCLUDED.codex_mode, linguacode_runs.codex_mode),
                        codex_subdir = COALESCE(EXCLUDED.codex_subdir, linguacode_runs.codex_subdir),
                        status_code = COALESCE(EXCLUDED.status_code, linguacode_runs.status_code),
                        codex_executed = COALESCE(EXCLUDED.codex_executed, linguacode_runs.codex_executed),
                        error = EXCLUDED.error,
                        result_payload = COALESCE(EXCLUDED.result_payload, linguacode_runs.result_payload),
                        updated_at = NOW()
                    """,
                    {
                        "task_id": task_id,
                        "status": status,
                        "selected_audio": selected_audio,
                        "base_url": base_url,
                        "mode": mode,
                        "run_codex": run_codex,
                        "codex_mode": codex_mode,
                        "codex_subdir": codex_subdir,
                        "status_code": status_code,
                        "codex_executed": codex_executed,
                        "error": error,
                        "result_payload": result_payload,
                    },
                )
    except Exception as exc:
        LOGGER.warning("Failed to persist run metadata for %s: %s", task_id, exc)
    finally:
        conn.close()
