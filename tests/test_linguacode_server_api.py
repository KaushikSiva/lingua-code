from __future__ import annotations

from fastapi.testclient import TestClient

from app.linguacode_server_api import TASKS, app


def test_run_endpoint_submits_uploaded_audio(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_start_task(task_id, options, audio_path):
        captured["task_id"] = task_id
        captured["audio"] = options.audio
        captured["run_codex"] = options.run_codex
        captured["audio_path_exists"] = audio_path.exists()

    monkeypatch.setattr("app.linguacode_server_api._start_task", fake_start_task)
    client = TestClient(app)
    response = client.post(
        "/run",
        files={"file": ("sample.wav", b"fake-audio", "audio/wav")},
        data={"mode": "direct", "run_codex": "true", "codex_mode": "exec"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "submitted"
    assert payload["codex_requested"] is True
    assert payload["message"] == "Codex task submitted."
    assert captured["audio_path_exists"] is True
    assert TASKS[payload["task_id"]]["status"] == "submitted"


def test_run_endpoint_rejects_interactive_codex() -> None:
    client = TestClient(app)
    response = client.post(
        "/run",
        files={"file": ("sample.wav", b"fake-audio", "audio/wav")},
        data={"run_codex": "true", "codex_mode": "interactive"},
    )
    assert response.status_code == 400
    assert "Interactive Codex mode" in response.json()["detail"]
