from __future__ import annotations

from app.server import _rewrite_to_codex_prompt


def test_rewrite_to_codex_prompt_prefixes_fallback(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    prompt = _rewrite_to_codex_prompt("Update the user's last_seen field during login.", "gpt-5-mini")
    assert prompt == "Write code to update the user's last_seen field during login."


def test_rewrite_to_codex_prompt_preserves_existing_prefix(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    prompt = _rewrite_to_codex_prompt("Write code to parse CSV and output JSON.", "gpt-5-mini")
    assert prompt == "Write code to parse CSV and output JSON."
