from __future__ import annotations

from app.linguacode_server import _build_codex_prompt


def test_build_codex_prompt_adds_subdir_constraint() -> None:
    prompt = _build_codex_prompt("Write code to parse CSV.", "generated/foo")
    assert "Write all new or modified code only under the subdirectory `generated/foo`" in prompt
    assert "Write code to parse CSV." in prompt
