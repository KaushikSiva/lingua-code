from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call the local FastAPI inference server with a random audio file.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI server base URL")
    parser.add_argument("--endpoint", default="/infer", help="Inference endpoint path")
    parser.add_argument(
        "--audio-root",
        default="data/hf_stage/audio",
        help="Directory to scan for audio files when --audio is not provided",
    )
    parser.add_argument("--audio", default=None, help="Specific audio file to upload")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument("--postbin-url", default=None, help="Optional postbin/webhook URL override")
    parser.add_argument("--adapter-path", default=None, help="Optional adapter path or HF adapter repo override")
    parser.add_argument("--mode", default="direct", choices=["auto", "baseline", "direct"])
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--in-process", action="store_true", help="Call the FastAPI app in-process instead of over HTTP")
    parser.add_argument(
        "--mock-postbin",
        action="store_true",
        help="In --in-process mode, intercept the outbound postbin request and echo its URL/payload",
    )
    return parser


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


def main() -> None:
    args = build_parser().parse_args()
    audio_path = choose_audio_file(args.audio, args.audio_root, args.seed)
    url = args.base_url.rstrip("/") + args.endpoint

    data = {"mode": args.mode}
    if args.postbin_url:
        data["postbin_url"] = args.postbin_url
    if args.adapter_path:
        data["adapter_path"] = args.adapter_path

    with audio_path.open("rb") as handle:
        files = {"file": (audio_path.name, handle, "audio/wav")}
        if args.in_process:
            from fastapi.testclient import TestClient
            import app.server as server_module

            if args.mock_postbin:
                server_module._rewrite_to_codex_prompt = lambda clean_english, openai_model: clean_english

                def fake_post_result(postbin_url: str, payload: dict[str, object]) -> tuple[int, str]:
                    body = json.dumps({"mock_postbin_url": postbin_url, "received": payload}, ensure_ascii=False)
                    return 200, body

                server_module._post_result = fake_post_result

            client = TestClient(server_module.app)
            response = client.post(
                args.endpoint,
                data=data,
                files=files,
            )
        else:
            response = httpx.post(
                url,
                data=data,
                files=files,
                timeout=args.timeout,
                follow_redirects=True,
            )

    print(f"Selected audio: {audio_path}")
    print(f"Status: {response.status_code}")
    try:
        payload = response.json()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except json.JSONDecodeError:
        print(response.text)

    response.raise_for_status()


if __name__ == "__main__":
    main()
