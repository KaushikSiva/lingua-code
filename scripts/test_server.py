from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.linguacode_server import ClientRunOptions, execute_test_server


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
        "--run-codex",
        action="store_true",
        help="Execute the returned codex_prompt locally with the Codex CLI after the request succeeds",
    )
    parser.add_argument(
        "--codex-mode",
        choices=["interactive", "exec"],
        default="interactive",
        help="How to launch Codex when --run-codex is set",
    )
    parser.add_argument(
        "--codex-subdir",
        default="codex_output",
        help="Subdirectory under the repo root where Codex should write generated code",
    )
    parser.add_argument(
        "--mock-postbin",
        action="store_true",
        help="In --in-process mode, intercept the outbound postbin request and echo its URL/payload",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    options = ClientRunOptions(
        base_url=args.base_url,
        endpoint=args.endpoint,
        audio_root=args.audio_root,
        audio=args.audio,
        seed=args.seed,
        postbin_url=args.postbin_url,
        adapter_path=args.adapter_path,
        mode=args.mode,
        timeout=args.timeout,
        in_process=args.in_process,
        run_codex=args.run_codex,
        codex_mode=args.codex_mode,
        codex_subdir=args.codex_subdir,
        mock_postbin=args.mock_postbin,
    )
    result = execute_test_server(options)
    print(f"Selected audio: {result.selected_audio}")
    print(f"Status: {result.status_code}")
    print(result.raw_text)


if __name__ == "__main__":
    main()
