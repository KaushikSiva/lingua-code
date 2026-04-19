from __future__ import annotations

import argparse

from app.config import OUTPUT_DIR, ensure_project_dirs
from app.infer import SpeechJsonInferenceService, save_inference_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run audio-to-JSON inference.")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument("--mode", choices=["auto", "baseline", "direct"], default="auto")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--transcriber-model", default="openai/whisper-small")
    parser.add_argument("--label-provider", choices=["heuristic", "openai"], default="heuristic")
    parser.add_argument("--translation-backend", choices=["nllb", "rule"], default="nllb")
    parser.add_argument("--direct-model-name", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    return parser


def main() -> None:
    ensure_project_dirs()
    parser = build_parser()
    args = parser.parse_args()
    service = SpeechJsonInferenceService(
        mode=args.mode,
        transcriber_model=args.transcriber_model,
        label_provider=args.label_provider,
        translation_backend=args.translation_backend,
        direct_model_name=args.direct_model_name,
        adapter_path=args.adapter_path,
    )
    label = service.run(args.audio)
    output_path = save_inference_output(
        label=label,
        audio_path=args.audio,
        output_dir=args.output_dir,
    )
    print(output_path)


if __name__ == "__main__":
    main()

