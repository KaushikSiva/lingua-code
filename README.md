# Linguacode

Linguacode turns Hindi/Hinglish, Tamil/Tanglish, or Bengali/Banglish speech into strict JSON:

```json
{
  "lang": "",
  "raw_mixed": "",
  "clean_native": "",
  "clean_english": ""
}
```

`clean_english` is the downstream-friendly field. It is intended to feed coding tools, agents, or automation that work best on concise English instructions while preserving technical intent from the original utterance.

## Purpose

The project builds a practical end-to-end pipeline for:

1. ingesting speech data from public AI4Bharat datasets
2. converting transcripts into supervised audio-to-JSON targets
3. fine-tuning `Qwen/Qwen2-Audio-7B-Instruct` with LoRA or QLoRA
4. running inference either through a direct audio-to-JSON model or a baseline ASR-to-label pipeline

## Architecture

- `app/datasets.py` loads AI4Bharat datasets and materializes local audio/transcript examples
- `app/transliteration.py` uses Aksharantar lexicons plus fallback transliteration helpers
- `app/label_transcript.py` converts transcripts into strict JSON labels with validation, retries, and OpenAI batching
- `app/dataset_builder.py` builds intermediate, labeled, curated, stratified train/valid JSONL exports and stages Hub uploads
- `app/train_utils.py` loads `Qwen2-Audio-7B-Instruct`, applies LoRA/QLoRA, and trains on exported JSONL
- `app/transcribe.py` provides the baseline ASR path
- `app/infer.py` runs either baseline inference or direct model inference and writes JSON outputs

## Data Sources

This implementation uses the public AI4Bharat sources requested in the spec:

1. `ai4bharat/IndicVoices`
   Used as the primary speech dataset. The ingest path filters Hindi, Tamil, and Bengali rows and stores normalized intermediate records with local audio paths plus transcripts.
2. `ai4bharat/Aksharantar`
   Used as transliteration support for Hindi, Tamil, and Bengali via subset lexicons (`hin`, `tam`, `ben`).
3. `ai4bharat/Bhasha-Abhijnaanam`
   Used as optional language-ID evaluation data for native-script and romanized text through `scripts/prepare_data.py lid-eval`.

## Data Prep Flow

The data-prep pipeline is implemented first and can run stage by stage or end to end.

1. Ingest IndicVoices:

```bash
source .venv/bin/activate
python scripts/prepare_data.py ingest \
  --language hindi \
  --audio-dir data/intermediate/audio \
  --hf-token "$HF_TOKEN"
```

2. Convert transcripts to labels:

```bash
python scripts/prepare_data.py label \
  --language hindi \
  --translation-backend nllb
```

3. Validate labeled rows:

```bash
python scripts/prepare_data.py validate \
  --language hindi
```

4. Run the same per-language ingest/label flow in separate terminals for `tamil` and `bengali`, then export the combined training set with basic label-quality filtering and stratified splits:

```bash
python scripts/prepare_data.py export \
  --train data/exports/train.jsonl \
  --valid data/exports/valid.jsonl
```

5. Optional language-ID evaluation using Bhasha-Abhijnaanam:

```bash
python scripts/prepare_data.py lid-eval \
  --output data/intermediate/bhasha_lid_eval.jsonl
```

Or run the full pipeline:

```bash
python scripts/prepare_data.py all \
  --language tamil \
  --audio-dir data/intermediate/audio \
  --hf-token "$HF_TOKEN"
```

Per-language runs write to language-specific manifests such as:

- `data/intermediate/hindi_intermediate.jsonl`
- `data/intermediate/hindi_labeled.jsonl`
- `data/intermediate/tamil_intermediate.jsonl`
- `data/intermediate/tamil_labeled.jsonl`
- `data/intermediate/bengali_intermediate.jsonl`
- `data/intermediate/bengali_labeled.jsonl`

After all three language jobs finish, run `python scripts/prepare_data.py export` once to merge them into the combined `train.jsonl` and `valid.jsonl`.

The final train and valid exports follow the exact required format:

```json
{"audio":"path/to/file.wav","target":"{\"lang\":\"hinglish\",\"raw_mixed\":\"login ke time user ka last_seen update karo\",\"clean_native\":\"लॉगिन के समय user का last_seen अपडेट करो\",\"clean_english\":\"Update the user's last_seen field during login.\"}"}
```

## Training Flow

Training targets `Qwen/Qwen2-Audio-7B-Instruct` directly for audio-to-JSON generation.

- The trainer reads the exported JSONL files
- It can also pull `train.jsonl`, `valid.jsonl`, and relative audio paths from a Hugging Face dataset repo
- Each sample is framed as a chat turn with audio plus a strict JSON instruction
- The assistant target is the exact JSON string from the dataset
- LoRA is enabled by default
- QLoRA is used when CUDA and 4-bit quantization are available
- Loss, checkpoints, and validation sample generations are logged during training

Run training with:

```bash
python scripts/train.py \
  --train data/exports/train.jsonl \
  --valid data/exports/valid.jsonl \
  --config configs/train.yaml
```

Or train from a dataset repo on the Hub:

```bash
python scripts/train.py \
  --hf-dataset-repo yourname/linguacode-audio-json-v1 \
  --config configs/train.yaml
```

## Hugging Face Dataset Upload

Package the local exports into a portable dataset repo layout:

```bash
python scripts/upload_dataset.py --stage-only
```

Upload the staged folder to a private dataset repo:

```bash
python scripts/upload_dataset.py
```

The upload workflow creates a dataset repo with:

- `train.jsonl`
- `valid.jsonl`
- `audio/train/<language>/*.wav`
- `audio/valid/<language>/*.wav`
- `README.md`

Audio paths inside the manifests are rewritten to repo-relative paths so the dataset can be downloaded on a GPU machine and trained directly.

If your local `transformers` build does not include Qwen2-Audio, install a newer release or a source build. Hugging Face documents `Qwen2AudioForConditionalGeneration` and `AutoProcessor` in the official Qwen2-Audio model docs and model card.

## Inference Flow

Two inference paths are implemented:

1. Baseline:
   audio -> Whisper transcription -> transcript labeling -> strict JSON
2. Direct:
   audio -> fine-tuned Qwen2-Audio -> strict JSON

`auto` mode tries the direct model first and falls back to baseline if direct inference is unavailable or invalid.

Run inference with:

```bash
python -m app.main --audio path/to/file.wav
```

Or explicitly select a mode:

```bash
python -m app.main \
  --audio path/to/file.wav \
  --mode baseline \
  --translation-backend nllb
```

Direct inference with a trained adapter:

```bash
python -m app.main \
  --audio path/to/file.wav \
  --mode direct \
  --adapter-path artifacts/qwen2_audio_json
```

Outputs are written to `data/output/<audio_stem>.json`.

## FastAPI Server

The repo also includes a single-endpoint FastAPI server at [app/server.py](/Users/kaushiksivakumar/linguacode/app/server.py:1) for end-to-end inference and forwarding:

- accept an uploaded audio file over HTTP
- or accept a direct audio URL / S3 URL
- or accept a Supabase record id and resolve its `url` column through the Supabase REST API
- run local inference with either the uploaded full merged model or the LoRA adapter from Hugging Face
- rewrite `clean_english` into a Codex-style prompt using OpenAI
- POST the result JSON to a webhook / postbin URL

Environment variables:

- `DIRECT_MODEL_NAME` for the direct inference model id
- `HF_ADAPTER_REPO` optional, only needed when using a separate LoRA adapter repo
- `OPENAI_API_KEY` for the Codex-prompt rewrite step
- `POSTBIN_URL` default destination webhook if not provided per request
- `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_TABLE` for `source_id` lookups

Use the uploaded full merged model:

```bash
export DIRECT_MODEL_NAME=kaushiksiva/linguacode-qwen2-audio-full
unset HF_ADAPTER_REPO
```

Or use the base model plus LoRA adapter:

```bash
export DIRECT_MODEL_NAME=Qwen/Qwen2-Audio-7B-Instruct
export HF_ADAPTER_REPO=kaushiksiva/linguacode-qwen2-audio-lora
```

Run the server:

```bash
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

JSON request with a direct URL:

```bash
curl -X POST http://localhost:8000/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "source_url": "https://example.com/audio.wav",
    "postbin_url": "https://postb.in/your-endpoint"
  }'
```

Multipart upload:

```bash
curl -X POST http://localhost:8000/infer \
  -F file=@sample.wav \
  -F postbin_url=https://postb.in/your-endpoint
```

## Prompt Contract

The transcript labeling prompt lives at [prompts/transcript_to_json.txt](/Users/kaushiksivakumar/linguacode/prompts/transcript_to_json.txt:1). It enforces:

- JSON-only output
- exact key order
- Hindi -> `hinglish`, Tamil -> `tanglish`, Bengali -> `banglish`
- preservation of technical English tokens where appropriate

## Tests

Run the lightweight test suite with:

```bash
pytest tests/test_schema.py tests/test_dataset_builder.py tests/test_labeling.py tests/test_inference.py tests/test_cli.py tests/test_train_utils.py
```

## Setup

Use the local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You can also drive data prep from `.env` instead of repeating flags. Copy `.env.example` to `.env`, fill in your secrets and defaults, then run:

```bash
python scripts/prepare_data.py all
```

Recommended `.env` entries for OpenAI-backed labeling:

```env
HF_TOKEN=your_hf_token
HF_DATASET_REPO=yourname/linguacode-audio-json-v1
HF_DATASET_PRIVATE=true
HF_DATASET_REVISION=
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-5-mini
OPENAI_BATCH_SIZE=16
LABEL_PROVIDER=openai
TRANSLATION_BACKEND=nllb
PREP_STREAMING=true
PREP_MAX_SAMPLES_PER_LANGUAGE=15000
PREP_PROGRESS_EVERY=100
```

During long prep runs, the script now logs progress lines like:

```text
label progress 3200/15000 | 21.3% | elapsed 00:27:10 | eta 01:40:20 | accepted=3018 rejected=182
```

OpenAI-backed labeling also logs batch lifecycle details and resumes safely on reruns:

```text
resume state | existing labeled rows=3240 | remaining=11760 | total=15000
label batch start | rows 3241-3270 | size=30 | attempt=1
OpenAI request waiting | elapsed 00:00:30 | rows 3241-3270 | size=30 | attempt=1
label batch complete | rows 3241-3270 | size=30 | attempt=1 | elapsed 00:00:44
```
# lingua-code
