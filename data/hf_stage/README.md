---
language:
- hi
- ta
- bn
license: apache-2.0
task_categories:
- automatic-speech-recognition
- text-generation
pretty_name: Linguacode Audio JSON Dataset
size_categories:
- 1K<n<10K
---

# Linguacode Audio JSON Dataset

This dataset is packaged for fine-tuning `Qwen/Qwen2-Audio-7B-Instruct` on Hindi/Hinglish, Tamil/Tanglish, and Bengali/Banglish audio-to-JSON generation.

## Repo

- Dataset repo: `kaushiksiva/linguacode-audio-json-v1`
- Format: local audio files plus `train.jsonl` and `valid.jsonl`
- Schema: `{"audio":"audio/train/...wav","target":"{\"lang\":...\"clean_english\":...}"}`

## Sources

- `ai4bharat/IndicVoices`
- `ai4bharat/Aksharantar`
- `ai4bharat/Bhasha-Abhijnaanam`

## Split Stats

### Train

- Rows: 5484
- Language counts: {"banglish": 1832, "hinglish": 1824, "tanglish": 1828}

### Valid

- Rows: 610
- Language counts: {"banglish": 204, "hinglish": 203, "tanglish": 203}
