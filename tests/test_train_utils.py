from __future__ import annotations

from app.train_utils import AudioJsonTrainingDataset, resolve_training_files
from app.utils import write_jsonl


def test_dataset_resolves_relative_audio_paths(tmp_path) -> None:
    audio = tmp_path / "audio" / "sample.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"RIFF")
    train_jsonl = tmp_path / "train.jsonl"
    write_jsonl(
        train_jsonl,
        [
            {
                "audio": "audio/sample.wav",
                "target": '{"lang":"hinglish","raw_mixed":"ghar","clean_native":"घर","clean_english":"Home."}',
            }
        ],
    )
    dataset = AudioJsonTrainingDataset(train_jsonl)
    item = dataset[0]
    assert item["audio"] == str(audio.resolve())


def test_resolve_training_files_from_hub_snapshot(tmp_path, monkeypatch) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    valid_jsonl = tmp_path / "valid.jsonl"
    train_jsonl.write_text("", encoding="utf-8")
    valid_jsonl.write_text("", encoding="utf-8")

    monkeypatch.setattr("app.train_utils.snapshot_download", lambda **kwargs: str(tmp_path))
    train_file, valid_file = resolve_training_files(
        hf_dataset_repo="owner/repo",
        hf_dataset_revision="main",
        hf_token="token",
    )
    assert train_file == str(train_jsonl)
    assert valid_file == str(valid_jsonl)
