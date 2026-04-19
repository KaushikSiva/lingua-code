from __future__ import annotations

import json
import logging

from app.dataset_builder import (
    build_intermediate_dataset,
    discover_language_labeled_paths,
    export_training_jsonl,
    label_intermediate_dataset,
    merge_labeled_datasets,
    resolve_default_dataset_repo,
    stage_training_dataset_for_hub,
    stratified_split_rows,
    upload_staged_dataset_to_hub,
)
from app.utils import read_jsonl, write_jsonl


def test_dataset_export_format(tmp_path, monkeypatch) -> None:
    intermediate = tmp_path / "intermediate.jsonl"
    labeled = tmp_path / "labeled.jsonl"
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"

    write_jsonl(
        intermediate,
        [
            {"audio": "/tmp/a.wav", "transcript": "कल मीटिंग है"},
            {"audio": "/tmp/b.wav", "transcript": "কাল meeting আছে"},
        ],
    )

    class FakeLabel:
        def __init__(self, lang: str, raw_mixed: str, clean_native: str, clean_english: str) -> None:
            self.lang = lang
            self.raw_mixed = raw_mixed
            self.clean_native = clean_native
            self.clean_english = clean_english

        def ordered_dict(self) -> dict:
            return {
                "lang": self.lang,
                "raw_mixed": self.raw_mixed,
                "clean_native": self.clean_native,
                "clean_english": self.clean_english,
            }

    class FakeLabeler:
        def label_transcript(self, transcript: str):
            if "কাল" in transcript:
                return FakeLabel("banglish", "kal meeting ache", "কাল meeting আছে", "There is a meeting tomorrow.")
            return FakeLabel("hinglish", "kal meeting hai", "कल मीटिंग है", "There is a meeting tomorrow.")

    monkeypatch.setattr(
        "app.dataset_builder.TranscriptLabeler.from_defaults",
        lambda **kwargs: FakeLabeler(),
    )

    label_intermediate_dataset(
        input_path=intermediate,
        output_path=labeled,
        provider="heuristic",
        translation_backend="rule",
        use_aksharantar=False,
    )
    train_rows, valid_rows = export_training_jsonl(
        labeled_path=labeled,
        train_path=train,
        valid_path=valid,
        valid_ratio=0.5,
        seed=7,
    )

    assert train_rows or valid_rows
    exported_rows = list(read_jsonl(train)) + list(read_jsonl(valid))
    assert exported_rows
    sample = exported_rows[0]
    assert set(sample.keys()) == {"audio", "target"}
    payload = json.loads(sample["target"])
    assert list(payload.keys()) == ["lang", "raw_mixed", "clean_native", "clean_english"]


def test_build_intermediate_dataset_filters_to_requested_language(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_ingest(**kwargs):
        calls.append(kwargs)
        return [{"audio": str(tmp_path / "audio.wav"), "transcript": "வணக்கம்"}]

    monkeypatch.setattr("app.dataset_builder.ingest_indicvoices", fake_ingest)
    output = tmp_path / "tamil_intermediate.jsonl"
    rows = build_intermediate_dataset(
        output_path=output,
        audio_dir=tmp_path / "audio",
        language_name="tamil",
    )
    assert rows[0]["transcript"] == "வணக்கம்"
    assert calls[0]["languages"] == ["tamil"]
    persisted = list(read_jsonl(output))
    assert len(persisted) == 1


def test_openai_labeling_uses_batching(tmp_path, monkeypatch) -> None:
    intermediate = tmp_path / "intermediate.jsonl"
    labeled = tmp_path / "labeled.jsonl"
    write_jsonl(
        intermediate,
        [
            {"audio": "/tmp/a.wav", "transcript": "one"},
            {"audio": "/tmp/b.wav", "transcript": "two"},
            {"audio": "/tmp/c.wav", "transcript": "three"},
        ],
    )

    class FakeLabel:
        def __init__(self, value: str) -> None:
            self.value = value

        def ordered_dict(self) -> dict:
            return {
                "lang": "hinglish",
                "raw_mixed": self.value,
                "clean_native": self.value,
                "clean_english": self.value,
            }

    class FakeLabeler:
        def label_transcripts(self, transcripts, batch_start=None, batch_end=None, log_prefix=None):
            return [FakeLabel(text.upper()) for text in transcripts]

    monkeypatch.setattr(
        "app.dataset_builder.TranscriptLabeler.from_defaults",
        lambda **kwargs: FakeLabeler(),
    )

    rows = label_intermediate_dataset(
        input_path=intermediate,
        output_path=labeled,
        provider="openai",
        openai_batch_size=2,
    )
    assert len(rows) == 3
    assert rows[0]["label"]["raw_mixed"] == "ONE"
    assert rows[2]["label"]["raw_mixed"] == "THREE"


def test_labeling_resumes_and_appends_without_duplicates(tmp_path, monkeypatch) -> None:
    intermediate = tmp_path / "intermediate.jsonl"
    labeled = tmp_path / "labeled.jsonl"
    write_jsonl(
        intermediate,
        [
            {"audio": "/tmp/a.wav", "transcript": "one"},
            {"audio": "/tmp/b.wav", "transcript": "two"},
            {"audio": "/tmp/c.wav", "transcript": "three"},
        ],
    )
    write_jsonl(
        labeled,
        [
            {
                "audio": "/tmp/a.wav",
                "label": {
                    "lang": "hinglish",
                    "raw_mixed": "ONE",
                    "clean_native": "ONE",
                    "clean_english": "ONE",
                },
            }
        ],
    )

    class FakeLabel:
        def __init__(self, value: str) -> None:
            self.value = value

        def ordered_dict(self) -> dict:
            return {
                "lang": "hinglish",
                "raw_mixed": self.value,
                "clean_native": self.value,
                "clean_english": self.value,
            }

    class FakeLabeler:
        def label_transcripts(self, transcripts, batch_start=None, batch_end=None, log_prefix=None):
            return [FakeLabel(text.upper()) for text in transcripts]

    monkeypatch.setattr(
        "app.dataset_builder.TranscriptLabeler.from_defaults",
        lambda **kwargs: FakeLabeler(),
    )

    rows = label_intermediate_dataset(
        input_path=intermediate,
        output_path=labeled,
        provider="openai",
        openai_batch_size=2,
        resume=True,
    )
    persisted = list(read_jsonl(labeled))
    assert len(rows) == 3
    assert len(persisted) == 3
    assert persisted[0]["label"]["raw_mixed"] == "ONE"
    assert persisted[1]["label"]["raw_mixed"] == "TWO"
    assert persisted[2]["label"]["raw_mixed"] == "THREE"


def test_language_specific_resume_does_not_touch_other_outputs(tmp_path, monkeypatch) -> None:
    hindi_intermediate = tmp_path / "hindi_intermediate.jsonl"
    tamil_labeled = tmp_path / "tamil_labeled.jsonl"
    hindi_labeled = tmp_path / "hindi_labeled.jsonl"
    write_jsonl(
        hindi_intermediate,
        [
            {"audio": "/tmp/a.wav", "transcript": "one"},
            {"audio": "/tmp/b.wav", "transcript": "two"},
        ],
    )
    write_jsonl(
        hindi_labeled,
        [
            {
                "audio": "/tmp/a.wav",
                "label": {
                    "lang": "hinglish",
                    "raw_mixed": "ONE",
                    "clean_native": "ONE",
                    "clean_english": "ONE",
                },
            }
        ],
    )
    write_jsonl(
        tamil_labeled,
        [
            {
                "audio": "/tmp/t.wav",
                "label": {
                    "lang": "tanglish",
                    "raw_mixed": "vanakkam",
                    "clean_native": "வணக்கம்",
                    "clean_english": "Hello.",
                },
            }
        ],
    )

    class FakeLabel:
        def __init__(self, value: str) -> None:
            self.value = value

        def ordered_dict(self) -> dict:
            return {
                "lang": "hinglish",
                "raw_mixed": self.value,
                "clean_native": self.value,
                "clean_english": self.value,
            }

    class FakeLabeler:
        def label_transcripts(self, transcripts, batch_start=None, batch_end=None, log_prefix=None):
            return [FakeLabel(text.upper()) for text in transcripts]

    monkeypatch.setattr(
        "app.dataset_builder.TranscriptLabeler.from_defaults",
        lambda **kwargs: FakeLabeler(),
    )

    label_intermediate_dataset(
        input_path=hindi_intermediate,
        output_path=hindi_labeled,
        provider="openai",
        openai_batch_size=2,
        resume=True,
        language_name="hindi",
    )
    tamil_rows = list(read_jsonl(tamil_labeled))
    hindi_rows = list(read_jsonl(hindi_labeled))
    assert len(tamil_rows) == 1
    assert len(hindi_rows) == 2
    assert hindi_rows[1]["label"]["raw_mixed"] == "TWO"


def test_labeling_incrementally_writes_output(tmp_path, monkeypatch) -> None:
    intermediate = tmp_path / "intermediate.jsonl"
    labeled = tmp_path / "labeled.jsonl"
    write_jsonl(
        intermediate,
        [
            {"audio": "/tmp/a.wav", "transcript": "one"},
            {"audio": "/tmp/b.wav", "transcript": "two"},
        ],
    )

    class FakeLabel:
        def __init__(self, value: str) -> None:
            self.value = value

        def ordered_dict(self) -> dict:
            return {
                "lang": "hinglish",
                "raw_mixed": self.value,
                "clean_native": self.value,
                "clean_english": self.value,
            }

    observed_counts = []

    class FakeLabeler:
        def label_transcripts(self, transcripts, batch_start=None, batch_end=None, log_prefix=None):
            if labeled.exists():
                observed_counts.append(sum(1 for _ in read_jsonl(labeled)))
            else:
                observed_counts.append(0)
            return [FakeLabel(text.upper()) for text in transcripts]

    monkeypatch.setattr(
        "app.dataset_builder.TranscriptLabeler.from_defaults",
        lambda **kwargs: FakeLabeler(),
    )

    label_intermediate_dataset(
        input_path=intermediate,
        output_path=labeled,
        provider="openai",
        openai_batch_size=1,
        resume=True,
    )
    assert observed_counts == [0, 1]


def test_labeling_supports_row_slices_with_resume(tmp_path, monkeypatch) -> None:
    intermediate = tmp_path / "indicvoices_intermediate.jsonl"
    tamil_labeled = tmp_path / "tamil_labeled.jsonl"
    write_jsonl(
        intermediate,
        [
            {"audio": f"/tmp/{index}.wav", "transcript": f"row-{index}"}
            for index in range(1, 11)
        ],
    )
    write_jsonl(
        tamil_labeled,
        [
            {
                "audio": "/tmp/6.wav",
                "label": {
                    "lang": "tanglish",
                    "raw_mixed": "ROW-6",
                    "clean_native": "ROW-6",
                    "clean_english": "ROW-6",
                },
            }
        ],
    )

    seen_batches = []

    class FakeLabel:
        def __init__(self, value: str) -> None:
            self.value = value

        def ordered_dict(self) -> dict:
            return {
                "lang": "tanglish",
                "raw_mixed": self.value,
                "clean_native": self.value,
                "clean_english": self.value,
            }

    class FakeLabeler:
        def label_transcripts(self, transcripts, batch_start=None, batch_end=None, log_prefix=None):
            seen_batches.append((batch_start, batch_end, list(transcripts), log_prefix))
            return [FakeLabel(text.upper()) for text in transcripts]

    monkeypatch.setattr(
        "app.dataset_builder.TranscriptLabeler.from_defaults",
        lambda **kwargs: FakeLabeler(),
    )

    rows = label_intermediate_dataset(
        input_path=intermediate,
        output_path=tamil_labeled,
        provider="openai",
        openai_batch_size=2,
        resume=True,
        language_name="tamil",
        start_row=6,
        end_row=10,
    )
    persisted = list(read_jsonl(tamil_labeled))
    assert len(rows) == 5
    assert len(persisted) == 5
    assert seen_batches[0][0:2] == (7, 8)
    assert seen_batches[1][0:2] == (9, 10)
    assert seen_batches[0][2] == ["row-7", "row-8"]
    assert persisted[-1]["label"]["raw_mixed"] == "ROW-10"


def test_language_tagged_progress_logging(caplog, tmp_path, monkeypatch) -> None:
    caplog.set_level(logging.INFO)
    intermediate = tmp_path / "bengali_intermediate.jsonl"
    labeled = tmp_path / "bengali_labeled.jsonl"
    write_jsonl(
        intermediate,
        [
            {"audio": "/tmp/a.wav", "transcript": "one"},
            {"audio": "/tmp/b.wav", "transcript": "two"},
        ],
    )

    class FakeLabel:
        def __init__(self, value: str) -> None:
            self.value = value

        def ordered_dict(self) -> dict:
            return {
                "lang": "banglish",
                "raw_mixed": self.value,
                "clean_native": self.value,
                "clean_english": self.value,
            }

    class FakeLabeler:
        def label_transcripts(self, transcripts, batch_start=None, batch_end=None, log_prefix=None):
            return [FakeLabel(text.upper()) for text in transcripts]

    monkeypatch.setattr(
        "app.dataset_builder.TranscriptLabeler.from_defaults",
        lambda **kwargs: FakeLabeler(),
    )

    label_intermediate_dataset(
        input_path=intermediate,
        output_path=labeled,
        provider="openai",
        openai_batch_size=1,
        progress_every=10,
        resume=True,
        language_name="bengali",
    )
    messages = [record.message for record in caplog.records]
    assert any("label[bengali] progress 1/2" in message for message in messages)


def test_stratified_split_preserves_language_balance() -> None:
    rows = [
        {"audio": "/tmp/1.wav", "label": {"lang": "hinglish", "raw_mixed": "a", "clean_native": "a", "clean_english": "a"}},
        {"audio": "/tmp/2.wav", "label": {"lang": "hinglish", "raw_mixed": "b", "clean_native": "b", "clean_english": "b"}},
        {"audio": "/tmp/3.wav", "label": {"lang": "tanglish", "raw_mixed": "c", "clean_native": "c", "clean_english": "c"}},
        {"audio": "/tmp/4.wav", "label": {"lang": "tanglish", "raw_mixed": "d", "clean_native": "d", "clean_english": "d"}},
        {"audio": "/tmp/5.wav", "label": {"lang": "banglish", "raw_mixed": "e", "clean_native": "e", "clean_english": "e"}},
        {"audio": "/tmp/6.wav", "label": {"lang": "banglish", "raw_mixed": "f", "clean_native": "f", "clean_english": "f"}},
    ]
    train_rows, valid_rows = stratified_split_rows(rows, valid_ratio=0.5, seed=3)
    assert len(train_rows) == 3
    assert len(valid_rows) == 3
    assert sorted(row["label"]["lang"] for row in train_rows) == ["banglish", "hinglish", "tanglish"]
    assert sorted(row["label"]["lang"] for row in valid_rows) == ["banglish", "hinglish", "tanglish"]


def test_export_merges_language_specific_labeled_files(tmp_path) -> None:
    intermediate_dir = tmp_path / "intermediate"
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        intermediate_dir / "hindi_labeled.jsonl",
        [
            {"audio": "/tmp/1.wav", "label": {"lang": "hinglish", "raw_mixed": "a", "clean_native": "a", "clean_english": "Alpha."}},
            {"audio": "/tmp/2.wav", "label": {"lang": "hinglish", "raw_mixed": "b", "clean_native": "b", "clean_english": "Beta."}},
        ],
    )
    write_jsonl(
        intermediate_dir / "tamil_labeled.jsonl",
        [
            {"audio": "/tmp/3.wav", "label": {"lang": "tanglish", "raw_mixed": "c", "clean_native": "c", "clean_english": "Gamma."}},
            {"audio": "/tmp/4.wav", "label": {"lang": "tanglish", "raw_mixed": "d", "clean_native": "d", "clean_english": "Delta."}},
        ],
    )
    write_jsonl(
        intermediate_dir / "bengali_labeled.jsonl",
        [
            {"audio": "/tmp/5.wav", "label": {"lang": "banglish", "raw_mixed": "e", "clean_native": "e", "clean_english": "Epsilon."}},
            {"audio": "/tmp/6.wav", "label": {"lang": "banglish", "raw_mixed": "f", "clean_native": "f", "clean_english": "Zeta."}},
        ],
    )

    discovered = discover_language_labeled_paths(intermediate_dir)
    assert [path.name for path in discovered] == [
        "hindi_labeled.jsonl",
        "tamil_labeled.jsonl",
        "bengali_labeled.jsonl",
    ]

    train_rows, valid_rows = export_training_jsonl(
        labeled_path=discovered,
        train_path=train,
        valid_path=valid,
        valid_ratio=0.5,
        seed=5,
    )
    assert len(train_rows) == 3
    assert len(valid_rows) == 3
    payloads = [json.loads(row["target"]) for row in list(read_jsonl(train)) + list(read_jsonl(valid))]
    assert sorted(payload["lang"] for payload in payloads) == [
        "banglish",
        "banglish",
        "hinglish",
        "hinglish",
        "tanglish",
        "tanglish",
    ]


def test_discover_language_labeled_paths_prefers_legacy_hindi_file(tmp_path) -> None:
    intermediate_dir = tmp_path / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        intermediate_dir / "indicvoices_labeled.jsonl",
        [{"audio": "/tmp/h.wav", "label": {"lang": "hinglish", "raw_mixed": "a", "clean_native": "a", "clean_english": "A."}}],
    )
    write_jsonl(
        intermediate_dir / "hindi_labeled.jsonl",
        [{"audio": "/tmp/h2.wav", "label": {"lang": "hinglish", "raw_mixed": "b", "clean_native": "b", "clean_english": "B."}}],
    )
    write_jsonl(
        intermediate_dir / "tamil_labeled.jsonl",
        [{"audio": "/tmp/t.wav", "label": {"lang": "tanglish", "raw_mixed": "c", "clean_native": "c", "clean_english": "C."}}],
    )

    discovered = discover_language_labeled_paths(intermediate_dir)
    assert [path.name for path in discovered] == [
        "indicvoices_labeled.jsonl",
        "tamil_labeled.jsonl",
    ]


def test_stage_training_dataset_rewrites_audio_paths(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    audio_a = tmp_path / "source_a.wav"
    audio_b = tmp_path / "source_b.wav"
    audio_a.write_bytes(b"RIFFa")
    audio_b.write_bytes(b"RIFFb")
    write_jsonl(
        train,
        [
            {
                "audio": str(audio_a),
                "target": json.dumps(
                    {
                        "lang": "hinglish",
                        "raw_mixed": "ghar",
                        "clean_native": "घर",
                        "clean_english": "House.",
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        ],
    )
    write_jsonl(
        valid,
        [
            {
                "audio": str(audio_b),
                "target": json.dumps(
                    {
                        "lang": "banglish",
                        "raw_mixed": "bari",
                        "clean_native": "বাড়ি",
                        "clean_english": "Home.",
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        ],
    )
    stats = stage_training_dataset_for_hub(train, valid, tmp_path / "stage", repo_id="me/repo")
    staged_train = list(read_jsonl(tmp_path / "stage" / "train.jsonl"))
    staged_valid = list(read_jsonl(tmp_path / "stage" / "valid.jsonl"))
    assert staged_train[0]["audio"].startswith("audio/train/hindi/")
    assert staged_valid[0]["audio"].startswith("audio/valid/bengali/")
    assert (tmp_path / "stage" / staged_train[0]["audio"]).exists()
    assert (tmp_path / "stage" / staged_valid[0]["audio"]).exists()
    assert stats["train"]["language_counts"] == {"hinglish": 1}
    assert stats["valid"]["language_counts"] == {"banglish": 1}


def test_upload_helpers_call_hub_api(tmp_path, monkeypatch) -> None:
    calls = []

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def whoami(self):
            return {"name": "tester"}

        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs))

        def upload_folder(self, **kwargs):
            calls.append(("upload_folder", kwargs))

    monkeypatch.setattr("app.dataset_builder.HfApi", FakeApi)
    repo_id = resolve_default_dataset_repo("token")
    assert repo_id == "tester/linguacode-audio-json-v1"
    returned = upload_staged_dataset_to_hub(tmp_path, token="token", repo_id=repo_id, private=True)
    assert returned == repo_id
    assert calls[0][0] == "create_repo"
    assert calls[1][0] == "upload_folder"
