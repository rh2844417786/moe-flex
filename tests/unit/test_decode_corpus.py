from __future__ import annotations

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from flexmoe.datasets.decode_corpus import (
    load_unique_workload,
    prepare_decode_corpus,
    verify_decode_corpus,
)
from flexmoe.datasets.sharegpt import PromptRecord, verify_subset, write_jsonl_zst
from flexmoe.manifest import sha256_file


class StubChatTokenizer:
    eos_token_id = 0

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        return conversation[0]["content"]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [int(part) for part in text.split()]


def write_source(path: Path, prompts: list[str]) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": str(index),
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt", "value": "acknowledged"},
                    ],
                }
                for index, prompt in enumerate(prompts)
            ]
        ),
        encoding="utf-8",
    )


def write_dataset(
    tmp_path: Path, tokens: list[tuple[int, ...]]
) -> tuple[Path, Path]:
    dataset = tmp_path / "decode.jsonl.zst"
    records = [
        PromptRecord(f"request-{index}", prompt, len(prompt))
        for index, prompt in enumerate(tokens)
    ]
    digest = write_jsonl_zst(records, dataset)
    counts: dict[str, int] = {}
    unique: dict[str, int] = {}
    for prompt in tokens:
        key = str(len(prompt))
        counts[key] = counts.get(key, 0) + 1
        unique.setdefault(key, 0)
    for length in {len(prompt) for prompt in tokens}:
        unique[str(length)] = len({prompt for prompt in tokens if len(prompt) == length})
    manifest = tmp_path / "decode-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sha256": digest,
                "record_count": len(records),
                "counts_by_context": counts,
                "unique_by_context": unique,
                "no_repetition": True,
            }
        ),
        encoding="utf-8",
    )
    return dataset, manifest


def test_prepare_decode_corpus_deduplicates_full_token_sequences(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    write_source(
        source,
        ["1 2 3 4", "1 2 3 4", "5 6 7 8", "9 10 11 12"],
    )
    first_data = tmp_path / "first.zst"
    first_manifest = tmp_path / "first.json"
    second_data = tmp_path / "second.zst"
    second_manifest = tmp_path / "second.json"

    first = prepare_decode_corpus(
        source,
        StubChatTokenizer(),
        first_data,
        first_manifest,
        lengths=(4,),
        unique_per_length=3,
        seed=0,
    )
    second = prepare_decode_corpus(
        source,
        StubChatTokenizer(),
        second_data,
        second_manifest,
        lengths=(4,),
        unique_per_length=3,
        seed=0,
    )

    assert first["sha256"] == second["sha256"]
    assert first["counts_by_context"] == {"4": 3}
    assert first["unique_by_context"] == {"4": 3}
    assert first["no_repetition"] is True
    assert first["candidate_generation"] == {
        "bounded": True,
        "source_wrap": False,
        "candidates_by_context": {"4": 4},
        "duplicate_candidates_by_context": {"4": 1},
        "source_rows_consumed_by_context": {"4": 4},
    }
    assert first["packing"] == {
        "policy": "shuffle-pack-eos-truncate",
        "order": "seeded-source-row-shuffle",
        "overflow": "discarded-per-candidate",
    }
    verify_subset(first_data, first_manifest)
    verify_decode_corpus(first_data, first_manifest)


def test_prepare_decode_corpus_fails_without_source_wrap(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    write_source(source, ["1 2 3 4", "1 2 3 4"])
    dataset = tmp_path / "decode.zst"
    manifest = tmp_path / "manifest.json"

    with pytest.raises(ValueError, match="only 1 unique prompts.*2 required"):
        prepare_decode_corpus(
            source,
            StubChatTokenizer(),
            dataset,
            manifest,
            lengths=(4,),
            unique_per_length=2,
            seed=0,
        )

    assert not dataset.exists()
    assert not manifest.exists()


def test_prepare_decode_corpus_never_overwrites_its_source(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    write_source(source, ["1 2 3 4", "5 6 7 8"])
    original = source.read_bytes()

    with pytest.raises(ValueError, match="must be distinct"):
        prepare_decode_corpus(
            source,
            StubChatTokenizer(),
            source,
            tmp_path / "manifest.json",
            lengths=(4,),
            unique_per_length=1,
            seed=0,
        )

    assert source.read_bytes() == original


@pytest.mark.parametrize("existing_name", ["decode.zst", "manifest.json"])
def test_prepare_decode_corpus_requires_fresh_destinations(
    tmp_path: Path, existing_name: str
) -> None:
    source = tmp_path / "source.json"
    write_source(source, ["1 2 3 4", "5 6 7 8"])
    output = tmp_path / "decode.zst"
    manifest = tmp_path / "manifest.json"
    existing = tmp_path / existing_name
    existing.write_bytes(b"do-not-overwrite")

    with pytest.raises(FileExistsError, match="fresh paths"):
        prepare_decode_corpus(
            source,
            StubChatTokenizer(),
            output,
            manifest,
            lengths=(4,),
            unique_per_length=1,
            seed=0,
        )

    assert existing.read_bytes() == b"do-not-overwrite"
    assert output == existing or not output.exists()
    assert manifest == existing or not manifest.exists()


def test_prepare_decode_corpus_stages_source_manifest_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    write_source(source, ["1 2 3 4", "5 6 7 8"])
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": {
                    "repo_id": "fixture/sharegpt",
                    "revision": "source-revision",
                    "filename": source.name,
                    "sha256": sha256_file(source),
                },
                "tokenizer": {
                    "repo_id": "fixture/tokenizer",
                    "revision": "tokenizer-revision",
                    "files": {"tokenizer.json": "a" * 64},
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "decode.zst"
    manifest = tmp_path / "manifest.json"

    generated = prepare_decode_corpus(
        source,
        StubChatTokenizer(),
        output,
        manifest,
        lengths=(4,),
        unique_per_length=1,
        seed=0,
        source_manifest=source_manifest,
    )

    assert generated == json.loads(manifest.read_text(encoding="utf-8"))
    assert generated["source_manifest_sha256"] == sha256_file(source_manifest)
    assert generated["source_provenance"]["revision"] == "source-revision"
    assert generated["source_provenance"]["cleaning"]["rows"] == 2
    assert generated["tokenizer_provenance"]["revision"] == "tokenizer-revision"
    assert generated["tokenizer_provenance"]["runtime"]["eos_token_id"] == 0


def test_verify_decode_corpus_rejects_duplicate_tokens_with_distinct_ids(
    tmp_path: Path,
) -> None:
    dataset, manifest = write_dataset(
        tmp_path, [(1, 2), (1, 2), (3, 4), (5, 6)]
    )

    with pytest.raises(ValueError, match="duplicate prompt token hash"):
        verify_decode_corpus(dataset, manifest)


def test_load_unique_workload_uses_file_order_and_no_wrap(tmp_path: Path) -> None:
    tokens = [(index, index + 100) for index in range(40)]
    dataset, manifest = write_dataset(tmp_path, tokens)

    prompts, calibration, metadata = load_unique_workload(
        dataset,
        manifest,
        context_length=2,
        request_count=3,
        calibration_count=32,
        selection_offset=2,
    )

    assert calibration == tuple(tokens[:32])
    assert prompts == tuple(tokens[34:37])
    assert metadata["source_request_count"] == 40
    assert metadata["unique_evaluation_pool_count"] == 8
    assert metadata["selection_offset"] == 2
    assert len(metadata["calibration_input_hashes"]) == 32
    assert len(metadata["selected_input_hashes"]) == 3
    assert not set(metadata["calibration_input_hashes"]) & set(
        metadata["selected_input_hashes"]
    )

    with pytest.raises(ValueError, match="no-wrap evaluation pool"):
        load_unique_workload(
            dataset,
            manifest,
            context_length=2,
            request_count=7,
            calibration_count=32,
            selection_offset=2,
        )


def test_load_unique_workload_allows_1000_only_with_enough_unique_rows(
    tmp_path: Path,
) -> None:
    enough = [(index, index + 2000) for index in range(1032)]
    dataset, manifest = write_dataset(tmp_path / "enough", enough)

    prompts, calibration, _ = load_unique_workload(
        dataset,
        manifest,
        context_length=2,
        request_count=1000,
    )

    assert len(prompts) == 1000
    assert len(calibration) == 32

    short = [(index, index + 2000) for index in range(1031)]
    short_data, short_manifest = write_dataset(tmp_path / "short", short)
    with pytest.raises(ValueError, match="only 999.*1000 requested"):
        load_unique_workload(
            short_data,
            short_manifest,
            context_length=2,
            request_count=1000,
        )


def test_local_cli_builds_and_verifies_without_downloading(tmp_path: Path) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    source = tmp_path / "source.json"
    write_source(
        source,
        [
            "alpha beta gamma delta",
            "echo foxtrot golf hotel",
            "india juliet kilo lima",
        ],
    )
    vocabulary = {
        "[UNK]": 0,
        "</s>": 1,
        **{
            word: index + 2
            for index, word in enumerate(
                [
                    "alpha",
                    "beta",
                    "gamma",
                    "delta",
                    "echo",
                    "foxtrot",
                    "golf",
                    "hotel",
                    "india",
                    "juliet",
                    "kilo",
                    "lima",
                ]
            )
        },
    }
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocabulary, unk_token="[UNK]")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="</s>",
        chat_template=(
            "{% for message in messages %}"
            "{{ message['content'] if message['role'] == 'user' else '' }} "
            "{% endfor %}"
        ),
    )
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    tokenizer_files = {
        path.name: sha256_file(path)
        for path in tokenizer_dir.iterdir()
        if path.is_file()
    }
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": {
                    "repo_id": "fixture/sharegpt",
                    "revision": "fixture-revision",
                    "filename": source.name,
                    "sha256": sha256_file(source),
                },
                "tokenizer": {
                    "repo_id": "fixture/tokenizer",
                    "revision": "fixture-tokenizer-revision",
                    "files": tokenizer_files,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "decode.jsonl.zst"
    manifest = tmp_path / "decode-manifest.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_decode_corpus.py",
            "--source",
            str(source),
            "--tokenizer",
            str(tokenizer_dir),
            "--source-manifest",
            str(source_manifest),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--lengths",
            "4",
            "--unique-per-length",
            "2",
            "--seed",
            "17",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert completed.stderr == ""
    assert payload["record_count"] == 2
    assert payload["source_manifest_sha256"] == sha256(
        source_manifest.read_bytes()
    ).hexdigest()
    assert payload["source_provenance"]["revision"] == "fixture-revision"
    assert payload["source_provenance"]["cleaning"]["rows"] == 3
    assert (
        payload["tokenizer_provenance"]["revision"]
        == "fixture-tokenizer-revision"
    )
    assert payload["tokenizer_provenance"]["runtime"]["eos_token_id"] == 1
    assert output.stat().st_size < 100_000_000
    verify_decode_corpus(output, manifest)
