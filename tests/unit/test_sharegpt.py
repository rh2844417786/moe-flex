import json
import subprocess
import sys
from pathlib import Path

import pytest

from flexmoe.datasets.sharegpt import (
    build_fixed_requests,
    load_sharegpt_conversations,
    main,
    prepare_subset,
    verify_subset,
    write_jsonl_zst,
)


class TinyTokenizer:
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) % 251 + 1 for char in text]


class TinyChatTokenizer(TinyTokenizer):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        return "|".join(
            f"{message['role']}:{message['content']}" for message in conversation
        )


def test_build_fixed_requests_is_exact_and_deterministic() -> None:
    conversations = [f"conversation-{index}-" * 8 for index in range(20)]

    first = build_fixed_requests(conversations, TinyTokenizer(), (32, 64), 3, 7)
    second = build_fixed_requests(conversations, TinyTokenizer(), (32, 64), 3, 7)

    assert first == second
    assert [len(item.prompt_token_ids) for item in first] == [32] * 3 + [64] * 3
    assert len({item.request_id for item in first}) == 6


@pytest.mark.parametrize(
    ("conversations", "lengths", "per_length", "message"),
    [
        ([], (32,), 1, "conversations must be non-empty"),
        (["valid"], (), 1, "lengths must contain positive values"),
        (["valid"], (0,), 1, "lengths must contain positive values"),
        (["valid"], (32,), 0, "per_length must be positive"),
    ],
)
def test_build_fixed_requests_rejects_invalid_shape(
    conversations: list[str],
    lengths: tuple[int, ...],
    per_length: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_fixed_requests(conversations, TinyTokenizer(), lengths, per_length, 7)


def test_build_fixed_requests_requires_eos_token() -> None:
    tokenizer = TinyTokenizer()
    tokenizer.eos_token_id = None

    with pytest.raises(ValueError, match="tokenizer must define eos_token_id"):
        build_fixed_requests(["valid"], tokenizer, (32,), 1, 7)


def test_load_sharegpt_conversations_applies_chat_template() -> None:
    source = Path("tests/fixtures/sharegpt_mini.json")

    result = load_sharegpt_conversations(source, TinyChatTokenizer())

    assert result.conversations == (
        (
            "user:Explain sparse experts briefly.|assistant:Only selected experts "
            "process each token."
        ),
        (
            "user:Why does KV cache consume memory?|assistant:It stores attention keys "
            "and values for prior tokens."
        ),
    )
    assert result.rows_seen == 2
    assert result.rows_accepted == 2
    assert result.skipped_by_reason == ()


def test_load_sharegpt_conversations_skips_unknown_role(tmp_path: Path) -> None:
    source = tmp_path / "bad.json"
    source.write_text(
        json.dumps(
            [
                {
                    "id": "bad",
                    "conversations": [{"from": "system", "value": "unsafe"}],
                }
            ]
        )
    )

    result = load_sharegpt_conversations(source, TinyChatTokenizer())

    assert result.conversations == ()
    assert result.skipped_by_reason == (("unsupported_role", 1),)


def test_load_sharegpt_conversations_requires_list_root(tmp_path: Path) -> None:
    source = tmp_path / "not-a-list.json"
    source.write_text(json.dumps({"not": "a list"}))

    with pytest.raises(TypeError, match="ShareGPT root must be a list"):
        load_sharegpt_conversations(source, TinyChatTokenizer())


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ([42], "bad_row_type"),
        (
            [{"conversations": [42]}],
            "bad_turn_type",
        ),
    ],
)
def test_load_sharegpt_conversations_skips_wrong_types(
    tmp_path: Path, payload: object, reason: str
) -> None:
    source = tmp_path / "wrong-type.json"
    source.write_text(json.dumps(payload))

    result = load_sharegpt_conversations(source, TinyChatTokenizer())

    assert result.conversations == ()
    assert result.skipped_by_reason == ((reason, 1),)


def test_load_sharegpt_conversations_reports_cleaning_reasons(tmp_path: Path) -> None:
    source = tmp_path / "dirty.json"
    source.write_text(
        json.dumps(
            [
                {"conversations": []},
                {"conversations": [{"from": "human", "value": ""}]},
                {"conversations": [{"from": "gpt", "value": "starts wrong"}]},
                {
                    "conversations": [
                        {"from": "human", "value": "first"},
                        {"from": "human", "value": "not alternating"},
                    ]
                },
                {
                    "conversations": [
                        {"from": "human", "value": "valid"},
                        {"from": "gpt", "value": "answer"},
                    ]
                },
            ]
        )
    )

    result = load_sharegpt_conversations(source, TinyChatTokenizer())

    assert result.rows_seen == 5
    assert result.rows_accepted == 1
    assert result.skipped_by_reason == (
        ("empty_or_bad_list", 1),
        ("empty_value", 1),
        ("non_alternating", 1),
        ("starts_non_human", 1),
    )


def test_write_and_verify_subset_round_trip(tmp_path: Path) -> None:
    records = build_fixed_requests(
        ["alpha" * 20, "beta" * 20], TinyTokenizer(), (8, 16), 1, 11
    )
    subset = tmp_path / "subset.jsonl.zst"
    digest = write_jsonl_zst(records, subset)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sha256": digest,
                "record_count": 2,
                "counts_by_context": {"8": 1, "16": 1},
            }
        )
    )

    verify_subset(subset, manifest)


def test_verify_subset_rejects_wrong_hash(tmp_path: Path) -> None:
    records = build_fixed_requests(["alpha" * 20], TinyTokenizer(), (8,), 1, 11)
    subset = tmp_path / "subset.jsonl.zst"
    write_jsonl_zst(records, subset)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sha256": "0" * 64,
                "record_count": 1,
                "counts_by_context": {"8": 1},
            }
        )
    )

    with pytest.raises(ValueError, match="subset SHA256 mismatch"):
        verify_subset(subset, manifest)


def test_prepare_subset_writes_verifiable_manifest(tmp_path: Path) -> None:
    source = Path("tests/fixtures/sharegpt_mini.json")
    subset = tmp_path / "subset.jsonl.zst"
    manifest = tmp_path / "manifest.json"

    generated = prepare_subset(
        source=source,
        tokenizer=TinyChatTokenizer(),
        output=subset,
        manifest_path=manifest,
        lengths=(16, 32),
        per_length=2,
        seed=19,
    )

    assert generated["record_count"] == 4
    assert generated["counts_by_context"] == {"16": 2, "32": 2}
    assert generated == json.loads(manifest.read_text())
    verify_subset(subset, manifest)


def test_verify_cli_returns_success_for_valid_subset(tmp_path: Path) -> None:
    records = build_fixed_requests(["alpha" * 20], TinyTokenizer(), (8,), 1, 11)
    subset = tmp_path / "subset.jsonl.zst"
    digest = write_jsonl_zst(records, subset)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sha256": digest,
                "record_count": 1,
                "counts_by_context": {"8": 1},
            }
        )
    )

    assert main(["verify", str(subset), str(manifest)]) == 0


def test_module_verify_cli_has_no_runtime_warning(tmp_path: Path) -> None:
    records = build_fixed_requests(["alpha" * 20], TinyTokenizer(), (8,), 1, 11)
    subset = tmp_path / "subset.jsonl.zst"
    digest = write_jsonl_zst(records, subset)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sha256": digest,
                "record_count": 1,
                "counts_by_context": {"8": 1},
            }
        )
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "flexmoe.datasets.sharegpt",
            "verify",
            str(subset),
            str(manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stderr == ""
