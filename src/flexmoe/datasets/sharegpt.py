"""Deterministic ShareGPT subset construction."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import TextIOWrapper
from pathlib import Path
from random import Random
from typing import Protocol

import zstandard as zstd

from flexmoe.manifest import sha256_file


class TokenizerLike(Protocol):
    """Minimum tokenizer behavior required by the subset builder."""

    eos_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text into token IDs."""


class ChatTokenizerLike(TokenizerLike, Protocol):
    """Tokenizer behavior required to render ShareGPT conversations."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        """Render one chat without tokenizing it."""


@dataclass(frozen=True)
class PromptRecord:
    """One fixed-length tokenized benchmark request."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    context_length: int


@dataclass(frozen=True)
class ConversationLoadResult:
    """Accepted rendered conversations and deterministic cleaning statistics."""

    conversations: tuple[str, ...]
    rows_seen: int
    rows_accepted: int
    skipped_by_reason: tuple[tuple[str, int], ...]


def build_fixed_requests(
    conversations: Sequence[str],
    tokenizer: TokenizerLike,
    lengths: tuple[int, ...],
    per_length: int,
    seed: int,
) -> list[PromptRecord]:
    """Pack deterministic conversation order into exact token lengths."""

    if not conversations:
        raise ValueError("conversations must be non-empty")
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("lengths must contain positive values")
    if per_length <= 0:
        raise ValueError("per_length must be positive")
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    order = list(range(len(conversations)))
    Random(seed).shuffle(order)
    records: list[PromptRecord] = []
    cursor = 0

    for target in lengths:
        for sample_index in range(per_length):
            packed: list[int] = []
            source_ids: list[int] = []
            while len(packed) < target:
                source_index = order[cursor % len(order)]
                cursor += 1
                source_ids.append(source_index)
                packed.extend(
                    tokenizer.encode(
                        conversations[source_index], add_special_tokens=False
                    )
                )
                packed.append(eos_token_id)
            request_id = sha256(
                f"{seed}:{target}:{sample_index}:{source_ids}".encode()
            ).hexdigest()[:20]
            records.append(
                PromptRecord(
                    request_id=request_id,
                    prompt_token_ids=tuple(packed[:target]),
                    context_length=target,
                )
            )

    return records


def load_sharegpt_conversations(
    source: Path, tokenizer: ChatTokenizerLike
) -> ConversationLoadResult:
    """Render strictly alternating human/GPT rows and report skipped rows."""

    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError("ShareGPT root must be a list")

    role_map = {"human": "user", "gpt": "assistant"}
    rendered: list[str] = []
    skipped: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            skipped["bad_row_type"] += 1
            continue
        turns = row.get("conversations")
        if not isinstance(turns, list) or not turns:
            skipped["empty_or_bad_list"] += 1
            continue
        if any(not isinstance(turn, dict) for turn in turns):
            skipped["bad_turn_type"] += 1
            continue
        if any(turn.get("from") not in role_map for turn in turns):
            skipped["unsupported_role"] += 1
            continue
        if any(
            not isinstance(turn.get("value"), str)
            or not turn["value"].strip()
            for turn in turns
        ):
            skipped["empty_value"] += 1
            continue
        if turns[0]["from"] != "human":
            skipped["starts_non_human"] += 1
            continue
        if any(
            turn["from"] != ("human" if turn_index % 2 == 0 else "gpt")
            for turn_index, turn in enumerate(turns)
        ):
            skipped["non_alternating"] += 1
            continue

        messages: list[dict[str, str]] = []
        for turn in turns:
            source_role = turn.get("from")
            value = turn.get("value")
            assert isinstance(source_role, str)
            assert isinstance(value, str)
            messages.append({"role": role_map[source_role], "content": value})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        if not isinstance(text, str) or not text:
            raise ValueError(f"chat template returned empty text for row {row_index}")
        rendered.append(text)
    return ConversationLoadResult(
        conversations=tuple(rendered),
        rows_seen=len(rows),
        rows_accepted=len(rendered),
        skipped_by_reason=tuple(sorted(skipped.items())),
    )


def write_jsonl_zst(records: Iterable[PromptRecord], path: Path) -> str:
    """Write compact JSONL through a deterministic zstd compressor."""

    path.parent.mkdir(parents=True, exist_ok=True)
    compressor = zstd.ZstdCompressor(level=19, threads=0)
    with (
        path.open("wb") as raw,
        compressor.stream_writer(raw, closefd=False) as stream,
    ):
        for record in records:
            payload = json.dumps(
                asdict(record), sort_keys=True, separators=(",", ":")
            )
            stream.write(f"{payload}\n".encode())
    return sha256_file(path)


def read_jsonl_zst(path: Path) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    decompressor = zstd.ZstdDecompressor()
    with (
        path.open("rb") as raw,
        decompressor.stream_reader(raw, closefd=False) as reader,
        TextIOWrapper(reader, encoding="utf-8") as text,
    ):
        for line_number, line in enumerate(text, start=1):
            payload = json.loads(line)
            try:
                record = PromptRecord(
                    request_id=str(payload["request_id"]),
                    prompt_token_ids=tuple(
                        int(token) for token in payload["prompt_token_ids"]
                    ),
                    context_length=int(payload["context_length"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid subset record at line {line_number}"
                ) from error
            records.append(record)
    return records


def verify_subset(path: Path, manifest_path: Path) -> None:
    """Verify content hash, record uniqueness, lengths, and context counts."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(path)
    if actual_hash != manifest.get("sha256"):
        raise ValueError("subset SHA256 mismatch")

    records = read_jsonl_zst(path)
    expected_count = int(manifest["record_count"])
    if len(records) != expected_count:
        raise ValueError(
            f"subset record count mismatch: expected {expected_count}, got {len(records)}"
        )
    request_ids = [record.request_id for record in records]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("subset request IDs must be unique")
    for record in records:
        if len(record.prompt_token_ids) != record.context_length:
            raise ValueError(f"context length mismatch for {record.request_id}")

    actual_counts = Counter(record.context_length for record in records)
    expected_counts = {
        int(context): int(count)
        for context, count in manifest["counts_by_context"].items()
    }
    if dict(sorted(actual_counts.items())) != dict(sorted(expected_counts.items())):
        raise ValueError("subset context counts mismatch")


def prepare_subset(
    source: Path,
    tokenizer: ChatTokenizerLike,
    output: Path,
    manifest_path: Path,
    lengths: tuple[int, ...],
    per_length: int,
    seed: int,
) -> dict[str, object]:
    """Generate a fixed subset from a local source and write its manifest."""

    load_result = load_sharegpt_conversations(source, tokenizer)
    records = build_fixed_requests(
        conversations=load_result.conversations,
        tokenizer=tokenizer,
        lengths=lengths,
        per_length=per_length,
        seed=seed,
    )
    subset_hash = write_jsonl_zst(records, output)
    counts = Counter(record.context_length for record in records)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "sha256": subset_hash,
        "source_sha256": sha256_file(source),
        "record_count": len(records),
        "counts_by_context": {
            str(context): count for context, count in sorted(counts.items())
        },
        "sampling_seed": seed,
        "packing_policy": "shuffle-pack-eos-truncate",
        "source_rows": load_result.rows_seen,
        "accepted_rows": load_result.rows_accepted,
        "skipped_by_reason": dict(load_result.skipped_by_reason),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify_subset(output, manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Run dataset verification from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify", help="verify a fixed subset")
    verify_parser.add_argument("subset", type=Path)
    verify_parser.add_argument("manifest", type=Path)
    arguments = parser.parse_args(argv)

    if arguments.command == "verify":
        verify_subset(arguments.subset, arguments.manifest)
        return 0
    raise AssertionError(f"unhandled command {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
