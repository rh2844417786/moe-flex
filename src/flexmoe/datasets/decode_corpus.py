"""Unique, bounded decode-corpus construction from local ShareGPT data."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from flexmoe.datasets.sharegpt import (
    ChatTokenizerLike,
    PromptRecord,
    load_sharegpt_conversations,
    read_jsonl_zst,
    verify_subset,
    write_jsonl_zst,
)
from flexmoe.manifest import sha256_file

_MAX_OUTPUT_BYTES = 100_000_000


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive int")
    return value


def _token_hash(tokens: Sequence[int]) -> str:
    payload = json.dumps(list(tokens), separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def _validate_lengths(lengths: tuple[int, ...]) -> tuple[int, ...]:
    if not lengths or any(type(length) is not int or length < 1 for length in lengths):
        raise ValueError("lengths must contain positive ints")
    if len(set(lengths)) != len(lengths):
        raise ValueError("lengths must not contain duplicates")
    return lengths


def _manifest_provenance(
    source_manifest: Path,
    source: Path,
    local_source: dict[str, object],
    local_tokenizer: dict[str, object],
) -> dict[str, object]:
    parsed = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError("source manifest must be a JSON object")
    dataset = parsed.get("dataset")
    tokenizer = parsed.get("tokenizer")
    if not isinstance(dataset, dict) or not isinstance(tokenizer, dict):
        raise TypeError("source manifest must define dataset and tokenizer objects")
    if dataset.get("filename") != source.name:
        raise ValueError("source filename does not match source manifest")
    if dataset.get("sha256") != local_source["sha256"]:
        raise ValueError("source file SHA256 does not match source manifest")
    return {
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_provenance": {**dataset, "cleaning": local_source},
        "tokenizer_provenance": {**tokenizer, "runtime": local_tokenizer},
    }


def _promote_fresh(staged: Path, destination: Path) -> None:
    try:
        os.link(staged, destination)
    except FileExistsError as error:
        raise FileExistsError(
            "decode corpus output and manifest must be fresh paths"
        ) from error


def _build_unique_requests(
    conversations: Sequence[str],
    tokenizer: ChatTokenizerLike,
    lengths: tuple[int, ...],
    unique_per_length: int,
    seed: int,
) -> tuple[list[PromptRecord], dict[str, dict[str, int]]]:
    if not conversations:
        raise ValueError("source contains no accepted conversations")
    eos_token_id = tokenizer.eos_token_id
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise ValueError("tokenizer must define a nonnegative integer eos_token_id")
    order = list(range(len(conversations)))
    Random(seed).shuffle(order)
    records: list[PromptRecord] = []
    candidate_counts: dict[str, int] = {}
    duplicate_counts: dict[str, int] = {}
    consumed_counts: dict[str, int] = {}

    for target in lengths:
        cursor = 0
        candidates = 0
        duplicates = 0
        seen: dict[str, tuple[int, ...]] = {}
        while cursor < len(order) and len(seen) < unique_per_length:
            packed: list[int] = []
            while cursor < len(order) and len(packed) < target:
                source_index = order[cursor]
                cursor += 1
                encoded = tokenizer.encode(
                    conversations[source_index], add_special_tokens=False
                )
                if any(type(token) is not int or token < 0 for token in encoded):
                    raise ValueError("tokenizer returned an invalid token ID")
                packed.extend(encoded)
                packed.append(eos_token_id)
            if len(packed) < target:
                break
            candidates += 1
            tokens = tuple(packed[:target])
            digest = _token_hash(tokens)
            previous = seen.get(digest)
            if previous is not None:
                if previous != tokens:
                    raise RuntimeError("SHA256 collision between prompt token sequences")
                duplicates += 1
                continue
            seen[digest] = tokens
            records.append(PromptRecord(digest, tokens, target))

        if len(seen) < unique_per_length:
            raise ValueError(
                f"context {target}: only {len(seen)} unique prompts could be built "
                f"without source wrap; {unique_per_length} required"
            )
        key = str(target)
        candidate_counts[key] = candidates
        duplicate_counts[key] = duplicates
        consumed_counts[key] = cursor

    return records, {
        "candidates_by_context": candidate_counts,
        "duplicate_candidates_by_context": duplicate_counts,
        "source_rows_consumed_by_context": consumed_counts,
    }


def verify_decode_corpus(dataset: Path, manifest: Path) -> None:
    """Verify base subset integrity and the decode corpus no-repetition contract."""

    verify_subset(dataset, manifest)
    parsed = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError("decode corpus manifest must be a JSON object")
    if parsed.get("no_repetition") is not True:
        raise ValueError("decode corpus manifest must declare no_repetition=true")
    raw_unique = parsed.get("unique_by_context")
    if not isinstance(raw_unique, dict):
        raise TypeError("decode corpus manifest must define unique_by_context")

    records = read_jsonl_zst(dataset)
    hashes_by_context: dict[int, set[str]] = {}
    for record in records:
        digest = _token_hash(record.prompt_token_ids)
        bucket = hashes_by_context.setdefault(record.context_length, set())
        if digest in bucket:
            raise ValueError(
                f"duplicate prompt token hash in context {record.context_length}: "
                f"{digest}"
            )
        bucket.add(digest)
    actual_unique = {
        str(context): len(hashes)
        for context, hashes in sorted(hashes_by_context.items())
    }
    try:
        expected_unique = {
            str(context): int(count) for context, count in raw_unique.items()
        }
    except (TypeError, ValueError) as error:
        raise ValueError("unique_by_context must contain integer counts") from error
    if actual_unique != expected_unique:
        raise ValueError(
            "decode corpus unique_by_context mismatch: "
            f"expected {expected_unique}, got {actual_unique}"
        )


def prepare_decode_corpus(
    source: Path,
    tokenizer: ChatTokenizerLike,
    output: Path,
    manifest_path: Path,
    *,
    lengths: tuple[int, ...] = (1024, 4096),
    unique_per_length: int = 1536,
    seed: int = 20260912,
    source_manifest: Path | None = None,
) -> dict[str, object]:
    """Build and atomically promote a deterministic, unique local corpus."""

    destinations = {
        source.resolve(),
        output.resolve(),
        manifest_path.resolve(),
    }
    if len(destinations) != 3:
        raise ValueError("source, output, and manifest paths must be distinct")
    if os.path.lexists(output) or os.path.lexists(manifest_path):
        raise FileExistsError(
            "decode corpus output and manifest must be fresh paths"
        )
    context_lengths = _validate_lengths(lengths)
    per_length = _positive_int(unique_per_length, "unique_per_length")
    if type(seed) is not int:
        raise ValueError("seed must be an int")
    load_result = load_sharegpt_conversations(source, tokenizer)
    records, generation = _build_unique_requests(
        load_result.conversations,
        tokenizer,
        context_lengths,
        per_length,
        seed,
    )
    counts = Counter(record.context_length for record in records)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    local_source: dict[str, object] = {
        "filename": source.name,
        "sha256": sha256_file(source),
        "rows": load_result.rows_seen,
        "accepted_rows": load_result.rows_accepted,
        "skipped_by_reason": dict(load_result.skipped_by_reason),
    }
    local_tokenizer: dict[str, object] = {
        "implementation": (
            f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
        ),
        "eos_token_id": tokenizer.eos_token_id,
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "sha256": "",
        "record_count": len(records),
        "counts_by_context": {
            str(context): count for context, count in sorted(counts.items())
        },
        "unique_by_context": {
            str(context): count for context, count in sorted(counts.items())
        },
        "no_repetition": True,
        "sampling_seed": seed,
        "packing_policy": "shuffle-pack-eos-truncate",
        "packing": {
            "policy": "shuffle-pack-eos-truncate",
            "order": "seeded-source-row-shuffle",
            "overflow": "discarded-per-candidate",
        },
        "deduplication": {
            "scope": "per-context-length",
            "key": "sha256-full-prompt-token-ids",
        },
        "candidate_generation": {
            "bounded": True,
            "source_wrap": False,
            **generation,
        },
        "source_provenance": local_source,
        "tokenizer_provenance": local_tokenizer,
    }
    if source_manifest is not None:
        manifest.update(
            _manifest_provenance(
                source_manifest, source, local_source, local_tokenizer
            )
        )

    with (
        TemporaryDirectory(prefix=".decode-corpus-", dir=output.parent) as data_tmp,
        TemporaryDirectory(
            prefix=".decode-manifest-", dir=manifest_path.parent
        ) as manifest_tmp,
    ):
        staged_output = Path(data_tmp) / output.name
        staged_manifest = Path(manifest_tmp) / manifest_path.name
        manifest["sha256"] = write_jsonl_zst(records, staged_output)
        staged_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        verify_decode_corpus(staged_output, staged_manifest)
        if staged_output.stat().st_size >= _MAX_OUTPUT_BYTES:
            raise ValueError("generated decode corpus exceeds GitHub 100MB file limit")
        _promote_fresh(staged_output, output)
        try:
            _promote_fresh(staged_manifest, manifest_path)
        except BaseException:
            output.unlink()
            raise
    return manifest


def load_unique_workload(
    dataset: Path,
    manifest: Path,
    *,
    context_length: int,
    request_count: int,
    calibration_count: int = 32,
    selection_offset: int = 0,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    dict[str, object],
]:
    """Load a disjoint calibration/evaluation split without cycling inputs."""

    context = _positive_int(context_length, "context_length")
    requested = _positive_int(request_count, "request_count")
    if type(calibration_count) is not int or calibration_count < 0:
        raise ValueError("calibration_count must be a nonnegative int")
    if type(selection_offset) is not int or selection_offset < 0:
        raise ValueError("selection_offset must be a nonnegative int")
    verify_decode_corpus(dataset, manifest)

    source = [
        record.prompt_token_ids
        for record in read_jsonl_zst(dataset)
        if record.context_length == context
    ]
    unique: dict[str, tuple[int, ...]] = {}
    for prompt in source:
        unique.setdefault(_token_hash(prompt), prompt)
    ordered_hashes = tuple(unique)
    if len(ordered_hashes) < calibration_count:
        raise ValueError(
            f"only {len(ordered_hashes)} unique prompts are available for "
            f"{calibration_count} calibration rows"
        )
    calibration_hashes = ordered_hashes[:calibration_count]
    evaluation_hashes = ordered_hashes[calibration_count:]
    available = len(evaluation_hashes) - selection_offset
    if available < requested:
        raise ValueError(
            f"no-wrap evaluation pool has only {max(0, available)} prompts after "
            f"selection_offset={selection_offset}; {requested} requested"
        )
    selected_hashes = evaluation_hashes[
        selection_offset : selection_offset + requested
    ]
    calibration = tuple(unique[digest] for digest in calibration_hashes)
    prompts = tuple(unique[digest] for digest in selected_hashes)
    if set(calibration_hashes) & set(selected_hashes):
        raise ValueError("calibration/evaluation input overlap")

    parsed = json.loads(manifest.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    metadata: dict[str, object] = {
        "dataset_sha256": parsed["sha256"],
        "dataset_manifest_sha256": sha256_file(manifest),
        "context_length": context,
        "source_request_count": len(source),
        "unique_source_count": len(unique),
        "calibration_count": calibration_count,
        "unique_evaluation_pool_count": len(evaluation_hashes),
        "selection_offset": selection_offset,
        "request_count": requested,
        "calibration_input_hashes": list(calibration_hashes),
        "selected_input_hashes": list(selected_hashes),
        "no_repetition": True,
        "sampling_policy": "file-order-first-unique-no-wrap",
        "split_scope": "full-prompt-token-hashes",
    }
    return prompts, calibration, metadata


__all__ = [
    "load_unique_workload",
    "prepare_decode_corpus",
    "verify_decode_corpus",
]
