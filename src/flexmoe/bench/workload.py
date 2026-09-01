"""Exact deterministic request selection from the committed ShareGPT subset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from flexmoe.datasets.sharegpt import (
    PromptRecord,
    read_jsonl_zst,
    verify_subset,
)


@dataclass(frozen=True)
class Workload:
    requests: tuple[PromptRecord, ...]
    batch_size: int
    context_length: int
    dataset_sha256: str


def load_workload(
    *,
    dataset: Path,
    manifest: Path,
    batch_size: int,
    context_length: int,
) -> Workload:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive int")
    if type(context_length) is not int or context_length <= 0:
        raise ValueError("context_length must be a positive int")
    verify_subset(dataset, manifest)
    parsed = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError("dataset manifest must be a JSON object")
    manifest_data = cast(dict[str, object], parsed)
    digest = manifest_data.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("dataset manifest has an invalid SHA256")

    matching = tuple(
        record
        for record in read_jsonl_zst(dataset)
        if record.context_length == context_length
    )
    if len(matching) < batch_size:
        raise ValueError(
            f"only {len(matching)} requests have context {context_length}; "
            f"batch {batch_size} was requested"
        )
    return Workload(
        requests=matching[:batch_size],
        batch_size=batch_size,
        context_length=context_length,
        dataset_sha256=digest,
    )


__all__ = ["Workload", "load_workload"]
