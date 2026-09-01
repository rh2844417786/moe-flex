from __future__ import annotations

import json
from pathlib import Path

import pytest

from flexmoe.bench.workload import load_workload
from flexmoe.datasets.sharegpt import PromptRecord, write_jsonl_zst


def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    records = [
        PromptRecord(f"r{length}-{index}", tuple(range(length)), length)
        for length in (8, 16)
        for index in range(3)
    ]
    dataset = tmp_path / "requests.jsonl.zst"
    digest = write_jsonl_zst(records, dataset)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sha256": digest,
                "record_count": 6,
                "counts_by_context": {"8": 3, "16": 3},
            }
        ),
        encoding="utf-8",
    )
    return dataset, manifest


def test_load_workload_selects_exact_batch_and_context(tmp_path: Path) -> None:
    dataset, manifest = _dataset(tmp_path)

    workload = load_workload(
        dataset=dataset,
        manifest=manifest,
        batch_size=2,
        context_length=16,
    )

    assert len(workload.requests) == 2
    assert all(len(request.prompt_token_ids) == 16 for request in workload.requests)
    assert workload.dataset_sha256 == json.loads(manifest.read_text())["sha256"]


def test_load_workload_rejects_insufficient_matching_requests(
    tmp_path: Path,
) -> None:
    dataset, manifest = _dataset(tmp_path)

    with pytest.raises(ValueError, match="only 3 requests"):
        load_workload(
            dataset=dataset,
            manifest=manifest,
            batch_size=4,
            context_length=8,
        )
