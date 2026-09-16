"""Public evidence cannot turn partial or incompatible runs into gains."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from flexmoe.analysis import parallel_report as report


@pytest.fixture
def raw() -> dict[str, Any]:
    from test_parallel_runner import FakeGPU, policy

    from flexmoe.bench.parallel_runner import ParallelConfig

    config = ParallelConfig(
        "ep4-dp2",
        Path("model"),
        Path("data"),
        Path("manifest"),
        batch_size=5,
        context_length=2,
        output_length=10,
    )
    workers = []
    for rank in range(2):
        gpu = object.__new__(FakeGPU)
        gpu.config, gpu.rank = config, rank
        workers.append(
            {**gpu.metadata(), "dp_rank": rank, "indices": [[0, 1, 2], [3, 4]][rank]}
        )
    ranks = [
        {
            "dp_rank": r,
            "indices": indices,
            "status": "complete",
            "request_count": len(indices),
            "generated_tokens": len(indices) * 10,
            "output_counts": [10] * len(indices),
            "output_hashes": ["a" * 64] * len(indices),
            "latencies_s": [0.2] * len(indices),
            "memory": workers[r]["memory"],
        }
        for r, indices in enumerate([[0, 1, 2], [3, 4]])
    ]
    return {
        "schema_version": 1,
        "artifact_kind": "parallel-run",
        "diagnostic_only": True,
        "formal_offload_gain": False,
        "deployment_gain_proven": False,
        "config": "ep4-dp2",
        "mode": "full-resident-native",
        "status": "complete",
        "timing_scope": "global_wallclock",
        "partitions": [[0, 1, 2], [3, 4]],
        "workers": workers,
        "requested_policy": policy(config),
        "contract": {
            "model_identity_sha256": "1" * 64,
            "input_sha256": "2" * 64,
            "dataset_sha256": "3" * 64,
            "dataset_manifest_sha256": "4" * 64,
            "commit": "5" * 40,
            "hardware_sha256": sha256(
                json.dumps(
                    [
                        {"uuid": f"GPU-{i}", "total_memory": 80_000_000_000}
                        for i in range(4)
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "versions": workers[0]["versions"],
            "batch_size": 5,
            "context_length": 2,
            "output_length": 10,
            "repetitions_requested": 3,
            "max_num_seqs": 1024,
            "max_num_batched_tokens": 8192,
            "seed": 20260912,
            "warmups": 1,
            "gpu_memory_utilization": 0.9,
            "dtype": "bfloat16",
            "calibration_count": 32,
            "no_repetition": True,
            "selected_input_hashes": [str(i) * 64 for i in range(5)],
        },
        "repetitions": [
            {
                "iteration": i,
                "status": "complete",
                "elapsed_s": 2,
                "generated_tokens": 50,
                "tokens_s": 25,
                "timing_scope": "global_wallclock",
                "latency": {
                    "scope": "arrival-to-finished-offline",
                    "count": 5,
                    "p50_s": 0.2,
                    "p95_s": 0.2,
                },
                "per_rank": copy.deepcopy(ranks),
            }
            for i in (1, 2, 3)
        ],
    }


def save(tmp_path: Path, raw: dict[str, Any], name: str = "summary.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(raw))
    return path


def test_global_summary_and_private_fields_stripped(raw: Any, tmp_path: Path) -> None:
    raw["error"] = "/secret/private/path"
    public = report.summarize_parallel(save(tmp_path, raw))
    assert public["status"] == "complete"
    assert public["median_tokens_s"] == 25
    assert public["timing_scope"] == "global_wallclock"
    assert len(public["memory"]) == 4
    serialized = json.dumps(public)
    for secret in (
        "GPU-",
        "output_hash",
        "latencies_s",
        "/secret",
        "selected_input",
        "aaaaaaa",
    ):
        assert secret not in serialized


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-rank",
        "duplicate-partition",
        "wrong-count",
        "wrong-policy",
        "missing-kv",
        "missing-memory",
        "nan",
        "partial",
        "bad-hash",
        "duplicate-device",
        "rank-failed",
        "two-repetitions",
    ],
)
def test_invalid_evidence_stays_failed(raw: Any, tmp_path: Path, mutation: str) -> None:
    rep = raw["repetitions"][0]
    if mutation == "missing-rank":
        rep["per_rank"].pop()
    elif mutation == "duplicate-partition":
        rep["per_rank"][1]["indices"] = [0, 1]
    elif mutation == "wrong-count":
        rep["per_rank"][0]["output_counts"] = [9, 10, 11]
    elif mutation == "wrong-policy":
        raw["workers"][0]["resolved_policy"]["parallel_config"][
            "data_parallel_size"
        ] = 1
    elif mutation == "missing-kv":
        raw["workers"][0]["memory"][0]["kv_cache_allocated_bytes"] = None
    elif mutation == "missing-memory":
        rep["per_rank"][0]["memory"] = []
    elif mutation == "nan":
        rep["elapsed_s"] = float("nan")
    elif mutation == "partial":
        raw["status"] = "failed"
    elif mutation == "bad-hash":
        rep["per_rank"][0]["output_hashes"] = ["private/path"] * 3
    elif mutation == "duplicate-device":
        raw["workers"][1]["devices"][0]["uuid"] = "GPU-0"
    elif mutation == "rank-failed":
        rep["per_rank"][0]["status"] = "failed"
    else:
        raw["repetitions"].pop()
    result = report.summarize_parallel(save(tmp_path, raw))
    assert result["status"] == "failed"
    assert result["median_tokens_s"] is None


def test_output_variation_audit_and_latency_unavailable(
    raw: Any, tmp_path: Path
) -> None:
    raw["repetitions"][1]["per_rank"][0]["output_hashes"][0] = "b" * 64
    for rep in raw["repetitions"]:
        for row in rep["per_rank"]:
            row["latencies_s"] = []
    result = report.summarize_parallel(save(tmp_path, raw))
    assert result["status"] == "complete"
    assert result["output_variation_observed"] is True
    assert result["repetitions"][0]["latency"]["p50_s"] is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("commit", "a" * 40),
        ("hardware_sha256", "b" * 64),
        ("input_sha256", "c" * 64),
        ("max_num_seqs", 2048),
    ],
)
def test_comparison_rejects_identity_or_global_budget_change(
    raw: Any, tmp_path: Path, key: str, value: Any
) -> None:
    first = save(tmp_path, raw, "one.json")
    raw["contract"][key] = value
    result = report.compare_parallel([first, save(tmp_path, raw, "two.json")])
    assert result["status"] == "ineligible"
    assert len(result["rows"]) == 2


def test_export_retains_missing_failed_rows_and_stdlib_cli(
    raw: Any, tmp_path: Path
) -> None:
    source = save(tmp_path, raw)
    missing = tmp_path / "tp4" / "summary.json"
    result = report.export_parallel([source, missing], tmp_path / "public")
    assert len(result["rows"]) == 2
    assert result["rows"][1]["status"] == "failed"
    assert result["rows"][1]["median_tokens_s"] is None
    assert (tmp_path / "public" / "parallel-summary.json").is_file()
    assert (tmp_path / "public" / "parallel-summary.md").is_file()
    completed = subprocess.run(
        [sys.executable, "-S", report.__file__, "--help"],
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0


def test_different_actual_hardware_cannot_reuse_claimed_digest(
    raw: Any, tmp_path: Path
) -> None:
    raw["workers"][1]["devices"][0]["uuid"] = "GPU-different"
    assert report.summarize_parallel(save(tmp_path, raw))["status"] == "failed"
