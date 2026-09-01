from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from flexmoe.bench.runner import (
    create_run_directory,
    load_run_config,
    variant_environment,
)


def test_load_run_config_enforces_formal_contract(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
schema_version: 1
variant: fluxmoe-fixed
model_path: /mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct
dataset_path: benchmarks/data/sharegpt/qwen3next_1024_requests.jsonl.zst
dataset_manifest: benchmarks/data/sharegpt/dataset_manifest.json
dataset_sha256: ae5e2428733c21f39efdd1c6ba45d9dac78e4f00d43a0a930151034eda178445
batch_size: 32
context_length: 1024
output_length: 128
tensor_parallel_size: 4
dtype: bfloat16
greedy: true
enforce_eager: true
gpu_memory_utilization: 0.60
warmups: 3
repetitions: 3
seed: 20260901
gpu_compressed_budget_bytes: 67108864
host_capacity_bytes: 214748364800
gpu_decode_bytes_per_second: 100000000000.0
host_h2d_bytes_per_second: 25000000000.0
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.variant == "fluxmoe-fixed"
    assert config.tensor_parallel_size == 4
    assert config.enforce_eager
    assert config.gpu_memory_utilization == 0.60


def test_run_directory_is_immutable_and_sha_named(tmp_path: Path) -> None:
    timestamp = datetime(2026, 9, 1, 12, 30, tzinfo=timezone.utc)

    run_dir = create_run_directory(tmp_path, "a" * 40, timestamp)

    assert run_dir.name == "20260901T123000Z-aaaaaaaaaaaa"
    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path, "a" * 40, timestamp)


def test_variant_environment_is_explicit_and_fail_closed() -> None:
    assert variant_environment("resident") == {"FLUXMOE_ENABLE": "0"}
    assert variant_environment("fluxmoe-fixed") == {
        "FLUXMOE_ENABLE": "1",
        "FLUXMOE_PLANNER_MODE": "fixed",
        "FLUXMOE_VERIFY_WEIGHTS": "1",
    }
    with pytest.raises(ValueError, match="unsupported benchmark variant"):
        variant_environment("unknown")
