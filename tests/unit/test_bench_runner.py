from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from flexmoe.bench.runner import (
    RunConfig,
    _aggregate_worker_counters,
    compare_reference,
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
    }
    with pytest.raises(ValueError, match="unsupported benchmark variant"):
        variant_environment("unknown")


def test_reference_comparison_checks_tokens_router_and_delta(tmp_path: Path) -> None:
    reference = tmp_path / "resident"
    reference.mkdir()
    config = RunConfig(
        schema_version=1,
        variant="fluxmoe-fixed",
        model_path=Path("/model"),
        dataset_path=Path("dataset.zst"),
        dataset_manifest=Path("manifest.json"),
        dataset_sha256="d" * 64,
        batch_size=4,
        context_length=1024,
        output_length=16,
        tensor_parallel_size=4,
        dtype="bfloat16",
        greedy=True,
        enforce_eager=True,
        gpu_memory_utilization=0.6,
        warmups=3,
        repetitions=3,
        seed=7,
        gpu_compressed_budget_bytes=1,
        host_capacity_bytes=100,
        gpu_decode_bytes_per_second=2.0,
        host_h2d_bytes_per_second=1.0,
    )
    reference_config = {
        "variant": "resident",
        "model_path": "/model",
        "dataset_sha256": "d" * 64,
        "batch_size": 4,
        "context_length": 1024,
        "output_length": 16,
        "tensor_parallel_size": 4,
        "dtype": "bfloat16",
        "seed": 7,
    }
    repetitions = [
        {"output_tokens_per_second": value, "output_token_ids": [[1, 2]]}
        for value in (90.0, 100.0, 110.0)
    ]
    reference_router = {
        "rank-0.jsonl": {
            "sha256": "e" * 64,
            "line_count": 2,
            "probe_sha256": "a" * 64,
            "probe_line_count": 1,
        }
    }
    router_manifest = {
        "rank-0.jsonl": {
            "sha256": "f" * 64,
            "line_count": 2,
            "probe_sha256": "a" * 64,
            "probe_line_count": 1,
        }
    }
    (reference / "config.json").write_text(
        json.dumps(reference_config), encoding="utf-8"
    )
    (reference / "metrics.json").write_text(
        json.dumps({"repetitions": repetitions, "router_trace": reference_router}),
        encoding="utf-8",
    )
    current = {
        "repetitions": [
            {"output_tokens_per_second": value, "output_token_ids": [[1, 2]]}
            for value in (108.0, 120.0, 132.0)
        ]
    }

    tokens_match, router_match, delta = compare_reference(
        config, current, router_manifest, reference
    )

    assert tokens_match and router_match
    assert current["performance_output_tokens_match"] is True
    assert current["router_full_trace_match"] is False
    assert delta == pytest.approx(0.2)


def test_reference_comparison_records_performance_token_mismatch(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "resident"
    reference.mkdir()
    config = RunConfig(
        schema_version=1,
        variant="fluxmoe-fixed",
        model_path=Path("/model"),
        dataset_path=Path("dataset.zst"),
        dataset_manifest=Path("manifest.json"),
        dataset_sha256="d" * 64,
        batch_size=4,
        context_length=1024,
        output_length=16,
        tensor_parallel_size=4,
        dtype="bfloat16",
        greedy=True,
        enforce_eager=True,
        gpu_memory_utilization=0.6,
        warmups=3,
        repetitions=1,
        seed=7,
        gpu_compressed_budget_bytes=1,
        host_capacity_bytes=100,
        gpu_decode_bytes_per_second=2.0,
        host_h2d_bytes_per_second=1.0,
    )
    (reference / "config.json").write_text(
        json.dumps(
            {
                "variant": "resident",
                "model_path": "/model",
                "dataset_sha256": "d" * 64,
                "batch_size": 4,
                "context_length": 1024,
                "output_length": 16,
                "tensor_parallel_size": 4,
                "dtype": "bfloat16",
                "seed": 7,
            }
        ),
        encoding="utf-8",
    )
    (reference / "metrics.json").write_text(
        json.dumps(
            {
                "repetitions": [
                    {"output_tokens_per_second": 10.0, "output_token_ids": [[1]]}
                ],
                "router_trace": {},
            }
        ),
        encoding="utf-8",
    )
    current: dict[str, object] = {
        "repetitions": [
            {"output_tokens_per_second": 10.0, "output_token_ids": [[2]]}
        ]
    }

    tokens_match, _, _ = compare_reference(config, current, {}, reference)

    assert tokens_match is False
    assert current["performance_output_tokens_match"] is False


def test_sampling_seed_is_part_of_reproducibility_contract() -> None:
    source = Path(__file__).parents[2] / "src/flexmoe/bench/runner.py"
    assert "seed=config.seed" in source.read_text(encoding="utf-8")


def test_worker_mechanism_counters_are_aggregated_across_tp_ranks() -> None:
    counters = {
        "mapped_bytes": 10,
        "mapping_count": 2,
        "h2d_bytes": 20,
        "decompressed_bytes": 30,
        "weights_verified": 48,
        "weights_expected": 48,
    }

    totals = _aggregate_worker_counters(
        [dict(counters) for _ in range(4)], expected_workers=4
    )

    assert totals == {
        "mapped_bytes": 40,
        "mapping_count": 8,
        "h2d_bytes": 80,
        "decompressed_bytes": 120,
        "weights_verified": 192,
        "weights_expected": 192,
    }


def test_worker_mechanism_counters_fail_closed() -> None:
    counters = {
        "mapped_bytes": 10,
        "mapping_count": 2,
        "h2d_bytes": 20,
        "decompressed_bytes": 30,
        "weights_verified": 48,
        "weights_expected": 48,
    }

    with pytest.raises(RuntimeError, match="expected 4"):
        _aggregate_worker_counters([counters], expected_workers=4)
    invalid = [dict(counters) for _ in range(4)]
    invalid[2].pop("weights_verified")
    with pytest.raises(RuntimeError, match="unexpected fields"):
        _aggregate_worker_counters(invalid, expected_workers=4)
