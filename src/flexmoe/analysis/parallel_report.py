"""Standard-library-only native parallel evidence validation and public export."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

CONFIGS = {"tp4": (4, 1, False), "ep4-dp4": (1, 4, True), "ep4-dp2": (2, 2, True)}
MEMORY = (
    "rank",
    "total_gpu_bytes",
    "free_gpu_bytes",
    "torch_allocated_bytes",
    "torch_reserved_bytes",
    "torch_peak_allocated_bytes",
    "torch_peak_reserved_bytes",
    "available_kv_cache_bytes",
    "kv_cache_allocated_bytes",
    "kv_cache_declared_bytes",
    "num_gpu_blocks",
    "model_memory_bytes",
)
IDENTITY = (
    "model_identity_sha256",
    "input_sha256",
    "dataset_sha256",
    "dataset_manifest_sha256",
    "commit",
    "hardware_sha256",
    "versions",
    "batch_size",
    "context_length",
    "output_length",
    "max_num_seqs",
    "max_num_batched_tokens",
    "seed",
    "warmups",
    "gpu_memory_utilization",
    "dtype",
    "calibration_count",
    "no_repetition",
)


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _hash(value: Any, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value) is not None
    )


def _read(source: Path) -> dict[str, Any]:
    value = json.loads(
        (source / "summary.json" if source.is_dir() else source).read_text()
    )
    if not isinstance(value, dict):
        raise TypeError("invalid root")
    return value


def _memory(rows: Any, tp: int) -> None:
    if (
        not isinstance(rows, list)
        or len(rows) != tp
        or len({r["rank"] for r in rows}) != tp
    ):
        raise ValueError("missing memory")
    for row in rows:
        for key in (
            "rank",
            "total_gpu_bytes",
            "free_gpu_bytes",
            "kv_cache_allocated_bytes",
            "num_gpu_blocks",
        ):
            if type(row.get(key)) is not int or row[key] < (0 if key == "rank" else 1):
                raise ValueError("missing memory")
        if (
            row["free_gpu_bytes"] < 2_000_000_000
            or row.get("kv_cache_accounting_consistent") is not True
        ):
            raise ValueError("unsafe memory")


def _policy(raw: Any, contract: Any, tp: int, dp: int, ep: bool) -> None:
    expected = {
        "model_config": {
            "enforce_eager": False,
            "quantization": None,
            "seed": 20260912,
            "max_model_len": contract["context_length"] + contract["output_length"],
        },
        "cache_config": {
            "enable_prefix_caching": False,
            "cpu_offload_gb": 0,
            "swap_space_bytes": 0,
            "gpu_memory_utilization": 0.9,
        },
        "scheduler_config": {
            "max_num_seqs": contract["max_num_seqs"] // dp,
            "max_num_batched_tokens": contract["max_num_batched_tokens"] // dp,
            "enable_chunked_prefill": True,
        },
        "parallel_config": {
            "tensor_parallel_size": tp,
            "data_parallel_size": dp,
            "pipeline_parallel_size": 1,
            "enable_expert_parallel": ep,
            "enable_eplb": False,
            "disable_custom_all_reduce": False,
        },
    }
    for group, fields in expected.items():
        for name, value in fields.items():
            if name not in raw.get(group, {}) or raw[group][name] != value:
                raise ValueError("resolved policy mismatch")
    if raw["model_config"].get("dtype") not in ("bfloat16", "torch.bfloat16"):
        raise ValueError("wrong dtype")
    compile_config = raw["compilation_config"]
    if (
        compile_config.get("level") in (None, 0, "0")
        or compile_config.get("cudagraph_mode") is None
    ):
        raise ValueError("native optimization unavailable")


def _validate(raw: Mapping[str, Any]) -> None:
    if (
        raw["artifact_kind"] != "parallel-run"
        or raw["schema_version"] != 1
        or raw["status"] != "complete"
        or raw["mode"] != "full-resident-native"
        or raw["timing_scope"] != "global_wallclock"
    ):
        raise ValueError("incomplete artifact")
    tp, dp, ep = CONFIGS[raw["config"]]
    c = raw["contract"]
    for key in IDENTITY:
        if key not in c or c[key] is None:
            raise ValueError("missing identity")
    for key in (
        "batch_size",
        "context_length",
        "output_length",
        "max_num_seqs",
        "max_num_batched_tokens",
    ):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError("invalid budget")
    if (
        c["max_num_seqs"] % dp
        or c["max_num_batched_tokens"] % dp
        or c["batch_size"] < dp
    ):
        raise ValueError("invalid global division")
    for key in (
        "model_identity_sha256",
        "input_sha256",
        "dataset_sha256",
        "dataset_manifest_sha256",
        "hardware_sha256",
    ):
        if not _hash(c[key]):
            raise ValueError("invalid identity digest")
    if (
        not _hash(c["commit"], 40)
        or c["versions"].get("vllm", "").split("+")[0] != "0.10.2"
    ):
        raise ValueError("invalid runtime identity")
    if (
        c["dtype"] != "bfloat16"
        or c["seed"] != 20260912
        or c["warmups"] != 1
        or c["calibration_count"] != 32
        or c["no_repetition"] is not True
        or c["gpu_memory_utilization"] != 0.9
    ):
        raise ValueError("workload protocol mismatch")
    selected = c["selected_input_hashes"]
    if (
        len(selected) != c["batch_size"]
        or len(set(selected)) != c["batch_size"]
        or not all(_hash(v) for v in selected)
    ):
        raise ValueError("duplicate or missing selected input")
    partitions = raw["partitions"]
    if (
        len(partitions) != dp
        or any(not row for row in partitions)
        or sorted(i for row in partitions for i in row) != list(range(c["batch_size"]))
    ):
        raise ValueError("invalid partition")
    workers = raw["workers"]
    if len(workers) != dp or sorted(row["dp_rank"] for row in workers) != list(
        range(dp)
    ):
        raise ValueError("invalid workers")
    uuids = []
    inventory = []
    for worker in workers:
        if (
            worker["indices"] != partitions[worker["dp_rank"]]
            or worker["versions"] != c["versions"]
        ):
            raise ValueError("worker identity differs")
        _policy(worker["resolved_policy"], c, tp, dp, ep)
        _memory(worker["memory"], tp)
        devices = worker["devices"]
        if len(devices) != tp or {d["rank"] for d in devices} != {
            m["rank"] for m in worker["memory"]
        }:
            raise ValueError("device mapping missing")
        for device in devices:
            if (
                not isinstance(device.get("uuid"), str)
                or not device["uuid"]
                or type(device.get("total_memory")) is not int
                or device["total_memory"] <= 0
            ):
                raise ValueError("device identity missing")
            uuids.append(device["uuid"])
            inventory.append(
                {"uuid": device["uuid"], "total_memory": device["total_memory"]}
            )
    if len(set(uuids)) != 4:
        raise ValueError("duplicate physical GPU")
    observed_hardware = sha256(
        json.dumps(
            sorted(inventory, key=lambda d: d["uuid"]),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if observed_hardware != c["hardware_sha256"]:
        raise ValueError("hardware digest differs from observed devices")
    repetitions = raw["repetitions"]
    if (
        c["repetitions_requested"] != 3
        or len(repetitions) != 3
        or [r["iteration"] for r in repetitions] != [1, 2, 3]
    ):
        raise ValueError("three complete repetitions required")
    for repetition in repetitions:
        elapsed = repetition["elapsed_s"]
        if (
            repetition["status"] != "complete"
            or not _number(elapsed)
            or elapsed <= 0
            or repetition["timing_scope"] != "global_wallclock"
        ):
            raise ValueError("invalid coordinated duration")
        rows = repetition["per_rank"]
        if len(rows) != dp or sorted(r["dp_rank"] for r in rows) != list(range(dp)):
            raise ValueError("missing DP output")
        for row in rows:
            indices = partitions[row["dp_rank"]]
            expected = len(indices) * c["output_length"]
            if (
                row["status"] != "complete"
                or row["indices"] != indices
                or row["request_count"] != len(indices)
                or row["generated_tokens"] != expected
                or row["output_counts"] != [c["output_length"]] * len(indices)
            ):
                raise ValueError("fixed outputs incomplete")
            if len(row["output_hashes"]) != len(indices) or not all(
                _hash(v) for v in row["output_hashes"]
            ):
                raise ValueError("output hashes missing")
            if len(row["latencies_s"]) > len(indices) or not all(
                _number(v) for v in row["latencies_s"]
            ):
                raise ValueError("invalid latency evidence")
            _memory(row["memory"], tp)
        total = c["batch_size"] * c["output_length"]
        if (
            repetition["generated_tokens"] != total
            or not _number(repetition["tokens_s"])
            or not math.isclose(repetition["tokens_s"], total / elapsed)
        ):
            raise ValueError("invalid global reduction")


def _latency(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = sorted(v for row in rows for v in row.get("latencies_s", []) if _number(v))
    requests = sum(row.get("request_count", 0) for row in rows)
    return {
        "scope": "arrival-to-finished-offline",
        "count": len(values),
        "availability": "complete"
        if values and len(values) == requests
        else "partial"
        if values
        else "unavailable",
        "p50_s": values[math.ceil(len(values) * 0.5) - 1] if values else None,
        "p95_s": values[math.ceil(len(values) * 0.95) - 1] if values else None,
    }


def summarize_parallel(source: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "config": "unavailable",
        "status": "failed",
        "mode": "full-resident-native",
        "timing_scope": "global_wallclock",
        "median_tokens_s": None,
        "failure_category": "invalid-evidence",
        "memory": [],
        "repetitions": [],
        "workload": {},
        "output_variation_observed": None,
        "diagnostic_only": True,
        "formal_offload_gain": False,
        "deployment_gain_proven": False,
    }
    try:
        raw = _read(source)
        if raw.get("config") in CONFIGS:
            result["config"] = raw["config"]
        else:
            for part in source.parts:
                if part in CONFIGS:
                    result["config"] = part
        if raw.get("failure_category") in (
            "timeout",
            "worker-failed",
            "invalid-evidence",
        ):
            result["failure_category"] = raw["failure_category"]
        c = raw.get("contract", {})
        result["workload"] = {
            key: c[key]
            for key in (
                "batch_size",
                "context_length",
                "output_length",
                "max_num_seqs",
                "max_num_batched_tokens",
                "gpu_memory_utilization",
                "seed",
                "warmups",
            )
            if _number(c.get(key))
        }
        for worker in raw.get("workers", []):
            if type(worker.get("dp_rank")) is int:
                for memory in worker.get("memory", []):
                    result["memory"].append(
                        {
                            "dp_rank": worker["dp_rank"],
                            **{
                                key: memory[key]
                                for key in MEMORY
                                if _number(memory.get(key))
                            },
                        }
                    )
        _validate(raw)
        for repetition in raw["repetitions"]:
            result["repetitions"].append(
                {
                    key: repetition[key]
                    for key in (
                        "iteration",
                        "elapsed_s",
                        "generated_tokens",
                        "tokens_s",
                    )
                }
            )
            result["repetitions"][-1]["latency"] = _latency(repetition["per_rank"])
        hashes = [
            [
                row["output_hashes"]
                for row in sorted(r["per_rank"], key=lambda row: row["dp_rank"])
            ]
            for r in raw["repetitions"]
        ]
        result.update(
            status="complete",
            failure_category=None,
            median_tokens_s=statistics.median(
                r["tokens_s"] for r in raw["repetitions"]
            ),
            output_variation_observed=any(h != hashes[0] for h in hashes[1:]),
        )
        tp, dp, ep = CONFIGS[raw["config"]]
        result["parallelism"] = {
            "tp": tp,
            "dp": dp,
            "ep_enabled": ep,
            "physical_gpus": 4,
        }
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
        for part in source.parts:
            if part in CONFIGS:
                result["config"] = part
    return result


def compare_parallel(sources: Sequence[Path]) -> dict[str, Any]:
    rows = [summarize_parallel(source) for source in sources]
    eligible = bool(rows) and all(row["status"] == "complete" for row in rows)
    if eligible:
        contracts = [_read(source)["contract"] for source in sources]
        eligible = all(
            all(c[key] == contracts[0][key] for key in IDENTITY) for c in contracts
        )
        eligible = eligible and len({row["config"] for row in rows}) == len(rows)
    return {
        "schema_version": 1,
        "status": "eligible" if eligible else "ineligible",
        "timing_scope": "global_wallclock",
        "rows": rows,
        "diagnostic_only": True,
        "formal_offload_gain": False,
        "deployment_gain_proven": False,
    }


def _write(path: Path, content: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def export_parallel(sources: Sequence[Path], output: Path) -> dict[str, Any]:
    result = compare_parallel(sources)
    output.mkdir(parents=True, exist_ok=True)
    _write(
        output / "parallel-summary.json",
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
    )
    lines = [
        "# Native parallel comparison",
        "",
        f"Comparison: {result['status']}. Timing: global_wallclock.",
        "Latency: arrival-to-finished offline; missing values remain unavailable.",
        "",
        "| Config | Status | Median output tokens/s |",
        "| --- | --- | ---: |",
    ]
    for row in result["rows"]:
        value = row["median_tokens_s"]
        lines.append(
            f"| {row['config']} | {row['status']} | {value if value is not None else 'unavailable'} |"
        )
    lines.extend(
        [
            "",
            "Per-GPU KV allocations and repetition latency counts/P50/P95 are retained in parallel-summary.json.",
            "Native full-resident diagnostic; no offload or deployment gain claim.",
            "",
        ]
    )
    _write(output / "parallel-summary.md", "\n".join(lines))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    export_parallel(args.sources, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
