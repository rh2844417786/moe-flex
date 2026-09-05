"""Small offline partial-offload suite and allowlisted result export.

Run this file directly on the server host: it needs only the Python standard
library. GPU execution always uses the existing pinned offline container.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, cast

ARMS = ("resident", "partial-fixed-kv", "partial-auto-kv")
EXPECTED_ROOT = Path("/home/jovyan/wangtonghan/moe-flex")
MEMORY_FIELDS = (
    "rank",
    "total_gpu_bytes",
    "free_gpu_bytes",
    "torch_allocated_bytes",
    "torch_reserved_bytes",
    "available_kv_cache_bytes",
    "model_memory_bytes",
    "kv_cache_allocated_bytes",
    "kv_cache_declared_bytes",
    "num_gpu_blocks",
)
TIMING_FIELDS = (
    "sample_count",
    "load_cuda_s",
    "wait_cuda_s",
    "compute_cuda_s",
    "cpu_enqueue_s",
)
COUNTER_FIELDS = ("h2d_bytes", "copy_launches", "offload_forwards", "resident_forwards")
REPETITION_FIELDS = (
    "repetition",
    "elapsed_s",
    "generated_tokens",
    "request_count",
    "output_tokens_per_second",
    "latency_available_requests",
    "ttft_available_requests",
    "ttft_median_s",
    "request_latency_median_s",
)


@dataclass(frozen=True)
class Point:
    arm: str
    offload_count: int
    staging_slots: int

    def __post_init__(self) -> None:
        if (
            self.arm not in ARMS
            or self.staging_slots not in (1, 2)
            or self.offload_count < 0
        ):
            raise ValueError("invalid partial-offload point")
        if self.arm == "resident" and self.offload_count != 0:
            raise ValueError("resident point cannot offload layers")
        if self.offload_count and (
            self.offload_count <= self.staging_slots
            or self.offload_count % self.staging_slots
        ):
            raise ValueError("offload count must exceed and be divisible by slots")


def scan_points() -> list[Point]:
    return [Point("resident", 0, 1), Point("partial-auto-kv", 0, 1)] + [
        Point("partial-auto-kv", count, slots)
        for count in (2, 4, 6, 8)
        for slots in (1, 2)
        if count > slots and count % slots == 0
    ]


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return cast(dict[str, Any], value)


def _atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _identifier(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}", value
    ):
        raise ValueError("invalid public run ID")
    return value


def _numbers(raw: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in names:
        if name not in raw:
            continue
        value = raw[name]
        if value is None or (
            type(value) in (int, float) and math.isfinite(value) and value >= 0
        ):
            result[name] = value
        else:
            raise ValueError(f"invalid numeric field {name}")
    return result


def _hashes(raw: Mapping[str, Any], names: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        if name in raw:
            value = raw[name]
            if not isinstance(value, str) or not re.fullmatch(
                r"[0-9a-f]{40}|[0-9a-f]{64}", value
            ):
                raise ValueError(f"invalid hash field {name}")
            result[name] = value
    return result


def _bools(raw: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    return {
        name: raw[name]
        for name in names
        if name in raw and (type(raw[name]) is bool or raw[name] is None)
    }


def _public_repetition(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        **_numbers(row, REPETITION_FIELDS),
        **_hashes(row, ("output_sha256",)),
        **_bools(row, ("timing_available",)),
    }
    counters = row.get("diagnostics", {})
    result["diagnostics"] = {
        **_numbers(counters, COUNTER_FIELDS),
        "timing": _numbers(counters.get("timing", {}), TIMING_FIELDS),
    }
    result["diagnostics"]["per_rank"] = [
        {
            **_numbers(item, ("rank", *COUNTER_FIELDS)),
            "timing": _numbers(item.get("timing", {}), TIMING_FIELDS),
        }
        for item in counters.get("per_rank", [])
    ]
    return result


def public_run(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Typed allowlist, including nested values. Never copy arbitrary objects."""
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": _identifier(raw["run_id"]),
        "arm": raw["arm"] if raw.get("arm") in ARMS else "unknown",
        "status": raw["status"]
        if raw.get("status")
        in ("running", "complete", "failed", "timeout", "interrupted")
        else "failed",
        **_numbers(
            raw,
            (
                "offload_count",
                "staging_slots",
                "repetitions_completed",
                "requested_kv_cache_bytes",
                "exit_code",
                "launcher_elapsed_s",
            ),
        ),
        **_bools(
            raw,
            (
                "smoke_matches_resident",
                "performance_outputs_stable",
                "performance_outputs_match_resident",
                "mechanism_validated",
            ),
        ),
    }
    layers = raw.get("offload_layers", [])
    if not isinstance(layers, list) or any(
        type(layer) is not int or layer < 0 for layer in layers
    ):
        raise ValueError("invalid offload layers")
    result["offload_layers"] = list(layers)
    source = raw.get("contract", {})
    contract: dict[str, Any] = {
        **_numbers(
            source,
            (
                "batch_size",
                "context_length",
                "output_length",
                "tensor_parallel_size",
                "gpu_memory_utilization",
                "seed",
                "warmups",
                "timing_samples",
                "smoke_output_length",
                "source_request_count",
                "unique_selected_request_count",
                "repeated_request_count",
            ),
        ),
        **_hashes(
            source,
            (
                "commit",
                "model_identity_sha256",
                "model_config_sha256",
                "dataset_sha256",
                "dataset_manifest_sha256",
                "input_sha256",
                "engine_policy_sha256",
            ),
        ),
    }
    if source.get("dtype") == "bfloat16":
        contract["dtype"] = "bfloat16"
    if source.get("sampling_policy") in (
        "existing-prompts",
        "repeated-existing-prompts",
    ):
        contract["sampling_policy"] = source["sampling_policy"]
    versions = source.get("versions", {})
    contract["versions"] = {
        name: value
        for name in ("torch", "vllm", "cuda", "vllm_commit")
        if (value := versions.get(name)) is None
        or (
            isinstance(value, str)
            and re.fullmatch(r"[0-9][0-9A-Za-z.+_-]{0,63}", value)
        )
    }
    result["contract"] = contract
    for name in ("memory", "final_memory"):
        result[name] = [
            {
                **_numbers(row, MEMORY_FIELDS),
                **_bools(row, ("kv_cache_accounting_consistent",)),
            }
            for row in raw.get(name, [])
        ]
    budget = raw.get("hardware_budget", {})
    result["hardware_budget"] = {}
    for name in ("total_gpu_bytes_by_rank", "requested_gpu_bytes_by_rank"):
        if name in budget:
            values = budget[name]
            if not isinstance(values, list) or any(
                type(value) is not int or value <= 0 for value in values
            ):
                raise ValueError("invalid hardware budget")
            result["hardware_budget"][name] = list(values)
    smoke = raw.get("smoke", {})
    result["smoke"] = {
        **_numbers(smoke, REPETITION_FIELDS),
        **_hashes(smoke, ("input_sha256", "output_sha256")),
    }
    result["repetitions"] = [
        _public_repetition(row) for row in raw.get("repetitions", [])
    ]
    result["partial_stats"] = [
        {
            **_numbers(
                row,
                (
                    "schema_version",
                    "rank",
                    "total_layers",
                    "staging_slots",
                    "layer_bytes",
                    "host_source_bytes",
                    "gpu_staging_bytes",
                    "resident_routed_bytes",
                    "net_freed_bytes",
                    "weights_expected",
                    "weights_verified",
                    *COUNTER_FIELDS,
                ),
            ),
            "timing": _numbers(row.get("timing", {}), TIMING_FIELDS),
        }
        for row in raw.get("partial_stats", [])
    ]
    return result


def _actual_kv(row: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    workers = row["contract"]["tensor_parallel_size"]
    memory = row["memory"]
    if len(memory) != workers or {item.get("rank") for item in memory} != set(
        range(workers)
    ):
        raise ValueError("worker KV coverage is incomplete")
    if any(item.get("kv_cache_accounting_consistent") is not True for item in memory):
        raise ValueError(
            "actual KV allocation accounting is inconsistent or unavailable"
        )
    pairs = tuple(
        (item.get("kv_cache_allocated_bytes"), item.get("num_gpu_blocks"))
        for item in sorted(memory, key=lambda item: item["rank"])
    )
    if any(type(value) is not int or value <= 0 for pair in pairs for value in pair):
        raise ValueError("actual KV bytes or block capacity unavailable")
    return cast(tuple[tuple[int, int], ...], pairs)


def _speeds(row: Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    config = row["contract"]
    repetitions = row["repetitions"]
    if not repetitions or len(repetitions) != row["repetitions_completed"]:
        raise ValueError("repetition coverage is incomplete")
    for index, item in enumerate(repetitions):
        speed = item.get("output_tokens_per_second")
        elapsed = item.get("elapsed_s")
        if (
            item.get("repetition") != index
            or item.get("request_count") != config["batch_size"]
            or item.get("generated_tokens")
            != config["batch_size"] * config["output_length"]
        ):
            raise ValueError("fixed workload token count differs")
        if (
            type(speed) not in (int, float)
            or not math.isfinite(speed)
            or speed <= 0
            or type(elapsed) not in (int, float)
            or elapsed <= 0
        ):
            raise ValueError("throughput or elapsed time is unavailable")
        if not math.isclose(speed * elapsed, item["generated_tokens"], rel_tol=1e-6):
            raise ValueError("throughput does not match measured elapsed time")
        values.append(float(speed))
    return values


def analyze_triplet(
    resident: Mapping[str, Any], fixed: Mapping[str, Any], auto: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "incomplete",
        "stable_three_repetition_gain": False,
    }
    runs = (resident, fixed, auto)
    if any(row.get("status") != "complete" for row in runs):
        return result
    try:
        if tuple(row.get("arm") for row in runs) != ARMS:
            raise ValueError("R/B/C arm mapping differs")
        if any(row.get("contract") != resident.get("contract") for row in runs[1:]):
            raise ValueError(
                "model, workload, budget, software or engine policy contract differs"
            )
        required = (
            "commit",
            "model_identity_sha256",
            "input_sha256",
            "dataset_manifest_sha256",
            "engine_policy_sha256",
            "dtype",
            "gpu_memory_utilization",
            "seed",
        )
        if any(name not in resident["contract"] for name in required):
            raise ValueError("comparison contract is incomplete")
        if any(
            row.get("hardware_budget") != resident.get("hardware_budget")
            for row in runs[1:]
        ):
            raise ValueError("physical GPU memory budget differs")
        if (
            fixed.get("offload_count", 0) <= 0
            or fixed.get("offload_layers") != auto.get("offload_layers")
            or fixed.get("staging_slots") != auto.get("staging_slots")
        ):
            raise ValueError("B/C offload placement differs or is the zero control")
        if any(row.get("mechanism_validated") is not True for row in runs[1:]):
            raise ValueError("weight/transfer mechanism evidence is unavailable")
        for row in runs[1:]:
            if (
                row.get("smoke_matches_resident") is not True
                or any(
                    row.get("smoke", {}).get(name)
                    != resident.get("smoke", {}).get(name)
                    for name in (
                        "input_sha256",
                        "output_sha256",
                        "request_count",
                        "generated_tokens",
                    )
                )
                or row.get("smoke", {}).get("request_count") != 1
            ):
                raise ValueError("batch1 smoke output hashes do not match")
        rkv, bkv, ckv = (_actual_kv(row) for row in runs)
        if rkv != bkv:
            raise ValueError("B/R actual KV capacity differs")
        r, b, c = (_speeds(row) for row in runs)
        if len(r) != len(b) or len(b) != len(c):
            raise ValueError("repetition counts differ")
        kv_gain = sum(pair[0] for pair in ckv) - sum(pair[0] for pair in bkv)
        stable = len(c) >= 3 and min(c) > max(r) and kv_gain > 0
        result.update(
            {
                "status": "validated-throughput-gain"
                if stable
                else "no-stable-net-gain",
                "stable_three_repetition_gain": stable,
                "b_over_r": median(b) / median(r),
                "c_over_b": median(c) / median(b),
                "c_over_r": median(c) / median(r),
                "c_over_r_conservative_min": min(c) / max(r),
                "c_over_r_conservative_max": max(c) / min(r),
                "kv_gain_bytes_total": kv_gain,
                "repetitions": len(r),
                "performance_output_hashes_match": all(
                    [rep.get("output_sha256") for rep in row["repetitions"]]
                    == [rep.get("output_sha256") for rep in resident["repetitions"]]
                    for row in runs[1:]
                ),
                "correctness_scope": "batch1-smoke-only",
            }
        )
    except (KeyError, TypeError, ValueError) as error:
        result.update({"status": "invalid-comparison", "reason": str(error)})
    return result


def point_command(
    root: Path,
    point: Point,
    run_dir: Path,
    reference: Path | None,
    options: Mapping[str, Any],
) -> list[str]:
    command = [
        "bash",
        str(root / "scripts/server/run_partial_offload.sh"),
        "point",
        "--timeout-s",
        str(options["timeout_s"]),
        "--arm",
        point.arm,
        "--offload-count",
        str(point.offload_count),
        "--staging-slots",
        str(point.staging_slots),
        "--run-dir",
        str(run_dir),
    ]
    for name in (
        "batch_size",
        "context_length",
        "output_length",
        "gpu_memory_utilization",
        "warmups",
        "repetitions",
        "max_num_seqs",
        "max_num_batched_tokens",
        "timing_samples",
        "seed",
    ):
        command.extend(["--" + name.replace("_", "-"), str(options[name])])
    for name in ("dataset_path", "dataset_manifest"):
        if name in options:
            command.extend(["--" + name.replace("_", "-"), str(options[name])])
    if reference is not None:
        command.extend(["--resident-run", str(reference)])
    return command


def execute_point(
    root: Path,
    point: Point,
    run_dir: Path,
    reference: Path | None,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    if run_dir.exists():
        raise FileExistsError(run_dir)
    log_root = run_dir.parent / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    command = point_command(root, point, run_dir, reference, options)
    print(
        f"Starting {run_dir.name}: {point.arm}, offload={point.offload_count}, slots={point.staging_slots}",
        flush=True,
    )
    started = perf_counter()
    with (
        (log_root / f"{run_dir.name}.stdout.log").open("w") as stdout,
        (log_root / f"{run_dir.name}.stderr.log").open("w") as stderr,
    ):
        outcome = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    path = run_dir / "summary.json"
    summary = (
        _read(path)
        if path.is_file()
        else {
            "schema_version": 1,
            "run_id": run_dir.name,
            "arm": point.arm,
            "offload_count": point.offload_count,
            "staging_slots": point.staging_slots,
            "status": "failed",
            "repetitions_completed": 0,
            "repetitions": [],
        }
    )
    summary["exit_code"] = (
        outcome.returncode if outcome.returncode >= 0 else 128 - outcome.returncode
    )
    summary["launcher_elapsed_s"] = perf_counter() - started
    if outcome.returncode in (124, 137):
        summary["status"] = "timeout"
    elif outcome.returncode != 0 or summary.get("status") != "complete":
        summary["status"] = "failed"
    _atomic(path, summary)
    print(f"Finished {run_dir.name}: {summary['status']}", flush=True)
    return summary


def _scan_choice(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    residents = [
        row
        for row in runs
        if row.get("arm") == "resident" and row.get("status") == "complete"
    ]
    if len(residents) != 1:
        return None
    resident = residents[0]
    candidates: list[tuple[float, Mapping[str, Any]]] = []
    for row in runs:
        if (
            row.get("arm") != "partial-auto-kv"
            or row.get("status") != "complete"
            or row.get("offload_count", 0) <= 0
        ):
            continue
        if (
            row.get("contract") != resident.get("contract")
            or row.get("smoke_matches_resident") is not True
        ):
            continue
        try:
            _actual_kv(row)
            candidates.append((median(_speeds(row)), row))
        except (KeyError, TypeError, ValueError):
            continue
    if not candidates:
        return None
    speed, selected = max(candidates, key=lambda pair: pair[0])
    return {
        "run_id": selected["run_id"],
        "offload_count": selected["offload_count"],
        "staging_slots": selected["staging_slots"],
        "output_tokens_per_second": speed,
        "status": "diagnostic-candidate-only",
    }


def export_suite(source: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    # An export may be repeated from a suite that already contains its
    # generated public package; that package is not a benchmark run.
    paths = [path for path in source.glob("*/summary.json") if path.parent.name != "public"]
    runs = [public_run(_read(path)) for path in sorted(paths)]
    if not runs:
        raise ValueError("suite has no run summaries")
    by_id = {row["run_id"]: row for row in runs}
    manifest = _read(source / "suite.json") if (source / "suite.json").is_file() else {}
    comparisons: list[dict[str, Any]] = []
    groups: list[dict[str, str]] = []
    for raw_group in manifest.get("groups", []):
        group = {arm: _identifier(raw_group[arm]) for arm in ARMS}
        groups.append(group)
        comparison = analyze_triplet(*(by_id.get(group[arm], {}) for arm in ARMS))
        comparison["run_ids"] = group
        comparisons.append(comparison)
    exported: dict[str, Any] = {
        "schema_version": 1,
        "runs": runs,
        "comparisons": comparisons,
        "selected_candidate": _scan_choice(runs),
        "both_orders_confirmed": len(comparisons) >= 2
        and all(row["stable_three_repetition_gain"] for row in comparisons),
    }
    _atomic(output / "summary.json", exported)
    fields = (
        "run_id",
        "arm",
        "status",
        "offload_count",
        "staging_slots",
        "repetitions",
        "batch_size",
        "context_length",
        "output_length",
        "gpu_memory_utilization",
        "throughput_median",
        "throughput_min",
        "throughput_max",
        "kv_bytes_total",
        "smoke_matches_resident",
        "performance_outputs_match_resident",
    )
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in runs:
            speeds = [rep["output_tokens_per_second"] for rep in row["repetitions"]]
            csv_row = {key: row.get(key) for key in fields if key in row}
            csv_row.update(
                {
                    key: row["contract"].get(key)
                    for key in (
                        "batch_size",
                        "context_length",
                        "output_length",
                        "gpu_memory_utilization",
                    )
                }
            )
            csv_row.update(
                {
                    "repetitions": len(speeds),
                    "throughput_median": median(speeds) if speeds else None,
                    "throughput_min": min(speeds) if speeds else None,
                    "throughput_max": max(speeds) if speeds else None,
                    "kv_bytes_total": sum(
                        item.get("kv_cache_allocated_bytes") or 0
                        for item in row["memory"]
                    )
                    if row["memory"]
                    else None,
                }
            )
            writer.writerow(csv_row)
    lines = [
        "# 部分 BF16 专家卸载离线吞吐结果",
        "",
        "主指标为相同固定 token 工作量的端到端输出吞吐；所有容量以十进制 GB 展示。",
        "",
        "| Run | Arm | 状态 | 卸载层 / slots | 输出 token/s 中位数 [最小, 最大] | 实际 KV 总 GB |",
        "|---|---|---|---|---|---|",
    ]
    for row in runs:
        speeds = [rep["output_tokens_per_second"] for rep in row["repetitions"]]
        speed_text = (
            f"{median(speeds):.3f} [{min(speeds):.3f}, {max(speeds):.3f}]"
            if speeds
            else "不可用"
        )
        values = [item.get("kv_cache_allocated_bytes") for item in row["memory"]]
        kv = (
            f"{sum(values) / 1e9:.6f}"
            if values and all(type(value) is int for value in values)
            else "不可用"
        )
        lines.append(
            f"| {row['run_id']} | {row['arm']} | {row['status']} | {row.get('offload_count')} / {row.get('staging_slots')} | {speed_text} | {kv} |"
        )
    for index, comparison in enumerate(comparisons, 1):
        lines.extend(["", f"第 {index} 组：{comparison['status']}。"])
        if "c_over_r" in comparison:
            lines.append(
                f"B/R={comparison['b_over_r']:.4f}；C/B={comparison['c_over_b']:.4f}；C/R={comparison['c_over_r']:.4f}。三轮稳定净收益={comparison['stable_three_repetition_gain']}。"
            )
            lines.append(
                f"性能批量输出哈希一致={comparison['performance_output_hashes_match']}。"
            )
        elif "reason" in comparison:
            lines.append(f"原因：{comparison['reason']}。")
    lines.extend(
        [
            "",
            "正确性结论仅覆盖相同输入的 batch=1 greedy smoke 输出哈希。性能批量的输出哈希另行比较；不一致时不声称性能批量输出等价。",
            "缺失 RequestOutput.metrics 的时延记为不可用。CUDA 抽样计时只解释机制，不能替代端到端吞吐。短扫描用于选候选，不能证明三轮稳定收益。",
            "模型身份由 config、权重索引和模型路径的哈希识别；未重读并计算全部权重内容哈希。初始化和 warmup 不进入 measured counter delta。",
            "原始 stdout/stderr 留在服务器私有 runs 目录；本包仅含白名单数值、哈希、版本、相对 run ID 和结果文字。",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return exported


def execute_suite(args: argparse.Namespace) -> Path:
    root = cast(Path, args.project_root).resolve()
    if root != EXPECTED_ROOT:
        raise ValueError(f"server execution requires project root {EXPECTED_ROOT}")
    if not os.environ.get("GPU_IDS"):
        raise ValueError("GPU_IDS must explicitly select the four exclusive H100 GPUs")
    suite_id = _identifier(
        args.suite_id
        or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.mode
    )
    source = root / "runs" / "partial" / suite_id
    source.mkdir(parents=True, exist_ok=False)
    options = vars(args).copy()
    if args.mode == "scan":
        points = scan_points()
    else:
        if args.scan_dir:
            scan_runs = [_read(path) for path in args.scan_dir.glob("*/summary.json")]
            selected = _scan_choice(scan_runs)
            if selected is None:
                raise ValueError("scan has no valid nonzero-offload candidate")
            count, slots = selected["offload_count"], selected["staging_slots"]
        else:
            count, slots = args.offload_count, args.staging_slots
        if not count:
            raise ValueError("confirmation needs --offload-count or --scan-dir")
        points = [
            Point("resident", 0, 1),
            Point("partial-fixed-kv", count, slots),
            Point("partial-auto-kv", count, slots),
        ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "mode": args.mode,
        "run_ids": [],
        "groups": [],
        "status": "running",
    }
    rows: list[dict[str, Any]] = []
    resident: Path | None = None
    group: dict[str, str] = {}
    for index, point in enumerate(points, 1):
        run_id = (
            f"{index:02d}-{point.arm}-o{point.offload_count}-s{point.staging_slots}"
        )
        manifest["run_ids"].append(run_id)
        _atomic(source / "suite.json", manifest)
        row = execute_point(root, point, source / run_id, resident, options)
        rows.append(row)
        group[point.arm] = run_id
        if point.arm == "resident":
            if row["status"] != "complete":
                break
            resident = source / run_id
    if args.mode == "confirm" and all(arm in group for arm in ARMS):
        manifest["groups"].append(group)
        first_comparison = analyze_triplet(*rows)
        if args.reverse_on_gain and first_comparison["stable_three_repetition_gain"]:
            reverse_group: dict[str, str] = {}
            for index, point in enumerate(reversed(points), 4):
                run_id = f"{index:02d}-reverse-{point.arm}-o{point.offload_count}-s{point.staging_slots}"
                manifest["run_ids"].append(run_id)
                _atomic(source / "suite.json", manifest)
                execute_point(
                    root,
                    point,
                    source / run_id,
                    resident if point.arm != "resident" else None,
                    options,
                )
                reverse_group[point.arm] = run_id
            manifest["groups"].append(reverse_group)
    manifest["status"] = "finished"
    _atomic(source / "suite.json", manifest)
    export_suite(source, source / "public")
    print(f"Numeric results: {source / 'public'}", flush=True)
    return source


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("scan", "confirm"):
        command = commands.add_parser(mode)
        command.add_argument("--project-root", type=Path, default=EXPECTED_ROOT)
        command.add_argument("--suite-id")
        command.add_argument("--gpu-memory-utilization", type=float, default=0.60)
        defaults = {
            "timeout-s": 1800,
            "batch-size": 256 if mode == "scan" else 512,
            "context-length": 4096,
            "output-length": 32 if mode == "scan" else 256,
            "warmups": 0 if mode == "scan" else 1,
            "repetitions": 1 if mode == "scan" else 3,
            "max-num-seqs": 256,
            "max-num-batched-tokens": 8192,
            "timing-samples": 128,
            "seed": 20260905,
        }
        for name, default in defaults.items():
            command.add_argument("--" + name, type=int, default=default)
        command.add_argument(
            "--dataset-path",
            type=Path,
            default=Path("benchmarks/data/sharegpt/qwen3next_1024_requests.jsonl.zst"),
        )
        command.add_argument(
            "--dataset-manifest",
            type=Path,
            default=Path("benchmarks/data/sharegpt/dataset_manifest.json"),
        )
        if mode == "confirm":
            selection = command.add_mutually_exclusive_group(required=True)
            selection.add_argument("--offload-count", type=int)
            command.add_argument("--staging-slots", type=int, choices=(1, 2), default=1)
            selection.add_argument("--scan-dir", type=Path)
            command.add_argument(
                "--reverse-on-gain",
                action="store_true",
                help="After a stable forward gain, independently rerun C/B/R in reverse order",
            )
    export = commands.add_parser("export")
    export.add_argument("--project-root", type=Path, default=Path.cwd())
    export.add_argument("--suite-dir", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "export":
        if not args.output_dir.resolve().is_relative_to(args.project_root.resolve()):
            parser.error("output-dir must remain inside project-root")
        export_suite(args.suite_dir.resolve(), args.output_dir.resolve())
    else:
        if args.timeout_s <= 0 or args.repetitions <= 0 or args.warmups < 0:
            parser.error("timeouts/repetitions must be positive; warmups nonnegative")
        execute_suite(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
