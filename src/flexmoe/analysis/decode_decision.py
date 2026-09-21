"""One-command, fail-closed follow-up to the TP4 decode offload experiment.

The host controller uses only the standard library. All GPU work is delegated to
the existing pinned, offline decode point wrapper; this module never simulates a
transfer or calls a replay result an end-to-end prefetch speedup.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import runpy
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, cast
from uuid import uuid4

MODEL = "/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct"
DATA = "benchmarks/data/decode-mechanism/qwen3next_unique_1536x_1k4k.jsonl.zst"
MANIFEST = "benchmarks/data/decode-mechanism/dataset_manifest.json"
SAFETY_RESERVE = 2_000_000_000


def validate_run_id(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,55}", value) or ".." in value:
        raise ValueError("run ID must be a short path-free name")
    return value


def validate_branch(value: str) -> str:
    if value != "repro/fluxmoe":
        raise ValueError("only the repro/fluxmoe branch may run this experiment")
    return value


def _actual_kv(summary: dict[str, Any]) -> tuple[int, int]:
    if summary.get("status") != "complete":
        raise ValueError("incomplete point")
    actual = summary.get("actual_kv", {})
    allocated, block = (
        actual.get("allocated_bytes_per_rank"),
        actual.get("bytes_per_block"),
    )
    if (
        type(allocated) is not int
        or type(block) is not int
        or allocated <= 0
        or block <= 0
        or allocated % block
        or actual.get("num_gpu_blocks") != allocated // block
    ):
        raise ValueError("no rank-agreed actual KV or block size")
    return allocated, block


def zero_miss_gate(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("profile") or summary.get("mode") != "offload":
        return {
            "status": "not-measured",
            "reason": "requires an unprofiled offload run",
        }
    reps = summary.get("repetitions", [])
    if (
        summary.get("status") != "complete"
        or summary.get("repetitions_completed") != 3
        or len(reps) != 3
    ):
        return {"status": "incomplete-evidence"}
    observed = []
    for i, rep in enumerate(reps):
        if (
            rep.get("repetition") != i
            or rep.get("timing_eligible") is not True
            or rep.get("generation_status") != "complete"
            or rep.get("capture_status") != "complete"
            or rep.get("measurement_status") == "rejected"
        ):
            return {"status": "incomplete-evidence"}
        rows = rep.get("diagnostics", {}).get("per_rank")
        if (
            not isinstance(rows, list)
            or {r.get("rank") for r in rows if isinstance(r, dict)} != set(range(4))
            or len(rows) != 4
        ):
            return {"status": "incomplete-evidence"}
        for row in rows:
            values = (
                row.get("policy", {}).get("cache_misses"),
                row.get("h2d_bytes"),
                row.get("copy_launches"),
            )
            if any(type(v) is not int or v < 0 for v in values):
                return {"status": "incomplete-evidence"}
            observed.append(
                {
                    "repetition": i,
                    "rank": row["rank"],
                    "misses": values[0],
                    "payload_bytes": values[1],
                    "copy_launches": values[2],
                }
            )
    return {
        "status": "proven-zero-miss"
        if all(
            all(row[key] == 0 for key in ("misses", "payload_bytes", "copy_launches"))
            for row in observed
        )
        else "not-zero-miss",
        "per_rank_repetition": observed,
    }


def capacity_gate(
    native: dict[str, Any], eager: dict[str, Any], offload: dict[str, Any]
) -> dict[str, Any]:
    try:
        native_kv, native_block = _actual_kv(native)
        eager_kv, eager_block = _actual_kv(eager)
        k1, b1 = _actual_kv(offload)
        if (
            native.get("mode") != "native"
            or eager.get("mode") != "matched-resident"
            or offload.get("mode") != "offload"
        ):
            raise ValueError("capacity arms differ")
        fields = (
            "commit",
            "input_sha256",
            "context_length",
            "batch_size",
            "output_length",
            "max_num_seqs",
            "max_num_batched_tokens",
            "seed",
            "gpu_memory_utilization",
            "model_identity_sha256",
            "hardware_sha256",
            "physical_safety_reserve_bytes",
        )
        contracts = [run["contract"] for run in (native, eager, offload)]
        if (
            any(
                contracts[0].get(field) is None
                or any(c.get(field) != contracts[0][field] for c in contracts[1:])
                for field in fields
            )
            or any(
                run.get("hardware_budget") != native.get("hardware_budget")
                for run in (eager, offload)
            )
            or native.get("source_hashes") is None
            or any(
                run.get("source_hashes") != native["source_hashes"]
                for run in (eager, offload)
            )
            or contracts[0].get("expert_cache", {}).get("profile_sha256") is None
            or any(
                c.get("expert_cache", {}).get("profile_sha256")
                != contracts[0]["expert_cache"]["profile_sha256"]
                for c in contracts[1:]
            )
        ):
            raise ValueError("capacity controls differ")
    except (ValueError, TypeError, KeyError):
        return {"status": "incomplete-evidence"}
    resident_max = max(native_kv, eager_kv)
    extra = k1 - resident_max
    status = (
        "extra-capacity-measured"
        if extra >= max(native_block, eager_block, b1)
        else "no-extra-capacity"
        if extra <= 0
        else "incomplete-evidence"
    )
    return {
        "status": status,
        "native_bytes_per_rank": native_kv,
        "eager_bytes_per_rank": eager_kv,
        "resident_bytes_per_rank": resident_max,
        "offload_bytes_per_rank": k1,
        "extra_bytes_per_rank": extra,
        "minimum_resolvable_increment_bytes": max(native_block, eager_block, b1),
        "interpretation": "compares actual auto KV against BOTH native and eager resident controls after block rounding; not a successful long-load throughput test",
    }


def select_budgets(
    native: dict[str, Any], eager: dict[str, Any], offload: dict[str, Any]
) -> dict[str, int | None]:
    gate = capacity_gate(native, eager, offload)
    if gate["status"] == "incomplete-evidence":
        raise ValueError("unmatched or incomplete auto-KV capacity points")
    native_kv, _ = _actual_kv(native)
    eager_kv, _ = _actual_kv(eager)
    offload_kv, _ = _actual_kv(offload)
    return {
        "native_kv": native_kv,
        "matched_eager_kv": min(eager_kv, offload_kv),
        "offload_extra_kv": (
            offload_kv if gate["status"] == "extra-capacity-measured" else None
        ),
    }


def _same_eager_controls(a: dict[str, Any], b: dict[str, Any]) -> bool:
    fields = (
        "commit",
        "batch_size",
        "context_length",
        "output_length",
        "max_num_seqs",
        "max_num_batched_tokens",
        "input_sha256",
        "calibration_input_hashes_sha256",
        "seed",
        "model_identity_sha256",
        "hardware_sha256",
        "gpu_memory_utilization",
        "physical_safety_reserve_bytes",
    )
    ca, cb = a["contract"], b["contract"]
    return (
        a.get("profile") is False
        and b.get("profile") is False
        and all(ca.get(k) is not None and ca.get(k) == cb.get(k) for k in fields)
        and a.get("source_hashes") is not None
        and a["source_hashes"] == b.get("source_hashes")
        and a.get("engine_policy") is not None
        and a["engine_policy"] == b.get("engine_policy")
        and ca.get("expert_cache", {}).get("profile_sha256") is not None
        and ca["expert_cache"]["profile_sha256"]
        == cb.get("expert_cache", {}).get("profile_sha256")
        and a.get("smoke", {}).get("output_sha256") is not None
        and a["smoke"]["output_sha256"] == b.get("smoke", {}).get("output_sha256")
        and a.get("hardware_budget") == b.get("hardware_budget")
        and all(
            run.get("repetitions_completed") == 3
            and len(run.get("repetitions", [])) == 3
            for run in (a, b)
        )
    )


def compare_pair(resident: dict[str, Any], offload: dict[str, Any]) -> dict[str, Any]:
    try:
        a, _ = _actual_kv(resident)
        b, _ = _actual_kv(offload)
        if (
            resident.get("mode") != "matched-resident"
            or offload.get("mode") != "offload"
            or a != b
            or not _same_eager_controls(resident, offload)
        ):
            raise ValueError("pair differs")
        at, bt = (
            median(float(rep["output_tokens_per_second"]) for rep in run["repetitions"])
            for run in (resident, offload)
        )
        if min(at, bt) <= 0:
            raise ValueError("unmeasured throughput")
    except (KeyError, TypeError, ValueError):
        return {"status": "invalid-pair"}
    return {
        "status": "measured-pair",
        "resident_median_tokens_s": at,
        "offload_median_tokens_s": bt,
        "offload_over_resident": bt / at,
        "scope": "same eager mode, workload and actual KV; wall clock includes prefill",
    }


def compare_offload_kv(small: dict[str, Any], large: dict[str, Any]) -> dict[str, Any]:
    try:
        k0, _ = _actual_kv(small)
        k1, _ = _actual_kv(large)
        if (
            small.get("mode") != "offload"
            or large.get("mode") != "offload"
            or k1 <= k0
            or not _same_eager_controls(small, large)
            or small["contract"].get("expert_cache")
            != large["contract"].get("expert_cache")
        ):
            raise ValueError("not a same-path KV intervention")
        a, b = (
            median(float(row["output_tokens_per_second"]) for row in run["repetitions"])
            for run in (small, large)
        )
        if min(a, b) <= 0:
            raise ValueError("missing throughput")
    except (KeyError, TypeError, ValueError):
        return {"status": "invalid-pair"}
    return {
        "status": "measured-kv-intervention",
        "small_kv_bytes": k0,
        "large_kv_bytes": k1,
        "large_over_small_throughput": b / a,
        "scope": "same offload policy/workload/hardware; only actual KV differs; whole generation wallclock",
    }


def point_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    actual, _ = _actual_kv(summary)
    reps = summary.get("repetitions", [])
    values: dict[str, Any] = {
        "actual_kv": actual,
        "throughput": median(rep["output_tokens_per_second"] for rep in reps),
        "elapsed_median_s": median(rep["elapsed_s"] for rep in reps),
        "ttft_median_s": median(
            [
                rep["ttft_median_s"]
                for rep in reps
                if type(rep.get("ttft_median_s")) in (int, float)
            ]
        )
        if any(type(rep.get("ttft_median_s")) in (int, float) for rep in reps)
        else None,
        "request_latency_median_s": median(
            [
                rep["request_latency_median_s"]
                for rep in reps
                if type(rep.get("request_latency_median_s")) in (int, float)
            ]
        )
        if any(
            type(rep.get("request_latency_median_s")) in (int, float) for rep in reps
        )
        else None,
        "decode_batch_mean": None,
        "decode_batch_steps": None,
        "kv_usage_peak": None,
        "preemptions_total": None,
        "waiting_requests_peak": None,
        "misses_per_rank": None,
        "payload_bytes_per_rank": None,
        "phase_status": "unavailable",
        "phase_per_repetition": [],
        "prefill_wall_time_median_s": None,
        "decode_wall_time_median_s": None,
        "request_metrics_per_repetition": [],
        "kv_admission_blocked": {
            "status": "unavailable",
            "reason": "waiting-request samples do not expose a direct KV allocation block reason",
        },
        "swap_outs": {"status": "policy-disabled", "measured_count": None},
        "recomputed_tokens": {
            "status": "unavailable",
            "reason": "pinned scheduler statistics do not expose recomputed token counts",
        },
        "oracle_per_repetition": [],
    }
    phases = [rep.get("phase") for rep in reps]
    phase_fields = (
        "max_rank_prefill_wall_time_s",
        "max_rank_decode_wall_time_s",
        "max_rank_decode_step_ms_p50",
        "max_rank_decode_step_ms_p95",
    )
    if len(phases) == len(reps) and all(
        isinstance(row, dict)
        and row.get("status") == "measured"
        and all(type(row.get(field)) in (int, float) for field in phase_fields)
        for row in phases
    ):
        measured_phases = cast(list[dict[str, Any]], phases)
        values["phase_status"] = "measured"
        values["phase_per_repetition"] = measured_phases
        values["prefill_wall_time_median_s"] = median(
            row["max_rank_prefill_wall_time_s"] for row in measured_phases
        )
        values["decode_wall_time_median_s"] = median(
            row["max_rank_decode_wall_time_s"] for row in measured_phases
        )
    request_metrics = [rep.get("request_metrics") for rep in reps]
    if len(request_metrics) == len(reps) and all(
        isinstance(row, dict) for row in request_metrics
    ):
        values["request_metrics_per_repetition"] = request_metrics
    oracle_metrics = [rep.get("oracle") for rep in reps]
    if len(oracle_metrics) == len(reps) and all(
        isinstance(row, dict) and row.get("status") == "measured"
        for row in oracle_metrics
    ):
        values["oracle_per_repetition"] = oracle_metrics
    distributions = []
    for rep in reps:
        obs = rep.get("worker_observations", [])
        if len(obs) != 4 or {row.get("rank") for row in obs} != set(range(4)):
            break
        batch = [row.get("decode_batch_step_counts", {}) for row in obs]
        if not all(row == batch[0] for row in batch):
            break
        if not all(
            str(key).isdigit() and type(count) is int and count >= 0
            for key, count in batch[0].items()
        ):
            break
        distributions.append(batch[0])
    if len(distributions) == len(reps):
        counts: dict[int, int] = {}
        for dist in distributions:
            for size, count in dist.items():
                counts[int(size)] = counts.get(int(size), 0) + count
        steps = sum(counts.values())
        values["decode_batch_steps"] = steps
        values["decode_batch_mean"] = (
            sum(size * count for size, count in counts.items()) / steps
            if steps
            else None
        )
    scheduler = [rep.get("scheduler", {}) for rep in reps]
    for field, item, operation in (
        ("kv_usage_peak", "kv_cache_usage", max),
        ("waiting_requests_peak", "waiting_requests", max),
        ("preemptions_total", "preemptions", sum),
    ):
        subkey = "total" if item == "preemptions" else "peak"
        samples = [row.get(item, {}) for row in scheduler]
        if len(samples) == len(reps) and all(
            sample.get("status") == "measured"
            and type(sample.get(subkey)) in (int, float)
            for sample in samples
        ):
            values[field] = operation(sample[subkey] for sample in samples)
    per_rep = [rep.get("diagnostics", {}).get("per_rank") for rep in reps]
    if len(per_rep) == len(reps) and all(
        isinstance(rows, list)
        and len(rows) == 4
        and {row.get("rank") for row in rows} == set(range(4))
        for rows in per_rep
    ):
        for source, dest in (("h2d_bytes", "payload_bytes_per_rank"),):
            if all(type(row.get(source)) is int for rows in per_rep for row in rows):
                values[dest] = [
                    sum(rows[rank][source] for rows in per_rep) for rank in range(4)
                ]
        if all(
            type(row.get("policy", {}).get("cache_misses")) is int
            for rows in per_rep
            for row in rows
        ):
            values["misses_per_rank"] = [
                sum(rows[rank]["policy"]["cache_misses"] for rows in per_rep)
                for rank in range(4)
            ]
    return values


def fixed_actual_batch_gate(summary: dict[str, Any], *, target: int) -> dict[str, Any]:
    if target not in (16, 32):
        raise ValueError("fixed actual batch target must be 16 or 32")
    repetitions = summary.get("repetitions")
    if (
        summary.get("status") != "complete"
        or summary.get("repetitions_completed") != 3
        or not isinstance(repetitions, list)
        or len(repetitions) != 3
    ):
        return {"status": "rejected", "reason": "incomplete-repetitions"}
    per_repetition = []
    for repetition, row in enumerate(repetitions):
        observations = row.get("worker_observations")
        if (
            not isinstance(observations, list)
            or len(observations) != 4
            or {item.get("rank") for item in observations} != set(range(4))
        ):
            return {"status": "rejected", "reason": "incomplete-rank-coverage"}
        distributions = [item.get("decode_batch_step_counts") for item in observations]
        if (
            any(value != distributions[0] for value in distributions[1:])
            or not isinstance(distributions[0], dict)
            or set(distributions[0]) != {str(target)}
            or type(distributions[0][str(target)]) is not int
            or distributions[0][str(target)] <= 0
        ):
            return {
                "status": "rejected",
                "reason": "decode-batch-distribution-differs",
            }
        per_repetition.append(
            {"repetition": repetition, "decode_steps": distributions[0][str(target)]}
        )
    return {
        "status": "measured",
        "target_actual_batch": target,
        "per_repetition": per_repetition,
    }


def aggregate_replay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scoped = {(row.get("repetition"), row.get("rank")) for row in rows}
    expected = {(rep, rank) for rep in range(3) for rank in range(4)}
    valid = (
        len(rows) == 12
        and scoped == expected
        and all(row.get("status") == "baseline-reproduced" for row in rows)
        and all(
            type(row.get(key)) is int and row[key] >= 0
            for row in rows
            for key in (
                "current_policy_misses",
                "lru_misses",
                "future_aware_misses",
                "future_aware_saved_bytes",
            )
        )
    )
    if not valid:
        return {"status": "incomplete-evidence", "per_rank_repetition": rows}
    return {
        "status": "baseline-reproduced-all-ranks",
        "current_policy_misses": sum(row["current_policy_misses"] for row in rows),
        "lru_misses": sum(row["lru_misses"] for row in rows),
        "future_aware_misses": sum(row["future_aware_misses"] for row in rows),
        "future_aware_saved_bytes": sum(
            row["future_aware_saved_bytes"] for row in rows
        ),
        "scope": "identical recorded demand/state, resident set and persistent cache capacity; eviction-only; no prefetch or measured throughput",
        "per_rank_repetition": rows,
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    return ordered[low] + (ordered[math.ceil(position)] - ordered[low]) * (
        position - low
    )


def profile_timing_file(path: Path) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            raw = json.load(stream)
        if (
            raw.get("artifact_kind") != "decode-profile"
            or raw.get("profile") is not True
        ):
            raise ValueError("not an instrumented decode profile")
        obs = raw["observation"]
        spans = obs["model_step_spans"]
        if (
            obs.get("coverage_status") != "complete"
            or not spans
            or any(
                row.get("status") != "measured"
                or type(row.get("cuda_s")) not in (int, float)
                or not math.isfinite(row["cuda_s"])
                or row["cuda_s"] < 0
                for row in spans
            )
        ):
            raise ValueError("incomplete CUDA step spans")
        result: dict[str, Any] = {
            "status": "instrumented",
            "model_step_cuda_p50_ms": _percentile(
                [row["cuda_s"] * 1000 for row in spans], 0.5
            ),
            "model_step_cuda_p95_ms": _percentile(
                [row["cuda_s"] * 1000 for row in spans], 0.95
            ),
            "step_count": len(spans),
            "host_gather_cpu_sum_s": None,
            "payload_bytes": None,
            "scope": "per-rank instrumented prepared-input to execute-model-return CUDA span; not unsampled decode wall",
        }
        pool = obs.get("pool_profile")
        if isinstance(pool, dict):
            rows = pool.get("rows", [])
            if (
                rows
                and pool.get("dropped_rows") == 0
                and all(
                    row.get("cpu_timing", {}).get("status") == "measured"
                    and type(row["cpu_timing"].get("host_gather_s")) in (int, float)
                    and type(row.get("loaded_bytes")) is int
                    for row in rows
                )
            ):
                result["host_gather_cpu_sum_s"] = sum(
                    row["cpu_timing"]["host_gather_s"] for row in rows
                )
                result["payload_bytes"] = sum(row["loaded_bytes"] for row in rows)
        return result
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "incomplete-evidence", "error_type": type(exc).__name__}


def profile_timing_point(root: Path, name: str) -> dict[str, Any]:
    rows = []
    for repetition in range(3):
        for rank in range(4):
            path = (
                root
                / "runs/decode-mechanism"
                / name
                / f"decode-rep-{repetition:03d}-rank-{rank}.json.gz"
            )
            rows.append(
                {"repetition": repetition, "rank": rank, **profile_timing_file(path)}
            )
    return {
        "status": "instrumented-all-ranks"
        if all(row["status"] == "instrumented" for row in rows)
        else "incomplete-evidence",
        "per_rank_repetition": rows,
    }


def point_arguments(
    mode: str,
    name: str,
    context: int,
    kv_bytes: int | None,
    batch: int,
    common: dict[str, str],
    *,
    profile: bool = False,
    resident_ratio: str = "0.8",
    output_length: int = 256,
    capture_steps: int = 64,
    target_batch: int | None = None,
    oracle_trace_path: str | None = None,
    prefetch_horizon: int | None = None,
    force_omission: str | None = None,
) -> dict[str, Any]:
    if mode not in ("calibrate", "native", "matched-resident", "offload"):
        raise ValueError("unsupported point mode")
    if (
        context not in (1024, 4096)
        or not 1 <= batch <= 64
        or kv_bytes is not None
        and kv_bytes <= 0
    ):
        raise ValueError("point exceeds bounded workload")
    argv = [
        "bash",
        "scripts/server/run_decode_mechanism.sh",
        mode,
        "--run-id",
        name,
        "--model-path",
        MODEL,
        "--dataset-path",
        DATA,
        "--dataset-manifest",
        MANIFEST,
        "--context-length",
        str(context),
        "--batch-size",
        str(batch),
        "--output-length",
        str(output_length),
        "--max-num-seqs",
        "32",
        "--max-num-batched-tokens",
        "8192",
        "--gpu-memory-utilization",
        "0.90",
        "--warmups",
        "1",
        "--repetitions",
        "3",
        "--calibration-count",
        "32",
        "--seed",
        "20260912",
    ]
    if mode != "calibrate":
        argv.extend(
            (
                "--profile-path",
                common["profile_path"],
                "--safety-reserve-bytes",
                str(SAFETY_RESERVE),
            )
        )
    if kv_bytes is not None:
        argv.extend(("--kv-bytes", str(kv_bytes)))
    if mode == "offload":
        argv.extend(("--resident-ratio", resident_ratio, "--cache-slots", "512"))
    if target_batch is not None:
        if target_batch not in (1, 16, 32):
            raise ValueError("target_batch must be 1, 16, or 32")
        argv.extend(("--target-batch", str(target_batch)))
    if (oracle_trace_path is None) != (prefetch_horizon is None):
        raise ValueError("oracle trace and horizon must be provided together")
    if oracle_trace_path is not None:
        if mode != "offload" or prefetch_horizon not in (0, 1, 2):
            raise ValueError("oracle points require offload horizon 0, 1, or 2")
        if target_batch not in (16, 32):
            raise ValueError("oracle points require target batch 16 or 32")
        argv.extend(
            (
                "--oracle-trace",
                oracle_trace_path,
                "--prefetch-horizon",
                str(prefetch_horizon),
            )
        )
    if force_omission is not None:
        if oracle_trace_path is None or prefetch_horizon not in (1, 2):
            raise ValueError("forced omission requires oracle horizon 1 or 2")
        argv.extend(("--force-omission", force_omission))
    if profile:
        if mode not in ("matched-resident", "offload"):
            raise ValueError("native profiling is not supported")
        argv.extend(
            (
                "--profile",
                "--capture-steps",
                str(capture_steps),
                "--min-capture-steps",
                str(min(capture_steps, output_length // 2)),
            )
        )
    return {"name": name, "mode": mode, "argv": argv}


def build_decision_plan(
    *,
    run_id: str,
    capacities: dict[str, int],
    trace_paths: dict[int, str],
    forced_omission: str = "0:1:0",
) -> list[dict[str, Any]]:
    validate_run_id(run_id)
    required = (
        "native_kv",
        "eager_kv",
        "offload_kv",
        "resident_ceiling",
        "k_pair",
        "legacy_kv",
        "kv_block_bytes",
    )
    if any(
        type(capacities.get(key)) is not int or capacities[key] <= 0 for key in required
    ):
        raise ValueError("complete positive capacity selection is required")
    if set(trace_paths) != {16, 32} or any(not value for value in trace_paths.values()):
        raise ValueError("batch 16 and 32 trace paths are required")
    points: list[dict[str, Any]] = []

    def add(
        label: str,
        mode: str,
        context: int,
        kv: int | None,
        batch: int,
        **options: Any,
    ) -> None:
        profile_context = 1024 if context == 1024 else 4096
        point = point_arguments(
            mode,
            f"{run_id}-{label}",
            context,
            kv,
            batch,
            {
                "profile_path": f"runs/decode-mechanism/{run_id}-cal-{profile_context}/profile.json"
            },
            **options,
        )
        points.append({"label": label, "status": "planned", **point})

    add("cal-1024", "calibrate", 1024, None, 32, output_length=8)
    add("cal-4096", "calibrate", 4096, None, 32, output_length=8)
    add("native-auto", "native", 4096, None, 1)
    add("eager-auto", "matched-resident", 4096, None, 1)
    add("offload-auto", "offload", 4096, None, 1)
    add("zero-resident", "matched-resident", 1024, capacities["k_pair"], 1)
    add(
        "zero-offload",
        "offload",
        1024,
        capacities["k_pair"],
        1,
        resident_ratio="0.9",
    )
    add(
        "zero-offload-profile",
        "offload",
        1024,
        capacities["k_pair"],
        1,
        resident_ratio="0.9",
        profile=True,
        target_batch=1,
    )
    for label, mode in (
        ("legacy-native", "native"),
        ("legacy-eager-resident", "matched-resident"),
        ("legacy-offload", "offload"),
    ):
        add(
            label,
            mode,
            4096,
            capacities["legacy_kv"],
            32,
            target_batch=32,
        )
    add(
        "mechanism-native-32",
        "native",
        4096,
        capacities["native_kv"],
        32,
        target_batch=32,
    )
    add(
        "mechanism-eager-32",
        "matched-resident",
        4096,
        capacities["k_pair"],
        32,
        target_batch=32,
    )
    add(
        "mechanism-offload-32",
        "offload",
        4096,
        capacities["k_pair"],
        32,
        target_batch=32,
    )
    add("service-native-k0", "native", 4096, capacities["native_kv"], 64)
    if capacities["offload_kv"] >= capacities["native_kv"]:
        add("service-offload-k0", "offload", 4096, capacities["native_kv"], 64)
    else:
        points.append(
            {
                "label": "service-offload-k0",
                "name": f"{run_id}-service-offload-k0",
                "mode": "offload",
                "status": "skipped",
                "reason": "offload-cannot-run-native-k0",
                "argv": [],
            }
        )
    if (
        capacities["offload_kv"] - capacities["resident_ceiling"]
        >= capacities["kv_block_bytes"]
    ):
        add("service-offload-k1", "offload", 4096, capacities["offload_kv"], 64)
    else:
        points.append(
            {
                "label": "service-offload-k1",
                "name": f"{run_id}-service-offload-k1",
                "mode": "offload",
                "status": "skipped",
                "reason": "offload-kv-does-not-exceed-resident-ceiling",
                "argv": [],
            }
        )
    add(
        "service-eager-kpair",
        "matched-resident",
        4096,
        capacities["k_pair"],
        64,
    )
    add("service-offload-kpair", "offload", 4096, capacities["k_pair"], 64)
    for batch in (16, 32):
        add(
            f"trace-{batch}",
            "offload",
            4096,
            capacities["k_pair"],
            batch,
            profile=True,
            target_batch=batch,
        )
    for batch in (16, 32):
        for horizon in (0, 1, 2):
            add(
                f"oracle-{batch}-h{horizon}",
                "offload",
                4096,
                capacities["k_pair"],
                batch,
                target_batch=batch,
                oracle_trace_path=trace_paths[batch],
                prefetch_horizon=horizon,
            )
        for horizon in (0, 1, 2):
            add(
                f"oracle-{batch}-h{horizon}-profile",
                "offload",
                4096,
                capacities["k_pair"],
                batch,
                profile=True,
                target_batch=batch,
                oracle_trace_path=trace_paths[batch],
                prefetch_horizon=horizon,
            )
    add(
        "oracle-forced-omission",
        "offload",
        4096,
        capacities["k_pair"],
        32,
        profile=True,
        target_batch=32,
        oracle_trace_path=trace_paths[32],
        prefetch_horizon=1,
        force_omission=forced_omission,
    )
    return points


def derive_forced_omission(path: Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        artifact = json.load(stream)
    if artifact.get("artifact_kind") != "decode-logical-cache-trace":
        raise ValueError("forced omission requires a logical cache trace")
    rows = artifact.get("rows")
    if not isinstance(rows, list):
        raise TypeError("logical cache trace rows are unavailable")
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("logical cache trace row is not an object")
        step, layer, ids = (
            row.get("step_id"),
            row.get("layer_id"),
            row.get("actual_expert_ids"),
        )
        if (
            type(step) is int
            and step >= 0
            and type(layer) is int
            and layer > 0
            and isinstance(ids, list)
            and ids
            and all(type(expert) is int and expert >= 0 for expert in ids)
        ):
            return f"{step}:{layer}:{ids[0]}"
    raise ValueError("logical trace has no nonfirst-layer expert to omit")


def oracle_output_gate(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    required = ("h0", "h1", "h2", "forced")
    if set(runs) != set(required):
        return {"status": "incomplete-evidence", "reason": "oracle-runs-missing"}
    hashes: dict[str, list[str]] = {}
    for label in required:
        run = runs[label]
        repetitions = run.get("repetitions")
        if (
            run.get("status") != "complete"
            or run.get("repetitions_completed") != 3
            or not isinstance(repetitions, list)
            or len(repetitions) != 3
            or any(not isinstance(row.get("output_sha256"), str) for row in repetitions)
        ):
            return {
                "status": "incomplete-evidence",
                "reason": f"{label}-repetitions-incomplete",
            }
        hashes[label] = [row["output_sha256"] for row in repetitions]
    baseline = hashes["h0"]
    matches = all(values == baseline for values in hashes.values())
    return {
        "status": "outputs-match" if matches else "output-mismatch",
        "per_run_output_sha256": hashes,
        "scope": "three greedy repetitions per horizon; forced omission must match horizon 0",
    }


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"invalid artifact: {path}")
    return value


def _save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}-{uuid4().hex}.new")
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def _point_summary(root: Path, name: str) -> dict[str, Any]:
    return _read(root / "runs/decode-mechanism" / name / "summary.json")


def analyze_trace_point(root: Path, name: str, commit: str) -> dict[str, Any]:
    # Replay ran in the pinned container immediately after profiling. The -S
    # host controller reads only a small numeric result and never imports torch.
    try:
        saved = _read(
            root / "runs/decode-mechanism" / name / "decode-cache-replay.json"
        )
        if saved.get("run_id") != name or saved.get("commit") != commit:
            raise ValueError("replay result identity differs from measured point")
        rows = saved.get("rows")
        if not isinstance(rows, list):
            raise TypeError("replay rank rows missing")
        return aggregate_replay(rows)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return {"status": "incomplete-evidence", "error_type": type(exc).__name__}


def point_matches_commit(summary: dict[str, Any], commit: str, mode: str) -> bool:
    recorded = (
        summary.get("commit")
        if mode == "calibrate"
        else summary.get("contract", {}).get("commit")
    )
    return summary.get("status") == "complete" and recorded == commit


def build_image(
    root: Path, state_dir: Path, state: dict[str, Any], commit: str
) -> None:
    """Build the exact committed image without using the server network."""
    existing = state.get("build")
    if isinstance(existing, dict) and existing.get("status") == "complete":
        if existing.get("source_sha") != commit:
            raise RuntimeError("completed image build belongs to another commit")
        metadata = (root / "build/image.env").read_text(encoding="utf-8").splitlines()
        if (
            f"GIT_SHA={commit}" not in metadata
            or f"IMAGE=moe-flex-local:{commit}" not in metadata
        ):
            raise RuntimeError("completed image metadata differs from checkout")
        return
    if isinstance(existing, dict) and existing.get("status") in ("running", "failed"):
        raise RuntimeError("incomplete image build requires a fresh suite ID")
    state["build"] = {"status": "running", "source_sha": commit}
    _save(state_dir / "state.json", state)
    try:
        with (
            (state_dir / "build.stdout.log").open("x") as stdout,
            (state_dir / "build.stderr.log").open("x") as stderr,
        ):
            result = subprocess.run(
                ["bash", "scripts/server/build.sh"],
                cwd=root,
                env={**os.environ, "FLEXMOE_OFFLINE_BUILD": "1"},
                stdout=stdout,
                stderr=stderr,
                timeout=1800,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(
                "pinned offline image build failed; inspect build.stderr.log"
            )
        metadata = (root / "build/image.env").read_text(encoding="utf-8").splitlines()
        if (
            f"GIT_SHA={commit}" not in metadata
            or f"IMAGE=moe-flex-local:{commit}" not in metadata
        ):
            raise RuntimeError("offline image metadata differs from committed checkout")
        state["build"] = {"status": "complete", "source_sha": commit}
        _save(state_dir / "state.json", state)
        print("offline pinned image: validated", flush=True)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        state["build"] = {
            "status": "failed",
            "source_sha": commit,
            "error_type": type(exc).__name__,
        }
        _save(state_dir / "state.json", state)
        raise


def _run_point(
    root: Path,
    state_dir: Path,
    state: dict[str, Any],
    point: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    name = point["name"]
    existing = state["points"].get(name)
    if existing:
        if (
            existing.get("status") != "complete"
            or existing.get("argv") != point["argv"]
        ):
            raise RuntimeError(
                f"refusing changed or failed prior point: {name}; use a new run ID"
            )
        result = _point_summary(root, name)
        if not point_matches_commit(result, state["commit"], point["mode"]):
            raise RuntimeError(f"saved point no longer matches this commit: {name}")
        return result
    print(f"{name}: started", flush=True)
    state["points"][name] = {**point, "status": "running"}
    _save(state_dir / "state.json", state)
    out = state_dir / (name + ".stdout.log")
    err = state_dir / (name + ".stderr.log")
    try:
        with out.open("x") as stdout, err.open("x") as stderr:
            run = subprocess.run(
                point["argv"] + ["--timeout-s", str(timeout)],
                cwd=root,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout + 300,
                check=False,
            )
        state["points"][name]["exit_code"] = run.returncode
        if run.returncode:
            raise RuntimeError(f"point {name} exited {run.returncode}; see {err}")
        result = _point_summary(root, name)
        if not point_matches_commit(result, state["commit"], point["mode"]):
            raise RuntimeError(f"point {name} missing complete pinned result")
        state["points"][name]["status"] = "complete"
        _save(state_dir / "state.json", state)
        print(f"{name}: complete", flush=True)
        return result
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        state["points"][name]["status"] = "failed"
        state["points"][name]["error_type"] = type(exc).__name__
        _save(state_dir / "state.json", state)
        raise


def _markdown(state: dict[str, Any]) -> str:
    def fmt(value: object, digits: int = 2) -> str:
        return f"{value:,.{digits}f}" if type(value) in (int, float) else "—"

    lines = [
        "# TP4 卸载可行性补充实验",
        "",
        f"- Commit: `{state['commit']}`",
        f"- Run ID: `{state['run_id']}`",
        "- 四张 H100、GPU 使用率上限 0.90、物理安全余量每卡 2,000,000,000 bytes；模型挂载只读。",
        "- 端到端吞吐 = 输出 token 数 / 包含 prefill 的整批墙钟；instrumented 点不纳入该对照。",
        "- 基准工作量与显存预算必须通过配对验证，缺失字段记为不可判定。",
        "",
        "| 实验点 | 模式 | 状态 | 实际 KV/GPU GiB | 输出 token/s 中位数 | decode 平均实际 batch | KV 峰值比例 | 抢占总次 | 等待队列峰值 | TTFT 中位数 s | 请求延迟中位数 s |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in state["points"].items():
        value = state.get("numeric", {}).get(name, {})
        kv = value.get("actual_kv")
        lines.append(
            "| "
            + " | ".join(
                (
                    name,
                    row["mode"] + (" sampled" if "--profile" in row["argv"] else ""),
                    row["status"],
                    fmt(kv / 2**30 if type(kv) is int else None),
                    fmt(value.get("throughput")),
                    fmt(value.get("decode_batch_mean")),
                    fmt(value.get("kv_usage_peak"), 3),
                    fmt(value.get("preemptions_total"), 0),
                    fmt(value.get("waiting_requests_peak"), 0),
                    fmt(value.get("ttft_median_s"), 4),
                    fmt(value.get("request_latency_median_s"), 4),
                )
            )
            + " |"
        )
    capacity = state.get("capacity", {})
    lines += [
        "",
        "## 容量与端到端对照",
        "",
        f"容量门槛：`{capacity.get('status', 'unavailable')}`；native 全驻留={capacity.get('native_bytes_per_rank')}，eager 全驻留={capacity.get('eager_bytes_per_rank')}，两者上限={capacity.get('resident_bytes_per_rank')}，卸载={capacity.get('offload_bytes_per_rank')}，超越较强 resident 的增量={capacity.get('extra_bytes_per_rank')} bytes/GPU。只有超过至少一个真实 KV block 且相同硬件/预算/工作量，才称为卸载独有容量。",
        f"工作预算：native 参考={state.get('budgets', {}).get('native_kv')}；同为 eager 的 resident/offload 比较={state.get('budgets', {}).get('matched_eager_kv')} bytes/GPU（两种模式都能稳定自动分配的较小值）。",
        f"无采样零 miss 四 rank × 三重复：`{state.get('zero_miss', {}).get('status', 'unavailable')}`；同 KV eager 配对：`{state.get('zero_pair', {}).get('status', 'unavailable')}`，卸载/全驻留吞吐比={fmt(state.get('zero_pair', {}).get('offload_over_resident'), 4)}。",
        f"4K、64 个不同请求、并发上限 32，同 KV eager 配对：`{state.get('low_pair', {}).get('status', 'unavailable')}`，卸载/全驻留吞吐比={fmt(state.get('low_pair', {}).get('offload_over_resident'), 4)}。native 用自身 auto 上限作为强基线，不强制与 eager K 相等。上表的 actual batch 是实际观测值，不等同于请求数。",
    ]
    kv_pair = state.get("kv_intervention", {})
    if kv_pair.get("status") == "measured-kv-intervention":
        lines.append(
            f"真正额外 KV 的同卸载路径干预：`{kv_pair['status']}`；大/小 KV 吞吐比={fmt(kv_pair['large_over_small_throughput'], 4)}；两点 workload、硬件、source SHA、执行配置与缓存策略均已核对。同时检查上表 KV 使用峰值、等待队列与抢占，不把单纯分配容量视作性能收益。"
        )
    else:
        lines.append(
            f"额外 KV 吞吐干预：`{kv_pair.get('status', 'not-run')}`；没有通过容量/匹配门槛时，不报告 K1/K0 性能净收益。"
        )
    lines += [
        "",
        "四 rank × 三重复的真实零 miss 判据（全部三项均为 0 才通过）：",
        "",
        "| repetition | rank | miss 次数 | payload bytes | copy launches |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in state.get("zero_miss", {}).get("per_rank_repetition", []):
        lines.append(
            f"| {row['repetition']} | {row['rank']} | {row['misses']} | {row['payload_bytes']} | {row['copy_launches']} |"
        )
    lines += [
        "",
        "## 逐 rank 传输与 instrumented 时间",
        "",
        "`host_gather` 从 host reuse wait 结束后开始：miss 非空才 `index_select` 专家权重，然后总是填充所有专家的 CPU 映射；它在 H2D enqueue 之前结束。此 CPU 累计区间不是纯权重拷贝或 TP 全局净损失。",
        "| 实验点 | payload bytes/GPU（3 次之和，rank 0–3） | cache miss/GPU（3 次之和，rank 0–3） |",
        "|---|---|---|",
    ]
    for label in (
        "zero-offload",
        "low-offload",
        "low-extra-kv",
        "trace-16",
        "trace-32",
    ):
        value = state.get("numeric", {}).get(f"{state['run_id']}-{label}", {})
        if value:
            lines.append(
                f"| {label} | {value.get('payload_bytes_per_rank')} | {value.get('misses_per_rank')} |"
            )
    lines += [
        "",
        "instrumented CUDA span 为 prepared inputs → execute_model return，含测量扰动；host_gather 为逐 rank 的 CPU 区间和，绝不能从端到端时间中直接相减。",
        "",
        "| 样本 | repetition | rank | CUDA step p50 ms | CUDA step p95 ms | host_gather CPU sum s | 采样 payload bytes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("eager-resident", "zero_timing_resident"),
        ("zero-miss-offload", "zero_timing_offload"),
        ("batch-16-offload", "trace_timing_16"),
        ("batch-32-offload", "trace_timing_32"),
    ):
        for row in state.get(key, {}).get("per_rank_repetition", []):
            lines.append(
                f"| {label} | {row['repetition']} | {row['rank']} | {fmt(row.get('model_step_cuda_p50_ms'), 3)} | {fmt(row.get('model_step_cuda_p95_ms'), 3)} | {fmt(row.get('host_gather_cpu_sum_s'), 3)} | {row.get('payload_bytes')} |"
            )
    lines += ["", "## 同预算缓存重放（不是预取）", ""]
    for size in (16, 32):
        data = state.get(f"replay_{size}", {})
        lines.append(
            f"batch={size}：`{data.get('status', 'unavailable')}`；当前策略 miss={data.get('current_policy_misses')}，LRU miss={data.get('lru_misses')}，未来知情淘汰 miss={data.get('future_aware_misses')}，未来淘汰可避免 payload={data.get('future_aware_saved_bytes')} bytes（四 rank、三次采样合计）。"
        )
    lines += [
        "",
        "仅在原策略以捕获时的真实驻留/缓存状态逐层复现 miss 和 payload，且所有 rank 完整连续 64 步时，重放统计才成立。若生产策略为 LFU，其先前 LRU 顺序未被记录，LRU 从相同占用以 slot 顺序初始化，仅作条件性对照。未来知情淘汰不提前加载，也不改变 ingress、KV 容量或执行成本。",
        "",
        "## 尚未测量的边界",
        "",
        "独立 prefill/decode 墙钟、KV 阻塞准入时长、TPOT/ITL 的逐 token 尾延迟、跨层真实提前窗口、权重提前到位比例、强制漏预测补载与端到端 Oracle 预取收益均 **未测量**；没有输出不会被填成零。后续只有修复零 miss 固定开销并观察到目标负载的真实 KV 压力后，才值得实施有限提前窗口的实际 H2D 预取。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=7200)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    if root != Path("/home/jovyan/wangtonghan/moe-flex"):
        parser.error("only the authorized server project can run points")
    if os.environ.get("GPU_IDS") != "0,1,2,3" or args.timeout_s < 120:
        parser.error("four explicitly selected GPUs and a bounded timeout required")
    run_id = validate_run_id(
        args.run_id
        or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:6]
        )
    )
    state_dir = root / "runs/decode-decision" / run_id
    validate_branch(
        subprocess.check_output(
            ["git", "-C", str(root), "branch", "--show-current"], text=True
        ).strip()
    )
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--",
            ".",
            ":(exclude)docs/results/**",
        ],
        text=True,
    )
    if status.strip():
        raise RuntimeError("commit the server code before the experiment")
    if args.resume:
        state = _read(state_dir / "state.json")
        if (
            state.get("commit") != commit
            or state.get("run_id") != run_id
            or state.get("status") not in ("running", "failed")
        ):
            raise RuntimeError("resume must use a matching, incomplete run")
        state["status"] = "running"
    else:
        state_dir.mkdir(parents=True, exist_ok=False)
        state = {
            "run_id": run_id,
            "commit": commit,
            "status": "running",
            "points": {},
            "numeric": {},
        }
    _save(state_dir / "state.json", state)

    def point(
        mode: str,
        label: str,
        context: int,
        kv: int | None,
        batch: int,
        *,
        profile: bool = False,
        ratio: str = "0.8",
        output: int = 256,
    ) -> dict[str, Any]:
        name = f"{run_id}-{label}"
        profile_path = f"runs/decode-mechanism/{run_id}-cal-{context}/profile.json"
        record = point_arguments(
            mode,
            name,
            context,
            kv,
            batch,
            {"profile_path": profile_path},
            profile=profile,
            resident_ratio=ratio,
            output_length=output,
        )
        result = _run_point(root, state_dir, state, record, args.timeout_s)
        if mode != "calibrate":
            state["numeric"][name] = point_metrics(result)
            _save(state_dir / "state.json", state)
        return result

    try:
        build_image(root, state_dir, state, commit)
        point("calibrate", "cal-1024", 1024, None, 32, output=8)
        point("calibrate", "cal-4096", 4096, None, 32, output=8)
        resident_auto = point("native", "native-auto", 4096, None, 1, output=256)
        eager_auto = point("matched-resident", "eager-auto", 4096, None, 1, output=256)
        offload_auto = point("offload", "offload-auto", 4096, None, 1, output=256)
        state["capacity"] = capacity_gate(resident_auto, eager_auto, offload_auto)
        if state["capacity"]["status"] == "incomplete-evidence":
            raise RuntimeError(
                "capacity point incomplete; no budget selection possible"
            )
        state["budgets"] = select_budgets(resident_auto, eager_auto, offload_auto)
        native_kv, native_block = _actual_kv(resident_auto)
        eager_kv, eager_block = _actual_kv(eager_auto)
        offload_kv, offload_block = _actual_kv(offload_auto)
        capacities = {
            "native_kv": native_kv,
            "eager_kv": eager_kv,
            "offload_kv": offload_kv,
            "resident_ceiling": max(native_kv, eager_kv),
            "k_pair": min(eager_kv, offload_kv),
            "legacy_kv": 32_815_054_848,
            "kv_block_bytes": max(native_block, eager_block, offload_block),
        }
        trace_paths = {
            batch: (
                f"runs/decode-mechanism/{run_id}-trace-{batch}/"
                "decode-logical-trace-rep-000.json.gz"
            )
            for batch in (16, 32)
        }
        plan = build_decision_plan(
            run_id=run_id,
            capacities=capacities,
            trace_paths=trace_paths,
        )
        state["selected_capacities"] = capacities
        state["point_plan"] = plan
        completed: dict[str, dict[str, Any]] = {
            "native-auto": resident_auto,
            "eager-auto": eager_auto,
            "offload-auto": offload_auto,
        }
        _save(state_dir / "state.json", state)

        for spec in plan:
            label, name = spec["label"], spec["name"]
            if spec["status"] == "skipped":
                state["points"].setdefault(name, dict(spec))
                _save(state_dir / "state.json", state)
                continue
            if label == "oracle-forced-omission":
                omission = derive_forced_omission(root / trace_paths[32])
                index = spec["argv"].index("--force-omission")
                spec["argv"][index + 1] = omission
                state["forced_omission"] = omission
            result = _run_point(root, state_dir, state, spec, args.timeout_s)
            completed[label] = result
            if spec["mode"] != "calibrate":
                state["numeric"][name] = point_metrics(result)
            if label.startswith("mechanism-"):
                state.setdefault("fixed_batch_gates", {})[label] = (
                    fixed_actual_batch_gate(result, target=32)
                )
            if label == "zero-offload":
                state["zero_miss"] = zero_miss_gate(result)
                state["zero_pair"] = compare_pair(completed["zero-resident"], result)
            if label == "zero-offload-profile":
                state["zero_timing_offload"] = profile_timing_point(root, name)
            if label == "service-offload-kpair":
                state["low_pair"] = compare_pair(
                    completed["service-eager-kpair"], result
                )
            if label == "service-offload-k1" and "service-offload-k0" in completed:
                state["kv_intervention"] = compare_offload_kv(
                    completed["service-offload-k0"], result
                )
            if label in ("trace-16", "trace-32"):
                batch = int(label.rsplit("-", 1)[1])
                state[f"trace_timing_{batch}"] = profile_timing_point(root, name)
                state[f"replay_{batch}"] = analyze_trace_point(root, name, commit)
            _save(state_dir / "state.json", state)

        for batch in (16, 32):
            horizon_runs = {
                f"h{horizon}": completed[f"oracle-{batch}-h{horizon}"]
                for horizon in (0, 1, 2)
            }
            if batch == 32:
                horizon_runs["forced"] = completed["oracle-forced-omission"]
                state["oracle_output_gate_32"] = oracle_output_gate(horizon_runs)
            else:
                hashes = {
                    label: [row["output_sha256"] for row in run["repetitions"]]
                    for label, run in horizon_runs.items()
                }
                state["oracle_output_gate_16"] = {
                    "status": "outputs-match"
                    if hashes["h0"] == hashes["h1"] == hashes["h2"]
                    else "output-mismatch",
                    "per_run_output_sha256": hashes,
                }
        state["status"] = "complete"
        _save(state_dir / "state.json", state)
        public = root / "docs/results" / f"decode-decision-{run_id}"
        public.mkdir(parents=True, exist_ok=False)
        report_module = runpy.run_path(
            str(root / "src/flexmoe/analysis/decode_oracle_report.py")
        )
        report_json, report_markdown = report_module["build_oracle_report"](state)
        _save(public / "report.json", report_json)
        (public / "report.md").write_text(report_markdown, encoding="utf-8")
        print(f"Report: {public / 'report.md'}", flush=True)
        return 0
    except BaseException:
        state["status"] = "failed"
        _save(state_dir / "state.json", state)
        raise


if __name__ == "__main__":
    sys.exit(main())
