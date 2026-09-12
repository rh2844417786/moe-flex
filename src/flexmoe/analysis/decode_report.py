"""Typed public decode reports; raw traces and identifying strings stay local."""

from __future__ import annotations

import csv
import gzip
import json
import zlib
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

from .decode_counts import summarize_activations, validate_activation_row
from .decode_suite import (
    compare_runs,
    digest,
    integer,
    labels_valid,
    mapping,
    number,
    read_artifact,
    validate_run,
)
from .schema import diagnostic_artifact

STATUSES = {
    "complete",
    "failed",
    "running",
    "rejected",
    "measured",
    "instrumented",
    "unavailable",
    "partial",
    "unreached",
    "insufficient-window",
    "incomplete-layers",
    "not-requested",
    "not-applicable",
    "eligible",
    "ineligible",
}
STAGES = {"capture_stop", "capture_save", "scheduler", "memory", "cpu_memory"}
ERROR_TYPES = {
    "ValueError",
    "RuntimeError",
    "OSError",
    "TypeError",
    "KeyError",
    "AssertionError",
    "OutOfMemoryError",
    "TimeoutError",
    "MemoryError",
    "FileNotFoundError",
    "PermissionError",
}
COUNTERS = (
    "resident_hits",
    "cache_hits",
    "unique_misses",
    "unique_demands",
    "first_loads",
    "reloads",
    "loaded_bytes",
    "metadata_bytes",
    "promotion_bytes",
    "evictions",
    "bypasses",
)
CUDA_TIMES = ("load_s", "payload_s", "metadata_s", "compute_s", "promotion_s", "span_s")
CPU_TIMES = (
    "route_d2h_s",
    "policy_cpu_s",
    "host_reuse_wait_s",
    "host_gather_s",
    "h2d_enqueue_s",
    "compute_enqueue_s",
)
MEMORY = (
    "rank",
    "total_gpu_bytes",
    "free_gpu_bytes",
    "torch_allocated_bytes",
    "torch_reserved_bytes",
    "torch_peak_allocated_bytes",
    "torch_peak_reserved_bytes",
    "available_kv_cache_bytes",
    "model_memory_bytes",
    "kv_cache_allocated_bytes",
    "kv_cache_declared_bytes",
    "num_gpu_blocks",
)


def status(value: object) -> str:
    return str(value) if isinstance(value, str) and value in STATUSES else "unavailable"


def numbers(raw: object, keys: Sequence[str], *, ints: bool = False) -> dict[str, Any]:
    row = mapping(raw)
    check = integer if ints else number
    return {key: row[key] for key in keys if check(row.get(key))}


def entries(raw: object, limit: int = 16384) -> list[dict[str, Any]]:
    return (
        [mapping(x) for x in raw[:limit] if isinstance(x, dict)]
        if isinstance(raw, list)
        else []
    )


def distribution(raw: object, allowed: set[str] | None = None) -> dict[str, int]:
    return {
        k: v
        for k, v in list(mapping(raw).items())[:2048]
        if isinstance(k, str)
        and (k in allowed if allowed else k.isdigit() and 0 < int(k) <= 1000000)
        and integer(v)
    }


def scheduler(raw: object) -> dict[str, Any]:
    result: dict[str, Any] = {
        "scope": "frontend samples; not aligned with worker steps; arithmetic sample means; KV usage is a fraction, not occupied bytes"
    }
    for name in (
        "kv_cache_usage",
        "running_requests",
        "waiting_requests",
        "preemptions",
    ):
        row = mapping(mapping(raw).get(name))
        item = {
            "status": status(row.get("status")),
            **numbers(row, ("samples",), ints=True),
        }
        for key in ("mean", "peak", "total"):
            value = row.get(key)
            item[key] = (
                value
                if number(value) and (name != "kv_cache_usage" or value <= 1)
                else None
            )
        if item["status"] in ("unavailable", "partial"):
            item["unavailable_reason"] = (
                "invalid-samples"
                if mapping(mapping(raw).get("invalid_samples")).get(name, 0)
                else "unavailable-at-source"
            )
        result[name] = item
    result["invalid_samples"] = numbers(
        mapping(raw).get("invalid_samples"),
        ("kv_cache_usage", "running_requests", "waiting_requests", "preemptions"),
        ints=True,
    )
    return result


def observation(raw: object) -> dict[str, Any]:
    row = mapping(raw)
    return {
        **numbers(
            row,
            (
                "rank",
                "observed_steps",
                "captured_steps",
                "selected_steps",
                "missing_layer_events",
                "target_batch",
                "capture_steps",
                "min_capture_steps",
                "buffer_bytes",
                "trace_budget_bytes",
            ),
            ints=True,
        ),
        "coverage_status": status(row.get("coverage_status")),
        "phase_step_counts": distribution(
            row.get("phase_step_counts"), {"prefill", "mixed", "decode", "unknown"}
        ),
        "decode_batch_step_counts": distribution(row.get("decode_batch_step_counts")),
        "skipped_steps": numbers(
            row.get("skipped_steps"),
            ("non_decode", "non_target_batch", "capacity"),
            ints=True,
        ),
    }


def repetition(raw: object) -> dict[str, Any]:
    row = mapping(raw)
    final = mapping(row.get("observer_finalization"))
    errors = []
    for error in entries(final.get("errors"), 32):
        stage = error.get("stage") if error.get("stage") in STAGES else "other-stage"
        kind = (
            error.get("error_type")
            if error.get("error_type") in ERROR_TYPES
            else "other-error"
        )
        errors.append(f"{stage}:{kind}")
    result = {
        **numbers(row, ("repetition", "generated_tokens", "request_count"), ints=True),
        **numbers(row, ("elapsed_s", "ttft_median_s", "request_latency_median_s")),
        "generation_status": status(row.get("generation_status")),
        "measurement_status": status(row.get("measurement_status", "complete")),
        "capture_status": status(row.get("capture_status")),
        "memory_status": status(row.get("memory_status")),
        "observer_finalization": status(final.get("status")),
        "observer_errors": errors,
        "evidence_kind": status(row.get("evidence_kind")),
        "memory": [
            numbers(x, MEMORY, ints=True) for x in entries(row.get("memory"), 4)
        ],
        "cpu_memory": {
            "status": status(mapping(row.get("cpu_memory")).get("status")),
            **numbers(row.get("cpu_memory"), ("peak_rss_bytes",), ints=True),
            "scope": "frontend lifetime high-water; excludes live workers",
        },
        "scheduler": scheduler(row.get("scheduler")),
        "worker_observations": [
            observation(x) for x in entries(row.get("worker_observations"), 4)
        ],
    }
    # The saved numeric sample remains available even when rejected. It is never a headline.
    if (
        number(row.get("elapsed_s"))
        and row["elapsed_s"] > 0
        and integer(row.get("generated_tokens"))
    ):
        result["sample_tokens_s"] = row["generated_tokens"] / row["elapsed_s"]
    return result


def _worker_batch(obs: Mapping[str, Any], minimum: int) -> dict[str, Any]:
    counts = mapping(obs.get("decode_batch_step_counts"))
    steps = sum(counts.values())
    return {
        **numbers(obs, ("rank",), ints=True),
        "status": "unavailable"
        if not steps
        else "insufficient"
        if steps < minimum
        else "measured",
        "decode_steps": steps,
        "mean_actual_batch": sum(int(batch) * count for batch, count in counts.items())
        / steps
        if steps
        else None,
        "peak_actual_batch": max(map(int, counts)) if steps else None,
        "scope": "per-rank pure decode step-weighted mean; never summed over TP ranks",
    }


def comparison_telemetry(
    runs: Sequence[Mapping[str, Any]], checks: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Bounded public admission evidence, independent of measured timing eligibility."""
    arms: list[dict[str, Any]] = []
    for label, run, check in zip(("A", "B", "C", "native"), runs, checks):
        raw_reps = entries(run.get("repetitions"), 1000)
        minimum = mapping(run.get("contract")).get("min_capture_steps")
        minimum = minimum if integer(minimum, 1) else 64
        reps = []
        for raw_rep in raw_reps:
            observers = [
                observation(x) for x in entries(raw_rep.get("worker_observations"), 4)
            ]
            observers.sort(key=lambda x: x.get("rank", 4))
            reps.append(
                {
                    **numbers(
                        raw_rep,
                        ("repetition", "request_count", "generated_tokens"),
                        ints=True,
                    ),
                    "generation_status": status(raw_rep.get("generation_status")),
                    "capture_status": status(raw_rep.get("capture_status")),
                    "measurement_status": status(
                        raw_rep.get("measurement_status", "complete")
                    ),
                    "worker_observations": observers,
                    "worker_batch": [_worker_batch(x, minimum) for x in observers],
                    "scheduler": scheduler(raw_rep.get("scheduler")),
                }
            )
        arms.append(
            {
                "arm": label,
                "output_variation": check.get("output_variation"),
                "timing_eligible": check.get("timing_eligible") is True,
                "repetitions_exported": len(reps),
                "repetitions_requested": numbers(
                    run.get("contract"), ("repetitions_requested",), ints=True
                ).get("repetitions_requested"),
                "repetitions": reps,
            }
        )
    b, c = arms[1:3]
    deltas = []
    worker_available = (
        len(b["repetitions"])
        == len(c["repetitions"])
        == b["repetitions_requested"]
        == c["repetitions_requested"]
        and len(b["repetitions"]) >= 3
    )
    for left, right in zip(b["repetitions"], c["repetitions"]):
        for rep in (left, right):
            batch = rep["worker_batch"]
            observers = rep["worker_observations"]
            if (
                len(batch) != 4
                or [x.get("rank") for x in batch] != [0, 1, 2, 3]
                or any(x["status"] != "measured" for x in batch)
                or any(
                    x["decode_batch_step_counts"]
                    != observers[0]["decode_batch_step_counts"]
                    for x in observers
                )
            ):
                worker_available = False
        if worker_available:
            deltas.append(
                right["worker_batch"][0]["mean_actual_batch"]
                - left["worker_batch"][0]["mean_actual_batch"]
            )
    worker_status = "insufficient"
    if worker_available and deltas:
        worker_status = (
            "observed-increase"
            if max(deltas) > 0 and min(deltas) >= 0
            else "no-observed-increase"
            if max(deltas) <= 0
            else "insufficient"
        )
    frontend = [
        mapping(rep["scheduler"].get(metric))
        for arm in (b, c)
        for rep in arm["repetitions"]
        for metric in (
            "kv_cache_usage",
            "running_requests",
            "waiting_requests",
            "preemptions",
        )
    ]
    frontend_present = [field for field in frontend if integer(field.get("samples"), 1)]
    frontend_complete = bool(frontend) and all(
        field.get("status") == "measured"
        and integer(field.get("samples"), 1)
        and (number(field.get("mean")) or integer(field.get("total")))
        for field in frontend
    )
    causal_status = (
        "unavailable"
        if not frontend_present
        else "available"
        if frontend_complete and worker_available
        else "incomplete"
    )
    if not (b["timing_eligible"] and c["timing_eligible"]):
        causal_status = "incomplete"
    return {
        "arms": arms,
        "admission_evidence": {
            "worker_batch_status": worker_status,
            "causal_evidence_status": causal_status,
            "scheduler_status": "measured"
            if frontend_complete
            else "partial"
            if frontend_present
            else "unavailable",
            "paired_repetitions": len(deltas) if worker_available else 0,
            "scope": "B to C mean actual pure-decode batch per repetition; require consistent TP rank observations, use rank0 once; frontend samples are separate and not aligned to worker steps",
            "interpretation": "timing remains measured with unchanged batch or unavailable scheduler; availability is not proof the entire throughput change was caused by admission; KV usage fraction is not occupied bytes",
        },
    }


def pool_summary(raw: object) -> dict[str, Any]:
    pool = mapping(raw)
    rows = entries(pool.get("rows"), 65536)
    complete = [x for x in rows if x.get("status") == "complete"]
    cuda = [
        x for x in complete if mapping(x.get("cuda_timing")).get("status") == "measured"
    ]
    result: dict[str, Any] = {
        **numbers(pool, ("capacity", "dropped_rows", "unmeasured_rows"), ints=True),
        "observed_rows": len(rows),
        "complete_rows": len(complete),
        "failed_rows": len(rows) - len(complete),
        "cuda_measured_rows": len(cuda),
        "cuda_unavailable_rows": len(complete) - len(cuda),
        "timing_scope": "local per-rank serialized service; never TP-global net loss; CPU and GPU intervals must not be added",
        "prefetch_status": "not-applicable",
        "transfer_schedule": "serialized-on-compute-stream",
    }
    for key in COUNTERS:
        values = [x[key] for x in complete if integer(x.get(key))]
        result[key] = sum(values) if len(values) == len(complete) and complete else None
    for key in CUDA_TIMES:
        values = [mapping(x.get("cuda_timing")).get(key) for x in cuda]
        result[key] = sum(values) if cuda and all(number(x) for x in values) else None
    for key in CPU_TIMES:
        values = [mapping(x.get("cpu_timing")).get(key) for x in complete]
        result[key] = (
            sum(values) if complete and all(number(x) for x in values) else None
        )
    return result


def capture_summary(
    raw: Mapping[str, Any], parent: Mapping[str, Any]
) -> dict[str, Any]:
    obs, geometry = mapping(raw.get("observation")), mapping(raw.get("geometry"))
    result: dict[str, Any] = {
        **numbers(raw, ("rank", "repetition"), ints=True),
        "generation_status": status(raw.get("generation_status")),
        "evidence_kind": status(raw.get("evidence_kind")),
        **observation(obs),
        "activation_status": "unavailable",
        "activation_summaries": [],
        "pool": pool_summary(obs.get("pool_profile")),
        "model_step_span_scope": "per-rank instrumented prepared-inputs through execute_model return; not whole generation",
    }
    spans = [
        x["cuda_s"]
        for x in entries(obs.get("model_step_spans"))
        if x.get("status") == "measured" and number(x.get("cuda_s"))
    ]
    result["model_step_spans"] = {
        "status": "measured" if spans else "unavailable",
        "samples": len(spans),
        "sum_cuda_s": sum(spans) if spans else None,
    }
    if not labels_valid(raw, "decode-profile"):
        result["activation_status"] = "artifact-labels"
        return result
    if parent and (
        raw.get("contract") != parent.get("contract")
        or geometry != parent.get("geometry")
    ):
        result["activation_status"] = "capture-cohort-mismatch"
        return result
    rows = obs.get("activation_rows")
    if not rows:
        return result
    try:
        layers, experts, top_k = (
            geometry[k] for k in ("total_layers", "num_experts", "top_k")
        )
        if (
            not all(integer(x, 1) for x in (layers, experts, top_k))
            or layers > 256
            or experts > 4096
            or not isinstance(rows, list)
            or len(rows) > 65536
        ):
            raise ValueError("bounded-geometry")
        valid = [
            validate_activation_row(x, layers=layers, experts=experts, top_k=top_k)
            for x in rows
        ]
        groups: dict[int, list[dict[str, Any]]] = {}
        for row in valid:
            step = row["step"]
            if integer(step):
                groups.setdefault(step, []).append(row)
        complete = [
            x
            for group in groups.values()
            if len(group) == layers and len({r["layer"] for r in group}) == layers
            for x in group
        ]
        minimum = mapping(raw.get("contract")).get("min_capture_steps", 64)
        if not integer(minimum, 1):
            raise ValueError("minimum")
        summaries = [
            summarize_activations(
                complete,
                target_batch=batch,
                layers=layers,
                experts=experts,
                top_k=top_k,
                min_steps=minimum,
            )
            for batch in sorted({x["actual_batch"] for x in complete})
        ]
        result["activation_summaries"] = summaries
        result["activation_status"] = (
            "complete"
            if complete and len(complete) == len(valid)
            else "incomplete-layers"
        )
        result["incomplete_rows"] = len(valid) - len(complete)
    except (ValueError, TypeError, KeyError):
        result["activation_status"] = "invalid-histogram"
    return result


def _load(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        return read_artifact(path), None
    except FileNotFoundError:
        return {}, "missing-artifact"
    except EOFError:
        return {}, "truncated-compression"
    except (gzip.BadGzipFile, zlib.error):
        return {}, "invalid-compression"
    except (json.JSONDecodeError, UnicodeError):
        return {}, "malformed-json"
    except OSError:
        return {}, "io-error"
    except (ValueError, TypeError):
        return {}, "artifact-schema"


def summarize_directory(source: Path) -> dict[str, Any]:
    raw, error = _load(source / "summary.json")
    try:
        validation = validate_run(raw)
    except (TypeError, ValueError, KeyError):
        validation = {
            "status": "ineligible",
            "timing_eligible": False,
            "reasons": ["artifact-schema"],
        }
    if error:
        validation["reasons"] = sorted(set(validation["reasons"]) | {error})
    contract = mapping(raw.get("contract"))
    mode = (
        raw.get("mode")
        if raw.get("mode") in {"native", "matched-resident", "offload"}
        else "unavailable"
    )
    result: dict[str, Any] = diagnostic_artifact(
        "decode-report",
        {
            "mode": mode,
            "status": status(raw.get("status")),
            "evidence_kind": status(raw.get("evidence_kind")),
            "validation": validation,
            "cohort_sha256": digest(
                {
                    k: v
                    for k, v in contract.items()
                    if k not in ("engine_policy_sha256",)
                }
            ),
            "workload": numbers(
                contract,
                (
                    "batch_size",
                    "context_length",
                    "output_length",
                    "max_num_seqs",
                    "max_num_batched_tokens",
                    "target_batch",
                    "repetitions_requested",
                    "calibration_count",
                    "unique_selected_request_count",
                    "repeated_request_count",
                ),
                ints=True,
            ),
            "actual_kv": numbers(
                raw.get("actual_kv"),
                (
                    "allocated_bytes_per_rank",
                    "requested_bytes",
                    "num_gpu_blocks",
                    "bytes_per_block",
                    "rounding_bytes",
                ),
                ints=True,
            ),
            "geometry": numbers(
                raw.get("geometry"),
                ("total_layers", "num_experts", "top_k", "expert_bytes"),
                ints=True,
            ),
            "cache": numbers(
                contract.get("expert_cache"),
                ("resident_ratio", "cache_slots", "calibration_count"),
            ),
            "memory": [
                numbers(x, MEMORY, ints=True) for x in entries(raw.get("memory"), 4)
            ],
            "final_memory": [
                numbers(x, MEMORY, ints=True)
                for x in entries(raw.get("final_memory"), 4)
            ],
            "expert_cache_stats": [
                numbers(
                    x,
                    (
                        "rank",
                        "resident_slots",
                        "cache_slots",
                        "ingress_slots",
                        "physical_slots",
                        "expert_bytes",
                        "host_source_bytes",
                        "gpu_pool_bytes",
                        "gpu_resident_bytes",
                        "gpu_cache_bytes",
                        "gpu_ingress_bytes",
                        "gpu_metadata_bytes",
                        "net_freed_bytes",
                        "startup_resident_h2d_bytes",
                        "h2d_bytes",
                        "copy_launches",
                        "unique_demands",
                    ),
                    ints=True,
                )
                for x in entries(raw.get("expert_cache_stats"), 4)
            ],
            "repetitions": [],
            "captures": [],
            "artifact_errors": [],
            "interpretation": "elapsed includes prefill; profile is instrumented; routed-expert histograms are logically replicated across TP ranks and must not be summed; selection probability is edge share; allocated KV is not scheduler occupancy",
        },
    )
    smoke, smoke_error = _load(source / "smoke.json")
    if smoke_error and mapping(raw.get("smoke")):
        smoke = mapping(raw["smoke"])
        smoke_error = None
    result["smoke"] = {
        **numbers(smoke, ("generated_tokens", "request_count"), ints=True),
        **numbers(smoke, ("elapsed_s",)),
        "status": "complete" if labels_valid(smoke, "decode-smoke") else "unavailable",
    }
    launcher, _ = _load(source / "launcher.json")
    result["launcher"] = {
        "status": status(launcher.get("status")),
        **numbers(launcher, ("exit_code",), ints=True),
    }
    if smoke_error:
        result["artifact_errors"].append({"kind": "smoke", "reason": smoke_error})
    files = sorted(source.glob("rep-[0-9][0-9][0-9].json")) + sorted(
        source.glob("failed-rep-[0-9][0-9][0-9].json")
    )
    for path in files:
        row, failure = _load(path)
        if failure:
            result["artifact_errors"].append({"kind": "repetition", "reason": failure})
        else:
            result["repetitions"].append(repetition(row))
    if not files:
        result["repetitions"] = [
            repetition(x) for x in entries(raw.get("repetitions"), 1000)
        ]
        if isinstance(raw.get("failed_measurement"), dict):
            result["repetitions"].append(repetition(raw["failed_measurement"]))
    for path in sorted(source.glob("decode-rep-*-rank-*.json.gz"))[:4000]:
        row, failure = _load(path)
        if failure:
            result["artifact_errors"].append({"kind": "capture", "reason": failure})
        else:
            result["captures"].append(capture_summary(row, raw))
    return result


def _svg(
    path: Path,
    points: Sequence[tuple[float, float]],
    xlabel: str,
    ylabel: str,
    *,
    labels: Sequence[str] = (),
) -> None:
    if not points:
        return
    xmax, ymax = max(x for x, _ in points), max(y for _, y in points)
    dots = []
    colors = {
        "A-matched-resident": "#2455a4",
        "B-offload": "#b64b38",
        "C-offload": "#287a4d",
    }
    for index, (x, y) in enumerate(points):
        px, py = 70 + 480 * x / (xmax or 1), 300 - 240 * y / (ymax or 1)
        label = labels[index] if index < len(labels) and labels[index] in colors else ""
        color = colors.get(label, "#2455a4")
        dots.append(
            f'<circle cx="{px:.3f}" cy="{py:.3f}" r="4" data-x="{x:g}" data-y="{y:g}" data-arm="{label}" fill="{color}"><title>{label}: {x:g}, {y:g}</title></circle>'
        )
        if label:
            dots.append(
                f'<text x="{px + 7:.3f}" y="{py:.3f}" fill="{color}">{label[0]}</text>'
            )
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360">'
        '<rect width="640" height="360" fill="white"/><path d="M70 40V300H580" fill="none" stroke="#333"/>'
        f'<text x="180" y="340">{escape(xlabel)}</text><text x="75" y="24">{escape(ylabel)}</text>'
        f'<text x="70" y="318">0</text><text x="530" y="318">{xmax:g}</text><text x="8" y="60">{ymax:g}</text>'
        + "".join(dots)
        + "</svg>\n"
    )


def _admission_markdown(comparison: Mapping[str, Any]) -> list[str]:
    evidence = mapping(comparison.get("admission_evidence"))
    lines = [
        f"B→C 实际batch观察：{evidence.get('worker_batch_status', 'unavailable')}；指标可用性：{evidence.get('causal_evidence_status', 'unavailable')}；frontend scheduler：{evidence.get('scheduler_status', 'unavailable')}。",
        "",
        "下表 worker 使用 rank0 一份逻辑计数，mean 按纯decode步骤加权；scheduler 使用独立frontend样本，二者不逐步对齐。batch未增加或scheduler不可用不会抹掉有效墙钟结果，指标齐全也不证明全部时间变化由准入导致。",
        "",
        "| arm | rep | worker mean / decode steps / status | KV usage mean / samples / status | running mean / samples / status | waiting mean / samples / status | preemptions total / samples / status | output audit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def value(raw: object) -> str:
        return f"{raw:.6g}" if number(raw) else "unavailable"

    for arm in entries(comparison.get("arms"), 4):
        audit = mapping(arm.get("output_variation")).get("status", "unavailable")
        for rep in entries(arm.get("repetitions"), 1000):
            worker = next(
                (x for x in entries(rep.get("worker_batch"), 4) if x.get("rank") == 0),
                {},
            )
            fields = [
                f"{value(worker.get('mean_actual_batch'))} / {value(worker.get('decode_steps'))} / {worker.get('status', 'unavailable')}"
            ]
            for metric in (
                "kv_cache_usage",
                "running_requests",
                "waiting_requests",
                "preemptions",
            ):
                item = mapping(mapping(rep.get("scheduler")).get(metric))
                fields.append(
                    f"{value(item.get('total' if metric == 'preemptions' else 'mean'))} / {value(item.get('samples'))} / {item.get('status', 'unavailable')}"
                )
            lines.append(
                f"| {arm['arm']} | {rep.get('repetition')} | "
                + " | ".join(fields)
                + f" | {audit} |"
            )
    return [*lines, ""]


def export_report(
    source: Path, output: Path, *, comparisons: Sequence[Path] = ()
) -> dict[str, Any]:
    result = summarize_directory(source)
    if comparisons:
        rows = [read_artifact(path / "summary.json") for path in [source, *comparisons]]
        result["comparison"] = compare_runs(
            rows[0], rows[1], rows[2], rows[3] if len(rows) == 4 else None
        )
        if result["comparison"]["status"] == "measured":
            result["comparison"]["points"] = [
                {
                    "arm": arm,
                    "allocated_kv_bytes_per_rank": row["actual_kv"][
                        "allocated_bytes_per_rank"
                    ],
                    "throughput_tokens_s": validate_run(row)[
                        "throughput_median_tokens_s"
                    ],
                }
                for arm, row in zip(
                    ("A-matched-resident", "B-offload", "C-offload"), rows[:3]
                )
            ]
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    with (output / "report.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("record", "rank", "repetition", "metric", "value", "status"))
        for key, value in mapping(result.get("comparison")).items():
            if number(value, -1e30):
                writer.writerow(
                    ("comparison", "", "", key, value, result["comparison"]["status"])
                )
        for row in result["repetitions"]:
            for key in ("elapsed_s", "generated_tokens", "sample_tokens_s"):
                writer.writerow(
                    (
                        "repetition",
                        "",
                        row.get("repetition"),
                        key,
                        row.get(key),
                        row["measurement_status"],
                    )
                )
        for row in result["captures"]:
            for key in (*COUNTERS, *CUDA_TIMES, *CPU_TIMES):
                writer.writerow(
                    (
                        "local-pool",
                        row.get("rank"),
                        row.get("repetition"),
                        key,
                        row["pool"].get(key),
                        row["evidence_kind"],
                    )
                )
            for summary in row["activation_summaries"]:
                for layer in summary["per_layer"]:
                    for expert, count in enumerate(layer["histogram_total"]):
                        writer.writerow(
                            (
                                "histogram",
                                row.get("rank"),
                                row.get("repetition"),
                                f"batch={summary['target_batch']};layer={layer['layer']};expert={expert}",
                                count,
                                "instrumented",
                            )
                        )
    val = result["validation"]
    markdown = [
        "# 解码机制实测报告",
        "",
        f"模式：{result['mode']}；证据：{result['evidence_kind']}；计时资格：{val['status']}。",
        "",
        "验证原因：" + (", ".join(val["reasons"]) or "通过") + "。",
        "",
        "整批输出重复变化审计："
        + str(mapping(val.get("output_variation")).get("status", "unavailable"))
        + "；变化不作为吞吐资格门槛，固定输出长度、有效hash、smoke与失败校验仍保留。",
        "",
        "时间为包含 prefill 的整批生成墙钟；采集运行不进入正式吞吐。专家覆盖只针对 routed experts。TP 各 rank 的逻辑选择计数不能相加。",
        "",
        "KV allocated 是实际分配量；scheduler usage 是采样占用比例，不能冒充精确已使用字节。CPU 等待和 GPU 本地服务间隔不可直接相加；本地 load span 不代表 TP 全局净损失。",
        "",
        f"保存 {len(result['repetitions'])} 个数值样本、{len(result['captures'])} 个 rank 采集。失败状态及数值见 report.json / report.csv。",
        "",
    ]
    if val["timing_eligible"]:
        markdown += [
            f"三次以上有效重复的吞吐中位数：{val['throughput_median_tokens_s']:.6g} output tokens/s。",
            "",
        ]
        _svg(
            output / "kv-throughput.svg",
            [
                (point["allocated_kv_bytes_per_rank"], point["throughput_tokens_s"])
                for point in mapping(result.get("comparison")).get("points", [])
            ]
            or [
                (
                    result["actual_kv"]["allocated_bytes_per_rank"],
                    val["throughput_median_tokens_s"],
                )
            ],
            "actual allocated KV bytes / rank",
            "measured output tokens / s",
            labels=[
                point["arm"]
                for point in mapping(result.get("comparison")).get("points", [])
            ],
        )
    if result.get("comparison"):
        comp = result["comparison"]
        markdown += ["配对比较状态：" + comp["status"] + "。", ""]
        for key in (
            "offload_overhead_s",
            "kv_recovery_s",
            "matched_net_ratio",
            "native_reference_ratio",
            "engine_mode_tax_s",
        ):
            if key in comp:
                markdown += [f"{key}: {comp[key]:.6g}", ""]
        markdown += _admission_markdown(comp)
    coverage, misses = [], []
    for row in result["captures"]:
        if (
            result["status"] != "complete"
            or set(val["reasons"]) - {"instrumented-timing"}
            or row.get("rank") != 0
            or row["generation_status"] != "complete"
            or row["activation_status"] != "complete"
        ):
            continue
        for summary in row["activation_summaries"]:
            if summary["status"] == "complete":
                coverage += [
                    (summary["target_batch"], layer["coverage_mean"])
                    for layer in summary["per_layer"]
                ]
        pool = row["pool"]
        if (
            pool["failed_rows"] == 0
            and pool["cuda_unavailable_rows"] == 0
            and number(pool.get("unique_misses"))
            and number(pool.get("load_s"))
        ):
            misses.append((pool["unique_misses"], pool["load_s"]))
    _svg(
        output / "batch-coverage.svg",
        coverage,
        "actual pure decode batch",
        "instrumented routed-expert coverage / layer; rank 0",
    )
    _svg(
        output / "miss-service.svg",
        misses,
        "unique expert misses in captured window",
        "instrumented local serialized load service seconds; rank 0",
    )
    (output / "report.md").write_text("\n".join(markdown))
    return {
        "status": "exported",
        "files": sorted(x.name for x in output.iterdir()),
        "validation": val,
    }
