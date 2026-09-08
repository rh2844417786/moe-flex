"""Small public, typed allowlists; raw traces, paths and commands remain local."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io import atomic_json, native_timing, oracle_adapter, read_json
from .schema import diagnostic_artifact, parse_diagnostic_artifact

# Strings are admitted by literal value, never by length or character class.
_ENUMS = frozenset(
    [
        "complete",
        "truncated",
        "incomplete",
        "failed",
        "running",
        "invalid",
        "timeout",
        "unreadable",
        "missing",
        "candidate",
        "model-no-headroom",
        "serial-profitable",
        "overlap-required",
        "neither-profitable",
        "native",
        "eager",
        "trace",
        "resident",
        "native-measured",
        "kv-oracle-diagnostic",
        "measured",
        "instrumented-not-throughput",
        "contiguous",
        "fragmented",
        "gather",
        "isolated",
        "gemm-nccl-proxy",
        "prefill",
        "decode",
        "mixed",
        "unknown",
        "lru",
        "decayed-lfu",
        "future",
        "native-summary",
        "analysis-smoke",
        "analysis-repetition",
        "demand-trace",
        "transfer-samples",
        "transfer-worker",
        "replay-suite",
        "analysis-suite",
        "feasibility-analysis",
        "launcher-failure",
        "capture-failure",
        "oracle-adaptation",
        "offload-analysis-plan",
        "offload-analysis-report",
        "over-budget-diagnostic-counterfactual",
        "matching-kv-reference",
        "incremental-transfer-calibration",
        "hardware-identity-unavailable",
        "full-workload-four-rank-trace",
        "staging-overflow-compute-cost",
        "positive-actual-kv-increment",
        "kv-increment-exceeds-net-freed",
        "four-rank-transfer-samples",
        "unmeasured-exact-transfer-shape",
        "native-measured-baseline",
        "native-engine-baseline",
        "no-complete-step-layer-grid",
        "legacy-hardware-sha256",
        "legacy-prompt-hashes",
        "matching-native-reference-required",
        "diagnostic-model-not-deployment-measurement",
        "serial-layer-barrier-is-a-modelling-scenario-not-a-hardware-bound",
        "full-overlap-is-optimistic-and-not-a-hardware-bound",
        "no-rank-bandwidth-aggregation",
        "synthetic-workload",
        "kv-reference-engine-mode-differs",
        "kv-reference-engine-policy-differs",
        "kv-reference-utilization-differs",
        "kv-reference-is-diagnostic-counterfactual",
        "trace-engine-policy-differs",
        "trace-utilization-differs",
        "median-measured-wall-s-per-exact-chunk",
        "largest-first-exact-decomposition-with-backtracking",
        "NUMA/PCIe-unavailable",
        "coordinator",
        "launcher",
        "export",
    ]
)
_NUMBERS = frozenset(
    [
        "rank",
        "device_rank",
        "repetition",
        "exit_code",
        "generated_tokens",
        "request_count",
        "elapsed_s",
        "output_tokens_per_second",
        "throughput_median",
        "throughput_min",
        "throughput_max",
        "median_elapsed_s",
        "throughput_tps",
        "gpu_memory_utilization",
        "repetitions_completed",
        "batch_size",
        "context_length",
        "output_length",
        "max_num_seqs",
        "max_num_batched_tokens",
        "seed",
        "unique_selected_request_count",
        "repeated_request_count",
        "source_request_count",
        "evaluation_pool_count",
        "selection_offset",
        "total_gpu_bytes",
        "free_gpu_bytes",
        "torch_allocated_bytes",
        "torch_reserved_bytes",
        "torch_peak_allocated_bytes",
        "torch_peak_reserved_bytes",
        "kv_cache_allocated_bytes",
        "kv_cache_declared_bytes",
        "num_gpu_blocks",
        "physical_safety_reserve_bytes",
        "observed_steps",
        "captured_steps",
        "missing_layer_events",
        "discarded_present_events",
        "buffer_bytes",
        "max_trace_steps",
        "trace_budget_bytes",
        "total_layers",
        "num_experts",
        "top_k",
        "expert_bytes",
        "loaded_bytes",
        "loaded_experts",
        "resident_hits",
        "cache_hits",
        "demands",
        "bytes_per_generated_token",
        "net_freed_bytes",
        "cache_slots",
        "staging_experts",
        "extra_gpu_bytes",
        "staging_overflow_events",
        "experts_per_batch",
        "payload_bytes",
        "wall_s",
        "copy_s",
        "gather_s",
        "compute_s",
        "copy_only_wall_s",
        "compute_only_wall_s",
        "joint_wall_s",
        "bottleneck_rank",
        "bottleneck_wall_s",
        "bottleneck_bytes_per_s",
        "copy_count",
        "total_service_s",
        "effective_bandwidth_bytes_s",
        "optimistic_resource_service_s",
        "serial_layer_barrier_service_s",
        "time_headroom_s",
        "required_overlap_to_match_baseline_s",
    ]
)
_ARRAY_NUMBERS = frozenset(
    [
        "actual_kv_increment_bytes_by_rank",
        "net_freed_bytes_by_rank",
        "bandwidth_requirements_bytes_s",
        "bandwidth_to_fit_headroom_bytes_s_by_rank",
        "throughput_range_tps",
        "kv_bytes_by_rank",
        "gpu_total_bytes_by_rank",
        "per_rank_payload_bytes",
        "elapsed_s",
    ]
)
_BOOLS = frozenset(
    [
        "timing_eligible",
        "full_workload",
        "truncated",
        "coverage_complete",
        "evidence_complete",
        "synthetic",
        "future_policy_reference",
        "kv_cache_accounting_consistent",
    ]
)
_STRINGS = frozenset(
    [
        "status",
        "generation_status",
        "capture_status",
        "artifact_kind",
        "engine_mode",
        "source_kind",
        "measurement_evidence",
        "mode",
        "contention",
        "policy",
        "scope",
        "candidate_class",
        "phase",
        "topology_status",
        "service_metric",
        "chunk_rule",
    ]
)
_OBJECTS = frozenset(
    [
        "contract",
        "geometry",
        "memory",
        "final_memory",
        "failed_measurement",
        "smoke",
        "repetitions",
        "measurements",
        "samples",
        "partial_worker_artifacts",
        "worker_metadata",
        "capture",
        "config",
        "replays",
        "analyses",
        "baseline",
        "kv_reference",
        "transport",
        "per_rank",
        "predictions",
        "serial_layer_barrier",
        "optimistic_full_overlap",
        "copy_only_components",
        "compute_only_components",
        "trace",
        "legacy_reference",
        "baseline_evidence",
        "reference_evidence",
    ]
)
_KINDS = frozenset(
    {
        "native-summary",
        "analysis-smoke",
        "analysis-repetition",
        "demand-trace",
        "transfer-samples",
        "transfer-worker",
        "replay-suite",
        "analysis-suite",
        "feasibility-analysis",
        "launcher-failure",
        "capture-failure",
        "oracle-adaptation",
        "offload-analysis-plan",
        "offload-analysis-report",
    }
)


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def public_fields(raw: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in raw.items():
        if key in {
            "model_identity_sha256",
            "model_config_sha256",
            "input_sha256",
            "dataset_sha256",
            "dataset_manifest_sha256",
            "engine_policy_sha256",
            "hardware_sha256",
            "benchmark_policy_sha256",
            "commit",
        } and isinstance(value, str):
            if re.fullmatch(
                r"[a-f0-9]{40}" if key == "commit" else r"[a-f0-9]{64}", value
            ):
                result[key] = value
        elif key == "versions" and isinstance(value, Mapping):
            result[key] = {
                name: v
                for name, v in value.items()
                if name in {"torch", "vllm", "cuda", "python", "vllm_commit"}
                and isinstance(v, str)
                and re.fullmatch(
                    r"[a-f0-9]{40}"
                    if name == "vllm_commit"
                    else r"[0-9]+(?:\.[0-9]+){1,3}(?:\+(?:cpu|cu[0-9]+|[a-f0-9]{7,40}))?",
                    v,
                )
            }
        elif (
            key in _NUMBERS
            and (_number(value) or value is None)
            or key in _BOOLS
            and type(value) is bool
            or key in _STRINGS
            and (value is None or isinstance(value, str) and value in _ENUMS)
        ):
            result[key] = value
        elif key in ("missing_evidence", "assumptions") and isinstance(value, list):
            result[key] = [v for v in value if isinstance(v, str) and v in _ENUMS]
            result["unpublished_reason_count"] = sum(
                not isinstance(v, str) or v not in _ENUMS for v in value
            )
        elif key in _ARRAY_NUMBERS and isinstance(value, (tuple, list)):
            result[key] = [v for v in value if _number(v) or v is None]
        elif key in _OBJECTS:
            if isinstance(value, Mapping):
                result[key] = public_fields(value)
            elif isinstance(value, list):
                result[key] = [
                    public_fields(v) for v in value if isinstance(v, Mapping)
                ]
            elif value is None:
                result[key] = None
    return result


def public_artifact(raw: Mapping[str, Any]) -> dict[str, Any]:
    kind = raw.get("artifact_kind")
    if raw.get("comparison_backend") == "kv-oracle":
        return public_fields(oracle_adapter(raw))
    if not isinstance(kind, str) or kind not in _KINDS:
        raise ValueError("unsupported artifact kind")
    fields = parse_diagnostic_artifact(raw, kind)
    if (
        kind == "native-summary"
        and fields.get("status") == "complete"
        and fields.get("timing_eligible") is True
    ):
        native_timing(raw)
    result = public_fields(fields)
    result["artifact_kind"] = kind
    if kind == "native-summary" and fields.get("engine_mode") == "trace":
        result["generation_status"] = result.get("status", "unknown")
        if (
            fields.get("status") == "complete"
            and fields.get("capture_status") != "complete"
        ):
            result["status"] = "incomplete"
    if kind == "demand-trace":
        capture = fields.get("capture")
        trace = fields.get("trace")
        if fields.get("generation_status") == "failed":
            result["status"] = "failed"
        elif (
            isinstance(capture, dict)
            and capture.get("full_workload") is True
            and isinstance(trace, dict)
            and trace.get("full_workload") is True
            and fields.get("generation_status") == "complete"
        ):
            result["status"] = "complete"
        else:
            result["status"] = "incomplete"
    if isinstance(fields.get("run_id"), str):
        result["run_identity_sha256"] = hashlib.sha256(
            str(fields["run_id"]).encode()
        ).hexdigest()
    return result


def metric_rows(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten only the already sanitized public tree, never the input artifact."""
    if isinstance(value, dict):
        return [
            row
            for key, item in value.items()
            for row in metric_rows(item, f"{prefix}.{key}" if prefix else key)
        ]
    if isinstance(value, list):
        return [
            row
            for index, item in enumerate(value)
            for row in metric_rows(item, f"{prefix}[{index}]")
        ]
    return [(prefix, value)]


def export_report(source: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("public output must be fresh")
    if source.is_file():
        paths = [source]
    else:
        # No logs or arbitrary file content. Independent smoke always survives.
        patterns = (
            "summary.json",
            "smoke.json",
            "rep-*.json",
            "failed-rep-*.json",
            "launcher.json",
            "samples.json",
            "worker-*.json",
            "analysis.json",
            "replay.json",
            "trace-rank-*.json.gz",
            "capture-failure.json",
        )
        paths = sorted(
            {
                p
                for pattern in patterns
                for p in source.rglob(pattern)
                if output not in p.parents
            }
        )
    if not paths:
        raise ValueError("no evidence artifacts found")
    records = []
    for index, path in enumerate(paths):
        try:
            item = public_artifact(read_json(path))
        except (OSError, ValueError, TypeError, KeyError, AssertionError):
            item = {"status": "unreadable", "artifact_kind": "unknown"}
        records.append({"record_index": index, **item})
    result = diagnostic_artifact("offload-analysis-report", {"records": records})
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["record_index", "metric", "value"])
    for row in records:
        for key, value in metric_rows(row):
            writer.writerow([row["record_index"], key, value])
    markdown = (
        "# 专家卸载可行性诊断\n\n所有结果仅用于诊断；未证明真实卸载或部署吞吐收益。\n\n"
    )
    markdown += "单位：bytes；GB = bytes / 1,000,000,000（十进制）。\n\n"
    markdown += "|记录|类型|状态|\n|---|---|---|\n"
    markdown += "".join(
        f"|{r['record_index']}|{r.get('artifact_kind', 'unknown')}|{r.get('status', 'unknown')}|\n"
        for r in records
    )
    markdown += (
        "\n完整数值见 report.json 与逐字段 CSV。原始轨迹、日志、路径、提示词、UUID、命令不公开。\n"
        "\n原生基线为实测，trace 为采样观测，回放成本为预测。代理 joint wall 含 GEMM/NCCL，禁止重复加入 K 时间。"
        "未配对 K 或完整轨迹时保留字节/带宽观测，不解释为无收益。\n"
    )
    for row in records:
        markdown += f"\n## 记录 {row['record_index']}\n\n"
        markdown += "|指标|值|\n|---|---|\n"
        # Per-repetition transport observations remain in JSON/CSV; the human
        # view keeps aggregate native, coverage, replay and feasibility fields.
        view = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "samples",
                "measurements",
                "partial_worker_artifacts",
                "repetitions",
            }
        }
        for key, value in metric_rows(view):
            markdown += f"|{key}|{value}|\n"
    markdown += (
        "\n连续拷贝假设权重已连续/预打包；未计任意路由子集的额外打包或 overfetch。"
        "fragmented 计每次 copy launch；gather 计当前 CPU 准备，受 CPU/affinity/拓扑限制。"
        "现有数据仅 1K–4K；重复与 synthetic 长上下文仅为压力样本。\n"
    )
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "report.json", result)
    (output / "report.csv").write_text(buffer.getvalue(), encoding="utf-8")
    (output / "report.md").write_text(markdown, encoding="utf-8")
    return result
