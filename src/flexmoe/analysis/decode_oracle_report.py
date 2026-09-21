"""Public, evidence-bounded report for the constrained decode Oracle suite."""

from __future__ import annotations

import csv
import io
import json
import math
from hashlib import sha256
from typing import Any, cast


def _fmt(value: object, digits: int = 3) -> str:
    return f"{value:,.{digits}f}" if type(value) in (int, float) else "—"


def _public_metrics(raw: dict[str, Any], *, timing_eligible: bool) -> dict[str, Any]:
    allowed = (
        "identity",
        "actual_kv",
        "actual_kv_geometry",
        "elapsed_median_s",
        "throughput_samples_tokens_s",
        "throughput_range_tokens_s",
        "decode_batch_mean",
        "decode_batch_steps",
        "decode_batch_distribution",
        "kv_usage_peak",
        "preemptions_total",
        "waiting_requests_peak",
        "phase_status",
        "phase_per_repetition",
        "prefill_wall_time_median_s",
        "decode_wall_time_median_s",
        "request_metrics_per_repetition",
        "kv_admission_blocked",
        "swap_outs",
        "recomputed_tokens",
        "misses_per_rank",
        "payload_bytes_per_rank",
        "oracle_per_repetition",
        "safety_headroom_per_repetition",
    )
    public = {key: raw[key] for key in allowed if key in raw}
    public["throughput_tokens_s"] = raw.get("throughput") if timing_eligible else None
    if not timing_eligible:
        public["throughput_samples_tokens_s"] = None
        public["throughput_range_tokens_s"] = {"min": None, "max": None}
    public["timing_eligible"] = timing_eligible
    if public.get("oracle_per_repetition"):
        public["too_early_prefetch_bytes"] = {
            "status": "unavailable",
            "reason": "the bounded ingress has no eviction/expiry event that distinguishes too-early from merely unused prefetch; unused_prefetch_bytes remains measured separately",
        }
    return public


def _point_map(points: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row["label"]): row for row in points if isinstance(row.get("label"), str)
    }


def _formal_ratio(
    points: dict[str, dict[str, Any]], numerator: str, denominator: str
) -> float | None:
    try:
        top = points[numerator]["metrics"]
        bottom = points[denominator]["metrics"]
        a, b = top["throughput_tokens_s"], bottom["throughput_tokens_s"]
        if (
            top.get("timing_eligible") is not True
            or bottom.get("timing_eligible") is not True
            or type(a) not in (int, float)
            or type(b) not in (int, float)
            or b <= 0
        ):
            return None
        return float(a / b)
    except (KeyError, TypeError):
        return None


def _decision_gates(
    state: dict[str, Any], points: dict[str, dict[str, Any]]
) -> tuple[list[str], dict[str, Any]]:
    zero_ratio = state.get("zero_pair", {}).get("offload_over_resident")
    runtime_first = (
        state.get("zero_miss", {}).get("status") == "proven-zero-miss"
        and type(zero_ratio) in (int, float)
        and zero_ratio < 1
    )
    replay_saves = False
    replay_gates: dict[str, Any] = {}
    for batch in (16, 32):
        replay = state.get(f"replay_{batch}", {})
        current = replay.get("current_policy_misses")
        future = replay.get("future_aware_misses")
        valid = (
            replay.get("status") == "baseline-reproduced-all-ranks"
            and type(current) is int
            and type(future) is int
        )
        replay_gates[str(batch)] = {
            "status": replay.get("status", "unavailable"),
            "future_aware_reduces_misses": valid and future < current,
        }
        replay_saves = replay_saves or (valid and future < current)

    oracle_gates: dict[str, Any] = {}
    prefetch_improves = False
    oracle_complete = True
    for batch in (16, 32):
        output_gate = state.get(f"oracle_output_gate_{batch}", {}).get("status")
        h1 = _formal_ratio(points, f"oracle-{batch}-h1", f"oracle-{batch}-h0")
        h2 = _formal_ratio(points, f"oracle-{batch}-h2", f"oracle-{batch}-h0")
        valid = output_gate == "outputs-match" and h1 is not None and h2 is not None
        oracle_complete = oracle_complete and valid
        prefetch_improves = prefetch_improves or (
            valid and (cast(float, h1) > 1 or cast(float, h2) > 1)
        )
        oracle_gates[str(batch)] = {
            "output_gate": output_gate or "unavailable",
            "h1_over_h0": h1,
            "h2_over_h0": h2,
        }

    service_native = points.get("service-native-k0", {}).get("metrics", {})
    admission = service_native.get("kv_admission_blocked", {})
    recompute = service_native.get("recomputed_tokens", {})
    no_kv_pressure = (
        admission.get("status") == "measured"
        and admission.get("count") == 0
        and service_native.get("preemptions_total") == 0
        and recompute.get("status") == "measured"
        and recompute.get("count") == 0
    )

    decisions: list[str] = []
    if runtime_first:
        decisions.append("runtime-first")
    if no_kv_pressure:
        decisions.append("current-load-does-not-trigger-kv-gain")
    if replay_saves:
        decisions.append("cache-policy-first")
    if prefetch_improves:
        decisions.append("prefetch-predictor-has-direct-value")
    if oracle_complete and not prefetch_improves:
        decisions.append("prediction-accuracy-not-priority-under-current-budget")
    if not decisions:
        decisions.append("inconclusive")
    return decisions, {
        "runtime": {
            "zero_miss": state.get("zero_miss", {}).get("status", "unavailable"),
            "offload_over_resident": zero_ratio,
        },
        "service_native_k0": {
            "kv_admission_blocked": admission,
            "preemptions_total": service_native.get("preemptions_total"),
            "recomputed_tokens": recompute,
            "no_kv_pressure_proven": no_kv_pressure,
        },
        "replay": replay_gates,
        "oracle": oracle_gates,
    }


def build_oracle_report(
    state: dict[str, Any], point_artifacts: dict[str, dict[str, Any]] | None = None
) -> tuple[dict[str, Any], str]:
    points = state.get("points", {})
    numeric = (
        point_artifacts if point_artifacts is not None else state.get("numeric", {})
    )
    public_points = []
    failed = []
    oracle: dict[str, Any] = {}
    for name, row in points.items():
        label = row.get("label", name)
        argv = row.get("argv", [])
        timing_eligible = "--profile" not in argv and row.get("status") == "complete"
        point = {
            "name": name,
            "label": label,
            "mode": row.get("mode"),
            "status": row.get("status"),
            "error_type": row.get("error_type"),
            "metrics": _public_metrics(
                numeric.get(name, {}), timing_eligible=timing_eligible
            ),
        }
        public_points.append(point)
        if row.get("status") in ("failed", "timeout", "skipped", "unreached"):
            failed.append(
                {
                    "label": label,
                    "status": row.get("status"),
                    "reason": row.get("reason"),
                    "error_type": row.get("error_type"),
                }
            )
        if isinstance(label, str) and label.startswith("oracle-"):
            oracle[label] = point["metrics"]
    labeled_points = _point_map(public_points)
    decisions, decision_gates = _decision_gates(state, labeled_points)
    public = {
        "schema_version": 1,
        "artifact_kind": "decode-oracle-decision-report",
        "run_id": state.get("run_id"),
        "commit": state.get("commit"),
        "status": state.get("status"),
        "suite_error_type": state.get("suite_error_type"),
        "report_error_type": state.get("report_error_type"),
        "capacity": state.get("capacity", {}),
        "selected_capacities": state.get("selected_capacities", {}),
        "zero_miss": state.get("zero_miss", {}),
        "mode_decomposition": {
            "zero_pair": state.get("zero_pair", {}),
            "fixed_batch_gates": state.get("fixed_batch_gates", {}),
            "instrumented_timing": {
                "zero_offload": state.get("zero_timing_offload", {}),
                "trace_16": state.get("trace_timing_16", {}),
                "trace_32": state.get("trace_timing_32", {}),
            },
        },
        "service": {
            "kv_intervention": state.get("kv_intervention", {}),
            "low_pair": state.get("low_pair", {}),
        },
        "replay": {
            "batch_16": state.get("replay_16", {}),
            "batch_32": state.get("replay_32", {}),
        },
        "oracle": oracle,
        "oracle_output_gates": {
            "batch_16": state.get("oracle_output_gate_16", {}),
            "batch_32": state.get("oracle_output_gate_32", {}),
        },
        "forced_omission": state.get("forced_omission"),
        "points": public_points,
        "failed_points": failed,
        "decision": decisions[0],
        "decisions": decisions,
        "decision_gates": decision_gates,
        "decision_scope": "ratios and evidence gates are reported without inventing a significance threshold",
    }

    def point_metrics(label: str) -> dict[str, Any]:
        value = labeled_points.get(label, {}).get("metrics")
        return value if isinstance(value, dict) else {}

    def point_status(label: str) -> str:
        return str(labeled_points.get(label, {}).get("status", "unavailable"))

    lines = [
        "# TP4 低并发卸载与受约束 Oracle 报告",
        "",
        f"- Run ID: `{state.get('run_id')}`",
        f"- Commit: `{state.get('commit')}`",
        f"- Suite status: `{state.get('status', 'unavailable')}`",
        "- 正式吞吐只来自 profile-off 的三次 repetition；profile-on 数值仅用于机制诊断。",
        "- TP 时间采用四个 rank 中的最大墙钟或逐 rank 展示，绝不把四个 rank 的时间相加。",
        "- `—`、`unavailable`、失败和跳过都不解释为数值 0。吞吐为包含 prefill 的整批墙钟。",
        "",
        "## 结论门槛",
        "",
        f"- 结论：`{', '.join(decisions)}`",
        f"- 完整门槛：`{json.dumps(decision_gates, ensure_ascii=False, sort_keys=True)}`",
        "- 未使用人为显著性阈值；报告保留比值、范围和证据门槛供判断。",
        "",
        "## 无采样零 miss 固定开销",
        "",
        f"- 零 miss 门槛：`{state.get('zero_miss', {}).get('status', 'unavailable')}`",
        f"- 同 KV eager offload/resident 正式吞吐比：{_fmt(state.get('zero_pair', {}).get('offload_over_resident'))}",
        "- host preparation 边界：`wait_host()` 返回后开始；包含条件性权重 gather 和无条件 host map 构建；在 H2D enqueue 前结束。instrumented 结果分别记录 `weight_gather_cpu_s`、`host_map_cpu_s`，兼容字段 `host_gather_s` 是两者之和。",
        "",
        "| repetition | rank | misses | payload bytes | copy launches |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in state.get("zero_miss", {}).get("per_rank_repetition", []):
        lines.append(
            f"| {row['repetition']} | {row['rank']} | {row['misses']} | {row['payload_bytes']} | {row['copy_launches']} |"
        )
    lines += [
        "",
        "### 零 miss 配对的 phase 与请求延迟",
        "",
        "| 点 | 状态 | KV/GPU bytes | output tokens/s | range | prefill wall s | decode wall s | actual decode batch mean |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for label in ("zero-resident", "zero-offload"):
        metrics = point_metrics(label)
        interval = metrics.get("throughput_range_tokens_s", {})
        lines.append(
            f"| {label} | {point_status(label)} | {metrics.get('actual_kv', '—')} | {_fmt(metrics.get('throughput_tokens_s'), 2)} | {_fmt(interval.get('min'), 2)}–{_fmt(interval.get('max'), 2)} | {_fmt(metrics.get('prefill_wall_time_median_s'))} | {_fmt(metrics.get('decode_wall_time_median_s'))} | {_fmt(metrics.get('decode_batch_mean'))} |"
        )
    lines += [
        "",
        "### 零 miss instrumented 逐 rank 时间（不作为正式吞吐）",
        "",
        "| rep | rank | CUDA step p50/p95 ms | host reuse wait sum s | weight gather sum s | host map sum s | host gather兼容和 s | payload bytes |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in state.get("zero_timing_offload", {}).get("per_rank_repetition", []):
        lines.append(
            f"| {row.get('repetition', '—')} | {row.get('rank', '—')} | {_fmt(row.get('model_step_cuda_p50_ms'))}/{_fmt(row.get('model_step_cuda_p95_ms'))} | {_fmt(row.get('host_reuse_wait_cpu_sum_s'))} | {_fmt(row.get('weight_gather_cpu_sum_s'))} | {_fmt(row.get('host_map_cpu_sum_s'))} | {_fmt(row.get('host_gather_cpu_sum_s'))} | {row.get('payload_bytes', '—')} |"
        )
    lines += [
        "",
        "## 执行模式分解",
        "",
        "同一历史 KV 点用于 Native → eager resident → eager offload 分解；机制点只有固定 actual batch 门槛通过时才支持固定-batch 解释。",
        "",
        "| 点 | 状态 | KV/GPU bytes | output tokens/s | samples | prefill wall s | decode wall s |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for label in (
        "legacy-native",
        "legacy-eager-resident",
        "legacy-offload",
        "mechanism-native-32",
        "mechanism-eager-32",
        "mechanism-offload-32",
    ):
        metrics = point_metrics(label)
        lines.append(
            f"| {label} | {point_status(label)} | {metrics.get('actual_kv', '—')} | {_fmt(metrics.get('throughput_tokens_s'), 2)} | {metrics.get('throughput_samples_tokens_s', '—')} | {_fmt(metrics.get('prefill_wall_time_median_s'))} | {_fmt(metrics.get('decode_wall_time_median_s'))} |"
        )
    lines += [
        "",
        f"- 固定 actual batch 门槛：`{json.dumps(state.get('fixed_batch_gates', {}), ensure_ascii=False, sort_keys=True)}`",
        "- 若 Native 的 K0 与 eager 的 K_pair 不同，Native 对比带容量混杂；严格卸载管理税只看 eager resident 与 eager offload 的 K_pair 配对。",
        "",
        "## K0 与 K1 稳定容量",
        "",
        f"- 容量结论：`{state.get('capacity', {}).get('status', 'unavailable')}`",
        f"- 完整容量证据：`{json.dumps(state.get('capacity', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- 选择：`{json.dumps(state.get('selected_capacities', {}), ensure_ascii=False, sort_keys=True)}`",
        "- K1 只有严格超过 Native 与 eager-resident 两种上限至少一个实际 KV block，才视为卸载独有容量。",
        "",
        "## 最大并发 32 服务收益",
        "",
        f"- K1/K0 同卸载路径干预：`{state.get('kv_intervention', {}).get('status', 'unavailable')}`；大/小 KV 吞吐比={_fmt(state.get('kv_intervention', {}).get('large_over_small_throughput'))}",
        f"- 同 K_pair eager offload/resident：`{state.get('low_pair', {}).get('status', 'unavailable')}`；吞吐比={_fmt(state.get('low_pair', {}).get('offload_over_resident'))}",
        "",
        "| 点 | 状态 | KV/GPU bytes | blocks × bytes/block | output tokens/s | batch mean | batch histogram | KV peak | preemptions | waiting peak | TTFT median s | latency median s |",
        "|---|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for label in (
        "service-native-k0",
        "service-offload-k0",
        "service-offload-k1",
        "service-eager-kpair",
        "service-offload-kpair",
    ):
        metrics = point_metrics(label)
        geometry = metrics.get("actual_kv_geometry", {})
        lines.append(
            f"| {label} | {point_status(label)} | {metrics.get('actual_kv', '—')} | {geometry.get('num_gpu_blocks', '—')} × {geometry.get('bytes_per_block', '—')} | {_fmt(metrics.get('throughput_tokens_s'), 2)} | {_fmt(metrics.get('decode_batch_mean'))} | {metrics.get('decode_batch_distribution', '—')} | {_fmt(metrics.get('kv_usage_peak'))} | {_fmt(metrics.get('preemptions_total'), 0)} | {_fmt(metrics.get('waiting_requests_peak'), 0)} | {_fmt(metrics.get('ttft_median_s'))} | {_fmt(metrics.get('request_latency_median_s'))} |"
        )
    lines += [
        "",
        "### 服务指标可用性",
        "",
    ]
    for label in (
        "service-native-k0",
        "service-offload-k0",
        "service-offload-k1",
        "service-eager-kpair",
        "service-offload-kpair",
    ):
        metrics = point_metrics(label)
        lines.append(
            f"- `{label}`: KV admission={json.dumps(metrics.get('kv_admission_blocked', {'status': 'unavailable'}), ensure_ascii=False, sort_keys=True)}; swap={json.dumps(metrics.get('swap_outs', {'status': 'unavailable'}), ensure_ascii=False, sort_keys=True)}; recompute={json.dumps(metrics.get('recomputed_tokens', {'status': 'unavailable'}), ensure_ascii=False, sort_keys=True)}"
        )
        lines.append(
            f"  - physical safety headroom: {json.dumps(metrics.get('safety_headroom_per_repetition', []), ensure_ascii=False, sort_keys=True)}"
        )
        for repetition, request in enumerate(
            metrics.get("request_metrics_per_repetition", [])
        ):
            lines.append(
                f"  - repetition {repetition}: TTFT p50/p95/p99={json.dumps(request.get('ttft_s', {}), ensure_ascii=False, sort_keys=True)}; request latency p50/p95/p99={json.dumps(request.get('request_latency_s', {}), ensure_ascii=False, sort_keys=True)}; derived TPOT p50/p95/p99={json.dumps(request.get('derived_tpot_s', {}), ensure_ascii=False, sort_keys=True)}; ITL={json.dumps(request.get('itl_s', {'status': 'unavailable'}), ensure_ascii=False, sort_keys=True)}"
            )
    lines += [
        "",
        "## 同预算缓存重放",
        "",
        "重放只改变 eviction/admission，不提前加载、不增加 cache/ingress/KV，也不支持吞吐结论。只有逐行重现生产策略后才展示 LRU 与 future-aware。",
        "",
        "| batch | 状态 | current miss | LRU miss | future-aware miss | future-aware saved bytes |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for batch in (16, 32):
        replay = state.get(f"replay_{batch}", {})
        lines.append(
            f"| {batch} | {replay.get('status', 'unavailable')} | {replay.get('current_policy_misses', '—')} | {replay.get('lru_misses', '—')} | {replay.get('future_aware_misses', '—')} | {replay.get('future_aware_saved_bytes', '—')} |"
        )
        lines.append("")
        lines.append(f"### batch {batch} 的逐 rank/repetition replay")
        lines.append("")
        for row in replay.get("per_rank_repetition", []):
            lines.append(
                f"- repetition={row.get('repetition')} rank={row.get('rank')} status={row.get('status')}; current={json.dumps(row.get('current', {}), ensure_ascii=False, sort_keys=True)}; LRU={json.dumps(row.get('lru', {}), ensure_ascii=False, sort_keys=True)}; future-aware={json.dumps(row.get('future_aware', {}), ensure_ascii=False, sort_keys=True)}"
            )
    lines += [
        "",
        "## 真实 Oracle horizon 0/1/2",
        "",
        "正式点完整执行 host gather、transfer queue、真实 H2D、映射/物化、ready wait、promotion 和原专家计算；profile 点只提供局部机制计时。",
        "",
        "| 点 | 状态 | timing eligible | output tokens/s | range | prefill wall s | decode wall s |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for label, metrics in sorted(oracle.items()):
        interval = metrics.get("throughput_range_tokens_s", {})
        lines.append(
            f"| {label} | {point_status(label)} | {str(metrics['timing_eligible']).lower()} | {_fmt(metrics.get('throughput_tokens_s'), 2)} | {_fmt(interval.get('min'), 2)}–{_fmt(interval.get('max'), 2)} | {_fmt(metrics.get('prefill_wall_time_median_s'))} | {_fmt(metrics.get('decode_wall_time_median_s'))} |"
        )
    lines += [
        "",
        "### Oracle 逐 repetition/rank 物理证据",
        "",
        "| 点 | rep | rank | mandatory bytes | prefetched bytes | unused bytes | lookahead ms | ready-before-use | all-ready layer batch | H2D p50/p95/p99 ms | host-buffer wait p50/p95/p99 ms | exposed wait p50/p95/p99 ms | fallback experts |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|---:|",
    ]
    for label, metrics in sorted(oracle.items()):
        for repetition, oracle_rep in enumerate(
            metrics.get("oracle_per_repetition", [])
        ):
            for row in oracle_rep.get("per_rank", []):
                waits = "/".join(
                    _fmt(row.get(key))
                    for key in (
                        "exposed_wait_ms_p50",
                        "exposed_wait_ms_p95",
                        "exposed_wait_ms_p99",
                    )
                )
                h2d = "/".join(
                    _fmt(row.get(key))
                    for key in (
                        "transfer_h2d_ms_p50",
                        "transfer_h2d_ms_p95",
                        "transfer_h2d_ms_p99",
                    )
                )
                host_wait = "/".join(
                    _fmt(row.get(key))
                    for key in (
                        "host_buffer_reuse_wait_ms_p50",
                        "host_buffer_reuse_wait_ms_p95",
                        "host_buffer_reuse_wait_ms_p99",
                    )
                )
                lines.append(
                    f"| {label} | {repetition} | {row.get('rank', '—')} | {row.get('mandatory_loaded_bytes', '—')} | {row.get('prefetch_loaded_bytes', '—')} | {row.get('unused_prefetch_bytes', '—')} | {row.get('lookahead_ms', '—')} | {_fmt(row.get('ready_before_use_ratio'))} | {_fmt(row.get('layer_batch_all_ready_ratio'))} | {h2d} | {host_wait} | {waits} | {row.get('fallback_on_demand_count', '—')} |"
                )
    lines.append("")
    lines.append(
        "- too-early prefetch bytes：`unavailable`；当前 bounded ingress 不驱逐/过期预取项，无法把“过早”与“最终未使用”拆成两个测量量。`unused_prefetch_bytes` 单独保留为实测。"
    )
    lines.append(
        "- transfer queue wait：`unavailable`；当前只记录 H2D event duration、host-buffer reuse wait 与 exposed compute wait，不把它们相加成全局净损失。"
    )
    lines += [
        "",
        "### Oracle 请求级延迟",
        "",
    ]
    for label, metrics in sorted(oracle.items()):
        if not metrics.get("timing_eligible"):
            continue
        for repetition, request in enumerate(
            metrics.get("request_metrics_per_repetition", [])
        ):
            lines.append(
                f"- `{label}` repetition {repetition}: TTFT p50/p95/p99={json.dumps(request.get('ttft_s', {}), ensure_ascii=False, sort_keys=True)}; request latency p50/p95/p99={json.dumps(request.get('request_latency_s', {}), ensure_ascii=False, sort_keys=True)}; derived TPOT p50/p95/p99={json.dumps(request.get('derived_tpot_s', {}), ensure_ascii=False, sort_keys=True)}; ITL={json.dumps(request.get('itl_s', {'status': 'unavailable'}), ensure_ascii=False, sort_keys=True)}"
            )
    lines += [
        "",
        "## 强制漏预测正确性",
        "",
        f"- 强制遗漏：`{state.get('forced_omission', 'unavailable')}`",
        f"- batch 16 输出门槛：`{state.get('oracle_output_gate_16', {}).get('status', 'unavailable')}`",
        f"- batch 32（含 forced omission）输出门槛：`{state.get('oracle_output_gate_32', {}).get('status', 'unavailable')}`",
        f"- 输出哈希证据：`{json.dumps({'batch_16': state.get('oracle_output_gate_16', {}), 'batch_32': state.get('oracle_output_gate_32', {})}, ensure_ascii=False, sort_keys=True)}`",
        "- 只有 forced omission 计数与 on-demand fallback 计数均为 1，且三次 greedy 输出与 horizon 0 完全一致，才通过正确性门槛。",
        "",
        "## 失败、超时与不可用指标",
        "",
        f"- Suite status=`{state.get('status', 'unavailable')}`; suite error={state.get('suite_error_type', '—')}; report error={state.get('report_error_type', '—')}",
    ]
    if failed:
        for row in failed:
            lines.append(
                f"- `{row['label']}`: {row['status']}; reason={row.get('reason')}; error={row.get('error_type')}"
            )
    else:
        lines.append("- 无失败或跳过点。")
    lines += [
        "",
        "### 所有点",
        "",
        "| 点 | 状态 | 模式 | KV bytes/GPU | output tokens/s | timing eligible |",
        "|---|---|---|---:|---:|---:|",
    ]
    for point in public_points:
        metrics = point["metrics"]
        lines.append(
            f"| {point['label']} | {point['status']} | {point['mode']} | {metrics.get('actual_kv', '—')} | {_fmt(metrics.get('throughput_tokens_s'), 2)} | {str(metrics.get('timing_eligible', False)).lower()} |"
        )
    lines += [
        "",
        "## 完整聚合数值证据",
        "",
        "以下 JSON 与同目录 `report.json` 内容一致；它直接嵌入报告，便于离线查看，不含 prompt/output token IDs、完整专家 route、主机名、模型路径或私有日志。",
        "",
        "```json",
        json.dumps(public, indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
    ]
    return public, "\n".join(lines)


def repair_summary_conflict(summary: dict[str, Any], raw_csv: str) -> dict[str, Any]:
    source_hash = sha256(raw_csv.encode()).hexdigest()
    rows = list(csv.DictReader(io.StringIO(raw_csv)))
    identity = [
        row.get("value")
        for row in rows
        if row.get("kind") == "identity" and row.get("metric") == "contract_sha256"
    ]
    if identity != [summary.get("contract_sha256")]:
        return {**summary, "repair_status": "identity-mismatch"}
    elapsed_rows = [
        row
        for row in rows
        if row.get("kind") == "repetition" and row.get("metric") == "elapsed_s"
    ]
    measured: dict[int, float] = {}
    for row in elapsed_rows:
        try:
            repetition = int(row.get("repetition", ""))
            value = float(row.get("value", ""))
        except (TypeError, ValueError):
            return {**summary, "repair_status": "invalid-repetition-evidence"}
        if (
            row.get("status") != "complete"
            or repetition not in (0, 1, 2)
            or repetition in measured
            or not math.isfinite(value)
            or value <= 0
        ):
            return {**summary, "repair_status": "invalid-repetition-evidence"}
        measured[repetition] = value
    if set(measured) != {0, 1, 2} or len(elapsed_rows) != 3:
        return {**summary, "repair_status": "three-repetition-evidence-unavailable"}
    return {
        **summary,
        "original_status": summary.get("status"),
        "status": "measured-repaired",
        "repair_status": "complete",
        "repair_source_sha256": source_hash,
        "repair_scope": "derived export status only; immutable source CSV retained",
    }


__all__ = ["build_oracle_report", "repair_summary_conflict"]
