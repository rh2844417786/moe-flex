"""Public, evidence-bounded report for the constrained decode Oracle suite."""

from __future__ import annotations

import csv
import io
import json
from hashlib import sha256
from typing import Any


def _fmt(value: object, digits: int = 3) -> str:
    return f"{value:,.{digits}f}" if type(value) in (int, float) else "—"


def _public_metrics(raw: dict[str, Any], *, timing_eligible: bool) -> dict[str, Any]:
    allowed = (
        "actual_kv",
        "elapsed_median_s",
        "decode_batch_mean",
        "decode_batch_steps",
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
    )
    public = {key: raw[key] for key in allowed if key in raw}
    public["throughput_tokens_s"] = raw.get("throughput") if timing_eligible else None
    public["timing_eligible"] = timing_eligible
    return public


def build_oracle_report(state: dict[str, Any]) -> tuple[dict[str, Any], str]:
    points = state.get("points", {})
    numeric = state.get("numeric", {})
    public_points = []
    failed = []
    oracle: dict[str, Any] = {}
    zero_ratio = state.get("zero_pair", {}).get("offload_over_resident")
    decision = (
        "runtime-first"
        if state.get("zero_miss", {}).get("status") == "proven-zero-miss"
        and type(zero_ratio) in (int, float)
        and zero_ratio < 1
        else "inconclusive"
    )
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
        if row.get("status") in ("failed", "timeout", "skipped"):
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
    public = {
        "schema_version": 1,
        "artifact_kind": "decode-oracle-decision-report",
        "run_id": state.get("run_id"),
        "commit": state.get("commit"),
        "status": state.get("status"),
        "capacity": state.get("capacity", {}),
        "selected_capacities": state.get("selected_capacities", {}),
        "zero_miss": state.get("zero_miss", {}),
        "mode_decomposition": {
            "zero_pair": state.get("zero_pair", {}),
            "fixed_batch_gates": state.get("fixed_batch_gates", {}),
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
        "decision": decision,
        "decision_scope": "ratios and evidence gates are reported without inventing a significance threshold",
    }
    lines = [
        "# TP4 低并发卸载与受约束 Oracle 报告",
        "",
        f"- Run ID: `{state.get('run_id')}`",
        f"- Commit: `{state.get('commit')}`",
        "- 正式吞吐只来自 profile-off repetition；局部 rank/CPU/GPU 时间不相加。",
        "",
        "## 无采样零 miss 固定开销",
        "",
        f"- 零 miss 门槛：`{state.get('zero_miss', {}).get('status', 'unavailable')}`",
        f"- 同 KV eager offload/resident：{_fmt(state.get('zero_pair', {}).get('offload_over_resident'))}",
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
        "## 执行模式分解",
        "",
        f"- 固定 batch 门槛：`{json.dumps(state.get('fixed_batch_gates', {}), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## K0 与 K1 稳定容量",
        "",
        f"- 容量结论：`{state.get('capacity', {}).get('status', 'unavailable')}`",
        f"- 选择：`{json.dumps(state.get('selected_capacities', {}), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## 最大并发 32 服务收益",
        "",
        f"- K1/K0 干预：`{state.get('kv_intervention', {}).get('status', 'unavailable')}`",
        f"- 吞吐比：{_fmt(state.get('kv_intervention', {}).get('large_over_small_throughput'))}",
        "",
        "## 同预算缓存重放",
        "",
    ]
    for batch in (16, 32):
        replay = state.get(f"replay_{batch}", {})
        lines.append(
            f"- batch {batch}: `{replay.get('status', 'unavailable')}`, current={replay.get('current_policy_misses')}, LRU={replay.get('lru_misses')}, future-aware={replay.get('future_aware_misses')}"
        )
    lines += [
        "",
        "## 真实 Oracle horizon 0/1/2",
        "",
        "| 点 | 状态 | timing eligible | output tokens/s |",
        "|---|---|---:|---:|",
    ]
    for label, metrics in sorted(oracle.items()):
        point_status = next(
            (row["status"] for row in public_points if row["label"] == label),
            "unavailable",
        )
        lines.append(
            f"| {label} | {point_status} | {str(metrics['timing_eligible']).lower()} | {_fmt(metrics.get('throughput_tokens_s'), 2)} |"
        )
    lines += [
        "",
        "## 强制漏预测正确性",
        "",
        f"- 强制遗漏：`{state.get('forced_omission', 'unavailable')}`",
        f"- batch 32 输出门槛：`{state.get('oracle_output_gate_32', {}).get('status', 'unavailable')}`",
        "",
        "## 失败、超时与不可用指标",
        "",
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
        "| 点 | 状态 | 模式 | KV bytes/GPU | output tokens/s |",
        "|---|---|---|---:|---:|",
    ]
    for point in public_points:
        metrics = point["metrics"]
        lines.append(
            f"| {point['label']} | {point['status']} | {point['mode']} | {metrics.get('actual_kv', '—')} | {_fmt(metrics.get('throughput_tokens_s'), 2)} |"
        )
    return public, "\n".join(lines) + "\n"


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
    measured = {
        int(row["repetition"])
        for row in rows
        if row.get("kind") == "repetition"
        and row.get("metric") == "elapsed_s"
        and row.get("status") == "complete"
        and row.get("repetition", "").isdigit()
    }
    if measured != {0, 1, 2}:
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
