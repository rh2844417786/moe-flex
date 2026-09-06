"""Stdlib host CLI: screen, forward/reverse confirm, and diagnostic-only export."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4


def _sibling(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        "_oracle_host_" + name, Path(__file__).with_name(name + ".py")
    )
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


shared = _sibling("partial_suite")
evidence = _sibling("kv_oracle_evidence")
PRESETS = {
    "control": (32, 4096, 64),
    "main": (512, 4096, 256),
    "long": (512, 4096, 1024),
}
OPTIONS = (
    "model_path",
    "dataset_path",
    "dataset_manifest",
    "batch_size",
    "context_length",
    "output_length",
    "warmups",
    "repetitions",
    "max_num_seqs",
    "max_num_batched_tokens",
    "seed",
    "small_add_bytes",
    "large_add_bytes",
    "safety_fraction",
    "safety_margin_bytes",
)


def point_command(
    root: Path,
    role: str,
    run_dir: Path,
    reference: Path | None,
    options: Mapping[str, Any],
) -> list[str]:
    command = [
        "bash",
        str(root / "scripts/server/run_kv_oracle.sh"),
        "point",
        "--timeout-s",
        str(options["timeout_s"]),
        "--role",
        role,
        "--run-dir",
        str(run_dir),
    ]
    for name in OPTIONS:
        command.extend(["--" + name.replace("_", "-"), str(options[name])])
    if reference is not None:
        command.extend(["--baseline-reference", str(reference)])
    return command


def execute_point(
    root: Path,
    role: str,
    run_dir: Path,
    reference: Path | None,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    if run_dir.exists():
        raise FileExistsError(run_dir)
    logs = run_dir.parent / "logs"
    logs.mkdir(exist_ok=True)
    started = perf_counter()
    returncode = 1
    try:
        with (
            (logs / (role + ".stdout.log")).open("w") as stdout,
            (logs / (role + ".stderr.log")).open("w") as stderr,
        ):
            process = subprocess.run(
                point_command(root, role, run_dir, reference, options),
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
            returncode = process.returncode
    except (OSError, KeyboardInterrupt):
        returncode = 130
    path = run_dir / "summary.json"
    row = (
        shared._read(path)
        if path.is_file()
        else {
            **evidence.LABELS,
            "schema_version": 1,
            "run_id": run_dir.name,
            "role": role,
            "arm": "resident",
            "status": "failed",
            "repetitions": [],
            "repetitions_completed": 0,
        }
    )
    row["exit_code"] = returncode if returncode >= 0 else 128 - returncode
    row["launcher_elapsed_s"] = perf_counter() - started
    if returncode in (124, 137):
        row["status"] = "timeout"
        row.update(failure_code="timeout", failure_stage="launcher")
    elif returncode or row.get("status") != "complete":
        row["status"] = "failed"
        row.setdefault("failure_code", "launcher-failed")
        row.setdefault("failure_stage", "launcher")
    shared._atomic(path, row)
    print(f"{role}: {row['status']}", flush=True)
    return row


def export_suite(source: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    manifest = shared._read(source / "suite.json")
    runs = []
    for role in evidence.ROLES:
        run_id = evidence.identifier(manifest["roles"][role])
        try:
            raw = shared._read(source / run_id / "summary.json")
            smoke = source / run_id / "smoke.json"
            if not raw.get("smoke") and smoke.is_file():
                raw = {**raw, "smoke": shared._read(smoke)}
            row = evidence.public_run(raw)
            if row["run_id"] != run_id or row["role"] != role:
                raise ValueError("manifest role differs")
        except (ValueError, TypeError, KeyError, AttributeError, OSError):
            row = {
                **evidence.LABELS,
                "run_id": run_id,
                "role": role,
                "status": "failed",
                "evidence_status": "malformed-or-missing",
                "failure_code": "malformed-or-missing",
                "failure_stage": "export",
                "repetitions": [],
            }
        runs.append(row)
    anchor = None
    anchor_invalid = False
    if manifest.get("order") == "reverse":
        try:
            anchor = evidence.public_run(shared._read(source / "anchor.json"))
            evidence.validate_run(anchor)
        except (ValueError, TypeError, KeyError, AttributeError, OSError):
            anchor_invalid = True
    comparison = evidence.analyze(runs, anchor)
    if anchor_invalid:
        comparison = {
            **evidence.LABELS,
            "status": "invalid",
            "decision": "insufficient-evidence",
        }
    result = {
        **evidence.LABELS,
        "schema_version": 1,
        "suite_id": evidence.identifier(source.name),
        "order": manifest.get("order")
        if manifest.get("order") in ("forward", "reverse")
        else "unknown",
        "runs": runs,
        "comparison": comparison,
    }
    if anchor is not None and not anchor_invalid:
        result["anchor"] = anchor
    shared._atomic(output / "results.json", result)
    with (output / "points.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "role",
                "status",
                "rank",
                "actual_KV_GB",
                "used_GPU_GB",
                "torch_peak_reserved_GB",
            ]
        )
        for row in runs:
            for memory in row.get("final_memory", []) or [{}]:

                def gb(name: str, memory: dict[str, Any] = memory) -> float | None:
                    value = memory.get(name)
                    return value / 1e9 if type(value) is int else None

                used = (
                    (memory["total_gpu_bytes"] - memory["free_gpu_bytes"]) / 1e9
                    if "total_gpu_bytes" in memory and "free_gpu_bytes" in memory
                    else None
                )
                writer.writerow(
                    [
                        row["role"],
                        row["status"],
                        memory.get("rank"),
                        gb("kv_cache_allocated_bytes"),
                        used,
                        gb("torch_peak_reserved_bytes"),
                    ]
                )
    decision_text = {
        "consider-small-offload": "small 至少20%且有调度机制改善，可讨论后续小比例卸载；本次未运行卸载。",
        "stop-throughput-route": "small/large 均低于10%且无已观测机制改善，建议停止该负载吞吐路线。",
        "low-cost-only": "存在有限信号，仅考虑低成本改动。",
        "large-only-not-small-offload-evidence": "large 有信号，不能证明5层/1槽方案可行。",
        "insufficient-evidence": "证据不足或尚未重复确认；不据此判断没有收益，也不授权卸载。",
    }
    lines = [
        "# KV Oracle 容量诊断",
        "",
        "仅诊断：diagnostic_only=true；formal_offload_gain=false。",
        "O组显式KV可能超过r0名义0.60预算，不能算公平卸载收益或严格数学上界。",
        "安全检查为估算加观测边界，不是硬显存上限；非Torch瞬态峰值可能未捕获。",
        "表内显存使用十进制GB。请求观测是运行时批次成员数，不是等待队列；不可用字段为null。",
        "",
        f"状态：{comparison['status']}。",
        decision_text[comparison["decision"]],
        "",
    ]
    for role in ("small", "large"):
        if role in comparison:
            point = comparison[role]
            lines.append(
                f"- {role}: 吞吐比 {point['throughput_ratio']:.4f}；耗时中位数 "
                f"{point['elapsed_median_s']:.3f}s，范围 {point['elapsed_range_s']}；"
                f"逐轮时间收益空间 {point['time_savings_s']}s。"
            )
    for row in runs:
        lines.append(
            f"- {row['role']} 状态：{row['status']}；"
            f"失败码：{row.get('failure_code', 'none')}。"
        )
    lines.extend(
        [
            "",
            "输出正确性范围仅为batch1 smoke；性能批次哈希保留供检查。",
            "正序和反序都要检查；screen单轮不提供稳定结论。",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def execute_suite(args: argparse.Namespace) -> Path:
    root: Path = args.project_root.resolve()
    options = vars(args).copy()
    options.update(
        zip(
            ("batch_size", "context_length", "output_length"),
            PRESETS[args.preset],
            strict=True,
        )
    )
    options.update(warmups=1, repetitions=1 if args.command == "screen" else 3)
    for name in (
        "timeout_s",
        "max_num_seqs",
        "max_num_batched_tokens",
        "small_add_bytes",
        "large_add_bytes",
    ):
        evidence.integer(options[name], 1)
    evidence.integer(args.seed)
    evidence.integer(args.safety_margin_bytes)
    if (
        not 0 < evidence.number(args.safety_fraction) <= 1
        or args.large_add_bytes <= args.small_add_bytes
    ):
        raise ValueError("invalid safety or increment settings")
    for name in ("dataset_path", "dataset_manifest"):
        if not Path(options[name]).is_absolute():
            options[name] = root / options[name]
    if args.command == "screen" and args.order != "forward":
        raise ValueError("screen requires forward order")
    anchor = None
    if args.order == "reverse":
        if args.anchor is None:
            raise ValueError("reverse requires prior completed r0 anchor")
        anchor = shared._read(args.anchor / "summary.json")
        evidence.validate_run(anchor)
        if anchor["role"] != "r0":
            raise ValueError("reverse anchor must be r0")
        for name in (
            "batch_size",
            "context_length",
            "output_length",
            "max_num_seqs",
            "max_num_batched_tokens",
            "seed",
            "warmups",
        ):
            if anchor["contract"][name] != options[name]:
                raise ValueError("reverse anchor workload/engine differs")
    suite_id: str = evidence.identifier(
        args.suite_id
        or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
    )
    destination = root / "runs/kv-oracle" / suite_id
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        **evidence.LABELS,
        "suite_id": suite_id,
        "status": "running",
        "order": args.order,
        "preset": args.preset,
        "roles": {role: role for role in evidence.ROLES},
    }
    if anchor is not None:
        shared._atomic(destination / "anchor.json", evidence.public_run(anchor))
    shared._atomic(destination / "suite.json", manifest)
    order = (
        evidence.ROLES if args.order == "forward" else tuple(reversed(evidence.ROLES))
    )
    for role in order:
        reference = (
            None
            if role == "r0"
            else (destination / "r0" if args.order == "forward" else args.anchor)
        )
        row = execute_point(root, role, destination / role, reference, options)
        if row.get("status") != "complete":
            manifest["status"] = "failed"
            break
    else:
        manifest["status"] = "finished"
    shared._atomic(destination / "suite.json", manifest)
    export_suite(destination, root / "docs/results" / ("kv-oracle-" + suite_id))
    return destination


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    for mode in ("screen", "confirm", "export"):
        sub = commands.add_parser(mode)
        sub.add_argument("--project-root", type=Path, default=Path.cwd())
        if mode == "export":
            sub.add_argument("--source", type=Path, required=True)
            sub.add_argument("--output", type=Path, required=True)
            continue
        sub.add_argument("--preset", choices=tuple(PRESETS), default="main")
        sub.add_argument("--suite-id")
        sub.add_argument("--order", choices=("forward", "reverse"), default="forward")
        sub.add_argument("--anchor", type=Path)
        sub.add_argument(
            "--model-path",
            type=Path,
            default=Path("/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct"),
        )
        sub.add_argument(
            "--dataset-path",
            type=Path,
            default=Path("benchmarks/data/sharegpt/qwen3next_1024_requests.jsonl.zst"),
        )
        sub.add_argument(
            "--dataset-manifest",
            type=Path,
            default=Path("benchmarks/data/sharegpt/dataset_manifest.json"),
        )
        for name, default in {
            "timeout-s": 7200,
            "max-num-seqs": 512,
            "max-num-batched-tokens": 8192,
            "seed": 20260905,
            "small-add-bytes": 3_200_000_000,
            "large-add-bytes": 18_100_000_000,
            "safety-margin-bytes": 2_000_000_000,
        }.items():
            sub.add_argument("--" + name, type=int, default=default)
        sub.add_argument("--safety-fraction", type=float, default=0.90)
    return cli


def main(argv: Sequence[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    root = args.project_root.resolve()
    if root != shared.EXPECTED_ROOT:
        cli.error("server project root required")
    if args.command == "export":
        if not args.source.resolve().is_relative_to(
            root / "runs/kv-oracle"
        ) or not args.output.resolve().is_relative_to(root / "docs/results"):
            cli.error("export paths must remain inside project Oracle runs/results")
        export_suite(args.source, args.output)
        return 0
    if args.anchor is not None and not args.anchor.resolve().is_relative_to(
        root / "runs/kv-oracle"
    ):
        cli.error("anchor must remain inside project Oracle runs")
    path = execute_suite(args)
    manifest = shared._read(path / "suite.json")
    return 0 if manifest["status"] == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
