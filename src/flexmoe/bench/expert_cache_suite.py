"""Stdlib-only offline calibration, equal-budget confirmation, and safe export."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, cast


def _sibling(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        "_expert_host_" + name, Path(__file__).with_name(name + ".py")
    )
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


shared = _sibling("partial_suite")
evidence = _sibling("expert_cache_evidence")
OPTIONS = (
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
    "dataset_path",
    "dataset_manifest",
    "model_path",
    "profile_path",
    "resident_ratio",
    "cache_slots",
    "cache_policy",
    "calibration_count",
)


def point_command(
    root: Path,
    arm: str,
    run_dir: Path,
    reference: Path | None,
    options: Mapping[str, Any],
    *,
    mode: str = "point",
) -> list[str]:
    command = [
        "bash",
        str(root / "scripts/server/run_expert_cache.sh"),
        mode,
        "--timeout-s",
        str(options["timeout_s"]),
        "--arm",
        arm,
        "--run-dir",
        str(run_dir),
    ]
    for name in OPTIONS:
        command.extend(["--" + name.replace("_", "-"), str(options[name])])
    if reference is not None:
        command.extend(["--resident-run", str(reference)])
    return command


def execute_point(
    root: Path,
    arm: str,
    run_dir: Path,
    reference: Path | None,
    options: Mapping[str, Any],
    *,
    mode: str = "point",
) -> dict[str, Any]:
    if run_dir.exists():
        raise FileExistsError(run_dir)
    logs = run_dir.parent / "logs"
    logs.mkdir(exist_ok=True)
    started = perf_counter()
    command = point_command(root, arm, run_dir, reference, options, mode=mode)
    with (
        (logs / (run_dir.name + ".stdout.log")).open("w") as stdout,
        (logs / (run_dir.name + ".stderr.log")).open("w") as stderr,
    ):
        process = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    path = run_dir / "summary.json"
    row = (
        shared._read(path)
        if path.exists()
        else {
            "schema_version": 1,
            "run_id": run_dir.name,
            "arm": arm,
            "storage_backend": "native" if arm == "resident" else "expert-cache",
            "status": "failed",
            "repetitions": [],
            "repetitions_completed": 0,
        }
    )
    row["exit_code"] = (
        process.returncode if process.returncode >= 0 else 128 - process.returncode
    )
    row["launcher_elapsed_s"] = perf_counter() - started
    if process.returncode in (124, 137):
        row["status"] = "timeout"
    elif process.returncode or row.get("status") != "complete":
        row["status"] = "failed"
    shared._atomic(path, row)
    print(f"{run_dir.name}: {row['status']}", flush=True)
    return row


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        return [
            pair
            for key, item in value.items()
            for pair in _flatten(item, f"{prefix}.{key}" if prefix else key)
        ]
    if isinstance(value, list):
        return [
            pair
            for index, item in enumerate(value)
            for pair in _flatten(item, f"{prefix}[{index}]")
        ]
    return [(prefix, value)]


def export_suite(source: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    manifest = shared._read(source / "suite.json")
    runs = []
    for run_id in manifest.get("run_ids", []):
        run_id = shared._identifier(run_id)
        path = source / run_id / "summary.json"
        try:
            raw = shared._read(path)
            # A worker terminated after smoke may leave smoke.json while the
            # parent has not yet merged it into summary.json. Preserve that
            # measured, allowlisted artifact in the public failure report.
            if not raw.get("smoke"):
                smoke_path = source / run_id / "smoke.json"
                if smoke_path.is_file():
                    raw = {**raw, "smoke": shared._read(smoke_path)}
            row = shared.public_run(raw)
            if row["run_id"] != run_id:
                raise ValueError("manifest and summary run ID differ")
            # A malformed completed measurement must not be rendered as a gain.
            if row["status"] == "complete":
                shared._speeds(row)
        except (ValueError, TypeError, KeyError, AttributeError, OSError):
            row = {
                "schema_version": 1,
                "run_id": run_id,
                "arm": "unknown",
                "status": "failed",
                "evidence_status": "malformed-or-missing",
                "contract": {},
                "repetitions": [],
                "memory": [],
            }
        runs.append(row)
    by_id = {row["run_id"]: row for row in runs}
    comparisons = []
    for group in manifest.get("groups", []):
        ids = {arm: shared._identifier(group[arm]) for arm in shared.ARMS}
        comparison = shared.analyze_triplet(
            *(by_id.get(ids[arm], {}) for arm in shared.ARMS)
        )
        comparisons.append({**comparison, "run_ids": ids})
    result: dict[str, Any] = {
        "schema_version": 1,
        **shared._hashes(manifest, ("commit",)),
        "backend": "expert-cache",
        "runs": runs,
        "comparisons": comparisons,
        "suite_status": manifest.get("status")
        if manifest.get("status") in ("finished", "failed", "running")
        else "failed",
        "cuda_preflight": manifest.get("cuda_preflight")
        if manifest.get("cuda_preflight") in ("passed", "failed", "pending")
        else "pending",
    }
    calibration_path = source / "calibration" / "summary.json"
    if calibration_path.is_file():
        try:
            result["calibration"] = evidence.public_calibration(
                shared._read(calibration_path)
            )
        except (KeyError, TypeError, ValueError):
            result["calibration"] = {"status": "failed", "evidence_status": "malformed"}
    shared._atomic(output / "summary.json", result)
    fields = (
        "run_id",
        "storage_backend",
        "arm",
        "status",
        "throughput_median",
        "throughput_min",
        "throughput_max",
        "kv_bytes_total",
    )
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in runs:
            try:
                speeds = shared._speeds(row)
            except (KeyError, TypeError, ValueError):
                speeds = []
            writer.writerow(
                {
                    **{key: row.get(key) for key in fields[:4]},
                    "throughput_median": median(speeds) if speeds else None,
                    "throughput_min": min(speeds) if speeds else None,
                    "throughput_max": max(speeds) if speeds else None,
                    "kv_bytes_total": sum(
                        x.get("kv_cache_allocated_bytes") or 0 for x in row["memory"]
                    )
                    if row["memory"]
                    else None,
                }
            )
    # Long-form CSV preserves all nested allowlisted counters, layer arrays,
    # kernel configurations, requested settings, hashes and actual memory.
    with (output / "evidence.csv").open("w", newline="", encoding="utf-8") as stream:
        long_writer = csv.writer(stream)
        long_writer.writerow(("run_id", "field", "value"))
        for row in runs:
            for field, value in _flatten(row):
                long_writer.writerow((row["run_id"], field, value))
        for field, value in _flatten(result.get("calibration", {})):
            long_writer.writerow(("calibration", field, value))
    lines = [
        "# BF16 热专家常驻与共享缓存离线结果",
        "",
        f"Suite 状态：{result['suite_status']}；CUDA 前置检查：{result['cuda_preflight']}。",
        f"Suite 执行 SHA：{result.get('commit', '不可用')}。",
        "",
        "R=原生 resident；B=专家缓存且实际 KV 等于 R；C=相同专家缓存、相同显存利用率下自动 KV。",
        "实际吞吐由固定输出 token 数 / 测得秒数核对；未完成或损坏结果不产生增益结论。容量采用十进制 GB。",
        "",
    ]
    for row in runs:
        lines.append(f"## {row['run_id']}：{row['status']}")
        try:
            speeds = shared._speeds(row)
            lines.append(
                f"输出 token/s 中位数 {median(speeds):.3f}，范围 [{min(speeds):.3f}, {max(speeds):.3f}]。"
            )
        except (KeyError, TypeError, ValueError):
            lines.append("完整吞吐证据不可用；保留失败状态和已完成数值。")
        values = [m.get("kv_cache_allocated_bytes") for m in row["memory"]]
        if values and all(type(x) is int for x in values):
            lines.append(f"各 rank 实际 KV 合计 {sum(values) / 1e9:.6f} GB。")
        settings = row.get("contract", {}).get("expert_cache", {})
        if settings:
            lines.append(
                f"常驻比例 {settings['resident_ratio']}，共享 cache slots={settings['cache_slots']}，policy={settings['cache_policy']}，profile SHA={settings['profile_sha256']}。"
            )
            contract = row["contract"]
            command = (
                'bash scripts/server/run_expert_cache.sh confirm --suite-id "$NEW_SUITE_ID" '
                '--model-path "$MODEL_PATH" --profile-path "$PROFILE_PATH" '
                '--dataset-path "$DATASET_PATH" --dataset-manifest "$DATASET_MANIFEST"'
            )
            for key in (
                "batch_size",
                "context_length",
                "output_length",
                "gpu_memory_utilization",
                "warmups",
                "seed",
                "max_num_seqs",
                "max_num_batched_tokens",
                "timing_samples",
            ):
                if key in contract:
                    command += " --" + key.replace("_", "-") + " " + str(contract[key])
            if "repetitions_requested" in contract:
                command += " --repetitions " + str(contract["repetitions_requested"])
            for key in (
                "resident_ratio",
                "cache_slots",
                "cache_policy",
                "calibration_count",
            ):
                command += " --" + key.replace("_", "-") + " " + str(settings[key])
            lines.extend(
                [
                    f"执行 SHA：{contract.get('commit')}。以下为完整 R/B/C 的脱敏重跑模板，私有路径需在服务器设定变量：",
                    "",
                    "```bash",
                    command,
                    "```",
                    "",
                ]
            )
        for rep in row["repetitions"]:
            diag = rep.get("diagnostics", {})
            lines.append(
                f"第 {rep['repetition']} 轮：H2D={diag.get('h2d_bytes')} bytes；resident hits={diag.get('resident_hits')}；cold cache hit ratio={diag.get('policy', {}).get('cache_hit_ratio')}。"
            )
        detail = {
            key: row.get(key)
            for key in ("memory", "final_memory", "expert_cache_stats")
        }
        detail["measured"] = [
            {
                key: rep.get(key)
                for key in ("repetition", "diagnostics", "memory", "memory_peak_scope")
            }
            for rep in row["repetitions"]
        ]
        lines.extend(
            [
                "",
                "<details><summary>逐 rank / 逐层数值、kernel 选择与显存证据（bytes）</summary>",
                "",
                "```json",
                json.dumps(detail, ensure_ascii=False, indent=2, allow_nan=False),
                "```",
                "",
                "</details>",
            ]
        )
        lines.append("")
    if "calibration" in result:
        lines.append(
            f"原生校准状态：{result['calibration']['status']}。逐 rank 校准计数与完整输入哈希保存在同包 JSON/CSV。"
        )
    for comparison in comparisons:
        lines.append(f"比较：{comparison['status']}。")
        if "c_over_r" in comparison:
            lines.append(
                f"B/R={comparison['b_over_r']:.4f}；C/B={comparison['c_over_b']:.4f}；C/R={comparison['c_over_r']:.4f}；三轮稳定净收益={comparison['stable_three_repetition_gain']}。性能批量输出哈希一致={comparison['performance_output_hashes_match']}。"
            )
        elif "reason" in comparison:
            lines.append("原因：" + comparison["reason"])
    lines.extend(
        [
            "",
            "每层 forward / unique demand 增量与覆盖率、状态最大值、嵌套 policy / timing、实际 kernel 选择和每 rank 容量均保存在 summary.json 与 evidence.csv。覆盖率分母为该层 measured forward 数乘逻辑专家数；最大值是累计状态快照，不做相减。",
            "H2D=0 可以是全命中；命中率分母只含 cold cache hits+misses。命中率或正 H2D 均不能替代端到端收益。吞吐变慢同样是有效负结果。",
            "正确性只由独立 CUDA parity tests 与 batch=1 smoke 各自覆盖；性能批量输出哈希单列。runtime weights_verified=0，未执行运行时大权重 D2H 验证。",
            "模型身份绑定 config/index/本地路径哈希，未逐字节散列全 checkpoint。校准和评估按完整 prompt token 哈希分离，未声称源会话独立。",
            "原始 prompt、生成文本、权重路径、机器身份及日志不在此导出包内。",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def execute_suite(args: argparse.Namespace) -> Path:
    root = args.project_root.resolve()
    if root != shared.EXPECTED_ROOT or not os.environ.get("GPU_IDS"):
        raise ValueError("server root and explicit four exclusive GPU_IDS required")
    suite_id = shared._identifier(
        args.suite_id
        or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.mode
    )
    source = root / "runs" / "expert-cache" / suite_id
    source.mkdir(parents=True, exist_ok=False)
    options = vars(args).copy()
    profile_path = args.profile_path or source / "profile.json"
    if args.mode == "confirm" and not profile_path.is_file():
        raise ValueError("confirmation requires an existing calibration profile")
    if args.mode == "calibrate" and not profile_path.resolve().is_relative_to(root):
        raise ValueError("profile output must remain inside project root")
    options["profile_path"] = profile_path
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "run_ids": [],
        "groups": [],
        "status": "running",
        "cuda_preflight": "pending",
    }
    shared._atomic(source / "suite.json", manifest)
    check = subprocess.run(
        [
            "bash",
            str(root / "scripts/server/run_expert_cache.sh"),
            "cuda-check",
            "--timeout-s",
            str(args.timeout_s),
            "--model-path",
            str(args.model_path),
            "--output",
            str(source / "cuda-preflight.json"),
        ],
        check=False,
    )
    manifest["cuda_preflight"] = "passed" if check.returncode == 0 else "failed"
    if check.returncode:
        manifest["status"] = "failed"
    elif args.mode == "calibrate":
        row = execute_point(
            root,
            "resident",
            source / "calibration",
            None,
            options,
            mode="calibrate-point",
        )
        manifest["status"] = "finished" if row["status"] == "complete" else "failed"
        # Calibration counts remain a sanitized profile/provenance artifact, not a performance arm.
        manifest["calibration_status"] = row["status"]
        manifest["profile_sha256"] = row.get("profile_sha256")
    else:
        resident = source / "01-resident"
        group = {arm: f"{index:02d}-{arm}" for index, arm in enumerate(shared.ARMS, 1)}
        manifest["groups"].append(group)
        completed = []
        for arm in shared.ARMS:
            run_id = group[arm]
            manifest["run_ids"].append(run_id)
            shared._atomic(source / "suite.json", manifest)
            row = execute_point(
                root,
                arm,
                source / run_id,
                None if arm == "resident" else resident,
                options,
            )
            if row["status"] != "complete":
                manifest["status"] = "failed"
                break
            completed.append(row)
        else:
            comparison = shared.analyze_triplet(*completed)
            manifest["status"] = (
                "finished"
                if comparison["status"]
                in ("validated-throughput-gain", "no-stable-net-gain")
                else "failed"
            )
    shared._atomic(source / "suite.json", manifest)
    export_suite(source, source / "public")
    print(source / "public", flush=True)
    return cast(Path, source)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="mode", required=True)
    for mode in ("calibrate", "confirm"):
        command = commands.add_parser(mode)
        command.add_argument("--project-root", type=Path, default=shared.EXPECTED_ROOT)
        command.add_argument("--suite-id")
        command.add_argument("--profile-path", type=Path, required=mode == "confirm")
        command.add_argument(
            "--model-path",
            type=Path,
            default=Path("/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct"),
        )
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
        command.add_argument("--resident-ratio", type=float, default=0.5)
        command.add_argument("--cache-slots", type=int, default=256)
        command.add_argument(
            "--cache-policy", choices=("lru", "decayed-lfu"), default="decayed-lfu"
        )
        command.add_argument("--gpu-memory-utilization", type=float, default=0.6)
        for flag, default in (
            ("calibration-count", 32),
            ("timeout-s", 7200),
            ("batch-size", 512),
            ("context-length", 4096),
            ("output-length", 256),
            ("warmups", 1),
            ("repetitions", 3),
            ("max-num-seqs", 256),
            ("max-num-batched-tokens", 8192),
            ("timing-samples", 128),
            ("seed", 20260905),
        ):
            command.add_argument("--" + flag, type=int, default=default)
    export = commands.add_parser("export")
    export.add_argument("--project-root", type=Path, default=Path.cwd())
    export.add_argument("--suite-dir", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.mode == "export":
        if not args.output_dir.resolve().is_relative_to(args.project_root.resolve()):
            arguments.error("output must remain inside project root")
        export_suite(args.suite_dir, args.output_dir)
    else:
        if args.timeout_s <= 0 or args.repetitions <= 0 or args.warmups < 0:
            arguments.error("invalid timeout/repetition/warmup")
        source = execute_suite(args)
        return 0 if shared._read(source / "suite.json")["status"] == "finished" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
