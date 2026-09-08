"""Offline diagnostic CLI, also runnable as python3 -S path/to/cli.py."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shlex
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if sys.version_info < (3, 10):  # noqa: UP036 - direct entry may use macOS Python 3.9
    raise SystemExit("Python >=3.10 required; use the existing project Python with -S")

# A single private package bootstrap avoids evaluating flexmoe.__init__ (Torch).
if not __package__:
    package_name = "_flexmoe_offline_analysis"
    spec = importlib.util.spec_from_file_location(
        package_name,
        Path(__file__).with_name("__init__.py"),
        submodule_search_locations=[str(Path(__file__).parent)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("analysis package unavailable")
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    __package__ = package_name

from .cost import analyze_feasibility
from .io import (
    atomic_json,
    load_replays,
    load_timing,
    load_trace,
    native_timing,
    oracle_adapter,
    read_json,
    transfer_groups,
)
from .replay import replay_trace, select_resident
from .report import export_report, public_artifact
from .schema import ReplayConfig, diagnostic_artifact

MODEL = "/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct"
DATA = "benchmarks/data/sharegpt/qwen3next_1024_requests.jsonl.zst"
MANIFEST = "benchmarks/data/sharegpt/dataset_manifest.json"


class StrictParser(argparse.ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["allow_abbrev"] = False
        super().__init__(*args, **kwargs)


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("positive integer required")
    return number


def plan(args: argparse.Namespace) -> dict[str, Any]:
    prefix = args.id_prefix or "oa-" + uuid.uuid4().hex[:12]
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}", prefix):
        raise ValueError("invalid plan ID prefix")
    wrapper = "scripts/server/run_offload_analysis.sh"
    base = [
        "--model-path",
        args.model_path,
        "--dataset-manifest",
        args.dataset_manifest,
        "--seed",
        str(args.seed),
        "--warmups",
        str(args.warmups),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--safety-reserve-bytes",
        str(args.safety_reserve_bytes),
    ]

    def command(
        mode: str,
        suffix: str,
        context: int,
        output: int,
        batch: int,
        utilization: float = 0.9,
        extra: Sequence[str] = (),
    ) -> dict[str, Any]:
        argv = [
            "bash",
            wrapper,
            mode,
            "--run-id",
            prefix + "-" + suffix,
            "--timeout-s",
            str(args.timeout_s),
            *base,
            "--dataset-path",
            str(args.dataset_path),
            "--batch-size",
            str(batch),
            "--context-length",
            str(context),
            "--output-length",
            str(output),
            "--max-num-seqs",
            str(batch),
            "--gpu-memory-utilization",
            str(utilization),
            "--repetitions",
            "1" if mode == "trace" else str(args.repetitions),
            *extra,
        ]
        return {"argv": argv, "shell": shlex.join(argv)}

    scan = [
        command(
            "resident",
            f"{name}-b{batch}-u{int(util * 100)}",
            context,
            output,
            batch,
            util,
        )
        for util in ([0.9, 0.6] if args.include_low_budget else [0.9])
        for name, context, output in (
            ("short", 1024, 128),
            ("generation", 1024, 2048),
            ("medium", 4096, 512),
        )
        for batch in (16, 32, 64, 128, 256, 512)
    ]
    cal_id = prefix + "-calibration"
    eval_id = prefix + "-evaluation"
    cal_path = f"runs/offload-analysis/{cal_id}/trace-rank-0.json.gz"
    traces = [
        f"runs/offload-analysis/{eval_id}/trace-rank-{r}.json.gz" for r in range(4)
    ]
    calibrations = [
        f"runs/offload-analysis/{cal_id}/trace-rank-{r}.json.gz" for r in range(4)
    ]
    selected = [
        command(
            "trace",
            "calibration",
            args.selected_context,
            args.selected_output,
            32,
            extra=("--max-trace-steps", str(args.max_trace_steps)),
        ),
        command(
            "resident",
            "matched-native",
            args.selected_context,
            args.selected_output,
            args.selected_batch,
            extra=("--exclude-trace", cal_path),
        ),
        command(
            "trace",
            "evaluation",
            args.selected_context,
            args.selected_output,
            args.selected_batch,
            extra=(
                "--exclude-trace",
                cal_path,
                "--max-trace-steps",
                str(args.max_trace_steps),
            ),
        ),
    ]
    transport = [
        "bash",
        wrapper,
        "transport",
        "--run-id",
        prefix + "-transport",
        "--model-path",
        str(args.model_path),
        "--timeout-s",
        str(args.timeout_s),
        "--warmups",
        str(args.warmups),
        "--repetitions",
        str(args.repetitions),
        "--safety-reserve-bytes",
        str(args.safety_reserve_bytes),
        "--expert-bytes",
        str(args.expert_bytes),
        "--experts-per-batch",
        "1,8,32,128,512",
        "--modes",
        "contiguous,fragmented,gather",
        "--contention",
        "isolated",
    ]
    replay_path = f"runs/offload-analysis/{prefix}-offline/replay.json"
    analysis_path = f"runs/offload-analysis/{prefix}-offline/analysis.json"
    offline = [
        [
            "python3",
            "-S",
            "src/flexmoe/analysis/cli.py",
            "replay",
            "--traces",
            *traces,
            "--calibration",
            *calibrations,
            "--output",
            replay_path,
        ],
        [
            "python3",
            "-S",
            "src/flexmoe/analysis/cli.py",
            "analyze",
            "--traces",
            *traces,
            "--replay",
            replay_path,
            "--samples",
            f"runs/offload-analysis/{prefix}-transport/samples.json",
            "--baseline",
            f"runs/offload-analysis/{prefix}-matched-native/summary.json",
            "--output",
            analysis_path,
        ],
    ]
    phases = [
        {"phase": "native-scan", "requires_inspection": False, "commands": scan},
        {
            "phase": "selected-independent-traces",
            "requires_inspection": True,
            "condition": "Inspect normal-budget scan; adjust selected workload before running these commands.",
            "commands": selected,
        },
        {
            "phase": "transport",
            "requires_inspection": True,
            "condition": "Verify actual expert_bytes, reserve and complete trace coverage; optional proxy is observations only.",
            "commands": [{"argv": transport, "shell": shlex.join(transport)}],
        },
        {
            "phase": "offline",
            "requires_inspection": True,
            "condition": "Same execution SHA/input required. Optional safe measured K must fit per-rank net freed bytes.",
            "commands": [{"argv": a, "shell": shlex.join(a)} for a in offline],
        },
    ]
    if args.include_long:
        phases.insert(
            1,
            {
                "phase": "synthetic-long",
                "requires_inspection": True,
                "condition": "Synthetic packed existing tokens only; inspect capacity first, not representative traffic.",
                "commands": [
                    command(
                        "resident",
                        f"synthetic-{c}",
                        c,
                        128,
                        16,
                        extra=("--synthetic-context",),
                    )
                    for c in (16384, 32768)
                ],
            },
        )
    return diagnostic_artifact(
        "offload-analysis-plan",
        {
            "execution": "manual-phased-manifest",
            "id_prefix": prefix,
            "phases": phases,
            "failure_policy": "retain original artifacts and independent smoke; fresh IDs only",
            "replay_scan": {
                "offload_fractions": [0.05, 0.10, 0.20, 0.30, 0.50],
                "cache_slots": [0, 64, 256, 512],
                "staging_experts": 512,
            },
        },
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="离线专家卸载诊断；不证明部署收益", allow_abbrev=False
    )
    sub = root.add_subparsers(
        dest="command",
        required=True,
        parser_class=StrictParser,
    )
    validate = sub.add_parser("validate")
    validate.add_argument("inputs", type=Path, nargs="+")
    replay = sub.add_parser("replay")
    replay.add_argument("--traces", type=Path, nargs="+", required=True)
    replay.add_argument("--calibration", type=Path, nargs="+", required=True)
    replay.add_argument("--offload-fraction", type=float, default=0.20)
    replay.add_argument("--cache-slots", type=int, default=0)
    replay.add_argument("--staging-experts", type=positive, default=512)
    replay.add_argument("--extra-gpu-bytes", type=int, default=0)
    replay.add_argument(
        "--policy", choices=("lru", "decayed-lfu", "future"), default="lru"
    )
    replay.add_argument("--output", type=Path, required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--traces", type=Path, nargs="+", required=True)
    analyze.add_argument("--replay", type=Path, required=True)
    analyze.add_argument("--samples", type=Path, nargs="+", required=True)
    analyze.add_argument("--baseline", type=Path, required=True)
    analyze.add_argument("--kv-reference", type=Path)
    analyze.add_argument("--output", type=Path, required=True)
    planner = sub.add_parser("plan")
    planner.add_argument("--output", type=Path, required=True)
    planner.add_argument("--id-prefix")
    planner.add_argument("--model-path", default=MODEL)
    planner.add_argument("--dataset-path", default=DATA)
    planner.add_argument("--dataset-manifest", default=MANIFEST)
    planner.add_argument("--include-low-budget", action="store_true")
    planner.add_argument("--include-long", action="store_true")
    for name, default in (
        ("seed", 0),
        ("warmups", 1),
        ("repetitions", 3),
        ("max-num-batched-tokens", 4096),
        ("timeout-s", 7200),
        ("safety-reserve-bytes", 2_000_000_000),
        ("selected-batch", 64),
        ("selected-context", 1024),
        ("selected-output", 128),
        ("max-trace-steps", 1024),
        ("expert-bytes", 1572864),
    ):
        planner.add_argument(
            "--" + name, type=int if name == "seed" else positive, default=default
        )
    export = sub.add_parser("export")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    failure = sub.add_parser("record-failure", help="Internal launcher failure record")
    failure.add_argument("--run-dir", type=Path, required=True)
    failure.add_argument("--exit-code", type=int, required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "validate":
            for path in args.inputs:
                raw = read_json(path)
                kind = raw.get("artifact_kind")
                if kind == "native-summary":
                    native_timing(raw)
                elif kind == "demand-trace":
                    load_trace(path)
                elif kind == "transfer-samples":
                    transfer_groups([path])
                elif kind == "replay-suite":
                    load_replays(path)
                elif raw.get("comparison_backend") == "kv-oracle":
                    oracle_adapter(raw)
                else:
                    public_artifact(raw)
            print(
                json.dumps(
                    diagnostic_artifact(
                        "validation", {"status": "complete", "files": len(args.inputs)}
                    )
                )
            )
        elif args.command == "replay":
            traces = [load_trace(p) for p in args.traces]
            calibration = [load_trace(p) for p in args.calibration]
            if sorted(t.rank for t in traces) != list(range(4)) or sorted(
                t.rank for t in calibration
            ) != list(range(4)):
                raise ValueError("four distinct evaluation/calibration ranks required")
            configs = {
                t.rank: ReplayConfig(
                    select_resident(
                        next(c for c in calibration if c.rank == t.rank),
                        t,
                        args.offload_fraction,
                    ),
                    args.cache_slots,
                    args.staging_experts,
                    args.policy,
                    args.extra_gpu_bytes,
                )
                for t in traces
            }
            atomic_json(
                args.output,
                diagnostic_artifact(
                    "replay-suite",
                    {
                        "replays": [
                            replay_trace(t, configs[t.rank]).to_dict() for t in traces
                        ],
                        "calibration_input_hashes": [
                            c.contract["input_sha256"] for c in calibration
                        ],
                    },
                ),
            )
        elif args.command == "analyze":
            traces = [load_trace(p) for p in args.traces]
            replays = load_replays(args.replay)
            baseline = load_timing(args.baseline)
            if baseline is None:
                raise ValueError(
                    "legacy Oracle cannot supply normal-budget native baseline"
                )
            reference = load_timing(args.kv_reference) if args.kv_reference else None
            results = [
                analyze_feasibility(traces, replays, group, baseline, reference)
                for group in transfer_groups(args.samples)
            ]
            fields: dict[str, Any] = {
                "analyses": results,
                "baseline_evidence": baseline.to_dict(),
                "reference_evidence": reference.to_dict() if reference else None,
            }
            if args.kv_reference and reference is None:
                fields["legacy_reference"] = oracle_adapter(
                    read_json(args.kv_reference)
                )
            atomic_json(args.output, diagnostic_artifact("analysis-suite", fields))
        elif args.command == "plan":
            atomic_json(args.output, plan(args))
        elif args.command == "export":
            export_report(args.source, args.output)
        elif args.command == "record-failure":
            atomic_json(
                args.run_dir / "launcher.json",
                diagnostic_artifact(
                    "launcher-failure",
                    {
                        "status": "failed",
                        "phase": "launcher",
                        "exit_code": args.exit_code,
                    },
                ),
            )
    except (OSError, ValueError, TypeError, KeyError, AssertionError) as error:
        print(
            f"invalid/incomplete evidence: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
