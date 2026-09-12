"""Measured decode evidence validation, pairing and finite manual server plans."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import re
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any, TypeGuard

if not __package__:
    spec = importlib.util.spec_from_file_location(
        "_decode_offline",
        Path(__file__).with_name("__init__.py"),
        submodule_search_locations=[str(Path(__file__).parent)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("analysis package unavailable")
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    __package__ = spec.name

from .schema import diagnostic_artifact

MODEL = "/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct"
DATA = "benchmarks/data/decode-mechanism/qwen3next_unique_1536x_1k4k.jsonl.zst"
MANIFEST = "benchmarks/data/decode-mechanism/dataset_manifest.json"
MODES = {"native", "matched-resident", "offload"}
KINDS = {"decode-run", "decode-smoke", "decode-repetition", "decode-profile"}
IDENTITIES = (
    "commit",
    "model_identity_sha256",
    "model_config_sha256",
    "hardware_sha256",
    "dataset_sha256",
    "dataset_manifest_sha256",
    "input_sha256",
    "calibration_input_hashes_sha256",
)
PAIR_FIELDS = (
    *IDENTITIES,
    "versions",
    "batch_size",
    "context_length",
    "output_length",
    "selected_input_hashes",
    "calibration_input_hashes",
    "seed",
    "dtype",
    "tensor_parallel_size",
    "max_num_seqs",
    "max_num_batched_tokens",
    "selection_offset",
    "warmups",
    "repetitions_requested",
    "timing_samples",
    "target_batch",
    "capture_steps",
    "min_capture_steps",
    "trace_budget_bytes",
    "physical_safety_reserve_bytes",
    "observation_policy",
    "gpu_memory_utilization",
)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def integer(value: object, minimum: int = 0) -> TypeGuard[int]:
    return type(value) is int and value >= minimum


def number(value: object, minimum: float = 0) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= minimum
    )


def hash_string(value: object, lengths: tuple[int, ...] = (64,)) -> bool:
    return (
        isinstance(value, str)
        and len(value) in lengths
        and re.fullmatch("[0-9a-f]+", value) is not None
    )


def ranked(value: object) -> list[dict[str, Any]]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(isinstance(x, dict) for x in value)
    ):
        raise ValueError("rank-evidence")
    rows = [mapping(x) for x in value]
    if any(not integer(x.get("rank")) for x in rows) or {x["rank"] for x in rows} != {
        0,
        1,
        2,
        3,
    }:
        raise ValueError("rank-evidence")
    return sorted(rows, key=lambda x: x["rank"])


def labels_valid(raw: Mapping[str, Any], kind: str) -> bool:
    profile = raw.get("profile")
    return (
        raw.get("artifact_kind") == kind
        and raw.get("schema_version") == 1
        and raw.get("comparison_backend") == "decode-mechanism"
        and raw.get("mode") in MODES
        and type(profile) is bool
        and raw.get("evidence_kind") == ("instrumented" if profile else "measured")
        and raw.get("timing_eligible") is (not profile)
        and raw.get("diagnostic_only") is True
        and raw.get("formal_offload_gain") is False
        and raw.get("deployment_gain_proven") is False
    )


def _memory_errors(raw: object, kv: Mapping[str, Any], reserve: object) -> set[str]:
    errors: set[str] = set()
    try:
        rows = ranked(raw)
    except ValueError:
        return {"memory-evidence"}
    actual, blocks = kv.get("allocated_bytes_per_rank"), kv.get("num_gpu_blocks")
    if not integer(actual, 1) or not integer(blocks, 1):
        return {"kv-evidence"}
    requested = kv.get("requested_bytes")
    for row in rows:
        fields = (
            "total_gpu_bytes",
            "free_gpu_bytes",
            "torch_allocated_bytes",
            "torch_reserved_bytes",
            "torch_peak_allocated_bytes",
            "torch_peak_reserved_bytes",
            "model_memory_bytes",
        )
        if any(not integer(row.get(k)) for k in fields) or not integer(reserve, 1):
            errors.add("memory-evidence")
        elif (
            row["free_gpu_bytes"] > row["total_gpu_bytes"]
            or row["free_gpu_bytes"]
            - max(0, row["torch_peak_reserved_bytes"] - row["torch_reserved_bytes"])
            < reserve
        ):
            errors.add("physical-safety")
        if (
            row.get("kv_cache_accounting_consistent") is not True
            or row.get("kv_cache_allocated_bytes") != actual
            or row.get("kv_cache_declared_bytes") != actual
            or row.get("num_gpu_blocks") != blocks
        ):
            errors.add("kv-evidence")
        if requested is not None and (
            not integer(requested, 1)
            or actual % blocks
            or not 0 <= requested - actual < actual // blocks
            or row.get("available_kv_cache_bytes") != requested
            or kv.get("rounding_bytes") != requested - actual
            or kv.get("bytes_per_block") != actual // blocks
        ):
            errors.add("kv-evidence")
    return errors


def validate_run(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on missing runtime evidence, retaining classified reasons."""
    errors: set[str] = set()
    c = mapping(raw.get("contract"))
    if not labels_valid(raw, "decode-run"):
        errors.add("artifact-labels")
    for key in (
        "comparison_backend",
        "mode",
        "profile",
        "evidence_kind",
        "timing_eligible",
    ):
        if c.get(key) != raw.get(key):
            errors.add("artifact-labels")
    if raw.get("profile") is not False:
        errors.add("instrumented-timing")
    if raw.get("status") != "complete" or raw.get("generation_status") != "complete":
        errors.add("generation-incomplete")
    if raw.get("capture_status") != "complete":
        errors.add("capture-failed")
    if mapping(raw.get("observer_finalization")).get("status") != "complete":
        errors.add("observer-finalization")
    if any(
        not hash_string(c.get(k), (40,) if k == "commit" else (64,)) for k in IDENTITIES
    ):
        errors.add("missing-identity")
    versions = mapping(c.get("versions"))
    if any(
        not isinstance(versions.get(k), str) or not versions[k]
        for k in ("torch", "vllm", "cuda", "python")
    ) or not hash_string(versions.get("vllm_commit"), (40,)):
        errors.add("missing-version")
    if (
        isinstance(versions.get("vllm"), str)
        and versions["vllm"].split("+")[0] != "0.10.2"
    ):
        errors.add("runtime-version")
    n, output = c.get("batch_size"), c.get("output_length")
    if any(
        not integer(c.get(k), 1)
        for k in (
            "batch_size",
            "output_length",
            "context_length",
            "max_num_seqs",
            "max_num_batched_tokens",
            "repetitions_requested",
            "min_capture_steps",
            "capture_steps",
        )
    ):
        errors.add("workload-contract")
    selected, calibration = (
        c.get("selected_input_hashes"),
        c.get("calibration_input_hashes"),
    )
    if (
        not isinstance(selected, list)
        or not isinstance(calibration, list)
        or not selected
        or not calibration
        or not all(hash_string(x) for x in selected + calibration)
    ) or (
        len(selected) != n
        or len(set(selected)) != n
        or len(set(calibration)) != len(calibration)
        or set(selected) & set(calibration)
        or c.get("prompt_hashes") != selected
        or c.get("calibration_input_hashes_sha256") != digest(calibration)
        or c.get("calibration_count") != len(calibration)
    ):
        errors.add("input-uniqueness")
    if (
        c.get("no_repetition") is not True
        or c.get("repeated_request_count") != 0
        or c.get("unique_selected_request_count") != n
    ):
        errors.add("input-uniqueness")
    policy = mapping(raw.get("engine_policy"))
    model, parallel, cache, sched = (
        mapping(policy.get(k))
        for k in ("model_config", "parallel_config", "cache_config", "scheduler_config")
    )
    if (
        c.get("engine_policy_sha256") != digest(policy)
        or c.get("tensor_parallel_size") != 4
        or c.get("dtype") != "bfloat16"
        or model.get("dtype") != "torch.bfloat16"
        or model.get("enforce_eager") is not (raw.get("mode") != "native")
        or "quantization" not in model
        or model["quantization"] is not None
        or parallel.get("tensor_parallel_size") != 4
        or parallel.get("pipeline_parallel_size") != 1
        or parallel.get("data_parallel_size") != 1
        or cache.get("cpu_offload_gb") != 0
        or cache.get("swap_space_bytes") != 0
        or cache.get("enable_prefix_caching") is not False
        or any(
            sched.get(k) != c.get(k) for k in ("max_num_seqs", "max_num_batched_tokens")
        )
    ):
        errors.add("engine-policy")
    kv, reserve = mapping(raw.get("actual_kv")), c.get("physical_safety_reserve_bytes")
    for memory in (raw.get("memory"), raw.get("final_memory")):
        errors.update(_memory_errors(memory, kv, reserve))
    if raw.get("mode") == "offload":
        settings = mapping(c.get("expert_cache"))
        if not hash_string(settings.get("profile_sha256")):
            errors.add("offload-identity")
        try:
            stats = ranked(raw.get("expert_cache_stats"))
            if raw.get("storage_backend") != "expert-cache" or any(
                x.get("failed") is not False
                or any(
                    x.get(k) != settings.get(k)
                    for k in (
                        "identity",
                        "profile_sha256",
                        "cache_slots",
                        "cache_policy",
                        "resident_ratio",
                    )
                )
                or not integer(x.get("startup_resident_h2d_bytes"))
                or any(
                    not integer(x.get(k), 1)
                    for k in (
                        "host_source_bytes",
                        "gpu_pool_bytes",
                        "net_freed_bytes",
                        "unique_demands",
                    )
                )
                or not isinstance(x.get("forward_counts"), list)
                or len(x["forward_counts"])
                != mapping(raw.get("geometry")).get("total_layers")
                or not all(integer(v, 1) for v in x["forward_counts"])
                for x in stats
            ):
                errors.add("offload-identity")
        except ValueError:
            errors.add("offload-identity")
    reps = raw.get("repetitions")
    if not isinstance(reps, list):
        reps = []
    if (
        len(reps) < 3
        or len(reps) != c.get("repetitions_requested")
        or raw.get("repetitions_completed") != len(reps)
        or [mapping(x).get("repetition") for x in reps] != list(range(len(reps)))
    ):
        errors.add("repetitions-incomplete")
    times: list[float] = []
    for value in reps:
        row = mapping(value)
        if not labels_valid(row, "decode-repetition") or any(
            row.get(k) != raw.get(k) for k in ("mode", "profile")
        ):
            errors.add("artifact-labels")
        if row.get("generation_status") != "complete":
            errors.add("generation-incomplete")
        if row.get("measurement_status") not in (None, "complete"):
            errors.add("measurement-rejected")
        if row.get("capture_status") != "complete":
            errors.add("capture-failed")
        if mapping(row.get("observer_finalization")).get("status") != "complete":
            errors.add("observer-finalization")
        if row.get("memory_status") != "measured":
            errors.add("memory-evidence")
        errors.update(_memory_errors(row.get("memory"), kv, reserve))
        elapsed = row.get("elapsed_s")
        if not number(elapsed) or elapsed <= 0:
            errors.add("elapsed-unavailable")
        else:
            times.append(float(elapsed))
        if (
            not integer(n, 1)
            or not integer(output, 1)
            or row.get("request_count") != n
            or row.get("generated_tokens") != n * output
            or not hash_string(row.get("output_sha256"))
        ):
            errors.add("fixed-output")
        try:
            obs = ranked(row.get("worker_observations"))
            for observation in obs:
                counts = mapping(observation.get("decode_batch_step_counts"))
                if not counts or any(
                    not str(k).isdigit() or int(k) <= 0 or not integer(v, 1)
                    for k, v in counts.items()
                ):
                    errors.add("batch-evidence")
                if (
                    c.get("target_batch") is not None
                    and observation.get("coverage_status") != "complete"
                ):
                    errors.add("target-unreached")
        except ValueError:
            errors.add("batch-evidence")
    smoke = mapping(raw.get("smoke"))
    if (
        not labels_valid(smoke, "decode-smoke")
        or not hash_string(smoke.get("output_sha256"))
        or not hash_string(smoke.get("input_sha256"))
        or smoke.get("generated_tokens") != c.get("smoke_output_length")
        or smoke.get("request_count") != 1
    ):
        errors.add("smoke-evidence")
    if (
        raw.get("performance_outputs_stable") is not True
        or len({mapping(x).get("output_sha256") for x in reps}) != 1
    ):
        errors.add("unstable-output")
    return {
        "status": "eligible" if not errors else "ineligible",
        "timing_eligible": not errors,
        "reasons": sorted(errors),
        "eligible_repetitions": len(times) if not errors else 0,
        "elapsed_median_s": median(times) if not errors else None,
        "throughput_median_tokens_s": median([n * output / t for t in times])
        if not errors and integer(n, 1) and integer(output, 1)
        else None,
    }


def compare_runs(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    c: Mapping[str, Any],
    native: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = diagnostic_artifact("decode-comparison", {"status": "invalid-comparison"})
    rows = [a, b, c] + ([native] if native is not None else [])
    checks = [validate_run(row) for row in rows]
    if any(not x["timing_eligible"] for x in checks):
        return {**result, "reasons": sorted({r for x in checks for r in x["reasons"]})}
    ac, bc, cc = (mapping(row.get("contract")) for row in (a, b, c))
    errors = set()
    if (
        a.get("mode") != "matched-resident"
        or b.get("mode") != "offload"
        or c.get("mode") != "offload"
    ):
        errors.add("execution-mode-mismatch")
    if any(
        any(ac.get(k) != mapping(row.get("contract")).get(k) for k in PAIR_FIELDS)
        for row in rows[1:]
    ):
        errors.add("cohort-mismatch")
    if any(a.get("source_hashes") != row.get("source_hashes") for row in rows[1:]):
        errors.add("source-mismatch")
    if a.get("engine_policy") != b.get("engine_policy") or b.get(
        "engine_policy"
    ) != c.get("engine_policy"):
        errors.add("engine-policy-mismatch")
    if bc.get("expert_cache") != cc.get("expert_cache"):
        errors.add("cache-profile-mismatch")
    if mapping(ac.get("expert_cache")).get("profile_sha256") != mapping(
        bc.get("expert_cache")
    ).get("profile_sha256"):
        errors.add("profile-mismatch")
    ak, bk, ck = (mapping(row.get("actual_kv")) for row in (a, b, c))
    if (
        any(
            ak.get(k) != bk.get(k)
            for k in ("allocated_bytes_per_rank", "num_gpu_blocks")
        )
        or ck["allocated_bytes_per_rank"] <= bk["allocated_bytes_per_rank"]
    ):
        errors.add("actual-kv-pairing")
    if any(
        mapping(row.get("smoke")).get("output_sha256")
        != mapping(a.get("smoke")).get("output_sha256")
        or mapping(row.get("smoke")).get("input_sha256")
        != mapping(a.get("smoke")).get("input_sha256")
        for row in rows[1:]
    ):
        errors.add("smoke-mismatch")
    if native is not None and (
        native.get("mode") != "native"
        or mapping(native.get("contract")).get("gpu_memory_utilization") != 0.9
    ):
        errors.add("native-reference-policy")
    if errors:
        return {**result, "reasons": sorted(errors)}
    at, bt, ct = (x["elapsed_median_s"] for x in checks[:3])
    result.update(
        status="measured",
        reasons=[],
        offload_overhead_s=bt - at,
        kv_recovery_s=bt - ct,
        matched_net_ratio=at / ct,
        paired_repetitions=[
            {
                "repetition": i,
                "matched_net_ratio": a["repetitions"][i]["elapsed_s"]
                / c["repetitions"][i]["elapsed_s"],
            }
            for i in range(len(a["repetitions"]))
        ],
        interpretation="whole-generation wallclock including prefill; implementation overhead, not pure miss time",
    )
    if native is not None:
        nt = checks[3]["elapsed_median_s"]
        result.update(
            native_reference_ratio=nt / ct, native_to_matched_elapsed_delta_s=at - nt
        )
        if all(
            mapping(native.get("actual_kv")).get(k) == ak.get(k)
            for k in ("allocated_bytes_per_rank", "num_gpu_blocks")
        ):
            result.update(
                engine_mode_tax_s=at - nt, engine_mode_tax_status="same-actual-kv"
            )
        else:
            result["engine_mode_tax_status"] = "confounded-kv"
    return result


def read_artifact(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        text = stream.read(64 * 1024 * 1024 + 1)
    if len(text) > 64 * 1024 * 1024:
        raise ValueError("artifact-too-large")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("artifact-root")
    return value


def build_plan(
    *,
    stage: str,
    run_id: str,
    model_path: str = MODEL,
    dataset_path: str = DATA,
    dataset_manifest: str = MANIFEST,
    context_length: int = 1024,
    profile_path: str | None = None,
    kv_bytes: Sequence[int] = (),
    batches: Sequence[int] | None = None,
    output_length: int = 512,
    repetitions: int = 3,
    seed: int = 20260912,
    max_num_seqs: int = 1024,
    max_num_batched_tokens: int = 8192,
    timeout_s: int = 7200,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,55}", run_id) or ".." in run_id:
        raise ValueError("safe fresh run ID required")
    if stage in {"kv", "offload", "overhead"} and (
        not kv_bytes or any(not integer(x, 1) for x in kv_bytes)
    ):
        raise ValueError("explicit actual KV budgets from baseline inspection required")
    if stage == "kv" and (len(kv_bytes) < 2 or list(kv_bytes) != sorted(set(kv_bytes))):
        raise ValueError("at least two increasing actual KV budgets required")
    if (stage == "offload" or (stage == "coverage" and context_length == 4096)) and (
        not batches or len(batches) > 3
    ):
        raise ValueError("provide at most three selected reached batches")
    if batches is None:
        batches = (1, 16, 50, 100, 200, 500, 1000)
    if not batches or any(not integer(x, 1) or x > 1000 for x in batches):
        raise ValueError("bounded batches required")
    commands = []

    def command(
        mode: str, suffix: str, context: int, extra: Sequence[str], *, total: int = 1024
    ) -> None:
        profile = (
            profile_path or f"runs/decode-mechanism/{run_id}-cal-{context}/profile.json"
        )
        argv = [
            "bash",
            "scripts/server/run_decode_mechanism.sh",
            mode,
            "--run-id",
            f"{run_id}-{suffix}",
            "--timeout-s",
            str(timeout_s),
            "--model-path",
            model_path,
            "--dataset-path",
            dataset_path,
            "--dataset-manifest",
            dataset_manifest,
            "--context-length",
            str(context),
            "--output-length",
            str(output_length),
            "--batch-size",
            str(total),
            "--repetitions",
            str(repetitions),
            "--seed",
            str(seed),
            "--max-num-seqs",
            str(max_num_seqs),
            "--max-num-batched-tokens",
            str(max_num_batched_tokens),
            "--gpu-memory-utilization",
            "0.90",
            "--calibration-count",
            "32",
        ]
        if mode != "calibrate":
            argv += ["--profile-path", profile]
        argv += list(extra)
        commands.append({"argv": argv, "shell": "GPU_IDS=0,1,2,3 " + shlex.join(argv)})

    if stage == "baseline":
        for context in (1024, 4096):
            command("calibrate", f"cal-{context}", context, [])
        command("native", f"native-{context_length}", context_length, [])
    elif stage == "coverage":
        for batch in batches:
            command(
                "matched-resident",
                f"coverage-{context_length}-{batch}",
                context_length,
                ["--profile", "--target-batch", str(batch)],
                total=batch,
            )
    elif stage == "offload":
        for batch in batches:
            for ratio, slots in (("0.8", "512"), ("0.8", "2048"), ("0.9", "512")):
                command(
                    "offload",
                    f"miss-{context_length}-{batch}-{ratio}-{slots}",
                    context_length,
                    [
                        "--profile",
                        "--target-batch",
                        str(batch),
                        "--kv-bytes",
                        str(kv_bytes[0]),
                        "--resident-ratio",
                        ratio,
                        "--cache-slots",
                        slots,
                        "--cache-policy",
                        "decayed-lfu",
                    ],
                    total=batch,
                )
    elif stage == "kv":
        command(
            "matched-resident",
            f"a-{context_length}-{kv_bytes[0]}",
            context_length,
            ["--kv-bytes", str(kv_bytes[0])],
        )
        for budget in kv_bytes:
            command(
                "offload",
                f"bc-{context_length}-{budget}",
                context_length,
                [
                    "--kv-bytes",
                    str(budget),
                    "--resident-ratio",
                    "0.8",
                    "--cache-slots",
                    "512",
                    "--cache-policy",
                    "decayed-lfu",
                ],
            )
    elif stage == "overhead":
        for profiled in (False, True):
            command(
                "matched-resident",
                f"overhead-{context_length}-{int(profiled)}",
                context_length,
                ["--kv-bytes", str(kv_bytes[0])] + (["--profile"] if profiled else []),
            )
    else:
        raise ValueError("unknown manual stage")
    return diagnostic_artifact(
        "decode-plan",
        {
            "stage": stage,
            "commands": commands,
            "auto_execute": False,
            "next_stage_requires": "inspect actual per-rank KV and reached batches; supply --kv-bytes and selected --batches; use separate profiles for each context",
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subs = parser.add_subparsers(dest="command", required=True)
    plan = subs.add_parser("plan", allow_abbrev=False)
    plan.add_argument(
        "--stage",
        choices=("baseline", "coverage", "offload", "kv", "overhead"),
        required=True,
    )
    plan.add_argument("--run-id", required=True)
    for name, default in (
        ("model-path", MODEL),
        ("dataset-path", DATA),
        ("dataset-manifest", MANIFEST),
        ("profile-path", None),
    ):
        plan.add_argument("--" + name, default=default)
    for name, int_default in (
        ("context-length", 1024),
        ("output-length", 512),
        ("repetitions", 3),
        ("seed", 20260912),
        ("max-num-seqs", 1024),
        ("max-num-batched-tokens", 8192),
        ("timeout-s", 7200),
    ):
        plan.add_argument("--" + name, type=int, default=int_default)
    plan.add_argument("--kv-bytes", nargs="+", type=int, default=[])
    plan.add_argument("--batches", nargs="+", type=int)
    for name in ("validate", "summarize", "export"):
        p = subs.add_parser(name, allow_abbrev=False)
        p.add_argument("--source", type=Path, required=True)
        if name == "export":
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--b", type=Path)
            p.add_argument("--c", type=Path)
            p.add_argument("--native", type=Path)
    compare = subs.add_parser("compare", allow_abbrev=False)
    for name in ("a", "b", "c", "native"):
        compare.add_argument("--" + name, type=Path, required=name != "native")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "plan":
            result = build_plan(**args)
        elif command == "compare":
            loaded = {
                k: read_artifact(v / "summary.json" if v.is_dir() else v)
                if v is not None
                else {}
                for k, v in args.items()
            }
            result = compare_runs(
                loaded["a"], loaded["b"], loaded["c"], loaded["native"] or None
            )
        elif command == "validate":
            source = args["source"]
            result = validate_run(
                read_artifact(source / "summary.json" if source.is_dir() else source)
            )
        else:
            from .decode_report import export_report, summarize_directory

            source = args["source"]
            if command == "summarize":
                result = summarize_directory(source)
            else:
                comparison_paths = [
                    args[k] for k in ("b", "c", "native") if args.get(k) is not None
                ]
                if comparison_paths and (
                    args.get("b") is None or args.get("c") is None
                ):
                    raise ValueError("comparison export requires both B and C")
                result = export_report(
                    source, args["output"], comparisons=comparison_paths
                )
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(
            json.dumps(
                {
                    "status": "invalid-input",
                    "reason": "io-error"
                    if isinstance(error, OSError)
                    else "invalid-contract",
                }
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
