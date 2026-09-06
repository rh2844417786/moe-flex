"""Stdlib-only strict numerical validation for diagnostic native KV experiments."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any

ROLES = ("r0", "small", "large")
LABELS = {
    "comparison_backend": "kv-oracle",
    "storage_backend": "native",
    "diagnostic_only": True,
    "formal_offload_gain": False,
}
FAILURE_CODES = (
    "safety-refusal",
    "oom",
    "timeout",
    "invalid-evidence",
    "interrupted",
    "runtime-error",
    "launcher-failed",
    "malformed-or-missing",
)
FAILURE_STAGES = ("budget-construction", "native-execution", "launcher", "export")


class SafetyRefusal(ValueError):
    """An estimated or observed memory boundary refuses this immutable point."""


def failure_code(error: BaseException) -> str:
    if isinstance(error, SafetyRefusal):
        return "safety-refusal"
    if (
        type(error).__name__ == "OutOfMemoryError"
        or "out of memory" in str(error).lower()
    ):
        return "oom"
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    if isinstance(error, (ValueError, TypeError, KeyError)):
        return "invalid-evidence"
    return "runtime-error"


HASH_FIELDS = (
    "model_identity_sha256",
    "model_config_sha256",
    "input_sha256",
    "dataset_sha256",
    "dataset_manifest_sha256",
    "engine_policy_sha256",
)
CONTRACT_INTS = (
    "batch_size",
    "context_length",
    "output_length",
    "tensor_parallel_size",
    "seed",
    "warmups",
    "timing_samples",
    "max_num_seqs",
    "max_num_batched_tokens",
    "repetitions_requested",
    "smoke_output_length",
    "source_request_count",
    "unique_selected_request_count",
    "repeated_request_count",
)
MEMORY_INTS = (
    "rank",
    "total_gpu_bytes",
    "free_gpu_bytes",
    "torch_allocated_bytes",
    "torch_reserved_bytes",
    "torch_peak_allocated_bytes",
    "torch_peak_reserved_bytes",
    "kv_cache_allocated_bytes",
    "kv_cache_declared_bytes",
    "num_gpu_blocks",
)
PROBE_FIELDS = (
    "rank",
    "forward_calls",
    "forward_telemetry_available",
    "request_telemetry_available",
    "request_observations",
    "request_sum",
    "request_peak",
    "request_mean",
    "parameter_count",
    "all_parameters_cuda",
    "kv_occupancy",
    "preemptions",
)
REP_FIELDS = (
    "repetition",
    "request_count",
    "generated_tokens",
    "elapsed_s",
    "output_tokens_per_second",
    "output_sha256",
    "input_sha256",
)


def integer(value: Any, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("invalid integer evidence")
    return value


def number(value: Any, minimum: float = 0) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError("invalid numeric evidence")
    return float(value)


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}", value
    ):
        raise ValueError("invalid run identifier")
    return value


def hash_value(value: Any, length: int = 64) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        "[a-f0-9]{" + str(length) + "}", value
    ):
        raise ValueError("invalid hash evidence")
    return value


def rank_rows(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError("four rank rows required")
    if any(not isinstance(row, dict) for row in raw):
        raise ValueError("invalid rank row")
    if sorted(integer(row.get("rank")) for row in raw) != [0, 1, 2, 3]:
        raise ValueError("rank coverage differs")
    return sorted(raw, key=lambda row: row["rank"])


def estimated_peak(row: Mapping[str, Any]) -> int:
    used = integer(row["total_gpu_bytes"]) - integer(row["free_gpu_bytes"])
    non_torch = max(0, used - integer(row["torch_reserved_bytes"]))
    return max(used, integer(row["torch_peak_reserved_bytes"]) + non_torch)


def validate_memory(raw: Any, fraction: float, margin: int) -> int:
    capacities = []
    for row in rank_rows(raw):
        for name in MEMORY_INTS:
            integer(row.get(name))
        total = integer(row["total_gpu_bytes"], 1)
        kv = integer(row["kv_cache_allocated_bytes"], 1)
        integer(row["num_gpu_blocks"], 1)
        if (
            row.get("kv_cache_accounting_consistent") is not True
            or kv != row["kv_cache_declared_bytes"]
        ):
            raise ValueError("actual KV accounting inconsistent")
        allocated, reserved = row["torch_allocated_bytes"], row["torch_reserved_bytes"]
        peak, peak_reserved = (
            row["torch_peak_allocated_bytes"],
            row["torch_peak_reserved_bytes"],
        )
        if not kv <= allocated <= reserved <= peak_reserved <= total:
            raise ValueError("invalid allocator accounting")
        if not allocated <= peak <= peak_reserved or row["free_gpu_bytes"] > total:
            raise ValueError("invalid peak/free accounting")
        used = estimated_peak(row)
        if used + margin > total * fraction:
            raise SafetyRefusal("observed safety boundary exceeded")
        capacities.append(kv)
    if len(set(capacities)) != 1:
        raise ValueError("rank KV capacities differ")
    return capacities[0]


def validate_probes(raw: Any, max_requests: int | None = None) -> bool:
    known = True
    for row in rank_rows(raw):
        if any(name not in row for name in PROBE_FIELDS):
            raise ValueError("probe evidence incomplete")
        integer(row["parameter_count"], 1)
        if row["all_parameters_cuda"] is not True:
            raise ValueError("native parameters are not entirely CUDA resident")
        for name in ("forward_telemetry_available", "request_telemetry_available"):
            if type(row[name]) is not bool:
                raise ValueError("invalid availability flag")
        if row["forward_telemetry_available"]:
            integer(row["forward_calls"], 1)
        elif row["forward_calls"] is not None:
            raise ValueError("unavailable forward count must be null")
        else:
            known = False
        observations = integer(row["request_observations"])
        if row["request_telemetry_available"]:
            integer(observations, 1)
            count = integer(row["request_sum"], observations)
            peak = integer(row["request_peak"], 1)
            if row["forward_calls"] != observations or count > peak * observations:
                raise ValueError("inconsistent request observations")
            if max_requests is not None and peak > max_requests:
                raise ValueError("observed requests exceed configured maximum")
            if not math.isclose(
                number(row["request_mean"]), count / observations, rel_tol=1e-9
            ):
                raise ValueError("request mean differs")
        else:
            if any(
                row[name] is not None
                for name in ("request_sum", "request_peak", "request_mean")
            ):
                raise ValueError("unavailable request telemetry must be null")
        if row["kv_occupancy"] is not None or row["preemptions"] is not None:
            raise ValueError("unsupported occupancy/preemption evidence")
    return known


def validate_sample(row: Any, requests: int, length: int) -> None:
    if integer(row.get("request_count"), 1) != requests:
        raise ValueError("request count differs")
    if integer(row.get("generated_tokens"), 1) != requests * length:
        raise ValueError("fixed token count differs")
    seconds = number(row.get("elapsed_s"))
    if seconds <= 0 or not math.isclose(
        number(row.get("output_tokens_per_second")),
        requests * length / seconds,
        rel_tol=1e-9,
    ):
        raise ValueError("tokens/seconds differ")
    hash_value(row.get("output_sha256"))


def validate_run(row: Mapping[str, Any]) -> int:
    if any(
        row.get(key) != value or type(row.get(key)) is not type(value)
        for key, value in LABELS.items()
    ):
        raise ValueError("Oracle diagnostic labels missing")
    identifier(row.get("run_id"))
    if row.get("status") != "complete" or row.get("role") not in ROLES:
        raise ValueError("completed Oracle role required")
    if (
        row.get("arm") != "resident"
        or integer(row.get("offload_count")) != 0
        or row.get("offload_layers") != []
    ):
        raise ValueError("Oracle must use native resident weights")
    contract = row["contract"]
    for key in HASH_FIELDS:
        hash_value(contract.get(key))
    hash_value(contract.get("commit"), 40)
    for key in CONTRACT_INTS:
        integer(contract.get(key))
    for key in (
        "batch_size",
        "context_length",
        "output_length",
        "repetitions_requested",
        "max_num_seqs",
        "max_num_batched_tokens",
        "smoke_output_length",
    ):
        integer(contract[key], 1)
    if (
        contract.get("comparison_backend") != "kv-oracle"
        or contract.get("dtype") != "bfloat16"
    ):
        raise ValueError("native BF16 contract required")
    if (
        contract["tensor_parallel_size"] != 4
        or contract.get("gpu_memory_utilization") != 0.60
    ):
        raise ValueError("TP4 / nominal r0 0.60 required")
    if (
        contract["warmups"] < 1
        or contract["smoke_output_length"] > contract["output_length"]
    ):
        raise ValueError("warmup/smoke contract differs")
    if (
        contract["unique_selected_request_count"] + contract["repeated_request_count"]
        != contract["batch_size"]
        or integer(contract["source_request_count"], 1)
        < contract["unique_selected_request_count"]
        or contract["unique_selected_request_count"]
        != min(contract["source_request_count"], contract["batch_size"])
    ):
        raise ValueError("input repetition accounting differs")
    versions = contract["versions"]
    for key in ("torch", "vllm", "cuda"):
        if not isinstance(versions.get(key), str) or not re.fullmatch(
            r"[0-9][0-9A-Za-z.+_-]{0,63}", versions[key]
        ):
            raise ValueError("software version unavailable")
    if versions["vllm"].split("+")[0] != "0.10.2":
        raise ValueError("pinned vLLM required")
    hash_value(versions.get("vllm_commit"), 40)
    fraction = number(row.get("safety_fraction"))
    if not 0 < fraction <= 1:
        raise ValueError("invalid safety fraction")
    margin = integer(row.get("safety_margin_bytes"))
    small = integer(row.get("small_add_bytes"), 1)
    if integer(row.get("large_add_bytes"), 1) <= small:
        raise ValueError("large increment must exceed small")
    kv = validate_memory(row.get("memory"), fraction, margin)

    def same_memory(raw: Any) -> None:
        if validate_memory(raw, fraction, margin) != kv:
            raise ValueError("actual KV changed within point")
        for old, new in zip(rank_rows(row["memory"]), rank_rows(raw), strict=True):
            if any(
                old[key] != new[key] for key in ("total_gpu_bytes", "num_gpu_blocks")
            ):
                raise ValueError("GPU/blocks changed within point")

    same_memory(row.get("final_memory"))
    if row["role"] == "r0":
        if (
            row.get("requested_kv_cache_bytes") is not None
            or row.get("baseline_kv_bytes") is not None
        ):
            raise ValueError("r0 must use automatic KV")
    else:
        baseline = integer(row.get("baseline_kv_bytes"), 1)
        target = baseline + row[row["role"] + "_add_bytes"]
        if integer(row.get("requested_kv_cache_bytes"), 1) != target:
            raise ValueError("explicit KV target differs")
        # Allocation is block-aligned; accept only less than one allocation block.
        for memory in rank_rows(row["memory"]):
            block = kv / memory["num_gpu_blocks"]
            if not baseline < kv <= target or target - kv >= block:
                raise ValueError("actual KV increment differs from aligned target")
    validate_sample(row["smoke"], 1, contract["smoke_output_length"])
    hash_value(row["smoke"].get("input_sha256"))
    reps = row.get("repetitions")
    if not isinstance(reps, list) or len(reps) != contract["repetitions_requested"]:
        raise ValueError("repetitions incomplete")
    if integer(row.get("repetitions_completed")) != len(reps):
        raise ValueError("completed repetition count differs")
    for index, rep in enumerate(reps):
        if integer(rep.get("repetition")) != index:
            raise ValueError("repetition sequence differs")
        validate_sample(rep, contract["batch_size"], contract["output_length"])
        same_memory(rep.get("memory"))
        validate_probes(
            rep.get("native_probe"),
            min(contract["max_num_seqs"], contract["batch_size"]),
        )
    validate_probes(
        row.get("native_probe"), min(contract["max_num_seqs"], contract["batch_size"])
    )
    return kv


def validate_anchor(anchor: Mapping[str, Any], current: Mapping[str, Any]) -> int:
    k0 = validate_run(anchor)
    if anchor["role"] != "r0":
        raise ValueError("only completed r0 may anchor capacity")
    left = {k: v for k, v in anchor["contract"].items() if k != "repetitions_requested"}
    right = {
        k: v for k, v in current["contract"].items() if k != "repetitions_requested"
    }
    if left != right:
        raise ValueError("anchor model/input/SHA/engine contract differs")
    if current.get("role") == "r0" and validate_run(current) != k0:
        raise ValueError("fresh reverse r0 K0 differs from anchor")
    if [r["total_gpu_bytes"] for r in rank_rows(anchor["memory"])] != [
        r["total_gpu_bytes"] for r in rank_rows(current["memory"])
    ]:
        raise ValueError("anchor hardware differs")
    return k0


def analyze(
    rows: Sequence[Mapping[str, Any]], anchor: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        **LABELS,
        "status": "incomplete",
        "decision": "insufficient-evidence",
    }
    if len(rows) != 3 or rows[0].get("status") != "complete":
        return result
    try:
        if [row.get("role") for row in rows] != list(ROLES):
            raise ValueError("three distinct Oracle roles required")
        r0 = rows[0]
        capacities = [validate_run(r0)]
        if anchor is not None:
            validate_anchor(anchor, r0)
        complete = [row for row in rows[1:] if row.get("status") == "complete"]
        for row in complete:
            capacities.append(validate_run(row))
            validate_anchor(r0, row)
            if (
                row["contract"] != r0["contract"]
                or row["baseline_kv_bytes"] != capacities[0]
            ):
                raise ValueError("comparison contract/K0 differs")
            for key in (
                "small_add_bytes",
                "large_add_bytes",
                "safety_fraction",
                "safety_margin_bytes",
            ):
                if row[key] != r0[key]:
                    raise ValueError("diagnostic budget/safety configuration differs")
            if any(
                row["smoke"][key] != r0["smoke"][key]
                for key in ("input_sha256", "output_sha256")
            ):
                raise ValueError("batch1 smoke differs")
        baseline = [rep["elapsed_s"] for rep in r0["repetitions"]]
        repeatable = len(baseline) >= 3
        mechanisms: list[bool | None] = []
        for row in complete:
            seconds = [rep["elapsed_s"] for rep in row["repetitions"]]
            ratio = median(baseline) / median(seconds)
            separated = max(seconds) < min(baseline) or min(seconds) > max(baseline)
            repeatable = repeatable and separated
            forward_known = all(
                p["forward_telemetry_available"]
                for run in (r0, row)
                for rep in run["repetitions"]
                for p in rep["native_probe"]
            )
            request_known = all(
                p["request_telemetry_available"]
                for run in (r0, row)
                for rep in run["repetitions"]
                for p in rep["native_probe"]
            )

            def counter(run: Mapping[str, Any], key: str) -> float:
                return float(
                    median(
                        sum(p[key] for p in rep["native_probe"])
                        for rep in run["repetitions"]
                    )
                )

            mechanism = (
                forward_known
                and counter(row, "forward_calls") < counter(r0, "forward_calls")
            ) or (
                request_known
                and counter(row, "request_mean") > counter(r0, "request_mean")
            )
            mechanisms.append(
                True
                if mechanism
                else False
                if forward_known and request_known
                else None
            )
            result[row["role"]] = {
                "throughput_ratio": ratio,
                "elapsed_median_s": median(seconds),
                "elapsed_range_s": [min(seconds), max(seconds)],
                "time_savings_s": [
                    a - b for a, b in zip(baseline, seconds, strict=True)
                ],
                "mechanism_improved": mechanisms[-1],
                "ranges_separated": separated,
                "actual_kv_bytes_per_rank": row["memory"][0][
                    "kv_cache_allocated_bytes"
                ],
            }
        if len(complete) != 2:
            return result
        result.update(
            status="confirmed-signal" if repeatable else "screening-only",
            r0_elapsed_median_s=median(baseline),
            r0_elapsed_range_s=[min(baseline), max(baseline)],
            order_check="reverse-anchor-checked" if anchor is not None else "forward",
            correctness_scope="batch1-smoke-only",
        )
        if repeatable:
            small, large = (
                result["small"]["throughput_ratio"],
                result["large"]["throughput_ratio"],
            )
            if small >= 1.2 and mechanisms[0] is True:
                result["decision"] = "consider-small-offload"
            elif small < 1.1 and large < 1.1 and mechanisms == [False, False]:
                result["decision"] = "stop-throughput-route"
            elif small >= 1.1 and mechanisms[0] is not None:
                result["decision"] = "low-cost-only"
            elif large >= 1.1:
                result["decision"] = "large-only-not-small-offload-evidence"
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        result.update(status="invalid", decision="insufficient-evidence")
    return result


def public_run(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only known numeric/hash fields; never export paths, prompts or errors."""
    if raw.get("status") == "complete":
        validate_run(raw)

    def fields(source: Any, names: Sequence[str]) -> dict[str, Any]:
        if not isinstance(source, Mapping):
            raise TypeError("invalid public evidence object")
        out: dict[str, Any] = {}
        for name in names:
            if name not in source:
                continue
            value = source[name]
            if name.endswith("sha256"):
                out[name] = hash_value(value)
            elif value is None or type(value) is bool:
                out[name] = value
            else:
                number(value)
                out[name] = value
        return out

    def memories(source: Any) -> list[dict[str, Any]]:
        if not isinstance(source, list):
            raise TypeError("invalid memory list")
        return [
            fields(
                row,
                (
                    *MEMORY_INTS,
                    "available_kv_cache_bytes",
                    "model_memory_bytes",
                    "kv_cache_accounting_consistent",
                ),
            )
            for row in source
        ]

    def probes(source: Any) -> list[dict[str, Any]]:
        if not isinstance(source, list):
            raise TypeError("invalid probe list")
        return [fields(row, PROBE_FIELDS) for row in source]

    result: dict[str, Any] = {
        **LABELS,
        "schema_version": 1,
        "run_id": identifier(raw.get("run_id")),
        "role": raw.get("role") if raw.get("role") in ROLES else "unknown",
        "arm": "resident",
        "status": raw.get("status")
        if raw.get("status")
        in ("complete", "running", "failed", "timeout", "interrupted")
        else "failed",
        **fields(
            raw,
            (
                "offload_count",
                "baseline_kv_bytes",
                "requested_kv_cache_bytes",
                "small_add_bytes",
                "large_add_bytes",
                "safety_fraction",
                "safety_margin_bytes",
                "repetitions_completed",
                "exit_code",
                "launcher_elapsed_s",
                "smoke_matches_resident",
                "nominal_r0_gpu_memory_utilization",
                "obeys_nominal_r0_budget",
            ),
        ),
    }
    layers = raw.get("offload_layers", [])
    if not isinstance(layers, list) or any(type(x) is not int or x < 0 for x in layers):
        raise ValueError("invalid offload layers")
    result["offload_layers"] = list(layers)
    for name, allowed in (
        ("failure_code", FAILURE_CODES),
        ("failure_stage", FAILURE_STAGES),
    ):
        if raw.get(name) in allowed:
            result[name] = raw[name]
    source = raw.get("contract", {})
    contract = fields(source, (*CONTRACT_INTS, *HASH_FIELDS, "gpu_memory_utilization"))
    if "commit" in source:
        contract["commit"] = hash_value(source["commit"], 40)
    for key, allowed_contract_values in {
        "dtype": ("bfloat16",),
        "comparison_backend": ("kv-oracle",),
        "sampling_policy": ("existing-prompts", "repeated-existing-prompts"),
    }.items():
        if source.get(key) in allowed_contract_values:
            contract[key] = source[key]
    versions = source.get("versions", {})
    contract["versions"] = {}
    for key in ("torch", "vllm", "cuda", "vllm_commit"):
        value = versions.get(key)
        if isinstance(value, str) and re.fullmatch(
            r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,63}", value
        ):
            contract["versions"][key] = value
    result["contract"] = contract
    for key in ("memory", "final_memory"):
        result[key] = memories(raw.get(key, []))
    result["native_probe"] = probes(raw.get("native_probe", []))
    result["smoke"] = fields(raw.get("smoke", {}), REP_FIELDS)
    result["repetitions"] = [
        dict(
            fields(rep, REP_FIELDS),
            memory=memories(rep.get("memory", [])),
            native_probe=probes(rep.get("native_probe", [])),
        )
        for rep in raw.get("repetitions", [])
    ]
    if result["status"] == "complete":
        validate_run(result)
    return result
