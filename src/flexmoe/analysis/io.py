"""Standard-library file adapters. No GPU runner or flexmoe root import."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any, cast

from .schema import (
    DemandTrace,
    ReplayResult,
    TimingPoint,
    TransferSample,
    diagnostic_artifact,
    parse_diagnostic_artifact,
    validate_contract,
    validate_transfer_contract,
)


def read_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        value = json.load(
            stream,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError("nonfinite JSON number")
            ),
        )
    if not isinstance(value, dict):
        raise TypeError("JSON root must be an object")
    return value


def atomic_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".analysis-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def evidence_validator() -> Any:
    """Load the existing independent stdlib validator, never its GPU frontend."""
    path = Path(__file__).parents[1] / "bench/kv_oracle_evidence.py"
    spec = importlib.util.spec_from_file_location("_analysis_oracle_evidence", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Oracle evidence validator unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_native_policy(
    policy: Mapping[str, Any], contract: Mapping[str, Any], mode: str
) -> None:
    """Check saved resolved settings independently of the GPU producer and hash."""
    eager = mode == "eager"
    required = {
        "model_config": {
            "enforce_eager": eager,
            "quantization": None,
            "max_model_len": contract["context_length"] + contract["output_length"],
            "seed": contract["seed"],
        },
        "cache_config": {
            "enable_prefix_caching": False,
            "cpu_offload_gb": 0.0,
            "swap_space_bytes": 0,
            "gpu_memory_utilization": contract["gpu_memory_utilization"],
        },
        "scheduler_config": {
            "max_num_seqs": min(contract["batch_size"], contract["max_num_seqs"]),
            "max_num_batched_tokens": contract["max_num_batched_tokens"],
            "enable_chunked_prefill": True,
        },
        "parallel_config": {
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "disable_custom_all_reduce": eager,
        },
    }
    if eager:
        # _capture_policy serializes the resolved custom_ops list as a string.
        required["compilation_config"] = {"level": 0, "custom_ops": "['all']"}
    for group, fields in required.items():
        resolved = policy.get(group)
        if not isinstance(resolved, dict):
            raise TypeError(f"resolved policy group missing: {group}")
        for name, expected in fields.items():
            value = resolved.get(name)
            valid_type = (
                type(value) in (int, float)
                if type(expected) is float
                else type(value) is type(expected)
            )
            if name not in resolved or not valid_type or value != expected:
                raise ValueError(f"resolved policy differs: {group}.{name}")
    if policy["model_config"].get("dtype") not in ("torch.bfloat16", "bfloat16"):
        raise ValueError("resolved native BF16 weights required")
    if not eager:
        compilation = policy.get("compilation_config")
        if not isinstance(compilation, dict) or any(
            type(compilation.get(name)) not in (int, str) or compilation[name] == ""
            for name in ("level", "cudagraph_mode")
        ):
            raise ValueError("resolved native compilation/graph state unavailable")


def native_timing(raw: Mapping[str, Any]) -> TimingPoint:
    fields = parse_diagnostic_artifact(raw, "native-summary")
    v = evidence_validator()
    v.integer(fields.get("offload_count"))
    if (
        fields.get("status") != "complete"
        or fields.get("arm") != "resident"
        or fields.get("offload_count") != 0
        or fields.get("offload_layers") != []
        or fields.get("timing_eligible") is not True
        or fields.get("engine_mode") not in ("native", "eager")
        or fields.get("source_kind") != "native-measured"
        or fields.get("measurement_evidence") != "measured"
        or fields.get("failed_measurement") is not None
    ):
        raise ValueError("complete native/eager resident timing evidence required")
    contract = validate_contract(fields.get("contract"), require_prompt_hashes=True)
    requests = v.integer(contract["batch_size"], 1)
    length = v.integer(contract["output_length"], 1)
    repetitions = v.integer(contract.get("repetitions_requested"), 1)
    unique = v.integer(contract.get("unique_selected_request_count"), 1)
    repeated = v.integer(contract.get("repeated_request_count"))
    if unique + repeated != requests or unique != len(
        cast(tuple[str, ...], contract["prompt_hashes"])
    ):
        raise ValueError("selected prompt identity/count differs")
    v.integer(contract.get("seed"))
    v.integer(contract.get("warmups"), 1)
    if contract.get("dtype") != "bfloat16":
        raise ValueError("native BF16 weights required")
    versions = contract["versions"]
    assert isinstance(versions, dict)
    for name in ("torch", "vllm", "cuda"):
        if not isinstance(versions.get(name), str) or not versions[name]:
            raise ValueError("software evidence missing")
    if str(versions["vllm"]).split("+")[0] != "0.10.2":
        raise ValueError("pinned vLLM required")
    v.hash_value(versions.get("vllm_commit"), 40)
    policy = fields.get("engine_policy")
    if not isinstance(policy, dict):
        raise TypeError("resolved engine policy required")
    digest = hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != contract["engine_policy_sha256"]:
        raise ValueError("engine policy hash differs")
    _validate_native_policy(policy, contract, str(fields["engine_mode"]))
    reserve = v.integer(fields.get("physical_safety_reserve_bytes"))
    memory = fields.get("memory")
    v.validate_memory(memory, 1.0, reserve)
    initial = v.rank_rows(memory)

    def same_memory(value: Any) -> None:
        v.validate_memory(value, 1.0, reserve)
        for before, after in zip(initial, v.rank_rows(value), strict=True):
            if any(
                before[key] != after[key]
                for key in (
                    "total_gpu_bytes",
                    "kv_cache_allocated_bytes",
                    "num_gpu_blocks",
                )
            ):
                raise ValueError("physical GPU/KV/blocks changed within timing point")

    same_memory(fields.get("final_memory"))
    smoke = parse_diagnostic_artifact(fields.get("smoke"), "analysis-smoke")
    if smoke.get("timing_eligible") is not True:
        raise ValueError("smoke timing label differs")
    v.validate_sample(smoke, 1, v.integer(contract.get("smoke_output_length"), 1))
    reps = fields.get("repetitions")
    if not isinstance(reps, list) or len(reps) != repetitions:
        raise ValueError("repetitions incomplete")
    if v.integer(fields.get("repetitions_completed")) != repetitions:
        raise ValueError("completed repetitions differ")
    times = []
    rates = []
    for index, raw_rep in enumerate(reps):
        rep = parse_diagnostic_artifact(raw_rep, "analysis-repetition")
        if (
            rep.get("timing_eligible") is not True
            or v.integer(rep.get("repetition")) != index
            or rep.get("measurement_status") == "rejected"
        ):
            raise ValueError("repetition eligibility differs")
        v.validate_sample(rep, requests, length)
        same_memory(rep.get("memory"))
        times.append(v.number(rep["elapsed_s"]))
        rates.append(v.number(rep["output_tokens_per_second"]))
    for key, expected in (
        ("throughput_median", median(rates)),
        ("throughput_min", min(rates)),
        ("throughput_max", max(rates)),
    ):
        if not math.isclose(v.number(fields.get(key)), expected, rel_tol=1e-9):
            raise ValueError("summary throughput differs from repetitions")
    return TimingPoint(
        v.identifier(fields.get("run_id")),
        contract,
        requests * length,
        tuple(times),
        tuple(row["kv_cache_allocated_bytes"] for row in initial),
        tuple(row["total_gpu_bytes"] for row in initial),
        "native-measured",
        str(fields["engine_mode"]),
    )


def oracle_adapter(raw: Mapping[str, Any]) -> dict[str, Any]:
    v = evidence_validator()
    v.validate_run(raw)
    # Legacy runs do not record prompt hashes or the four-card hardware identity.
    # Keep the validated original capacity observations; never invent these fields.
    missing = [
        name
        for name in ("hardware_sha256", "prompt_hashes")
        if not raw["contract"].get(name)
    ]
    return diagnostic_artifact(
        "oracle-adaptation",
        {
            "status": "incomplete",
            "source_kind": "kv-oracle-diagnostic",
            "scope": "over-budget-diagnostic-counterfactual",
            "timing_point": None,
            "missing_evidence": ["legacy-" + name.replace("_", "-") for name in missing]
            + ["matching-native-reference-required"],
            "kv_bytes_by_rank": [
                row["kv_cache_allocated_bytes"] for row in v.rank_rows(raw["memory"])
            ],
            "elapsed_s": [row["elapsed_s"] for row in raw["repetitions"]],
        },
    )


def load_timing(path: Path) -> TimingPoint | None:
    raw = read_json(path)
    if raw.get("comparison_backend") == "kv-oracle":
        oracle_adapter(raw)
        return None
    return native_timing(raw)


def load_trace(path: Path) -> DemandTrace:
    fields = parse_diagnostic_artifact(read_json(path), "demand-trace")
    if fields.get("timing_eligible") is not False or fields.get("trace") is None:
        raise ValueError("trace capture missing or timing eligibility forged")
    trace = DemandTrace.from_dict(fields["trace"])
    capture = fields.get("capture")
    if trace.full_workload and (
        fields.get("generation_status") != "complete"
        or not isinstance(capture, dict)
        or capture.get("full_workload") is not True
    ):
        raise ValueError("full-workload trace wrapper contradicts capture")
    return trace


def load_replays(path: Path) -> list[ReplayResult]:
    fields = parse_diagnostic_artifact(read_json(path), "replay-suite")
    rows = fields.get("replays")
    if not isinstance(rows, list):
        raise TypeError("replay suite requires replays")
    return [ReplayResult.from_dict(row) for row in rows]


def transfer_groups(paths: Sequence[Path]) -> list[list[TransferSample]]:
    groups: dict[str, list[TransferSample]] = {}
    for path in paths:
        fields = parse_diagnostic_artifact(read_json(path), "transfer-samples")
        if (
            fields.get("status") != "complete"
            or fields.get("evidence_complete") is False
        ):
            raise ValueError(
                "failed/partial transport is observations only; use export"
            )
        rows = fields.get("samples")
        if not isinstance(rows, list) or not rows:
            raise ValueError("transport requires nonempty samples")
        config = fields.get("config")
        if not isinstance(config, dict):
            raise TypeError("transport measurement config required")
        contract = validate_transfer_contract(fields.get("contract"))
        v = evidence_validator()
        repetitions = v.integer(config.get("repetitions"), 1)
        sizes, modes = config.get("experts_per_batch"), config.get("modes")
        if (
            not isinstance(sizes, list)
            or not sizes
            or not isinstance(modes, list)
            or not modes
            or any(type(size) is not int or size <= 0 for size in sizes)
            or any(mode not in {"contiguous", "fragmented", "gather"} for mode in modes)
            or len(set(sizes)) != len(sizes)
            or len(set(modes)) != len(modes)
        ):
            raise ValueError("invalid transport requested grid")
        parsed = [TransferSample.from_dict(raw) for raw in rows]
        expected = {
            (rank, size, mode, rep)
            for rank in range(4)
            for size in sizes
            for mode in modes
            for rep in range(repetitions)
        }
        actual = {
            (row.rank, row.experts_per_batch, row.mode, row.repetition)
            for row in parsed
        }
        if actual != expected or len(parsed) != len(expected):
            raise ValueError("transport requested grid incomplete or duplicated")
        if any(
            row.contract != contract
            or row.contention != config.get("contention")
            or row.expert_bytes != config.get("expert_bytes")
            for row in parsed
        ):
            raise ValueError("transport coordinator contract differs")
        for row in parsed:
            key = json.dumps([row.mode, row.contention, row.contract], sort_keys=True)
            groups.setdefault(key, []).append(row)
    for group in groups.values():
        identities = {(r.rank, r.experts_per_batch, r.repetition) for r in group}
        if len(identities) != len(group):
            raise ValueError("duplicate transfer evidence file/group")
    return [groups[key] for key in sorted(groups)]
