"""Strict standard-library evidence contracts for offload feasibility analysis."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SHA40_RE = re.compile(r"[0-9a-f]{40}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

_REQUIRED_HASHES = (
    "model_identity_sha256",
    "model_config_sha256",
    "dataset_sha256",
    "input_sha256",
    "engine_policy_sha256",
)
_OPTIONAL_HASHES = (
    "dataset_manifest_sha256",
    "calibration_input_hashes_sha256",
)
_REQUIRED_POSITIVE_INTS = (
    "batch_size",
    "context_length",
    "output_length",
    "max_num_seqs",
    "max_num_batched_tokens",
)
_OPTIONAL_POSITIVE_INTS = (
    "source_request_count",
    "unique_selected_request_count",
    "repetitions_requested",
    "smoke_output_length",
    "calibration_count",
    "evaluation_pool_count",
)
_OPTIONAL_NON_NEGATIVE_INTS = (
    "seed",
    "repeated_request_count",
    "warmups",
    "timing_samples",
)
_OPTIONAL_STRINGS = (
    "dtype",
    "sampling_policy",
    "comparison_backend",
    "storage_backend",
    "run_id",
    "role",
    "trace_mode",
    "synthetic_packing",
    "hardware_unavailable_reason",
    "split_scope",
)
_OPTIONAL_BOOLS = ("synthetic",)
_VERSION_KEYS = frozenset({"torch", "vllm", "cuda", "vllm_commit", "python"})
_TRANSFER_CONTRACT_KEYS = frozenset(
    {
        "commit",
        "tensor_parallel_size",
        "versions",
        "hardware_sha256",
        "hardware_unavailable_reason",
        "measurement_id",
        "benchmark_policy_sha256",
    }
)
_REPLAY_COUNTER_FIELDS = frozenset(
    {"demands", "resident_hits", "cache_hits", "loaded_experts", "loaded_bytes"}
)
_CONTRACT_KEYS = frozenset(
    _REQUIRED_HASHES
    + _OPTIONAL_HASHES
    + _REQUIRED_POSITIVE_INTS
    + _OPTIONAL_POSITIVE_INTS
    + _OPTIONAL_NON_NEGATIVE_INTS
    + _OPTIONAL_STRINGS
    + _OPTIONAL_BOOLS
    + (
        "commit",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "versions",
        "hardware_sha256",
        "prompt_hashes",
    )
)
_ARTIFACT_ROOT = {
    "schema_version": 1,
    "diagnostic_only": True,
    "formal_offload_gain": False,
    "deployment_gain_proven": False,
}


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    return cast(Sequence[object], value)


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, minimum: float = 0.0) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a finite real number")
    result = float(cast(float | int, value))
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _sha40(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA40_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 40-character commit SHA")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a portable identifier")
    return value


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")
    return value


def _versions(value: object) -> dict[str, str]:
    versions = _mapping(value, "versions")
    required_versions = {"torch", "vllm", "cuda", "vllm_commit"}
    if set(versions) - _VERSION_KEYS or not required_versions <= set(versions):
        raise ValueError("versions has unsupported or missing fields")
    result: dict[str, str] = {}
    for name in sorted(versions):
        item = versions[name]
        if name == "vllm_commit":
            result[name] = _sha40(item, "versions.vllm_commit")
        elif not isinstance(item, str) or not item:
            raise ValueError(f"versions.{name} must be a non-empty string")
        else:
            result[name] = item
    return result


def _hardware(raw: Mapping[str, object]) -> tuple[str | None, str | None]:
    if "hardware_sha256" not in raw:
        raise ValueError("contract requires hardware_sha256")
    hardware = raw["hardware_sha256"]
    reason = raw.get("hardware_unavailable_reason")
    if hardware is None:
        if not isinstance(reason, str) or not reason:
            raise ValueError(
                "null hardware_sha256 requires hardware_unavailable_reason"
            )
        return None, reason
    if reason is not None:
        raise ValueError(
            "hardware_unavailable_reason is only valid for unavailable hardware"
        )
    return _sha256(hardware, "hardware_sha256"), None


def validate_contract(
    value: object, *, require_prompt_hashes: bool = False
) -> dict[str, object]:
    """Validate and copy the shared identity/workload contract."""

    raw = _mapping(value, "contract")
    unknown = set(raw) - _CONTRACT_KEYS
    if unknown:
        raise ValueError(f"contract contains unsupported fields: {sorted(unknown)}")
    missing = set(_REQUIRED_HASHES + _REQUIRED_POSITIVE_INTS) - set(raw)
    missing.update(
        {
            "commit",
            "tensor_parallel_size",
            "gpu_memory_utilization",
            "versions",
            "hardware_sha256",
        }
        - set(raw)
    )
    if missing:
        raise ValueError(f"contract is missing required fields: {sorted(missing)}")

    result: dict[str, object] = {}
    for name in _REQUIRED_HASHES:
        result[name] = _sha256(raw[name], name)
    for name in _OPTIONAL_HASHES:
        if name in raw:
            result[name] = _sha256(raw[name], name)
    result["commit"] = _sha40(raw["commit"], "commit")
    if _integer(raw["tensor_parallel_size"], "tensor_parallel_size", 1) != 4:
        raise ValueError("tensor_parallel_size must be 4")
    result["tensor_parallel_size"] = 4
    for name in _REQUIRED_POSITIVE_INTS:
        result[name] = _integer(raw[name], name, 1)
    utilization = _number(raw["gpu_memory_utilization"], "gpu_memory_utilization")
    if not 0.0 < utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    result["gpu_memory_utilization"] = utilization

    result["versions"] = _versions(raw["versions"])
    hardware, unavailable_reason = _hardware(raw)
    result["hardware_sha256"] = hardware
    if unavailable_reason is not None:
        result["hardware_unavailable_reason"] = unavailable_reason
    prompts_raw = raw.get("prompt_hashes")
    if prompts_raw is not None:
        # These are unique selected request hashes. Repetition is represented
        # separately by repeated_request_count, so overlap tests remain exact.
        prompts = tuple(
            _sha256(item, "prompt_hash")
            for item in _sequence(prompts_raw, "prompt_hashes")
        )
        if not prompts or len(set(prompts)) != len(prompts):
            raise ValueError("prompt_hashes must be nonempty and distinct")
        result["prompt_hashes"] = prompts
    elif require_prompt_hashes:
        raise ValueError("trace contract requires prompt_hashes")

    for name in _OPTIONAL_POSITIVE_INTS:
        if name in raw:
            result[name] = _integer(raw[name], name, 1)
    for name in _OPTIONAL_NON_NEGATIVE_INTS:
        if name in raw:
            result[name] = _integer(raw[name], name)
    for name in _OPTIONAL_STRINGS:
        if name in raw:
            item = raw[name]
            if not isinstance(item, str) or not item:
                raise ValueError(f"{name} must be a non-empty string")
            result[name] = item
    for name in _OPTIONAL_BOOLS:
        if name in raw:
            result[name] = _bool(raw[name], name)
    return result


def validate_transfer_contract(value: object) -> dict[str, object]:
    """Validate the standalone four-rank transport measurement contract."""

    raw = _mapping(value, "transfer contract")
    unknown = set(raw) - _TRANSFER_CONTRACT_KEYS
    if unknown:
        raise ValueError(
            f"transfer contract contains unsupported fields: {sorted(unknown)}"
        )
    missing = {"commit", "tensor_parallel_size", "versions", "hardware_sha256"} - set(
        raw
    )
    if missing:
        raise ValueError(
            f"transfer contract is missing required fields: {sorted(missing)}"
        )
    if _integer(raw["tensor_parallel_size"], "tensor_parallel_size", 1) != 4:
        raise ValueError("tensor_parallel_size must be 4")
    result: dict[str, object] = {
        "commit": _sha40(raw["commit"], "commit"),
        "tensor_parallel_size": 4,
        "versions": _versions(raw["versions"]),
    }
    hardware, unavailable_reason = _hardware(raw)
    result["hardware_sha256"] = hardware
    if unavailable_reason is not None:
        result["hardware_unavailable_reason"] = unavailable_reason
    if "measurement_id" in raw:
        result["measurement_id"] = _identifier(raw["measurement_id"], "measurement_id")
    if "benchmark_policy_sha256" in raw:
        result["benchmark_policy_sha256"] = _sha256(
            raw["benchmark_policy_sha256"], "benchmark_policy_sha256"
        )
    return result


def _contract_to_dict(contract: Mapping[str, object]) -> dict[str, object]:
    result = dict(contract)
    versions = cast(Mapping[str, str], contract["versions"])
    result["versions"] = dict(versions)
    if "prompt_hashes" in result:
        result["prompt_hashes"] = list(cast(tuple[str, ...], result["prompt_hashes"]))
    return result


def _replay_totals(value: object, name: str) -> dict[str, int]:
    raw = _mapping(value, name)
    if set(raw) != _REPLAY_COUNTER_FIELDS:
        raise ValueError(f"{name} fields differ")
    return {key: _integer(raw[key], f"{name}.{key}") for key in raw}


def diagnostic_artifact(
    artifact_kind: str, fields: Mapping[str, object]
) -> dict[str, object]:
    """Build the single root format used for every saved analysis artifact."""

    kind = _identifier(artifact_kind, "artifact_kind")
    payload = dict(_mapping(fields, "artifact fields"))
    reserved = set(_ARTIFACT_ROOT) | {"artifact_kind"}
    if reserved & set(payload):
        raise ValueError("artifact fields cannot override fixed root metadata")
    return {**_ARTIFACT_ROOT, "artifact_kind": kind, **payload}


def parse_diagnostic_artifact(value: object, expected_kind: str) -> dict[str, object]:
    """Validate fixed root labels and return the artifact-specific fields."""

    raw = _mapping(value, "diagnostic artifact")
    if any(raw.get(key) != expected for key, expected in _ARTIFACT_ROOT.items()):
        raise ValueError("diagnostic artifact fixed labels differ")
    if raw.get("artifact_kind") != _identifier(expected_kind, "expected_kind"):
        raise ValueError("diagnostic artifact kind differs")
    reserved = set(_ARTIFACT_ROOT) | {"artifact_kind"}
    return {key: item for key, item in raw.items() if key not in reserved}


@dataclass(frozen=True)
class TraceEvent:
    step: int
    layer: int
    token_rows: int
    requests: int | None
    phase: str
    experts: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "step", _integer(self.step, "step"))
        object.__setattr__(self, "layer", _integer(self.layer, "layer"))
        object.__setattr__(
            self, "token_rows", _integer(self.token_rows, "token_rows", 1)
        )
        if self.requests is not None:
            requests = _integer(self.requests, "requests", 1)
            if requests > self.token_rows:
                raise ValueError("requests cannot exceed token_rows")
            object.__setattr__(self, "requests", requests)
        if self.phase not in {"prefill", "decode", "mixed", "unknown"}:
            raise ValueError("phase must be prefill, decode, mixed, or unknown")
        experts = tuple(
            _integer(item, "expert", 0) for item in _sequence(self.experts, "experts")
        )
        if not experts or experts != tuple(sorted(set(experts))):
            raise ValueError("experts must be nonempty, sorted, and distinct")
        object.__setattr__(self, "experts", experts)

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "layer": self.layer,
            "token_rows": self.token_rows,
            "requests": self.requests,
            "phase": self.phase,
            "experts": list(self.experts),
        }

    @classmethod
    def from_dict(cls, value: object) -> TraceEvent:
        raw = _mapping(value, "TraceEvent")
        expected = {"step", "layer", "token_rows", "requests", "phase", "experts"}
        if set(raw) != expected:
            raise ValueError("TraceEvent fields differ")
        return cls(
            cast(int, raw["step"]),
            cast(int, raw["layer"]),
            cast(int, raw["token_rows"]),
            cast(int | None, raw["requests"]),
            cast(str, raw["phase"]),
            tuple(cast(Sequence[int], _sequence(raw["experts"], "experts"))),
        )


@dataclass(frozen=True)
class DemandTrace:
    trace_id: str
    rank: int
    contract: dict[str, object]
    total_layers: int
    num_experts: int
    top_k: int
    expert_bytes: int
    observed_steps: int
    captured_steps: int
    full_workload: bool
    generated_tokens: int | None
    events: tuple[TraceEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_id", _identifier(self.trace_id, "trace_id"))
        rank = _integer(self.rank, "rank")
        if rank > 3:
            raise ValueError("rank must be in [0, 3]")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(
            self,
            "contract",
            validate_contract(self.contract, require_prompt_hashes=True),
        )
        total_layers = _integer(self.total_layers, "total_layers", 1)
        num_experts = _integer(self.num_experts, "num_experts", 1)
        top_k = _integer(self.top_k, "top_k", 1)
        if top_k > num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        expert_bytes = _integer(self.expert_bytes, "expert_bytes", 1)
        observed = _integer(self.observed_steps, "observed_steps", 1)
        captured = _integer(self.captured_steps, "captured_steps", 1)
        if captured > observed:
            raise ValueError("captured_steps cannot exceed observed_steps")
        full = _bool(self.full_workload, "full_workload")
        generated = self.generated_tokens
        if generated is not None:
            generated = _integer(generated, "generated_tokens", 1)
        if full and (captured != observed or generated is None):
            raise ValueError(
                "full_workload requires complete capture and generated tokens"
            )
        events = tuple(self.events)
        if any(not isinstance(event, TraceEvent) for event in events):
            raise TypeError("events must contain TraceEvent values")
        expected_grid = {
            (step, layer) for step in range(captured) for layer in range(total_layers)
        }
        actual_grid = {(event.step, event.layer) for event in events}
        if len(events) != len(actual_grid) or actual_grid != expected_grid:
            raise ValueError(
                "events must contain every layer exactly once for every step"
            )
        for event in events:
            if any(expert >= num_experts for expert in event.experts):
                raise ValueError("event expert is outside trace geometry")
            if len(event.experts) > event.token_rows * top_k:
                raise ValueError("event has more unique experts than routing permits")
        events = tuple(sorted(events, key=lambda event: (event.step, event.layer)))
        object.__setattr__(self, "total_layers", total_layers)
        object.__setattr__(self, "num_experts", num_experts)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "expert_bytes", expert_bytes)
        object.__setattr__(self, "observed_steps", observed)
        object.__setattr__(self, "captured_steps", captured)
        object.__setattr__(self, "full_workload", full)
        object.__setattr__(self, "generated_tokens", generated)
        object.__setattr__(self, "events", events)

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "rank": self.rank,
            "contract": _contract_to_dict(self.contract),
            "total_layers": self.total_layers,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "expert_bytes": self.expert_bytes,
            "observed_steps": self.observed_steps,
            "captured_steps": self.captured_steps,
            "full_workload": self.full_workload,
            "generated_tokens": self.generated_tokens,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, value: object) -> DemandTrace:
        raw = _mapping(value, "DemandTrace")
        expected = {
            "trace_id",
            "rank",
            "contract",
            "total_layers",
            "num_experts",
            "top_k",
            "expert_bytes",
            "observed_steps",
            "captured_steps",
            "full_workload",
            "generated_tokens",
            "events",
        }
        if set(raw) != expected:
            raise ValueError("DemandTrace fields differ")
        return cls(
            cast(str, raw["trace_id"]),
            cast(int, raw["rank"]),
            dict(_mapping(raw["contract"], "contract")),
            cast(int, raw["total_layers"]),
            cast(int, raw["num_experts"]),
            cast(int, raw["top_k"]),
            cast(int, raw["expert_bytes"]),
            cast(int, raw["observed_steps"]),
            cast(int, raw["captured_steps"]),
            cast(bool, raw["full_workload"]),
            cast(int | None, raw["generated_tokens"]),
            tuple(
                TraceEvent.from_dict(item)
                for item in _sequence(raw["events"], "events")
            ),
        )


@dataclass(frozen=True)
class ReplayConfig:
    resident_experts: tuple[tuple[int, ...], ...]
    cache_slots: int
    staging_experts: int
    policy: str
    extra_gpu_bytes: int = 0

    def __post_init__(self) -> None:
        rows: list[tuple[int, ...]] = []
        for layer, raw in enumerate(
            _sequence(self.resident_experts, "resident_experts")
        ):
            row = tuple(
                _integer(item, f"resident_experts[{layer}]")
                for item in _sequence(raw, "resident row")
            )
            if row != tuple(sorted(set(row))):
                raise ValueError("resident expert rows must be sorted and distinct")
            rows.append(row)
        if not rows:
            raise ValueError("resident_experts must contain at least one layer")
        object.__setattr__(self, "resident_experts", tuple(rows))
        object.__setattr__(
            self, "cache_slots", _integer(self.cache_slots, "cache_slots")
        )
        object.__setattr__(
            self,
            "staging_experts",
            _integer(self.staging_experts, "staging_experts", 1),
        )
        if self.policy not in {"lru", "decayed-lfu", "future"}:
            raise ValueError("policy must be lru, decayed-lfu, or future")
        object.__setattr__(
            self, "extra_gpu_bytes", _integer(self.extra_gpu_bytes, "extra_gpu_bytes")
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "resident_experts": [list(row) for row in self.resident_experts],
            "cache_slots": self.cache_slots,
            "staging_experts": self.staging_experts,
            "policy": self.policy,
            "extra_gpu_bytes": self.extra_gpu_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> ReplayConfig:
        raw = _mapping(value, "ReplayConfig")
        expected = {
            "resident_experts",
            "cache_slots",
            "staging_experts",
            "policy",
            "extra_gpu_bytes",
        }
        if set(raw) != expected:
            raise ValueError("ReplayConfig fields differ")
        residents = tuple(
            tuple(cast(Sequence[int], _sequence(row, "resident row")))
            for row in _sequence(raw["resident_experts"], "resident_experts")
        )
        return cls(
            residents,
            cast(int, raw["cache_slots"]),
            cast(int, raw["staging_experts"]),
            cast(str, raw["policy"]),
            cast(int, raw["extra_gpu_bytes"]),
        )


@dataclass(frozen=True)
class TransferSample:
    rank: int
    contract: dict[str, object]
    experts_per_batch: int
    expert_bytes: int
    mode: str
    contention: str
    repetition: int
    payload_bytes: int
    wall_s: float
    copy_s: float
    gather_s: float
    compute_s: float | None

    def __post_init__(self) -> None:
        rank = _integer(self.rank, "rank")
        if rank > 3:
            raise ValueError("rank must be in [0, 3]")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "contract", validate_transfer_contract(self.contract))
        experts = _integer(self.experts_per_batch, "experts_per_batch", 1)
        expert_bytes = _integer(self.expert_bytes, "expert_bytes", 1)
        payload = _integer(self.payload_bytes, "payload_bytes", 1)
        if payload != experts * expert_bytes:
            raise ValueError("payload_bytes must exactly equal measured expert payload")
        if self.mode not in {"contiguous", "fragmented", "gather"}:
            raise ValueError("unsupported transfer mode")
        if self.contention not in {"isolated", "gemm-nccl-proxy"}:
            raise ValueError("unsupported contention mode")
        repetition = _integer(self.repetition, "repetition")
        wall = _number(self.wall_s, "wall_s")
        copy = _number(self.copy_s, "copy_s")
        gather = _number(self.gather_s, "gather_s")
        if wall <= 0.0 or copy <= 0.0:
            raise ValueError("wall_s and copy_s must be positive")
        compute = (
            None if self.compute_s is None else _number(self.compute_s, "compute_s")
        )
        if wall + 1e-12 < copy + gather or (
            compute is not None and wall + 1e-12 < compute
        ):
            raise ValueError("wall_s cannot be shorter than measured component time")
        object.__setattr__(self, "experts_per_batch", experts)
        object.__setattr__(self, "expert_bytes", expert_bytes)
        object.__setattr__(self, "repetition", repetition)
        object.__setattr__(self, "payload_bytes", payload)
        object.__setattr__(self, "wall_s", wall)
        object.__setattr__(self, "copy_s", copy)
        object.__setattr__(self, "gather_s", gather)
        object.__setattr__(self, "compute_s", compute)

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "contract": _contract_to_dict(self.contract),
            "experts_per_batch": self.experts_per_batch,
            "expert_bytes": self.expert_bytes,
            "mode": self.mode,
            "contention": self.contention,
            "repetition": self.repetition,
            "payload_bytes": self.payload_bytes,
            "wall_s": self.wall_s,
            "copy_s": self.copy_s,
            "gather_s": self.gather_s,
            "compute_s": self.compute_s,
        }

    @classmethod
    def from_dict(cls, value: object) -> TransferSample:
        raw = _mapping(value, "TransferSample")
        expected = {
            "rank",
            "contract",
            "experts_per_batch",
            "expert_bytes",
            "mode",
            "contention",
            "repetition",
            "payload_bytes",
            "wall_s",
            "copy_s",
            "gather_s",
            "compute_s",
        }
        if set(raw) != expected:
            raise ValueError("TransferSample fields differ")
        return cls(
            cast(int, raw["rank"]),
            dict(_mapping(raw["contract"], "contract")),
            cast(int, raw["experts_per_batch"]),
            cast(int, raw["expert_bytes"]),
            cast(str, raw["mode"]),
            cast(str, raw["contention"]),
            cast(int, raw["repetition"]),
            cast(int, raw["payload_bytes"]),
            cast(float, raw["wall_s"]),
            cast(float, raw["copy_s"]),
            cast(float, raw["gather_s"]),
            cast(float | None, raw["compute_s"]),
        )


@dataclass(frozen=True)
class TimingPoint:
    point_id: str
    contract: dict[str, object]
    generated_tokens: int
    elapsed_s: tuple[float, ...]
    kv_bytes_by_rank: tuple[int, ...]
    gpu_total_bytes_by_rank: tuple[int, ...]
    source_kind: str
    engine_mode: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_id", _identifier(self.point_id, "point_id"))
        contract = validate_contract(self.contract)
        object.__setattr__(self, "contract", contract)
        generated = _integer(self.generated_tokens, "generated_tokens", 1)
        if generated != cast(int, contract["batch_size"]) * cast(
            int, contract["output_length"]
        ):
            raise ValueError("generated_tokens differs from fixed workload")
        elapsed = tuple(
            _number(item, "elapsed_s")
            for item in _sequence(self.elapsed_s, "elapsed_s")
        )
        if not elapsed or any(item <= 0.0 for item in elapsed):
            raise ValueError("elapsed_s must contain positive measurements")
        kv = tuple(
            _integer(item, "kv_bytes_by_rank")
            for item in _sequence(self.kv_bytes_by_rank, "kv_bytes_by_rank")
        )
        totals = tuple(
            _integer(item, "gpu_total_bytes_by_rank", 1)
            for item in _sequence(
                self.gpu_total_bytes_by_rank, "gpu_total_bytes_by_rank"
            )
        )
        if (
            len(kv) != 4
            or len(totals) != 4
            or any(k > total for k, total in zip(kv, totals))
        ):
            raise ValueError("timing point requires four valid physical GPU totals")
        if self.source_kind not in {"native-measured", "kv-oracle-diagnostic"}:
            raise ValueError("unsupported timing source_kind")
        if self.engine_mode not in {"native", "eager"}:
            raise ValueError("engine_mode must be native or eager")
        object.__setattr__(self, "generated_tokens", generated)
        object.__setattr__(self, "elapsed_s", elapsed)
        object.__setattr__(self, "kv_bytes_by_rank", kv)
        object.__setattr__(self, "gpu_total_bytes_by_rank", totals)

    def to_dict(self) -> dict[str, object]:
        return {
            "point_id": self.point_id,
            "contract": _contract_to_dict(self.contract),
            "generated_tokens": self.generated_tokens,
            "elapsed_s": list(self.elapsed_s),
            "kv_bytes_by_rank": list(self.kv_bytes_by_rank),
            "gpu_total_bytes_by_rank": list(self.gpu_total_bytes_by_rank),
            "source_kind": self.source_kind,
            "engine_mode": self.engine_mode,
        }

    @classmethod
    def from_dict(cls, value: object) -> TimingPoint:
        raw = _mapping(value, "TimingPoint")
        expected = {
            "point_id",
            "contract",
            "generated_tokens",
            "elapsed_s",
            "kv_bytes_by_rank",
            "gpu_total_bytes_by_rank",
            "source_kind",
            "engine_mode",
        }
        if set(raw) != expected:
            raise ValueError("TimingPoint fields differ")
        return cls(
            cast(str, raw["point_id"]),
            dict(_mapping(raw["contract"], "contract")),
            cast(int, raw["generated_tokens"]),
            tuple(cast(Sequence[float], _sequence(raw["elapsed_s"], "elapsed_s"))),
            tuple(
                cast(
                    Sequence[int],
                    _sequence(raw["kv_bytes_by_rank"], "kv_bytes_by_rank"),
                )
            ),
            tuple(
                cast(
                    Sequence[int],
                    _sequence(
                        raw["gpu_total_bytes_by_rank"], "gpu_total_bytes_by_rank"
                    ),
                )
            ),
            cast(str, raw["source_kind"]),
            cast(str, raw["engine_mode"]),
        )


@dataclass(frozen=True)
class ReplayResult:
    config: ReplayConfig
    trace_id: str
    rank: int
    model_identity_sha256: str
    model_config_sha256: str
    dataset_sha256: str
    input_sha256: str
    commit: str
    hardware_sha256: str | None
    loaded_experts: int
    loaded_bytes: int
    resident_hits: int
    cache_hits: int
    demands: int
    bytes_per_generated_token: float | None
    net_freed_bytes: int
    event_load_counts: tuple[int, ...]
    event_load_bytes: tuple[int, ...]
    final_cache: tuple[tuple[int, int], ...]
    per_layer_totals: tuple[dict[str, int], ...]
    per_phase_totals: dict[str, dict[str, int]]
    reuse_gap_summary: dict[str, int | float | None]
    staging_overflow_events: int
    future_policy_reference: bool
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.config, ReplayConfig):
            raise TypeError("config must be ReplayConfig")
        object.__setattr__(self, "trace_id", _identifier(self.trace_id, "trace_id"))
        rank = _integer(self.rank, "rank")
        if rank > 3:
            raise ValueError("rank must be in [0, 3]")
        object.__setattr__(self, "rank", rank)
        for name in (
            "model_identity_sha256",
            "model_config_sha256",
            "dataset_sha256",
            "input_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if self.hardware_sha256 is not None:
            object.__setattr__(
                self,
                "hardware_sha256",
                _sha256(self.hardware_sha256, "hardware_sha256"),
            )
        object.__setattr__(self, "commit", _sha40(self.commit, "commit"))
        for name in (
            "loaded_experts",
            "loaded_bytes",
            "resident_hits",
            "cache_hits",
            "demands",
            "staging_overflow_events",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.resident_hits + self.cache_hits + self.loaded_experts != self.demands:
            raise ValueError("replay hit/miss accounting differs from demands")
        if self.bytes_per_generated_token is not None:
            object.__setattr__(
                self,
                "bytes_per_generated_token",
                _number(self.bytes_per_generated_token, "bytes_per_generated_token"),
            )
        if type(self.net_freed_bytes) is not int:
            raise ValueError("net_freed_bytes must be an integer")
        counts = tuple(
            _integer(item, "event_load_counts") for item in self.event_load_counts
        )
        byte_rows = tuple(
            _integer(item, "event_load_bytes") for item in self.event_load_bytes
        )
        if (
            not counts
            or len(counts) != len(byte_rows)
            or sum(counts) != self.loaded_experts
            or sum(byte_rows) != self.loaded_bytes
        ):
            raise ValueError("per-event replay accounting differs from totals")
        expert_byte_values = {
            byte_count // count
            for count, byte_count in zip(counts, byte_rows)
            if count > 0 and byte_count % count == 0
        }
        if (
            any(
                (count == 0) != (byte_count == 0)
                for count, byte_count in zip(counts, byte_rows)
            )
            or any(
                count > 0 and byte_count % count != 0
                for count, byte_count in zip(counts, byte_rows)
            )
            or len(expert_byte_values) > 1
        ):
            raise ValueError("per-event bytes do not use one exact expert size")
        object.__setattr__(self, "event_load_counts", counts)
        object.__setattr__(self, "event_load_bytes", byte_rows)
        cache_keys: list[tuple[int, int]] = []
        for raw_key in _sequence(self.final_cache, "final_cache"):
            key = _sequence(raw_key, "final cache key")
            if len(key) != 2:
                raise ValueError("final cache keys must be (layer, expert)")
            layer = _integer(key[0], "cache layer")
            expert = _integer(key[1], "cache expert")
            if layer >= len(self.config.resident_experts):
                raise ValueError("final cache layer is outside replay geometry")
            if expert in self.config.resident_experts[layer]:
                raise ValueError("resident expert cannot also occupy cache")
            cache_keys.append((layer, expert))
        if (
            tuple(cache_keys) != tuple(sorted(set(cache_keys)))
            or len(cache_keys) > self.config.cache_slots
        ):
            raise ValueError("final_cache must be sorted, distinct, and bounded")
        object.__setattr__(self, "final_cache", tuple(cache_keys))
        layers = tuple(
            _replay_totals(row, "per_layer_totals")
            for row in _sequence(self.per_layer_totals, "per_layer_totals")
        )
        if len(layers) != len(self.config.resident_experts):
            raise ValueError("per_layer_totals differs from replay geometry")
        phases = {
            key: _replay_totals(row, f"per_phase_totals.{key}")
            for key, row in _mapping(self.per_phase_totals, "per_phase_totals").items()
        }
        expected_totals = {
            "demands": self.demands,
            "resident_hits": self.resident_hits,
            "cache_hits": self.cache_hits,
            "loaded_experts": self.loaded_experts,
            "loaded_bytes": self.loaded_bytes,
        }
        for rows, name in ((layers, "layer"), (tuple(phases.values()), "phase")):
            if any(
                sum(row[field] for row in rows) != total
                for field, total in expected_totals.items()
            ):
                raise ValueError(f"per-{name} replay totals differ from totals")
        object.__setattr__(self, "per_layer_totals", layers)
        object.__setattr__(self, "per_phase_totals", phases)
        reuse_raw = _mapping(self.reuse_gap_summary, "reuse_gap_summary")
        reuse_fields = {
            "observations",
            "reuse_count",
            "min_event_gap",
            "max_event_gap",
            "mean_event_gap",
        }
        if set(reuse_raw) != reuse_fields:
            raise ValueError("reuse_gap_summary fields differ")
        observations = _integer(reuse_raw["observations"], "observations")
        reuse_count = _integer(reuse_raw["reuse_count"], "reuse_count")
        if observations != self.demands or reuse_count > observations:
            raise ValueError("reuse summary observations differ from demands")
        if reuse_count == 0:
            if any(
                reuse_raw[name] is not None
                for name in ("min_event_gap", "max_event_gap", "mean_event_gap")
            ):
                raise ValueError("empty reuse gaps require null summary")
            reuse = dict(reuse_raw)
        else:
            minimum = _integer(reuse_raw["min_event_gap"], "min_event_gap", 1)
            maximum = _integer(reuse_raw["max_event_gap"], "max_event_gap", minimum)
            mean = _number(reuse_raw["mean_event_gap"], "mean_event_gap", minimum)
            if mean > maximum:
                raise ValueError("mean_event_gap exceeds maximum")
            reuse = {
                "observations": observations,
                "reuse_count": reuse_count,
                "min_event_gap": minimum,
                "max_event_gap": maximum,
                "mean_event_gap": mean,
            }
        object.__setattr__(self, "reuse_gap_summary", reuse)
        future = _bool(self.future_policy_reference, "future_policy_reference")
        if future != (self.config.policy == "future"):
            raise ValueError("future policy marker differs from ReplayConfig")
        object.__setattr__(self, "future_policy_reference", future)
        assumptions = tuple(_sequence(self.assumptions, "assumptions"))
        if any(not isinstance(item, str) or not item for item in assumptions):
            raise ValueError("assumptions must contain non-empty strings")
        if len(assumptions) != len(set(assumptions)):
            raise ValueError("assumptions must be distinct")
        object.__setattr__(self, "assumptions", cast(tuple[str, ...], assumptions))

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "trace_id": self.trace_id,
            "rank": self.rank,
            "model_identity_sha256": self.model_identity_sha256,
            "model_config_sha256": self.model_config_sha256,
            "dataset_sha256": self.dataset_sha256,
            "input_sha256": self.input_sha256,
            "commit": self.commit,
            "hardware_sha256": self.hardware_sha256,
            "loaded_experts": self.loaded_experts,
            "loaded_bytes": self.loaded_bytes,
            "resident_hits": self.resident_hits,
            "cache_hits": self.cache_hits,
            "demands": self.demands,
            "bytes_per_generated_token": self.bytes_per_generated_token,
            "net_freed_bytes": self.net_freed_bytes,
            "event_load_counts": list(self.event_load_counts),
            "event_load_bytes": list(self.event_load_bytes),
            "final_cache": [list(key) for key in self.final_cache],
            "per_layer_totals": [dict(row) for row in self.per_layer_totals],
            "per_phase_totals": {
                key: dict(value) for key, value in self.per_phase_totals.items()
            },
            "reuse_gap_summary": dict(self.reuse_gap_summary),
            "staging_overflow_events": self.staging_overflow_events,
            "future_policy_reference": self.future_policy_reference,
            "assumptions": list(self.assumptions),
        }

    @classmethod
    def from_dict(cls, value: object) -> ReplayResult:
        raw = _mapping(value, "ReplayResult")
        expected = set(cls.__dataclass_fields__)
        if set(raw) != expected:
            raise ValueError("ReplayResult fields differ")
        final_cache: list[tuple[int, int]] = []
        for raw_key in _sequence(raw["final_cache"], "final_cache"):
            key = _sequence(raw_key, "final cache key")
            if len(key) != 2:
                raise ValueError("final cache key must have two entries")
            final_cache.append((cast(int, key[0]), cast(int, key[1])))
        layer_totals = tuple(
            _replay_totals(row, "layer total")
            for row in _sequence(raw["per_layer_totals"], "per_layer_totals")
        )
        phase_totals = {
            key: _replay_totals(item, "phase total")
            for key, item in _mapping(
                raw["per_phase_totals"], "per_phase_totals"
            ).items()
        }
        reuse = dict(_mapping(raw["reuse_gap_summary"], "reuse_gap_summary"))
        return cls(
            ReplayConfig.from_dict(raw["config"]),
            cast(str, raw["trace_id"]),
            cast(int, raw["rank"]),
            cast(str, raw["model_identity_sha256"]),
            cast(str, raw["model_config_sha256"]),
            cast(str, raw["dataset_sha256"]),
            cast(str, raw["input_sha256"]),
            cast(str, raw["commit"]),
            cast(str | None, raw["hardware_sha256"]),
            cast(int, raw["loaded_experts"]),
            cast(int, raw["loaded_bytes"]),
            cast(int, raw["resident_hits"]),
            cast(int, raw["cache_hits"]),
            cast(int, raw["demands"]),
            cast(float | None, raw["bytes_per_generated_token"]),
            cast(int, raw["net_freed_bytes"]),
            tuple(
                cast(
                    Sequence[int],
                    _sequence(raw["event_load_counts"], "event_load_counts"),
                )
            ),
            tuple(
                cast(
                    Sequence[int],
                    _sequence(raw["event_load_bytes"], "event_load_bytes"),
                )
            ),
            tuple(final_cache),
            layer_totals,
            phase_totals,
            cast(dict[str, int | float | None], reuse),
            cast(int, raw["staging_overflow_events"]),
            cast(bool, raw["future_policy_reference"]),
            tuple(str(item) for item in _sequence(raw["assumptions"], "assumptions")),
        )


__all__ = [
    "DemandTrace",
    "ReplayConfig",
    "ReplayResult",
    "TimingPoint",
    "TraceEvent",
    "TransferSample",
    "diagnostic_artifact",
    "parse_diagnostic_artifact",
    "validate_contract",
    "validate_transfer_contract",
]
