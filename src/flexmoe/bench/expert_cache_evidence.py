"""Strict stdlib-only cache evidence shared by GPU runner and host exporter."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

COUNTERS = (
    "h2d_bytes",
    "copy_launches",
    "startup_resident_h2d_bytes",
    "promotion_d2d_bytes",
    "metadata_h2d_bytes",
    "unique_demands",
    "resident_hits",
)
POLICY_COUNTERS = (
    "observations",
    "unique_demands",
    "completed_forwards",
    "decay_events",
    "cache_hits",
    "cache_misses",
    "admissions",
    "evictions",
    "cache_bypasses",
    "reconfigurations",
)
POLICY_GAUGES = ("resident_experts", "cache_slots", "cache_entries", "cache_hit_ratio")
TIMINGS = (
    "route_d2h_s",
    "policy_cpu_s",
    "host_reuse_wait_s",
    "host_gather_s",
    "h2d_enqueue_s",
    "compute_enqueue_s",
    "cuda_sample_count",
    "load_cuda_s",
    "compute_cuda_s",
    "promotion_cuda_s",
)
CAPACITY = (
    "schema_version",
    "rank",
    "tensor_parallel_size",
    "total_layers",
    "num_experts",
    "physical_slots",
    "resident_slots",
    "cache_slots",
    "ingress_slots",
    "expert_bytes",
    "host_source_bytes",
    "pinned_gather_bytes",
    "gpu_pool_bytes",
    "gpu_resident_bytes",
    "gpu_cache_bytes",
    "gpu_ingress_bytes",
    "gpu_metadata_bytes",
    "pinned_metadata_bytes",
    "net_freed_bytes",
    "weights_verified",
)
ARRAYS = (
    "forward_counts",
    "per_layer_unique_demands",
    "per_layer_max_unique_per_forward",
)
SCOPES = {
    "weight_verification_scope": "CUDA parity tests required; no runtime weight D2H verification",
    "temporary_memory_scope": "native fused kernel workspace/output and CPU route IDs scale with scheduled tokens; no weight-sized GPU gather temporary",
    "memory_accounting_scope": "tensor payload bytes; excludes allocator rounding, CUDA event driver memory and Python policy metadata; native/chunk output workspace must be included in vLLM measured profiling peak",
    "kernel_config_scope": "native logical E geometry per actual native chunk size",
    "transfer_schedule": "demand-critical H2D on compute stream; no speculative prefetch",
}


def number(value: Any, *, integer: bool = False) -> int | float:
    if (
        type(value) not in ((int,) if integer else (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("invalid cache numeric evidence")
    return value  # type: ignore[no-any-return]


def numbers(
    raw: Mapping[str, Any], names: tuple[str, ...], *, integer: bool = False
) -> dict[str, Any]:
    return {name: number(raw[name], integer=integer) for name in names if name in raw}


def hash_value(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("invalid cache identity hash")
    return value


def public_identity(raw: Mapping[str, Any]) -> dict[str, Any]:
    geometry = raw.get("geometry", {})
    return {
        "geometry": numbers(
            geometry,
            ("total_layers", "num_experts", "hidden_size", "intermediate_size"),
            integer=True,
        ),
        "tensor_parallel_size": number(raw["tensor_parallel_size"], integer=True),
        "model_config_sha256": hash_value(raw["model_config_sha256"]),
        "model_identity_sha256": hash_value(raw["model_identity_sha256"]),
    }


def public_settings(raw: Mapping[str, Any]) -> dict[str, Any]:
    policy = raw.get("cache_policy")
    if policy not in ("lru", "decayed-lfu"):
        raise ValueError("unknown cache policy")
    ratio = number(raw["resident_ratio"])
    if ratio >= 1:
        raise ValueError("resident ratio must be below one")
    return {
        "resident_ratio": ratio,
        "cache_policy": policy,
        "cache_slots": number(raw["cache_slots"], integer=True),
        "calibration_count": number(raw["calibration_count"], integer=True),
        "profile_sha256": hash_value(raw["profile_sha256"]),
        "identity": public_identity(raw["identity"]),
    }


def public_stats(raw: Mapping[str, Any]) -> dict[str, Any]:
    if raw.get("storage_backend") == "native":
        # These are declared zero control accounts, not sampled registry evidence.
        return {
            "storage_backend": "native",
            "rank": number(raw["rank"], integer=True),
            **numbers(raw, COUNTERS, integer=True),
            "timing": numbers(raw.get("timing", {}), TIMINGS),
            "policy": numbers(raw.get("policy", {}), POLICY_COUNTERS, integer=True),
        }
    out = {
        **numbers(raw, CAPACITY + COUNTERS, integer=True),
        **numbers(
            raw, ("max_unique_per_forward", "mean_unique_coverage", "resident_ratio")
        ),
        "policy": {
            **numbers(
                raw.get("policy", {}),
                POLICY_COUNTERS + POLICY_GAUGES[:-1],
                integer=True,
            ),
            **numbers(raw.get("policy", {}), ("cache_hit_ratio",)),
        },
        "timing": numbers(raw.get("timing", {}), TIMINGS),
    }
    for name in ARRAYS:
        if name in raw:
            if not isinstance(raw[name], list):
                raise ValueError("invalid per-layer array")
            out[name] = [number(x, integer=True) for x in raw[name]]
    if "identity" in raw:
        out["identity"] = public_identity(raw["identity"])
    if "profile_sha256" in raw:
        out["profile_sha256"] = hash_value(raw["profile_sha256"])
    if raw.get("cache_policy") in ("decayed-lfu", "lru"):
        out["cache_policy"] = raw["cache_policy"]
    if type(raw.get("failed")) is bool:
        out["failed"] = raw["failed"]
    for name, value in SCOPES.items():
        if raw.get(name) == value:
            out[name] = value
    if "kernel_config_records" in raw:
        kernel = raw["kernel_config_records"]
        records = []
        allowed = {
            "BLOCK_SIZE_M",
            "BLOCK_SIZE_N",
            "BLOCK_SIZE_K",
            "GROUP_SIZE_M",
            "num_warps",
            "num_stages",
            "SPLIT_K",
        }
        for record in kernel["records"]:
            if set(record["config"]) - allowed:
                raise ValueError("unknown kernel configuration fields")
            records.append(
                {
                    **numbers(
                        record, ("chunk_tokens", "top_k", "selections"), integer=True
                    ),
                    "w13_shape": [number(x, integer=True) for x in record["w13_shape"]],
                    "w2_shape": [number(x, integer=True) for x in record["w2_shape"]],
                    "config": {
                        key: number(value, integer=True)
                        for key, value in record["config"].items()
                    },
                }
            )
        out["kernel_config_records"] = {
            **numbers(kernel, ("capacity", "unrecorded_selections"), integer=True),
            "records": records,
        }
    return out


def ranked(raw: object, workers: int) -> list[dict[str, Any]]:
    if (
        not isinstance(raw, list)
        or len(raw) != workers
        or any(not isinstance(x, dict) for x in raw)
    ):
        raise ValueError("cache worker coverage incomplete")
    if {x.get("rank") for x in raw} != set(range(workers)):
        raise ValueError("cache worker ranks differ")
    return sorted(raw, key=lambda x: x["rank"])


def _difference(
    old: Mapping[str, Any], new: Mapping[str, Any], names: tuple[str, ...]
) -> dict[str, Any]:
    return {key: number(number(new[key]) - number(old[key])) for key in names}


def cache_deltas(
    before: object, after: object, *, expected_workers: int
) -> dict[str, Any]:
    totals: dict[str, Any] = {
        **dict.fromkeys(COUNTERS, 0),
        "timing": dict.fromkeys(TIMINGS, 0),
        "policy": dict.fromkeys(POLICY_COUNTERS, 0),
        "per_rank": [],
    }
    for old, new in zip(
        ranked(before, expected_workers), ranked(after, expected_workers)
    ):
        public_stats(old)
        public_stats(new)
        if any(
            old.get(key) != new.get(key)
            for key in (
                *CAPACITY,
                "identity",
                "profile_sha256",
                "cache_policy",
                "resident_ratio",
            )
        ):
            raise ValueError("cache layout or identity changed during measurement")
        delta = {
            "rank": new["rank"],
            **_difference(old, new, COUNTERS),
            "timing": _difference(old["timing"], new["timing"], TIMINGS),
            "policy": _difference(old["policy"], new["policy"], POLICY_COUNTERS),
        }
        for field in ("forward_counts", "per_layer_unique_demands"):
            if new.get("storage_backend") != "native":
                if len(old[field]) != len(new[field]):
                    raise ValueError("layer count changed during measurement")
                delta[field] = [
                    number(b - a, integer=True) for a, b in zip(old[field], new[field])
                ]
        if new.get("storage_backend") != "native":
            delta["per_layer_mean_unique_coverage"] = [
                count / forwards / new["num_experts"] if forwards else None
                for count, forwards in zip(
                    delta["per_layer_unique_demands"], delta["forward_counts"]
                )
            ]
            # Gauge snapshots are explicitly separated from cumulative deltas.
            delta["gauge_snapshot"] = {
                key: new[key]
                for key in (
                    "max_unique_per_forward",
                    "mean_unique_coverage",
                    "per_layer_max_unique_per_forward",
                )
            }
            delta["policy_snapshot"] = numbers(new["policy"], POLICY_GAUGES)
            delta["kernel_config_records"] = public_stats(new).get(
                "kernel_config_records", {}
            )
        for key in COUNTERS:
            totals[key] += delta[key]
        for group, fields in (("policy", POLICY_COUNTERS), ("timing", TIMINGS)):
            for key in fields:
                totals[group][key] += delta[group][key]
        hits, misses = delta["policy"]["cache_hits"], delta["policy"]["cache_misses"]
        delta["policy"]["cache_hit_ratio"] = (
            hits / (hits + misses) if hits + misses else None
        )
        totals["per_rank"].append(delta)
    hits, misses = totals["policy"]["cache_hits"], totals["policy"]["cache_misses"]
    totals["policy"]["cache_hit_ratio"] = (
        hits / (hits + misses) if hits + misses else None
    )
    return totals


def validate_stats(
    raw: object, settings: Mapping[str, Any], workers: int, *, executed: bool = True
) -> None:
    settings = public_settings(settings)
    identity = settings["identity"]
    geometry = identity["geometry"]
    layers, experts = geometry["total_layers"], geometry["num_experts"]
    resident = layers * math.floor(experts * settings["resident_ratio"])
    slots = resident + settings["cache_slots"] + experts
    expert_bytes = 6 * geometry["hidden_size"] * geometry["intermediate_size"]
    for raw_row in ranked(raw, workers):
        row = public_stats(raw_row)
        expected = {
            "schema_version": 1,
            "identity": identity,
            "tensor_parallel_size": workers,
            "total_layers": layers,
            "num_experts": experts,
            "resident_slots": resident,
            "cache_slots": settings["cache_slots"],
            "physical_slots": slots,
            "ingress_slots": experts,
            "expert_bytes": expert_bytes,
            "host_source_bytes": layers * experts * expert_bytes,
            "pinned_gather_bytes": experts * expert_bytes,
            "gpu_pool_bytes": slots * expert_bytes,
            "gpu_resident_bytes": resident * expert_bytes,
            "gpu_cache_bytes": settings["cache_slots"] * expert_bytes,
            "gpu_ingress_bytes": experts * expert_bytes,
            "gpu_metadata_bytes": experts * 12,
            "pinned_metadata_bytes": experts * 12,
            "net_freed_bytes": (layers * experts - slots) * expert_bytes - experts * 12,
            "weights_verified": 0,
            "failed": False,
            **{
                key: settings[key]
                for key in ("cache_policy", "resident_ratio", "profile_sha256")
            },
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("worker cache identity, placement or capacity differs")
        if expected["net_freed_bytes"] <= 0:
            raise ValueError("cache point does not free routed tensor bytes")
        for key in ARRAYS:
            if len(row.get(key, [])) != layers:
                raise ValueError("per-layer cache evidence is incomplete")
        if sum(row["per_layer_unique_demands"]) != row["unique_demands"]:
            raise ValueError("per-layer demand accounting differs")
        validate_demand(row, expert_bytes)
        if executed and (
            not all(x > 0 for x in row["forward_counts"])
            or not row.get("kernel_config_records", {}).get("records")
        ):
            raise ValueError("cache forward or kernel evidence is missing")


def validate_demand(row: Mapping[str, Any], expert_bytes: int) -> None:
    policy = row["policy"]
    if (
        row["resident_hits"] + policy["cache_hits"] + policy["cache_misses"]
        != row["unique_demands"]
    ):
        raise ValueError("resident/cache/miss demand accounting differs")
    if row["h2d_bytes"] != policy["cache_misses"] * expert_bytes:
        raise ValueError("runtime miss transfer accounting differs")


def validate_triplet(runs: tuple[Mapping[str, Any], ...]) -> None:
    settings = runs[0]["contract"]["expert_cache"]
    if runs[0].get("storage_backend") != "native":
        raise ValueError("R must use native resident storage")
    workers = runs[0]["contract"]["tensor_parallel_size"]
    for native_row in ranked(runs[0]["expert_cache_stats"], workers):
        if native_row.get("storage_backend") != "native" or any(
            native_row.get(key) != 0 for key in COUNTERS
        ):
            raise ValueError("native resident control accounting differs")
    for row in runs[1:]:
        if row.get("storage_backend") != "expert-cache":
            raise ValueError("B/C must use expert cache storage")
        validate_stats(row["expert_cache_stats"], settings, workers)
        expert_bytes = row["expert_cache_stats"][0]["expert_bytes"]
        for rep in row["repetitions"]:
            for delta in ranked(rep["diagnostics"]["per_rank"], workers):
                if not delta.get("forward_counts") or not all(
                    x > 0 for x in delta["forward_counts"]
                ):
                    raise ValueError("measured per-layer forwards missing")
                validate_demand(delta, expert_bytes)


def public_diagnostics(raw: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        **numbers(raw, COUNTERS),
        "timing": numbers(raw.get("timing", {}), TIMINGS),
        "policy": numbers(raw.get("policy", {}), POLICY_COUNTERS),
    }
    hit_ratio = raw.get("policy", {}).get("cache_hit_ratio")
    out["policy"]["cache_hit_ratio"] = None if hit_ratio is None else number(hit_ratio)
    if "rank" in raw:
        out["rank"] = number(raw["rank"], integer=True)
    for key in (
        "forward_counts",
        "per_layer_unique_demands",
        "per_layer_mean_unique_coverage",
    ):
        if key in raw:
            out[key] = [None if x is None else number(x) for x in raw[key]]
    if "gauge_snapshot" in raw:
        gauges = raw["gauge_snapshot"]
        out["gauge_snapshot"] = numbers(
            gauges, ("max_unique_per_forward", "mean_unique_coverage")
        )
        out["gauge_snapshot"]["per_layer_max_unique_per_forward"] = [
            number(x, integer=True) for x in gauges["per_layer_max_unique_per_forward"]
        ]
    if "policy_snapshot" in raw:
        out["policy_snapshot"] = numbers(raw["policy_snapshot"], POLICY_GAUGES)
    if "kernel_config_records" in raw:
        out["kernel_config_records"] = public_stats(
            {"kernel_config_records": raw["kernel_config_records"]}
        )["kernel_config_records"]
    if "per_rank" in raw:
        out["per_rank"] = [public_diagnostics(row) for row in raw["per_rank"]]
    return out


def extend_public_run(raw: Mapping[str, Any], out: dict[str, Any]) -> None:
    out["storage_backend"] = raw["storage_backend"]
    source = raw.get("contract", {})
    if source.get("comparison_backend") == "expert-cache":
        out["contract"].update(
            {
                "comparison_backend": "expert-cache",
                "expert_cache": public_settings(source["expert_cache"]),
                **numbers(
                    source, ("calibration_count", "evaluation_pool_count"), integer=True
                ),
                "calibration_input_hashes_sha256": hash_value(
                    source["calibration_input_hashes_sha256"]
                ),
            }
        )
        if source.get("sampling_policy") == "heldout-token-hash-first-unique":
            out["contract"]["sampling_policy"] = source["sampling_policy"]
        if (
            source.get("split_scope")
            == "full-prompt-token-hashes; source-conversation-disjointness-not-claimed"
        ):
            out["contract"]["split_scope"] = source["split_scope"]
    out["expert_cache_stats"] = [
        public_stats(row) for row in raw.get("expert_cache_stats", [])
    ]
    for old, new in zip(raw.get("repetitions", []), out["repetitions"]):
        new["diagnostics"] = public_diagnostics(old.get("diagnostics", {}))
        new["memory"] = [
            numbers(
                row,
                (
                    "rank",
                    "total_gpu_bytes",
                    "free_gpu_bytes",
                    "torch_allocated_bytes",
                    "torch_reserved_bytes",
                    "torch_peak_allocated_bytes",
                    "torch_peak_reserved_bytes",
                    "kv_cache_allocated_bytes",
                    "num_gpu_blocks",
                ),
                integer=True,
            )
            for row in old.get("memory", [])
        ]
        if old.get("memory_peak_scope") == "measured-generate-after-synchronized-reset":
            new["memory_peak_scope"] = old["memory_peak_scope"]


def public_calibration(raw: Mapping[str, Any]) -> dict[str, Any]:
    if raw.get("status") != "complete":
        return {"status": "failed"}
    identity = public_identity(raw["identity"])
    geometry = identity["geometry"]
    profiles = []
    for row in ranked(raw["rank_profiles"], identity["tensor_parallel_size"]):
        if public_identity(row) != identity:
            raise ValueError("calibration worker identity differs")
        forwards = [number(x, integer=True) for x in row["forward_counts"]]
        counts = [[number(x) for x in layer] for layer in row["counts"]]
        if (
            len(forwards) != geometry["total_layers"]
            or len(counts) != len(forwards)
            or not all(forwards)
        ):
            raise ValueError("calibration layer coverage incomplete")
        if any(
            len(layer) != geometry["num_experts"] or any(x > forwards[i] for x in layer)
            for i, layer in enumerate(counts)
        ):
            raise ValueError("calibration demand geometry differs")
        profiles.append(
            {"rank": row["rank"], "counts": counts, "forward_counts": forwards}
        )
    return {
        "status": "complete",
        "identity": identity,
        "profile_sha256": hash_value(raw["profile_sha256"]),
        "calibration_input_hashes": [
            hash_value(x) for x in raw["calibration_input_hashes"]
        ],
        "rank_profiles": profiles,
    }
