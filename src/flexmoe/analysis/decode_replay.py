"""Exact fixed-budget cache replay; never a prefetch throughput simulator."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from flexmoe.runtime.expert_cache_policy import ExpertCachePolicy

_ID_FIELDS = (
    "actual_expert_ids",
    "resident_hit_ids",
    "cache_hit_ids",
    "miss_ids",
    "bypass_ids",
    "admitted_ids",
    "evicted_ids",
)
_TRANSITION_FIELDS = (
    "resident_hit_ids",
    "cache_hit_ids",
    "miss_ids",
    "bypass_ids",
    "admitted_ids",
    "evicted_ids",
    "cache_state_before",
    "cache_state_after",
    "loaded_bytes",
)


def _percentile(values: Sequence[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def _reuse(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    last: dict[tuple[int, int], int] = {}
    distances: list[int] = []
    for index, row in enumerate(rows):
        layer = cast(int, row["layer_id"])
        for expert in cast(list[int], row["actual_expert_ids"]):
            key = (layer, expert)
            if key in last:
                distances.append(index - last[key])
            last[key] = index
    histogram = {key: 0 for key in ("1", "2-4", "5-16", "17-64", "65+")}
    for distance in distances:
        bucket = (
            "1"
            if distance == 1
            else "2-4"
            if distance <= 4
            else "5-16"
            if distance <= 16
            else "17-64"
            if distance <= 64
            else "65+"
        )
        histogram[bucket] += 1
    return {
        "reuse_distance_histogram": histogram,
        "reuse_distance_p50": _percentile(distances, 0.50),
        "reuse_distance_p95": _percentile(distances, 0.95),
        "reuse_distance_p99": _percentile(distances, 0.99),
    }


def _ids(row: Mapping[str, Any], name: str, experts: int) -> list[int] | None:
    value = row.get(name)
    if (
        not isinstance(value, list)
        or any(type(expert) is not int or not 0 <= expert < experts for expert in value)
        or (name != "evicted_ids" and value != sorted(set(value)))
    ):
        return None
    return cast(list[int], value)


def _window(
    rows: list[dict[str, Any]], layers: int, experts: int, ingress_slots: int
) -> bool:
    if not rows or len(rows) % layers:
        return False
    first = rows[0].get("step_id")
    if type(first) is not int or first < 0:
        return False
    previous_after: object | None = None
    for index, row in enumerate(rows):
        id_values = {name: _ids(row, name, experts) for name in _ID_FIELDS}
        actual = id_values["actual_expert_ids"]
        residents = id_values["resident_hit_ids"]
        hits = id_values["cache_hit_ids"]
        misses = id_values["miss_ids"]
        bypasses = id_values["bypass_ids"]
        admissions = id_values["admitted_ids"]
        before, after = row.get("cache_state_before"), row.get("cache_state_after")
        if (
            row.get("step_id") != first + index // layers
            or row.get("layer_id") != index % layers
            or type(row.get("actual_batch")) is not int
            or row["actual_batch"] <= 0
            or any(value is None for value in id_values.values())
            or not actual
            or set(cast(list[int], residents))
            | set(cast(list[int], hits))
            | set(cast(list[int], misses))
            != set(actual)
            or set(cast(list[int], bypasses)) | set(cast(list[int], admissions))
            != set(cast(list[int], misses))
            or len(cast(list[int], misses)) > ingress_slots
            or not isinstance(before, dict)
            or not isinstance(after, dict)
            or type(row.get("loaded_bytes")) is not int
            or row["loaded_bytes"] < 0
            or (previous_after is not None and before != previous_after)
        ):
            return False
        previous_after = after
    return True


def _decision(
    policy: ExpertCachePolicy, layer: int, ids: Sequence[int]
) -> dict[str, Any]:
    before = policy.trace_snapshot()
    policy.observe(layer, ids)
    residents: list[int] = []
    hits: list[int] = []
    misses: list[int] = []
    protected: set[tuple[int, int]] = set()
    for expert in ids:
        if policy.resident_slot(layer, expert) is not None:
            residents.append(expert)
        elif policy.lookup(layer, expert) is None:
            misses.append(expert)
        else:
            hits.append(expert)
            protected.add((layer, expert))
    bypasses: list[int] = []
    admitted: list[int] = []
    evicted: list[int] = []
    for expert in misses:
        admission = policy.admit(layer, expert, protected)
        if admission is None:
            bypasses.append(expert)
        else:
            admitted.append(expert)
            if admission[1] is not None:
                evicted.append(admission[1][1])
            protected.add((layer, expert))
    return {
        "resident_hit_ids": residents,
        "cache_hit_ids": hits,
        "miss_ids": misses,
        "bypass_ids": bypasses,
        "admitted_ids": admitted,
        "evicted_ids": sorted(evicted),
        "cache_state_before": before,
        "cache_state_after": policy.trace_snapshot(),
    }


def _mismatch(row: Mapping[str, Any], field: str, replayed: object) -> dict[str, Any]:
    return {
        "status": "baseline-mismatch",
        "mismatch": {
            "step_id": row["step_id"],
            "layer_id": row["layer_id"],
            "field": field,
            "recorded": row[field],
            "replayed": replayed,
        },
        "meaning": "recorded production decisions must reproduce before cache controls are interpreted",
    }


def _summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    misses: int,
    bypasses: int,
    admissions: int,
    evictions: int,
    per_layer_misses: list[int],
    expert_bytes: int,
) -> dict[str, Any]:
    return {
        "misses": misses,
        "loaded_bytes": misses * expert_bytes,
        "bypasses": bypasses,
        "admissions": admissions,
        "evictions": evictions,
        "per_layer_misses": per_layer_misses,
        "per_layer_loaded_bytes": [value * expert_bytes for value in per_layer_misses],
        **_reuse(rows),
    }


def _policy_replay(
    policy: ExpertCachePolicy,
    rows: list[dict[str, Any]],
    *,
    layers: int,
    expert_bytes: int,
    compare: bool,
) -> dict[str, Any]:
    misses = bypasses = admissions = evictions = 0
    per_layer = [0] * layers
    for row in rows:
        decision = _decision(policy, row["layer_id"], row["actual_expert_ids"])
        decision["loaded_bytes"] = len(decision["miss_ids"]) * expert_bytes
        if compare:
            for field in _TRANSITION_FIELDS:
                if row[field] != decision[field]:
                    return _mismatch(row, field, decision[field])
        count = len(decision["miss_ids"])
        misses += count
        bypasses += len(decision["bypass_ids"])
        admissions += len(decision["admitted_ids"])
        evictions += len(decision["evicted_ids"])
        per_layer[row["layer_id"]] += count
    return {
        "status": "baseline-reproduced" if compare else "simulated-control",
        "summary": _summary(
            rows=rows,
            misses=misses,
            bypasses=bypasses,
            admissions=admissions,
            evictions=evictions,
            per_layer_misses=per_layer,
            expert_bytes=expert_bytes,
        ),
    }


def _future_aware(
    snapshot: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    layers: int,
    expert_bytes: int,
) -> dict[str, Any]:
    residents = [set(row) for row in snapshot["resident_ids"]]
    capacity = snapshot["cache_slots"]
    entries = {tuple(key) for key in snapshot["slots"] if key is not None}
    future: dict[tuple[int, int], deque[int]] = defaultdict(deque)
    for index, row in enumerate(rows):
        for expert in row["actual_expert_ids"]:
            future[(row["layer_id"], expert)].append(index)
    misses = bypasses = admissions = evictions = 0
    per_layer = [0] * layers
    for index, row in enumerate(rows):
        layer = row["layer_id"]
        for expert in row["actual_expert_ids"]:
            future[(layer, expert)].popleft()
        protected = {
            key
            for key in entries
            if key[0] == layer and key[1] in row["actual_expert_ids"]
        }
        for expert in row["actual_expert_ids"]:
            key = (layer, expert)
            if expert in residents[layer] or key in entries:
                continue
            misses += 1
            per_layer[layer] += 1
            if not capacity:
                bypasses += 1
                continue
            if len(entries) >= capacity:
                candidates = entries - protected
                if not candidates:
                    bypasses += 1
                    continue
                victim = max(
                    candidates,
                    key=lambda item: (
                        future[item][0] if future[item] else float("inf"),
                        item,
                    ),
                )
                incoming_next = future[key][0] if future[key] else float("inf")
                victim_next = future[victim][0] if future[victim] else float("inf")
                if incoming_next >= victim_next:
                    bypasses += 1
                    continue
                entries.remove(victim)
                evictions += 1
            entries.add(key)
            protected.add(key)
            admissions += 1
    return _summary(
        rows=rows,
        misses=misses,
        bypasses=bypasses,
        admissions=admissions,
        evictions=evictions,
        per_layer_misses=per_layer,
        expert_bytes=expert_bytes,
    )


def replay_trace(
    snapshot: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    expert_bytes: int,
    layers: int,
    ingress_slots: int,
) -> dict[str, Any]:
    if (
        type(expert_bytes) is not int
        or expert_bytes <= 0
        or type(layers) is not int
        or layers <= 0
        or type(ingress_slots) is not int
        or ingress_slots <= 0
    ):
        raise ValueError("physical expert and ingress geometry is required")
    try:
        if (
            snapshot.get("total_layers") != layers
            or snapshot.get("num_experts") is None
            or not _window(rows, layers, snapshot["num_experts"], ingress_slots)
            or rows[0]["cache_state_before"] != snapshot
        ):
            return {"status": "invalid-window"}
        current = _policy_replay(
            ExpertCachePolicy.from_replay_snapshot(snapshot),
            rows,
            layers=layers,
            expert_bytes=expert_bytes,
            compare=True,
        )
        if current["status"] != "baseline-reproduced":
            return current
        lru = _policy_replay(
            ExpertCachePolicy.from_replay_snapshot(snapshot, policy_override="lru"),
            rows,
            layers=layers,
            expert_bytes=expert_bytes,
            compare=False,
        )["summary"]
        oracle = _future_aware(snapshot, rows, layers=layers, expert_bytes=expert_bytes)
        current_summary = current["summary"]
        oracle["saved_loaded_bytes"] = (
            current_summary["loaded_bytes"] - oracle["loaded_bytes"]
        )
        return {
            "status": "baseline-reproduced",
            "complete_steps": len(rows) // layers,
            "layer_forwards": len(rows),
            "current": current_summary,
            "lru": lru,
            "future_aware": oracle,
            "current_policy_misses": current_summary["misses"],
            "lru_misses": lru["misses"],
            "future_aware_misses": oracle["misses"],
            "future_aware_saved_bytes": oracle["saved_loaded_bytes"],
            "resident_ids": snapshot["resident_ids"],
            "cache_slots": snapshot["cache_slots"],
            "ingress_slots": ingress_slots,
            "initial_cache_entries": sum(
                entry is not None for entry in snapshot["slots"]
            ),
            "lru_initial_recency": (
                "recorded"
                if snapshot["policy"] == "lru"
                else "unknown; slot-order seed, conditional control"
            ),
            "interpretation": "cache-eviction-only; no prefetch or throughput gain",
        }
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return {"status": "invalid-window", "error_type": type(exc).__name__}


def replay_file(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    if (
        trace.get("artifact_kind") != "decode-profile"
        or trace.get("profile") is not True
    ):
        return {"status": "invalid-profile"}
    observed = trace.get("observation", {})
    pool = observed.get("pool_profile", {})
    snapshot = pool.get("initial_policy_state")
    if (
        observed.get("coverage_status") != "complete"
        or pool.get("dropped_rows") != 0
        or observed.get("captured_steps", 0) < 64
        or not isinstance(snapshot, dict)
    ):
        return {"status": "insufficient-window"}
    return replay_trace(
        snapshot,
        pool.get("rows", []),
        expert_bytes=trace["geometry"]["expert_bytes"],
        layers=trace["geometry"]["total_layers"],
        ingress_slots=snapshot["num_experts"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay_file(args.profile), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
