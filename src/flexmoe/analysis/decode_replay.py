"""Fixed-capacity expert cache replay, not a prefetch or throughput simulator."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from flexmoe.runtime.expert_cache_policy import ExpertCachePolicy


def _window(rows: list[dict[str, Any]], layers: int, experts: int) -> bool:
    if not rows or len(rows) % layers:
        return False
    first = rows[0].get("step")
    if type(first) is not int or first < 0:
        return False
    for index, row in enumerate(rows):
        ids = row.get("demand_ids")
        if (
            row.get("status") != "complete"
            or row.get("step") != first + index // layers
            or row.get("layer") != index % layers
            or type(row.get("actual_batch")) is not int
            or row["actual_batch"] <= 0
            or not isinstance(ids, list)
            or not ids
            or any(
                type(expert) is not int or not 0 <= expert < experts for expert in ids
            )
            or ids != sorted(set(ids))
            or type(row.get("unique_misses")) is not int
            or row["unique_misses"] < 0
            or type(row.get("loaded_bytes")) is not int
            or row["loaded_bytes"] < 0
        ):
            return False
    return True


def _policy_replay(
    policy: ExpertCachePolicy, rows: list[dict[str, Any]]
) -> tuple[int, list[int]]:
    counts = []
    for row in rows:
        layer, ids = row["layer"], row["demand_ids"]
        policy.observe(layer, ids)
        misses = []
        protected = set()
        for expert in ids:
            if policy.resident_slot(layer, expert) is not None:
                continue
            if policy.lookup(layer, expert) is None:
                misses.append(expert)
            else:
                protected.add((layer, expert))
        for expert in misses:
            if policy.admit(layer, expert, protected) is not None:
                protected.add((layer, expert))
        counts.append(len(misses))
    return sum(counts), counts


def _future_aware(snapshot: dict[str, Any], rows: list[dict[str, Any]]) -> int:
    residents = [set(row) for row in snapshot["resident_ids"]]
    capacity = snapshot["cache_slots"]
    entries = {tuple(key) for key in snapshot["slots"] if key is not None}
    future: dict[tuple[int, int], deque[int]] = defaultdict(deque)
    for index, row in enumerate(rows):
        for expert in row["demand_ids"]:
            future[(row["layer"], expert)].append(index)
    misses = 0
    for index, row in enumerate(rows):
        layer = row["layer"]
        for expert in row["demand_ids"]:
            future[(layer, expert)].popleft()
        protected = {
            key for key in entries if key[0] == layer and key[1] in row["demand_ids"]
        }
        for expert in row["demand_ids"]:
            key = (layer, expert)
            if expert in residents[layer] or key in entries:
                continue
            misses += 1
            if not capacity:
                continue
            if len(entries) >= capacity:
                candidates = entries - protected
                if not candidates:
                    # The mandatory working set can use the existing ingress;
                    # this miss is not eliminated by the cache oracle.
                    continue
                victim = max(
                    candidates,
                    key=lambda item: (
                        future[item][0] if future[item] else float("inf"),
                        item,
                    ),
                )
                # A one-off demanded key is still loaded through ingress. Do
                # not evict an entry used sooner just to persist this one-off.
                incoming_next = future[key][0] if future[key] else float("inf")
                victim_next = future[victim][0] if future[victim] else float("inf")
                if incoming_next >= victim_next:
                    continue
                entries.remove(victim)
            entries.add(key)
            protected.add(key)
    return misses


def replay_trace(
    snapshot: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    expert_bytes: int,
    layers: int,
) -> dict[str, Any]:
    if (
        type(expert_bytes) is not int
        or expert_bytes <= 0
        or type(layers) is not int
        or layers <= 0
    ):
        raise ValueError("physical expert geometry is required")
    try:
        current = ExpertCachePolicy.from_replay_snapshot(snapshot)
        if snapshot["total_layers"] != layers or not _window(
            rows, layers, snapshot["num_experts"]
        ):
            return {"status": "invalid-window"}
        actual, per_layer = _policy_replay(current, rows)
        if any(
            count != row["unique_misses"] or count * expert_bytes != row["loaded_bytes"]
            for count, row in zip(per_layer, rows, strict=True)
        ):
            return {
                "status": "baseline-mismatch",
                "meaning": "cache state or demand window cannot reproduce real pool traffic",
            }
        lru, _ = _policy_replay(
            ExpertCachePolicy.from_replay_snapshot(snapshot, policy_override="lru"),
            rows,
        )
        oracle = _future_aware(snapshot, rows)
        return {
            "status": "baseline-reproduced",
            "complete_steps": len(rows) // layers,
            "layer_forwards": len(rows),
            "current_policy_misses": actual,
            "lru_misses": lru,
            "future_aware_misses": oracle,
            "current_payload_bytes": actual * expert_bytes,
            "future_aware_saved_bytes": (actual - oracle) * expert_bytes,
            "resident_ids": snapshot["resident_ids"],
            "cache_slots": snapshot["cache_slots"],
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
    if (
        observed.get("coverage_status") != "complete"
        or pool.get("dropped_rows") != 0
        or observed.get("captured_steps", 0) < 64
        or not pool.get("initial_policy_state")
    ):
        return {"status": "insufficient-window"}
    return replay_trace(
        pool["initial_policy_state"],
        pool.get("rows", []),
        expert_bytes=trace["geometry"]["expert_bytes"],
        layers=trace["geometry"]["total_layers"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay_file(args.profile), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
