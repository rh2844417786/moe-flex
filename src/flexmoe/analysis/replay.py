"""Deterministic CPU replay of layer-local expert demand traces."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import defaultdict
from typing import TypeAlias, cast

from .schema import DemandTrace, ReplayConfig, ReplayResult

ExpertKey: TypeAlias = tuple[int, int]
_COUNTER_FIELDS = (
    "demands",
    "resident_hits",
    "cache_hits",
    "loaded_experts",
    "loaded_bytes",
)
_LFU_DECAY = 0.5


def _ratio(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a finite real number")
    result = float(cast(float | int, value))
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _selection_compatibility(calibration: DemandTrace, evaluation: DemandTrace) -> None:
    if not isinstance(calibration, DemandTrace) or not isinstance(
        evaluation, DemandTrace
    ):
        raise TypeError("calibration and evaluation must be DemandTrace values")
    if (
        calibration.rank,
        calibration.total_layers,
        calibration.num_experts,
        calibration.top_k,
        calibration.expert_bytes,
    ) != (
        evaluation.rank,
        evaluation.total_layers,
        evaluation.num_experts,
        evaluation.top_k,
        evaluation.expert_bytes,
    ):
        raise ValueError("calibration and evaluation trace geometry differs")
    for name in (
        "model_identity_sha256",
        "model_config_sha256",
        "dataset_sha256",
        "commit",
        "tensor_parallel_size",
    ):
        if calibration.contract[name] != evaluation.contract[name]:
            raise ValueError(f"calibration and evaluation {name} differs")
    if calibration.contract["input_sha256"] == evaluation.contract["input_sha256"]:
        raise ValueError("calibration and evaluation input_sha256 must differ")
    calibration_prompts = set(
        cast(tuple[str, ...], calibration.contract["prompt_hashes"])
    )
    evaluation_prompts = set(
        cast(tuple[str, ...], evaluation.contract["prompt_hashes"])
    )
    if not calibration_prompts.isdisjoint(evaluation_prompts):
        raise ValueError("calibration and evaluation prompt_hashes overlap")


def select_resident(
    calibration: DemandTrace,
    evaluation: DemandTrace,
    offload_fraction: float,
) -> tuple[tuple[int, ...], ...]:
    """Select layer-local residents from a disjoint calibration trace."""

    _selection_compatibility(calibration, evaluation)
    fraction = _ratio(offload_fraction, "offload_fraction")
    resident_count = math.ceil(calibration.num_experts * (1.0 - fraction))
    counts: list[defaultdict[int, int]] = [
        defaultdict(int) for _ in range(calibration.total_layers)
    ]
    for event in calibration.events:
        for expert in event.experts:
            counts[event.layer][expert] += 1
    return tuple(
        tuple(
            sorted(
                sorted(
                    range(calibration.num_experts),
                    key=lambda expert: (-row[expert], expert),
                )[:resident_count]
            )
        )
        for row in counts
    )


def _empty_totals() -> dict[str, int]:
    return {name: 0 for name in _COUNTER_FIELDS}


def _add_totals(
    totals: dict[str, int],
    *,
    demands: int,
    resident_hits: int,
    cache_hits: int,
    loaded_experts: int,
    expert_bytes: int,
) -> None:
    totals["demands"] += demands
    totals["resident_hits"] += resident_hits
    totals["cache_hits"] += cache_hits
    totals["loaded_experts"] += loaded_experts
    totals["loaded_bytes"] += loaded_experts * expert_bytes


def _future_positions(trace: DemandTrace) -> dict[ExpertKey, tuple[int, ...]]:
    positions: dict[ExpertKey, list[int]] = defaultdict(list)
    for index, event in enumerate(trace.events):
        for expert in event.experts:
            positions[(event.layer, expert)].append(index)
    return {key: tuple(value) for key, value in positions.items()}


def _future_distance(
    key: ExpertKey, event_index: int, positions: dict[ExpertKey, tuple[int, ...]]
) -> float:
    rows = positions.get(key, ())
    offset = bisect_right(rows, event_index)
    if offset == len(rows):
        return math.inf
    return float(rows[offset])


def _retain_cache(
    candidates: set[ExpertKey],
    slots: int,
    policy: str,
    event_index: int,
    access: dict[ExpertKey, int],
    scores: dict[ExpertKey, tuple[float, int]],
    future: dict[ExpertKey, tuple[int, ...]] | None,
) -> set[ExpertKey]:
    if slots == 0:
        return set()
    if policy == "lru":
        ranked = sorted(candidates, key=lambda key: (-access.get(key, -1), key))
        return set(ranked[:slots])
    if policy == "decayed-lfu":
        ranked = sorted(
            candidates,
            key=lambda key: (-_lfu_value(scores, key, event_index), key),
        )
        return set(ranked[:slots])
    if future is None:
        raise AssertionError("future positions required by future policy")
    ranked = sorted(
        candidates,
        key=lambda key: (_future_distance(key, event_index, future), key),
    )
    return set(ranked[:slots])


def _lfu_value(
    scores: dict[ExpertKey, tuple[float, int]], key: ExpertKey, event_index: int
) -> float:
    score, updated_at = scores.get(key, (0.0, event_index))
    return score * _LFU_DECAY ** (event_index - updated_at)


def _reuse_gaps(trace: DemandTrace) -> dict[str, int | float | None]:
    previous: dict[ExpertKey, int] = {}
    observations = 0
    reuse_count = 0
    gap_sum = 0
    minimum: int | None = None
    maximum: int | None = None
    for event_index, event in enumerate(trace.events):
        for expert in event.experts:
            observations += 1
            key = (event.layer, expert)
            if key in previous:
                gap = event_index - previous[key]
                reuse_count += 1
                gap_sum += gap
                minimum = gap if minimum is None else min(minimum, gap)
                maximum = gap if maximum is None else max(maximum, gap)
            previous[key] = event_index
    return {
        "observations": observations,
        "reuse_count": reuse_count,
        "min_event_gap": minimum,
        "max_event_gap": maximum,
        "mean_event_gap": gap_sum / reuse_count if reuse_count else None,
    }


def _validate_config(trace: DemandTrace, config: ReplayConfig) -> None:
    if len(config.resident_experts) != trace.total_layers:
        raise ValueError("resident_experts rows differ from trace layers")
    for row in config.resident_experts:
        if any(expert >= trace.num_experts for expert in row):
            raise ValueError("resident expert is outside trace geometry")


def replay_trace(trace: DemandTrace, config: ReplayConfig) -> ReplayResult:
    """Replay unique per-event expert demand through one persistent cache."""

    if not isinstance(trace, DemandTrace) or not isinstance(config, ReplayConfig):
        raise TypeError("trace and config must use analysis schema types")
    _validate_config(trace, config)
    residents = {
        (layer, expert)
        for layer, row in enumerate(config.resident_experts)
        for expert in row
    }
    cache: set[ExpertKey] = set()
    access: dict[ExpertKey, int] = {}
    scores: dict[ExpertKey, tuple[float, int]] = {}
    future = _future_positions(trace) if config.policy == "future" else None
    layer_totals = [_empty_totals() for _ in range(trace.total_layers)]
    phase_totals: dict[str, dict[str, int]] = {}
    event_load_counts: list[int] = []
    event_load_bytes: list[int] = []
    total_demands = 0
    total_resident_hits = 0
    total_cache_hits = 0
    total_loaded = 0
    staging_overflow_events = 0

    for event_index, event in enumerate(trace.events):
        keys = {(event.layer, expert) for expert in event.experts}
        resident_hits = len(keys & residents)
        nonresident = keys - residents
        cache_hits = len(nonresident & cache)
        misses = sorted(nonresident - cache)
        loaded = len(misses)
        if loaded > config.staging_experts:
            staging_overflow_events += 1

        if config.policy == "decayed-lfu":
            for key in nonresident:
                scores[key] = (_lfu_value(scores, key, event_index) + 1.0, event_index)
        for key in sorted(nonresident):
            access[key] = event_index
        cache = _retain_cache(
            cache | nonresident,
            config.cache_slots,
            config.policy,
            event_index,
            access,
            scores,
            future,
        )

        demands = len(keys)
        total_demands += demands
        total_resident_hits += resident_hits
        total_cache_hits += cache_hits
        total_loaded += loaded
        event_load_counts.append(loaded)
        event_load_bytes.append(loaded * trace.expert_bytes)
        phase = phase_totals.setdefault(event.phase, _empty_totals())
        _add_totals(
            layer_totals[event.layer],
            demands=demands,
            resident_hits=resident_hits,
            cache_hits=cache_hits,
            loaded_experts=loaded,
            expert_bytes=trace.expert_bytes,
        )
        _add_totals(
            phase,
            demands=demands,
            resident_hits=resident_hits,
            cache_hits=cache_hits,
            loaded_experts=loaded,
            expert_bytes=trace.expert_bytes,
        )

    loaded_bytes = total_loaded * trace.expert_bytes
    assumptions = [
        "one-load-per-unique-layer-expert-per-event",
        "event-start-hit-classification",
        "persistent-global-cache",
    ]
    if not trace.full_workload:
        assumptions.append("partial-window-not-full-workload")
    if staging_overflow_events:
        assumptions.append("staging-overflow-requires-unmodelled-chunking")
    if config.policy == "future":
        assumptions.append("ideal-future-reference-not-deployed")
        assumptions.append("not-a-strict-all-system-optimum")
    contract = trace.contract
    resident_count = sum(len(row) for row in config.resident_experts)
    net_freed = (
        trace.total_layers * trace.num_experts
        - resident_count
        - config.cache_slots
        - config.staging_experts
    ) * trace.expert_bytes - config.extra_gpu_bytes
    bytes_per_token = (
        loaded_bytes / trace.generated_tokens
        if trace.full_workload and trace.generated_tokens is not None
        else None
    )
    return ReplayResult(
        config=config,
        trace_id=trace.trace_id,
        rank=trace.rank,
        model_identity_sha256=cast(str, contract["model_identity_sha256"]),
        model_config_sha256=cast(str, contract["model_config_sha256"]),
        dataset_sha256=cast(str, contract["dataset_sha256"]),
        input_sha256=cast(str, contract["input_sha256"]),
        commit=cast(str, contract["commit"]),
        hardware_sha256=cast(str | None, contract["hardware_sha256"]),
        loaded_experts=total_loaded,
        loaded_bytes=loaded_bytes,
        resident_hits=total_resident_hits,
        cache_hits=total_cache_hits,
        demands=total_demands,
        bytes_per_generated_token=bytes_per_token,
        net_freed_bytes=net_freed,
        event_load_counts=tuple(event_load_counts),
        event_load_bytes=tuple(event_load_bytes),
        final_cache=tuple(sorted(cache)),
        per_layer_totals=tuple(layer_totals),
        per_phase_totals=phase_totals,
        reuse_gap_summary=_reuse_gaps(trace),
        staging_overflow_events=staging_overflow_events,
        future_policy_reference=config.policy == "future",
        assumptions=tuple(assumptions),
    )


__all__ = ["replay_trace", "select_resident"]
