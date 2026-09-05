"""CPU-only hot-expert residency and shared persistent-cache policy."""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from collections.abc import Set as AbstractSet

from flexmoe.runtime.expert_profile import select_resident_ids

ExpertKey = tuple[int, int]
Admission = tuple[int, ExpertKey | None]
HeapEntry = tuple[float, int, int, int, int, ExpertKey]

_POLICIES = frozenset({"decayed-lfu", "lru"})
_DECAY_FACTOR = 0.5
_MIN_LAZY_SCALE = 2.0**-32


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _bounded_cache_slots(value: object, *, key_space: int) -> int:
    slots = _non_negative_int(value, "cache_slots")
    if slots > key_space:
        raise ValueError("cache_slots cannot exceed total_layers * num_experts")
    return slots


def _ratio(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("resident_ratio must be a real number")
    ratio = float(value)
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("resident_ratio must be finite and between 0 and 1")
    return ratio


def _counts(
    value: Sequence[Sequence[float]] | None,
    *,
    total_layers: int,
    num_experts: int,
) -> list[list[float]]:
    if value is None:
        return [[0.0] * num_experts for _ in range(total_layers)]
    if len(value) != total_layers or any(len(row) != num_experts for row in value):
        raise ValueError("initial_counts shape does not match policy geometry")
    result: list[list[float]] = []
    for layer, row in enumerate(value):
        converted: list[float] = []
        for expert, raw in enumerate(row):
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(
                    f"initial_counts[{layer}][{expert}] must be a real number"
                )
            count = float(raw)
            if not math.isfinite(count):
                raise ValueError("initial_counts must contain only finite values")
            if count < 0.0:
                raise ValueError("initial_counts must be non-negative")
            converted.append(count)
        result.append(converted)
    return result


class ExpertCachePolicy:
    """Plan resident/cache metadata without owning tensors or synchronization.

    A victim heap is prepared once per observed/admission layer. Admissions are
    then amortized ``O(log(cache_slots))``; monotonically protected candidates
    are skipped once per layer batch. The decayed-LFU clock advances from the
    minimum per-layer forward count so layer position cannot age heat early.
    """

    def __init__(
        self,
        total_layers: int,
        num_experts: int,
        resident_ratio: float,
        cache_slots: int,
        decay_interval: int = 64,
        policy: str = "decayed-lfu",
        initial_counts: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self._total_layers = _positive_int(total_layers, "total_layers")
        self._num_experts = _positive_int(num_experts, "num_experts")
        self._resident_ratio = _ratio(resident_ratio)
        self._cache_slots = _bounded_cache_slots(
            cache_slots, key_space=self._total_layers * self._num_experts
        )
        self._decay_interval = _positive_int(decay_interval, "decay_interval")
        if policy not in _POLICIES:
            raise ValueError(f"policy must be one of {sorted(_POLICIES)}")
        self._policy = policy
        self._scores = _counts(
            initial_counts,
            total_layers=self._total_layers,
            num_experts=self._num_experts,
        )
        self._scale = 1.0
        self._forward_counts = [0] * self._total_layers
        self._decay_events = 0
        self._resident_ids: tuple[tuple[int, ...], ...] = ()
        self._resident_ranks: tuple[dict[int, int], ...] = ()
        self._select_residents()

        self._assignments: dict[ExpertKey, int] = {}
        self._slots: list[ExpertKey | None] = [None] * self._cache_slots
        self._free_slots = list(range(self._cache_slots))
        heapq.heapify(self._free_slots)
        self._access_serial = 0
        self._access: dict[ExpertKey, int] = {}
        self._versions: dict[ExpertKey, int] = {}
        self._victim_heap: list[HeapEntry] = []
        self._heap_layer: int | None = None
        self._skipped_protected: set[ExpertKey] = set()

        self._observations = 0
        self._unique_demands = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._admissions = 0
        self._evictions = 0
        self._cache_bypasses = 0
        self._reconfigurations = 0

    @property
    def resident_ids(self) -> tuple[tuple[int, ...], ...]:
        return self._resident_ids

    def _validate_layer(self, layer: int) -> None:
        if type(layer) is not int or not 0 <= layer < self._total_layers:
            raise IndexError("layer is outside policy geometry")

    def _validate_expert(self, expert: int) -> None:
        if type(expert) is not int or not 0 <= expert < self._num_experts:
            raise IndexError("expert is outside policy geometry")

    def _validate_key(self, layer: int, expert: int) -> ExpertKey:
        self._validate_layer(layer)
        self._validate_expert(expert)
        return (layer, expert)

    def _select_residents(self) -> None:
        selected = select_resident_ids(self._scores, self._resident_ratio)
        self._resident_ids = selected
        self._resident_ranks = tuple(
            {expert: rank for rank, expert in enumerate(layer_ids)}
            for layer_ids in selected
        )

    def observe(self, layer: int, expert_ids: Sequence[int]) -> None:
        """Record one layer-forward's unique routed expert demand."""

        self._validate_layer(layer)
        unique = set(expert_ids)
        for expert in unique:
            self._validate_expert(expert)

        increment = 1.0 / self._scale
        for expert in unique:
            self._scores[layer][expert] += increment
        self._forward_counts[layer] += 1
        self._observations += 1
        self._unique_demands += len(unique)

        completed_forwards = min(self._forward_counts)
        decay_events = completed_forwards // self._decay_interval
        if decay_events > self._decay_events:
            elapsed = decay_events - self._decay_events
            self._scale *= _DECAY_FACTOR**elapsed
            self._decay_events = decay_events
            if self._scale < _MIN_LAZY_SCALE:
                self._renormalise_scores()
        self._heap_layer = None

    def _renormalise_scores(self) -> None:
        for row in self._scores:
            for expert, score in enumerate(row):
                row[expert] = score * self._scale
        self._scale = 1.0
        self._heap_layer = None

    def resident_slot(self, layer: int, expert: int) -> int | None:
        self._validate_key(layer, expert)
        rank = self._resident_ranks[layer].get(expert)
        if rank is None:
            return None
        residents_per_layer = len(self._resident_ids[layer])
        return layer * residents_per_layer + rank

    def lookup(self, layer: int, expert: int) -> int | None:
        key = self._validate_key(layer, expert)
        slot = self._assignments.get(key)
        if slot is None:
            self._cache_misses += 1
            return None
        self._cache_hits += 1
        if self._policy == "lru":
            self._touch_lru(key)
        return slot

    def _touch_lru(self, key: ExpertKey) -> None:
        self._access_serial += 1
        self._access[key] = self._access_serial
        self._versions[key] = self._versions.get(key, 0) + 1
        if self._heap_layer is not None:
            heapq.heappush(
                self._victim_heap, self._heap_entry(key, self._heap_layer)
            )
            self._compact_heap_if_needed()

    def _next_use_distance(self, current_layer: int, key_layer: int) -> int:
        distance = (key_layer - current_layer) % self._total_layers
        return self._total_layers if distance == 0 else distance

    def _heap_entry(self, key: ExpertKey, current_layer: int) -> HeapEntry:
        layer, expert = key
        priority = (
            float(self._access[key])
            if self._policy == "lru"
            else self._scores[layer][expert]
        )
        return (
            priority,
            -self._next_use_distance(current_layer, layer),
            layer,
            expert,
            self._versions[key],
            key,
        )

    def _prepare_heap(self, layer: int) -> None:
        self._victim_heap = [
            self._heap_entry(key, layer) for key in self._assignments
        ]
        heapq.heapify(self._victim_heap)
        self._heap_layer = layer
        self._skipped_protected.clear()

    def _compact_heap_if_needed(self) -> None:
        limit = max(2 * self._cache_slots, 1)
        if len(self._victim_heap) > limit and self._heap_layer is not None:
            self._prepare_heap(self._heap_layer)

    def _entry_is_current(self, entry: HeapEntry) -> bool:
        version = entry[4]
        key = entry[5]
        return (
            key in self._assignments
            and self._versions.get(key) == version
            and key not in self._skipped_protected
        )

    def _victim(
        self, layer: int, protected: AbstractSet[ExpertKey]
    ) -> tuple[ExpertKey, float] | None:
        if self._heap_layer != layer or not self._skipped_protected.issubset(
            protected
        ):
            self._prepare_heap(layer)
        while self._victim_heap:
            entry = heapq.heappop(self._victim_heap)
            if not self._entry_is_current(entry):
                continue
            key = entry[5]
            if key in protected:
                self._skipped_protected.add(key)
                continue
            return key, entry[0]
        return None

    def admit(
        self, layer: int, expert: int, protected: set[ExpertKey]
    ) -> Admission | None:
        """Synchronously plan a persistent slot assignment or cache bypass."""

        key = self._validate_key(layer, expert)
        existing = self._assignments.get(key)
        if existing is not None:
            return existing, None
        if expert in self._resident_ranks[layer] or self._cache_slots == 0:
            self._cache_bypasses += 1
            return None

        if self._free_slots:
            slot = heapq.heappop(self._free_slots)
            self._install(key, slot)
            return slot, None

        victim = self._victim(layer, protected)
        if victim is None:
            self._cache_bypasses += 1
            return None
        victim_key, victim_priority = victim
        if (
            self._policy == "decayed-lfu"
            and self._scores[layer][expert] <= victim_priority
        ):
            heapq.heappush(
                self._victim_heap,
                self._heap_entry(victim_key, layer),
            )
            self._cache_bypasses += 1
            return None

        slot = self._assignments.pop(victim_key)
        self._slots[slot] = None
        self._access.pop(victim_key, None)
        self._versions.pop(victim_key, None)
        self._evictions += 1
        self._install(key, slot)
        return slot, victim_key

    def _install(self, key: ExpertKey, slot: int) -> None:
        self._assignments[key] = slot
        self._slots[slot] = key
        self._versions[key] = 0
        self._admissions += 1
        if self._policy == "lru":
            self._touch_lru(key)
        elif self._heap_layer is not None:
            heapq.heappush(
                self._victim_heap, self._heap_entry(key, self._heap_layer)
            )

    def reconfigure(self, resident_ratio: float, cache_slots: int) -> None:
        """Preserve heat while replacing residents and clearing assignments."""

        next_ratio = _ratio(resident_ratio)
        next_slots = _bounded_cache_slots(
            cache_slots, key_space=self._total_layers * self._num_experts
        )
        self._resident_ratio = next_ratio
        self._cache_slots = next_slots
        self._select_residents()
        self._assignments.clear()
        self._slots = [None] * next_slots
        self._free_slots = list(range(next_slots))
        heapq.heapify(self._free_slots)
        self._access.clear()
        self._versions.clear()
        self._victim_heap.clear()
        self._heap_layer = None
        self._skipped_protected.clear()
        self._reconfigurations += 1

    def stats(self) -> dict[str, int | float]:
        attempts = self._cache_hits + self._cache_misses
        return {
            "observations": self._observations,
            "unique_demands": self._unique_demands,
            "completed_forwards": min(self._forward_counts),
            "decay_events": self._decay_events,
            "resident_experts": sum(len(ids) for ids in self._resident_ids),
            "cache_slots": self._cache_slots,
            "cache_entries": len(self._assignments),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_ratio": self._cache_hits / attempts if attempts else 0.0,
            "admissions": self._admissions,
            "evictions": self._evictions,
            "cache_bypasses": self._cache_bypasses,
            "reconfigurations": self._reconfigurations,
        }


__all__ = ["Admission", "ExpertCachePolicy", "ExpertKey"]
