"""JSON-safe expert cache decisions without tensor payloads."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


def _ids(value: tuple[int, ...], name: str) -> tuple[int, ...]:
    if any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{name} must contain nonnegative integer expert IDs")
    if tuple(sorted(set(value))) != value:
        raise ValueError(f"{name} must be sorted and unique")
    return value


@dataclass(frozen=True)
class CacheTraceRow:
    step_id: int
    layer_id: int
    actual_batch: int
    actual_expert_ids: tuple[int, ...]
    resident_hit_ids: tuple[int, ...]
    cache_hit_ids: tuple[int, ...]
    miss_ids: tuple[int, ...]
    bypass_ids: tuple[int, ...]
    admitted_ids: tuple[int, ...]
    evicted_ids: tuple[int, ...]
    cache_state_before: dict[str, object]
    cache_state_after: dict[str, object]
    loaded_bytes: int

    def __post_init__(self) -> None:
        for name in ("step_id", "layer_id"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.actual_batch) is not int or self.actual_batch <= 0:
            raise ValueError("actual_batch must be a positive integer")
        if type(self.loaded_bytes) is not int or self.loaded_bytes < 0:
            raise ValueError("loaded_bytes must be a nonnegative integer")
        for name in (
            "actual_expert_ids",
            "resident_hit_ids",
            "cache_hit_ids",
            "miss_ids",
            "bypass_ids",
            "admitted_ids",
            "evicted_ids",
        ):
            _ids(getattr(self, name), name)
        classified = (
            set(self.resident_hit_ids) | set(self.cache_hit_ids) | set(self.miss_ids)
        )
        if classified != set(self.actual_expert_ids):
            raise ValueError("resident/cache/miss IDs do not partition actual experts")
        if set(self.bypass_ids) | set(self.admitted_ids) != set(self.miss_ids):
            raise ValueError("bypass/admitted IDs do not partition misses")
        for state in (self.cache_state_before, self.cache_state_after):
            if "slots" not in state or "resident_ids" not in state:
                raise ValueError("cache trace state is incomplete")
            if any(key in state for key in ("weights", "w13", "w2", "payload")):
                raise ValueError("cache trace state cannot contain weight payloads")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "layer_id": self.layer_id,
            "actual_batch": self.actual_batch,
            "actual_expert_ids": list(self.actual_expert_ids),
            "resident_hit_ids": list(self.resident_hit_ids),
            "cache_hit_ids": list(self.cache_hit_ids),
            "miss_ids": list(self.miss_ids),
            "bypass_ids": list(self.bypass_ids),
            "admitted_ids": list(self.admitted_ids),
            "evicted_ids": list(self.evicted_ids),
            "cache_state_before": deepcopy(self.cache_state_before),
            "cache_state_after": deepcopy(self.cache_state_after),
            "loaded_bytes": self.loaded_bytes,
        }


__all__ = ["CacheTraceRow"]
