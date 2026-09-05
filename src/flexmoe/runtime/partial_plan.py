"""Static, cyclically safe assignment of offloaded layers to staging slots."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PartialPlan:
    total_layers: int
    offload_layers: tuple[int, ...]
    staging_slots: int = 1

    def __post_init__(self) -> None:
        if type(self.total_layers) is not int or self.total_layers < 2:
            raise ValueError("total_layers must be an integer >= 2")
        if type(self.staging_slots) is not int or self.staging_slots not in (1, 2):
            raise ValueError("staging_slots must be 1 or 2")
        if type(self.offload_layers) is not tuple or any(
            type(layer) is not int or not 0 <= layer < self.total_layers
            for layer in self.offload_layers
        ):
            raise ValueError("offload_layers must be a tuple of valid layer indices")
        if tuple(sorted(set(self.offload_layers))) != self.offload_layers:
            raise ValueError("offload_layers must be sorted and unique")
        count = len(self.offload_layers)
        if count and (count <= self.staging_slots or count % self.staging_slots):
            raise ValueError(
                "offload count must exceed and be divisible by staging_slots"
            )

    @classmethod
    def evenly_spaced(
        cls, total_layers: int, offload_count: int, staging_slots: int = 1
    ) -> PartialPlan:
        if type(total_layers) is not int or total_layers < 2:
            raise ValueError("total_layers must be an integer >= 2")
        if type(offload_count) is not int or not 0 <= offload_count <= total_layers:
            raise ValueError("offload_count is outside the model")
        layers = tuple(
            (index + 1) * total_layers // offload_count - 1
            for index in range(offload_count)
        )
        return cls(total_layers, layers, staging_slots)

    @property
    def initial_prefetch(self) -> tuple[int, ...]:
        return self.offload_layers[: self.staging_slots]

    def slot_for(self, layer: int) -> int | None:
        if type(layer) is not int or not 0 <= layer < self.total_layers:
            raise ValueError("layer is outside the model")
        try:
            return self.offload_layers.index(layer) % self.staging_slots
        except ValueError:
            return None

    def prefetch_after(self, layer: int) -> int:
        slot = self.slot_for(layer)
        if slot is None:
            raise ValueError("a resident layer cannot trigger staging reuse")
        position = self.offload_layers.index(layer)
        return self.offload_layers[
            (position + self.staging_slots) % len(self.offload_layers)
        ]

    def net_freed_bytes(self, layer_bytes: int) -> int:
        if type(layer_bytes) is not int or layer_bytes <= 0:
            raise ValueError("layer_bytes must be positive")
        if not self.offload_layers:
            return 0
        return (len(self.offload_layers) - self.staging_slots) * layer_bytes
