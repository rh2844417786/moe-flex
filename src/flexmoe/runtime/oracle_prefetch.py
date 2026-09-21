"""Bounded ingress ownership and real CUDA transfer dependencies for Oracle runs."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Protocol

import torch

OracleKey = tuple[int, int, int]
TensorCopy = tuple[torch.Tensor, torch.Tensor]


class IngressCapacityError(RuntimeError):
    pass


class OracleTransferBackend(Protocol):
    def enqueue(self, copies: Sequence[TensorCopy]) -> Any: ...
    def query(self, handle: Any) -> bool: ...
    def wait(self, handle: Any) -> Any | None: ...
    def elapsed_ms(self, handle: Any) -> float | None: ...
    def wait_elapsed_ms(self, handle: Any) -> float | None: ...
    def synchronize(self) -> None: ...


@dataclass(frozen=True)
class PrefetchTicket:
    step: int
    source_layer: int
    target_layer: int
    expert_ids: tuple[int, ...]
    slots: tuple[int, ...]
    handle: Any
    enqueued_ns: int


@dataclass(frozen=True)
class ResolveResult:
    prefetched_ids: tuple[int, ...]
    fallback_ids: tuple[int, ...]
    ticket: PrefetchTicket | None
    ready_before_use_count: int
    required_prefetch_count: int
    layer_batch_all_ready: bool
    resolved_ns: int


@dataclass
class _Entry:
    slot: int
    state: str
    ticket: PrefetchTicket | None


class OracleIngress:
    """Own existing ingress slots without changing cache or eviction policy."""

    def __init__(
        self,
        *,
        capacity: int,
        expert_bytes: int,
        transfer_backend: OracleTransferBackend,
        copy_factory: Callable[[OracleKey, int], Sequence[TensorCopy]] | None = None,
        clock_ns: Callable[[], int] = perf_counter_ns,
    ) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("oracle ingress capacity must be positive")
        if type(expert_bytes) is not int or expert_bytes <= 0:
            raise ValueError("oracle expert_bytes must be positive")
        self.capacity = capacity
        self.expert_bytes = expert_bytes
        self.backend = transfer_backend
        self.copy_factory = copy_factory
        self.clock_ns = clock_ns
        self._free: set[int] = set(range(capacity))
        self._entries: dict[OracleKey, _Entry] = {}
        self._tickets: dict[tuple[int, int], PrefetchTicket] = {}
        self._scheduled_bytes = 0
        self._ready_before_use = 0
        self._required_prefetch = 0
        self._fallback = 0
        self._capacity_failures = 0
        self._eligible_layer_batches = 0
        self._all_ready_layer_batches = 0
        self._wait_handles: list[Any] = []
        self._released_without_use = 0

    @property
    def free_slots(self) -> tuple[int, ...]:
        return tuple(sorted(self._free))

    @property
    def entries(self) -> dict[OracleKey, int]:
        return {key: entry.slot for key, entry in self._entries.items()}

    def _ids(
        self, values: Sequence[int], *, allow_empty: bool = False
    ) -> tuple[int, ...]:
        ids = tuple(values)
        if (
            (not ids and not allow_empty)
            or any(type(expert) is not int or expert < 0 for expert in ids)
            or tuple(sorted(set(ids))) != ids
        ):
            raise ValueError("oracle expert IDs must be sorted unique nonnegative")
        return ids

    def schedule(
        self,
        *,
        step: int,
        source_layer: int,
        target_layer: int,
        expert_ids: Sequence[int],
    ) -> PrefetchTicket:
        if any(type(value) is not int or value < 0 for value in (step, source_layer)):
            raise ValueError("oracle step and source layer must be nonnegative")
        if type(target_layer) is not int or target_layer <= source_layer:
            raise ValueError("oracle target layer must follow source layer")
        ids = self._ids(expert_ids)
        keys = tuple((step, target_layer, expert) for expert in ids)
        if (step, target_layer) in self._tickets or any(
            key in self._entries for key in keys
        ):
            raise ValueError("oracle target layer or expert is already scheduled")
        if len(keys) > len(self._free):
            self._capacity_failures += 1
            raise IngressCapacityError(
                f"oracle ingress needs {len(keys)} slots, only {len(self._free)} free"
            )
        slots = tuple(sorted(self._free)[: len(keys)])
        copies: list[TensorCopy] = []
        if self.copy_factory is not None:
            for key, slot in zip(keys, slots, strict=True):
                copies.extend(self.copy_factory(key, slot))
        handle = self.backend.enqueue(tuple(copies))
        ticket = PrefetchTicket(
            step=step,
            source_layer=source_layer,
            target_layer=target_layer,
            expert_ids=ids,
            slots=slots,
            handle=handle,
            enqueued_ns=self.clock_ns(),
        )
        self._tickets[(step, target_layer)] = ticket
        for key, slot in zip(keys, slots, strict=True):
            self._free.remove(slot)
            self._entries[key] = _Entry(slot=slot, state="inflight", ticket=ticket)
        self._scheduled_bytes += len(keys) * self.expert_bytes
        return ticket

    def resolve(
        self, *, step: int, layer: int, required_ids: Sequence[int]
    ) -> ResolveResult:
        required = self._ids(required_ids, allow_empty=True)
        ticket = self._tickets.get((step, layer))
        prefetched = tuple(
            expert for expert in required if (step, layer, expert) in self._entries
        )
        fallback = tuple(expert for expert in required if expert not in prefetched)
        ready = False
        wait_handle = None
        if ticket is not None and prefetched:
            ready = self.backend.query(ticket.handle)
            if not ready:
                wait_handle = self.backend.wait(ticket.handle)
                if wait_handle is not None:
                    self._wait_handles.append(wait_handle)
            for expert in prefetched:
                entry = self._entries[(step, layer, expert)]
                if entry.state not in ("inflight", "ready"):
                    raise RuntimeError("oracle ingress expert is not resolvable")
                entry.state = "in_use"
        ready_count = len(prefetched) if ready else 0
        self._ready_before_use += ready_count
        self._required_prefetch += len(prefetched)
        self._fallback += len(fallback)
        all_ready = (
            ticket is not None
            and not fallback
            and (not prefetched or ready_count == len(prefetched))
        )
        if ticket is not None:
            self._eligible_layer_batches += 1
            self._all_ready_layer_batches += int(all_ready)
        return ResolveResult(
            prefetched_ids=prefetched,
            fallback_ids=fallback,
            ticket=ticket,
            ready_before_use_count=ready_count,
            required_prefetch_count=len(prefetched),
            layer_batch_all_ready=all_ready,
            resolved_ns=self.clock_ns(),
        )

    def reserve_on_demand(
        self, *, step: int, layer: int, expert_ids: Sequence[int]
    ) -> dict[int, int]:
        if any(type(value) is not int or value < 0 for value in (step, layer)):
            raise ValueError("on-demand step and layer must be nonnegative")
        ids = self._ids(expert_ids)
        keys = tuple((step, layer, expert) for expert in ids)
        if any(key in self._entries for key in keys):
            raise ValueError("on-demand expert already owns an ingress slot")
        if len(keys) > len(self._free):
            self._capacity_failures += 1
            raise IngressCapacityError(
                f"on-demand ingress needs {len(keys)} slots, only {len(self._free)} free"
            )
        slots = tuple(sorted(self._free)[: len(keys)])
        for key, slot in zip(keys, slots, strict=True):
            self._free.remove(slot)
            self._entries[key] = _Entry(slot=slot, state="in_use", ticket=None)
        return {expert: slot for expert, slot in zip(ids, slots, strict=True)}

    def finish_layer(self, *, step: int, layer: int) -> None:
        ticket = self._tickets.pop((step, layer), None)
        keys = [key for key in self._entries if key[:2] == (step, layer)]
        if ticket is not None and any(
            self._entries[key].state == "inflight" for key in keys
        ):
            self.backend.wait(ticket.handle)
        for key in keys:
            entry = self._entries.pop(key)
            if entry.state != "in_use":
                self._released_without_use += 1
            self._free.add(entry.slot)

    def stats(self) -> dict[str, Any]:
        exposed = [
            value
            for handle in self._wait_handles
            if (value := self.backend.wait_elapsed_ms(handle)) is not None
        ]
        return {
            "capacity": self.capacity,
            "entries": len(self._entries),
            "free_slots": len(self._free),
            "scheduled_bytes": self._scheduled_bytes,
            "ready_before_use_count": self._ready_before_use,
            "required_prefetch_count": self._required_prefetch,
            "fallback_on_demand_count": self._fallback,
            "capacity_fallback_count": self._capacity_failures,
            "eligible_layer_batch_count": self._eligible_layer_batches,
            "layer_batch_all_ready_count": self._all_ready_layer_batches,
            "layer_batch_all_ready_ratio": (
                self._all_ready_layer_batches / self._eligible_layer_batches
                if self._eligible_layer_batches
                else None
            ),
            "released_without_use_count": self._released_without_use,
            "exposed_wait_ms_samples": exposed,
            "exposed_wait_ms_p50": _percentile(exposed, 0.50),
            "exposed_wait_ms_p95": _percentile(exposed, 0.95),
            "exposed_wait_ms_p99": _percentile(exposed, 0.99),
            "ready_before_use_ratio": (
                self._ready_before_use / self._required_prefetch
                if self._required_prefetch
                else None
            ),
        }


@dataclass(frozen=True)
class CudaTransferHandle:
    start: Any | None
    end: Any


@dataclass(frozen=True)
class CudaWaitHandle:
    start: Any
    end: Any


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = int(position), math.ceil(position)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


class CudaOracleTransferBackend:
    def __init__(self, device: torch.device, *, enable_timing: bool) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA oracle transfer backend requires a CUDA device")
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise ValueError("CUDA oracle transfer backend requires NVIDIA CUDA")
        self.device = device
        self.enable_timing = bool(enable_timing)
        self.stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]

    def enqueue(self, copies: Sequence[TensorCopy]) -> CudaTransferHandle:
        for destination, source in copies:
            if (
                destination.device != self.device
                or source.device.type != "cpu"
                or destination.shape != source.shape
                or destination.dtype != source.dtype
            ):
                raise ValueError("oracle transfer tensor geometry or device differs")
        start = (
            torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            if self.enable_timing
            else None
        )
        end = torch.cuda.Event(enable_timing=self.enable_timing)  # type: ignore[no-untyped-call]
        with torch.cuda.stream(self.stream):
            if start is not None:
                start.record(self.stream)  # type: ignore[no-untyped-call]
            for destination, source in copies:
                destination.copy_(source, non_blocking=True)
            end.record(self.stream)  # type: ignore[no-untyped-call]
        return CudaTransferHandle(start=start, end=end)

    def query(self, handle: CudaTransferHandle) -> bool:
        return bool(handle.end.query())

    def wait(self, handle: CudaTransferHandle) -> CudaWaitHandle | None:
        stream = torch.cuda.current_stream(self.device)
        if not self.enable_timing:
            stream.wait_event(handle.end)
            return None
        start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        start.record(stream)  # type: ignore[no-untyped-call]
        stream.wait_event(handle.end)
        end.record(stream)  # type: ignore[no-untyped-call]
        return CudaWaitHandle(start=start, end=end)

    def elapsed_ms(self, handle: CudaTransferHandle) -> float | None:
        if handle.start is None or not self.query(handle):
            return None
        return float(handle.start.elapsed_time(handle.end))

    def wait_elapsed_ms(self, handle: CudaWaitHandle) -> float | None:
        if not handle.end.query():
            return None
        return float(handle.start.elapsed_time(handle.end))

    def synchronize(self) -> None:
        self.stream.synchronize()


__all__ = [
    "CudaOracleTransferBackend",
    "CudaTransferHandle",
    "CudaWaitHandle",
    "IngressCapacityError",
    "OracleIngress",
    "OracleTransferBackend",
    "PrefetchTicket",
    "ResolveResult",
]
