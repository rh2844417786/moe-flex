"""Fixed BF16 staging with GPU-side ordering and bounded timing samples."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import torch


def should_sample_upload(
    upload_count: int,
    *,
    capacity: int,
    pending_count: int,
) -> bool:
    # 31 is coprime to the 2/4/6/8-layer candidate cycles, unlike a
    # power-of-two stride which would repeatedly sample the same slot.
    return upload_count % 31 == 0 and pending_count < capacity


class HostPair(Protocol):
    w13: torch.Tensor
    w2: torch.Tensor


class Stager(Protocol):
    pin_memory: bool

    def allocate(self, shape: tuple[int, ...]) -> torch.Tensor: ...

    def enqueue(
        self,
        slot: int,
        source: HostPair,
        destination: tuple[torch.Tensor, torch.Tensor],
        *,
        after_compute: bool,
    ) -> None: ...

    def begin(self, slot: int) -> object | None: ...

    def end(self, slot: int, ticket: object | None) -> None: ...

    def synchronize(self) -> None: ...

    def timing(self, *, reset: bool = False) -> dict[str, int | float]: ...


@dataclass
class _Timing:
    load_start: torch.cuda.Event
    load_end: torch.cuda.Event
    wait_start: torch.cuda.Event
    wait_end: torch.cuda.Event
    compute_end: torch.cuda.Event

    @classmethod
    def create(cls) -> _Timing:
        return cls(*(torch.cuda.Event(enable_timing=True) for _ in range(5)))  # type: ignore[no-untyped-call]


class CudaStager:
    """No background Python thread, per-copy receipts, or VMM remapping."""

    pin_memory = True

    def __init__(self, device: int, slots: int, timing_samples: int = 128) -> None:
        if type(timing_samples) is not int or timing_samples < 0:
            raise ValueError("timing_samples must be a non-negative integer")
        self.device = torch.device("cuda", device)
        with torch.cuda.device(self.device):
            self._load = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
            self._ready = [torch.cuda.Event() for _ in range(slots)]  # type: ignore[no-untyped-call]
            self._consumed = [torch.cuda.Event() for _ in range(slots)]  # type: ignore[no-untyped-call]
            self._last_compute = torch.cuda.current_stream(device)
        self._current: list[_Timing | None] = [None] * slots
        self._pending: deque[_Timing] = deque()
        self._timing_limit = timing_samples
        self._enqueue_count = 0
        self._totals: dict[str, int | float] = self._empty_timing()

    @staticmethod
    def _empty_timing() -> dict[str, int | float]:
        return {
            "sample_count": 0,
            "load_cuda_s": 0.0,
            "wait_cuda_s": 0.0,
            "compute_cuda_s": 0.0,
            "cpu_enqueue_s": 0.0,
        }

    def allocate(self, shape: tuple[int, ...]) -> torch.Tensor:
        # Allocate on the writer stream so recycled allocator blocks retain
        # their required ordering before the first side-stream H2D.
        with torch.cuda.device(self.device), torch.cuda.stream(self._load):
            return torch.empty(shape, dtype=torch.bfloat16, device=self.device)

    def enqueue(
        self,
        slot: int,
        source: HostPair,
        destination: tuple[torch.Tensor, torch.Tensor],
        *,
        after_compute: bool,
    ) -> None:
        started = perf_counter()
        self._collect()
        live_samples = len(self._pending) + sum(t is not None for t in self._current)
        sample = (
            _Timing.create()
            if should_sample_upload(
                self._enqueue_count,
                capacity=self._timing_limit,
                pending_count=live_samples,
            )
            else None
        )
        self._enqueue_count += 1
        with torch.cuda.device(self.device), torch.cuda.stream(self._load):
            if after_compute:
                self._load.wait_event(self._consumed[slot])
            if sample is not None:
                sample.load_start.record(self._load)  # type: ignore[no-untyped-call]
            destination[0].copy_(source.w13, non_blocking=True)
            destination[1].copy_(source.w2, non_blocking=True)
            if sample is not None:
                sample.load_end.record(self._load)  # type: ignore[no-untyped-call]
            self._ready[slot].record(self._load)  # type: ignore[no-untyped-call]
        self._current[slot] = sample
        self._totals["cpu_enqueue_s"] += perf_counter() - started

    def begin(self, slot: int) -> object | None:
        compute = torch.cuda.current_stream(self.device)
        self._last_compute = compute
        sample = self._current[slot]
        if sample is not None:
            sample.wait_start.record(compute)  # type: ignore[no-untyped-call]
        compute.wait_event(self._ready[slot])
        if sample is not None:
            sample.wait_end.record(compute)  # type: ignore[no-untyped-call]
        return sample

    def end(self, slot: int, ticket: object | None) -> None:
        compute = torch.cuda.current_stream(self.device)
        self._last_compute = compute
        if ticket is not None:
            if not isinstance(ticket, _Timing):
                raise TypeError("invalid CUDA timing ticket")
            ticket.compute_end.record(compute)  # type: ignore[no-untyped-call]
            self._pending.append(ticket)
        self._consumed[slot].record(compute)  # type: ignore[no-untyped-call]
        self._current[slot] = None

    def _collect(self) -> None:
        while self._pending and self._pending[0].compute_end.query():  # type: ignore[no-untyped-call]
            sample = self._pending.popleft()
            self._totals["sample_count"] += 1
            self._totals["load_cuda_s"] += (
                sample.load_start.elapsed_time(sample.load_end) / 1000  # type: ignore[no-untyped-call]
            )
            self._totals["wait_cuda_s"] += (
                sample.wait_start.elapsed_time(sample.wait_end) / 1000  # type: ignore[no-untyped-call]
            )
            self._totals["compute_cuda_s"] += (
                sample.wait_end.elapsed_time(sample.compute_end) / 1000  # type: ignore[no-untyped-call]
            )

    def synchronize(self) -> None:
        # Called at benchmark/teardown boundaries, never between model layers.
        self._last_compute.synchronize()
        self._load.synchronize()
        self._collect()

    def timing(self, *, reset: bool = False) -> dict[str, int | float]:
        self._collect()
        result = dict(self._totals)
        if reset:
            if self._pending:
                raise RuntimeError("synchronize before resetting CUDA timing")
            self._totals = self._empty_timing()
            self._current = [None] * len(self._current)
            self._enqueue_count = 0
        return result
