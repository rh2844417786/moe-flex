"""Two-layer expert-window state and CUDA lifecycle contracts."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from queue import Queue
from threading import Condition, Lock, Thread
from typing import Protocol, cast

import torch

from flexmoe.errors import ConfigurationError, IntegrityError


class LayerState(str, Enum):
    UNMAPPED = "unmapped"
    LOADING = "loading"
    RESIDENT = "resident"
    EVICTING = "evicting"


_NEXT_STATE: dict[LayerState, LayerState] = {
    LayerState.UNMAPPED: LayerState.LOADING,
    LayerState.LOADING: LayerState.RESIDENT,
    LayerState.RESIDENT: LayerState.EVICTING,
    LayerState.EVICTING: LayerState.UNMAPPED,
}


def _validate_layer(layer_idx: int, total_layers: int) -> None:
    if type(layer_idx) is not int:
        raise TypeError("layer_idx must be an int")
    if type(total_layers) is not int:
        raise TypeError("total_layers must be an int")
    if total_layers < 2:
        raise ValueError("total_layers must be at least two")
    if not 0 <= layer_idx < total_layers:
        raise IndexError(
            f"layer_idx {layer_idx} is outside [0, {total_layers})"
        )


def target_recycle_layer(layer_idx: int, total_layers: int) -> int:
    """Return the layer whose physical slot is two positions behind."""

    _validate_layer(layer_idx, total_layers)
    return (layer_idx - 2) % total_layers


def should_recycle(layer_idx: int, iteration: int, total_layers: int) -> bool:
    """Skip only the two empty slots at the beginning of the first iteration."""

    _validate_layer(layer_idx, total_layers)
    if type(iteration) is not int:
        raise TypeError("iteration must be an int")
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    return iteration > 0 or layer_idx >= 2


class LayerStateMachine:
    """Thread-safe fail-closed state machine for one tensor-kind window."""

    def __init__(self, total_layers: int) -> None:
        _validate_layer(0, total_layers)
        self._states = [LayerState.UNMAPPED] * total_layers
        self._lock = Lock()

    @property
    def total_layers(self) -> int:
        return len(self._states)

    def transition(self, layer_idx: int, destination: LayerState) -> None:
        _validate_layer(layer_idx, self.total_layers)
        if not isinstance(destination, LayerState):
            raise TypeError("destination must be a LayerState")
        with self._lock:
            source = self._states[layer_idx]
            if _NEXT_STATE[source] is not destination:
                raise IntegrityError(
                    f"illegal layer transition {source.value} -> {destination.value}"
                )
            self._states[layer_idx] = destination

    def state(self, layer_idx: int) -> LayerState:
        _validate_layer(layer_idx, self.total_layers)
        with self._lock:
            return self._states[layer_idx]

    def snapshot(self) -> tuple[LayerState, ...]:
        with self._lock:
            return tuple(self._states)


class LayerMaterializer(Protocol):
    """Operations that one tensor-kind worker launches for a layer."""

    def evict(self, layer_idx: int) -> None: ...

    def materialize(self, layer_idx: int, load_stream: int) -> None: ...


class CUDAStreamLike(Protocol):
    @property
    def cuda_stream(self) -> int: ...


class _NativeStreamLifecycle(Protocol):
    load_stream: int

    def record_load_done(self, layer_idx: int) -> None: ...

    def wait_load_done(self, layer_idx: int, compute_stream: int) -> None: ...

    def record_compute_done(self, layer_idx: int, compute_stream: int) -> None: ...

    def synchronize_compute_done(self, layer_idx: int) -> None: ...

    def synchronize_load_stream(self) -> None: ...


class _NativeModule(Protocol):
    StreamLifecycle: Callable[[int, int], _NativeStreamLifecycle]


@lru_cache(maxsize=1)
def _native_module() -> _NativeModule:
    try:
        module = importlib.import_module("flexmoe._C")
    except ImportError as error:
        raise RuntimeError(
            "flexmoe._C is unavailable; build the CUDA extension on Linux"
        ) from error
    if getattr(module, "StreamLifecycle", None) is None:
        raise RuntimeError("flexmoe._C does not expose StreamLifecycle")
    return cast(_NativeModule, module)


@dataclass(frozen=True)
class _LoadJob:
    layer_idx: int
    recycle_layer: int | None


class _TensorKindWorker:
    def __init__(
        self,
        *,
        kind: str,
        device: int,
        total_layers: int,
        materializer: LayerMaterializer,
    ) -> None:
        self.kind = kind
        self._device = device
        self._materializer = materializer
        self._states = LayerStateMachine(total_layers)
        self._native = _native_module().StreamLifecycle(device, total_layers)
        self._jobs: Queue[_LoadJob | None] = Queue()
        self._condition = Condition()
        self._error: Exception | None = None
        self._closed = False
        self._thread = Thread(
            target=self._run,
            name=f"flexmoe-{kind}-loader",
            daemon=True,
        )
        self._thread.start()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise IntegrityError(
                f"{self.kind} lifecycle worker failed: {self._error}"
            ) from self._error

    def check(self) -> None:
        with self._condition:
            self._raise_if_failed()
            if self._closed:
                raise RuntimeError(f"{self.kind} lifecycle worker is closed")

    def schedule(self, job: _LoadJob) -> None:
        self.check()
        self._jobs.put(job)

    def _run(self) -> None:
        try:
            torch.cuda.set_device(self._device)
        except Exception as error:  # noqa: BLE001 - cross-thread error handoff
            with self._condition:
                self._error = error
                self._condition.notify_all()

        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                with self._condition:
                    failed = self._error is not None
                if not failed:
                    self._execute(job)
            except Exception as error:  # noqa: BLE001 - cross-thread error handoff
                with self._condition:
                    if self._error is None:
                        self._error = error
                    self._condition.notify_all()
            finally:
                self._jobs.task_done()

    def _execute(self, job: _LoadJob) -> None:
        if job.recycle_layer is not None:
            self._states.transition(job.recycle_layer, LayerState.EVICTING)
            self._native.synchronize_compute_done(job.recycle_layer)
            self._materializer.evict(job.recycle_layer)
            self._states.transition(job.recycle_layer, LayerState.UNMAPPED)

        self._states.transition(job.layer_idx, LayerState.LOADING)
        self._materializer.materialize(job.layer_idx, self._native.load_stream)
        self._native.record_load_done(job.layer_idx)
        self._states.transition(job.layer_idx, LayerState.RESIDENT)
        with self._condition:
            self._condition.notify_all()

    def ensure_ready(self, layer_idx: int, compute_stream: int) -> None:
        with self._condition:
            while self._states.state(layer_idx) is not LayerState.RESIDENT:
                self._raise_if_failed()
                self._condition.wait()
            self._raise_if_failed()
        self._native.wait_load_done(layer_idx, compute_stream)

    def mark_consumed(self, layer_idx: int, compute_stream: int) -> None:
        self.check()
        if self._states.state(layer_idx) is not LayerState.RESIDENT:
            raise IntegrityError(
                f"cannot consume non-resident {self.kind} layer {layer_idx}"
            )
        self._native.record_compute_done(layer_idx, compute_stream)

    def snapshot(self) -> tuple[LayerState, ...]:
        return self._states.snapshot()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
        self._jobs.join()
        self._jobs.put(None)
        self._thread.join()
        self._native.synchronize_load_stream()
        with self._condition:
            self._raise_if_failed()


def _stream_handle(compute_stream: CUDAStreamLike) -> int:
    handle = compute_stream.cuda_stream
    if type(handle) is not int:
        raise TypeError("compute_stream.cuda_stream must be an int")
    if handle < 0:
        raise ValueError("compute stream handle must be non-negative")
    return handle


class LayerLifecycle:
    """Coordinate two physical layer slots for both routed-expert tensors."""

    def __init__(
        self,
        *,
        device: int,
        total_layers: int,
        materializers: Mapping[str, LayerMaterializer],
    ) -> None:
        _validate_layer(0, total_layers)
        if type(device) is not int:
            raise TypeError("device must be an int")
        if device < 0:
            raise ValueError("device must be non-negative")
        if set(materializers) != {"w13", "w2"}:
            raise ConfigurationError(
                "materializers must contain exactly the w13 and w2 tensor kinds"
            )
        self._total_layers = total_layers
        self._next_layer = 0
        self._iteration = 0
        self._closed = False
        self._lock = Lock()
        self._workers = {
            kind: _TensorKindWorker(
                kind=kind,
                device=device,
                total_layers=total_layers,
                materializer=materializers[kind],
            )
            for kind in sorted(materializers)
        }

    def schedule_next(self, layer_idx: int) -> None:
        _validate_layer(layer_idx, self._total_layers)
        with self._lock:
            if self._closed:
                raise RuntimeError("layer lifecycle is closed")
            if layer_idx != self._next_layer:
                raise IntegrityError(
                    f"expected layer {self._next_layer}, received layer {layer_idx}"
                )
            for worker in self._workers.values():
                worker.check()
            recycle_layer = (
                target_recycle_layer(layer_idx, self._total_layers)
                if should_recycle(
                    layer_idx, self._iteration, self._total_layers
                )
                else None
            )
            job = _LoadJob(layer_idx=layer_idx, recycle_layer=recycle_layer)
            for worker in self._workers.values():
                worker.schedule(job)
            self._next_layer = (layer_idx + 1) % self._total_layers
            if self._next_layer == 0:
                self._iteration += 1

    def ensure_ready(
        self, layer_idx: int, compute_stream: CUDAStreamLike
    ) -> None:
        _validate_layer(layer_idx, self._total_layers)
        stream = _stream_handle(compute_stream)
        for worker in self._workers.values():
            worker.ensure_ready(layer_idx, stream)

    def mark_consumed(
        self, layer_idx: int, compute_stream: CUDAStreamLike
    ) -> None:
        _validate_layer(layer_idx, self._total_layers)
        stream = _stream_handle(compute_stream)
        for worker in self._workers.values():
            worker.mark_consumed(layer_idx, stream)

    def snapshot(self) -> dict[str, tuple[LayerState, ...]]:
        return {
            kind: worker.snapshot() for kind, worker in self._workers.items()
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        errors: list[Exception] = []
        for worker in self._workers.values():
            try:
                worker.close()
            except Exception as error:  # noqa: BLE001 - close every worker
                errors.append(error)
        if errors:
            raise IntegrityError(f"failed to close layer lifecycle: {errors[0]}") from errors[
                0
            ]

__all__ = [
    "CUDAStreamLike",
    "LayerLifecycle",
    "LayerMaterializer",
    "LayerState",
    "LayerStateMachine",
    "should_recycle",
    "target_recycle_layer",
]
