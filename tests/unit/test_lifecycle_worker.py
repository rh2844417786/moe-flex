from __future__ import annotations

from dataclasses import dataclass, field

import pytest

import flexmoe.runtime.lifecycle as lifecycle_module
from flexmoe.errors import IntegrityError
from flexmoe.runtime.lifecycle import LayerLifecycle, LayerState


class _FakeNativeLifecycle:
    def __init__(self, device: int, total_layers: int) -> None:
        self.device = device
        self.total_layers = total_layers
        self.load_stream = 1000 + device
        self.calls: list[tuple[object, ...]] = []

    def record_load_done(self, layer_idx: int) -> None:
        self.calls.append(("record_load_done", layer_idx))

    def wait_load_done(self, layer_idx: int, stream: int) -> None:
        self.calls.append(("wait_load_done", layer_idx, stream))

    def record_compute_done(self, layer_idx: int, stream: int) -> None:
        self.calls.append(("record_compute_done", layer_idx, stream))

    def synchronize_compute_done(self, layer_idx: int) -> None:
        self.calls.append(("synchronize_compute_done", layer_idx))

    def synchronize_load_stream(self) -> None:
        self.calls.append(("synchronize_load_stream",))


class _FakeNativeModule:
    def __init__(self) -> None:
        self.instances: list[_FakeNativeLifecycle] = []

    def StreamLifecycle(
        self, device: int, total_layers: int
    ) -> _FakeNativeLifecycle:
        instance = _FakeNativeLifecycle(device, total_layers)
        self.instances.append(instance)
        return instance


@dataclass
class _Materializer:
    calls: list[tuple[object, ...]] = field(default_factory=list)

    def evict(self, layer_idx: int) -> None:
        self.calls.append(("evict", layer_idx))

    def materialize(self, layer_idx: int, load_stream: int) -> None:
        self.calls.append(("materialize", layer_idx, load_stream))


class _FailingMaterializer(_Materializer):
    def materialize(self, layer_idx: int, load_stream: int) -> None:
        raise RuntimeError(f"injected materialization failure for {layer_idx}")


@dataclass(frozen=True)
class _Stream:
    cuda_stream: int


@pytest.fixture
def fake_native(monkeypatch: pytest.MonkeyPatch) -> _FakeNativeModule:
    module = _FakeNativeModule()
    monkeypatch.setattr(lifecycle_module, "_native_module", lambda: module)
    monkeypatch.setattr(lifecycle_module.torch.cuda, "set_device", lambda _: None)
    return module


def test_workers_preserve_two_layer_recycle_order(
    fake_native: _FakeNativeModule,
) -> None:
    w13 = _Materializer()
    w2 = _Materializer()
    compute_stream = _Stream(cuda_stream=77)
    lifecycle = LayerLifecycle(
        device=0,
        total_layers=3,
        materializers={"w13": w13, "w2": w2},
    )
    try:
        lifecycle.schedule_next(0)
        lifecycle.schedule_next(1)
        lifecycle.ensure_ready(0, compute_stream)
        lifecycle.mark_consumed(0, compute_stream)
        lifecycle.schedule_next(2)
        lifecycle.ensure_ready(1, compute_stream)
        lifecycle.mark_consumed(1, compute_stream)
        lifecycle.schedule_next(0)
        lifecycle.ensure_ready(2, compute_stream)
        lifecycle.ensure_ready(0, compute_stream)

        snapshots = lifecycle.snapshot()
        assert snapshots["w13"][0] is LayerState.RESIDENT
        assert snapshots["w13"][1] is LayerState.UNMAPPED
        assert snapshots["w13"][2] is LayerState.RESIDENT
    finally:
        lifecycle.close()

    expected = [
        ("materialize", 0, 1000),
        ("materialize", 1, 1000),
        ("evict", 0),
        ("materialize", 2, 1000),
        ("evict", 1),
        ("materialize", 0, 1000),
    ]
    assert w13.calls == expected
    assert w2.calls == expected
    assert len(fake_native.instances) == 2


def test_scheduler_rejects_out_of_order_layer(
    fake_native: _FakeNativeModule,
) -> None:
    lifecycle = LayerLifecycle(
        device=0,
        total_layers=3,
        materializers={"w13": _Materializer(), "w2": _Materializer()},
    )
    try:
        with pytest.raises(IntegrityError, match="expected layer 0"):
            lifecycle.schedule_next(1)
    finally:
        lifecycle.close()


def test_worker_failure_is_propagated_without_hanging(
    fake_native: _FakeNativeModule,
) -> None:
    lifecycle = LayerLifecycle(
        device=0,
        total_layers=3,
        materializers={"w13": _FailingMaterializer(), "w2": _Materializer()},
    )
    lifecycle.schedule_next(0)

    with pytest.raises(IntegrityError, match="injected materialization failure"):
        lifecycle.ensure_ready(0, _Stream(cuda_stream=77))
    with pytest.raises(IntegrityError, match="failed to close"):
        lifecycle.close()
