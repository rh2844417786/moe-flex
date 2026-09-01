from __future__ import annotations

import pytest
import torch

from flexmoe import _C

pytestmark = pytest.mark.cuda


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def test_compute_stream_waits_for_load_event() -> None:
    _require_cuda()
    lifecycle = _C.StreamLifecycle(device=0, total_layers=3)
    load_stream = torch.cuda.ExternalStream(lifecycle.load_stream, device=0)
    compute_stream = torch.cuda.Stream(device=0)
    value = torch.zeros(1, dtype=torch.int64, device="cuda:0")

    with torch.cuda.stream(load_stream):
        value.fill_(41)
    lifecycle.record_load_done(0)
    lifecycle.wait_load_done(0, compute_stream.cuda_stream)
    with torch.cuda.stream(compute_stream):
        value.add_(1)
    lifecycle.record_compute_done(0, compute_stream.cuda_stream)
    lifecycle.synchronize_compute_done(0)

    assert value.item() == 42


def test_unrecorded_event_wait_fails_closed() -> None:
    _require_cuda()
    lifecycle = _C.StreamLifecycle(device=0, total_layers=3)
    compute_stream = torch.cuda.Stream(device=0)

    with pytest.raises(RuntimeError, match="load event has not been recorded"):
        lifecycle.wait_load_done(0, compute_stream.cuda_stream)
    with pytest.raises(RuntimeError, match="compute event has not been recorded"):
        lifecycle.synchronize_compute_done(0)


def test_ten_thousand_cyclic_transitions_are_ordered() -> None:
    _require_cuda()
    total_layers = 3
    lifecycle = _C.StreamLifecycle(device=0, total_layers=total_layers)
    load_stream = torch.cuda.ExternalStream(lifecycle.load_stream, device=0)
    compute_stream = torch.cuda.Stream(device=0)
    values = torch.zeros(total_layers, dtype=torch.int64, device="cuda:0")

    for step in range(10_000):
        layer_idx = step % total_layers
        with torch.cuda.stream(load_stream):
            values[layer_idx].fill_(step)
        lifecycle.record_load_done(layer_idx)
        lifecycle.wait_load_done(layer_idx, compute_stream.cuda_stream)
        with torch.cuda.stream(compute_stream):
            values[layer_idx].add_(1)
        lifecycle.record_compute_done(layer_idx, compute_stream.cuda_stream)
        lifecycle.synchronize_compute_done(layer_idx)

    expected = torch.tensor(
        [10_000, 9998, 9999], dtype=torch.int64, device="cuda:0"
    )
    assert torch.equal(values, expected)
