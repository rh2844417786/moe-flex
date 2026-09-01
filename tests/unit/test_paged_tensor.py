from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from flexmoe import paged_tensor
from flexmoe.paged_tensor import PagedTensorRegion


class _FakeRegion:
    def __init__(self, device: int, virtual_bytes: int) -> None:
        self.device = device
        self.base_address = 4096
        self.virtual_bytes = virtual_bytes
        self.granularity = 64
        self.calls: list[tuple[object, ...]] = []

    def create_block(self, nbytes: int) -> int:
        self.calls.append(("create_block", nbytes))
        return 7

    def map(self, offset: int, block_id: int, nbytes: int) -> None:
        self.calls.append(("map", offset, block_id, nbytes))

    def unmap(self, offset: int, nbytes: int) -> None:
        self.calls.append(("unmap", offset, nbytes))

    def tensor(
        self, offset: int, shape: Sequence[int], dtype_code: int
    ) -> torch.Tensor:
        self.calls.append(("tensor", offset, tuple(shape), dtype_code))
        return torch.empty(tuple(shape), dtype=torch.bfloat16)

    def snapshot(self) -> dict[str, int]:
        return {
            "device": self.device,
            "base_address": self.base_address,
            "virtual_bytes": self.virtual_bytes,
            "granularity": self.granularity,
            "block_count": 1,
            "mapping_count": 1,
            "mapped_bytes": 64,
        }


class _FakeModule:
    def __init__(self) -> None:
        self.last_region: _FakeRegion | None = None

    def PagedRegion(self, device: int, virtual_bytes: int) -> _FakeRegion:
        self.last_region = _FakeRegion(device, virtual_bytes)
        return self.last_region


@pytest.fixture
def fake_module(monkeypatch: pytest.MonkeyPatch) -> _FakeModule:
    module = _FakeModule()
    monkeypatch.setattr(paged_tensor, "_native_module", lambda: module)
    return module


def test_wrapper_validates_page_alignment(fake_module: _FakeModule) -> None:
    region = PagedTensorRegion(device=0, virtual_bytes=256)

    with pytest.raises(ValueError, match="aligned"):
        region.create_block(63)
    with pytest.raises(ValueError, match="aligned"):
        region.map(1, 7, 64)
    with pytest.raises(ValueError, match="exceeds"):
        region.map(256, 7, 64)

    native = fake_module.last_region
    assert native is not None
    assert native.calls == []


def test_wrapper_forwards_typed_operations(fake_module: _FakeModule) -> None:
    region = PagedTensorRegion(device=0, virtual_bytes=256)
    block_id = region.create_block(64)
    region.map(0, block_id, 64)
    tensor = region.tensor(0, (4, 8), torch.bfloat16)
    region.unmap(0, 64)

    assert tensor.shape == (4, 8)
    native = fake_module.last_region
    assert native is not None
    assert native.calls == [
        ("create_block", 64),
        ("map", 0, 7, 64),
        ("tensor", 0, (4, 8), 0),
        ("unmap", 0, 64),
    ]


def test_wrapper_returns_typed_snapshot(fake_module: _FakeModule) -> None:
    region = PagedTensorRegion(device=0, virtual_bytes=256)

    snapshot = region.snapshot()

    assert snapshot.base_address == 4096
    assert snapshot.mapped_bytes == 64
    assert snapshot.mapping_count == 1


def test_wrapper_rejects_unsupported_dtype(fake_module: _FakeModule) -> None:
    region = PagedTensorRegion(device=0, virtual_bytes=256)

    with pytest.raises(TypeError, match="unsupported dtype"):
        region.tensor(0, (1,), torch.bool)
