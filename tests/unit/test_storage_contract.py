from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch

from flexmoe.errors import IntegrityError
from flexmoe.storage.base import MaterializationReceipt
from flexmoe.storage.hierarchy import StorageHierarchy


@dataclass
class _FakeStore:
    backend: str
    calls: list[tuple[str, int]] = field(default_factory=list)

    def materialize(
        self, tensor_id: str, destination: torch.Tensor, stream: int
    ) -> MaterializationReceipt:
        self.calls.append((tensor_id, stream))
        return MaterializationReceipt(
            tensor_id=tensor_id,
            backend=self.backend,
            nbytes=destination.numel() * destination.element_size(),
            elapsed_s=0.001,
        )


def test_hierarchy_returns_receipts_sorted_by_tensor_id() -> None:
    gpu_store = _FakeStore("gpu_compressed")
    host_store = _FakeStore("host_pinned")
    hierarchy = StorageHierarchy(
        stores={"layer.0.w13": gpu_store, "layer.0.w2": host_store},
        tensor_layers={"layer.0.w13": 0, "layer.0.w2": 0},
    )
    destinations = {
        "layer.0.w2": torch.empty(4, dtype=torch.bfloat16),
        "layer.0.w13": torch.empty(8, dtype=torch.bfloat16),
    }

    receipts = hierarchy.materialize_layer(0, destinations, stream=17)

    assert tuple(receipt.tensor_id for receipt in receipts) == (
        "layer.0.w13",
        "layer.0.w2",
    )
    assert gpu_store.calls == [("layer.0.w13", 17)]
    assert host_store.calls == [("layer.0.w2", 17)]


def test_hierarchy_rejects_missing_or_cross_layer_destination() -> None:
    store = _FakeStore("host_pinned")
    hierarchy = StorageHierarchy(
        stores={"layer.0.w2": store, "layer.1.w2": store},
        tensor_layers={"layer.0.w2": 0, "layer.1.w2": 1},
    )

    with pytest.raises(IntegrityError, match="destination tensor IDs"):
        hierarchy.materialize_layer(
            0,
            {"layer.1.w2": torch.empty(1, dtype=torch.bfloat16)},
            stream=17,
        )


def test_receipt_rejects_zero_bytes_or_invalid_elapsed_time() -> None:
    with pytest.raises(ValueError, match="nbytes"):
        MaterializationReceipt(
            tensor_id="layer.0.w2",
            backend="host_pinned",
            nbytes=0,
            elapsed_s=0.1,
        )
    with pytest.raises(ValueError, match="elapsed_s"):
        MaterializationReceipt(
            tensor_id="layer.0.w2",
            backend="host_pinned",
            nbytes=2,
            elapsed_s=-1.0,
        )
