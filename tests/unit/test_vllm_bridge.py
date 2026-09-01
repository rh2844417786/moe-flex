from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

import flexmoe.vllm.bridge as bridge_module
from flexmoe.errors import UnsupportedModeError
from flexmoe.storage.base import MaterializationReceipt
from flexmoe.vllm.bridge import (
    FluxMoERegistry,
    RegistryStorageConfig,
    install_registry,
    maybe_create_weights,
    reset_registry,
)
from flexmoe.vllm.loader import store_expert_weight


class _FakeRegion:
    def __init__(self, device: int, virtual_bytes: int) -> None:
        self.device = device
        self.virtual_bytes = virtual_bytes
        self.granularity = 64
        self._next_block = 1

    def create_block(self, nbytes: int) -> int:
        assert nbytes % self.granularity == 0
        block = self._next_block
        self._next_block += 1
        return block

    def map(self, offset: int, block_id: int, nbytes: int) -> None:
        assert offset % self.granularity == 0
        assert block_id > 0
        assert nbytes % self.granularity == 0

    def unmap(self, offset: int, nbytes: int) -> None:
        assert offset % self.granularity == 0
        assert nbytes % self.granularity == 0

    def tensor(
        self, offset: int, shape: Sequence[int], dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.empty(tuple(shape), dtype=dtype)


class _Layer(torch.nn.Module):
    def __init__(self, layer_idx: int = 0) -> None:
        super().__init__()
        self.layer_name = f"model.layers.{layer_idx}.mlp.experts"
        self.tp_rank = 0
        self.tp_size = 1
        self.use_ep = False
        self.enable_eplb = False
        self.quant_config = None


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    reset_registry()
    yield
    reset_registry()


def test_disabled_bridge_leaves_vllm_creation_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FLUXMOE_ENABLE", raising=False)

    assert not maybe_create_weights(
        layer=_Layer(),
        num_experts=2,
        hidden_size=4,
        intermediate_size_per_partition=2,
        params_dtype=torch.bfloat16,
    )


def test_enabled_bridge_registers_paged_parameters_and_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLUXMOE_ENABLE", "1")
    registry = FluxMoERegistry(
        total_layers=2,
        device=0,
        tp_rank=0,
        tp_size=1,
        num_experts=2,
        region_factory=_FakeRegion,
    )
    install_registry(registry)
    layer = _Layer()

    created = maybe_create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=4,
        intermediate_size_per_partition=2,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *_: None,
    )

    assert created
    assert tuple(layer.w13_weight.shape) == (2, 4, 4)
    assert tuple(layer.w2_weight.shape) == (2, 4, 2)
    assert layer.w13_weight.weight_loader is store_expert_weight
    assert layer.w2_weight.weight_loader is store_expert_weight
    assert layer.w13_weight._fluxmoe_kind == "w13"
    assert layer.w2_weight._fluxmoe_kind == "w2"


def test_enabled_bridge_rejects_non_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLUXMOE_ENABLE", "1")

    with pytest.raises(UnsupportedModeError, match="BF16"):
        maybe_create_weights(
            layer=_Layer(),
            num_experts=2,
            hidden_size=4,
            intermediate_size_per_partition=2,
            params_dtype=torch.float16,
        )


class _FakePinnedStore:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def materialize(
        self, tensor_id: str, destination: torch.Tensor, stream: int
    ) -> MaterializationReceipt:
        destination.copy_(self.tensors[tensor_id])
        return MaterializationReceipt(
            tensor_id=tensor_id,
            backend="host_pinned",
            nbytes=destination.numel() * destination.element_size(),
            elapsed_s=0.001,
        )


class _FakeStream:
    cuda_stream = 77


class _FakeLifecycle:
    def __init__(
        self,
        *,
        device: int,
        total_layers: int,
        materializers: dict[str, object],
    ) -> None:
        self.total_layers = total_layers
        self.materializers = materializers
        self.scheduled: list[int] = []

    def schedule_next(self, layer_idx: int) -> None:
        if len(self.scheduled) >= 2:
            recycle = (layer_idx - 2) % self.total_layers
            for materializer in self.materializers.values():
                materializer.evict(recycle)
        for materializer in self.materializers.values():
            materializer.materialize(layer_idx, 77)
        self.scheduled.append(layer_idx)

    def ensure_ready(self, layer_idx: int, stream: _FakeStream) -> None:
        return None

    def mark_consumed(self, layer_idx: int, stream: _FakeStream) -> None:
        return None

    def close(self) -> None:
        return None


def test_complete_loader_builds_host_hierarchy_and_forward_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLUXMOE_ENABLE", "1")
    monkeypatch.setattr(bridge_module, "PinnedHostStore", _FakePinnedStore)
    monkeypatch.setattr(bridge_module, "LayerLifecycle", _FakeLifecycle)
    monkeypatch.setattr(
        bridge_module.torch.cuda, "current_stream", lambda _: _FakeStream()
    )
    registry = FluxMoERegistry(
        total_layers=2,
        device=0,
        tp_rank=0,
        tp_size=1,
        num_experts=1,
        region_factory=_FakeRegion,
        storage_config=RegistryStorageConfig(
            gpu_compressed_budget_bytes=0,
            host_capacity_bytes=1 << 20,
            gpu_decode_bytes_per_second=10.0,
            host_h2d_bytes_per_second=1.0,
        ),
    )
    install_registry(registry)
    layers = [_Layer(0), _Layer(1)]
    for layer in layers:
        assert maybe_create_weights(
            layer=layer,
            num_experts=1,
            hidden_size=4,
            intermediate_size_per_partition=2,
            params_dtype=torch.bfloat16,
        )

    for layer_idx, layer in enumerate(layers):
        base = float(layer_idx * 100)
        w1 = torch.full((2, 4), base + 1, dtype=torch.bfloat16)
        w3 = torch.full((2, 4), base + 3, dtype=torch.bfloat16)
        w2 = torch.full((4, 2), base + 2, dtype=torch.bfloat16)
        store_expert_weight(
            layer.w13_weight,
            w1,
            layer.layer_name,
            "w1",
            0,
        )
        store_expert_weight(
            layer.w13_weight,
            w3,
            layer.layer_name,
            "w3",
            0,
        )
        store_expert_weight(
            layer.w2_weight,
            w2,
            layer.layer_name,
            "w2",
            0,
        )

    assert registry.completed_experts() == 2
    assert len(registry.receipts()) == 4
    token = registry.before_forward(
        layers[0].layer_name, layers[0].w13_weight, layers[0].w2_weight
    )
    registry.after_forward(token)
    assert len(registry.receipts()) == 6
    assert torch.equal(
        layers[0].w13_weight[0, :2],
        torch.full((2, 4), 1, dtype=torch.bfloat16),
    )
