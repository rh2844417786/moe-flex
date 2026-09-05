from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from flexmoe.errors import IntegrityError
from flexmoe.runtime.partial_plan import PartialPlan
from flexmoe.vllm import bridge, partial
from flexmoe.vllm.loader import ExpertWeights
from flexmoe.vllm.partial import (
    HostExpertLayer,
    PartialRegistry,
    validate_execution_config,
)


def test_host_layer_retains_original_bf16_bits_and_rejects_missing_weights() -> None:
    layer = HostExpertLayer((2, 4, 4), (2, 4, 2), pin_memory=False)
    w13 = torch.arange(16, dtype=torch.int16).view(torch.bfloat16).reshape(4, 4)
    w2 = torch.arange(8, dtype=torch.int16).view(torch.bfloat16).reshape(4, 2)
    layer.add(ExpertWeights(1, w13, w2))
    with pytest.raises(IntegrityError, match="incomplete"):
        layer.require_complete()
    layer.add(ExpertWeights(0, w13.flip(0), w2.flip(0)))
    layer.require_complete()
    assert torch.equal(layer.w13[1].view(torch.int16), w13.view(torch.int16))
    assert torch.equal(layer.w2[0].view(torch.int16), w2.flip(0).view(torch.int16))
    with pytest.raises(IntegrityError, match="duplicate"):
        layer.add(ExpertWeights(1, w13, w2))


def test_host_store_ignores_model_construction_default_device() -> None:
    # vLLM constructs models with a CUDA default device. The meta device
    # reproduces this factory-device trap without needing a GPU on the Mac.
    with torch.device("meta"):
        layer = HostExpertLayer((2, 4, 4), (2, 4, 2), pin_memory=False)
    assert layer.w13.device.type == "cpu"
    assert layer.w2.device.type == "cpu"


class CpuStager:
    """Synchronous real tensor copies stand in only for CUDA stream operations."""

    pin_memory = False

    def allocate(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape, dtype=torch.bfloat16)

    def enqueue(
        self,
        slot: int,
        source: HostExpertLayer,
        destination: tuple[torch.Tensor, torch.Tensor],
        *,
        after_compute: bool,
    ) -> None:
        destination[0].copy_(source.w13)
        destination[1].copy_(source.w2)

    def begin(self, slot: int) -> object | None:
        return None

    def end(self, slot: int, ticket: object | None) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def timing(self, *, reset: bool = False) -> dict[str, int | float]:
        return {"sample_count": 0}


def build_registry(
    offload_count: int = 4,
    slots: int = 2,
) -> tuple[PartialRegistry, dict[int, tuple[torch.nn.Parameter, torch.nn.Parameter]]]:
    registry = PartialRegistry(
        plan=PartialPlan.evenly_spaced(8, offload_count, slots),
        device=0,
        tp_rank=0,
        tp_size=1,
        num_experts=2,
        stager=CpuStager(),
    )
    parameters = {}
    for layer in range(8):
        pair = registry.register_layer(
            f"model.layers.{layer}.mlp.experts",
            (2, 4, 4),
            (2, 4, 2),
            torch.bfloat16,
        )
        if pair is None:
            continue
        parameters[layer] = pair
        for expert in range(2):
            for part, shape in (("w1", (2, 4)), ("w3", (2, 4)), ("w2", (4, 2))):
                param = pair[1] if part == "w2" else pair[0]
                registry.ingest(
                    param=param,
                    loaded_weight=torch.full(
                        shape, layer + expert, dtype=torch.bfloat16
                    ),
                    weight_name=f"layers.{layer}.mlp.experts.{expert}.weight",
                    shard_id=part,
                    expert_id=expert,
                )
    return registry, parameters


@pytest.mark.parametrize("count, slots", [(4, 2), (3, 1), (2, 1)])
def test_real_weights_survive_many_slot_reuses(count: int, slots: int) -> None:
    registry, parameters = build_registry(count, slots)
    assert len(parameters) == count
    pointers = {
        layer: tuple(p.data_ptr() for p in pair) for layer, pair in parameters.items()
    }
    for _ in range(5):
        for layer in range(8):
            pair = parameters.get(layer)
            if pair is None:
                pair = (torch.nn.Parameter(torch.empty(0)),) * 2
            token = registry.before_forward(f"model.layers.{layer}.mlp.experts", *pair)
            if layer in parameters:
                assert tuple(p.data_ptr() for p in pair) == pointers[layer]
                for expert in range(2):
                    assert torch.all(pair[0][expert] == layer + expert)
                    assert torch.all(pair[1][expert] == layer + expert)
            registry.after_forward(token)
    stats = registry.stats()
    assert stats["offload_forwards"] == 5 * count
    assert stats["copy_launches"] == 2 * (slots + 5 * count)
    assert stats["gpu_staging_bytes"] == 96 * slots
    assert stats["host_source_bytes"] == 96 * count
    assert stats["net_freed_bytes"] == 96 * (count - slots)
    assert stats["h2d_bytes"] == 96 * (slots + 5 * count)
    assert stats["weights_verified"] == 2 * count


def test_changed_parameter_storage_is_detected_even_when_parameter_object_is_same() -> (
    None
):
    registry, parameters = build_registry()
    parameters[1][0].data = parameters[1][0].data.clone()
    with pytest.raises(IntegrityError, match="staging"):
        registry.before_forward("model.layers.1.mlp.experts", *parameters[1])


def test_prefetch_cannot_advance_out_of_model_order() -> None:
    registry, parameters = build_registry()
    with pytest.raises(IntegrityError, match="expected"):
        registry.before_forward("model.layers.3.mlp.experts", *parameters[3])


def test_zero_offload_preserves_normal_creation_without_any_buffers() -> None:
    registry, parameters = build_registry(0, 1)
    assert parameters == {}
    stats = registry.stats()
    for field in (
        "host_source_bytes",
        "gpu_staging_bytes",
        "h2d_bytes",
        "net_freed_bytes",
    ):
        assert stats[field] == 0
    assert stats["timing"]["sample_count"] == 0


def test_vllm_hook_preserves_resident_parameters_and_routes_partial_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = PartialRegistry(
        plan=PartialPlan.evenly_spaced(8, 2, 1),
        device=0,
        tp_rank=0,
        tp_size=1,
        num_experts=2,
        stager=CpuStager(),
    )
    monkeypatch.setenv("FLUXMOE_ENABLE", "1")
    monkeypatch.setenv("FLUXMOE_STORAGE_MODE", "partial-host")
    monkeypatch.setattr(partial, "registry_for_layer", lambda *_: registry)
    for idx in range(8):
        layer = torch.nn.Module()
        layer.layer_name = f"model.layers.{idx}.mlp.experts"
        layer.tp_rank = 0
        layer.tp_size = 1
        result = bridge.maybe_create_weights(
            layer,
            2,
            4,
            2,
            torch.bfloat16,
            custom_attr="kept",
        )
        assert result is (idx in (3, 7))
        if result:
            assert layer.w13_weight.weight_loader is partial.store_partial_weight
            assert layer.w13_weight.custom_attr == "kept"
        else:
            assert list(layer.parameters()) == []


def test_disabled_fused_hook_or_graphs_fail_before_allocating_partial_weights() -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True),
        compilation_config=SimpleNamespace(level=0, custom_ops=["all"]),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
    )
    validate_execution_config(config)
    config.compilation_config.custom_ops = ["none"]
    with pytest.raises(ValueError, match="fused"):
        validate_execution_config(config)
    config.compilation_config.custom_ops = ["all"]
    config.compilation_config.level = 3
    with pytest.raises(ValueError, match="compilation"):
        validate_execution_config(config)
