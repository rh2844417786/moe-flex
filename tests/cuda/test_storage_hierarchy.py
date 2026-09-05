from __future__ import annotations

import numpy as np
import pytest
import torch

from flexmoe.codec.packed import pack_layer_descriptor
from flexmoe.codec.reference import encode_bf16_bits
from flexmoe.paged_tensor import PagedTensorRegion
from flexmoe.storage.gpu_compressed import (
    BatchedGpuCompressedStore,
    GpuCompressedStore,
)
from flexmoe.storage.hierarchy import StorageHierarchy
from flexmoe.storage.host_pinned import PinnedHostStore

pytestmark = pytest.mark.cuda


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def test_host_store_owns_pinned_source_and_copies_non_blocking() -> None:
    _require_cuda()
    source = torch.arange(1024, dtype=torch.float32).to(torch.bfloat16)
    store = PinnedHostStore({"layer.0.w2": source})
    destination = torch.empty_like(source, device="cuda:0")
    stream = torch.cuda.Stream(device=0)

    receipt = store.materialize("layer.0.w2", destination, stream.cuda_stream)
    stream.synchronize()

    assert store.source("layer.0.w2").is_pinned()
    assert receipt.backend == "host_pinned"
    assert receipt.nbytes == source.numel() * source.element_size()
    assert torch.equal(destination.cpu(), source)


def test_host_store_copies_selected_experts_into_paged_layer() -> None:
    _require_cuda()
    experts = 16
    expert_shape = (128, 256)
    layer_shape = (experts, *expert_shape)
    raw_bytes = torch.empty(layer_shape, dtype=torch.bfloat16).numel() * 2
    probe = PagedTensorRegion(device=0, virtual_bytes=1)
    aligned_bytes = ((raw_bytes + probe.granularity - 1) // probe.granularity) * probe.granularity
    region = PagedTensorRegion(device=0, virtual_bytes=5 * aligned_bytes)
    block = region.create_block(aligned_bytes)
    stream = torch.cuda.Stream(device=0)

    for layer in (0, 4):
        offset = layer * aligned_bytes
        region.map(offset, block, aligned_bytes)
        destination = region.tensor(offset, layer_shape, torch.bfloat16)
        selected = (5, 13)
        sources = {
            f"layer.{layer}.expert.{expert}.w13": torch.full(
                expert_shape, layer * 20 + expert + 0.25, dtype=torch.bfloat16
            )
            for expert in selected
        }
        store = PinnedHostStore(sources)
        hierarchy = StorageHierarchy(
            stores={tensor_id: store for tensor_id in sources},
            tensor_layers={tensor_id: layer for tensor_id in sources},
        )
        hierarchy.materialize_tensors(
            layer,
            {
                f"layer.{layer}.expert.{expert}.w13": destination[expert]
                for expert in selected
            },
            stream=stream.cuda_stream,
        )
        stream.synchronize()

        for expert in selected:
            assert torch.equal(
                destination[expert].cpu(),
                sources[f"layer.{layer}.expert.{expert}.w13"],
            )
        del destination
        region.unmap(offset, aligned_bytes)


def test_hierarchy_materializes_both_backends_on_one_load_stream() -> None:
    _require_cuda()
    words = (
        np.random.default_rng(13)
        .integers(0, 65_536, 4096, dtype=np.uint16)
        .astype("<u2", copy=False)
    )
    encoded = encode_bf16_bits(words.tobytes(), (4096,))
    host_source = torch.arange(2048, dtype=torch.float32).to(torch.bfloat16)
    gpu_store = GpuCompressedStore({"layer.0.w13": encoded}, device=0)
    host_store = PinnedHostStore({"layer.0.w2": host_source})
    hierarchy = StorageHierarchy(
        stores={"layer.0.w13": gpu_store, "layer.0.w2": host_store},
        tensor_layers={"layer.0.w13": 0, "layer.0.w2": 0},
    )
    destinations = {
        "layer.0.w13": torch.empty(4096, dtype=torch.bfloat16, device="cuda:0"),
        "layer.0.w2": torch.empty(2048, dtype=torch.bfloat16, device="cuda:0"),
    }
    stream = torch.cuda.Stream(device=0)

    receipts = hierarchy.materialize_layer(0, destinations, stream=stream.cuda_stream)
    stream.synchronize()
    gpu_store.raise_for_decode_errors("layer.0.w13")

    assert tuple(receipt.backend for receipt in receipts) == (
        "gpu_compressed",
        "host_pinned",
    )
    assert (
        destinations["layer.0.w13"].view(torch.int16).cpu().numpy().tobytes()
        == words.tobytes()
    )
    assert torch.equal(destinations["layer.0.w2"].cpu(), host_source)


def test_batched_gpu_store_uses_one_layer_kind_launch() -> None:
    _require_cuda()
    raw_experts = [
        np.random.default_rng(seed)
        .integers(0, 65_536, 5000, dtype=np.uint16)
        .astype("<u2", copy=False)
        .tobytes()
        for seed in (41, 42)
    ]
    packed = pack_layer_descriptor(
        {
            expert: encode_bf16_bits(raw, (5000,))
            for expert, raw in enumerate(raw_experts)
        },
        destination_shape=(2, 5000),
    )
    store = BatchedGpuCompressedStore({"layer.0.w13": packed}, device=0)
    destination = torch.empty((2, 5000), dtype=torch.bfloat16, device="cuda:0")
    stream = torch.cuda.Stream(device=0)

    receipt = store.materialize_batched("layer.0.w13", destination, stream.cuda_stream)
    stream.synchronize()
    store.raise_for_decode_errors("layer.0.w13")

    assert receipt.nbytes == 20_000
    assert store.source_bytes == 20_000
    assert store.storage_bytes >= store.input_bytes("layer.0.w13")
    assert destination.view(torch.int16).cpu().numpy().tobytes() == b"".join(
        raw_experts
    )
