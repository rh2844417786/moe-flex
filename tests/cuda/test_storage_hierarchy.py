from __future__ import annotations

import numpy as np
import pytest
import torch

from flexmoe.codec.reference import encode_bf16_bits
from flexmoe.storage.gpu_compressed import GpuCompressedStore
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

    receipt = store.materialize(
        "layer.0.w2", destination, stream.cuda_stream
    )
    stream.synchronize()

    assert store.source("layer.0.w2").is_pinned()
    assert receipt.backend == "host_pinned"
    assert receipt.nbytes == source.numel() * source.element_size()
    assert torch.equal(destination.cpu(), source)


def test_hierarchy_materializes_both_backends_on_one_load_stream() -> None:
    _require_cuda()
    words = np.random.default_rng(13).integers(
        0, 65_536, 4096, dtype=np.uint16
    ).astype("<u2", copy=False)
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

    receipts = hierarchy.materialize_layer(
        0, destinations, stream=stream.cuda_stream
    )
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
