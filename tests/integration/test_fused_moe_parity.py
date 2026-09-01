from __future__ import annotations

import pytest
import torch

from flexmoe.paged_tensor import PagedTensorRegion

pytestmark = [pytest.mark.integration, pytest.mark.cuda]


def _paged_copy(source: torch.Tensor) -> tuple[PagedTensorRegion, torch.Tensor]:
    region = PagedTensorRegion(device=0, virtual_bytes=16 << 20)
    nbytes = source.numel() * source.element_size()
    mapped_bytes = ((nbytes + region.granularity - 1) // region.granularity) * (
        region.granularity
    )
    block = region.create_block(mapped_bytes)
    region.map(0, block, mapped_bytes)
    destination = region.tensor(0, tuple(source.shape), source.dtype)
    destination.copy_(source)
    return region, destination


@torch.inference_mode()
def test_vmm_weights_match_resident_fused_moe() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    fused_module = pytest.importorskip(
        "vllm.model_executor.layers.fused_moe.fused_moe"
    )
    torch.manual_seed(19)
    torch.cuda.manual_seed_all(19)
    experts, tokens, hidden, intermediate, top_k = 4, 8, 128, 128, 2
    inputs = torch.randn(
        (tokens, hidden), dtype=torch.bfloat16, device="cuda:0"
    )
    resident_w13 = torch.randn(
        (experts, 2 * intermediate, hidden),
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    resident_w2 = torch.randn(
        (experts, hidden, intermediate),
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    scores = torch.randn(
        (tokens, experts), dtype=torch.bfloat16, device="cuda:0"
    )
    w13_region, paged_w13 = _paged_copy(resident_w13)
    w2_region, paged_w2 = _paged_copy(resident_w2)
    torch.cuda.synchronize()

    expected = fused_module.fused_moe(
        inputs.clone(),
        resident_w13,
        resident_w2,
        scores,
        top_k,
        renormalize=False,
    )
    actual = fused_module.fused_moe(
        inputs.clone(),
        paged_w13,
        paged_w2,
        scores,
        top_k,
        renormalize=False,
    )

    assert torch.equal(actual, expected)
    assert w13_region.snapshot().mapped_bytes > 0
    assert w2_region.snapshot().mapped_bytes > 0
