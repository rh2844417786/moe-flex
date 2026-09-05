"""Real pinned-vLLM CUDA parity; CPU execution is explicitly not evidence."""

from __future__ import annotations

import pytest
import torch

from flexmoe.runtime.expert_cache_policy import ExpertCachePolicy
from flexmoe.runtime.expert_pool import CudaPoolBackend, ExpertPool
from flexmoe.vllm.expert_cache import logical_fused_experts
from flexmoe.vllm.loader import ExpertLoadAccumulator

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires NVIDIA CUDA and pinned vLLM"
    ),
]


@pytest.mark.parametrize("tp_rank", range(4))
def test_mapped_bf16_native_topk_parity_churn_full_demand_and_stream_reuse(tp_rank):
    vllm = pytest.importorskip("vllm")
    assert vllm.__version__ == "0.10.2"
    from vllm.model_executor.layers.fused_moe import fused_experts

    generator = torch.Generator(device="cpu").manual_seed(91)
    sources = []
    for layer in range(2):
        accum = ExpertLoadAccumulator(f"model.layers.{layer}.mlp.experts", tp_rank, 4)
        experts = []
        for expert in range(8):
            for shard in ("w1", "w3", "w2"):
                shape = (128, 64) if shard != "w2" else (64, 128)
                value = (torch.randn(shape, generator=generator) * 0.03).bfloat16()
                accum.ingest(shard, expert, value)
            experts.append(accum.finalize_expert(expert))
        sources.append(
            (
                torch.stack([e.w13 for e in experts]),
                torch.stack([e.w2 for e in experts]),
            )
        )
    pool = ExpertPool(
        ExpertCachePolicy(2, 8, 0.25, 2, policy="lru"),
        sources,
        backend=CudaPoolBackend(0),
    )
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    demands = [[2, 3], [2, 3], [4, 5], list(range(8)), [2, 3]]
    for step, demand in enumerate(demands):
        for layer in range(2):
            with torch.cuda.stream(streams[(step + layer) % 2]):
                ids = torch.tensor(
                    [demand[i : i + 2] for i in range(0, len(demand), 2)],
                    dtype=torch.int32,
                    device="cuda",
                )
                hidden = torch.randn(
                    (ids.shape[0], 64), device="cuda", dtype=torch.bfloat16
                )
                weights = torch.full(ids.shape, 0.5, device="cuda", dtype=torch.float32)
                native13, native2 = (t.cuda() for t in sources[layer])
                expected = fused_experts(
                    hidden,
                    native13,
                    native2,
                    weights,
                    ids,
                    inplace=False,
                    activation="silu",
                )

                def compute(
                    w13,
                    w2,
                    mapping,
                    demand=demand,
                    layer=layer,
                    hidden=hidden,
                    weights=weights,
                    ids=ids,
                ):
                    for expert in demand:
                        slot = int(mapping[expert].item())
                        assert slot >= 0
                        for gpu, source in (
                            (w13, sources[layer][0]),
                            (w2, sources[layer][1]),
                        ):
                            assert torch.equal(
                                gpu[slot].view(torch.int16).cpu(),
                                source[expert].view(torch.int16),
                            )
                    return logical_fused_experts(
                        hidden,
                        w13,
                        w2,
                        weights,
                        ids,
                        mapping,
                        num_experts=8,
                        activation="silu",
                        apply_router_weight_on_input=False,
                    )

                actual = pool.execute(layer, ids, compute)
                torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-2)
                if len(demand) == 2:
                    copied = pool.stats()["h2d_bytes"]
                    repeated = pool.execute(layer, ids, compute)
                    torch.testing.assert_close(repeated, expected, atol=1e-3, rtol=1e-2)
                    assert pool.stats()["h2d_bytes"] == copied
    stats = pool.stats()
    assert stats["max_unique_per_forward"] == 8
    assert stats["policy"]["evictions"] > 0
    assert stats["policy"]["cache_bypasses"] > 0
    assert stats["timing"]["cuda_sample_count"] == 1
