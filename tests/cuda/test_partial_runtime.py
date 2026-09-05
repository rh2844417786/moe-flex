from __future__ import annotations

import pytest
import torch

from flexmoe.runtime.partial_plan import PartialPlan
from flexmoe.vllm.partial import PartialRegistry

pytestmark = pytest.mark.cuda


def _registry(slots: int, tp_rank: int = 0, tp_size: int = 1):
    registry = PartialRegistry(
        plan=PartialPlan.evenly_spaced(8, 4, slots),
        device=0,
        tp_rank=tp_rank,
        tp_size=tp_size,
        num_experts=2,
    )
    parameters = {}
    reference = {}
    generator = torch.Generator().manual_seed(42)
    for idx in range(8):
        with torch.device("cuda"):
            pair = registry.register_layer(
                f"model.layers.{idx}.mlp.experts",
                (2, 256 // tp_size, 128),
                (2, 128, 128 // tp_size),
                torch.bfloat16,
            )
        if pair is None:
            continue
        parameters[idx] = pair
        reference[idx] = [[], []]
        for expert in range(2):
            weights = {
                part: torch.randn(shape, generator=generator).to(torch.bfloat16)
                for part, shape in (
                    ("w1", (128, 128)),
                    ("w3", (128, 128)),
                    ("w2", (128, 128)),
                )
            }
            start = tp_rank * (128 // tp_size)
            stop = start + 128 // tp_size
            reference[idx][0].append(
                torch.cat((weights["w1"][start:stop], weights["w3"][start:stop]), dim=0)
            )
            reference[idx][1].append(weights["w2"][:, start:stop].contiguous())
            for part in ("w1", "w3", "w2"):
                registry.ingest(
                    param=pair[1] if part == "w2" else pair[0],
                    loaded_weight=weights[part],
                    weight_name=f"layers.{idx}.mlp.experts.{expert}.weight",
                    shard_id=part,
                    expert_id=expert,
                )
        reference[idx] = tuple(torch.stack(value) for value in reference[idx])
    return registry, parameters, reference


@pytest.mark.parametrize("slots", [1, 2])
@pytest.mark.parametrize("tp_rank", [0, 1, 2, 3])
def test_side_stream_reuse_preserves_bf16_bits_across_tp_shards(
    slots: int,
    tp_rank: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA H100 required")
    registry, parameters, reference = _registry(slots, tp_rank, 4)
    compute = torch.cuda.Stream()
    saved = []
    try:
        with torch.cuda.stream(compute):
            for _ in range(10):
                for idx, pair in parameters.items():
                    token = registry.before_forward(
                        f"model.layers.{idx}.mlp.experts", *pair
                    )
                    # Deliberately keep the consumer live while the producer
                    # schedules reuse, including at the cycle boundary.
                    torch.cuda._sleep(100_000)
                    saved.append((idx, pair[0].clone(), pair[1].clone()))
                    registry.after_forward(token)
        stats = registry.stats()
        assert stats["weights_verified"] == 8
        assert stats["copy_launches"] == 2 * (slots + 40)
        for idx, w13, w2 in saved:
            assert torch.equal(
                w13.view(torch.int16).cpu(), reference[idx][0].view(torch.int16)
            )
            assert torch.equal(
                w2.view(torch.int16).cpu(), reference[idx][1].view(torch.int16)
            )
    finally:
        registry.close()


def test_staged_weights_match_native_fused_moe_after_repeated_reuse() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA H100 required")
    fused = pytest.importorskip("vllm.model_executor.layers.fused_moe.fused_moe")
    registry, parameters, reference = _registry(1)
    torch.manual_seed(27)
    inputs = torch.randn(16, 128, dtype=torch.bfloat16, device="cuda")
    scores = torch.randn(16, 2, dtype=torch.bfloat16, device="cuda")
    try:
        for _ in range(3):
            for idx, pair in parameters.items():
                dense = tuple(t.to("cuda") for t in reference[idx])
                expected = fused.fused_moe(
                    inputs.clone(), *dense, scores, 2, renormalize=False
                )
                token = registry.before_forward(
                    f"model.layers.{idx}.mlp.experts", *pair
                )
                actual = fused.fused_moe(
                    inputs.clone(), *pair, scores, 2, renormalize=False
                )
                registry.after_forward(token)
                assert torch.equal(actual, expected)
    finally:
        registry.close()
