from __future__ import annotations

import pytest
import torch

from flexmoe.errors import IntegrityError
from flexmoe.vllm.loader import ExpertLoadAccumulator


def test_loader_combines_w1_w3_and_shards_like_vllm() -> None:
    accumulator = ExpertLoadAccumulator(
        layer_name="model.layers.0.mlp.experts", tp_rank=1, tp_size=4
    )
    w1 = torch.arange(32, dtype=torch.float32).to(torch.bfloat16).reshape(8, 4)
    w3 = w1 + 100
    accumulator.ingest("w1", 0, w1)
    accumulator.ingest("w3", 0, w3)

    combined = accumulator.finalize_w13(0)

    assert combined.shape == (4, 4)
    assert torch.equal(combined[:2], w1[2:4])
    assert torch.equal(combined[2:], w3[2:4])


def test_loader_shards_w2_input_dimension_like_vllm() -> None:
    accumulator = ExpertLoadAccumulator(
        layer_name="model.layers.0.mlp.experts", tp_rank=2, tp_size=4
    )
    w2 = torch.arange(64, dtype=torch.float32).to(torch.bfloat16).reshape(8, 8)
    accumulator.ingest("w2", 3, w2)

    sharded = accumulator.finalize_w2(3)

    assert sharded.shape == (8, 2)
    assert torch.equal(sharded, w2[:, 4:6])


def test_loader_rejects_duplicate_or_unsupported_shard() -> None:
    accumulator = ExpertLoadAccumulator(
        layer_name="model.layers.0.mlp.experts", tp_rank=0, tp_size=1
    )
    weight = torch.ones((2, 2), dtype=torch.bfloat16)
    accumulator.ingest("w1", 0, weight)

    with pytest.raises(IntegrityError, match="duplicate"):
        accumulator.ingest("w1", 0, weight)
    with pytest.raises(IntegrityError, match="unexpected shard"):
        accumulator.ingest("bias", 0, weight)


def test_loader_rejects_non_bf16_and_non_matrix_weights() -> None:
    accumulator = ExpertLoadAccumulator(
        layer_name="model.layers.0.mlp.experts", tp_rank=0, tp_size=1
    )

    with pytest.raises(IntegrityError, match="BF16"):
        accumulator.ingest("w1", 0, torch.ones((2, 2), dtype=torch.float32))
    with pytest.raises(IntegrityError, match="two-dimensional"):
        accumulator.ingest("w1", 0, torch.ones(2, dtype=torch.bfloat16))
