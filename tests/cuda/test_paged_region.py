from __future__ import annotations

import gc

import pytest
import torch

from flexmoe.paged_tensor import PagedTensorRegion

pytestmark = pytest.mark.cuda


def _region() -> PagedTensorRegion:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return PagedTensorRegion(device=0, virtual_bytes=256 << 20)


def test_pointer_is_stable_across_remap() -> None:
    region = _region()
    granularity = region.granularity
    assert 2 * granularity <= region.virtual_bytes

    block_a = region.create_block(granularity)
    block_b = region.create_block(granularity)
    region.map(0, block_a, granularity)

    view = region.tensor(0, (granularity // 2,), torch.bfloat16)
    pointer = view.data_ptr()
    view.fill_(1)
    torch.cuda.synchronize()

    del view
    region.unmap(0, granularity)
    region.map(0, block_b, granularity)
    assert (
        region.tensor(0, (granularity // 2,), torch.bfloat16).data_ptr()
        == pointer
    )


def test_alignment_overlap_and_unmapped_views_fail_closed() -> None:
    region = _region()
    granularity = region.granularity
    block = region.create_block(granularity)

    with pytest.raises(ValueError, match="aligned"):
        region.map(1, block, granularity)

    region.map(0, block, granularity)
    with pytest.raises(RuntimeError, match="overlap"):
        region.map(0, block, granularity)
    with pytest.raises(RuntimeError, match="unmapped"):
        region.tensor(granularity, (1,), torch.bfloat16)


def test_snapshot_accounts_for_live_mapping() -> None:
    region = _region()
    granularity = region.granularity
    block = region.create_block(granularity)
    region.map(0, block, granularity)

    snapshot = region.snapshot()
    assert snapshot.device == 0
    assert snapshot.block_count == 1
    assert snapshot.mapping_count == 1
    assert snapshot.mapped_bytes == granularity


def test_tensor_view_keeps_region_owner_alive() -> None:
    region = _region()
    granularity = region.granularity
    block = region.create_block(granularity)
    region.map(0, block, granularity)
    view = region.tensor(0, (granularity // 2,), torch.bfloat16)

    del region
    gc.collect()
    view.fill_(2)
    torch.cuda.synchronize()
    assert view[0].item() == 2


def test_repeated_remap_keeps_base_address_stable() -> None:
    region = _region()
    granularity = region.granularity
    block = region.create_block(granularity)
    base_address = region.base_address

    for _ in range(10_000):
        region.map(0, block, granularity)
        assert region.base_address == base_address
        region.unmap(0, granularity)
