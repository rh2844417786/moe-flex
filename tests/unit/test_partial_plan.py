from __future__ import annotations

import pytest

from flexmoe.runtime.partial_plan import PartialPlan


def test_even_spacing_leaves_compute_between_offloaded_layers() -> None:
    plan = PartialPlan.evenly_spaced(8, 4, 2)
    assert plan.offload_layers == (1, 3, 5, 7)
    assert plan.initial_prefetch == (1, 3)
    assert plan.slot_for(0) is None
    assert [plan.slot_for(i) for i in (1, 3, 5, 7)] == [0, 1, 0, 1]
    assert [plan.prefetch_after(i) for i in (1, 3, 5, 7)] == [5, 7, 1, 3]
    assert plan.net_freed_bytes(128) == 256


def test_single_slot_wraps_without_overwriting_an_unconsumed_layer() -> None:
    plan = PartialPlan.evenly_spaced(8, 3, 1)
    assert plan.offload_layers == (1, 4, 7)
    assert plan.initial_prefetch == (1,)
    assert [plan.prefetch_after(i) for i in (1, 4, 7)] == [4, 7, 1]
    assert plan.net_freed_bytes(128) == 256


def test_zero_offload_needs_no_staging_and_frees_nothing() -> None:
    plan = PartialPlan.evenly_spaced(8, 0, 2)
    assert plan.offload_layers == ()
    assert plan.initial_prefetch == ()
    assert plan.slot_for(7) is None
    assert plan.net_freed_bytes(128) == 0


@pytest.mark.parametrize(
    "layers, slots",
    [((1, 3, 5), 2), ((1,), 1), ((1, 3), 2), ((3, 1), 1), ((1, 1), 1), ((1, 8), 1)],
)
def test_rejects_unsafe_or_non_saving_slot_plans(
    layers: tuple[int, ...], slots: int
) -> None:
    with pytest.raises(ValueError):
        PartialPlan(total_layers=8, offload_layers=layers, staging_slots=slots)


@pytest.mark.parametrize("count", [-1, 9, True])
def test_rejects_invalid_offload_counts(count: int) -> None:
    with pytest.raises(ValueError):
        PartialPlan.evenly_spaced(8, count, 1)


def test_resident_layer_cannot_trigger_slot_reuse() -> None:
    plan = PartialPlan.evenly_spaced(8, 4, 2)
    with pytest.raises(ValueError):
        plan.prefetch_after(0)
