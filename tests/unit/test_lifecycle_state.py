from __future__ import annotations

import pytest

from flexmoe.errors import IntegrityError
from flexmoe.runtime.lifecycle import (
    LayerState,
    LayerStateMachine,
    should_recycle,
    target_recycle_layer,
)


def test_two_layer_recycle_wraps_across_iterations() -> None:
    assert target_recycle_layer(layer_idx=2, total_layers=48) == 0
    assert target_recycle_layer(layer_idx=1, total_layers=48) == 47
    assert target_recycle_layer(layer_idx=0, total_layers=48) == 46


def test_first_two_initial_layers_do_not_recycle() -> None:
    assert not should_recycle(layer_idx=0, iteration=0, total_layers=48)
    assert not should_recycle(layer_idx=1, iteration=0, total_layers=48)
    assert should_recycle(layer_idx=2, iteration=0, total_layers=48)
    assert should_recycle(layer_idx=0, iteration=1, total_layers=48)


def test_state_enum_is_explicit() -> None:
    assert {state.value for state in LayerState} == {
        "unmapped",
        "loading",
        "resident",
        "evicting",
    }


def test_state_machine_accepts_only_the_lifecycle_cycle() -> None:
    machine = LayerStateMachine(total_layers=2)

    machine.transition(0, LayerState.LOADING)
    machine.transition(0, LayerState.RESIDENT)
    machine.transition(0, LayerState.EVICTING)
    machine.transition(0, LayerState.UNMAPPED)
    assert machine.snapshot() == (LayerState.UNMAPPED, LayerState.UNMAPPED)


def test_state_machine_rejects_illegal_transition() -> None:
    machine = LayerStateMachine(total_layers=2)

    with pytest.raises(IntegrityError, match="unmapped -> resident"):
        machine.transition(0, LayerState.RESIDENT)
    with pytest.raises(IndexError, match="layer_idx"):
        machine.transition(2, LayerState.LOADING)
