from __future__ import annotations

import math

import pytest

from flexmoe.runtime.expert_cache_policy import ExpertCachePolicy


def test_hand_derived_resident_and_admission_example() -> None:
    policy = ExpertCachePolicy(
        2,
        4,
        0.25,
        1,
        initial_counts=[[9, 3, 2, 1], [8, 2, 1, 0]],
    )

    assert policy.resident_ids == ((0,), (0,))
    assert policy.resident_slot(1, 0) == 1
    assert policy.admit(0, 1, set()) == (0, None)
    assert policy.admit(1, 3, set()) is None
    assert policy.lookup(0, 1) == 0


def test_observe_counts_repeated_ids_once_and_lookup_counts_misses() -> None:
    policy = ExpertCachePolicy(1, 4, 0.0, 1)

    policy.observe(0, [2, 2, 2])
    assert policy.lookup(0, 2) is None

    assert policy.stats()["observations"] == 1
    assert policy.stats()["unique_demands"] == 1
    assert policy.stats()["cache_misses"] == 1


def test_all_protected_cache_entries_bypass_without_mutation() -> None:
    policy = ExpertCachePolicy(
        1,
        3,
        0.0,
        2,
        initial_counts=[[1, 1, 4]],
    )
    assert policy.admit(0, 0, set()) == (0, None)
    assert policy.admit(0, 1, set()) == (1, None)

    assert policy.admit(0, 2, {(0, 0), (0, 1)}) is None
    assert policy.lookup(0, 0) == 0
    assert policy.lookup(0, 1) == 1
    assert policy.stats()["cache_bypasses"] == 1


def test_equal_expert_numbers_in_different_layers_have_distinct_keys() -> None:
    policy = ExpertCachePolicy(2, 2, 0.0, 2, policy="lru")

    assert policy.admit(0, 1, set()) == (0, None)
    assert policy.admit(1, 1, set()) == (1, None)

    assert policy.lookup(0, 1) == 0
    assert policy.lookup(1, 1) == 1


def test_decayed_lfu_replaces_stale_hot_entry_after_workload_changes() -> None:
    policy = ExpertCachePolicy(
        1,
        3,
        0.0,
        1,
        decay_interval=1,
        initial_counts=[[4, 0, 0]],
    )
    assert policy.admit(0, 0, set()) == (0, None)

    for _ in range(3):
        policy.observe(0, [1])

    assert policy.admit(0, 1, set()) == (0, (0, 0))
    assert policy.lookup(0, 0) is None
    assert policy.lookup(0, 1) == 0


def test_decay_epoch_is_shared_by_all_layers_in_a_forward() -> None:
    policy = ExpertCachePolicy(
        2,
        2,
        0.0,
        0,
        decay_interval=1,
        initial_counts=[[0.75, 0], [0.75, 0]],
    )

    policy.observe(0, [1])
    policy.observe(1, [1])
    policy.reconfigure(0.5, 0)

    assert policy.resident_ids == ((1,), (1,))


def test_same_score_victim_uses_next_layer_distance_deterministically() -> None:
    policy = ExpertCachePolicy(
        3,
        2,
        0.0,
        2,
        initial_counts=[[0, 3], [2, 0], [2, 0]],
    )
    assert policy.admit(1, 0, set()) == (0, None)
    assert policy.admit(2, 0, set()) == (1, None)

    assert policy.admit(0, 1, set()) == (1, (2, 0))
    assert policy.lookup(1, 0) == 0


def test_resident_entries_never_enter_persistent_eviction_cache() -> None:
    policy = ExpertCachePolicy(
        1,
        4,
        0.5,
        1,
        initial_counts=[[9, 8, 2, 1]],
    )

    assert policy.resident_ids == ((0, 1),)
    assert policy.resident_slot(0, 0) == 0
    assert policy.resident_slot(0, 1) == 1
    assert policy.admit(0, 0, set()) is None
    assert policy.lookup(0, 0) is None


def test_reconfigure_preserves_heat_and_clears_persistent_assignments() -> None:
    policy = ExpertCachePolicy(1, 4, 0.0, 1, initial_counts=[[3, 2, 1, 0]])
    assert policy.admit(0, 1, set()) == (0, None)
    policy.observe(0, [3])
    policy.observe(0, [3])
    policy.observe(0, [3])
    policy.observe(0, [3])

    policy.reconfigure(0.25, 2)

    assert policy.resident_ids == ((3,),)
    assert policy.lookup(0, 1) is None
    assert policy.stats()["reconfigurations"] == 1


def test_lru_always_admits_and_tracks_monotonic_access_order() -> None:
    policy = ExpertCachePolicy(
        1,
        3,
        0.0,
        2,
        policy="lru",
        initial_counts=[[100, 1, 0]],
    )
    assert policy.admit(0, 0, set()) == (0, None)
    assert policy.admit(0, 1, set()) == (1, None)
    assert policy.lookup(0, 0) == 0

    assert policy.admit(0, 2, set()) == (1, (0, 1))


@pytest.mark.parametrize("ratio", [0.0, 1.0])
def test_resident_ratio_boundaries_are_valid(ratio: float) -> None:
    policy = ExpertCachePolicy(2, 3, ratio, 0)
    expected = 0 if ratio == 0.0 else 3
    assert all(len(layer) == expected for layer in policy.resident_ids)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"total_layers": 0}, "total_layers"),
        ({"num_experts": 0}, "num_experts"),
        ({"resident_ratio": -0.1}, "resident_ratio"),
        ({"resident_ratio": 1.1}, "resident_ratio"),
        ({"resident_ratio": math.nan}, "resident_ratio"),
        ({"cache_slots": -1}, "cache_slots"),
        ({"cache_slots": 3}, "cache_slots"),
        ({"decay_interval": 0}, "decay_interval"),
        ({"policy": "fifo"}, "policy"),
    ],
)
def test_invalid_policy_configuration_fails_closed(
    kwargs: dict[str, object], message: str
) -> None:
    valid: dict[str, object] = {
        "total_layers": 1,
        "num_experts": 2,
        "resident_ratio": 0.5,
        "cache_slots": 1,
    }
    valid.update(kwargs)

    with pytest.raises((TypeError, ValueError), match=message):
        ExpertCachePolicy(**valid)  # type: ignore[arg-type]


def test_invalid_observation_and_initial_counts_fail_closed() -> None:
    with pytest.raises(ValueError, match="shape"):
        ExpertCachePolicy(2, 2, 0.0, 1, initial_counts=[[1, 2]])
    with pytest.raises(ValueError, match="finite"):
        ExpertCachePolicy(1, 2, 0.0, 1, initial_counts=[[1, math.inf]])

    policy = ExpertCachePolicy(1, 2, 0.0, 1)
    with pytest.raises(IndexError, match="layer"):
        policy.observe(1, [0])
    with pytest.raises(IndexError, match="expert"):
        policy.observe(0, [2])


def test_reconfigure_rejects_cache_larger_than_key_space_atomically() -> None:
    policy = ExpertCachePolicy(1, 2, 0.5, 1, initial_counts=[[2, 1]])

    with pytest.raises(ValueError, match="cache_slots"):
        policy.reconfigure(0.0, 3)

    assert policy.resident_ids == ((0,),)
    assert policy.stats()["cache_slots"] == 1
