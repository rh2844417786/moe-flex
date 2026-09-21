import copy

import pytest

from flexmoe.runtime.expert_cache_policy import ExpertCachePolicy


def _state(scores, forwards, slots, access, serial):
    return {
        "total_layers": 1,
        "num_experts": 4,
        "resident_ratio": 0.0,
        "cache_slots": 2,
        "decay_interval": 64,
        "policy": "lru",
        "scores": [scores],
        "scale": 1.0,
        "forward_counts": [forwards],
        "decay_events": 0,
        "resident_ids": [[]],
        "slots": slots,
        "access_by_slot": access,
        "access_serial": serial,
    }


def initial_state():
    return _state(
        [100.0, 90.0, 0.0, 0.0],
        0,
        [[0, 0], [0, 1]],
        [1, 2],
        2,
    )


def complete_rows():
    states = [
        initial_state(),
        _state([100.0, 90.0, 1.0, 0.0], 1, [[0, 2], [0, 1]], [3, 2], 3),
        _state([101.0, 90.0, 1.0, 0.0], 2, [[0, 2], [0, 0]], [3, 4], 4),
        _state([101.0, 90.0, 2.0, 0.0], 3, [[0, 2], [0, 0]], [5, 4], 5),
        _state([102.0, 90.0, 2.0, 0.0], 4, [[0, 2], [0, 0]], [5, 6], 6),
        _state([102.0, 90.0, 3.0, 0.0], 5, [[0, 2], [0, 0]], [7, 6], 7),
    ]
    experts = (2, 0, 2, 0, 2)
    misses = ((2,), (0,), (), (), ())
    cache_hits = ((), (), (2,), (0,), (2,))
    evicted = ((0,), (1,), (), (), ())
    return [
        {
            "step_id": step,
            "layer_id": 0,
            "actual_batch": 1,
            "actual_expert_ids": [expert],
            "resident_hit_ids": [],
            "cache_hit_ids": list(cache_hits[step]),
            "miss_ids": list(misses[step]),
            "bypass_ids": [],
            "admitted_ids": list(misses[step]),
            "evicted_ids": list(evicted[step]),
            "cache_state_before": states[step],
            "cache_state_after": states[step + 1],
            "loaded_bytes": 10 if misses[step] else 0,
        }
        for step, expert in enumerate(experts)
    ]


def profiled_rows(steps=64):
    from test_expert_cache_runtime import TensorBackend, invoke, make_pool

    from flexmoe.runtime.expert_pool import ExpertPool, PoolProfileObserver

    original, sources = make_pool(slots=4)
    pool = ExpertPool(
        original.policy, sources, backend=TensorBackend(), track_load_history=True
    )
    context = {"step": 0, "actual_batch": 16, "phase": "decode"}
    observer = PoolProfileObserver(
        lambda layer: context, capacity=steps * 2, device="cpu"
    )
    pool.profile_observer = observer
    for step in range(steps):
        context["step"] = step
        for layer in range(2):
            invoke(pool, sources, layer, [step % 4])
    captured = observer.snapshot()
    return captured["initial_policy_state"], captured["rows"], pool.expert_bytes


def test_snapshot_keeps_fixed_residency_scores_clock_and_cache_occupancy():
    policy = ExpertCachePolicy(
        2, 4, 0.25, 1, decay_interval=2, initial_counts=[[10, 3, 2, 0], [8, 5, 1, 0]]
    )
    assert policy.admit(0, 1, set()) == (0, None)
    policy.observe(0, [1, 2])
    policy.observe(1, [2])

    clone = ExpertCachePolicy.from_replay_snapshot(policy.replay_snapshot())
    assert clone.resident_ids == ((0,), (0,))
    for source in (policy, clone):
        source.observe(0, [2])
        assert source.lookup(0, 2) is None
        assert source.admit(0, 2, set()) is None
        assert source.lookup(0, 1) == 0
    assert clone.replay_snapshot() == policy.replay_snapshot()


def test_replay_reproduces_every_transition_before_comparing_cache_policies():
    from flexmoe.analysis.decode_replay import replay_trace

    result = replay_trace(
        initial_state(),
        complete_rows(),
        expert_bytes=10,
        layers=1,
        ingress_slots=4,
    )

    assert result["status"] == "baseline-reproduced"
    assert result["current"] == {
        "misses": 2,
        "loaded_bytes": 20,
        "bypasses": 0,
        "admissions": 2,
        "evictions": 2,
        "per_layer_misses": [2],
        "per_layer_loaded_bytes": [20],
        "reuse_distance_histogram": {
            "1": 0,
            "2-4": 3,
            "5-16": 0,
            "17-64": 0,
            "65+": 0,
        },
        "reuse_distance_p50": 2.0,
        "reuse_distance_p95": 2.0,
        "reuse_distance_p99": 2.0,
    }
    assert result["lru"]["misses"] == 2
    assert result["future_aware"]["misses"] == 1
    assert result["future_aware"]["saved_loaded_bytes"] == 10
    assert (
        result["interpretation"]
        == "cache-eviction-only; no prefetch or throughput gain"
    )


def test_replay_stops_before_oracles_on_recorded_transition_mismatch():
    from flexmoe.analysis.decode_replay import replay_trace

    rows = complete_rows()
    rows[1]["evicted_ids"] = [3]
    result = replay_trace(
        initial_state(), rows, expert_bytes=10, layers=1, ingress_slots=4
    )

    assert result["status"] == "baseline-mismatch"
    assert result["mismatch"] == {
        "step_id": 1,
        "layer_id": 0,
        "field": "evicted_ids",
        "recorded": [3],
        "replayed": [1],
    }
    assert "lru" not in result
    assert "future_aware" not in result


def test_replay_rejects_noncontiguous_window_and_ingress_overflow():
    from flexmoe.analysis.decode_replay import replay_trace

    rows = complete_rows()
    rows[2]["step_id"] = 5
    assert (
        replay_trace(initial_state(), rows, expert_bytes=10, layers=1, ingress_slots=4)[
            "status"
        ]
        == "invalid-window"
    )

    rows = complete_rows()
    rows[0]["actual_expert_ids"] = [0, 1, 2, 3]
    rows[0]["miss_ids"] = [0, 1, 2, 3]
    rows[0]["admitted_ids"] = [0, 1, 2, 3]
    assert (
        replay_trace(initial_state(), rows, expert_bytes=10, layers=1, ingress_slots=3)[
            "status"
        ]
        == "invalid-window"
    )


def test_snapshot_rejects_out_of_range_cached_expert():
    invalid = copy.deepcopy(initial_state())
    invalid["slots"][0] = [0, 99]
    with pytest.raises(ValueError):
        ExpertCachePolicy.from_replay_snapshot(invalid)


def test_profile_file_requires_full_trace_fields_and_sixty_four_steps(tmp_path):
    import gzip
    import json

    from flexmoe.analysis.decode_replay import replay_file

    initial, rows, expert_bytes = profiled_rows()
    profile = {
        "artifact_kind": "decode-profile",
        "profile": True,
        "geometry": {"total_layers": 2, "expert_bytes": expert_bytes},
        "observation": {
            "coverage_status": "complete",
            "captured_steps": 64,
            "pool_profile": {
                "initial_policy_state": initial,
                "rows": rows,
                "dropped_rows": 0,
            },
        },
    }
    path = tmp_path / "worker.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump(profile, stream)

    result = replay_file(path)

    assert result["status"] == "baseline-reproduced"
    assert result["complete_steps"] == 64
