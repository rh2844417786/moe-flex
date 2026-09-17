import copy

import pytest

from flexmoe.runtime.expert_cache_policy import ExpertCachePolicy


def seeded_policy():
    policy = ExpertCachePolicy(1, 4, 0.0, 2, initial_counts=[[100, 90, 0, 0]])
    assert policy.admit(0, 0, set()) == (0, None)
    assert policy.admit(0, 1, set()) == (1, None)
    return policy


def demand_rows():
    ids = (2, 0, 2, 0, 2)
    return [
        {
            "step": step,
            "layer": 0,
            "actual_batch": 1,
            "demand_ids": [expert],
            "unique_misses": 1 if expert == 2 else 0,
            "loaded_bytes": 10 if expert == 2 else 0,
            "status": "complete",
        }
        for step, expert in enumerate(ids)
    ]


def test_snapshot_keeps_fixed_residency_scores_clock_and_cache_occupancy():
    policy = ExpertCachePolicy(
        2, 4, 0.25, 1, decay_interval=2, initial_counts=[[10, 3, 2, 0], [8, 5, 1, 0]]
    )
    assert policy.admit(0, 1, set()) == (0, None)
    policy.observe(0, [1, 2])
    policy.observe(1, [2])
    from flexmoe.runtime.expert_cache_policy import ExpertCachePolicy as Policy

    clone = Policy.from_replay_snapshot(policy.replay_snapshot())
    assert clone.resident_ids == ((0,), (0,))
    for source in (policy, clone):
        source.observe(0, [2])
        assert source.lookup(0, 2) is None
        assert source.admit(0, 2, set()) is None
        assert source.lookup(0, 1) == 0
    assert clone.replay_snapshot() == policy.replay_snapshot()


def test_replay_exactly_reproduces_observed_misses_before_comparing_cache_policies():
    from flexmoe.analysis.decode_replay import replay_trace

    result = replay_trace(
        seeded_policy().replay_snapshot(), demand_rows(), expert_bytes=10, layers=1
    )
    assert result["status"] == "baseline-reproduced"
    assert result["current_policy_misses"] == 3
    assert result["lru_misses"] >= 1
    assert result["future_aware_misses"] == 1
    assert result["future_aware_saved_bytes"] == 20
    assert (
        result["interpretation"]
        == "cache-eviction-only; no prefetch or throughput gain"
    )


def test_replay_rejects_noncontiguous_window_and_mismatch_without_oracle_claim():
    from flexmoe.analysis.decode_replay import replay_trace

    rows = demand_rows()
    rows[2]["step"] = 5
    assert (
        replay_trace(
            seeded_policy().replay_snapshot(), rows, expert_bytes=10, layers=1
        )["status"]
        == "invalid-window"
    )
    rows = demand_rows()
    rows[2]["unique_misses"] = 0
    assert (
        replay_trace(
            seeded_policy().replay_snapshot(), rows, expert_bytes=10, layers=1
        )["status"]
        == "baseline-mismatch"
    )


def test_replay_rejects_faked_bytes_and_duplicate_demand_ids():
    from flexmoe.analysis.decode_replay import replay_trace

    rows = demand_rows()
    rows[0]["loaded_bytes"] = 0
    assert (
        replay_trace(
            seeded_policy().replay_snapshot(), rows, expert_bytes=10, layers=1
        )["status"]
        == "baseline-mismatch"
    )
    rows = demand_rows()
    rows[0]["demand_ids"] = [2, 2]
    assert (
        replay_trace(
            seeded_policy().replay_snapshot(), rows, expert_bytes=10, layers=1
        )["status"]
        == "invalid-window"
    )


def test_snapshot_rejects_out_of_range_cached_expert():
    snapshot = seeded_policy().replay_snapshot()
    invalid = copy.deepcopy(snapshot)
    invalid["slots"][0] = [0, 99]
    with pytest.raises(ValueError):
        ExpertCachePolicy.from_replay_snapshot(invalid)


def test_future_aware_cache_bypasses_one_off_expert_to_keep_future_hit():
    from flexmoe.analysis.decode_replay import replay_trace

    policy = ExpertCachePolicy(1, 4, 0.0, 1, initial_counts=[[100, 0, 0, 0]])
    assert policy.admit(0, 0, set()) == (0, None)
    rows = [
        {
            "step": i,
            "layer": 0,
            "actual_batch": 1,
            "demand_ids": [expert],
            "unique_misses": int(i == 0),
            "loaded_bytes": 10 if i == 0 else 0,
            "status": "complete",
        }
        for i, expert in enumerate((3, 0))
    ]
    result = replay_trace(policy.replay_snapshot(), rows, expert_bytes=10, layers=1)
    assert result["status"] == "baseline-reproduced"
    assert result["future_aware_misses"] == 1


def test_compressed_worker_profile_can_be_replayed_without_importing_gpu_runtime(
    tmp_path,
):
    import gzip
    import json

    from flexmoe.analysis.decode_replay import replay_file

    policy = ExpertCachePolicy(1, 4, 0.0, 1)
    rows = [
        {
            "step": step,
            "layer": 0,
            "actual_batch": 16,
            "demand_ids": [1],
            "unique_misses": int(step == 0),
            "loaded_bytes": 10 if step == 0 else 0,
            "status": "complete",
        }
        for step in range(64)
    ]
    profile = {
        "artifact_kind": "decode-profile",
        "profile": True,
        "geometry": {"total_layers": 1, "expert_bytes": 10},
        "observation": {
            "coverage_status": "complete",
            "captured_steps": 64,
            "pool_profile": {
                "dropped_rows": 0,
                "rows": rows,
                "initial_policy_state": policy.replay_snapshot(),
            },
        },
    }
    path = tmp_path / "worker.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump(profile, stream)
    result = replay_file(path)
    assert result["status"] == "baseline-reproduced"
    assert result["complete_steps"] == 64
