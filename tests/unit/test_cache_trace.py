from flexmoe.runtime.expert_cache_policy import ExpertCachePolicy


def test_cache_trace_row_serializes_exact_decisions_without_weights():
    from flexmoe.runtime.cache_trace import CacheTraceRow

    policy = ExpertCachePolicy(
        1, 4, 0.25, 1, policy="lru", initial_counts=[[1, 9, 2, 3]]
    )
    before = policy.trace_snapshot()
    policy.observe(0, [1, 2, 3])
    assert policy.admit(0, 2, set()) == (0, None)
    after = policy.trace_snapshot()
    row = CacheTraceRow(
        step_id=7,
        layer_id=0,
        actual_batch=16,
        actual_expert_ids=(1, 2, 3),
        resident_hit_ids=(1,),
        cache_hit_ids=(),
        miss_ids=(2, 3),
        bypass_ids=(3,),
        admitted_ids=(2,),
        evicted_ids=(),
        cache_state_before=before,
        cache_state_after=after,
        loaded_bytes=48,
    ).to_dict()

    assert row["step_id"] == 7
    assert row["actual_expert_ids"] == [1, 2, 3]
    assert row["cache_state_before"]["slots"] == [None]
    assert row["cache_state_after"]["slots"] == [[0, 2]]
    assert row["loaded_bytes"] == 48
    assert "weights" not in row["cache_state_after"]


def test_real_pool_trace_records_exact_decisions_and_cache_states():
    from test_expert_cache_runtime import TensorBackend, invoke, make_pool

    from flexmoe.runtime.expert_pool import ExpertPool, PoolProfileObserver

    original, sources = make_pool(ratio=0.25, slots=1)
    pool = ExpertPool(
        original.policy,
        sources,
        backend=TensorBackend(),
        track_load_history=True,
    )
    observer = PoolProfileObserver(
        lambda layer: {"step": 7, "actual_batch": 16, "phase": "decode"},
        capacity=1,
        device="cpu",
    )
    pool.profile_observer = observer

    invoke(pool, sources, layer=0, ids=[1, 2, 3])
    row = observer.snapshot()["rows"][0]

    assert row["step_id"] == 7
    assert row["layer_id"] == 0
    assert row["actual_expert_ids"] == [1, 2, 3]
    assert row["resident_hit_ids"] == [1]
    assert row["cache_hit_ids"] == []
    assert row["miss_ids"] == [2, 3]
    assert row["admitted_ids"] == [2]
    assert row["bypass_ids"] == [3]
    assert row["evicted_ids"] == []
    assert row["cache_state_before"]["slots"] == [None]
    assert row["cache_state_after"]["slots"] == [[0, 2]]
    assert row["loaded_bytes"] == 2 * pool.expert_bytes
