import pytest
import torch
from test_expert_cache_runtime import TensorBackend, invoke, make_pool


def observed_pool():
    from flexmoe.runtime.expert_pool import ExpertPool

    old, sources = make_pool(slots=1)
    return ExpertPool(
        old.policy, sources, backend=TensorBackend(), track_load_history=True
    ), sources


def test_real_pool_first_load_hit_eviction_reload_and_observer_parity():
    from flexmoe.runtime.expert_pool import PoolProfileObserver

    pool, sources = observed_pool()
    plain, other = make_pool(slots=1)
    # Warmup loads expert 1 before profile begins; lifecycle history survives.
    invoke(pool, sources, 0, [1])
    invoke(plain, other, 0, [1])
    current = {"step": 0, "actual_batch": 1, "phase": "decode"}
    observer = PoolProfileObserver(lambda layer: current, capacity=4, device="cpu")
    pool.profile_observer = observer
    for step, ids in enumerate(([1], [2], [1], [1, 3])):
        current["step"] = step
        invoke(pool, sources, 0, ids)
        invoke(plain, other, 0, ids)
    rows = observer.snapshot()["rows"]
    assert [r["unique_misses"] for r in rows] == [0, 1, 1, 1]
    assert [r["first_loads"] for r in rows] == [0, 1, 0, 1]
    assert [r["reloads"] for r in rows] == [0, 0, 1, 0]
    assert [r["cache_hits"] for r in rows] == [1, 0, 0, 1]
    assert [r["evictions"] for r in rows] == [0, 1, 1, 0]
    assert [r["bypasses"] for r in rows] == [0, 0, 0, 1]
    assert [r["loaded_bytes"] for r in rows] == [0, 24, 24, 24]
    assert rows[0]["cuda_timing"]["status"] == "unavailable"
    assert rows[0]["prefetch_status"] == "not-applicable"
    actual, reference = pool.stats(), plain.stats()
    for key in ("h2d_bytes", "gpu_pool_bytes", "policy", "forward_counts"):
        assert actual[key] == reference[key]


def test_actual_event_boundaries_not_cpu_plus_gpu_sum_and_capacity():
    from flexmoe.runtime.expert_pool import PoolProfileObserver

    pool, sources = observed_pool()
    times = iter([0.0, 0.0015, 0.002, 0.005, 0.006])

    class Event:
        def __init__(self):
            self.time = next(times)

        def query(self):
            return True

        def elapsed_time(self, other):
            return (other.time - self.time) * 1000

    observer = PoolProfileObserver(
        lambda layer: {"step": 2, "actual_batch": 1, "phase": "decode"},
        capacity=1,
        device="cpu",
        event_factory=Event,
    )
    pool.profile_observer = observer
    invoke(pool, sources, 0, [1])
    invoke(pool, sources, 0, [2])
    result = observer.snapshot()
    row = result["rows"][0]
    assert result["dropped_rows"] == 1
    assert row["cuda_timing"] == {
        "status": "measured",
        "load_s": 0.002,
        "payload_s": 0.0015,
        "metadata_s": 0.0005,
        "compute_s": 0.003,
        "promotion_s": 0.001,
        "span_s": 0.006,
    }
    assert "total_tax_s" not in row
    assert row["cpu_timing"]["status"] == "measured"


def test_unready_or_interrupted_events_preserved_without_forced_layer_sync():
    from flexmoe.runtime.expert_pool import PoolProfileObserver

    pool, sources = observed_pool()

    class Event:
        def query(self):
            return False

        def elapsed_time(self, other):
            raise AssertionError("read timing before event ready")

    observer = PoolProfileObserver(
        lambda layer: {"step": 0, "actual_batch": 1, "phase": "decode"},
        capacity=2,
        device="cpu",
        event_factory=Event,
    )
    pool.profile_observer = observer
    invoke(pool, sources, 0, [1])
    with pytest.raises(RuntimeError, match="compute"):
        pool.execute(
            0,
            torch.tensor([[2]]),
            lambda *args: (_ for _ in ()).throw(RuntimeError("compute")),
        )
    result = observer.snapshot()
    assert [r["cuda_timing"]["status"] for r in result["rows"]] == [
        "unavailable",
        "unavailable",
    ]
    assert [r["status"] for r in result["rows"]] == ["complete", "failed"]
