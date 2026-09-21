import pytest
import torch


class SyncTransfer:
    def __init__(self):
        self.serial = 0

    def enqueue(self, copies):
        self.serial += 1
        for destination, source in copies:
            destination.copy_(source)
        return self.serial

    def query(self, handle):
        return True

    def wait(self, handle):
        return None

    def wait_host(self, handle):
        pass

    def elapsed_ms(self, handle):
        return 0.0

    def wait_elapsed_ms(self, handle):
        return None

    def synchronize(self):
        pass

    def defer_reuse_until_current_stream(self):
        pass


def trace():
    from flexmoe.runtime.oracle_trace import OracleTrace

    return OracleTrace.from_rows(
        [
            {
                "step_id": 0,
                "layer_id": 0,
                "actual_batch": 1,
                "actual_expert_ids": [0, 1],
            },
            {
                "step_id": 0,
                "layer_id": 1,
                "actual_batch": 1,
                "actual_expert_ids": [2, 3],
            },
        ],
        {
            "geometry": {"total_layers": 2, "num_experts": 4, "top_k": 2},
            "actual_batch": 1,
        },
    )


def make_pair():
    from test_expert_cache_runtime import TensorBackend, make_pool

    from flexmoe.runtime.expert_cache_policy import ExpertCachePolicy
    from flexmoe.runtime.expert_pool import ExpertPool

    _, sources = make_pool(slots=2)

    def pool():
        policy = ExpertCachePolicy(
            2,
            4,
            0.0,
            2,
            policy="lru",
            initial_counts=[[1, 9, 2, 3], [8, 1, 2, 3]],
        )
        return ExpertPool(policy, sources, backend=TensorBackend())

    return pool(), pool(), sources


def execute_step(pool, sources):
    outputs = []
    for layer, ids in enumerate(([0, 1], [2, 3])):
        tensor_ids = torch.tensor([ids], dtype=torch.int32)

        def compute(w13, w2, mapping, *, current_layer=layer, current_ids=ids):
            for expert in current_ids:
                slot = int(mapping[expert])
                assert torch.equal(w13[slot], sources[current_layer][0][expert])
                assert torch.equal(w2[slot], sources[current_layer][1][expert])
            return torch.tensor([current_layer, sum(current_ids)])

        outputs.append(pool.execute(layer, tensor_ids, compute))
    return outputs


@pytest.mark.parametrize("horizon", [0, 1, 2])
def test_oracle_horizons_preserve_outputs_and_policy_transitions(horizon):
    baseline, oracle, sources = make_pair()
    context = {"step": 0, "actual_batch": 1, "phase": "decode"}
    oracle.configure_oracle(
        trace(),
        horizon=horizon,
        context=lambda layer: context,
        transfer_backend=SyncTransfer(),
    )

    baseline_outputs = execute_step(baseline, sources)
    oracle_outputs = execute_step(oracle, sources)

    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(oracle_outputs, baseline_outputs, strict=True)
    )
    assert oracle.policy.replay_snapshot() == baseline.policy.replay_snapshot()
    assert oracle.stats()["oracle"]["route_mismatch_count"] == 0
    assert oracle.stats()["oracle"]["prefetch_horizon"] == horizon


def test_forced_omission_falls_back_once_without_route_substitution():
    baseline, oracle, sources = make_pair()
    omitted = trace().with_forced_omission(step=0, layer=1, expert=3)
    context = {"step": 0, "actual_batch": 1, "phase": "decode"}
    oracle.configure_oracle(
        omitted,
        horizon=1,
        context=lambda layer: context,
        transfer_backend=SyncTransfer(),
    )

    expected = execute_step(baseline, sources)
    actual = execute_step(oracle, sources)
    stats = oracle.stats()["oracle"]

    assert all(
        torch.equal(value, reference)
        for value, reference in zip(actual, expected, strict=True)
    )
    assert stats["forced_omission_count"] == 1
    assert stats["forced_fallback_count"] == 1
    assert stats["fallback_on_demand_count"] == 3
    assert stats["route_mismatch_count"] == 0
    assert oracle.policy.replay_snapshot() == baseline.policy.replay_snapshot()


def test_steps_outside_bounded_trace_fall_back_without_route_mismatch():
    baseline, oracle, sources = make_pair()
    context = {"step": 1, "actual_batch": 1, "phase": "decode"}
    oracle.configure_oracle(
        trace(),
        horizon=1,
        context=lambda layer: context,
        transfer_backend=SyncTransfer(),
    )

    expected = execute_step(baseline, sources)
    actual = execute_step(oracle, sources)
    stats = oracle.stats()["oracle"]

    assert all(
        torch.equal(value, reference)
        for value, reference in zip(actual, expected, strict=True)
    )
    assert stats["route_mismatch_count"] == 0
    assert stats["trace_unavailable_count"] == 2
    assert stats["route_match_count"] == 0
    assert stats["fallback_on_demand_count"] == 4
    assert stats["timing_enabled"] is False
    assert stats["lookahead_ms"] == []
    assert stats["host_buffer_reuse_wait_timing_status"] == "unavailable"
