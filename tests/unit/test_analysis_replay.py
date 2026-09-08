from __future__ import annotations

import pytest


def contract(prompt: str = "a") -> dict[str, object]:
    return {
        "model_identity_sha256": "1" * 64,
        "model_config_sha256": "2" * 64,
        "dataset_sha256": "3" * 64,
        "input_sha256": prompt * 64,
        "commit": "4" * 40,
        "tensor_parallel_size": 4,
        "batch_size": 2,
        "context_length": 8,
        "output_length": 2,
        "max_num_seqs": 2,
        "max_num_batched_tokens": 16,
        "gpu_memory_utilization": 0.9,
        "engine_policy_sha256": "5" * 64,
        "versions": {
            "torch": "2.8.0",
            "vllm": "0.10.2",
            "cuda": "12.8",
            "vllm_commit": "6" * 40,
        },
        "hardware_sha256": "7" * 64,
        "prompt_hashes": ["8" * 64, "9" * 64],
        "seed": 7,
    }


def make_trace(
    expert_rows: list[list[tuple[int, ...]]],
    *,
    prompt: str = "a",
    prompt_hashes: list[str] | None = None,
    token_rows: int = 3,
    full_workload: bool = True,
):
    from flexmoe.analysis.schema import DemandTrace, TraceEvent

    raw_contract = contract(prompt)
    if prompt_hashes is not None:
        raw_contract["prompt_hashes"] = prompt_hashes
    layer_count = len(expert_rows[0])
    rows = tuple(
        TraceEvent(step, layer, token_rows, 2, "decode", experts)
        for step, step_rows in enumerate(expert_rows)
        for layer, experts in enumerate(step_rows)
    )
    return DemandTrace(
        f"trace-{prompt}",
        0,
        raw_contract,
        layer_count,
        5,
        2,
        100,
        len(expert_rows) + (0 if full_workload else 1),
        len(expert_rows),
        full_workload,
        4,
        rows,
    )


def test_select_resident_uses_per_event_frequency_ceil_and_id_ties():
    from flexmoe.analysis.replay import select_resident

    calibration = make_trace(
        [[(3,), (4,)], [(3,), (0,)]],
        prompt="a",
        prompt_hashes=["a" * 64, "b" * 64],
    )
    evaluation = make_trace(
        [[(1,), (2,)], [(2,), (3,)]],
        prompt="b",
        prompt_hashes=["c" * 64, "d" * 64],
    )
    # ceil(5 * (1 - 0.5)) = 3. Layer 0 breaks the zero-count tie by ID.
    assert select_resident(calibration, evaluation, 0.5) == ((0, 1, 3), (0, 1, 4))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.contract.update(input_sha256="a" * 64),
        lambda row: row.contract.update(prompt_hashes=("a" * 64, "c" * 64)),
        lambda row: object.__setattr__(row, "num_experts", 4),
        lambda row: row.contract.update(model_identity_sha256="0" * 64),
    ],
)
def test_select_resident_rejects_calibration_leakage_or_mismatch(mutate):
    from flexmoe.analysis.replay import select_resident

    calibration = make_trace(
        [[(3,), (4,)], [(3,), (0,)]],
        prompt="a",
        prompt_hashes=["a" * 64, "b" * 64],
    )
    evaluation = make_trace(
        [[(1,), (2,)], [(2,), (3,)]],
        prompt="b",
        prompt_hashes=["c" * 64, "d" * 64],
    )
    mutate(evaluation)
    with pytest.raises(ValueError):
        select_resident(calibration, evaluation, 0.5)


def test_replay_counts_unique_event_demands_and_persistent_cache_hits():
    from flexmoe.analysis.replay import replay_trace
    from flexmoe.analysis.schema import ReplayConfig, ReplayResult

    row = make_trace(
        [[(0, 1), (0, 1)], [(1, 2), (1, 2)]],
    )
    config = ReplayConfig(((0,), (0,)), 4, 2, "lru", 25)
    result = replay_trace(row, config)
    assert result.demands == 8
    assert result.resident_hits == 2
    assert result.cache_hits == 2
    assert result.loaded_experts == 4
    assert result.loaded_bytes == 400
    assert result.event_load_counts == (1, 1, 1, 1)
    assert result.event_load_bytes == (100, 100, 100, 100)
    assert result.bytes_per_generated_token == 100
    assert result.net_freed_bytes == 175  # (2*5 - 2 - 4 - 2)*100 - 25
    assert result.final_cache == ((0, 1), (0, 2), (1, 1), (1, 2))
    assert result.per_layer_totals[0] == {
        "demands": 4,
        "resident_hits": 1,
        "cache_hits": 1,
        "loaded_experts": 2,
        "loaded_bytes": 200,
    }
    assert result.per_phase_totals["decode"]["loaded_experts"] == 4
    assert ReplayResult.from_dict(result.to_dict()) == result

    malformed = result.to_dict()
    malformed["event_load_bytes"] = [101, 99, 100, 100]
    with pytest.raises(ValueError):
        ReplayResult.from_dict(malformed)


def test_net_freed_bytes_preserves_a_negative_capacity_result():
    from flexmoe.analysis.replay import replay_trace
    from flexmoe.analysis.schema import ReplayConfig

    row = make_trace([[(1,)], [(2,)]])
    result = replay_trace(row, ReplayConfig(((),), 5, 2, "lru", 25))
    assert result.net_freed_bytes == -225  # (1*5 - 0 - 5 - 2)*100 - 25


def test_replay_result_rejects_nonintegral_per_expert_event_bytes():
    from flexmoe.analysis.replay import replay_trace
    from flexmoe.analysis.schema import ReplayConfig, ReplayResult

    result = replay_trace(
        make_trace([[(1, 2)], [(3, 4)]]),
        ReplayConfig(((),), 0, 2, "lru"),
    )
    malformed = result.to_dict()
    malformed["event_load_bytes"] = [201, 199]
    with pytest.raises(ValueError):
        ReplayResult.from_dict(malformed)


def test_zero_cache_reloads_while_sufficient_cache_loads_only_first_visit():
    from flexmoe.analysis.replay import replay_trace
    from flexmoe.analysis.schema import ReplayConfig

    row = make_trace([[(1,)], [(1,)]])
    zero = replay_trace(row, ReplayConfig(((),), 0, 1, "lru"))
    enough = replay_trace(row, ReplayConfig(((),), 1, 1, "lru"))
    assert (zero.loaded_experts, zero.cache_hits) == (2, 0)
    assert (enough.loaded_experts, enough.cache_hits) == (1, 1)


def test_identical_expert_ids_in_different_layers_are_different_cache_keys():
    from flexmoe.analysis.replay import replay_trace
    from flexmoe.analysis.schema import ReplayConfig

    row = make_trace([[(1,), (1,)]])
    result = replay_trace(row, ReplayConfig(((), ()), 2, 1, "lru"))
    assert result.loaded_experts == 2
    assert result.final_cache == ((0, 1), (1, 1))


def test_current_event_hits_are_protected_while_misses_are_admitted():
    from flexmoe.analysis.replay import replay_trace
    from flexmoe.analysis.schema import ReplayConfig

    row = make_trace([[(1, 2)], [(1, 3, 4)]], token_rows=3)
    result = replay_trace(row, ReplayConfig(((),), 2, 3, "lru"))
    assert result.event_load_counts == (2, 2)
    assert result.cache_hits == 1


def test_token_rows_do_not_multiply_one_expert_load_for_an_event():
    from flexmoe.analysis.replay import replay_trace
    from flexmoe.analysis.schema import ReplayConfig

    row = make_trace([[(2,)]], token_rows=100)
    result = replay_trace(row, ReplayConfig(((),), 0, 1, "lru"))
    assert result.demands == 1
    assert result.loaded_experts == 1
    assert result.loaded_bytes == 100


def test_lru_decayed_lfu_and_future_are_deterministic_diagnostics():
    from flexmoe.analysis.replay import replay_trace
    from flexmoe.analysis.schema import ReplayConfig

    row = make_trace([[(0, 1)], [(0, 2)], [(1,)]])
    lru = replay_trace(row, ReplayConfig(((),), 2, 2, "lru"))
    lfu = replay_trace(row, ReplayConfig(((),), 2, 2, "decayed-lfu"))
    future = replay_trace(row, ReplayConfig(((),), 2, 2, "future"))
    assert (
        lru.final_cache
        == replay_trace(row, ReplayConfig(((),), 2, 2, "lru")).final_cache
    )
    assert (
        lfu.final_cache
        == replay_trace(row, ReplayConfig(((),), 2, 2, "decayed-lfu")).final_cache
    )
    assert future.final_cache == ((0, 0), (0, 1))
    assert future.loaded_experts == 3
    assert lru.loaded_experts == 4
    assert future.future_policy_reference is True
    assert "ideal-future-reference-not-deployed" in future.assumptions


def test_partial_trace_has_window_counts_without_generated_token_normalization():
    from flexmoe.analysis.replay import replay_trace
    from flexmoe.analysis.schema import ReplayConfig

    partial = make_trace([[(1,)], [(2,)]], full_workload=False)
    result = replay_trace(partial, ReplayConfig(((),), 0, 1, "lru"))
    assert result.loaded_bytes == 200
    assert result.bytes_per_generated_token is None
    assert "partial-window-not-full-workload" in result.assumptions


def test_staging_overflow_is_counted_as_unmodelled_cost():
    from flexmoe.analysis.replay import replay_trace
    from flexmoe.analysis.schema import ReplayConfig

    row = make_trace([[(1, 2, 3)]], token_rows=3)
    result = replay_trace(row, ReplayConfig(((),), 0, 2, "lru"))
    assert result.staging_overflow_events == 1
    assert "staging-overflow-requires-unmodelled-chunking" in result.assumptions


@pytest.mark.parametrize("policy", ["lru", "decayed-lfu", "future"])
def test_replay_scales_with_event_and_cache_state_not_event_cache_snapshots(policy):
    from flexmoe.analysis.replay import replay_trace
    from flexmoe.analysis.schema import ReplayConfig

    rows = [[tuple(sorted((step % 5, (step + 1) % 5)))] for step in range(200)]
    result = replay_trace(make_trace(rows), ReplayConfig(((),), 3, 2, policy))
    assert len(result.event_load_counts) == 200
    assert len(result.final_cache) <= 3
    assert "event_cache_after" not in result.to_dict()
