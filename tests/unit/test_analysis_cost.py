from __future__ import annotations

from dataclasses import replace

import pytest


def contract(*, hardware: str | None = "7" * 64) -> dict[str, object]:
    result: dict[str, object] = {
        "model_identity_sha256": "1" * 64,
        "model_config_sha256": "2" * 64,
        "dataset_sha256": "3" * 64,
        "input_sha256": "4" * 64,
        "commit": "5" * 40,
        "tensor_parallel_size": 4,
        "batch_size": 10,
        "context_length": 8,
        "output_length": 100,
        "max_num_seqs": 10,
        "max_num_batched_tokens": 80,
        "gpu_memory_utilization": 0.9,
        "engine_policy_sha256": "6" * 64,
        "versions": {
            "torch": "2.8.0",
            "vllm": "0.10.2",
            "cuda": "12.8",
            "vllm_commit": "8" * 40,
        },
        "hardware_sha256": hardware,
        "prompt_hashes": ["9" * 64, "a" * 64],
        "seed": 11,
        "dtype": "bfloat16",
        "sampling_policy": "existing-prompts",
        "dataset_manifest_sha256": "d" * 64,
        "source_request_count": 10,
        "unique_selected_request_count": 10,
        "repeated_request_count": 0,
    }
    if hardware is None:
        result["hardware_unavailable_reason"] = "uuid-query-unavailable"
    return result


def transport_contract(*, hardware: str | None = "7" * 64) -> dict[str, object]:
    result: dict[str, object] = {
        "commit": "5" * 40,
        "tensor_parallel_size": 4,
        "versions": {
            "torch": "2.8.0",
            "vllm": "0.10.2",
            "cuda": "12.8",
            "vllm_commit": "8" * 40,
        },
        "hardware_sha256": hardware,
        "measurement_id": "transport-001",
        "benchmark_policy_sha256": "c" * 64,
    }
    if hardware is None:
        result["hardware_unavailable_reason"] = "uuid-query-unavailable"
    return result


def trace(rank: int, *, event_rows=((5,), (6,)), hardware: str | None = "7" * 64):
    from flexmoe.analysis.schema import DemandTrace, TraceEvent

    return DemandTrace(
        f"trace-r{rank}",
        rank,
        contract(hardware=hardware),
        1,
        10,
        2,
        100,
        len(event_rows),
        len(event_rows),
        True,
        1000,
        tuple(
            TraceEvent(step, 0, 1, 1, "decode", experts)
            for step, experts in enumerate(event_rows)
        ),
    )


def trace_replays(*, event_rows=((5,), (6,)), hardware: str | None = "7" * 64):
    from flexmoe.analysis.replay import replay_trace
    from flexmoe.analysis.schema import ReplayConfig

    traces = tuple(
        trace(rank, event_rows=event_rows, hardware=hardware) for rank in range(4)
    )
    config = ReplayConfig(((0, 1, 2, 3, 4),), 0, 2, "lru")
    return traces, tuple(replay_trace(row, config) for row in traces)


def timing(
    point_id: str,
    seconds: float,
    *,
    kv: int,
    source_kind: str = "native-measured",
    engine_mode: str = "native",
    hardware: str | None = "7" * 64,
    utilization: float = 0.9,
):
    from flexmoe.analysis.schema import TimingPoint

    raw = contract(hardware=hardware)
    raw["gpu_memory_utilization"] = utilization
    if utilization != 0.9:
        raw["engine_policy_sha256"] = "b" * 64
    return TimingPoint(
        point_id,
        raw,
        1000,
        (seconds, seconds, seconds),
        (kv, kv, kv, kv),
        (1000, 1000, 1000, 1000),
        source_kind,
        engine_mode,
    )


def samples(
    *,
    slow_rank: int = 3,
    slow_s: float = 0.1,
    event_sizes=(1,),
    hardware: str | None = "7" * 64,
):
    from flexmoe.analysis.schema import TransferSample

    rows = []
    for rank in range(4):
        for size in event_sizes:
            wall = slow_s if rank == slow_rank else 0.05
            rows.append(
                TransferSample(
                    rank,
                    transport_contract(hardware=hardware),
                    size,
                    100,
                    "gather",
                    "isolated",
                    0,
                    size * 100,
                    wall,
                    wall * 0.75,
                    wall * 0.25,
                    None,
                )
            )
    return tuple(rows)


def analyze(*, sample_rows=None, kv_reference=True, hardware: str | None = "7" * 64):
    from flexmoe.analysis.cost import analyze_feasibility

    traces, replays = trace_replays(hardware=hardware)
    reference = timing("k", 2 / 3, kv=300, hardware=hardware) if kv_reference else None
    return analyze_feasibility(
        traces,
        replays,
        samples(hardware=hardware) if sample_rows is None else sample_rows,
        timing("r", 1, kv=100, hardware=hardware),
        reference,
    )


def test_literal_prediction_uses_bottleneck_rank_and_actual_kv_delta():
    result = analyze()
    assert result["diagnostic_only"] is True
    assert result["formal_offload_gain"] is False
    assert result["deployment_gain_proven"] is False
    assert result["status"] == "candidate"
    assert result["candidate_class"] == "serial-profitable"
    assert result["actual_kv_increment_bytes_by_rank"] == [200, 200, 200, 200]
    assert result["time_headroom_s"] == pytest.approx(1 / 3)
    assert result["transport"]["bottleneck_rank"] == 3
    assert result["transport"]["serial_layer_barrier_service_s"] == pytest.approx(0.2)
    assert result["predictions"]["serial_layer_barrier"][
        "throughput_tps"
    ] == pytest.approx(1153.8461538)
    assert result["predictions"]["optimistic_full_overlap"][
        "throughput_tps"
    ] == pytest.approx(1500)


def test_transfer_service_larger_than_headroom_is_overlap_dependent_candidate():
    result = analyze(sample_rows=samples(slow_s=0.2))
    assert result["transport"]["serial_layer_barrier_service_s"] == pytest.approx(0.4)
    assert result["predictions"]["serial_layer_barrier"][
        "throughput_tps"
    ] == pytest.approx(937.5)
    assert result["status"] == "candidate"
    assert result["candidate_class"] == "overlap-required"
    assert result["required_overlap_to_match_baseline_s"] == pytest.approx(1 / 15)


def test_even_optimistic_service_slower_than_baseline_is_model_no_headroom():
    result = analyze(sample_rows=samples(slow_s=0.55))
    assert result["transport"]["optimistic_resource_service_s"] == pytest.approx(1.1)
    assert result["predictions"]["optimistic_full_overlap"][
        "throughput_tps"
    ] == pytest.approx(1000 / 1.1)
    assert result["status"] == "model-no-headroom"
    assert result["candidate_class"] == "neither-profitable"


def test_serial_barrier_sums_per_event_rank_maxima_for_alternating_stalls():
    from flexmoe.analysis.cost import analyze_feasibility
    from flexmoe.analysis.schema import TransferSample

    event_rows = ((5,), (6, 7))
    traces, replays = trace_replays(event_rows=event_rows)
    rows = []
    for rank in range(4):
        for size in (1, 2):
            wall = 0.15 if (rank, size) in {(0, 1), (1, 2)} else 0.05
            rows.append(
                TransferSample(
                    rank,
                    transport_contract(),
                    size,
                    100,
                    "gather",
                    "isolated",
                    0,
                    size * 100,
                    wall,
                    wall * 0.75,
                    wall * 0.25,
                    None,
                )
            )
    result = analyze_feasibility(
        traces, replays, rows, timing("r", 1, kv=100), timing("k", 2 / 3, kv=300)
    )
    per_rank = result["transport"]["per_rank"]
    assert [row["total_service_s"] for row in per_rank[:2]] == pytest.approx([0.2, 0.2])
    assert result["transport"]["optimistic_resource_service_s"] == pytest.approx(0.2)
    assert result["transport"]["serial_layer_barrier_service_s"] == pytest.approx(0.3)


@pytest.mark.parametrize("bad_rank", [0, 1, 2, 3])
def test_missing_trace_or_replay_rank_is_rejected(bad_rank):
    from flexmoe.analysis.cost import analyze_feasibility

    traces, replays = trace_replays()
    with pytest.raises(ValueError):
        analyze_feasibility(
            tuple(row for row in traces if row.rank != bad_rank),
            replays,
            samples(),
            timing("r", 1, kv=100),
            timing("k", 2 / 3, kv=300),
        )


def test_unmeasured_exact_shape_is_incomplete_without_extrapolation():
    rows = samples(event_sizes=(2,))
    result = analyze(sample_rows=rows)
    assert result["status"] == "incomplete"
    assert "unmeasured-exact-transfer-shape" in result["missing_evidence"]
    assert result["predictions"] is None


def test_no_k_reference_still_reports_transfer_and_bandwidth_requirements():
    result = analyze(kv_reference=False)
    assert result["status"] == "incomplete"
    assert "matching-kv-reference" in result["missing_evidence"]
    assert result["predictions"] is None
    assert result["transport"]["per_rank"][3]["loaded_bytes"] == 200
    assert result["bandwidth_requirements_bytes_s"][3] == pytest.approx(200)


def test_unavailable_hardware_identity_is_insufficient_evidence_not_a_match():
    result = analyze(hardware=None)
    assert result["status"] == "incomplete"
    assert "hardware-identity-unavailable" in result["missing_evidence"]
    assert result["predictions"] is None


def test_partial_trace_cannot_produce_throughput_prediction():
    from flexmoe.analysis.cost import analyze_feasibility
    from flexmoe.analysis.replay import replay_trace

    traces, _ = trace_replays()
    partial = tuple(
        replace(row, observed_steps=3, full_workload=False) for row in traces
    )
    replays = tuple(replay_trace(row, trace_replays()[1][0].config) for row in partial)
    result = analyze_feasibility(
        partial,
        replays,
        samples(),
        timing("r", 1, kv=100),
        timing("k", 2 / 3, kv=300),
    )
    assert result["status"] == "incomplete"
    assert "full-workload-four-rank-trace" in result["missing_evidence"]
    assert result["predictions"] is None


def test_transfer_expert_bytes_must_match_rank_trace():
    from flexmoe.analysis.cost import analyze_feasibility

    traces, replays = trace_replays()
    rows = list(samples())
    rows[0] = replace(rows[0], expert_bytes=200, payload_bytes=200)
    with pytest.raises(ValueError):
        analyze_feasibility(
            traces,
            replays,
            rows,
            timing("r", 1, kv=100),
            timing("k", 2 / 3, kv=300),
        )


def test_trace_engine_counterfactual_differences_are_visible_not_equivalent():
    from flexmoe.analysis.cost import analyze_feasibility

    traces, replays = trace_replays()
    for row in traces:
        row.contract["engine_policy_sha256"] = "c" * 64
        row.contract["gpu_memory_utilization"] = 0.6
    result = analyze_feasibility(
        traces,
        replays,
        samples(),
        timing("r", 1, kv=100),
        timing("k", 2 / 3, kv=300),
    )
    assert result["status"] == "candidate"
    assert "trace-engine-policy-differs" in result["assumptions"]
    assert "trace-utilization-differs" in result["assumptions"]


def test_kv_increment_above_any_rank_net_free_is_incomplete():
    from flexmoe.analysis.cost import analyze_feasibility

    traces, replays = trace_replays()
    result = analyze_feasibility(
        traces,
        replays,
        samples(),
        timing("r", 1, kv=100),
        timing("k", 2 / 3, kv=500),
    )
    assert result["status"] == "incomplete"
    assert "kv-increment-exceeds-net-freed" in result["missing_evidence"]
    assert result["predictions"] is None


def test_stale_replay_cannot_inflate_net_free_past_trace_config_formula():
    from flexmoe.analysis.cost import analyze_feasibility

    traces, replays = trace_replays()
    forged = tuple(replace(row, net_freed_bytes=400) for row in replays)
    with pytest.raises(ValueError, match="net_freed_bytes"):
        analyze_feasibility(
            traces,
            forged,
            samples(),
            timing("r", 1, kv=100),
            timing("k", 2 / 3, kv=500),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("bytes_per_generated_token", 999.0),
        ("staging_overflow_events", 1),
    ],
)
def test_stale_replay_trace_dependent_scalar_is_rejected(field, value):
    from flexmoe.analysis.cost import analyze_feasibility

    traces, replays = trace_replays()
    stale = list(replays)
    stale[0] = replace(stale[0], **{field: value})
    with pytest.raises(ValueError, match=field):
        analyze_feasibility(
            traces,
            stale,
            samples(),
            timing("r", 1, kv=100),
            timing("k", 2 / 3, kv=300),
        )


def test_self_consistent_forged_zero_load_replay_is_rejected():
    from flexmoe.analysis.cost import analyze_feasibility
    from flexmoe.analysis.schema import ReplayResult

    traces, replays = trace_replays()
    forged_row = replays[0].to_dict()
    forged_row.update(
        loaded_experts=0,
        loaded_bytes=0,
        resident_hits=2,
        cache_hits=0,
        event_load_counts=[0, 0],
        event_load_bytes=[0, 0],
        bytes_per_generated_token=0.0,
        per_layer_totals=[
            {
                "demands": 2,
                "resident_hits": 2,
                "cache_hits": 0,
                "loaded_experts": 0,
                "loaded_bytes": 0,
            }
        ],
        per_phase_totals={
            "decode": {
                "demands": 2,
                "resident_hits": 2,
                "cache_hits": 0,
                "loaded_experts": 0,
                "loaded_bytes": 0,
            }
        },
    )
    forged = list(replays)
    forged[0] = ReplayResult.from_dict(forged_row)
    with pytest.raises(ValueError, match="canonical replay"):
        analyze_feasibility(
            traces,
            forged,
            samples(),
            timing("r", 1, kv=100),
            timing("k", 2 / 3, kv=300),
        )


def test_serialized_replay_roundtrip_remains_valid_cost_evidence():
    from flexmoe.analysis.cost import analyze_feasibility
    from flexmoe.analysis.schema import ReplayResult

    traces, replays = trace_replays()
    restored = tuple(ReplayResult.from_dict(row.to_dict()) for row in replays)
    result = analyze_feasibility(
        traces,
        restored,
        samples(),
        timing("r", 1, kv=100),
        timing("k", 2 / 3, kv=300),
    )
    assert result["status"] == "candidate"
    assert result["transport"]["serial_layer_barrier_service_s"] == pytest.approx(
        0.2
    )


def test_zero_loads_are_complete_and_both_scenarios_use_k_time():
    from flexmoe.analysis.cost import analyze_feasibility
    from flexmoe.analysis.replay import replay_trace
    from flexmoe.analysis.schema import ReplayConfig

    traces = tuple(trace(rank, event_rows=((5,), (6,))) for rank in range(4))
    config = ReplayConfig(((5, 6),), 0, 2, "lru")
    replays = tuple(replay_trace(row, config) for row in traces)
    result = analyze_feasibility(
        traces,
        replays,
        samples(),
        timing("r", 1, kv=100),
        timing("k", 2 / 3, kv=300),
    )
    assert result["missing_evidence"] == []
    assert result["transport"]["serial_layer_barrier_service_s"] == 0.0
    assert result["transport"]["optimistic_resource_service_s"] == 0.0
    assert result["predictions"]["serial_layer_barrier"]["elapsed_s"] == pytest.approx(
        2 / 3
    )
    assert result["predictions"]["optimistic_full_overlap"][
        "elapsed_s"
    ] == pytest.approx(2 / 3)
    assert result["status"] == "candidate"


@pytest.mark.parametrize(
    "field,value",
    [("dtype", "float16"), ("sampling_policy", "different-prompts")],
)
def test_timing_reference_rejects_execution_or_sampling_change(field, value):
    from flexmoe.analysis.cost import analyze_feasibility

    traces, replays = trace_replays()
    reference = timing("k", 2 / 3, kv=300)
    reference.contract[field] = value
    with pytest.raises(ValueError, match=field):
        analyze_feasibility(
            traces,
            replays,
            samples(),
            timing("r", 1, kv=100),
            reference,
        )


def test_exact_decomposition_is_iterative_and_backtracks_deterministically():
    from flexmoe.analysis.cost import _decompose_exact

    unit_chunks = _decompose_exact(1024, (1,))
    assert unit_chunks is not None
    assert len(unit_chunks) == 1024
    assert unit_chunks[:3] == (1, 1, 1)
    assert _decompose_exact(6, (4, 3)) == (3, 3)


def test_counterfactual_engine_and_utilization_differences_are_visible():
    from flexmoe.analysis.cost import analyze_feasibility

    traces, replays = trace_replays()
    result = analyze_feasibility(
        traces,
        replays,
        samples(),
        timing("r", 1, kv=100),
        timing(
            "k",
            2 / 3,
            kv=300,
            source_kind="kv-oracle-diagnostic",
            engine_mode="eager",
            utilization=1.0,
        ),
    )
    assert result["status"] == "candidate"
    assert "kv-reference-engine-mode-differs" in result["assumptions"]
    assert "kv-reference-engine-policy-differs" in result["assumptions"]
    assert "kv-reference-utilization-differs" in result["assumptions"]
    assert "kv-reference-is-diagnostic-counterfactual" in result["assumptions"]


@pytest.mark.parametrize(
    "kind",
    ["mixed-mode", "mixed-contention", "mixed-contract", "mixed-replay-config"],
)
def test_ambiguous_sample_or_replay_grouping_is_rejected(kind):
    from flexmoe.analysis.cost import analyze_feasibility

    traces, replays = trace_replays()
    rows = list(samples())
    if kind == "mixed-mode":
        rows[0] = replace(rows[0], mode="contiguous")
    elif kind == "mixed-contention":
        rows[0] = replace(rows[0], contention="gemm-nccl-proxy")
    elif kind == "mixed-contract":
        changed = dict(rows[0].contract)
        changed["measurement_id"] = "transport-002"
        rows[0] = replace(rows[0], contract=changed)
    else:
        from flexmoe.analysis.replay import replay_trace
        from flexmoe.analysis.schema import ReplayConfig

        replays = list(replays)
        replays[0] = replay_trace(traces[0], ReplayConfig(((0, 1, 2, 3),), 0, 2, "lru"))
    with pytest.raises(ValueError):
        analyze_feasibility(
            traces, replays, rows, timing("r", 1, kv=100), timing("k", 2 / 3, kv=300)
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "trace-workload",
        "trace-model",
        "sample-sha",
        "sample-version",
        "sample-hardware",
        "timing-gpu-total",
        "baseline-eager",
    ],
)
def test_identity_or_workload_mismatch_never_claims_gain(mutation):
    from flexmoe.analysis.cost import analyze_feasibility

    traces, replays = trace_replays()
    rows = list(samples())
    baseline = timing("r", 1, kv=100)
    reference = timing("k", 2 / 3, kv=300)
    if mutation == "trace-workload":
        traces[0].contract["batch_size"] = 11
    elif mutation == "trace-model":
        traces[0].contract["model_identity_sha256"] = "0" * 64
    elif mutation == "sample-sha":
        rows[0].contract["commit"] = "0" * 40
    elif mutation == "sample-version":
        rows[0].contract["versions"] = dict(rows[0].contract["versions"])
        rows[0].contract["versions"]["cuda"] = "12.9"
    elif mutation == "sample-hardware":
        rows[0].contract["hardware_sha256"] = "0" * 64
    elif mutation == "timing-gpu-total":
        reference = replace(reference, gpu_total_bytes_by_rank=(999, 1000, 1000, 1000))
    else:
        baseline = replace(baseline, engine_mode="eager")
    try:
        result = analyze_feasibility(traces, replays, rows, baseline, reference)
    except ValueError:
        return
    assert result["status"] == "incomplete"
    assert result["predictions"] is None


def test_standalone_transport_samples_can_be_reused_for_another_workload():
    from flexmoe.analysis.cost import analyze_feasibility

    traces, replays = trace_replays()
    for trace_row in traces:
        trace_row.contract["input_sha256"] = "d" * 64
        trace_row.contract["prompt_hashes"] = ("e" * 64, "f" * 64)
    for replay in replays:
        object.__setattr__(replay, "input_sha256", "d" * 64)
    baseline = timing("r", 1, kv=100)
    reference = timing("k", 2 / 3, kv=300)
    baseline.contract["input_sha256"] = "d" * 64
    baseline.contract["prompt_hashes"] = ("e" * 64, "f" * 64)
    reference.contract["input_sha256"] = "d" * 64
    reference.contract["prompt_hashes"] = ("e" * 64, "f" * 64)
    result = analyze_feasibility(traces, replays, samples(), baseline, reference)
    assert result["status"] == "candidate"
