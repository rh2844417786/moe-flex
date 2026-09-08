"""Evidence-bounded transfer-cost and KV-headroom feasibility model."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any, cast

from .replay import replay_trace
from .schema import (
    DemandTrace,
    ReplayResult,
    TimingPoint,
    TransferSample,
    diagnostic_artifact,
)

_RANKS = (0, 1, 2, 3)
_TIMING_COUNTERFACTUAL_FIELDS = frozenset(
    {
        "engine_policy_sha256",
        "gpu_memory_utilization",
        "hardware_unavailable_reason",
        "run_id",
        "role",
    }
)
_EVIDENCE_PAIR_FIELDS = (
    "model_identity_sha256",
    "model_config_sha256",
    "dataset_sha256",
    "dataset_manifest_sha256",
    "input_sha256",
    "commit",
    "tensor_parallel_size",
    "batch_size",
    "context_length",
    "output_length",
    "max_num_seqs",
    "max_num_batched_tokens",
    "seed",
    "dtype",
    "sampling_policy",
    "source_request_count",
    "unique_selected_request_count",
    "repeated_request_count",
    "versions",
    "hardware_sha256",
    "prompt_hashes",
)
_TRANSPORT_TRACE_PAIR_FIELDS = (
    "commit",
    "tensor_parallel_size",
    "versions",
    "hardware_sha256",
)


def _ranked(values: Sequence[Any], name: str) -> dict[int, Any]:
    result: dict[int, Any] = {}
    for value in values:
        rank = value.rank
        if rank in result:
            raise ValueError(f"{name} has duplicate rank {rank}")
        result[rank] = value
    if tuple(sorted(result)) != _RANKS:
        raise ValueError(f"{name} must cover ranks 0, 1, 2, 3")
    return result


def _require_equal_contracts(
    contracts: Sequence[Mapping[str, object]], name: str
) -> Mapping[str, object]:
    if not contracts:
        raise ValueError(f"{name} contracts must not be empty")
    anchor = contracts[0]
    if any(contract != anchor for contract in contracts[1:]):
        raise ValueError(f"{name} mixes measurement contracts")
    return anchor


def _validate_trace_group(traces: Mapping[int, DemandTrace]) -> Mapping[str, object]:
    contract = _require_equal_contracts(
        [traces[rank].contract for rank in _RANKS], "traces"
    )
    anchor = traces[0]
    geometry = (
        anchor.total_layers,
        anchor.num_experts,
        anchor.top_k,
        anchor.expert_bytes,
        anchor.observed_steps,
        anchor.captured_steps,
        anchor.full_workload,
        anchor.generated_tokens,
    )
    event_shape = tuple(
        (event.step, event.layer, event.token_rows, event.requests, event.phase)
        for event in anchor.events
    )
    for rank in _RANKS[1:]:
        trace = traces[rank]
        if (
            trace.total_layers,
            trace.num_experts,
            trace.top_k,
            trace.expert_bytes,
            trace.observed_steps,
            trace.captured_steps,
            trace.full_workload,
            trace.generated_tokens,
        ) != geometry:
            raise ValueError("trace geometry or coverage differs across ranks")
        if (
            tuple(
                (event.step, event.layer, event.token_rows, event.requests, event.phase)
                for event in trace.events
            )
            != event_shape
        ):
            raise ValueError("trace event coverage differs across ranks")
    return contract


def _validate_replays(
    traces: Mapping[int, DemandTrace], replays: Mapping[int, ReplayResult]
) -> dict[int, ReplayResult]:
    config = replays[0].config
    if any(replays[rank].config != config for rank in _RANKS[1:]):
        raise ValueError("replays mix cache/residency configurations")
    verified: dict[int, ReplayResult] = {}
    for rank in _RANKS:
        trace = traces[rank]
        replay = replays[rank]
        canonical = replay_trace(trace, replay.config)
        saved_fields = replay.to_dict()
        canonical_fields = canonical.to_dict()
        if saved_fields != canonical_fields:
            differing = next(
                name
                for name in canonical_fields
                if saved_fields.get(name) != canonical_fields[name]
            )
            raise ValueError(
                f"rank {rank} {differing} differs from canonical replay"
            )
        verified[rank] = canonical
    return verified


def _validate_baseline(
    trace_contract: Mapping[str, object], baseline: TimingPoint
) -> tuple[list[str], list[str]]:
    if any(
        trace_contract.get(name) != baseline.contract.get(name)
        for name in _EVIDENCE_PAIR_FIELDS
    ):
        raise ValueError("baseline contract differs from trace workload/identity")
    missing: list[str] = []
    assumptions: list[str] = []
    if baseline.source_kind != "native-measured":
        missing.append("native-measured-baseline")
    if baseline.engine_mode != "native":
        missing.append("native-engine-baseline")
    if (
        trace_contract["engine_policy_sha256"]
        != baseline.contract["engine_policy_sha256"]
    ):
        assumptions.append("trace-engine-policy-differs")
    if (
        trace_contract["gpu_memory_utilization"]
        != baseline.contract["gpu_memory_utilization"]
    ):
        assumptions.append("trace-utilization-differs")
    return missing, assumptions


def _validate_timing_pair(baseline: TimingPoint, reference: TimingPoint) -> None:
    fields = (
        set(baseline.contract) | set(reference.contract)
    ) - _TIMING_COUNTERFACTUAL_FIELDS
    for name in sorted(fields):
        if baseline.contract.get(name) != reference.contract.get(name):
            raise ValueError(f"timing reference {name} differs")
    if baseline.generated_tokens != reference.generated_tokens:
        raise ValueError("timing reference generated token count differs")
    if baseline.gpu_total_bytes_by_rank != reference.gpu_total_bytes_by_rank:
        raise ValueError("timing reference physical GPU totals differ")


def _decompose_exact(total: int, sizes: tuple[int, ...]) -> tuple[int, ...] | None:
    if total == 0:
        return ()
    ordered = tuple(sorted(set(sizes), reverse=True))
    reachable = [False] * (total + 1)
    reachable[0] = True
    for amount in range(1, total + 1):
        reachable[amount] = any(
            size <= amount and reachable[amount - size] for size in ordered
        )
    if not reachable[total]:
        return None
    chunks: list[int] = []
    remaining = total
    while remaining:
        for size in ordered:
            if size <= remaining and reachable[remaining - size]:
                chunks.append(size)
                remaining -= size
                break
        else:
            raise AssertionError("reachable decomposition lost during reconstruction")
    return tuple(chunks)


def _sample_medians(
    rows: Sequence[TransferSample], rank: int
) -> dict[int, dict[str, float]]:
    grouped: dict[int, list[TransferSample]] = defaultdict(list)
    for row in rows:
        if row.rank == rank:
            grouped[row.experts_per_batch].append(row)
    result: dict[int, dict[str, float]] = {}
    for size, repetitions in grouped.items():
        repetition_ids = [row.repetition for row in repetitions]
        if len(repetition_ids) != len(set(repetition_ids)):
            raise ValueError("transfer sample repetitions must be distinct per shape")
        result[size] = {
            "wall_s": median(row.wall_s for row in repetitions),
            "copy_s": median(row.copy_s for row in repetitions),
            "gather_s": median(row.gather_s for row in repetitions),
        }
    return result


def _transport_model(
    traces: Mapping[int, DemandTrace],
    replays: Mapping[int, ReplayResult],
    samples: Sequence[TransferSample],
) -> tuple[dict[str, object], list[str]]:
    missing: list[str] = []
    if not samples:
        return {
            "mode": None,
            "contention": None,
            "coverage_complete": False,
            "per_rank": [],
            "bottleneck_rank": None,
            "optimistic_resource_service_s": None,
            "serial_layer_barrier_service_s": None,
        }, ["four-rank-transfer-samples"]
    modes = {row.mode for row in samples}
    contentions = {row.contention for row in samples}
    if len(modes) != 1 or len(contentions) != 1:
        raise ValueError("one analysis call requires one transfer mode/contention")
    sample_contract = _require_equal_contracts(
        [row.contract for row in samples], "transfer samples"
    )
    if any(
        sample_contract.get(name) != traces[0].contract.get(name)
        for name in _TRANSPORT_TRACE_PAIR_FIELDS
    ):
        raise ValueError(
            "transfer sample contract differs from trace workload/identity"
        )
    if {row.rank for row in samples} != set(_RANKS):
        missing.append("four-rank-transfer-samples")
    per_rank: list[dict[str, object]] = []
    event_services: list[list[float] | None] = []
    for rank in _RANKS:
        replay = replays[rank]
        if any(
            row.expert_bytes != traces[rank].expert_bytes
            for row in samples
            if row.rank == rank
        ):
            raise ValueError("transfer sample expert_bytes differs from rank trace")
        medians = _sample_medians(samples, rank)
        sizes = tuple(size for size in medians if size <= replay.config.staging_experts)
        chunks_by_event: list[list[int] | None] = []
        services: list[float] = []
        copy_count = 0
        for count in replay.event_load_counts:
            chunks = _decompose_exact(count, sizes)
            chunks_by_event.append(None if chunks is None else list(chunks))
            if chunks is None:
                continue
            copy_count += len(chunks)
            services.append(
                sum((medians[size]["wall_s"] for size in chunks), 0.0)
            )
        complete = all(chunks is not None for chunks in chunks_by_event)
        if not complete:
            missing.append("unmeasured-exact-transfer-shape")
            event_services.append(None)
            total_service: float | None = None
        else:
            event_services.append(services)
            total_service = sum(services, 0.0)
        bandwidth = (
            replay.loaded_bytes / total_service
            if total_service is not None and total_service > 0.0
            else None
        )
        per_rank.append(
            {
                "rank": rank,
                "loaded_bytes": replay.loaded_bytes,
                "event_service_s": services if complete else None,
                "chunks_by_event": chunks_by_event,
                "copy_count": copy_count if complete else None,
                "total_service_s": total_service,
                "effective_bandwidth_bytes_s": bandwidth,
            }
        )
    coverage = all(row is not None for row in event_services)
    if coverage:
        aligned = cast(list[list[float]], event_services)
        event_count = len(aligned[0])
        if any(len(row) != event_count for row in aligned):
            raise ValueError("rank replay event counts are not aligned")
        serial_service = sum(
            (max(row[index] for row in aligned) for index in range(event_count)),
            0.0,
        )
        totals = [cast(float, row["total_service_s"]) for row in per_rank]
        optimistic_service = max(totals)
        bottleneck_rank = max(_RANKS, key=lambda rank: (totals[rank], -rank))
    else:
        serial_service = None
        optimistic_service = None
        bottleneck_rank = None
    return {
        "mode": next(iter(modes)),
        "contention": next(iter(contentions)),
        "coverage_complete": coverage,
        "per_rank": per_rank,
        "bottleneck_rank": bottleneck_rank,
        "optimistic_resource_service_s": optimistic_service,
        "serial_layer_barrier_service_s": serial_service,
        "service_metric": "median-measured-wall-s-per-exact-chunk",
        "chunk_rule": "largest-first-exact-decomposition-with-backtracking",
    }, sorted(set(missing))


def analyze_feasibility(
    traces: Sequence[DemandTrace],
    replays: Sequence[ReplayResult],
    samples: Sequence[TransferSample],
    baseline: TimingPoint,
    kv_reference: TimingPoint | None,
) -> dict[str, object]:
    """Estimate diagnostic-only headroom without claiming deployment gain."""

    trace_by_rank = _ranked(traces, "traces")
    saved_replay_by_rank = _ranked(replays, "replays")
    trace_contract = _validate_trace_group(trace_by_rank)
    replay_by_rank = _validate_replays(trace_by_rank, saved_replay_by_rank)
    if not isinstance(baseline, TimingPoint):
        raise TypeError("baseline must be a TimingPoint")
    missing, baseline_assumptions = _validate_baseline(trace_contract, baseline)
    transport, transport_missing = _transport_model(
        trace_by_rank, replay_by_rank, samples
    )
    missing.extend(transport_missing)
    if any(sample.contention != "isolated" for sample in samples):
        # Joint proxy wall already includes GEMM/NCCL. It is not an additive
        # transfer tax on the independently measured native K compute time.
        missing.append("incremental-transfer-calibration")
    assumptions = [
        "diagnostic-model-not-deployment-measurement",
        "serial-layer-barrier-is-a-modelling-scenario-not-a-hardware-bound",
        "full-overlap-is-optimistic-and-not-a-hardware-bound",
        "no-rank-bandwidth-aggregation",
        *baseline_assumptions,
    ]
    if any(
        not trace.full_workload
        or trace.generated_tokens != baseline.generated_tokens
        or replay_by_rank[trace.rank].bytes_per_generated_token is None
        for trace in traces
    ):
        missing.append("full-workload-four-rank-trace")
    if any(trace.contract.get("synthetic") is True for trace in traces):
        assumptions.append("synthetic-workload")
    if any(replay.staging_overflow_events for replay in replays):
        missing.append("staging-overflow-compute-cost")

    hardware_values = {
        trace.contract["hardware_sha256"] for trace in trace_by_rank.values()
    }
    hardware_values.add(baseline.contract["hardware_sha256"])
    hardware_values.update(row.contract["hardware_sha256"] for row in samples)
    if None in hardware_values:
        missing.append("hardware-identity-unavailable")
    if len(hardware_values - {None}) > 1:
        raise ValueError("hardware identity differs across evidence")

    baseline_time = median(baseline.elapsed_s)
    baseline_tps = baseline.generated_tokens / baseline_time
    bandwidth_requirements = [
        replay_by_rank[rank].loaded_bytes / baseline_time for rank in _RANKS
    ]
    actual_kv_increment: list[int] | None = None
    headroom: float | None = None
    predictions: dict[str, object] | None = None
    reference_summary: dict[str, object] | None = None
    candidate_class: str | None = None
    required_overlap: float | None = None
    if kv_reference is None:
        missing.append("matching-kv-reference")
    else:
        if not isinstance(kv_reference, TimingPoint):
            raise TypeError("kv_reference must be a TimingPoint or None")
        _validate_timing_pair(baseline, kv_reference)
        if kv_reference.contract["hardware_sha256"] is None:
            missing.append("hardware-identity-unavailable")
        if baseline.engine_mode != kv_reference.engine_mode:
            assumptions.append("kv-reference-engine-mode-differs")
        if (
            baseline.contract["engine_policy_sha256"]
            != kv_reference.contract["engine_policy_sha256"]
        ):
            assumptions.append("kv-reference-engine-policy-differs")
        if (
            baseline.contract["gpu_memory_utilization"]
            != kv_reference.contract["gpu_memory_utilization"]
        ):
            assumptions.append("kv-reference-utilization-differs")
        if kv_reference.source_kind == "kv-oracle-diagnostic":
            assumptions.append("kv-reference-is-diagnostic-counterfactual")
        reference_time = median(kv_reference.elapsed_s)
        headroom = baseline_time - reference_time
        actual_kv_increment = [
            kv_reference.kv_bytes_by_rank[rank] - baseline.kv_bytes_by_rank[rank]
            for rank in _RANKS
        ]
        if any(value <= 0 for value in actual_kv_increment):
            missing.append("positive-actual-kv-increment")
        if any(
            actual_kv_increment[rank] > replay_by_rank[rank].net_freed_bytes
            for rank in _RANKS
        ):
            missing.append("kv-increment-exceeds-net-freed")
        reference_summary = {
            "point_id": kv_reference.point_id,
            "median_elapsed_s": reference_time,
            "throughput_tps": kv_reference.generated_tokens / reference_time,
            "source_kind": kv_reference.source_kind,
            "engine_mode": kv_reference.engine_mode,
            "gpu_memory_utilization": kv_reference.contract["gpu_memory_utilization"],
            "engine_policy_sha256": kv_reference.contract["engine_policy_sha256"],
        }

    missing = sorted(set(missing))
    serial_service = transport["serial_layer_barrier_service_s"]
    optimistic_service = transport["optimistic_resource_service_s"]
    if (
        not missing
        and kv_reference is not None
        and headroom is not None
        and isinstance(serial_service, float)
        and isinstance(optimistic_service, float)
    ):
        reference_time = median(kv_reference.elapsed_s)
        serial_time = reference_time + serial_service
        optimistic_time = max(reference_time, optimistic_service)
        required_overlap = max(0.0, serial_time - baseline_time)
        serial_tps = baseline.generated_tokens / serial_time
        optimistic_tps = baseline.generated_tokens / optimistic_time
        predictions = {
            "serial_layer_barrier": {
                "elapsed_s": serial_time,
                "throughput_tps": serial_tps,
            },
            "optimistic_full_overlap": {
                "elapsed_s": optimistic_time,
                "throughput_tps": optimistic_tps,
            },
            "throughput_range_tps": [
                serial_tps,
                optimistic_tps,
            ],
        }
        if serial_tps > baseline_tps:
            candidate_class = "serial-profitable"
            status = "candidate"
        elif optimistic_tps > baseline_tps:
            candidate_class = "overlap-required"
            status = "candidate"
        else:
            candidate_class = "neither-profitable"
            status = "model-no-headroom"
    else:
        status = "incomplete"

    return diagnostic_artifact(
        "feasibility-analysis",
        {
            "status": status,
            "candidate_class": candidate_class,
            "baseline": {
                "point_id": baseline.point_id,
                "median_elapsed_s": baseline_time,
                "throughput_tps": baseline_tps,
                "source_kind": baseline.source_kind,
                "engine_mode": baseline.engine_mode,
            },
            "kv_reference": reference_summary,
            "actual_kv_increment_bytes_by_rank": actual_kv_increment,
            "net_freed_bytes_by_rank": [
                replay_by_rank[rank].net_freed_bytes for rank in _RANKS
            ],
            "time_headroom_s": headroom,
            "required_overlap_to_match_baseline_s": required_overlap,
            "transport": transport,
            "bandwidth_requirements_bytes_s": bandwidth_requirements,
            "bandwidth_to_fit_headroom_bytes_s_by_rank": (
                [replay_by_rank[rank].loaded_bytes / headroom for rank in _RANKS]
                if headroom is not None and headroom > 0.0
                else None
            ),
            "predictions": predictions,
            "assumptions": sorted(set(assumptions)),
            "missing_evidence": missing,
        },
    )


__all__ = ["analyze_feasibility"]
