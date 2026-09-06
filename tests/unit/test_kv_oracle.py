from copy import deepcopy
from pathlib import Path

import pytest

from flexmoe.bench.partial_runner import PartialRunConfig, engine_arguments


def config():
    return PartialRunConfig(
        "resident", Path("/model"), Path("/data"), Path("/manifest")
    )


def fixture(role="r0", seconds=100.0, reps=3):
    kv = {"r0": 1000, "small": 1200, "large": 1400}[role]
    contract = {
        name: "a" * 64
        for name in (
            "model_identity_sha256",
            "model_config_sha256",
            "input_sha256",
            "dataset_sha256",
            "dataset_manifest_sha256",
            "engine_policy_sha256",
        )
    }
    contract.update(
        commit="b" * 40,
        batch_size=4,
        context_length=4096,
        output_length=10,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.6,
        seed=20260905,
        warmups=1,
        timing_samples=0,
        max_num_seqs=512,
        max_num_batched_tokens=8192,
        repetitions_requested=reps,
        smoke_output_length=8,
        source_request_count=4,
        unique_selected_request_count=4,
        repeated_request_count=0,
        dtype="bfloat16",
        sampling_policy="existing-prompts",
        comparison_backend="kv-oracle",
        versions={
            "torch": "2.8.0",
            "vllm": "0.10.2",
            "cuda": "12.8",
            "vllm_commit": "c" * 40,
        },
    )
    memory = [
        {
            "rank": i,
            "total_gpu_bytes": 10000,
            "free_gpu_bytes": 8000 - kv,
            "torch_allocated_bytes": 1000 + kv,
            "torch_reserved_bytes": 1500 + kv,
            "torch_peak_allocated_bytes": 1200 + kv,
            "torch_peak_reserved_bytes": 1600 + kv,
            "available_kv_cache_bytes": 1200,
            "model_memory_bytes": 1000,
            "kv_cache_allocated_bytes": kv,
            "kv_cache_declared_bytes": kv,
            "kv_cache_accounting_consistent": True,
            "num_gpu_blocks": kv // 100,
        }
        for i in range(4)
    ]
    probes = [
        {
            "rank": i,
            "forward_calls": 100 if role == "r0" else 70,
            "forward_telemetry_available": True,
            "request_telemetry_available": True,
            "request_observations": 100 if role == "r0" else 70,
            "request_sum": 200 if role == "r0" else 210,
            "request_peak": 4,
            "request_mean": 2 if role == "r0" else 3,
            "parameter_count": 10,
            "all_parameters_cuda": True,
            "kv_occupancy": None,
            "preemptions": None,
        }
        for i in range(4)
    ]
    return {
        "schema_version": 1,
        "run_id": role,
        "role": role,
        "arm": "resident",
        "status": "complete",
        "comparison_backend": "kv-oracle",
        "storage_backend": "native",
        "diagnostic_only": True,
        "formal_offload_gain": False,
        "offload_count": 0,
        "offload_layers": [],
        "contract": contract,
        "requested_kv_cache_bytes": None if role == "r0" else kv,
        "baseline_kv_bytes": None if role == "r0" else 1000,
        "small_add_bytes": 200,
        "large_add_bytes": 400,
        "safety_fraction": 0.9,
        "safety_margin_bytes": 100,
        "memory": deepcopy(memory),
        "final_memory": deepcopy(memory),
        "native_probe": deepcopy(probes),
        "repetitions_completed": reps,
        "smoke": {
            "input_sha256": "d" * 64,
            "output_sha256": "e" * 64,
            "request_count": 1,
            "generated_tokens": 8,
            "elapsed_s": 1.0,
            "output_tokens_per_second": 8.0,
        },
        "repetitions": [
            {
                "repetition": i,
                "request_count": 4,
                "generated_tokens": 40,
                "elapsed_s": seconds,
                "output_tokens_per_second": 40 / seconds,
                "output_sha256": "f" * 64,
                "memory": deepcopy(memory),
                "native_probe": deepcopy(probes),
            }
            for i in range(reps)
        ],
    }


def test_budget_uses_completed_actual_kv_and_keeps_native_guard():
    from flexmoe.bench.kv_oracle_runner import OracleBackend

    with pytest.raises(ValueError):
        engine_arguments(config(), 1000)
    backend = OracleBackend("small", fixture(), 200, 400, 0.9, 100)
    args = backend.engine_arguments(config(), None)
    assert args["kv_cache_memory_bytes"] == 1200  # 1000 allocated + 200
    assert args["cpu_offload_gb"] == 0
    assert args["enforce_eager"] is True


@pytest.mark.parametrize(
    "change",
    [
        {"role": "small"},
        {"status": "failed"},
        {"repetitions_completed": 0},
        {"diagnostic_only": False},
    ],
)
def test_ineligible_anchor_rejected(change):
    from flexmoe.bench.kv_oracle_runner import OracleBackend

    row = fixture()
    row.update(change)
    with pytest.raises(ValueError):
        OracleBackend("small", row, 200, 400, 0.9, 100)


@pytest.mark.parametrize(
    "small,large,fraction,margin",
    [
        (-1, 400, 0.9, 100),
        (200, 100, 0.9, 100),
        (200, 400, float("nan"), 100),
        (200, 400, 1.1, 100),
        (200, 400, 0.9, -1),
        (200, 400, 0.1, 100),
    ],
)
def test_invalid_or_unsafe_budget_rejected(small, large, fraction, margin):
    from flexmoe.bench.kv_oracle_runner import OracleBackend

    with pytest.raises(ValueError):
        OracleBackend("small", fixture(), small, large, fraction, margin)


def test_diagnostic_analysis_and_formal_rejection():
    from flexmoe.bench.kv_oracle_evidence import analyze
    from flexmoe.bench.partial_suite import analyze_triplet

    rows = [fixture(), fixture("small", 80), fixture("large", 70)]
    result = analyze(rows)
    assert result["status"] == "confirmed-signal"
    assert result["small"]["throughput_ratio"] == pytest.approx(1.25)
    assert result["small"]["time_savings_s"] == [20, 20, 20]
    assert result["formal_offload_gain"] is False
    assert result["decision"] == "consider-small-offload"
    assert analyze_triplet(*rows)["status"] == "invalid-comparison"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["contract"].update(input_sha256="0" * 64),
        lambda r: r["contract"].update(commit="0" * 40),
        lambda r: r["repetitions"][0].update(generated_tokens=39),
        lambda r: r["repetitions"][0].update(output_tokens_per_second=99),
        lambda r: r["repetitions"][0].update(elapsed_s=float("nan")),
        lambda r: r["repetitions"][0]["memory"].pop(),
        lambda r: r["repetitions"][0]["native_probe"].pop(),
        lambda r: r["memory"][0].update(kv_cache_allocated_bytes=1300),
        lambda r: r["repetitions"][0]["memory"][0].update(
            torch_peak_reserved_bytes=9900
        ),
        lambda r: r["repetitions"][0]["native_probe"][0].pop("forward_calls"),
        lambda r: r.update(status="timeout"),
    ],
)
def test_invalid_evidence_never_confirms(mutate):
    from flexmoe.bench.kv_oracle_evidence import analyze

    rows = [fixture(), fixture("small", 80), fixture("large", 70)]
    mutate(rows[1])
    assert analyze(rows)["status"] != "confirmed-signal"


def test_reverse_anchor_capacity_not_speed_or_repetition_count():
    from flexmoe.bench.kv_oracle_evidence import validate_anchor

    validate_anchor(fixture(reps=1), fixture(reps=3))
    bad = fixture()
    bad["memory"][0]["kv_cache_allocated_bytes"] = 900
    with pytest.raises(ValueError):
        validate_anchor(bad, fixture())


def test_formal_sanitizer_cannot_strip_oracle_markers():
    from flexmoe.bench.partial_suite import public_run

    with pytest.raises(ValueError):
        public_run(fixture())


def test_target_mismatch_rejects_before_model_smoke():
    from flexmoe.bench.kv_oracle_runner import OracleBackend

    class NoCalls:
        def collective_rpc(self, *args, **kwargs):
            pytest.fail("target must be checked before probe or generate")

    backend = OracleBackend("small", fixture(), 200, 400, 0.9, 100)
    row = fixture("small")
    row["memory"] = fixture()["memory"]  # engine ignored explicit request
    row["hardware_budget"] = {"requested_gpu_bytes_by_rank": [6000] * 4}
    with pytest.raises(ValueError):
        backend.initialized(row, NoCalls())


def test_sanitized_roundtrip_revalidates_and_drops_private_fields():
    import json

    from flexmoe.bench.kv_oracle_evidence import public_run, validate_run

    row = fixture()
    row.update(private_path="/secret/weights", prompts=["private text"])
    row["memory"][0]["hostname"] = "private-host"
    clean = public_run(row)
    assert validate_run(clean) == 1000
    assert "private" not in json.dumps(clean)


def test_unknown_mechanism_and_overlap_do_not_authorize_offloading():
    from flexmoe.bench.kv_oracle_evidence import analyze

    rows = [fixture(), fixture("small", 80), fixture("large", 70)]
    for row in rows:
        for probes in [
            row["native_probe"],
            *(r["native_probe"] for r in row["repetitions"]),
        ]:
            for p in probes:
                p.update(
                    forward_telemetry_available=False,
                    forward_calls=None,
                    request_telemetry_available=False,
                    request_observations=0,
                    request_sum=None,
                    request_mean=None,
                    request_peak=None,
                )
    assert analyze(rows)["decision"] != "consider-small-offload"
    rows = [fixture(), fixture("small", 80), fixture("large", 70)]
    rows[1]["repetitions"][0].update(elapsed_s=101, output_tokens_per_second=40 / 101)
    assert analyze(rows)["status"] == "screening-only"


def test_safety_estimate_includes_current_non_torch_at_torch_peak():
    from flexmoe.bench.kv_oracle_evidence import validate_memory
    from flexmoe.bench.kv_oracle_runner import OracleBackend

    with pytest.raises(ValueError):
        OracleBackend("small", fixture(), 200, 400, 0.33, 100)
    with pytest.raises(ValueError):
        validate_memory(fixture("small")["memory"], 0.33, 100)


def test_failed_large_retains_valid_small_pair_without_decision():
    from flexmoe.bench.kv_oracle_evidence import analyze

    failed = fixture("large", 70)
    failed.update(status="timeout")
    result = analyze([fixture(), fixture("small", 80), failed])
    assert result["status"] == "incomplete"
    assert result["small"]["time_savings_s"] == [20, 20, 20]
    assert result["small"]["throughput_ratio"] == 1.25
    assert result["decision"] == "insufficient-evidence"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(repetitions_completed=3.0),
        lambda r: r.update(offload_count=False),
        lambda r: r["repetitions"][0].update(repetition=False),
        lambda r: r["contract"].update(source_request_count=0),
    ],
)
def test_malformed_types_and_input_accounting_rejected(mutate):
    from flexmoe.bench.kv_oracle_evidence import validate_run

    row = fixture()
    mutate(row)
    with pytest.raises(ValueError):
        validate_run(row)


def test_preload_safety_refusal_is_saved_and_sanitized(tmp_path):
    import json

    from flexmoe.bench.kv_oracle_evidence import public_run
    from flexmoe.bench.kv_oracle_runner import run_point

    path = tmp_path / "unsafe"
    with pytest.raises(ValueError):
        run_point(
            config(),
            project_root=tmp_path,
            run_dir=path,
            role="small",
            reference=fixture(),
            small_add_bytes=200,
            large_add_bytes=400,
            safety_fraction=0.33,
            safety_margin_bytes=100,
        )
    row = public_run(json.loads((path / "summary.json").read_text()))
    assert row["status"] == "failed"
    assert row["failure_code"] == "safety-refusal"
    assert row["failure_stage"] == "budget-construction"
    assert row["safety_fraction"] == 0.33
    assert row["small_add_bytes"] == 200
    assert row["formal_offload_gain"] is False


def test_valid_but_changed_fresh_r0_k0_rejects_reverse():
    from flexmoe.bench.kv_oracle_evidence import validate_anchor

    row = fixture()
    for memory in [
        row["memory"],
        row["final_memory"],
        *(r["memory"] for r in row["repetitions"]),
    ]:
        for rank in memory:
            rank.update(
                kv_cache_allocated_bytes=900,
                kv_cache_declared_bytes=900,
                num_gpu_blocks=9,
            )
    with pytest.raises(ValueError):
        validate_anchor(fixture(reps=1), row)
