"""Literal saved-evidence tests; these fixtures are not hardware measurements."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from flexmoe.analysis import decode_suite as suite


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def evidence(mode="matched-resident", elapsed=10, kv=1000, profile=False):
    labels = {
        "comparison_backend": "decode-mechanism",
        "mode": mode,
        "profile": profile,
        "evidence_kind": "instrumented" if profile else "measured",
        "timing_eligible": not profile,
        "diagnostic_only": True,
        "formal_offload_gain": False,
        "deployment_gain_proven": False,
        "schema_version": 1,
    }
    hashes = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(1000)]
    policy = {
        "model_config": {
            "enforce_eager": mode != "native",
            "dtype": "torch.bfloat16",
            "quantization": None,
        },
        "parallel_config": {
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "disable_custom_all_reduce": mode != "native",
        },
        "cache_config": {
            "cpu_offload_gb": 0,
            "swap_space_bytes": 0,
            "enable_prefix_caching": False,
            "gpu_memory_utilization": 0.9,
        },
        "scheduler_config": {"max_num_seqs": 1024, "max_num_batched_tokens": 8192},
    }
    contract = dict(
        labels,
        commit="a" * 40,
        model_identity_sha256="a" * 64,
        model_config_sha256="b" * 64,
        hardware_sha256="c" * 64,
        dataset_sha256="d" * 64,
        dataset_manifest_sha256="e" * 64,
        input_sha256="f" * 64,
        calibration_input_hashes_sha256=digest(["b" * 64]),
        selected_input_hashes=hashes,
        prompt_hashes=hashes,
        calibration_input_hashes=["b" * 64],
        calibration_count=1,
        no_repetition=True,
        repeated_request_count=0,
        unique_selected_request_count=1000,
        request_count=1000,
        batch_size=1000,
        output_length=1,
        context_length=1024,
        tensor_parallel_size=4,
        dtype="bfloat16",
        seed=1,
        max_num_seqs=1024,
        max_num_batched_tokens=8192,
        repetitions_requested=3,
        warmups=1,
        timing_samples=0,
        smoke_output_length=1,
        selection_offset=0,
        gpu_memory_utilization=0.9,
        physical_safety_reserve_bytes=100,
        target_batch=None,
        capture_steps=256,
        min_capture_steps=64,
        trace_budget_bytes=134217728,
        observation_policy="prepared",
        versions={
            "torch": "2.8.0",
            "vllm": "0.10.2",
            "cuda": "12.8",
            "python": "3.12.1",
            "vllm_commit": "a" * 40,
        },
        expert_cache={
            "resident_ratio": 0.8,
            "cache_slots": 512,
            "cache_policy": "decayed-lfu",
            "calibration_count": 1,
            "timing_samples": 0,
            "profile_sha256": "a" * 64,
        },
        engine_policy_sha256=digest(policy),
    )
    memory = [
        {
            "rank": i,
            "total_gpu_bytes": 80000,
            "free_gpu_bytes": 38000,
            "torch_allocated_bytes": 40000,
            "torch_reserved_bytes": 41000,
            "torch_peak_allocated_bytes": 40500,
            "torch_peak_reserved_bytes": 41000,
            "model_memory_bytes": 39000,
            "available_kv_cache_bytes": kv,
            "kv_cache_allocated_bytes": kv,
            "kv_cache_declared_bytes": kv,
            "kv_cache_accounting_consistent": True,
            "num_gpu_blocks": kv // 100,
        }
        for i in range(4)
    ]
    observations = [
        {
            "rank": i,
            "profile": profile,
            "target_batch": None,
            "coverage_status": "complete" if profile else "not-requested",
            "observed_steps": 256,
            "captured_steps": 256 if profile else 0,
            "phase_step_counts": {"decode": 256},
            "decode_batch_step_counts": {"1000": 256},
        }
        for i in range(4)
    ]
    repetitions = [
        dict(
            labels,
            artifact_kind="decode-repetition",
            repetition=i,
            generation_status="complete",
            elapsed_s=elapsed,
            generated_tokens=1000,
            request_count=1000,
            output_sha256="1" * 64,
            output_tokens_per_second=1000 / elapsed,
            memory=copy.deepcopy(memory),
            capture_status="complete",
            memory_status="measured",
            observer_finalization={"status": "complete", "errors": []},
            worker_observations=copy.deepcopy(observations),
        )
        for i in range(3)
    ]
    contract["expert_cache"]["identity"] = {
        "model_config_sha256": "b" * 64,
        "model_identity_sha256": "9" * 64,
        "tensor_parallel_size": 4,
        "geometry": {
            "total_layers": 2,
            "num_experts": 4,
            "hidden_size": 16,
            "intermediate_size": 2,
        },
    }
    stats = [
        {
            "rank": i,
            "storage_backend": "expert-cache" if mode == "offload" else "native",
            "startup_resident_h2d_bytes": 100 if mode == "offload" else 0,
            "host_source_bytes": 100,
            "resident_routed_bytes": 80,
            "cache_capacity_bytes": 10,
            "unique_demands": 1,
        }
        for i in range(4)
    ]
    if mode == "offload":
        for stat in stats:
            stat.pop("storage_backend")
            stat.update(
                failed=False,
                identity=contract["expert_cache"]["identity"],
                profile_sha256="a" * 64,
                cache_slots=512,
                resident_ratio=0.8,
                cache_policy="decayed-lfu",
                gpu_pool_bytes=100,
                net_freed_bytes=10,
                forward_counts=[1, 1],
                h2d_bytes=0,
            )
    return dict(
        labels,
        artifact_kind="decode-run",
        status="complete",
        generation_status="complete",
        capture_status="complete",
        observer_finalization={"status": "complete", "errors": []},
        contract=contract,
        engine_policy=policy,
        memory=memory,
        final_memory=copy.deepcopy(memory),
        actual_kv={
            "allocated_bytes_per_rank": kv,
            "num_gpu_blocks": kv // 100,
            "requested_bytes": kv,
            "bytes_per_block": 100,
            "rounding_bytes": 0,
        },
        geometry={"total_layers": 2, "num_experts": 4, "top_k": 2, "expert_bytes": 192},
        source_hashes={
            "decode_mechanism_runner.py": "a" * 64,
            "decode_trace.py": "b" * 64,
            "expert_pool.py": "c" * 64,
        },
        expert_cache_stats=stats,
        storage_backend="expert-cache" if mode == "offload" else "native",
        repetitions=repetitions,
        repetitions_completed=3,
        performance_outputs_stable=True,
        smoke=dict(
            labels,
            artifact_kind="decode-smoke",
            request_count=1,
            generated_tokens=1,
            input_sha256="2" * 64,
            output_sha256="3" * 64,
            elapsed_s=0.1,
        ),
    )


def test_literal_measured_triplet_keeps_native_reference_separate():
    a, b, c, n = (
        evidence(),
        evidence("offload", 12),
        evidence("offload", 9, 1400),
        evidence("native", 8),
    )
    result = suite.compare_runs(a, b, c, n)
    assert result["status"] == "measured"
    assert result["offload_overhead_s"] == 2
    assert result["kv_recovery_s"] == 3
    assert result["matched_net_ratio"] == pytest.approx(10 / 9)
    assert result["native_reference_ratio"] == pytest.approx(8 / 9)
    assert result["engine_mode_tax_s"] == 2


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("profile", "instrumented-timing"),
        ("version", "missing-version"),
        ("rank", "memory-evidence"),
        ("rejected", "measurement-rejected"),
        ("observer", "observer-finalization"),
        ("kv", "kv-evidence"),
        ("unreached", "target-unreached"),
        ("tokens", "fixed-output"),
        ("repeat", "input-uniqueness"),
        ("policy", "engine-policy"),
        ("count", "repetitions-incomplete"),
    ],
)
def test_eligibility_fails_closed(defect, reason):
    row = evidence(profile=defect == "profile")
    if defect == "version":
        row["contract"]["versions"]["vllm_commit"] = None
    if defect == "rank":
        row["repetitions"][0]["memory"].pop()
    if defect == "rejected":
        row["repetitions"][0]["measurement_status"] = "rejected"
    if defect == "observer":
        row["repetitions"][0]["observer_finalization"]["status"] = "failed"
    if defect == "kv":
        row["actual_kv"]["allocated_bytes_per_rank"] = 1001
    if defect == "unreached":
        row["contract"]["target_batch"] = 1000
        row["repetitions"][0]["worker_observations"][0]["coverage_status"] = "unreached"
    if defect == "tokens":
        row["repetitions"][0]["generated_tokens"] = 999
    if defect == "repeat":
        row["contract"]["selected_input_hashes"][1] = row["contract"][
            "selected_input_hashes"
        ][0]
    if defect == "policy":
        row["engine_policy"]["model_config"]["enforce_eager"] = False
    if defect == "count":
        row["repetitions"].pop()
    assert reason in suite.validate_run(row)["reasons"]
    assert suite.validate_run(row)["timing_eligible"] is False


@pytest.mark.parametrize(
    "key,value",
    [("cache_slots", 2048), ("resident_ratio", 0.9), ("profile_sha256", "f" * 64)],
)
def test_bc_cache_and_profile_must_be_frozen(key, value):
    c = evidence("offload", 9, 1400)
    c["contract"]["expert_cache"][key] = value
    assert (
        suite.compare_runs(evidence(), evidence("offload", 12), c)["status"]
        == "invalid-comparison"
    )


def test_mode_tax_cannot_be_misreported_as_offload_overhead():
    result = suite.compare_runs(
        evidence("native"), evidence("offload", 12), evidence("offload", 9, 1400)
    )
    assert result["status"] == "invalid-comparison"
    assert "offload_overhead_s" not in result


def test_real_offload_abi_does_not_require_nonexistent_worker_backend_label():
    row = evidence("offload")
    assert "storage_backend" not in row["expert_cache_stats"][0]
    assert suite.validate_run(row)["timing_eligible"] is True
    row["expert_cache_stats"][0]["profile_sha256"] = "e" * 64
    assert "offload-identity" in suite.validate_run(row)["reasons"]


def test_unmatched_native_kv_cannot_be_labeled_engine_mode_tax():
    result = suite.compare_runs(
        evidence(),
        evidence("offload", 12),
        evidence("offload", 9, 1400),
        evidence("native", 8, 1800),
    )
    assert result["status"] == "measured"
    assert "engine_mode_tax_s" not in result
    assert result["engine_mode_tax_status"] == "confounded-kv"


@pytest.mark.parametrize("group", ["cache_config", "requested"])
def test_native_reference_requires_actual_requested_resolved_utilization(group):
    native = evidence("native", 8)
    native["engine_policy"].setdefault(group, {})["gpu_memory_utilization"] = 0.6
    native["contract"]["engine_policy_sha256"] = digest(native["engine_policy"])
    result = suite.compare_runs(
        evidence(), evidence("offload", 12), evidence("offload", 9, 1400), native
    )
    assert result["status"] == "invalid-comparison"
    assert "engine-policy" in result["reasons"]
    assert "native_reference_ratio" not in result


@pytest.mark.parametrize(
    "group,key,value",
    [
        ("scheduler_config", "max_num_partial_prefills", 99),
        ("cache_config", "block_size", 32),
        ("requested", "disable_log_stats", True),
    ],
)
def test_native_policy_confound_keeps_reference_without_engine_only_tax(
    group, key, value
):
    native = evidence("native", 8)
    native["engine_policy"].setdefault(group, {})[key] = value
    native["contract"]["engine_policy_sha256"] = digest(native["engine_policy"])
    result = suite.compare_runs(
        evidence(), evidence("offload", 12), evidence("offload", 9, 1400), native
    )
    assert result["status"] == "measured"
    assert result["native_reference_ratio"] == pytest.approx(8 / 9)
    assert result["native_to_matched_elapsed_delta_s"] == 2
    assert result["engine_mode_tax_status"] == "confounded-policy"
    assert "engine_mode_tax_s" not in result


def test_recorded_native_eager_compilation_differences_remain_supported():
    rows = [
        evidence(),
        evidence("offload", 12),
        evidence("offload", 9, 1400),
        evidence("native", 8),
    ]
    for row in rows:
        native = row["mode"] == "native"
        policy = row["engine_policy"]
        policy["compilation_config"] = {
            "level": 3 if native else 0,
            "cudagraph_mode": "FULL_AND_PIECEWISE" if native else "NONE",
            "custom_ops": "['all']",
        }
        policy["requested"] = {
            "gpu_memory_utilization": 0.9,
            "enforce_eager": not native,
            "disable_custom_all_reduce": not native,
            "disable_log_stats": False,
        }
        if not native:
            policy["requested"]["compilation_config"] = {
                "level": 0,
                "custom_ops": ["all"],
            }
        row["contract"]["engine_policy_sha256"] = digest(policy)
    result = suite.compare_runs(*rows)
    assert result["status"] == "measured"
    assert result["engine_mode_tax_s"] == 2


def test_plan_is_finite_executable_and_requires_actual_kv_input():
    p = suite.build_plan(
        stage="baseline",
        run_id="custom",
        model_path="/mnt/public_data/custom model",
        dataset_path="data/custom.zst",
        dataset_manifest="data/custom.json",
    )
    assert len(p["commands"]) == 3
    assert p["commands"][0]["argv"][:3] == [
        "bash",
        "scripts/server/run_decode_mechanism.sh",
        "calibrate",
    ]
    assert "/mnt/public_data/custom model" in p["commands"][0]["argv"]
    assert "data/custom.zst" in p["commands"][2]["argv"]
    with pytest.raises(ValueError, match="actual KV"):
        suite.build_plan(stage="kv", run_id="kv")
    p = suite.build_plan(stage="kv", run_id="kv", kv_bytes=[1000, 1400])
    assert len(p["commands"]) == 3
    assert [x["argv"][2] for x in p["commands"]] == [
        "matched-resident",
        "offload",
        "offload",
    ]
    assert all("--target-batch" not in x["argv"] for x in p["commands"])
    assert all(
        x["argv"][x["argv"].index("--batch-size") + 1] == "1024" for x in p["commands"]
    )


def test_standard_library_cli_roundtrip(tmp_path):
    run = tmp_path / "raw"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps(evidence()))
    cli = Path(suite.__file__)
    result = subprocess.run(
        [sys.executable, "-S", str(cli), "validate", "--source", str(run)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["timing_eligible"] is True


def test_later_scans_require_selected_batches():
    with pytest.raises(ValueError, match="selected"):
        suite.build_plan(stage="offload", run_id="miss", kv_bytes=[1000])
    with pytest.raises(ValueError, match="selected"):
        suite.build_plan(stage="coverage", run_id="four", context_length=4096)


@pytest.mark.parametrize(
    "kind", ["decode-run", "decode-profile", "decode-smoke", "decode-repetition"]
)
def test_old_formal_consumer_rejects_decode_kind_without_flags(kind):
    from flexmoe.bench.partial_suite import public_run

    with pytest.raises(ValueError, match="diagnostic"):
        public_run({"artifact_kind": kind})


WRAPPER = Path(__file__).parents[2] / "scripts/server/run_decode_mechanism.sh"


def shell(code, *args):
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; shift; ' + code,
            "test",
            str(WRAPPER),
            *map(str, args),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_wrapper_container_recorder_preserves_literal_arguments_and_timeout(tmp_path):
    scripts = tmp_path / "scripts/server"
    scripts.mkdir(parents=True)
    recorder = scripts / "run_container.sh"
    recorder.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    result = shell(
        'decode_root="$1"; shift; decode_mode=offload; decode_parse "$@"; decode_run="$decode_root/runs/decode-mechanism/abc"; decode_run_gpu',
        tmp_path,
        "--run-id",
        "abc",
        "--timeout-s",
        "17",
        "--profile-path",
        "profiles/a space.json",
        "--dataset-path",
        "data/a;$x.zst",
        "--kv-bytes",
        "1234",
    )
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert args.index("timeout") < args.index("python3")
    assert args[args.index("--kill-after=30") + 1] == "17"
    assert "HF_HUB_OFFLINE=1" in args
    assert args[args.index("--dataset-path") + 1] == "data/a;$x.zst"
    assert args[args.index("--profile-path") + 1] == "profiles/a space.json"
    assert args[args.index("--kv-bytes") + 1] == "1234"


@pytest.mark.parametrize(
    "flag",
    [
        "--run-dir",
        "--run",
        "--project-root",
        "--project",
        "--mode",
        "--output",
        "--kv-bytes=1",
        "--unknown",
    ],
)
def test_wrapper_rejects_owned_flags_and_abbreviations(flag):
    result = shell('decode_mode=native; decode_parse "$@"', flag, "escape")
    assert result.returncode == 3


def test_wrapper_fresh_canonical_outputs_and_dirty_scope(tmp_path):
    (tmp_path / "runs/decode-mechanism").mkdir(parents=True)
    result = shell('decode_paths "$1" fresh', tmp_path)
    assert result.returncode == 0, result.stderr
    (tmp_path / "runs/decode-mechanism/fresh").mkdir()
    assert shell('decode_paths "$1" fresh', tmp_path).returncode != 0
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "runs/decode-mechanism/escape").symlink_to(
        outside, target_is_directory=True
    )
    assert shell('decode_paths "$1" escape', tmp_path).returncode != 0
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    results = tmp_path / "docs/results/decode-mechanism-test"
    results.mkdir(parents=True)
    (results / "report.json").write_text("{}")
    # Only the generated prefix can be ignored; any source edit blocks another run.
    (tmp_path / ".gitignore").write_text("runs/\noutside/\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", ".gitignore"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    assert shell('decode_clean "$1"', tmp_path).returncode == 0
    (tmp_path / "changed.py").write_text("x=1")
    assert shell('decode_clean "$1"', tmp_path).returncode != 0
