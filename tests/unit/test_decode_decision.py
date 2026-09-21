"""The host decision gate must not turn a rank-local or simulated gain into evidence."""

import copy

import pytest


def summary(*, mode="offload", kv=1024, misses=0, profile=False):
    return {
        "status": "complete",
        "mode": mode,
        "profile": profile,
        "repetitions_completed": 3,
        "actual_kv": {
            "allocated_bytes_per_rank": kv,
            "num_gpu_blocks": 32,
            "bytes_per_block": kv // 32,
        },
        "hardware_budget": {
            "total_gpu_bytes_by_rank": [80000] * 4,
            "requested_gpu_bytes_by_rank": [72000] * 4,
        },
        "source_hashes": {"decode_mechanism_runner.py": "same-code"},
        "engine_policy": {"model_config": {"enforce_eager": True}},
        "smoke": {"output_sha256": "same-output"},
        "contract": {
            "commit": "same-commit",
            "engine_policy_sha256": "same-policy",
            "expert_cache": {"profile_sha256": "same-profile"},
            "batch_size": 1,
            "context_length": 1024,
            "output_length": 256,
            "max_num_seqs": 32,
            "max_num_batched_tokens": 8192,
            "input_sha256": "workload",
            "calibration_input_hashes_sha256": "same-calibration",
            "seed": 20260912,
            "gpu_memory_utilization": 0.9,
            "model_identity_sha256": "model",
            "hardware_sha256": "same-h100s",
            "physical_safety_reserve_bytes": 2_000_000_000,
        },
        "repetitions": [
            {
                "repetition": i,
                "measurement_status": "complete",
                "generation_status": "complete",
                "capture_status": "complete",
                "timing_eligible": not profile,
                "output_tokens_per_second": 100.0,
                "elapsed_s": 2.56,
                "ttft_median_s": 0.25,
                "scheduler": {
                    "preemptions": {"status": "measured", "total": 0},
                    "kv_cache_usage": {"status": "measured", "peak": 0.7},
                },
                "diagnostics": {
                    "per_rank": [
                        {
                            "rank": rank,
                            "policy": {"cache_misses": misses},
                            "h2d_bytes": misses * 1024,
                            "copy_launches": 2 if misses else 0,
                        }
                        for rank in range(4)
                    ]
                },
            }
            for i in range(3)
        ],
    }


def test_zero_miss_requires_every_rank_of_every_unprofiled_repetition():
    from flexmoe.analysis.decode_decision import zero_miss_gate

    clean = summary()
    assert zero_miss_gate(clean)["status"] == "proven-zero-miss"
    one_miss = copy.deepcopy(clean)
    one_miss["repetitions"][2]["diagnostics"]["per_rank"][3]["policy"][
        "cache_misses"
    ] = 1
    assert zero_miss_gate(one_miss)["status"] == "not-zero-miss"
    assert zero_miss_gate(summary(profile=True))["status"] == "not-measured"
    incomplete = copy.deepcopy(clean)
    incomplete["repetitions"][0]["diagnostics"]["per_rank"].pop()
    assert zero_miss_gate(incomplete)["status"] == "incomplete-evidence"


def test_capacity_gate_requires_strict_growth_of_actual_allocation():
    from flexmoe.analysis.decode_decision import capacity_gate

    resident, eager, offload = (
        summary(mode="native"),
        summary(mode="matched-resident"),
        summary(kv=1024),
    )
    assert capacity_gate(resident, eager, offload)["status"] == "no-extra-capacity"
    offload["actual_kv"]["allocated_bytes_per_rank"] = 1056
    offload["actual_kv"]["num_gpu_blocks"] = 33
    assert (
        capacity_gate(resident, eager, offload)["status"] == "extra-capacity-measured"
    )
    assert capacity_gate(resident, eager, offload)["extra_bytes_per_rank"] == 32
    offload["actual_kv"]["allocated_bytes_per_rank"] = 1040
    offload["actual_kv"]["num_gpu_blocks"] = 33
    assert capacity_gate(resident, eager, offload)["status"] == "incomplete-evidence"
    offload = summary(kv=1088)
    offload["contract"]["gpu_memory_utilization"] = 0.8
    assert capacity_gate(resident, eager, offload)["status"] == "incomplete-evidence"
    eager = summary(mode="matched-resident", kv=1152)
    offload = summary(kv=1088)
    assert capacity_gate(resident, eager, offload)["status"] == "no-extra-capacity"
    assert capacity_gate(resident, eager, offload)["resident_bytes_per_rank"] == 1152
    eager["source_hashes"]["decode_mechanism_runner.py"] = "changed"
    assert capacity_gate(resident, eager, offload)["status"] == "incomplete-evidence"


def test_pair_budget_is_feasible_even_when_offload_auto_kv_is_smaller():
    from flexmoe.analysis.decode_decision import select_budgets

    native = summary(mode="native", kv=1024)
    eager = summary(mode="matched-resident", kv=1152)
    offload = summary(kv=1088)
    assert select_budgets(native, eager, offload) == {
        "native_kv": 1024,
        "matched_eager_kv": 1088,
        "offload_extra_kv": None,
    }
    offload = summary(kv=1280)
    assert select_budgets(native, eager, offload)["offload_extra_kv"] == 1280


def test_eager_resident_and_offload_pair_rejects_kv_or_workload_drift():
    from flexmoe.analysis.decode_decision import compare_pair

    resident = summary(mode="matched-resident")
    offload = summary()
    assert compare_pair(resident, offload)["status"] == "measured-pair"
    offload["actual_kv"]["allocated_bytes_per_rank"] = 1056
    assert compare_pair(resident, offload)["status"] == "invalid-pair"
    offload = summary()
    offload["contract"]["input_sha256"] = "different"
    assert compare_pair(resident, offload)["status"] == "invalid-pair"
    offload = summary()
    offload["hardware_budget"]["requested_gpu_bytes_by_rank"][2] = 60000
    assert compare_pair(resident, offload)["status"] == "invalid-pair"
    for field, replacement in (
        ("source_hashes", {"decode_mechanism_runner.py": "other"}),
        ("engine_policy", {"model_config": {"enforce_eager": False}}),
        ("smoke", {"output_sha256": "different-output"}),
    ):
        offload = summary()
        offload[field] = replacement
        assert compare_pair(resident, offload)["status"] == "invalid-pair"


def test_kv_gain_ratio_rejects_non_kv_drift():
    from flexmoe.analysis.decode_decision import compare_offload_kv

    small = summary(kv=1024)
    large = summary(kv=1088)
    assert compare_offload_kv(small, large)["status"] == "measured-kv-intervention"
    large["source_hashes"]["decode_mechanism_runner.py"] = "changed"
    assert compare_offload_kv(small, large)["status"] == "invalid-pair"


def test_selected_points_keep_low_concurrency_and_matched_work():
    from flexmoe.analysis.decode_decision import point_arguments

    common = {"profile_path": "runs/decode-decision/id/cal-1024/profile.json"}
    resident = point_arguments(
        "matched-resident", "zero-resident", 1024, 1024, 1, common
    )
    offload = point_arguments(
        "offload", "zero-offload", 1024, 1024, 1, common, resident_ratio="0.9"
    )
    for point in (resident, offload):
        argv = point["argv"]
        assert argv[argv.index("--max-num-seqs") + 1] == "32"
        assert argv[argv.index("--warmups") + 1] == "1"
        assert argv[argv.index("--repetitions") + 1] == "3"
        assert argv[argv.index("--kv-bytes") + 1] == "1024"
        assert argv[argv.index("--batch-size") + 1] == "1"
    assert "--resident-ratio" not in resident["argv"]
    assert offload["argv"][offload["argv"].index("--resident-ratio") + 1] == "0.9"


def test_decision_plan_contains_only_approved_points_and_horizons():
    from flexmoe.analysis.decode_decision import build_decision_plan

    plan = build_decision_plan(
        run_id="r",
        capacities={
            "native_kv": 1_000,
            "eager_kv": 900,
            "offload_kv": 1_200,
            "resident_ceiling": 1_000,
            "k_pair": 900,
            "legacy_kv": 32_815_054_848,
            "kv_block_bytes": 100,
        },
        trace_paths={16: "runs/r-trace-16.json.gz", 32: "runs/r-trace-32.json.gz"},
    )
    labels = [point["label"] for point in plan]

    assert labels == [
        "cal-1024",
        "cal-4096",
        "native-auto",
        "eager-auto",
        "offload-auto",
        "zero-resident",
        "zero-offload",
        "zero-offload-profile",
        "legacy-native",
        "legacy-eager-resident",
        "legacy-offload",
        "mechanism-native-32",
        "mechanism-eager-32",
        "mechanism-offload-32",
        "service-native-k0",
        "service-offload-k0",
        "service-offload-k1",
        "service-eager-kpair",
        "service-offload-kpair",
        "trace-16",
        "trace-32",
        "oracle-16-h0",
        "oracle-16-h1",
        "oracle-16-h2",
        "oracle-16-h0-profile",
        "oracle-16-h1-profile",
        "oracle-16-h2-profile",
        "oracle-32-h0",
        "oracle-32-h1",
        "oracle-32-h2",
        "oracle-32-h0-profile",
        "oracle-32-h1-profile",
        "oracle-32-h2-profile",
        "oracle-forced-omission",
    ]
    allowed_request_counts = {1, 16, 32, 64}
    for point in plan:
        argv = point["argv"]
        if point["status"] == "skipped":
            assert argv == []
            continue
        assert int(argv[argv.index("--batch-size") + 1]) in allowed_request_counts
        if "--prefetch-horizon" in argv:
            assert int(argv[argv.index("--prefetch-horizon") + 1]) in (0, 1, 2)


def test_decision_plan_skips_k1_when_offload_does_not_exceed_resident_ceiling():
    from flexmoe.analysis.decode_decision import build_decision_plan

    plan = build_decision_plan(
        run_id="r",
        capacities={
            "native_kv": 1_000,
            "eager_kv": 1_100,
            "offload_kv": 1_050,
            "resident_ceiling": 1_100,
            "k_pair": 1_050,
            "legacy_kv": 32_815_054_848,
            "kv_block_bytes": 100,
        },
        trace_paths={16: "trace16", 32: "trace32"},
    )
    k1 = [point for point in plan if point["label"] == "service-offload-k1"]

    assert k1 == [
        {
            "label": "service-offload-k1",
            "name": "r-service-offload-k1",
            "mode": "offload",
            "status": "skipped",
            "reason": "offload-kv-does-not-exceed-resident-ceiling",
            "argv": [],
        }
    ]


def test_decision_plan_skips_legacy_points_above_each_mode_capacity():
    from flexmoe.analysis.decode_decision import build_decision_plan

    plan = build_decision_plan(
        run_id="r",
        capacities={
            "native_kv": 40_000_000_000,
            "eager_kv": 16_000_000_000,
            "offload_kv": 40_000_000_000,
            "resident_ceiling": 40_000_000_000,
            "k_pair": 16_000_000_000,
            "legacy_kv": 32_815_054_848,
            "kv_block_bytes": 100,
        },
        trace_paths={16: "trace16", 32: "trace32"},
    )
    legacy = {row["label"]: row for row in plan if row["label"].startswith("legacy-")}

    assert legacy["legacy-native"]["status"] == "planned"
    assert legacy["legacy-eager-resident"]["status"] == "skipped"
    assert (
        legacy["legacy-eager-resident"]["reason"] == "legacy-kv-exceeds-mode-capacity"
    )
    assert legacy["legacy-offload"]["status"] == "planned"


def test_forced_omission_is_derived_from_a_real_nonfirst_layer_route(tmp_path):
    import gzip
    import json

    from flexmoe.analysis.decode_decision import derive_forced_omission

    path = tmp_path / "trace.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump(
            {
                "artifact_kind": "decode-logical-cache-trace",
                "rows": [
                    {
                        "step_id": 4,
                        "layer_id": 0,
                        "actual_expert_ids": [1, 2],
                        "miss_ids": [1],
                    },
                    {
                        "step_id": 4,
                        "layer_id": 1,
                        "actual_expert_ids": [3, 7],
                        "resident_hit_ids": [3],
                        "cache_hit_ids": [],
                        "miss_ids": [7],
                    },
                ],
            },
            stream,
        )

    assert derive_forced_omission(path) == "4:1:7"


def test_oracle_output_gate_requires_all_three_horizons_and_forced_hash_match():
    from flexmoe.analysis.decode_decision import oracle_output_gate

    runs = {
        label: {
            "status": "complete",
            "repetitions_completed": 3,
            "repetitions": [{"output_sha256": "same"}] * 3,
        }
        for label in ("h0", "h1", "h2", "forced")
    }
    assert oracle_output_gate(runs)["status"] == "outputs-match"

    runs["h2"] = {
        **runs["h2"],
        "repetitions": [{"output_sha256": "different"}] * 3,
    }
    assert oracle_output_gate(runs)["status"] == "output-mismatch"


def test_controller_refuses_path_unsafe_run_id():
    from flexmoe.analysis.decode_decision import validate_run_id

    for bad in ("../tmp", "", "a/b", "x..y", "x" * 65):
        with pytest.raises(ValueError):
            validate_run_id(bad)


def test_controller_rejects_wrong_branch_before_build():
    from flexmoe.analysis.decode_decision import validate_branch

    assert validate_branch("repro/fluxmoe") == "repro/fluxmoe"
    with pytest.raises(ValueError, match="repro/fluxmoe"):
        validate_branch("main")


def test_calibration_does_not_pass_wrapper_owned_profile_or_budget_flags():
    from flexmoe.analysis.decode_decision import point_arguments

    point = point_arguments(
        "calibrate", "cal-1024", 1024, None, 32, {"profile_path": "not-used"}
    )
    for forbidden in ("--profile-path", "--kv-bytes", "--safety-reserve-bytes"):
        assert forbidden not in point["argv"]


def test_interrupted_atomic_write_does_not_prevent_resume(tmp_path):
    import json

    from flexmoe.analysis.decode_decision import _save

    path = tmp_path / "state.json"
    (tmp_path / "state.json.new").write_text("old partial content")
    _save(path, {"status": "running"})
    assert json.loads(path.read_text()) == {"status": "running"}
    assert (tmp_path / "state.json.new").read_text() == "old partial content"


def test_failed_suite_publishes_partial_report_and_marks_unreached_points(tmp_path):
    import json
    from pathlib import Path

    from flexmoe.analysis.decode_decision import _publish_report

    source_root = Path(__file__).resolve().parents[2]
    root = tmp_path / "project"
    output = root / "docs/results/decode-decision-failed-run"
    state = {
        "run_id": "failed-run",
        "commit": "a" * 40,
        "status": "failed",
        "suite_error_type": "RuntimeError",
        "points": {
            "failed-run-zero-offload": {
                "label": "zero-offload",
                "mode": "offload",
                "status": "failed",
                "error_type": "RuntimeError",
                "argv": [],
            }
        },
        "point_plan": [
            {
                "label": "oracle-16-h0",
                "name": "failed-run-oracle-16-h0",
                "mode": "offload",
                "status": "planned",
                "argv": [],
            }
        ],
        "numeric": {},
    }

    _publish_report(
        root,
        output,
        state,
        report_script=source_root / "src/flexmoe/analysis/decode_oracle_report.py",
    )

    saved = json.loads((output / "report.json").read_text())
    assert saved["status"] == "failed"
    assert saved["suite_error_type"] == "RuntimeError"
    assert {row["status"] for row in saved["failed_points"]} == {
        "failed",
        "unreached",
    }
    markdown = (output / "report.md").read_text()
    assert "zero-offload" in markdown
    assert "oracle-16-h0" in markdown


def test_report_publisher_rejects_preexisting_symlink(tmp_path):
    from pathlib import Path

    from flexmoe.analysis.decode_decision import _publish_report

    source_root = Path(__file__).resolve().parents[2]
    root = tmp_path / "project"
    results = root / "docs/results"
    results.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    output = results / "decode-decision-r"
    output.symlink_to(outside, target_is_directory=True)
    state = {
        "run_id": "r",
        "commit": "a" * 40,
        "status": "failed",
        "points": {},
        "numeric": {},
    }

    with pytest.raises(RuntimeError, match="report output"):
        _publish_report(
            root,
            output,
            state,
            report_script=source_root / "src/flexmoe/analysis/decode_oracle_report.py",
        )


def test_calibration_commit_lives_at_top_level_not_contract():
    from flexmoe.analysis.decode_decision import point_matches_commit

    assert point_matches_commit(
        {"status": "complete", "commit": "abc"}, "abc", "calibrate"
    )
    assert not point_matches_commit(
        {"status": "failed", "commit": "abc"}, "abc", "calibrate"
    )
    assert point_matches_commit(
        {"status": "complete", "contract": {"commit": "abc"}}, "abc", "native"
    )


def test_server_launcher_rejects_mac_and_unselected_gpu_set():
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    run = subprocess.run(
        ["bash", "scripts/server/run_decode_decision.sh"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 2
    assert "authorized server project" in run.stderr


@pytest.mark.parametrize("mode", ["calibrate", "native", "matched-resident", "offload"])
def test_generated_point_flags_are_accepted_by_actual_offline_wrapper(mode):
    import subprocess
    from pathlib import Path

    from flexmoe.analysis.decode_decision import point_arguments

    root = Path(__file__).resolve().parents[2]
    script = root / "scripts/server/run_decode_mechanism.sh"
    point = point_arguments(
        mode,
        "point",
        1024,
        None,
        1,
        {"profile_path": "runs/decode-mechanism/id-cal-1024/profile.json"},
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; shift; decode_mode="$1"; shift; decode_parse "$@"',
            "test",
            str(script),
            mode,
            *point["argv"][3:],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_point_summary_keeps_scheduler_pressure_and_actual_batch_distribution():
    from flexmoe.analysis.decode_decision import point_metrics

    run = summary(mode="native")
    for rep in run["repetitions"]:
        rep["worker_observations"] = [
            {"rank": rank, "decode_batch_step_counts": {"16": 20, "32": 10}}
            for rank in range(4)
        ]
        rep["scheduler"]["preemptions"]["total"] = 2
        rep["scheduler"]["kv_cache_usage"]["peak"] = 0.9
        rep["memory"] = [
            {
                "rank": rank,
                "free_gpu_bytes": 3_000_000_000 + rank,
                "torch_reserved_bytes": 1_000,
                "torch_peak_reserved_bytes": 2_000,
            }
            for rank in range(4)
        ]
    result = point_metrics(run)
    assert result["actual_kv"] == 1024
    assert result["throughput"] == 100.0
    assert result["decode_batch_mean"] == pytest.approx(64 / 3)  # (16*20+32*10)/30
    assert result["preemptions_total"] == 6
    assert result["kv_usage_peak"] == 0.9
    assert result["kv_admission_blocked"] == {
        "status": "unavailable",
        "reason": "waiting-request samples do not expose a direct KV allocation block reason",
    }
    assert result["swap_outs"] == {
        "status": "policy-disabled",
        "measured_count": None,
    }
    assert result["recomputed_tokens"]["status"] == "unavailable"
    assert result["decode_batch_distribution"] == {"16": 60, "32": 30}
    assert result["actual_kv_geometry"] == {
        "allocated_bytes_per_rank": 1024,
        "num_gpu_blocks": 32,
        "bytes_per_block": 32,
    }
    assert result["safety_headroom_per_repetition"][0]["min_bytes"] == 2_999_999_000
    assert result["throughput_samples_tokens_s"] == [100.0, 100.0, 100.0]
    assert result["throughput_range_tokens_s"] == {"min": 100.0, "max": 100.0}
    assert result["identity"] == {
        "commit": "same-commit",
        "input_sha256": "workload",
        "calibration_input_hashes_sha256": "same-calibration",
        "model_identity_sha256": "model",
        "hardware_sha256": "same-h100s",
        "profile_sha256": "same-profile",
        "batch_size": 1,
        "context_length": 1024,
        "output_length": 256,
        "max_num_seqs": 32,
        "max_num_batched_tokens": 8192,
        "seed": 20260912,
        "gpu_memory_utilization": 0.9,
        "physical_safety_reserve_bytes": 2_000_000_000,
        "engine_policy_sha256": "same-policy",
        "source_hashes": {"decode_mechanism_runner.py": "same-code"},
        "smoke_output_sha256": "same-output",
    }


def test_point_summary_keeps_phase_and_request_percentiles_per_repetition():
    from flexmoe.analysis.decode_decision import point_metrics

    run = summary(mode="native")
    for repetition, rep in enumerate(run["repetitions"]):
        rep["phase"] = {
            "status": "measured",
            "max_rank_prefill_wall_time_s": float(repetition + 1),
            "max_rank_decode_wall_time_s": float(repetition + 2),
            "max_rank_decode_step_ms_p50": float(repetition + 3),
            "max_rank_decode_step_ms_p95": float(repetition + 4),
        }
        rep["request_metrics"] = {
            "ttft_s": {"status": "measured", "p50": 1.0, "p95": 2.0, "p99": 3.0},
            "request_latency_s": {
                "status": "measured",
                "p50": 4.0,
                "p95": 5.0,
                "p99": 6.0,
            },
            "derived_tpot_s": {
                "status": "measured",
                "p50": 0.1,
                "p95": 0.2,
                "p99": 0.3,
            },
            "itl_s": {"status": "unavailable", "reason": "per-token timestamps absent"},
        }

    result = point_metrics(run)

    assert result["prefill_wall_time_median_s"] == 2.0
    assert result["decode_wall_time_median_s"] == 3.0
    assert result["phase_per_repetition"][2]["max_rank_decode_step_ms_p95"] == 6.0
    assert result["request_metrics_per_repetition"][0]["ttft_s"]["p99"] == 3.0


def test_fixed_actual_batch_gate_requires_all_ranks_and_repetitions_exactly_32():
    from flexmoe.analysis.decode_decision import fixed_actual_batch_gate

    run = summary(mode="offload")
    for rep in run["repetitions"]:
        rep["worker_observations"] = [
            {"rank": rank, "decode_batch_step_counts": {"32": 256}} for rank in range(4)
        ]
    assert fixed_actual_batch_gate(run, target=32)["status"] == "measured"

    run["repetitions"][1]["worker_observations"][3]["decode_batch_step_counts"] = {
        "31": 1,
        "32": 255,
    }
    rejected = fixed_actual_batch_gate(run, target=32)
    assert rejected["status"] == "rejected"
    assert rejected["reason"] == "decode-batch-distribution-differs"


def test_replay_is_only_reported_when_all_four_ranks_three_reps_reproduce():
    from flexmoe.analysis.decode_decision import aggregate_replay

    rows = [
        {
            "rank": rank,
            "repetition": rep,
            "status": "baseline-reproduced",
            "current_policy_misses": 3,
            "lru_misses": 2,
            "future_aware_misses": 1,
            "future_aware_saved_bytes": 20,
        }
        for rep in range(3)
        for rank in range(4)
    ]
    complete = aggregate_replay(rows)
    assert complete["status"] == "baseline-reproduced-all-ranks"
    assert complete["current_policy_misses"] == 3
    assert complete["future_aware_misses"] == 1
    assert len(complete["logical_rank0_per_repetition"]) == 3
    rows[10]["status"] = "baseline-mismatch"
    assert aggregate_replay(rows)["status"] == "incomplete-evidence"


def test_replay_rejects_rank_divergence_instead_of_summing_logical_routes():
    from flexmoe.analysis.decode_decision import aggregate_replay

    rows = [
        {
            "rank": rank,
            "repetition": rep,
            "status": "baseline-reproduced",
            "current_policy_misses": 3,
            "lru_misses": 2,
            "future_aware_misses": 1,
            "future_aware_saved_bytes": 20,
        }
        for rep in range(3)
        for rank in range(4)
    ]
    rows[7]["lru_misses"] = 99

    assert aggregate_replay(rows)["status"] == "incomplete-evidence"


def test_host_reads_container_written_replay_without_importing_torch(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    from flexmoe.analysis.decode_decision import analyze_trace_point

    folder = tmp_path / "runs/decode-mechanism/point"
    folder.mkdir(parents=True)
    (folder / "decode-cache-replay.json").write_text(
        json.dumps(
            {
                "run_id": "point",
                "commit": "expected",
                "rows": [
                    {
                        "rank": rank,
                        "repetition": rep,
                        "status": "baseline-reproduced",
                        "current_policy_misses": 3,
                        "lru_misses": 2,
                        "future_aware_misses": 1,
                        "future_aware_saved_bytes": 20,
                    }
                    for rep in range(3)
                    for rank in range(4)
                ],
            }
        )
    )
    assert (
        analyze_trace_point(tmp_path, "point", "expected")["status"]
        == "baseline-reproduced-all-ranks"
    )
    assert (
        analyze_trace_point(tmp_path, "point", "wrong")["status"]
        == "incomplete-evidence"
    )
    source = (
        Path(__file__).resolve().parents[2] / "src/flexmoe/analysis/decode_decision.py"
    )
    script = (
        "import runpy,sys; m=runpy.run_path(sys.argv[1]); "
        "print(m['analyze_trace_point'](__import__('pathlib').Path(sys.argv[2]), 'point', 'expected')['status'])"
    )
    cold_host = subprocess.run(
        [sys.executable, "-S", "-c", script, str(source), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cold_host.returncode == 0, cold_host.stderr
    assert cold_host.stdout.strip() == "baseline-reproduced-all-ranks"


def test_profile_span_is_labeled_cuda_instrumented_not_pure_decode_wall(tmp_path):
    import gzip
    import json

    from flexmoe.analysis.decode_decision import profile_timing_file

    path = tmp_path / "rank.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump(
            {
                "artifact_kind": "decode-profile",
                "profile": True,
                "observation": {
                    "coverage_status": "complete",
                    "model_step_spans": [
                        {"status": "measured", "cuda_s": 0.01},
                        {"status": "measured", "cuda_s": 0.03},
                    ],
                    "pool_profile": {
                        "dropped_rows": 0,
                        "rows": [
                            {
                                "cpu_timing": {
                                    "status": "measured",
                                    "host_gather_s": 0.004,
                                    "weight_gather_cpu_s": 0.003,
                                    "host_map_cpu_s": 0.001,
                                    "host_reuse_wait_s": 0.005,
                                },
                                "loaded_bytes": 10,
                            },
                            {
                                "cpu_timing": {
                                    "status": "measured",
                                    "host_gather_s": 0.002,
                                    "weight_gather_cpu_s": 0.0,
                                    "host_map_cpu_s": 0.002,
                                    "host_reuse_wait_s": 0.001,
                                },
                                "loaded_bytes": 0,
                            },
                        ],
                    },
                },
            },
            stream,
        )
    measured = profile_timing_file(path)
    assert measured["status"] == "instrumented"
    assert measured["model_step_cuda_p50_ms"] == 20
    assert measured["model_step_cuda_p95_ms"] == pytest.approx(29)
    assert measured["host_gather_cpu_sum_s"] == pytest.approx(0.006)
    assert measured["weight_gather_cpu_sum_s"] == pytest.approx(0.003)
    assert measured["host_map_cpu_sum_s"] == pytest.approx(0.003)
    assert measured["host_reuse_wait_cpu_sum_s"] == pytest.approx(0.006)
    assert measured["payload_bytes"] == 10


def test_build_is_offline_and_runs_before_any_gpu_point(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from flexmoe.analysis.decode_decision import build_image

    seen = []

    def fake_run(argv, *, cwd, env, stdout, stderr, timeout, check):
        seen.append((argv, cwd, env["FLEXMOE_OFFLINE_BUILD"], timeout))
        (cwd / "build").mkdir()
        (cwd / "build/image.env").write_text("IMAGE=moe-flex-local:sha\nGIT_SHA=sha\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("flexmoe.analysis.decode_decision.subprocess.run", fake_run)
    state = {"build": {"status": "pending"}}
    build_image(tmp_path, tmp_path, state, "sha")
    assert seen == [(["bash", "scripts/server/build.sh"], tmp_path, "1", 1800)]
    assert state["build"]["status"] == "complete"


def test_point_wrapper_exit_124_is_preserved_as_timeout(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace

    from flexmoe.analysis.decode_decision import _run_point

    monkeypatch.setattr(
        "flexmoe.analysis.decode_decision.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=124),
    )
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state = {"commit": "sha", "points": {}}
    point = {
        "name": "timed-out",
        "label": "timed-out",
        "mode": "offload",
        "argv": ["false"],
    }

    with pytest.raises(subprocess.TimeoutExpired):
        _run_point(tmp_path, state_dir, state, point, 120)

    assert state["points"]["timed-out"]["status"] == "timeout"
    assert state["points"]["timed-out"]["error_type"] == "TimeoutExpired"


def test_report_embeds_the_twelve_rank_level_zero_miss_proofs():
    from flexmoe.analysis.decode_oracle_report import build_oracle_report

    state = {
        "commit": "sha",
        "run_id": "id",
        "points": {},
        "budgets": {},
        "capacity": {},
        "zero_miss": {
            "status": "proven-zero-miss",
            "per_rank_repetition": [
                {
                    "repetition": rep,
                    "rank": rank,
                    "misses": 0,
                    "payload_bytes": 0,
                    "copy_launches": 0,
                }
                for rep in range(3)
                for rank in range(4)
            ],
        },
    }
    _, report = build_oracle_report(state)
    assert "| 2 | 3 | 0 | 0 | 0 |" in report
    assert sum("| 0 | 0 | 0 |" in line for line in report.splitlines()) == 12
