from __future__ import annotations

import importlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest


def module():
    return importlib.import_module("flexmoe.bench.partial_suite")


def rows():
    contract = {
        "commit": "a" * 40,
        "model_identity_sha256": "b" * 64,
        "model_config_sha256": "c" * 64,
        "input_sha256": "d" * 64,
        "dataset_sha256": "e" * 64,
        "dataset_manifest_sha256": "f" * 64,
        "engine_policy_sha256": "1" * 64,
        "batch_size": 4,
        "context_length": 1024,
        "output_length": 16,
        "tensor_parallel_size": 4,
        "dtype": "bfloat16",
        "seed": 7,
        "gpu_memory_utilization": 0.6,
        "warmups": 1,
        "timing_samples": 128,
        "smoke_output_length": 8,
        "source_request_count": 256,
        "unique_selected_request_count": 4,
        "repeated_request_count": 0,
        "sampling_policy": "existing-prompts",
        "versions": {
            "torch": "2.8.0+cu128",
            "vllm": "0.10.2",
            "cuda": "12.8",
            "vllm_commit": "2" * 40,
        },
    }
    result = []
    for arm, label, speeds, kv in [
        ("resident", "r", (100, 101, 102), 1000),
        ("partial-fixed-kv", "b", (90, 91, 92), 1000),
        ("partial-auto-kv", "c", (120, 121, 122), 2000),
    ]:
        result.append(
            {
                "schema_version": 1,
                "run_id": label,
                "arm": arm,
                "status": "complete",
                "offload_count": 0 if arm == "resident" else 4,
                "staging_slots": 1,
                "offload_layers": [] if arm == "resident" else [1, 3, 5, 7],
                "contract": deepcopy(contract),
                "engine_policy": {"cache_config": {"enable_prefix_caching": False}},
                "hardware_budget": {
                    "total_gpu_bytes_by_rank": [80000] * 4,
                    "requested_gpu_bytes_by_rank": [48000] * 4,
                },
                "memory": [
                    {
                        "rank": rank,
                        "kv_cache_allocated_bytes": kv,
                        "kv_cache_declared_bytes": kv,
                        "kv_cache_accounting_consistent": True,
                        "num_gpu_blocks": kv // 100,
                        "available_kv_cache_bytes": kv + 10,
                    }
                    for rank in range(4)
                ],
                "smoke": {
                    "input_sha256": "3" * 64,
                    "output_sha256": "4" * 64,
                    "request_count": 1,
                    "generated_tokens": 8,
                },
                "smoke_matches_resident": None if arm == "resident" else True,
                "mechanism_validated": arm != "resident",
                "repetitions_completed": 3,
                "repetitions": [
                    {
                        "repetition": index,
                        "output_tokens_per_second": speed,
                        "elapsed_s": 64 / speed,
                        "request_count": 4,
                        "generated_tokens": 64,
                        "output_sha256": "5" * 64,
                        "ttft_median_s": None,
                        "request_latency_median_s": None,
                        "latency_available_requests": 0,
                        "ttft_available_requests": 0,
                        "timing_available": False,
                        "diagnostics": {
                            "h2d_bytes": 0,
                            "copy_launches": 0,
                            "offload_forwards": 0,
                            "resident_forwards": 0,
                            "timing": {
                                "sample_count": 0,
                                "load_cuda_s": 0,
                                "wait_cuda_s": 0,
                                "compute_cuda_s": 0,
                                "cpu_enqueue_s": 0,
                            },
                        },
                    }
                    for index, speed in enumerate(speeds)
                ],
            }
        )
    return result


def test_scan_has_only_valid_small_candidates_and_zero_overhead_control():
    points = module().scan_points()
    assert [
        (point.arm, point.offload_count, point.staging_slots) for point in points
    ] == [
        ("resident", 0, 1),
        ("partial-auto-kv", 0, 1),
        ("partial-auto-kv", 2, 1),
        ("partial-auto-kv", 4, 1),
        ("partial-auto-kv", 4, 2),
        ("partial-auto-kv", 6, 1),
        ("partial-auto-kv", 6, 2),
        ("partial-auto-kv", 8, 1),
        ("partial-auto-kv", 8, 2),
    ]


def test_analyzer_separates_offload_cost_capacity_gain_and_net_gain():
    analysis = module().analyze_triplet(*rows())
    assert analysis["status"] == "validated-throughput-gain"
    assert analysis["b_over_r"] == pytest.approx(91 / 101)
    assert analysis["c_over_b"] == pytest.approx(121 / 91)
    assert analysis["c_over_r"] == pytest.approx(121 / 101)
    assert analysis["stable_three_repetition_gain"] is True
    assert analysis["kv_gain_bytes_total"] == 4000


def test_analyzer_refuses_mismatched_policy_capacity_or_failed_run():
    for mutation in ("policy", "capacity", "failed", "smoke", "length"):
        r, b, c = rows()
        if mutation == "policy":
            b["contract"]["engine_policy_sha256"] = "9" * 64
        elif mutation == "capacity":
            b["memory"][0]["num_gpu_blocks"] = 9
        elif mutation == "failed":
            c["status"] = "failed"
        elif mutation == "smoke":
            c["smoke"]["output_sha256"] = "9" * 64
        else:
            c["repetitions"][0]["generated_tokens"] = 63
        result = module().analyze_triplet(r, b, c)
        assert result["status"] in ("invalid-comparison", "incomplete")
        assert result["stable_three_repetition_gain"] is False


def test_fast_single_rep_cannot_satisfy_stable_gain_and_perf_mismatch_is_visible():
    r, b, c = rows()
    c["repetitions"][0]["output_sha256"] = "9" * 64
    result = module().analyze_triplet(r, b, c)
    assert result["performance_output_hashes_match"] is False
    for row in (r, b, c):
        row["repetitions"] = row["repetitions"][:1]
        row["repetitions_completed"] = 1
    assert module().analyze_triplet(r, b, c)["stable_three_repetition_gain"] is False


def test_public_export_uses_typed_allowlist_even_for_nested_fields(tmp_path: Path):
    row = rows()[0]
    row.update(
        {
            "model_path": "/mnt/private/secret",
            "stderr": "private-machine",
            "prompts": [123],
        }
    )
    row["contract"]["machine"] = "private-machine"
    row["contract"]["versions"]["hostname"] = "private-machine"
    row["memory"][0]["path"] = "/mnt/private/secret"
    row["repetitions"][0]["output_token_ids"] = [[123456]]
    clean = module().public_run(row)
    text = json.dumps(clean)
    assert (
        "secret" not in text and "private-machine" not in text and "123456" not in text
    )
    assert clean["contract"]["commit"] == "a" * 40
    assert clean["contract"]["versions"]["vllm"] == "0.10.2"
    with pytest.raises(ValueError, match="run ID"):
        module().public_run({**row, "run_id": "../private"})


def test_point_command_preserves_argument_boundaries_and_contains_container_timeout(
    tmp_path: Path,
):
    point = module().Point("partial-fixed-kv", 4, 2)
    command = module().point_command(
        tmp_path,
        point,
        tmp_path / "r-test",
        tmp_path / "r-base",
        {
            "timeout_s": 60,
            "batch_size": 512,
            "context_length": 4096,
            "output_length": 128,
            "gpu_memory_utilization": 0.6,
            "warmups": 1,
            "repetitions": 3,
            "max_num_seqs": 256,
            "max_num_batched_tokens": 8192,
            "timing_samples": 128,
            "seed": 7,
        },
    )
    assert command[:3] == [
        "bash",
        str(tmp_path / "scripts/server/run_partial_offload.sh"),
        "point",
    ]
    assert command[command.index("--timeout-s") + 1] == "60"
    assert command[command.index("--resident-run") + 1] == str(tmp_path / "r-base")
    assert command[command.index("--staging-slots") + 1] == "2"


def test_launcher_help_and_root_guard_execute_without_gpu():
    path = Path("scripts/server/run_partial_offload.sh")
    help_result = subprocess.run(
        ["bash", str(path), "--help"], capture_output=True, text=True, check=False
    )
    assert help_result.returncode == 0
    assert "scan" in help_result.stdout and "confirm" in help_result.stdout
    rejected = subprocess.run(
        ["bash", str(path), "scan"], capture_output=True, text=True, check=False
    )
    assert rejected.returncode == 2
    assert "expected /home/jovyan/wangtonghan/moe-flex" in rejected.stderr


def test_export_produces_numeric_json_csv_and_honest_markdown(tmp_path: Path):
    source = tmp_path / "suite"
    source.mkdir()
    for row in rows():
        path = source / row["run_id"]
        path.mkdir()
        (path / "summary.json").write_text(json.dumps(row))
    (source / "public").mkdir()
    (source / "public" / "summary.json").write_text(json.dumps({"schema_version": 1}))
    (source / "suite.json").write_text(
        json.dumps(
            {
                "mode": "confirm",
                "groups": [
                    {"resident": "r", "partial-fixed-kv": "b", "partial-auto-kv": "c"}
                ],
            }
        )
    )
    output = tmp_path / "public"
    module().export_suite(source, output)
    assert (
        json.loads((output / "summary.json").read_text())["comparisons"][0]["c_over_r"]
        > 1
    )
    assert "partial-auto-kv" in (output / "summary.csv").read_text()
    assert "batch=1" in (output / "report.md").read_text()
    assert str(source) not in (output / "summary.json").read_text()
    with pytest.raises(FileExistsError):
        module().export_suite(source, output)


def test_point_timeout_keeps_numeric_status_and_private_stderr(tmp_path: Path):
    script = tmp_path / "scripts/server/run_partial_offload.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\necho private-diagnostic >&2\nexit 124\n")
    target = tmp_path / "runs/test/r-one"
    options = {
        "timeout_s": 1,
        "batch_size": 4,
        "context_length": 1024,
        "output_length": 16,
        "gpu_memory_utilization": 0.6,
        "warmups": 0,
        "repetitions": 1,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 1024,
        "timing_samples": 0,
        "seed": 7,
    }
    row = module().execute_point(
        tmp_path, module().Point("resident", 0, 1), target, None, options
    )
    assert row["status"] == "timeout" and row["exit_code"] == 124
    assert "private-diagnostic" not in (target / "summary.json").read_text()
    assert "private-diagnostic" in (target.parent / "logs/r-one.stderr.log").read_text()


def test_confirm_cli_requires_one_candidate_source():
    with pytest.raises(SystemExit) as error:
        module().main(["confirm", "--offload-count", "4", "--scan-dir", "/unused"])
    assert error.value.code == 2
