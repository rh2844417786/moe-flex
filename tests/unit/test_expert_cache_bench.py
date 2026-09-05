from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from flexmoe.bench.partial_runner import digest_json
from flexmoe.datasets.sharegpt import PromptRecord, write_jsonl_zst


def runner():
    return importlib.import_module("flexmoe.bench.expert_cache_runner")


def evidence():
    return importlib.import_module("flexmoe.bench.expert_cache_evidence")


def inputs(tmp_path):
    dataset = tmp_path / "data.zst"
    tokens = [(1, 2), (1, 2), (3, 4), (5, 6)]
    sha = write_jsonl_zst(
        [PromptRecord(str(i), t, 2) for i, t in enumerate(tokens)], dataset
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"sha256": sha, "record_count": 4, "counts_by_context": {"2": 4}})
    )
    return dataset, manifest


def identity(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_next",
                "num_hidden_layers": 2,
                "num_experts": 8,
                "hidden_size": 16,
                "moe_intermediate_size": 32,
            }
        )
    )
    (model / "model.safetensors.index.json").write_text("{}")
    from flexmoe.vllm.expert_calibration import model_profile_identity

    return model, model_profile_identity(model, 4)


def test_split_excludes_duplicate_token_hashes_before_repetition(tmp_path):
    data, manifest = inputs(tmp_path)
    split = runner().split_workload(data, manifest, 2, 5, 1)
    assert split.calibration == ((1, 2),)
    assert split.evaluation.prompts == ((3, 4), (5, 6), (3, 4), (5, 6), (3, 4))
    assert split.evaluation.metadata["calibration_input_hashes_sha256"] == digest_json(
        [digest_json((1, 2))]
    )
    with pytest.raises(ValueError, match="evaluation"):
        runner().split_workload(data, manifest, 2, 5, 3)


def test_profile_aggregation_validates_each_rank_identity_and_heldout_hashes(tmp_path):
    model, ident = identity(tmp_path)
    hashes = [digest_json((1, 2))]
    rows = [
        {
            "schema_version": 1,
            "rank": i,
            **ident,
            "counts": [[1] * 8, [2] * 8],
            "forward_counts": [2, 3],
        }
        for i in range(4)
    ]
    profile = runner().profile_from_workers(rows, ident, hashes)
    assert profile.forward_counts == (8, 12)
    assert profile.counts[0] == (4.0,) * 8
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile.to_dict()))
    runner().load_profile(path, model, 4, hashes)
    for key, replacement in (
        ("tensor_parallel_size", 2),
        ("model_identity_sha256", "a" * 64),
        ("geometry", {**ident["geometry"], "hidden_size": 17}),
    ):
        bad = deepcopy(rows)
        bad[3][key] = replacement
        with pytest.raises(ValueError):
            runner().profile_from_workers(bad, ident, hashes)
    with pytest.raises(ValueError):
        runner().load_profile(path, model, 4, [digest_json((3, 4))])


def cache_row(rank=0):
    return {
        "schema_version": 1,
        "rank": rank,
        "total_layers": 2,
        "num_experts": 8,
        "tensor_parallel_size": 4,
        "physical_slots": 13,
        "resident_slots": 4,
        "cache_slots": 1,
        "ingress_slots": 8,
        "expert_bytes": 100,
        "host_source_bytes": 1600,
        "pinned_gather_bytes": 800,
        "gpu_pool_bytes": 1300,
        "gpu_resident_bytes": 400,
        "gpu_cache_bytes": 100,
        "gpu_ingress_bytes": 800,
        "gpu_metadata_bytes": 96,
        "pinned_metadata_bytes": 96,
        "net_freed_bytes": 204,
        "h2d_bytes": 0,
        "copy_launches": 0,
        "startup_resident_h2d_bytes": 400,
        "promotion_d2d_bytes": 0,
        "metadata_h2d_bytes": 0,
        "forward_counts": [2, 2],
        "unique_demands": 4,
        "max_unique_per_forward": 1,
        "mean_unique_coverage": 0.125,
        "resident_hits": 4,
        "per_layer_unique_demands": [2, 2],
        "per_layer_max_unique_per_forward": [1, 1],
        "policy": {
            "observations": 4,
            "unique_demands": 4,
            "completed_forwards": 2,
            "decay_events": 0,
            "resident_experts": 4,
            "cache_slots": 1,
            "cache_entries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_hit_ratio": 0.0,
            "admissions": 0,
            "evictions": 0,
            "cache_bypasses": 0,
            "reconfigurations": 0,
        },
        "timing": {
            key: 0
            for key in (
                "route_d2h_s",
                "policy_cpu_s",
                "host_reuse_wait_s",
                "host_gather_s",
                "h2d_enqueue_s",
                "compute_enqueue_s",
                "cuda_sample_count",
                "load_cuda_s",
                "compute_cuda_s",
                "promotion_cuda_s",
            )
        },
        "weights_verified": 0,
        "cuda_timing_capacity": 128,
        "failed": False,
        "identity": {
            "geometry": {
                "total_layers": 2,
                "num_experts": 8,
                "hidden_size": 16,
                "intermediate_size": 8,
            },
            "tensor_parallel_size": 4,
            "model_config_sha256": "a" * 64,
            "model_identity_sha256": "b" * 64,
        },
        "kernel_config_records": {
            "capacity": 64,
            "unrecorded_selections": 0,
            "records": [
                {
                    "chunk_tokens": 1,
                    "w13_shape": [8, 16, 16],
                    "w2_shape": [8, 16, 8],
                    "top_k": 2,
                    "config": {"BLOCK_SIZE_M": 16},
                    "selections": 4,
                }
            ],
        },
    }


def test_counter_delta_does_not_subtract_gauges_or_ratios_and_preserves_layers():
    old = cache_row()
    new = deepcopy(old)
    new.update(
        forward_counts=[3, 4],
        unique_demands=10,
        per_layer_unique_demands=[4, 6],
        resident_hits=10,
    )
    new["policy"].update(
        observations=7,
        unique_demands=10,
        completed_forwards=3,
        cache_hit_ratio=0.0,
        cache_entries=0,
    )
    result = evidence().cache_deltas([old], [new], expected_workers=1)
    assert result["unique_demands"] == 6
    assert result["per_rank"][0]["forward_counts"] == [1, 2]
    assert result["per_rank"][0]["per_layer_mean_unique_coverage"] == [0.25, 0.25]
    assert "cache_entries" not in result["policy"]
    assert result["policy"]["cache_hit_ratio"] is None
    bad = deepcopy(new)
    bad["h2d_bytes"] = -1
    with pytest.raises(ValueError):
        evidence().cache_deltas([old], [bad], expected_workers=1)


def test_native_control_never_calls_uninitialized_cache_rpc(tmp_path, monkeypatch):
    from flexmoe.bench.partial_runner import PartialRunConfig

    model, _ = identity(tmp_path)
    data, manifest = inputs(tmp_path)
    cfg = PartialRunConfig(
        "resident", model, data, manifest, context_length=2, timing_samples=7
    )
    monkeypatch.setattr(os, "environ", os.environ.copy())
    backend = runner().ExpertBackend(cfg, tmp_path / "unused", 0.25, 1, "lru", 1)
    backend.configure(cfg, tmp_path, ())
    assert os.environ["FLUXMOE_ENABLE"] == "0"
    assert os.environ["FLUXMOE_STORAGE_MODE"] == "expert-cache"
    assert os.environ["FLUXMOE_EXPERT_TIMING_SAMPLES"] == "7"

    class Engine:
        def collective_rpc(self, *args, **kwargs):
            raise AssertionError("native must not call cache RPC")

    rows = backend.snapshot(Engine(), 4)
    assert all(row["h2d_bytes"] == 0 for row in rows)
    assert (
        backend.kv_budget(
            [
                {
                    "rank": i,
                    "kv_cache_allocated_bytes": 1000,
                    "available_kv_cache_bytes": 1200,
                    "num_gpu_blocks": 10,
                    "kv_cache_accounting_consistent": True,
                }
                for i in range(4)
            ],
            4,
        )
        == 1000
    )


def test_cache_public_schema_rejects_nan_and_drops_nested_private_strings():
    row = cache_row()
    row["secret"] = "private-prompt"
    row["policy"]["prompt"] = "private-prompt"
    row["kernel_config_records"]["records"][0]["config"]["private"] = "private-prompt"
    with pytest.raises(ValueError):
        evidence().public_stats(row)
    del row["kernel_config_records"]["records"][0]["config"]["private"]
    clean = evidence().public_stats(row)
    assert "private-prompt" not in json.dumps(clean)
    assert clean["weights_verified"] == 0
    assert clean["kernel_config_records"]["records"][0]["config"]["BLOCK_SIZE_M"] == 16
    row["timing"]["load_cuda_s"] = float("nan")
    with pytest.raises(ValueError):
        evidence().public_stats(row)


def test_host_entrypoint_is_stdlib_only_and_forwards_custom_dataset():
    path = Path(__file__).parents[2] / "src/flexmoe/bench/expert_cache_suite.py"
    result = subprocess.run(
        [sys.executable, "-S", str(path), "confirm", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--dataset-path" in result.stdout
    suite = importlib.import_module("flexmoe.bench.expert_cache_suite")
    args = suite.parser().parse_args(
        [
            "confirm",
            "--suite-id",
            "x",
            "--profile-path",
            "/p",
            "--dataset-path",
            "/custom.zst",
            "--dataset-manifest",
            "/custom.json",
            "--model-path",
            "/existing-model",
            "--resident-ratio",
            ".75",
        ]
    )
    command = suite.point_command(
        Path("/root"), "resident", Path("/r"), None, vars(args)
    )
    assert command[command.index("--dataset-path") + 1] == "/custom.zst"
    assert command[command.index("--model-path") + 1] == "/existing-model"
    assert command[command.index("--resident-ratio") + 1] == "0.75"


def test_native_calibration_then_shared_rbc_excludes_inputs_and_exports_negative_results(
    tmp_path, monkeypatch
):
    from test_partial_runner import Engine, memory_rows

    from flexmoe.bench import partial_suite
    from flexmoe.bench.partial_runner import PartialRunConfig

    model, ident = identity(tmp_path)
    data, manifest = inputs(tmp_path)
    profile_path = tmp_path / "profile.json"
    cfg = PartialRunConfig(
        "resident",
        model,
        data,
        manifest,
        batch_size=5,
        context_length=2,
        output_length=3,
        repetitions=3,
        warmups=1,
    )
    monkeypatch.setattr(os, "environ", os.environ.copy())
    captured = []

    class Boundary(Engine):
        active = False

        def generate(self, prompts, sampling, **kwargs):
            captured.append((self.active, [p["prompt_token_ids"] for p in prompts]))
            return super().generate(prompts, sampling, **kwargs)

        def collective_rpc(self, method, kwargs=None):
            if method == "fluxmoe_reset_memory_peaks":
                return list(range(4))
            if method == "fluxmoe_expert_calibration":
                assert os.environ["FLUXMOE_ENABLE"] == "0"
                self.active = kwargs["action"] == "start"
                return [
                    {
                        "rank": rank,
                        "schema_version": 1,
                        **ident,
                        "active": self.active,
                        "counts": [[1] * 8, [1] * 8],
                        "forward_counts": [1, 1],
                    }
                    for rank in range(4)
                ]
            if method == "fluxmoe_expert_cache_stats":
                assert os.environ["FLUXMOE_ENABLE"] == "1"
                rows = []
                for rank in range(4):
                    row = cache_row(rank)
                    row.update(
                        identity=ident,
                        resident_ratio=0.25,
                        cache_policy="lru",
                        profile_sha256=digest_json(
                            json.loads(profile_path.read_text())
                        ),
                    )
                    for key in (
                        "expert_bytes",
                        "host_source_bytes",
                        "pinned_gather_bytes",
                        "gpu_pool_bytes",
                        "gpu_resident_bytes",
                        "gpu_cache_bytes",
                        "gpu_ingress_bytes",
                        "startup_resident_h2d_bytes",
                    ):
                        row[key] = row[key] // 100 * 768
                    row["net_freed_bytes"] = 2208
                    row.update(
                        forward_counts=[self.calls] * 2,
                        per_layer_unique_demands=[self.calls] * 2,
                        unique_demands=2 * self.calls,
                        resident_hits=2 * self.calls,
                    )
                    row["policy"].update(
                        observations=2 * self.calls,
                        completed_forwards=self.calls,
                        unique_demands=2 * self.calls,
                    )
                    rows.append(row)
                return rows
            if method == "fluxmoe_worker_memory_stats":
                return [
                    {
                        **row,
                        "torch_peak_allocated_bytes": 42000,
                        "torch_peak_reserved_bytes": 43000,
                    }
                    for row in memory_rows()
                ]
            return super().collective_rpc(method, kwargs)

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            LLM=Boundary, SamplingParams=SimpleNamespace, __version__="0.10.2"
        ),
    )
    runner().run_calibration(
        cfg,
        project_root=Path.cwd(),
        run_dir=tmp_path / "calibration",
        profile_path=profile_path,
        calibration_count=1,
    )
    assert captured == [(True, [[1, 2]])]
    captured.clear()
    runs = []
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    for arm in partial_suite.ARMS:
        point = replace(cfg, arm=arm)
        backend = runner().ExpertBackend(point, profile_path, 0.25, 1, "lru", 1)
        run = suite_dir / arm
        runner().shared.run_benchmark(
            point,
            project_root=Path.cwd(),
            run_dir=run,
            resident_run=None if arm == "resident" else suite_dir / "resident",
            backend=backend,
        )
        runs.append(json.loads((run / "summary.json").read_text()))
    assert all(not active and [1, 2] not in prompts for active, prompts in captured)
    assert runs[1]["requested_kv_cache_bytes"] == 1000
    assert runs[1]["repetitions"][0]["diagnostics"]["h2d_bytes"] == 0
    assert runs[1]["repetitions"][0]["diagnostics"]["per_rank"][0][
        "forward_counts"
    ] == [1, 1]
    assert runs[0]["contract"] == runs[1]["contract"] == runs[2]["contract"]
    assert partial_suite.analyze_triplet(*runs)["status"] == "no-stable-net-gain"
    for key in ("cache_policy", "profile_sha256", "resident_ratio"):
        bad = deepcopy(runs)
        bad[1]["expert_cache_stats"][0][key] = "invalid"
        assert partial_suite.analyze_triplet(*bad)["status"] == "invalid-comparison"
    (suite_dir / "suite.json").write_text(
        json.dumps(
            {
                "run_ids": list(partial_suite.ARMS),
                "groups": [{arm: arm for arm in partial_suite.ARMS}],
                "status": "finished",
                "cuda_preflight": "passed",
            }
        )
    )
    suite = importlib.import_module("flexmoe.bench.expert_cache_suite")
    result = suite.export_suite(suite_dir, tmp_path / "public")
    assert result["comparisons"][0]["status"] == "no-stable-net-gain"
    assert result["runs"][1]["expert_cache_stats"][0]["weights_verified"] == 0
    assert (
        "per_layer_mean_unique_coverage"
        in (tmp_path / "public/evidence.csv").read_text()
    )
    rendered = (tmp_path / "public/report.md").read_text()
    assert "BLOCK_SIZE_M" in rendered and "42000" in rendered
    assert "$MODEL_PATH" in rendered and "--max-num-batched-tokens 8192" in rendered
    assert str(model) not in rendered
    bad = runs[2]
    bad["repetitions"][0]["output_tokens_per_second"] = 1e30
    (suite_dir / partial_suite.ARMS[2] / "summary.json").write_text(json.dumps(bad))
    malformed = suite.export_suite(suite_dir, tmp_path / "invalid")
    assert malformed["runs"][2]["status"] == "failed"
    assert malformed["comparisons"][0]["stable_three_repetition_gain"] is False


def test_cuda_gate_rejects_all_skipped_or_empty_junit(tmp_path):
    module = importlib.import_module("flexmoe.bench.expert_cache_preflight")
    path = tmp_path / "junit.xml"
    for attributes in (
        'tests="5" skipped="5" errors="0" failures="0"',
        'tests="0" skipped="0" errors="0" failures="0"',
        'tests="5" skipped="0" errors="0" failures="1"',
    ):
        path.write_text(f"<testsuites><testsuite {attributes}/></testsuites>")
        with pytest.raises(ValueError):
            module.validate_junit(path)
    path.write_text(
        '<testsuites><testsuite tests="14" skipped="0" errors="0" failures="0"/></testsuites>'
    )
    assert module.validate_junit(path)["tests"] == 14


def test_counter_schema_rejects_fractional_counts_and_changed_layout():
    old = cache_row()
    new = deepcopy(old)
    new["policy"]["cache_misses"] = 0.5
    with pytest.raises(ValueError):
        evidence().public_stats(new)
    new = deepcopy(old)
    new["resident_slots"] += 1
    with pytest.raises(ValueError):
        evidence().cache_deltas([old], [new], expected_workers=1)


def test_suite_stops_after_failed_point_and_exports_failure(tmp_path, monkeypatch):
    suite = importlib.import_module("flexmoe.bench.expert_cache_suite")
    monkeypatch.setattr(suite.shared, "EXPECTED_ROOT", tmp_path)
    monkeypatch.setenv("GPU_IDS", "0,1,2,3")
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    args = suite.parser().parse_args(
        [
            "confirm",
            "--project-root",
            str(tmp_path),
            "--suite-id",
            "stop-on-failure",
            "--profile-path",
            str(profile),
        ]
    )
    calls = []

    def external(command, **kwargs):
        if command[0] == "git":
            return SimpleNamespace(returncode=0, stdout="a" * 40)
        if "cuda-check" in command:
            return SimpleNamespace(returncode=0)
        calls.append(command[command.index("--arm") + 1])
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(suite.subprocess, "run", external)
    source = suite.execute_suite(args)
    assert calls == ["resident"]
    report = json.loads((source / "public/summary.json").read_text())
    assert report["suite_status"] == "failed"
    assert report["commit"] == "a" * 40
    assert report["comparisons"][0]["stable_three_repetition_gain"] is False


def test_export_preserves_sanitized_calibration_rank_counts(tmp_path):
    suite = importlib.import_module("flexmoe.bench.expert_cache_suite")
    _, ident = identity(tmp_path)
    source = tmp_path / "cal-suite"
    (source / "calibration").mkdir(parents=True)
    (source / "suite.json").write_text(
        json.dumps(
            {
                "run_ids": [],
                "groups": [],
                "status": "finished",
                "cuda_preflight": "passed",
            }
        )
    )
    rows = [
        {
            "schema_version": 1,
            "rank": i,
            **ident,
            "active": False,
            "counts": [[1] * 8, [2] * 8],
            "forward_counts": [2, 3],
            "prompt": "private-prompt",
        }
        for i in range(4)
    ]
    (source / "calibration/summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "identity": ident,
                "profile_sha256": "a" * 64,
                "calibration_input_hashes": ["b" * 64],
                "rank_profiles": rows,
                "private_path": "/secret",
                "commit": "a" * 40,
            }
        )
    )
    result = suite.export_suite(source, tmp_path / "public-cal")
    assert result["calibration"]["rank_profiles"][3]["counts"][1] == [2] * 8
    assert "private-prompt" not in json.dumps(result)
    assert "/secret" not in json.dumps(result)


def test_suite_does_not_mark_invalid_completed_triplet_successful(
    tmp_path, monkeypatch
):
    suite = importlib.import_module("flexmoe.bench.expert_cache_suite")
    monkeypatch.setattr(suite.shared, "EXPECTED_ROOT", tmp_path)
    monkeypatch.setenv("GPU_IDS", "0,1,2,3")
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    args = suite.parser().parse_args(
        [
            "confirm",
            "--project-root",
            str(tmp_path),
            "--suite-id",
            "invalid-triplet",
            "--profile-path",
            str(profile),
        ]
    )
    monkeypatch.setattr(
        suite.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="a" * 40),
    )
    monkeypatch.setattr(
        suite,
        "execute_point",
        lambda root, arm, run, *a, **kw: {
            "arm": arm,
            "status": "complete",
            "run_id": run.name,
        },
    )
    source = suite.execute_suite(args)
    assert json.loads((source / "suite.json").read_text())["status"] == "failed"
