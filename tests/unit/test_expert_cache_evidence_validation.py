from copy import deepcopy
from types import SimpleNamespace

import pytest

from flexmoe.bench import expert_cache_evidence as evidence
from flexmoe.bench import expert_cache_suite as suite
from flexmoe.bench import partial_suite


def complete_cache_triplet():
    from test_expert_cache_bench import cache_row
    from test_partial_suite import rows

    result = rows()
    sample = cache_row()
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
        sample[key] = sample[key] // 100 * 768
    sample.update(
        net_freed_bytes=2208,
        resident_ratio=0.25,
        cache_policy="lru",
        profile_sha256="f" * 64,
    )
    settings = {
        "identity": sample["identity"],
        "resident_ratio": 0.25,
        "cache_slots": 1,
        "timing_samples": 128,
        "cache_policy": "lru",
        "calibration_count": 1,
        "profile_sha256": "f" * 64,
    }
    for index, row in enumerate(result):
        row.update(
            offload_count=0,
            offload_layers=[],
            storage_backend="native" if index == 0 else "expert-cache",
        )
        row["contract"].update(
            comparison_backend="expert-cache",
            expert_cache=deepcopy(settings),
            calibration_count=1,
            evaluation_pool_count=255,
            calibration_input_hashes_sha256="a" * 64,
        )
        row["expert_cache_stats"] = [
            (
                {
                    "rank": rank,
                    "storage_backend": "native",
                    **dict.fromkeys(evidence.COUNTERS, 0),
                    "timing": dict.fromkeys(evidence.TIMINGS, 0),
                    "policy": dict.fromkeys(evidence.POLICY_COUNTERS, 0),
                }
                if index == 0
                else {**deepcopy(sample), "rank": rank}
            )
            for rank in range(4)
        ]
        for memory in row["memory"]:
            memory.update(
                total_gpu_bytes=80000,
                free_gpu_bytes=20000,
                torch_allocated_bytes=40000,
                torch_reserved_bytes=41000,
                torch_peak_allocated_bytes=42000,
                torch_peak_reserved_bytes=43000,
            )
        for rep in row["repetitions"]:
            per_rank = []
            for rank in range(4):
                delta = {
                    "rank": rank,
                    **dict.fromkeys(evidence.COUNTERS, 0),
                    "timing": dict.fromkeys(evidence.TIMINGS, 0),
                    "policy": {
                        **dict.fromkeys(evidence.POLICY_COUNTERS, 0),
                        "cache_hit_ratio": None,
                    },
                }
                if index:
                    delta.update(
                        unique_demands=2,
                        resident_hits=2,
                        forward_counts=[1, 1],
                        per_layer_unique_demands=[1, 1],
                        per_layer_mean_unique_coverage=[0.125, 0.125],
                        gauge_snapshot={
                            "max_unique_per_forward": 1,
                            "mean_unique_coverage": 0.125,
                            "per_layer_max_unique_per_forward": [1, 1],
                        },
                        policy_snapshot={
                            key: sample["policy"][key] for key in evidence.POLICY_GAUGES
                        },
                        kernel_config_records=deepcopy(sample["kernel_config_records"]),
                    )
                    delta["policy"].update(
                        observations=2, unique_demands=2, completed_forwards=1
                    )
                per_rank.append(delta)
            rep["diagnostics"] = {
                **dict.fromkeys(evidence.COUNTERS, 0),
                "timing": dict.fromkeys(evidence.TIMINGS, 0),
                "policy": {
                    **dict.fromkeys(evidence.POLICY_COUNTERS, 0),
                    "cache_hit_ratio": None,
                },
                "per_rank": per_rank,
            }
            if index:
                rep["diagnostics"].update(unique_demands=8, resident_hits=8)
                rep["diagnostics"]["policy"].update(
                    observations=8, unique_demands=8, completed_forwards=4
                )
            rep["memory"] = deepcopy(row["memory"])
            rep["memory_peak_scope"] = "measured-generate-after-synchronized-reset"
    return result


@pytest.mark.parametrize("stage", ["raw", "sanitized"])
@pytest.mark.parametrize(
    "mutation",
    [
        "layers",
        "demands",
        "coverage",
        "timing",
        "peaks",
        "memory_rank",
        "aggregate",
        "policy_aggregate",
        "policy_large_delta",
        "timing_aggregate",
        "native_counter",
        "rank_bool",
        "impossible_peak",
    ],
)
def test_rejects_malformed_measured_evidence_raw_and_sanitized(mutation, stage):
    good = complete_cache_triplet()
    if stage == "sanitized":
        good = [partial_suite.public_run(row) for row in good]
    assert partial_suite.analyze_triplet(*good)["status"] == "validated-throughput-gain"
    broken = deepcopy(good)
    rep = broken[1]["repetitions"][0]
    delta = rep["diagnostics"]["per_rank"][0]
    if mutation == "layers":
        delta["forward_counts"] = [1]
    elif mutation == "demands":
        del delta["per_layer_unique_demands"]
    elif mutation == "coverage":
        delta["per_layer_mean_unique_coverage"] = [1.0, 1.0]
    elif mutation == "timing":
        del delta["timing"]
    elif mutation == "peaks":
        del rep["memory"][0]["torch_peak_allocated_bytes"]
    elif mutation == "memory_rank":
        rep["memory"].pop()
    elif mutation == "aggregate":
        rep["diagnostics"]["h2d_bytes"] = 999999
    elif mutation == "policy_aggregate":
        rep["diagnostics"]["policy"]["cache_hits"] = 9
    elif mutation == "policy_large_delta":
        delta["policy"]["decay_events"] = 10**15
        rep["diagnostics"]["policy"]["decay_events"] = 10**15 + 1
    elif mutation == "timing_aggregate":
        rep["diagnostics"]["timing"]["load_cuda_s"] = 10
    elif mutation == "native_counter":
        broken[0]["repetitions"][0]["diagnostics"]["resident_hits"] = 100
    elif mutation == "rank_bool":
        delta["rank"] = False
    else:
        for key in (
            "torch_allocated_bytes",
            "torch_reserved_bytes",
            "torch_peak_allocated_bytes",
            "torch_peak_reserved_bytes",
        ):
            rep["memory"][0][key] = 0
    assert partial_suite.analyze_triplet(*broken)["status"] == "invalid-comparison"
    try:
        sanitized = [partial_suite.public_run(row) for row in broken]
    except (ValueError, KeyError, TypeError):
        return
    assert partial_suite.analyze_triplet(*sanitized)["status"] == "invalid-comparison"


def test_cuda_check_forwards_selected_existing_model(tmp_path, monkeypatch):
    monkeypatch.setattr(suite.shared, "EXPECTED_ROOT", tmp_path)
    monkeypatch.setenv("GPU_IDS", "0,1,2,3")
    custom = tmp_path / "custom-model"
    custom.mkdir()
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    args = suite.parser().parse_args(
        [
            "confirm",
            "--project-root",
            str(tmp_path),
            "--suite-id",
            "custom-model",
            "--profile-path",
            str(profile),
            "--model-path",
            str(custom),
        ]
    )
    observed = []

    def external(command, **kwargs):
        if command[0] == "git":
            return SimpleNamespace(returncode=0, stdout="a" * 40)
        if "cuda-check" in command:
            observed.append(command)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(suite.subprocess, "run", external)
    suite.execute_suite(args)
    assert observed[0][observed[0].index("--model-path") + 1] == str(custom)


def test_zero_cuda_sampling_and_zero_h2d_remain_valid_but_wrong_capacity_fails():
    good = complete_cache_triplet()
    for row in good:
        row["contract"]["timing_samples"] = 0
        row["contract"]["expert_cache"]["timing_samples"] = 0
        if row["storage_backend"] == "expert-cache":
            for stat in row["expert_cache_stats"]:
                stat["cuda_timing_capacity"] = 0
    assert partial_suite.analyze_triplet(*good)["status"] == "validated-throughput-gain"
    clean = [partial_suite.public_run(row) for row in good]
    assert (
        partial_suite.analyze_triplet(*clean)["status"] == "validated-throughput-gain"
    )
    clean[1]["expert_cache_stats"][0]["cuda_timing_capacity"] = 128
    assert partial_suite.analyze_triplet(*clean)["status"] == "invalid-comparison"


def test_cuda_preflight_parser_uses_custom_model_before_cuda_gate(
    tmp_path, monkeypatch
):
    import json
    import sys

    import torch
    from test_expert_cache_bench import identity

    from flexmoe.bench import expert_cache_preflight

    model, ident = identity(tmp_path)
    output = tmp_path / "preflight.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight",
            "--project-root",
            str(tmp_path),
            "--model-path",
            str(model),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="0.10.2"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="four visible GPUs"):
        expert_cache_preflight.main()
    assert json.loads(output.read_text())["identity"] == ident
