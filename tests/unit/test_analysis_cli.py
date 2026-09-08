from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from test_analysis_cost import samples, trace

from flexmoe.analysis.schema import diagnostic_artifact

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "src/flexmoe/analysis/cli.py"


def invoke(*args):
    return subprocess.run(
        [sys.executable, "-S", str(CLI), *map(str, args)],
        text=True,
        capture_output=True,
        check=False,
    )


def save(path, value):
    path.write_text(json.dumps(value))
    return path


def transport_artifact(rows):
    return diagnostic_artifact(
        "transfer-samples",
        {
            "status": "complete",
            "samples": [row.to_dict() for row in rows],
            "contract": rows[0].contract,
            "config": {
                "experts_per_batch": sorted({row.experts_per_batch for row in rows}),
                "modes": sorted({row.mode for row in rows}),
                "repetitions": 1,
                "contention": rows[0].contention,
                "expert_bytes": rows[0].expert_bytes,
            },
        },
    )


def native(kv=100, seconds=10):
    contract = trace(0).to_dict()["contract"]
    contract.update(
        repetitions_requested=1,
        warmups=1,
        smoke_output_length=1,
        unique_selected_request_count=2,
        repeated_request_count=8,
    )
    policy = {
        "model_config": {"enforce_eager": False},
        "parallel_config": {"tensor_parallel_size": 4},
        "scheduler_config": {"max_num_seqs": 10, "max_num_batched_tokens": 80},
    }
    contract["engine_policy_sha256"] = hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    memory = [
        {
            "rank": r,
            "total_gpu_bytes": 1000,
            "free_gpu_bytes": 300,
            "torch_allocated_bytes": kv + 100,
            "torch_reserved_bytes": kv + 100,
            "torch_peak_allocated_bytes": kv + 100,
            "torch_peak_reserved_bytes": kv + 100,
            "kv_cache_allocated_bytes": kv,
            "kv_cache_declared_bytes": kv,
            "kv_cache_accounting_consistent": True,
            "num_gpu_blocks": 10,
        }
        for r in range(4)
    ]
    sample = {
        "request_count": 10,
        "generated_tokens": 1000,
        "elapsed_s": seconds,
        "output_tokens_per_second": 1000 / seconds,
        "output_sha256": "a" * 64,
    }
    rep = diagnostic_artifact(
        "analysis-repetition",
        dict(sample, timing_eligible=True, repetition=0, memory=deepcopy(memory)),
    )
    smoke = diagnostic_artifact(
        "analysis-smoke",
        {
            "request_count": 1,
            "generated_tokens": 1,
            "elapsed_s": 1,
            "output_tokens_per_second": 1,
            "output_sha256": "a" * 64,
            "timing_eligible": True,
        },
    )
    return diagnostic_artifact(
        "native-summary",
        {
            "run_id": "native-1",
            "status": "complete",
            "arm": "resident",
            "offload_count": 0,
            "offload_layers": [],
            "engine_mode": "native",
            "timing_eligible": True,
            "measurement_evidence": "measured",
            "source_kind": "native-measured",
            "physical_safety_reserve_bytes": 0,
            "contract": contract,
            "memory": memory,
            "engine_policy": policy,
            "final_memory": deepcopy(memory),
            "repetitions": [rep],
            "repetitions_completed": 1,
            "smoke": smoke,
            "throughput_median": 1000 / seconds,
            "throughput_min": 1000 / seconds,
            "throughput_max": 1000 / seconds,
        },
    )


@pytest.mark.parametrize(
    "command", [None, "validate", "replay", "analyze", "plan", "export"]
)
def test_stdlib_help(command):
    result = invoke(*([command] if command else []), "--help")
    assert result.returncode == 0, result.stderr


def test_native_conversion_and_rejects_contaminated_or_incomplete(tmp_path):
    path = save(tmp_path / "native.json", native())
    result = invoke("validate", path)
    assert result.returncode == 0, result.stderr
    for mutate in (
        lambda x: x.update(engine_mode="trace"),
        lambda x: x.update(timing_eligible=False),
        lambda x: x["repetitions"][0].update(generated_tokens=999),
        lambda x: x["repetitions"][0].update(output_tokens_per_second=99),
        lambda x: x["final_memory"][2].update(num_gpu_blocks=9),
        lambda x: x["contract"].update(prompt_hashes=[]),
        lambda x: x.update(formal_offload_gain=0),
    ):
        raw = deepcopy(native())
        mutate(raw)
        save(path, raw)
        assert invoke("validate", path).returncode != 0


def test_real_trace_replay_analyze_and_atomic_failure(tmp_path):
    paths = []
    calibrations = []
    for rank in range(4):
        raw = trace(rank).to_dict()
        raw["contract"].update(
            repetitions_requested=1,
            warmups=1,
            smoke_output_length=1,
            unique_selected_request_count=2,
            repeated_request_count=8,
        )
        wrapper = lambda t: diagnostic_artifact(
            "demand-trace",
            {
                "trace": t,
                "capture": {"full_workload": True},
                "generation_status": "complete",
                "timing_eligible": False,
                "missing_evidence": [],
            },
        )
        path = tmp_path / f"trace-{rank}.json.gz"
        with gzip.open(path, "wt") as handle:
            json.dump(wrapper(raw), handle)
        paths.append(path)
        calibration = deepcopy(raw)
        calibration["contract"].update(input_sha256="c" * 64, prompt_hashes=["b" * 64])
        calibrations.append(save(tmp_path / f"cal-{rank}.json", wrapper(calibration)))
    replay = tmp_path / "replay.json"
    result = invoke(
        "replay",
        "--traces",
        *paths,
        "--calibration",
        *calibrations,
        "--offload-fraction",
        0.5,
        "--cache-slots",
        0,
        "--staging-experts",
        2,
        "--output",
        replay,
    )
    assert result.returncode == 0, result.stderr
    transport = save(
        tmp_path / "samples.json",
        transport_artifact(samples()),
    )
    baseline = save(tmp_path / "r.json", native())
    reference = save(tmp_path / "k.json", native(kv=200, seconds=8))
    output = tmp_path / "analysis.json"
    result = invoke(
        "analyze",
        "--traces",
        *paths,
        "--replay",
        replay,
        "--samples",
        transport,
        "--baseline",
        baseline,
        "--kv-reference",
        reference,
        "--output",
        output,
    )
    assert result.returncode == 0, result.stderr
    analysis = json.loads(output.read_text())
    assert analysis["analyses"][0]["time_headroom_s"] == 2
    assert analysis["diagnostic_only"] is True
    assert analysis["baseline_evidence"]["contract"]["gpu_memory_utilization"] == 0.9
    assert analysis["baseline_evidence"]["contract"]["repeated_request_count"] == 8
    # Malformed second evidence file must not produce partial output.
    save(reference, {"status": "complete"})
    previous = output.read_bytes()
    assert (
        invoke(
            "analyze",
            "--traces",
            *paths,
            "--replay",
            replay,
            "--samples",
            transport,
            "--baseline",
            baseline,
            "--kv-reference",
            reference,
            "--output",
            output,
        ).returncode
        != 0
    )
    assert output.read_bytes() == previous


def test_finite_plan_preserves_custom_flags_and_does_not_execute(tmp_path):
    output = tmp_path / "plan.json"
    result = invoke(
        "plan",
        "--output",
        output,
        "--model-path",
        "/mnt/public_data/custom model",
        "--dataset-path",
        "data/custom.jsonl",
        "--seed",
        77,
        "--max-num-batched-tokens",
        12345,
        "--include-long",
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads(output.read_text())
    scan = manifest["phases"][0]["commands"]
    assert len(scan) == 18
    assert len({tuple(row["argv"]) for row in scan}) == 18
    for row in scan:
        argv = row["argv"]
        assert argv[argv.index("--seed") + 1] == "77"
        assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.9"
        assert argv[argv.index("--model-path") + 1] == "/mnt/public_data/custom model"
    assert all(phase["requires_inspection"] for phase in manifest["phases"][1:])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["plan.json"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["engine_policy"]["model_config"].update(enforce_eager=True),
        lambda r: r["contract"].update(engine_policy_sha256="f" * 64),
        lambda r: r["contract"].update(dtype="float16"),
        lambda r: r.update(offload_count=False),
    ],
)
def test_native_rejects_forged_policy_and_weight_evidence(tmp_path, mutation):
    raw = native()
    mutation(raw)
    assert invoke("validate", save(tmp_path / "forged.json", raw)).returncode != 0


def test_direct_bootstrap_does_not_import_gpu_package():
    script = (
        "import runpy,sys; sys.argv=['cli.py','--help']; "
        "\ntry: runpy.run_path(sys.argv[1] if False else "
        + repr(str(CLI))
        + ",run_name='__main__')"
        "\nexcept SystemExit as e: assert e.code == 0"
        "\nassert 'torch' not in sys.modules and 'flexmoe' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", script], capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_transport_grid_rejects_missing_duplicate_wrong_contract_and_partial(tmp_path):
    for mutation in (
        lambda x: x["samples"].pop(),
        lambda x: x["samples"].append(x["samples"][0]),
        lambda x: x["contract"].update(commit="a" * 40),
        lambda x: x.update(status="failed", evidence_complete=False),
    ):
        raw = deepcopy(transport_artifact(samples()))
        mutation(raw)
        assert invoke("validate", save(tmp_path / "samples.json", raw)).returncode != 0


def test_transport_modes_and_measurement_contracts_stay_separate(tmp_path):
    from dataclasses import replace

    from flexmoe.analysis.io import transfer_groups

    first = samples()
    second = tuple(replace(r, mode="fragmented", gather_s=0) for r in first)
    third = tuple(
        replace(r, contract={**r.contract, "measurement_id": "second"}) for r in first
    )
    paths = [
        save(tmp_path / "first.json", transport_artifact(first + second)),
        save(tmp_path / "second.json", transport_artifact(third)),
    ]
    groups = transfer_groups(paths)
    assert len(groups) == 3 and all(len(group) == 4 for group in groups)


def test_legacy_oracle_is_validated_and_keeps_missing_identity(tmp_path):
    from test_kv_oracle import fixture

    from flexmoe.analysis.io import oracle_adapter

    raw = fixture("small")
    result = oracle_adapter(raw)
    assert result["timing_point"] is None
    assert result["scope"] == "over-budget-diagnostic-counterfactual"
    assert "legacy-hardware-sha256" in result["missing_evidence"]
    assert "legacy-prompt-hashes" in result["missing_evidence"]
    assert invoke("validate", save(tmp_path / "old.json", raw)).returncode == 0
    raw["repetitions"][0]["generated_tokens"] = 999
    with pytest.raises(ValueError):
        oracle_adapter(raw)


def test_actual_producer_to_stdlib_converter(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace

    import torch
    from test_analysis_runner import AnalysisEngine, supported_config

    from flexmoe.bench.analysis_runner import AnalysisBackend
    from flexmoe.bench.partial_runner import run_benchmark

    monkeypatch.setattr(os, "environ", os.environ.copy())

    class CompleteMemoryEngine(AnalysisEngine):
        def collective_rpc(self, method, kwargs=None):
            result = super().collective_rpc(method, kwargs)
            if method == "fluxmoe_worker_memory_stats":
                for row in result:
                    row.update(
                        torch_peak_allocated_bytes=40500,
                        torch_peak_reserved_bytes=42000,
                    )
            return result

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            LLM=CompleteMemoryEngine,
            SamplingParams=SimpleNamespace,
            __version__="0.10.2",
        ),
    )
    monkeypatch.setenv("FLEXMOE_VLLM_COMMIT", "a" * 40)
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    cfg = supported_config(tmp_path)
    run_benchmark(
        cfg,
        project_root=ROOT,
        run_dir=tmp_path / "native",
        backend=AnalysisBackend(mode="native", safety_reserve_bytes=100),
    )
    result = invoke("validate", tmp_path / "native/summary.json")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("custom", [False, True])
def test_plan_preserves_one_dataset_across_all_length_buckets(tmp_path, custom):
    output = tmp_path / "plan.json"
    path = "benchmarks/data/sharegpt/qwen3next_1024_requests.jsonl.zst"
    flags = []
    if custom:
        path = "data/my existing corpus.jsonl"
        flags = ["--dataset-path", path]
    result = invoke(
        "plan", "--include-long", "--selected-context", 4096, "--output", output, *flags
    )
    assert result.returncode == 0, result.stderr
    raw = json.loads(output.read_text())
    for phase in raw["phases"]:
        for command in phase["commands"]:
            argv = command["argv"]
            if "--dataset-path" in argv:
                assert argv[argv.index("--dataset-path") + 1] == path
                if not custom:
                    assert (ROOT / path).is_file()
                    assert (ROOT / argv[argv.index("--dataset-manifest") + 1]).is_file()


def test_unsupported_system_python_has_explicit_version_error():
    probe = subprocess.run(
        ["python3", "-S", "-c", "import sys; print(sys.version_info < (3,10))"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "True":
        pytest.skip("system Python already supports the project version contract")
    result = subprocess.run(
        ["python3", "-S", str(CLI), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Python >=3.10" in result.stderr and "Traceback" not in result.stderr


def test_plan_custom_settings_reach_every_gpu_phase_and_shell_argv(tmp_path):
    import shlex

    output = tmp_path / "plan.json"
    settings = {
        "--model-path": "/mnt/public_data/custom model",
        "--timeout-s": "999",
        "--warmups": "7",
        "--repetitions": "9",
        "--safety-reserve-bytes": "3000000000",
    }
    flags = [part for pair in settings.items() for part in pair]
    result = invoke("plan", "--output", output, "--include-long", *flags)
    assert result.returncode == 0, result.stderr
    checked_modes = set()
    for phase in json.loads(output.read_text())["phases"]:
        for command in phase["commands"]:
            argv = command["argv"]
            assert shlex.split(command["shell"]) == argv
            if argv[0] != "bash":
                continue
            checked_modes.add(argv[2])
            for flag, expected in settings.items():
                if argv[2] == "trace" and flag == "--repetitions":
                    expected = "1"
                assert argv[argv.index(flag) + 1] == expected, (phase["phase"], flag)
    assert checked_modes == {"resident", "trace", "transport"}
