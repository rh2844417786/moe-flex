import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_kv_oracle import fixture
from test_partial_runner import Engine, config


@pytest.mark.parametrize("unsafe", [False, True])
def test_real_shared_loop_native_r0_and_oracle(tmp_path, monkeypatch, unsafe):
    from flexmoe.bench.kv_oracle_evidence import SafetyRefusal, public_run
    from flexmoe.bench.kv_oracle_runner import OracleBackend, run_point
    from flexmoe.bench.kv_oracle_suite import export_suite
    from flexmoe.bench.partial_runner import run_benchmark

    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setenv("FLEXMOE_VLLM_COMMIT", "c" * 40)
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    engines = []

    class NativeBoundary(Engine):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.role = "small" if kwargs["kv_cache_memory_bytes"] else "r0"
            self.active = False
            self.resets = 0
            self.events = []
            engines.append(self)

        def generate(self, prompts, sampling, **kwargs):
            self.events.append(
                (
                    "generate",
                    self.active,
                    len(prompts),
                    sampling.min_tokens,
                    sampling.max_tokens,
                )
            )
            assert sampling.ignore_eos is True and sampling.detokenize is False
            return super().generate(prompts, sampling, **kwargs)

        def collective_rpc(self, method, kwargs=None):
            self.events.append((method, (kwargs or {}).get("action")))
            if method == "fluxmoe_worker_memory_stats":
                memory = fixture(self.role)["memory"]
                if unsafe and self.role == "small" and self.resets >= 2:
                    memory[2]["torch_peak_reserved_bytes"] = 9800
                return memory
            if method == "fluxmoe_reset_memory_peaks":
                self.resets += 1
                return [0, 1, 2, 3]
            if method == "fluxmoe_native_probe":
                action = kwargs["action"]
                if action == "start":
                    assert not self.active
                    self.active = True
                elif action == "stop":
                    self.active = False
                return fixture(self.role)["native_probe"]
            return super().collective_rpc(method, kwargs)

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            LLM=NativeBoundary, SamplingParams=SimpleNamespace, __version__="0.10.2"
        ),
    )
    cfg = replace(config(tmp_path), timing_samples=0, max_num_seqs=512, repetitions=3)
    run_benchmark(
        cfg,
        project_root=Path.cwd(),
        run_dir=tmp_path / "r0",
        backend=OracleBackend("r0", None, 200, 400, 0.9, 100),
    )
    r0 = json.loads((tmp_path / "r0/summary.json").read_text())

    def execute_small():
        run_point(
            cfg,
            project_root=Path.cwd(),
            run_dir=tmp_path / "small",
            role="small",
            reference=r0,
            small_add_bytes=200,
            large_add_bytes=400,
            safety_fraction=0.9,
            safety_margin_bytes=100,
        )

    if unsafe:
        with pytest.raises(SafetyRefusal):
            execute_small()
        raw = json.loads((tmp_path / "small/summary.json").read_text())
        assert raw["status"] == "failed"
        assert raw["repetitions_completed"] == 1
        rejected = raw["failed_measurement"]
        assert rejected["measurement_status"] == "rejected"
        assert rejected["repetition"] == 1
        assert rejected["generated_tokens"] == 15 and rejected["elapsed_s"] > 0
        assert rejected["memory"][2]["torch_peak_reserved_bytes"] == 9800
        assert len(rejected["native_probe"]) == 4
        assert (tmp_path / "small/failed-rep-001.json").is_file()
        clean = public_run(raw)
        assert (
            clean["failed_measurement"]["memory"][2]["torch_peak_reserved_bytes"]
            == 9800
        )
        (tmp_path / "suite.json").write_text(
            json.dumps(
                {"roles": {r: r for r in ("r0", "small", "large")}, "order": "forward"}
            )
        )
        result = export_suite(tmp_path, tmp_path / "export")
        assert (
            result["runs"][1]["failed_measurement"]["memory"][2][
                "torch_peak_reserved_bytes"
            ]
            == 9800
        )
        assert result["comparison"]["decision"] == "insufficient-evidence"
        return
    execute_small()
    small = json.loads((tmp_path / "small/summary.json").read_text())
    assert small["requested_kv_cache_bytes"] == 1200
    assert small["contract"] == r0["contract"]
    assert small["contract"]["repeated_request_count"] == 3
    assert small["smoke_matches_resident"] is True
    assert small["diagnostic_only"] is True and small["formal_offload_gain"] is False
    assert small["repetitions_completed"] == 3
    assert "requested_gpu_bytes_by_rank" not in small["hardware_budget"]
    for engine, row in zip(engines, (r0, small), strict=True):
        assert engine.resets == 3
        assert [event[1] for event in engine.events if event[0] == "generate"] == [
            False,
            False,
            True,
            True,
            True,
        ]
        assert all(
            rep["generated_tokens"] == 15 and rep["elapsed_s"] > 0
            for rep in row["repetitions"]
        )
        assert all(
            len(rep["memory"]) == len(rep["native_probe"]) == 4
            for rep in row["repetitions"]
        )
        assert engine.arguments["cpu_offload_gb"] == 0
        assert engine.arguments["compilation_config"]["level"] == 0
