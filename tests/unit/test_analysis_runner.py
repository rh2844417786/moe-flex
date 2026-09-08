import gzip
import importlib
import json
import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_partial_runner import Engine, config


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())


def module():
    return importlib.import_module("flexmoe.bench.analysis_runner")


def test_native_policy_preserves_optimization_and_old_default(tmp_path):
    from flexmoe.bench.partial_runner import BenchmarkBackend

    cfg = config(tmp_path)
    backend = module().AnalysisBackend(mode="native")
    args = backend.engine_arguments(cfg, None)
    assert args["enforce_eager"] is False
    assert args["disable_custom_all_reduce"] is False
    assert "compilation_config" not in args
    engine = Engine(**args)
    resolved = engine.llm_engine.vllm_config
    resolved.model_config.enforce_eager = False
    resolved.model_config.quantization = None
    resolved.cache_config.cpu_offload_gb = 0
    resolved.cache_config.swap_space_bytes = 0
    resolved.parallel_config.disable_custom_all_reduce = False
    resolved.compilation_config.level = 3
    resolved.compilation_config.cudagraph_mode = "FULL_AND_PIECEWISE"
    assert backend.resolved_policy(engine, args)["compilation_config"]["level"] == 3
    with pytest.raises(RuntimeError):
        BenchmarkBackend().resolved_policy(engine, args)
    with pytest.raises(ValueError, match="KV"):
        backend.engine_arguments(cfg, 123)


def test_analysis_workload_excludes_prompts_before_repeating_and_packs_explicitly(
    tmp_path,
):
    cfg = config(tmp_path)
    backend = module().AnalysisBackend(mode="native", selection_offset=1)
    w = backend.workload(cfg)
    assert w.prompts[0] == (21, 22)
    assert w.metadata["repeated_request_count"] == 3
    exclusion = {module().digest_json((11, 12))}
    other = module().select_workload(cfg, excluded_hashes=exclusion)
    assert other.prompts == ((21, 22),) * 5
    assert other.metadata["unique_selected_request_count"] == 1
    assert other.metadata["repeated_request_count"] == 4
    with pytest.raises(ValueError, match="empty"):
        module().select_workload(cfg, excluded_hashes=set(w.metadata["prompt_hashes"]))
    with pytest.raises(ValueError, match="synthetic"):
        module().select_workload(replace(cfg, context_length=4))
    packed = module().select_workload(
        replace(cfg, context_length=4), synthetic_context=True
    )
    assert packed.prompts[0] == (11, 12, 21, 22)
    assert packed.metadata["synthetic"] is True


def test_hardware_hash_uses_actual_uuid_capacity_order_and_unavailable():
    devices = [SimpleNamespace(uuid=f"uuid-{i}", total_memory=80_000) for i in range(4)]
    first = module().hardware_identity(devices)
    assert (
        first["hardware_sha256"]
        != module().hardware_identity(devices[::-1])["hardware_sha256"]
    )
    devices[2].uuid = None
    assert module().hardware_identity(devices)["hardware_sha256"] is None


def test_trace_requires_single_window_and_clears_environment(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    backend = module().AnalysisBackend(mode="trace")
    with pytest.raises(ValueError, match="repetitions=1"):
        backend.workload(cfg)
    monkeypatch.setenv("FLUXMOE_ANALYSIS_TRACE", "1")
    module().AnalysisBackend(mode="native").configure(cfg, tmp_path, ())
    assert "FLUXMOE_ANALYSIS_TRACE" not in __import__("os").environ


def supported_config(tmp_path, repetitions=1):
    cfg = replace(config(tmp_path), repetitions=repetitions)
    (cfg.model_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_next",
                "num_hidden_layers": 2,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "hidden_size": 16,
                "moe_intermediate_size": 8,
            }
        )
    )
    return cfg


class AnalysisEngine(Engine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = False
        cfg = self.llm_engine.vllm_config
        cfg.model_config.enforce_eager = kwargs["enforce_eager"]
        cfg.model_config.quantization = None
        cfg.cache_config.cpu_offload_gb = 0
        cfg.cache_config.swap_space_bytes = 0
        cfg.parallel_config.disable_custom_all_reduce = kwargs[
            "disable_custom_all_reduce"
        ]
        cfg.compilation_config.cudagraph_mode = (
            "NONE" if kwargs["enforce_eager"] else "FULL_AND_PIECEWISE"
        )
        cfg.compilation_config.level = 0 if kwargs["enforce_eager"] else 3

    def collective_rpc(self, method, kwargs=None):
        if method == "fluxmoe_analysis_device":
            return [
                {"rank": r, "uuid": f"gpu-{r}", "total_memory": 80000} for r in range(4)
            ]
        if method == "fluxmoe_reset_memory_peaks":
            return list(range(4))
        if method == "fluxmoe_analysis_trace":
            if kwargs["action"] == "start":
                self.capture = True
                self.capture_start_calls = self.calls
                return [{"rank": r} for r in range(4)]
            self.capture = False
            return [
                {
                    "rank": r,
                    "full_workload": True,
                    "observed_steps": 1,
                    "captured_steps": 1,
                    "status": "complete",
                    "events": [
                        {
                            "step": 0,
                            "layer": layer,
                            "token_rows": 5,
                            "requests": 5,
                            "phase": "decode",
                            "experts": [0, 1],
                        }
                        for layer in range(2)
                    ],
                }
                for r in range(4)
            ]
        return super().collective_rpc(method, kwargs)


@pytest.mark.parametrize("mode", ["native", "eager", "trace"])
def test_shared_analysis_runner_writes_valid_contract_and_capture(
    tmp_path, monkeypatch, mode
):
    from flexmoe.analysis.schema import (
        DemandTrace,
        parse_diagnostic_artifact,
        validate_contract,
    )
    from flexmoe.bench.partial_runner import run_benchmark

    cfg = supported_config(tmp_path)
    engines = []

    def create_engine(**kwargs):
        engine = AnalysisEngine(**kwargs)
        engines.append(engine)
        return engine

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            LLM=create_engine, SamplingParams=SimpleNamespace, __version__="0.10.2"
        ),
    )
    monkeypatch.setenv("FLEXMOE_VLLM_COMMIT", "a" * 40)
    import torch

    def no_frontend_cuda(*args):
        raise AssertionError("frontend CUDA properties query")

    monkeypatch.setattr(torch.cuda, "get_device_properties", no_frontend_cuda)

    monkeypatch.setattr(torch.version, "cuda", "12.8")
    run = tmp_path / mode
    backend = module().AnalysisBackend(mode=mode, safety_reserve_bytes=100)
    run_benchmark(
        cfg, project_root=__import__("pathlib").Path.cwd(), run_dir=run, backend=backend
    )
    summary = parse_diagnostic_artifact(
        json.loads((run / "summary.json").read_text()), "native-summary"
    )
    assert summary["status"] == "complete"
    assert summary["contract"]["hardware_sha256"] is not None
    assert json.loads((run / "smoke.json").read_text())["diagnostic_only"] is True
    assert (
        json.loads((run / "rep-000.json").read_text())["deployment_gain_proven"]
        is False
    )
    assert summary["timing_eligible"] == (mode != "trace")
    assert (
        validate_contract(summary["contract"], require_prompt_hashes=True)[
            "repeated_request_count"
        ]
        == 3
    )
    assert summary["geometry"]["expert_bytes"] == 192
    assert summary["repetitions"][0]["memory"][0]["kv_cache_allocated_bytes"] == 1000
    if mode == "trace":
        assert engines[0].capture_start_calls == 2  # smoke plus warmup
        with gzip.open(run / "trace-rank-0.json.gz", "rt") as stream:
            saved = parse_diagnostic_artifact(json.load(stream), "demand-trace")
        assert DemandTrace.from_dict(saved["trace"]).generated_tokens == 15
        assert saved["generation_status"] == "complete"
        assert not engines[0].capture


@pytest.mark.parametrize("drop_after_calls", [2, 3])
def test_shared_runner_reserve_drop_preserves_failed_sample_and_export(
    tmp_path, monkeypatch, drop_after_calls
):
    import torch
    from test_analysis_cli import ROOT, invoke

    from flexmoe.bench.partial_runner import run_benchmark

    engines = []

    class ReserveDropEngine(AnalysisEngine):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            engines.append(self)

        def collective_rpc(self, method, kwargs=None):
            result = super().collective_rpc(method, kwargs)
            if method == "fluxmoe_worker_memory_stats":
                for row in result:
                    row.update(
                        torch_peak_allocated_bytes=40500,
                        torch_peak_reserved_bytes=42000,
                    )
                    if self.calls >= drop_after_calls:
                        row.update(
                            free_gpu_bytes=50,
                            torch_reserved_bytes=79950,
                            torch_peak_reserved_bytes=79950,
                        )
            return result

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            LLM=ReserveDropEngine, SamplingParams=SimpleNamespace, __version__="0.10.2"
        ),
    )
    monkeypatch.setenv("FLEXMOE_VLLM_COMMIT", "a" * 40)
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    run = tmp_path / "native-failed"
    with pytest.raises(RuntimeError, match="physical GPU safety reserve"):
        run_benchmark(
            supported_config(tmp_path),
            project_root=ROOT,
            run_dir=run,
            backend=module().AnalysisBackend(safety_reserve_bytes=100),
        )
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["repetitions_completed"] == 0
    assert engines[0].calls == drop_after_calls
    assert not (run / "rep-000.json").exists()
    if drop_after_calls == 2:
        # The pre-measurement gate still refuses to run the measured generation.
        assert summary.get("failed_measurement") is None
        assert not (run / "failed-rep-000.json").exists()
        return

    failed = summary.get("failed_measurement")
    assert isinstance(failed, dict)
    assert failed["measurement_status"] == "rejected"
    assert failed["generated_tokens"] == 15
    assert failed["request_count"] == 5
    assert failed["elapsed_s"] > 0
    assert failed["output_tokens_per_second"] == 15 / failed["elapsed_s"]
    assert failed == json.loads((run / "failed-rep-000.json").read_text())
    assert [row["free_gpu_bytes"] for row in failed["memory"]] == [50] * 4
    assert [row["torch_peak_reserved_bytes"] for row in failed["memory"]] == [79950] * 4

    public = tmp_path / "public"
    exported = invoke("export", "--source", run, "--output", public)
    assert exported.returncode == 0, exported.stderr
    records = json.loads((public / "report.json").read_text())["records"]
    saved_summary = next(
        row for row in records if row["artifact_kind"] == "native-summary"
    )
    saved_failed = next(
        row for row in records if row["artifact_kind"] == "analysis-repetition"
    )
    assert saved_summary["status"] == "failed"
    for sample in (saved_summary["failed_measurement"], saved_failed):
        assert sample["generated_tokens"] == 15
        assert sample["elapsed_s"] == failed["elapsed_s"]
        assert [row["torch_peak_reserved_bytes"] for row in sample["memory"]] == [
            79950
        ] * 4
    for path in public.iterdir():
        rendered = path.read_text()
        assert "79950" in rendered
        assert str(tmp_path) not in rendered and "gpu-0" not in rendered


def test_no_complete_capture_saves_null_and_generation_failure_cleanup(tmp_path):
    backend = module().AnalysisBackend(mode="trace")
    backend.run_dir = tmp_path
    backend.summary = {}
    rows = [
        {
            "rank": r,
            "captured_steps": 0,
            "observed_steps": 2,
            "events": [],
            "full_workload": False,
            "status": "incomplete",
        }
        for r in range(4)
    ]
    backend._save_capture(rows, "failed")
    with gzip.open(tmp_path / "trace-rank-0.json.gz", "rt") as stream:
        saved = json.load(stream)
    assert saved["trace"] is None
    assert saved["generation_status"] == "failed"
    assert saved["missing_evidence"] == ["no-complete-step-layer-grid"]


def test_duplicate_source_rows_are_disclosed_exactly(tmp_path):
    from flexmoe.datasets.sharegpt import PromptRecord, write_jsonl_zst

    cfg = config(tmp_path)
    digest = write_jsonl_zst(
        [
            PromptRecord("a", (11, 12), 2),
            PromptRecord("b", (11, 12), 2),
            PromptRecord("c", (21, 22), 2),
        ],
        cfg.dataset_path,
    )
    cfg.dataset_manifest.write_text(
        json.dumps({"sha256": digest, "record_count": 3, "counts_by_context": {"2": 3}})
    )
    selected = module().select_workload(cfg)
    assert selected.metadata["unique_selected_request_count"] == 2
    assert selected.metadata["repeated_request_count"] == 3
    assert selected.metadata["evaluation_pool_count"] == 2


def test_analysis_cli_passes_all_workload_and_capture_flags(tmp_path, monkeypatch):
    observed = []
    monkeypatch.setattr(
        module(), "run_benchmark", lambda cfg, **kwargs: observed.append((cfg, kwargs))
    )
    assert (
        module().main(
            [
                "--mode",
                "trace",
                "--project-root",
                str(tmp_path),
                "--run-dir",
                str(tmp_path / "trace"),
                "--repetitions",
                "1",
                "--batch-size",
                "17",
                "--context-length",
                "16384",
                "--output-length",
                "512",
                "--max-num-seqs",
                "17",
                "--max-num-batched-tokens",
                "8192",
                "--seed",
                "12",
                "--synthetic-context",
                "--selection-offset",
                "32",
                "--trace-budget-bytes",
                "8192",
                "--max-trace-steps",
                "31",
                "--safety-reserve-bytes",
                "1234",
                "--gpu-memory-utilization",
                "0.93",
            ]
        )
        == 0
    )
    cfg, kw = observed[0]
    assert (
        cfg.batch_size,
        cfg.context_length,
        cfg.seed,
        cfg.gpu_memory_utilization,
    ) == (17, 16384, 12, 0.93)
    assert kw["backend"].selection_offset == 32
    assert kw["backend"].trace_budget_bytes == 8192
    assert kw["backend"].synthetic_context is True
