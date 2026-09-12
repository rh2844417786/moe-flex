import gzip
import importlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_analysis_runner import AnalysisEngine, supported_config
from test_decode_corpus import write_dataset
from test_decode_trace import PreparedRunner


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())


def module():
    return importlib.import_module("flexmoe.bench.decode_mechanism_runner")


def config(tmp_path):
    cfg = supported_config(tmp_path)
    (cfg.model_path / "model.safetensors.index.json").write_text("{}")
    data, manifest = write_dataset(tmp_path, [(i, i + 1) for i in range(40)])
    return replace(cfg, dataset_path=data, dataset_manifest=manifest, timing_samples=0)


class DecodeEngine(AnalysisEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert kwargs["disable_log_stats"] is False
        assert "stat_loggers" not in kwargs
        self.llm_engine.stat_logger = SimpleNamespace(record=lambda **kw: None)
        self.runners = [PreparedRunner() for _ in range(4)]
        self.captures = []
        self.start_calls = []

    def collective_rpc(self, method, kwargs=None):
        if method == "fluxmoe_decode_mechanism":
            from flexmoe.vllm.decode_trace import DecodeTraceCollector

            if kwargs["action"] == "start":
                self.start_calls.append(self.calls)
                self.captures = [
                    DecodeTraceCollector(
                        runner,
                        device="cpu",
                        **{k: v for k, v in kwargs.items() if k != "action"},
                    )
                    for runner in self.runners
                ]
                return [
                    {"rank": rank, **capture.start()}
                    for rank, capture in enumerate(self.captures)
                ]
            return [
                {"rank": rank, **capture.stop()}
                for rank, capture in enumerate(self.captures)
            ]
        result = super().collective_rpc(method, kwargs)
        if method == "fluxmoe_worker_memory_stats":
            for row in result:
                row.update(
                    available_kv_cache_bytes=self.arguments["kv_cache_memory_bytes"]
                    or 1000,
                    num_gpu_blocks=10,
                    torch_peak_reserved_bytes=41000,
                    torch_reserved_bytes=41000,
                )
        return result

    def generate(self, prompts, sampling, **kwargs):
        outputs = super().generate(prompts, sampling, **kwargs)
        # A capacity-limited scheduler produces batch 2 even with five submitted requests.
        for _ in range(3):
            for rank, runner in enumerate(self.runners):
                runner.execute_model(batch=2)
                if self.captures:
                    for layer in range(2):
                        self.captures[rank].record(
                            layer, torch.tensor([[0, 1], [0, 2]])
                        )
            self.llm_engine.stat_logger.record(
                scheduler_stats=SimpleNamespace(
                    kv_cache_usage=0.7,
                    num_running_reqs=2,
                    num_waiting_reqs=3,
                    step_counter=1,
                ),
                iteration_stats=SimpleNamespace(num_preempted_reqs=1),
                engine_idx=0,
            )
        return outputs


def install(monkeypatch, engine_class=DecodeEngine):
    engines = []

    def create(**kwargs):
        engine = engine_class(**kwargs)
        engines.append(engine)
        return engine

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            LLM=create, SamplingParams=SimpleNamespace, __version__="0.10.2"
        ),
    )
    monkeypatch.setenv("FLEXMOE_VLLM_COMMIT", "a" * 40)
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    return engines


def test_explicit_kv_only_new_adapter_and_native_profile_rejected(tmp_path):
    from flexmoe.bench.partial_runner import engine_arguments

    cfg = config(tmp_path)
    for mode in ("native", "matched-resident"):
        backend = module().DecodeMechanismBackend(cfg, mode=mode, kv_bytes=1050)
        args = backend.engine_arguments(cfg, None)
        assert args["kv_cache_memory_bytes"] == 1050
        assert args["enforce_eager"] == (mode != "native")
        assert args["max_num_seqs"] == cfg.max_num_seqs
    with pytest.raises(ValueError, match="KV"):
        engine_arguments(cfg, 1050)
    with pytest.raises(ValueError, match="native.*profile"):
        module().DecodeMechanismBackend(cfg, mode="native", profile=True)


@pytest.mark.parametrize(
    "mode,profile",
    [("native", False), ("matched-resident", False), ("matched-resident", True)],
)
def test_shared_runner_unique_inputs_kv_real_batch_scoped_stats_and_saved_labels(
    tmp_path, monkeypatch, mode, profile
):
    from flexmoe.bench.partial_runner import run_benchmark

    cfg = config(tmp_path)
    engines = install(monkeypatch)
    backend = module().DecodeMechanismBackend(
        cfg,
        mode=mode,
        profile=profile,
        kv_bytes=1050,
        target_batch=5 if profile else None,
        capture_steps=2,
        min_capture_steps=1,
        safety_reserve_bytes=100,
    )
    run = tmp_path / "decode-run"
    run_benchmark(cfg, project_root=Path.cwd(), run_dir=run, backend=backend)
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["artifact_kind"] == "decode-run"
    assert summary["mode"] == mode
    assert summary["evidence_kind"] == ("instrumented" if profile else "measured")
    assert summary["contract"]["repeated_request_count"] == 0
    assert len(summary["contract"]["selected_input_hashes"]) == 5
    assert summary["actual_kv"]["rounding_bytes"] == 50
    rep = summary["repetitions"][0]
    assert rep["artifact_kind"] == "decode-repetition"
    assert rep["generation_status"] == "complete"
    assert rep["timing_eligible"] is (not profile)
    assert rep["worker_observations"][0]["decode_batch_step_counts"] == {"2": 3}
    assert rep["worker_observations"][0]["coverage_status"] == (
        "unreached" if profile else "not-requested"
    )
    assert rep["scheduler"]["kv_cache_usage"]["peak"] == 0.7
    assert rep["scheduler"]["preemptions"]["total"] == 3
    assert rep["cpu_memory"]["peak_rss_bytes"] > 0
    assert engines[0].start_calls == [2]
    assert all(not c.active for c in engines[0].captures)
    assert (
        json.loads((run / "smoke.json").read_text())["artifact_kind"] == "decode-smoke"
    )
    with gzip.open(run / "decode-rep-000-rank-0.json.gz", "rt") as stream:
        saved = json.load(stream)
    assert saved["artifact_kind"] == "decode-profile"
    assert saved["evidence_kind"] == ("instrumented" if profile else "measured")
    assert saved["observation"]["activation_rows"] == []


def test_duplicate_or_insufficient_inputs_rejected_before_engine(tmp_path):
    cfg = config(tmp_path)
    backend = module().DecodeMechanismBackend(cfg)
    with pytest.raises(ValueError, match="no-wrap"):
        backend.workload(replace(cfg, batch_size=9))


@pytest.mark.parametrize("mutation", ["rank", "requested", "rounding"])
def test_kv_measurement_mismatch_fails_closed(tmp_path, mutation):
    cfg = config(tmp_path)
    backend = module().DecodeMechanismBackend(cfg, kv_bytes=1050)
    rows = [
        {
            "rank": rank,
            "kv_cache_allocated_bytes": 1000,
            "kv_cache_declared_bytes": 1000,
            "num_gpu_blocks": 10,
            "available_kv_cache_bytes": 1050,
            "kv_cache_accounting_consistent": True,
        }
        for rank in range(4)
    ]
    if mutation == "rank":
        rows[2].update(kv_cache_allocated_bytes=900, kv_cache_declared_bytes=900)
    elif mutation == "requested":
        rows[2]["available_kv_cache_bytes"] = 1000
    else:
        for row in rows:
            row.update(kv_cache_allocated_bytes=800, kv_cache_declared_bytes=800)
    with pytest.raises(ValueError, match="KV"):
        backend.validate_kv(rows)


def test_mid_generation_failure_saves_partial_capture_and_smoke(tmp_path, monkeypatch):
    from flexmoe.bench.partial_runner import run_benchmark

    class Failure(DecodeEngine):
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            if self.calls == 3:
                raise RuntimeError("mid-generation")
            return result

    cfg = config(tmp_path)
    engines = install(monkeypatch, Failure)
    run = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="mid-generation"):
        run_benchmark(
            cfg,
            project_root=Path.cwd(),
            run_dir=run,
            backend=module().DecodeMechanismBackend(
                cfg,
                mode="matched-resident",
                profile=True,
                min_capture_steps=1,
                capture_steps=2,
                safety_reserve_bytes=100,
            ),
        )
    assert json.loads((run / "summary.json").read_text())["status"] == "failed"
    assert (run / "smoke.json").is_file()
    with gzip.open(run / "decode-rep-000-rank-0.json.gz", "rt") as stream:
        saved = json.load(stream)
    assert saved["generation_status"] == "failed"
    assert len(saved["observation"]["activation_rows"]) == 4
    assert not engines[0].captures[0].active


def test_post_measurement_physical_drop_preserves_numeric_rejected_sample(
    tmp_path, monkeypatch
):
    from flexmoe.bench.partial_runner import run_benchmark

    class Drop(DecodeEngine):
        def collective_rpc(self, method, kwargs=None):
            rows = super().collective_rpc(method, kwargs)
            if method == "fluxmoe_worker_memory_stats" and self.calls == 3:
                for row in rows:
                    row["free_gpu_bytes"] = 50
            return rows

    cfg = config(tmp_path)
    install(monkeypatch, Drop)
    run = tmp_path / "failed-memory"
    with pytest.raises(RuntimeError, match="physical GPU safety reserve"):
        run_benchmark(
            cfg,
            project_root=Path.cwd(),
            run_dir=run,
            backend=module().DecodeMechanismBackend(cfg, safety_reserve_bytes=100),
        )
    failed = json.loads((run / "failed-rep-000.json").read_text())
    assert failed["generated_tokens"] == 15
    assert failed["elapsed_s"] > 0
    assert failed["measurement_status"] == "rejected"
    assert failed["memory"][0]["free_gpu_bytes"] == 50
    assert failed["artifact_kind"] == "decode-repetition"


def test_cli_passes_explicit_new_parameters_to_shared_runner(tmp_path, monkeypatch):
    got = []
    monkeypatch.setattr(
        module().shared,
        "run_benchmark",
        lambda cfg, **kwargs: got.append((cfg, kwargs)),
    )
    module().main(
        [
            "--mode",
            "offload",
            "--profile",
            "--project-root",
            str(tmp_path),
            "--run-dir",
            str(tmp_path / "cli"),
            "--profile-path",
            "profile.json",
            "--kv-bytes",
            "123456",
            "--batch-size",
            "1000",
            "--target-batch",
            "500",
            "--max-num-seqs",
            "1024",
            "--capture-steps",
            "128",
            "--min-capture-steps",
            "64",
            "--resident-ratio",
            "0.9",
            "--cache-slots",
            "2048",
        ]
    )
    cfg, kwargs = got[0]
    backend = kwargs["backend"]
    assert cfg.arm == "partial-auto-kv" and cfg.batch_size == 1000
    assert cfg.timing_samples == 0 and cfg.max_num_seqs == 1024
    assert backend.kv_bytes == 123456 and backend.target_batch == 500
    assert (
        backend.profile
        and backend.resident_ratio == 0.9
        and backend.cache_slots == 2048
    )


@pytest.mark.parametrize("profile", [False, True])
def test_shared_offload_runner_uses_real_registry_pool_and_heldout_profile(
    tmp_path, monkeypatch, profile
):
    from test_expert_cache_runtime import TensorBackend

    from flexmoe.bench import expert_cache_evidence as evidence
    from flexmoe.bench.partial_runner import digest_json, run_benchmark
    from flexmoe.datasets.decode_corpus import load_unique_workload
    from flexmoe.runtime.expert_pool import PoolProfileObserver
    from flexmoe.runtime.expert_profile import ExpertProfile
    from flexmoe.vllm import expert_cache
    from flexmoe.vllm.expert_calibration import model_profile_identity

    cfg = replace(config(tmp_path), arm="partial-auto-kv")
    identity = model_profile_identity(cfg.model_path, 4)
    _, _, metadata = load_unique_workload(
        cfg.dataset_path, cfg.dataset_manifest, context_length=2, request_count=5
    )
    heat = ExpertProfile.from_dict(
        {
            "schema_version": 1,
            **identity,
            "calibration_input_hashes": metadata["calibration_input_hashes"],
            "counts": [[1] * 4] * 2,
            "forward_counts": [1, 1],
        }
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(heat.to_dict()))

    class CPUBackend(TensorBackend):
        def timing(self, reset):
            return {
                "cuda_sample_count": 0,
                "load_cuda_s": 0.0,
                "compute_cuda_s": 0.0,
                "promotion_cuda_s": 0.0,
            }

    def kernel(hidden, w13, w2, weights, ids, mapping, **kwargs):
        assert w13.shape[0] == 5  # shared pool capacity, not logical E
        for expert in ids.unique().tolist():
            assert torch.all(w13[int(mapping[expert])] == expert + 1)
            assert torch.all(w2[int(mapping[expert])] == expert + 1)
        kwargs["config_recorder"].record(
            hidden.shape[0], (4, 4, 16), (4, 16, 2), 2, {"BLOCK_SIZE_M": 16}
        )
        return hidden + 1

    monkeypatch.setattr(expert_cache, "logical_fused_experts", kernel)

    class OffloadEngine(DecodeEngine):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            assert os.environ["FLUXMOE_ENABLE"] == "1"
            assert os.environ["FLUXMOE_STORAGE_MODE"] == "expert-cache"
            self.registries = []
            for rank in range(4):
                registry = expert_cache.ExpertCacheRegistry(
                    identity=identity,
                    profile=heat,
                    tp_rank=rank,
                    resident_ratio=0.0,
                    cache_slots=1,
                    backend=CPUBackend(),
                    policy="lru",
                )
                for layer in range(2):
                    name = f"model.layers.{layer}.mlp.experts"
                    params = registry.register_layer(
                        name, (4, 4, 16), (4, 16, 2), torch.bfloat16
                    )
                    for expert in range(4):
                        full = torch.full((8, 16), expert + 1, dtype=torch.bfloat16)
                        for shard, param, tensor in (
                            ("w1", params[0], full),
                            ("w3", params[0], full),
                            ("w2", params[1], full.T.contiguous()),
                        ):
                            registry.ingest(
                                param=param,
                                loaded_weight=tensor,
                                weight_name=f"{name}.{expert}.{shard}.weight",
                                shard_id=shard,
                                expert_id=expert,
                            )
                registry.start()
                self.registries.append(registry)

        def collective_rpc(self, method, kwargs=None):
            if method == "fluxmoe_expert_cache_stats":
                return [r.stats(**(kwargs or {})) for r in self.registries]
            result = super().collective_rpc(method, kwargs)
            if method == "fluxmoe_decode_mechanism" and profile:
                for rank, capture in enumerate(self.captures):
                    pool = self.registries[rank].pool
                    if kwargs["action"] == "start":
                        pool.profile_observer = PoolProfileObserver(
                            capture.pool_context, capacity=4, device="cpu"
                        )
                    else:
                        result[rank]["pool_profile"] = pool.profile_observer.snapshot()
                        pool.profile_observer = None
            return result

        def generate(self, prompts, sampling, **kwargs):
            from test_partial_runner import Engine

            outputs = Engine.generate(self, prompts, sampling, **kwargs)
            for _ in range(3):
                ids = torch.tensor([[0, 1], [0, 2]], dtype=torch.int32)
                for rank, runner in enumerate(self.runners):
                    runner.execute_model(batch=2)
                    for layer in range(2):
                        if self.captures:
                            self.captures[rank].record(layer, ids)
                        result = self.registries[rank].forward(
                            f"model.layers.{layer}.mlp.experts",
                            torch.zeros(2, 16, dtype=torch.bfloat16),
                            torch.ones(2, 2),
                            ids,
                            activation="silu",
                            apply_router_weight_on_input=False,
                        )
                        assert torch.equal(result, torch.ones_like(result))
                self.llm_engine.stat_logger.record(
                    scheduler_stats=SimpleNamespace(
                        kv_cache_usage=0.7, num_running_reqs=2, num_waiting_reqs=3
                    ),
                    iteration_stats=SimpleNamespace(num_preempted_reqs=0),
                )
            return outputs

    engines = install(monkeypatch, OffloadEngine)
    run = tmp_path / "offload"
    backend = module().DecodeMechanismBackend(
        cfg,
        mode="offload",
        profile=profile,
        profile_path=profile_path,
        resident_ratio=0.0,
        cache_slots=1,
        cache_policy="lru",
        kv_bytes=1050,
        capture_steps=2,
        min_capture_steps=1,
        safety_reserve_bytes=100,
    )
    run_benchmark(cfg, project_root=Path.cwd(), run_dir=run, backend=backend)
    saved = json.loads((run / "summary.json").read_text())
    assert saved["status"] == "complete" and saved["mechanism_validated"] is True
    assert saved["contract"]["expert_cache"]["profile_sha256"] == digest_json(
        heat.to_dict()
    )
    evidence.validate_stats(saved["expert_cache_stats"], backend.settings, 4)
    assert saved["repetitions"][0]["diagnostics"]["h2d_bytes"] > 0
    assert all(r.pool._loaded_history for r in engines[0].registries)
    if profile:
        with gzip.open(run / "decode-rep-000-rank-0.json.gz", "rt") as stream:
            capture = json.load(stream)["observation"]
        assert (
            len(capture["pool_profile"]["rows"]) == len(capture["activation_rows"]) == 4
        )
        assert all(
            row["first_loads"] == 0 and row["reloads"] > 0
            for row in capture["pool_profile"]["rows"]
        )
