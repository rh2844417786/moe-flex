from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from flexmoe.datasets.sharegpt import PromptRecord, write_jsonl_zst


@pytest.fixture(autouse=True)
def isolated_process_environment(monkeypatch):
    # A real point runs in its own process. Reproduce that isolation when the
    # GPU boundary is replaced inside this shared pytest process.
    monkeypatch.setattr(os, "environ", os.environ.copy())


def module():
    return importlib.import_module("flexmoe.bench.partial_runner")


def dataset(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "input.zst"
    digest = write_jsonl_zst(
        [PromptRecord("a", (11, 12), 2), PromptRecord("b", (21, 22), 2)], source
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"sha256": digest, "record_count": 2, "counts_by_context": {"2": 2}})
    )
    return source, manifest


def config(tmp_path: Path, **kwargs):
    source, manifest = dataset(tmp_path)
    model = tmp_path / "weights"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(json.dumps({"num_hidden_layers": 8}))
    return module().PartialRunConfig(
        arm="resident",
        model_path=model,
        dataset_path=source,
        dataset_manifest=manifest,
        batch_size=5,
        context_length=2,
        output_length=3,
        tensor_parallel_size=4,
        warmups=1,
        repetitions=2,
        **kwargs,
    )


def test_workload_repeats_committed_tokens_and_records_provenance(tmp_path: Path):
    source, manifest = dataset(tmp_path)
    workload = module().load_partial_workload(source, manifest, 5, 2)
    assert workload.prompts == ((11, 12), (21, 22), (11, 12), (21, 22), (11, 12))
    assert workload.metadata["source_request_count"] == 2
    assert workload.metadata["repeated_request_count"] == 3
    assert workload.metadata["sampling_policy"] == "repeated-existing-prompts"
    assert len(workload.metadata["input_sha256"]) == 64
    assert (
        workload.metadata["dataset_manifest_sha256"]
        != workload.metadata["dataset_sha256"]
    )


def test_engine_policy_keeps_all_arms_comparable_and_enables_resident_memory_rpc(
    tmp_path: Path,
):
    cfg = config(tmp_path)
    resident = module().engine_arguments(cfg, None)
    fixed = module().engine_arguments(
        replace(cfg, arm="partial-fixed-kv", offload_count=4), 1234
    )
    auto = module().engine_arguments(
        replace(cfg, arm="partial-auto-kv", offload_count=4), None
    )
    assert resident["enable_prefix_caching"] is False
    assert resident["enforce_eager"] is True
    assert resident["disable_custom_all_reduce"] is True
    assert (
        resident["worker_extension_cls"] == "flexmoe.vllm.bridge.FluxMoEWorkerExtension"
    )
    assert resident["max_model_len"] == 5
    assert resident["dtype"] == "bfloat16"
    assert resident["compilation_config"] == {"level": 0, "custom_ops": ["all"]}
    assert resident["data_parallel_size"] == 1
    assert resident["pipeline_parallel_size"] == 1
    assert resident == auto
    assert {
        key: value for key, value in fixed.items() if key != "kv_cache_memory_bytes"
    } == {
        key: value for key, value in resident.items() if key != "kv_cache_memory_bytes"
    }
    assert fixed["kv_cache_memory_bytes"] == 1234
    with pytest.raises(ValueError, match="KV"):
        module().engine_arguments(
            replace(cfg, arm="partial-fixed-kv", offload_count=4), None
        )


def test_missing_v1_latency_is_unavailable_and_fixed_output_is_enforced():
    outputs = [
        SimpleNamespace(outputs=[SimpleNamespace(token_ids=[8, 9])], metrics=None)
    ]
    summary = module().summarize_outputs(
        outputs, request_count=1, output_length=2, elapsed_s=0.5
    )
    assert summary["output_tokens_per_second"] == 4.0
    assert summary["latency_available_requests"] == 0
    assert summary["ttft_median_s"] is None
    assert summary["request_latency_median_s"] is None
    assert "output_token_ids" not in summary
    with pytest.raises(RuntimeError, match="fixed-output"):
        module().summarize_outputs(
            outputs, request_count=1, output_length=3, elapsed_s=0.5
        )


def test_worker_delta_excludes_warmup_and_rejects_missing_workers():
    before = [
        {
            "rank": rank,
            "h2d_bytes": 100,
            "copy_launches": 4,
            "offload_forwards": 2,
            "resident_forwards": 6,
            "timing": {
                "sample_count": 1,
                "load_cuda_s": 0.3,
                "wait_cuda_s": 0.2,
                "compute_cuda_s": 0.1,
                "cpu_enqueue_s": 0.02,
            },
        }
        for rank in range(4)
    ]
    after = [
        {
            **row,
            "h2d_bytes": 140,
            "copy_launches": 8,
            "offload_forwards": 4,
            "resident_forwards": 12,
            "timing": {
                "sample_count": 2,
                "load_cuda_s": 0.5,
                "wait_cuda_s": 0.3,
                "compute_cuda_s": 0.2,
                "cpu_enqueue_s": 0.03,
            },
        }
        for row in before
    ]
    delta = module().worker_deltas(before, after, expected_workers=4)
    assert delta["h2d_bytes"] == 160
    assert delta["copy_launches"] == 16
    assert delta["timing"]["load_cuda_s"] == pytest.approx(0.8)
    with pytest.raises(RuntimeError, match="workers"):
        module().worker_deltas(before, after[:3], expected_workers=4)


def test_smoke_checks_actual_hash_and_never_infers_correctness_from_counters():
    assert (
        module().compare_smoke(
            {
                "input_sha256": "a" * 64,
                "output_sha256": "b" * 64,
                "generated_tokens": 4,
            },
            {
                "input_sha256": "a" * 64,
                "output_sha256": "c" * 64,
                "generated_tokens": 4,
            },
        )
        is False
    )
    with pytest.raises(ValueError, match="smoke input"):
        module().compare_smoke(
            {"input_sha256": "a" * 64, "output_sha256": "b" * 64},
            {"input_sha256": "c" * 64, "output_sha256": "b" * 64},
        )


def test_atomic_json_does_not_leave_partial_final_document(tmp_path: Path):
    path = tmp_path / "rep-000.json"
    module().atomic_json(path, {"elapsed_s": 0.3})
    assert json.loads(path.read_text()) == {"elapsed_s": 0.3}
    assert list(tmp_path.iterdir()) == [path]


def memory_rows(kv_bytes=1000, blocks=10):
    return [
        {
            "rank": rank,
            "total_gpu_bytes": 80000,
            "free_gpu_bytes": 20000,
            "torch_allocated_bytes": 40000,
            "torch_reserved_bytes": 41000,
            "available_kv_cache_bytes": 1200 + rank,
            "model_memory_bytes": 39000,
            "kv_cache_allocated_bytes": kv_bytes,
            "kv_cache_declared_bytes": kv_bytes,
            "kv_cache_accounting_consistent": True,
            "num_gpu_blocks": blocks,
        }
        for rank in range(4)
    ]


def test_fixed_kv_uses_minimum_budget_and_verifies_actual_capacity():
    assert module().resident_kv_budget(memory_rows(), 4) == 1200
    module().validate_fixed_kv(memory_rows(), memory_rows(), 4)
    with pytest.raises(RuntimeError, match="actual KV"):
        module().validate_fixed_kv(memory_rows(kv_bytes=900), memory_rows(), 4)
    invalid = memory_rows()
    invalid[0]["kv_cache_allocated_bytes"] = None
    with pytest.raises(RuntimeError, match="actual KV"):
        module().validate_fixed_kv(invalid, memory_rows(), 4)
    invalid = memory_rows()
    invalid[0]["kv_cache_accounting_consistent"] = False
    with pytest.raises(RuntimeError, match="actual KV"):
        module().validate_fixed_kv(invalid, memory_rows(), 4)


def test_contract_rejects_changed_budget_prefix_policy_or_input():
    ref = {
        "gpu_memory_utilization": 0.6,
        "input_sha256": "a" * 64,
        "engine_policy_sha256": "b" * 64,
        "commit": "c" * 40,
    }
    for field, value in [
        ("gpu_memory_utilization", 0.9),
        ("input_sha256", "d" * 64),
        ("engine_policy_sha256", "e" * 64),
    ]:
        with pytest.raises(ValueError, match="contract"):
            module().validate_contract({**ref, field: value}, ref)


class Engine:
    def __init__(self, **kwargs):
        self.arguments = kwargs
        self.calls = 0
        self.llm_engine = SimpleNamespace(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(
                    dtype="torch.bfloat16",
                    enforce_eager=True,
                    max_model_len=kwargs["max_model_len"],
                    seed=kwargs["seed"],
                ),
                cache_config=SimpleNamespace(
                    enable_prefix_caching=False,
                    gpu_memory_utilization=kwargs["gpu_memory_utilization"],
                    cache_dtype="auto",
                    block_size=16,
                ),
                scheduler_config=SimpleNamespace(
                    max_num_seqs=kwargs["max_num_seqs"],
                    max_num_batched_tokens=kwargs["max_num_batched_tokens"],
                    enable_chunked_prefill=True,
                ),
                parallel_config=SimpleNamespace(
                    tensor_parallel_size=4,
                    data_parallel_size=1,
                    pipeline_parallel_size=1,
                    disable_custom_all_reduce=True,
                ),
                compilation_config=SimpleNamespace(level=0, custom_ops=["all"]),
            )
        )

    def generate(self, prompts, sampling, **kwargs):
        self.calls += 1
        return [
            SimpleNamespace(
                outputs=[SimpleNamespace(token_ids=[7] * sampling.max_tokens)],
                metrics=None,
            )
            for _ in prompts
        ]

    def collective_rpc(self, method, kwargs=None):
        if method == "fluxmoe_synchronize":
            return list(range(4))
        if method == "fluxmoe_worker_memory_stats":
            return memory_rows()
        assert method == "fluxmoe_partial_stats"
        return [
            {
                "schema_version": 1,
                "rank": rank,
                "total_layers": 0,
                "staging_slots": 0,
                "layer_bytes": 0,
                "host_source_bytes": 0,
                "gpu_staging_bytes": 0,
                "resident_routed_bytes": 0,
                "net_freed_bytes": 0,
                "h2d_bytes": 0,
                "copy_launches": 0,
                "offload_forwards": 0,
                "resident_forwards": self.calls * 8,
                "offload_layers": [],
                "timing": {
                    "sample_count": 0,
                    "load_cuda_s": 0,
                    "wait_cuda_s": 0,
                    "compute_cuda_s": 0,
                    "cpu_enqueue_s": 0,
                },
            }
            for rank in range(4)
        ]


def test_runner_writes_complete_repetitions_with_smoke_and_no_raw_tokens(
    tmp_path: Path, monkeypatch
):
    cfg = config(tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            LLM=Engine, SamplingParams=SimpleNamespace, __version__="0.10.2"
        ),
    )
    run = tmp_path / "r-first"
    module().run_benchmark(cfg, project_root=Path.cwd(), run_dir=run)
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["smoke"]["request_count"] == 1
    assert summary["smoke"]["generated_tokens"] == 3
    assert summary["smoke_matches_resident"] is None
    assert summary["repetitions_completed"] == 2
    assert summary["performance_outputs_stable"] is True
    assert summary["repetitions"][0]["diagnostics"]["resident_forwards"] == 32
    assert len(list(run.glob("rep-*.json"))) == 2
    assert "/weights" not in (run / "summary.json").read_text()
    assert "prompt_token_ids" not in (run / "summary.json").read_text()
    with pytest.raises(FileExistsError):
        module().run_benchmark(cfg, project_root=Path.cwd(), run_dir=run)


def test_runner_saves_failure_before_propagating_invalid_fixed_outputs(
    tmp_path: Path, monkeypatch
):
    class BrokenEngine(Engine):
        def generate(self, prompts, sampling, **kwargs):
            return []

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            LLM=BrokenEngine, SamplingParams=SimpleNamespace, __version__="0.10.2"
        ),
    )
    cfg = config(tmp_path)
    run = tmp_path / "r-broken"
    with pytest.raises(RuntimeError, match="fixed-output"):
        module().run_benchmark(cfg, project_root=Path.cwd(), run_dir=run)
    assert json.loads((run / "summary.json").read_text())["status"] == "failed"


def test_runner_rejects_resolved_policy_that_bypasses_custom_forward(
    tmp_path: Path, monkeypatch
):
    class CompiledEngine(Engine):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.llm_engine.vllm_config.compilation_config.level = 3

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            LLM=CompiledEngine, SamplingParams=SimpleNamespace, __version__="0.10.2"
        ),
    )
    with pytest.raises(RuntimeError, match="compilation_config.level"):
        module().run_benchmark(
            config(tmp_path), project_root=Path.cwd(), run_dir=tmp_path / "compiled"
        )


def test_mechanism_gate_requires_real_offload_and_bit_exact_weights():
    stats = [
        {
            "rank": rank,
            "total_layers": 8,
            "offload_layers": [1, 3, 5, 7],
            "staging_slots": 2,
            "layer_bytes": 100,
            "host_source_bytes": 400,
            "gpu_staging_bytes": 200,
            "net_freed_bytes": 200,
            "weights_expected": 8,
            "weights_verified": 8,
            "h2d_bytes": 800,
            "copy_launches": 16,
            "offload_forwards": 4,
        }
        for rank in range(4)
    ]
    module().validate_partial_mechanism(stats, (1, 3, 5, 7), 2, 4)
    stats[0]["weights_verified"] = 7
    with pytest.raises(RuntimeError, match="weight"):
        module().validate_partial_mechanism(stats, (1, 3, 5, 7), 2, 4)
