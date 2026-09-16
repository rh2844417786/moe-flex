"""Native DP coordination: real processes and artifacts, fake GPU boundary only."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from flexmoe.bench import parallel_runner as runner


def test_partition_and_global_wallclock() -> None:
    assert runner.split_indices(5, 2) == [[0, 1, 2], [3, 4]]
    assert (
        runner.summarize_round(
            [
                {"dp_rank": 0, "generated_tokens": 30, "tokens_s": 900},
                {"dp_rank": 1, "generated_tokens": 20, "tokens_s": 800},
            ],
            elapsed_s=2,
            expected_tokens=50,
        )["tokens_s"]
        == 25
    )
    for count, dp in [(1, 2), (0, 1)]:
        with pytest.raises(ValueError):
            runner.split_indices(count, dp)


@pytest.mark.parametrize(
    "rows,elapsed",
    [
        (
            [
                {"dp_rank": 0, "generated_tokens": 50},
                {"dp_rank": 0, "generated_tokens": 0},
            ],
            2,
        ),
        ([{"dp_rank": 1, "generated_tokens": 50}], 2),
        ([{"dp_rank": 0, "generated_tokens": 49}], 2),
        ([{"dp_rank": 0, "generated_tokens": 50, "status": "failed"}], 2),
        ([{"dp_rank": 0, "generated_tokens": 50}], float("nan")),
    ],
)
def test_invalid_round_never_yields_throughput(rows: Any, elapsed: float) -> None:
    with pytest.raises(ValueError):
        runner.summarize_round(rows, elapsed_s=elapsed, expected_tokens=50)


def policy(config: Any) -> dict[str, Any]:
    tp, dp, ep = runner.CONFIGS[config.config]
    return {
        "model_config": {
            "dtype": "torch.bfloat16",
            "enforce_eager": False,
            "quantization": None,
            "seed": 20260912,
            "max_model_len": config.context_length + config.output_length,
        },
        "cache_config": {
            "enable_prefix_caching": False,
            "cpu_offload_gb": 0,
            "swap_space_bytes": 0,
            "gpu_memory_utilization": 0.9,
        },
        "scheduler_config": {
            "max_num_seqs": 1024 // dp,
            "max_num_batched_tokens": 8192 // dp,
            "enable_chunked_prefill": True,
        },
        "parallel_config": {
            "tensor_parallel_size": tp,
            "data_parallel_size": dp,
            "pipeline_parallel_size": 1,
            "enable_expert_parallel": ep,
            "enable_eplb": False,
            "disable_custom_all_reduce": False,
        },
        "compilation_config": {"level": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"},
    }


class FakeGPU:
    """Strict vLLM boundary; never replaces supervision, reduction, or persistence."""

    def __init__(self, config: Any, rank: int, root: Path) -> None:
        import os

        assert os.environ["VLLM_DP_RANK"] == str(rank)
        assert os.environ["VLLM_DP_SIZE"] == str(runner.CONFIGS[config.config][1])
        assert os.environ["FLUXMOE_ENABLE"] == "0"
        assert "FLUXMOE_EXPERT_CALIBRATION" not in os.environ
        self.config, self.rank = config, rank

    def metadata(self) -> dict[str, Any]:
        tp = runner.CONFIGS[self.config.config][0]
        devices = [
            {
                "rank": r,
                "uuid": f"GPU-{self.rank * tp + r}",
                "total_memory": 80_000_000_000,
            }
            for r in range(tp)
        ]
        return {
            "resolved_policy": policy(self.config),
            "devices": devices,
            "versions": {
                "vllm": "0.10.2",
                "torch": "2.8.0",
                "cuda": "12.8",
                "python": "3.12.0",
                "vllm_commit": "a" * 40,
            },
            "memory": self.memory(),
            "expert_placement": None,
        }

    def memory(self) -> list[dict[str, Any]]:
        return [
            {
                "rank": r,
                "total_gpu_bytes": 80_000_000_000,
                "free_gpu_bytes": 4_000_000_000,
                "kv_cache_allocated_bytes": 1000,
                "kv_cache_declared_bytes": 1000,
                "kv_cache_accounting_consistent": True,
                "num_gpu_blocks": 10,
            }
            for r in range(runner.CONFIGS[self.config.config][0])
        ]

    def generate(self, prompts: Any) -> list[Any]:
        time.sleep(0.03)
        return [
            SimpleNamespace(
                prompt_token_ids=list(p),
                outputs=[SimpleNamespace(token_ids=[7] * self.config.output_length)],
                metrics=SimpleNamespace(arrival_time=10.0, finished_time=10.2),
            )
            for p in prompts
        ]


class FailingGPU(FakeGPU):
    def generate(self, prompts: Any) -> list[Any]:
        if self.rank == 1:
            raise RuntimeError("PRIVATE /weights secret failure")
        return super().generate(prompts)


class HangingGPU(FakeGPU):
    def generate(self, prompts: Any) -> list[Any]:
        time.sleep(30)
        return super().generate(prompts)


class ShortOutputGPU(FakeGPU):
    def generate(self, prompts: Any) -> list[Any]:
        outputs = super().generate(prompts)
        return outputs[:-1] if self.rank == 0 else outputs


class WrongIdentityGPU(FakeGPU):
    def generate(self, prompts: Any) -> list[Any]:
        outputs = super().generate(prompts)
        if self.rank == 0:
            outputs[0].prompt_token_ids = [-1]
        return outputs


class PostGenerationMemoryFailureGPU(FakeGPU):
    def generate(self, prompts: Any) -> list[Any]:
        outputs = super().generate(prompts)
        self.generated = True
        return outputs

    def memory(self) -> list[dict[str, Any]]:
        if self.rank == 0 and getattr(self, "generated", False):
            raise RuntimeError("private post-generation memory failure")
        return super().memory()


@pytest.fixture
def config(tmp_path: Path) -> Any:
    from flexmoe.datasets.sharegpt import PromptRecord, write_jsonl_zst

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3_next"}')
    (model / "model.safetensors.index.json").write_text("{}")
    data = tmp_path / "data.zst"
    records = [PromptRecord(str(i), (i, i + 100), 2) for i in range(40)]
    digest = write_jsonl_zst(records, data)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sha256": digest,
                "record_count": 40,
                "counts_by_context": {"2": 40},
                "unique_by_context": {"2": 40},
                "no_repetition": True,
            }
        )
    )
    return runner.ParallelConfig(
        config="ep4-dp2",
        model_path=model,
        dataset_path=data,
        dataset_manifest=manifest,
        batch_size=5,
        context_length=2,
        output_length=10,
        timeout_s=10,
    )


def test_coordinator_actual_processes_and_saved_evidence(
    config: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("FLUXMOE_EXPERT_CALIBRATION", "1")
    result = runner.run_parallel(
        config,
        project_root=Path.cwd(),
        run_dir=tmp_path / "run",
        backend_factory=FakeGPU,
    )
    assert result["status"] == "complete"
    assert result["partitions"] == [[0, 1, 2], [3, 4]]
    assert len(result["repetitions"]) == 3
    assert result["repetitions"][0]["generated_tokens"] == 50
    assert result["repetitions"][0]["latency"]["count"] == 5
    assert result["repetitions"][0]["latency"]["p50_s"] == pytest.approx(0.2)
    assert len(result["workers"]) == 2
    assert len(list((tmp_path / "run" / "private").glob("dp-*/round-*.json"))) == 8
    assert json.loads((tmp_path / "run" / "summary.json").read_text()) == result
    from flexmoe.analysis.parallel_report import summarize_parallel

    assert summarize_parallel(tmp_path / "run")["status"] == "complete"


@pytest.mark.parametrize(
    "backend,timeout,category",
    [
        (FailingGPU, 10, "worker-failed"),
        (HangingGPU, 1.5, "timeout"),
    ],
)
def test_failure_and_timeout_preserve_owned_process_evidence(
    config: Any, tmp_path: Path, backend: Any, timeout: float, category: str
) -> None:
    from dataclasses import replace

    start = time.monotonic()
    result = runner.run_parallel(
        replace(config, timeout_s=timeout),
        project_root=Path.cwd(),
        run_dir=tmp_path / "run",
        backend_factory=backend,
    )
    assert time.monotonic() - start < 10
    assert result["status"] == "failed"
    assert result["failure_category"] == category
    assert "secret" not in (tmp_path / "run" / "summary.json").read_text()
    assert (tmp_path / "run" / "summary.json").is_file()


def test_global_budget_divisibility(config: Any) -> None:
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(config, max_num_seqs=1023)


@pytest.mark.parametrize(
    "backend,request_count,tokens,output_validation",
    [
        (ShortOutputGPU, 2, 20, "failed"),
        (WrongIdentityGPU, 3, 30, "failed"),
        (PostGenerationMemoryFailureGPU, 3, 30, "complete"),
    ],
)
def test_failed_round_retains_counts_before_validation_or_telemetry(
    config: Any,
    tmp_path: Path,
    backend: Any,
    request_count: int,
    tokens: int,
    output_validation: str,
) -> None:
    from flexmoe.analysis.parallel_report import summarize_parallel

    run = tmp_path / "run"
    result = runner.run_parallel(
        config, project_root=Path.cwd(), run_dir=run, backend_factory=backend
    )
    assert result["status"] == "failed"
    rejected = result["incomplete_round"]
    assert rejected["status"] == "incomplete"
    assert rejected["iteration"] == 0
    assert rejected["elapsed_s"] > 0
    assert rejected["tokens_s"] is None
    partial = json.loads((run / "private/dp-0/round-0.json").read_text())
    assert partial["status"] in ("failed", "incomplete")
    assert partial["request_count"] == request_count
    assert partial["generated_tokens"] == tokens
    assert partial["output_counts"] == [10] * request_count
    assert partial["validation"]["outputs"] == output_validation
    assert partial["memory"] == []
    public = summarize_parallel(run)
    assert public["status"] == "failed"
    assert public["median_tokens_s"] is None


@pytest.mark.parametrize(
    "group,key,value",
    [
        ("parallel_config", "data_parallel_size", 1),
        ("parallel_config", "enable_expert_parallel", False),
        ("model_config", "dtype", "float16"),
        ("cache_config", "cpu_offload_gb", 2),
        ("compilation_config", "level", 0),
    ],
)
def test_resolved_policy_rejects_silent_fallback(
    config: Any, group: str, key: str, value: Any
) -> None:
    raw = policy(config)
    raw[group][key] = value
    with pytest.raises(ValueError):
        runner.validate_policy(raw, config)


@pytest.mark.parametrize(
    "name,partitions",
    [("tp4", [[0, 1, 2, 3, 4]]), ("ep4-dp4", [[0, 1], [2], [3], [4]])],
)
def test_all_native_configs_use_same_global_workload(
    config: Any, tmp_path: Path, name: str, partitions: Any
) -> None:
    from dataclasses import replace

    result = runner.run_parallel(
        replace(config, config=name),
        project_root=Path.cwd(),
        run_dir=tmp_path / "run",
        backend_factory=FakeGPU,
    )
    assert result["status"] == "complete"
    assert result["partitions"] == partitions
    assert result["repetitions"][0]["generated_tokens"] == 50


def test_pinned_vllm_boundary_preserves_native_bootstrap(
    config: Any, monkeypatch: Any
) -> None:
    import sys

    expected_policy = policy(config)

    class Engine:
        def __init__(self, **kwargs: Any) -> None:
            assert "data_parallel_size" not in kwargs
            assert "compilation_config" not in kwargs
            assert kwargs["enforce_eager"] is False
            assert kwargs["enable_expert_parallel"] is True
            assert kwargs["tensor_parallel_size"] == 2
            assert kwargs["max_num_seqs"] == 512
            assert kwargs["max_num_batched_tokens"] == 4096
            assert kwargs["cpu_offload_gb"] == 0
            assert kwargs["quantization"] is None
            assert (
                kwargs["worker_extension_cls"]
                == "flexmoe.vllm.bridge.FluxMoEWorkerExtension"
            )
            self.llm_engine = SimpleNamespace(
                vllm_config=SimpleNamespace(
                    **{k: SimpleNamespace(**v) for k, v in expected_policy.items()}
                )
            )

        def generate(self, prompts: Any, sampling: Any, *, use_tqdm: bool) -> list[Any]:
            assert prompts == [{"prompt_token_ids": [1, 2]}]
            assert sampling.max_tokens == sampling.min_tokens == 10
            assert sampling.ignore_eos is True
            assert sampling.temperature == 0
            assert use_tqdm is False
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[7] * 10)])]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            __version__="0.10.2", LLM=Engine, SamplingParams=SimpleNamespace
        ),
    )
    backend = runner.VLLMBackend(config, 0, Path.cwd())
    assert len(backend.generate([(1, 2)])[0].outputs[0].token_ids) == 10
