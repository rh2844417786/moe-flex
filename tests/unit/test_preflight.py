import json
from pathlib import Path

import pytest

from flexmoe.config import BenchmarkConfig, ModelSpec, PlannerConfig, RuntimeConfig
from flexmoe.errors import PreflightError
from flexmoe.runtime.preflight import (
    ComputeProcess,
    GpuInfo,
    run_preflight,
    validate_checkpoint_files,
)


def make_checkpoint(root: Path, shard_count: int = 2) -> ModelSpec:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3NextForCausalLM"],
                "torch_dtype": "bfloat16",
            }
        )
    )
    weight_map = {
        f"weight-{index}": f"model-{index:05d}-of-{shard_count:05d}.safetensors"
        for index in range(1, shard_count + 1)
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    for filename in sorted(set(weight_map.values())):
        (root / filename).write_bytes(b"x")
    return ModelSpec(root, "Qwen3NextForCausalLM", "bfloat16", shard_count)


class FakeProbe:
    def __init__(self, *, busy: bool = False) -> None:
        self.busy = busy

    def gpu_inventory(self) -> tuple[GpuInfo, ...]:
        return tuple(
            GpuInfo(index=index, name="NVIDIA H100 80GB HBM3", total_memory=80 << 30)
            for index in range(4)
        )

    def compute_processes(self) -> tuple[ComputeProcess, ...]:
        if not self.busy:
            return ()
        return (ComputeProcess(gpu_index=2, pid=999, process_name="other-job"),)

    def mount_options(self, path: Path) -> frozenset[str]:
        del path
        return frozenset({"ro", "nosuid"})

    def driver_version(self) -> str:
        return "580.173.02"

    def torch_version(self) -> str:
        return "2.8.0"

    def vllm_commit(self) -> str:
        return "01efc7ef781391e744ed08c3292817a773d654e6"

    def vmm_supported(self, device: int) -> bool:
        return device in {0, 1, 2, 3}


def runtime_config(tmp_path: Path) -> RuntimeConfig:
    project = tmp_path / "project"
    project.mkdir()
    model = make_checkpoint(tmp_path / "model")
    return RuntimeConfig(
        project_root=project,
        model=model,
        planner=PlannerConfig(),
        benchmark=BenchmarkConfig(
            variant="vllm-resident",
            batch_size=32,
            context_length=1024,
            output_length=16,
        ),
        gpu_ids=(0, 1, 2, 3),
    )


def test_checkpoint_requires_every_indexed_shard(tmp_path: Path) -> None:
    spec = make_checkpoint(tmp_path / "model")
    (spec.path / "model-00002-of-00002.safetensors").unlink()

    with pytest.raises(PreflightError, match="model-00002-of-00002.safetensors"):
        validate_checkpoint_files(spec)


def test_preflight_accepts_exact_environment(tmp_path: Path) -> None:
    report = run_preflight(runtime_config(tmp_path), FakeProbe())

    assert report.ok is True
    assert all(check.ok for check in report.checks)
    assert report.environment["driver_version"] == "580.173.02"


def test_preflight_rejects_busy_formal_gpu(tmp_path: Path) -> None:
    report = run_preflight(runtime_config(tmp_path), FakeProbe(busy=True))

    assert report.ok is False
    exclusive = next(check for check in report.checks if check.name == "exclusive_gpus")
    assert exclusive.ok is False
    assert "other-job" in exclusive.details
