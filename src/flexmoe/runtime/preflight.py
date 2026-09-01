"""Fail-closed execution preflight with injectable system probes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from flexmoe.config import ModelSpec, RuntimeConfig
from flexmoe.errors import PreflightError


@dataclass(frozen=True)
class GpuInfo:
    """GPU identity reported by the execution host."""

    index: int
    name: str
    total_memory: int


@dataclass(frozen=True)
class ComputeProcess:
    """One process currently consuming a GPU."""

    gpu_index: int
    pid: int
    process_name: str


@dataclass(frozen=True)
class CheckResult:
    """One named preflight check."""

    name: str
    ok: bool
    details: str


@dataclass(frozen=True)
class PreflightReport:
    """All preflight checks plus captured environment metadata."""

    ok: bool
    checks: tuple[CheckResult, ...]
    environment: dict[str, object]


class SystemProbe(Protocol):
    """Boundary implemented by real server probes and test fakes."""

    def gpu_inventory(self) -> tuple[GpuInfo, ...]: ...

    def compute_processes(self) -> tuple[ComputeProcess, ...]: ...

    def mount_options(self, path: Path) -> frozenset[str]: ...

    def driver_version(self) -> str: ...

    def torch_version(self) -> str: ...

    def vllm_commit(self) -> str: ...

    def vmm_supported(self, device: int) -> bool: ...


def validate_checkpoint_files(spec: ModelSpec) -> tuple[Path, ...]:
    """Validate model identity and every indexed non-empty shard."""

    config_path = spec.path / "config.json"
    index_path = spec.path / "model.safetensors.index.json"
    if not config_path.is_file():
        raise PreflightError(f"missing checkpoint config: {config_path}")
    if not index_path.is_file():
        raise PreflightError(f"missing checkpoint index: {index_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or spec.architecture not in architectures:
        raise PreflightError(
            f"checkpoint architecture mismatch: expected {spec.architecture}"
        )
    if config.get("torch_dtype") != spec.dtype:
        raise PreflightError(f"checkpoint dtype mismatch: expected {spec.dtype}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise PreflightError("checkpoint index weight_map must be an object")
    shard_names = sorted({str(filename) for filename in weight_map.values()})
    if len(shard_names) != spec.expected_shards:
        raise PreflightError(
            "checkpoint shard count mismatch: "
            f"expected {spec.expected_shards}, got {len(shard_names)}"
        )

    shard_paths = tuple(spec.path / name for name in shard_names)
    for shard in shard_paths:
        if not shard.is_file() or shard.stat().st_size <= 0:
            raise PreflightError(f"missing or empty checkpoint shard: {shard.name}")
    return shard_paths


def _check(name: str, condition: bool, details: str) -> CheckResult:
    return CheckResult(name=name, ok=condition, details=details)


def run_preflight(config: RuntimeConfig, probe: SystemProbe) -> PreflightReport:
    """Evaluate all declared prerequisites without silently changing config."""

    checks: list[CheckResult] = []
    project_ok = config.project_root.is_dir()
    checks.append(_check("project_root", project_ok, str(config.project_root)))

    try:
        shards = validate_checkpoint_files(config.model)
    except PreflightError as error:
        checks.append(_check("checkpoint", False, str(error)))
    else:
        checks.append(_check("checkpoint", True, f"{len(shards)} shards"))

    mount_options = probe.mount_options(config.model.path)
    checks.append(
        _check(
            "model_mount_read_only",
            "ro" in mount_options,
            ",".join(sorted(mount_options)),
        )
    )

    inventory = {gpu.index: gpu for gpu in probe.gpu_inventory()}
    selected = config.gpu_ids
    gpu_set_ok = (
        len(selected) == config.benchmark.tensor_parallel_size
        and len(set(selected)) == len(selected)
        and all(device in inventory for device in selected)
        and all("H100" in inventory[device].name for device in selected)
    )
    checks.append(_check("gpu_inventory", gpu_set_ok, repr(selected)))

    busy = [
        process for process in probe.compute_processes() if process.gpu_index in selected
    ]
    busy_details = ", ".join(
        f"gpu={process.gpu_index} pid={process.pid} name={process.process_name}"
        for process in busy
    )
    checks.append(_check("exclusive_gpus", not busy, busy_details or "idle"))

    torch_version = probe.torch_version().split("+")[0]
    checks.append(_check("torch_version", torch_version == "2.8.0", torch_version))
    actual_vllm_commit = probe.vllm_commit()
    checks.append(
        _check(
            "vllm_commit",
            actual_vllm_commit == config.vllm_commit,
            actual_vllm_commit,
        )
    )
    vmm_devices = [device for device in selected if probe.vmm_supported(device)]
    checks.append(
        _check(
            "cuda_vmm",
            len(vmm_devices) == len(selected),
            f"supported={vmm_devices}",
        )
    )

    environment: dict[str, object] = {
        "driver_version": probe.driver_version(),
        "torch_version": probe.torch_version(),
        "vllm_commit": actual_vllm_commit,
        "gpu_ids": list(selected),
        "gpu_names": [inventory[device].name for device in selected if device in inventory],
    }
    return PreflightReport(
        ok=all(result.ok for result in checks),
        checks=tuple(checks),
        environment=environment,
    )
