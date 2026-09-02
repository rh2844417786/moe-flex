"""Fail-closed execution preflight with injectable system probes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import torch

from flexmoe.config import (
    BenchmarkConfig,
    ModelSpec,
    PlannerConfig,
    RuntimeConfig,
)
from flexmoe.errors import PreflightError
from flexmoe.paged_tensor import PagedTensorRegion


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

    def cuda_compute_supported(self, device: int) -> bool: ...


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
    compute_devices = [
        device for device in selected if probe.cuda_compute_supported(device)
    ]
    checks.append(
        _check(
            "cuda_compute",
            len(compute_devices) == len(selected),
            f"supported={compute_devices}",
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


def _command(*arguments: str) -> str:
    try:
        return subprocess.run(
            list(arguments),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise PreflightError(f"command failed: {' '.join(arguments)}: {error}") from error


class NvidiaSystemProbe:
    """Real Linux/NVIDIA probe used only inside the pinned server image."""

    def gpu_inventory(self) -> tuple[GpuInfo, ...]:
        rows = _command(
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        )
        inventory: list[GpuInfo] = []
        for row in rows.splitlines():
            fields = [field.strip() for field in row.split(",")]
            if len(fields) != 3:
                raise PreflightError(f"unexpected nvidia-smi GPU row: {row}")
            inventory.append(
                GpuInfo(
                    index=int(fields[0]),
                    name=fields[1],
                    total_memory=int(fields[2]) * 1024 * 1024,
                )
            )
        return tuple(inventory)

    def compute_processes(self) -> tuple[ComputeProcess, ...]:
        uuid_rows = _command(
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        )
        uuid_to_index = {
            fields[1]: int(fields[0])
            for row in uuid_rows.splitlines()
            if len(fields := [field.strip() for field in row.split(",")]) == 2
        }
        rows = _command(
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        )
        processes: list[ComputeProcess] = []
        for row in rows.splitlines():
            if not row.strip():
                continue
            fields = [field.strip() for field in row.split(",", maxsplit=2)]
            if len(fields) != 3 or fields[0] not in uuid_to_index:
                raise PreflightError(f"unexpected nvidia-smi process row: {row}")
            processes.append(
                ComputeProcess(
                    gpu_index=uuid_to_index[fields[0]],
                    pid=int(fields[1]),
                    process_name=fields[2],
                )
            )
        return tuple(processes)

    def mount_options(self, path: Path) -> frozenset[str]:
        resolved = path.resolve()
        matches: list[tuple[int, frozenset[str]]] = []
        for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
            left, separator, right = line.partition(" - ")
            if not separator:
                continue
            fields = left.split()
            right_fields = right.split()
            if len(fields) < 6 or len(right_fields) < 3:
                continue
            mount_point = Path(fields[4].replace("\\040", " "))
            try:
                resolved.relative_to(mount_point)
            except ValueError:
                continue
            options = frozenset(fields[5].split(",")) | frozenset(
                right_fields[2].split(",")
            )
            matches.append((len(str(mount_point)), options))
        if not matches:
            raise PreflightError(f"cannot resolve mount options for {path}")
        return max(matches, key=lambda item: item[0])[1]

    def driver_version(self) -> str:
        return _command(
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ).splitlines()[0]

    def torch_version(self) -> str:
        return torch.__version__

    def vllm_commit(self) -> str:
        value = os.environ.get("FLEXMOE_VLLM_COMMIT")
        if not value:
            raise PreflightError("FLEXMOE_VLLM_COMMIT is not set")
        return value

    def vmm_supported(self, device: int) -> bool:
        try:
            region = PagedTensorRegion(device=device, virtual_bytes=1)
            return region.granularity > 0
        except (RuntimeError, ValueError):
            return False

    def cuda_compute_supported(self, device: int) -> bool:
        try:
            with torch.cuda.device(device):
                value = torch.ones(1, device=f"cuda:{device}")
                torch.cuda.synchronize(device)
                return value.item() == 1.0
        except (RuntimeError, ValueError):
            return False


def select_idle_h100s(probe: SystemProbe, count: int) -> tuple[int, ...]:
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive int")
    busy = {process.gpu_index for process in probe.compute_processes()}
    eligible = tuple(
        gpu.index
        for gpu in sorted(probe.gpu_inventory(), key=lambda item: item.index)
        if "H100" in gpu.name and gpu.index not in busy
    )
    if len(eligible) < count:
        raise PreflightError(
            f"requested {count} idle H100 GPUs, but only {len(eligible)} are available"
        )
    return eligible[:count]


def _report_json(report: PreflightReport) -> dict[str, object]:
    return {
        "ok": report.ok,
        "checks": [asdict(check) for check in report.checks],
        "environment": report.environment,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser("select-gpus")
    select_parser.add_argument("--count", type=int, required=True)
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--project-root", type=Path, required=True)
    check_parser.add_argument("--model-path", type=Path, required=True)
    check_parser.add_argument("--gpu-ids", required=True)
    check_parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    probe = NvidiaSystemProbe()
    if arguments.command == "select-gpus":
        print(",".join(str(device) for device in select_idle_h100s(probe, arguments.count)))
        return 0
    if arguments.command == "check":
        gpu_ids = tuple(int(item) for item in arguments.gpu_ids.split(","))
        config = RuntimeConfig(
            project_root=arguments.project_root,
            model=ModelSpec(
                path=arguments.model_path,
                architecture="Qwen3NextForCausalLM",
                dtype="bfloat16",
                expected_shards=41,
            ),
            planner=PlannerConfig(),
            benchmark=BenchmarkConfig(
                variant="vllm-resident",
                batch_size=4,
                context_length=128,
                output_length=16,
            ),
            gpu_ids=gpu_ids,
        )
        report = run_preflight(config, probe)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(_report_json(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0 if report.ok else 1
    raise AssertionError(f"unhandled command {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
