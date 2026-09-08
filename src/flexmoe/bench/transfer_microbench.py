"""Four-rank pinned transfer measurements; GEMM/NCCL joint windows are proxies."""

from __future__ import annotations

import argparse
import importlib
import math
import multiprocessing
import os
import platform
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

from flexmoe.analysis.schema import TransferSample, diagnostic_artifact
from flexmoe.bench.analysis_runner import hardware_identity
from flexmoe.bench.partial_runner import atomic_json, digest_json, read_json


@dataclass(frozen=True)
class TransferConfig:
    expert_bytes: int = 1572864
    experts_per_batch: tuple[int, ...] = (1, 8, 32, 128, 512)
    modes: tuple[str, ...] = ("contiguous", "fragmented", "gather")
    contention: str = "isolated"
    warmups: int = 2
    repetitions: int = 5
    iterations: int = 1
    memory_cap_bytes: int = 4_000_000_000
    safety_reserve_bytes: int = 2_000_000_000
    timeout_s: float = 600.0
    gemm_size: int = 2048

    def __post_init__(self) -> None:
        for name in (
            "expert_bytes",
            "repetitions",
            "iterations",
            "memory_cap_bytes",
            "safety_reserve_bytes",
            "gemm_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.warmups) is not int or self.warmups < 0:
            raise ValueError("warmups must be nonnegative")
        if (
            type(self.timeout_s) not in (float, int)
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ValueError("timeout_s must be positive and finite")
        if not self.experts_per_batch or any(
            type(v) is not int or v <= 0 for v in self.experts_per_batch
        ):
            raise ValueError("experts_per_batch must contain positive integers")
        if len(set(self.experts_per_batch)) != len(self.experts_per_batch):
            raise ValueError("experts_per_batch must be unique")
        if (
            not self.modes
            or len(set(self.modes)) != len(self.modes)
            or set(self.modes) - {"contiguous", "fragmented", "gather"}
        ):
            raise ValueError("invalid transfer modes")
        if self.contention not in ("isolated", "gemm-nccl-proxy"):
            raise ValueError("invalid contention")


def required_memory(config: TransferConfig, experts: int) -> dict[str, int]:
    payload = experts * config.expert_bytes
    proxy = 3 * config.gemm_size**2 * 2 if config.contention != "isolated" else 0
    return {
        "device_bytes": payload + proxy,
        "host_pinned_bytes": 2 * payload if "gather" in config.modes else payload,
    }


def payload_shape(config: TransferConfig, experts: int) -> tuple[int, int]:
    if type(experts) is not int or experts <= 0:
        raise ValueError("experts must be positive")
    if any(
        v > config.memory_cap_bytes for v in required_memory(config, experts).values()
    ):
        raise ValueError("transfer allocation exceeds explicit memory cap")
    return experts, config.expert_bytes


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(
            row[k] for k in ("experts_per_batch", "mode", "contention", "repetition")
        )
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in sorted(groups.items()):
        if len(group) != 4 or {r["rank"] for r in group} != set(range(4)):
            raise ValueError("transfer group requires exactly four ranks")
        if len({r["payload_bytes"] for r in group}) != 1:
            raise ValueError("rank payloads differ")
        worst = max(group, key=lambda r: r["wall_s"])
        result.append(
            dict(zip(("experts_per_batch", "mode", "contention", "repetition"), key))
            | {
                "bottleneck_rank": worst["rank"],
                "bottleneck_wall_s": worst["wall_s"],
                "per_rank_payload_bytes": worst["payload_bytes"],
                "bottleneck_bytes_per_s": worst["payload_bytes"] / worst["wall_s"],
            }
        )
    return result


def validate_sample_coverage(
    rows: Sequence[dict[str, Any]], config: TransferConfig, contract: dict[str, Any]
) -> None:
    expected = {
        (rank, experts, mode, repetition)
        for rank in range(4)
        for experts in config.experts_per_batch
        for mode in config.modes
        for repetition in range(config.repetitions)
    }
    actual = {
        (row["rank"], row["experts_per_batch"], row["mode"], row["repetition"])
        for row in rows
    }
    if actual != expected or len(rows) != len(expected):
        raise ValueError("transfer sample coverage differs from requested grid")
    if any(
        row["contract"] != contract
        or row["contention"] != config.contention
        or row["expert_bytes"] != config.expert_bytes
        for row in rows
    ):
        raise ValueError("transfer sample contract differs from coordinator")


def wait_owned_workers(children: Sequence[Any], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            if any(c.exitcode not in (None, 0) for c in children):
                raise RuntimeError("owned transfer worker failed")
            alive = [c for c in children if c.is_alive()]
            if not alive:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("owned transfer worker deadline exceeded")
            alive[0].join(timeout=min(0.1, max(0.0, deadline - time.monotonic())))
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            child.join(timeout=5)
            if child.is_alive():
                child.kill()
                child.join(timeout=5)


def _measure_case(
    torch: Any, dist: Any, config: TransferConfig, experts: int, mode: str
) -> list[dict[str, Any]]:
    shape = payload_shape(config, experts)
    free, _ = torch.cuda.mem_get_info()
    if (
        required_memory(config, experts)["device_bytes"] + config.safety_reserve_bytes
        > free
    ):
        raise RuntimeError("transfer physical safety reserve would be exceeded")
    destination = torch.empty(shape, dtype=torch.uint8, device="cuda")
    parts = (
        [
            torch.full(
                (config.expert_bytes,), i % 251, dtype=torch.uint8, pin_memory=True
            )
            for i in range(experts)
        ]
        if mode != "contiguous"
        else []
    )
    host = (
        torch.empty(shape, dtype=torch.uint8, pin_memory=True)
        if mode != "fragmented"
        else None
    )
    if mode == "contiguous":
        assert host is not None
        host.fill_(17)
    copy_stream, compute_stream = torch.cuda.Stream(), torch.cuda.Stream()
    proxy = config.contention == "gemm-nccl-proxy"
    matrices = None
    if proxy:
        a = torch.full(
            (config.gemm_size, config.gemm_size),
            1 / config.gemm_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        matrices = (a, a.clone(), torch.empty_like(a))
    torch.cuda.synchronize()

    def window(copy: bool, compute: bool) -> dict[str, float]:
        dist.barrier()
        torch.cuda.synchronize()
        copy_start, copy_end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        compute_start, compute_end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        gather_s = 0.0
        started = time.perf_counter()
        # Gather all iterations up front would overwrite an in-flight pinned source.
        # Use a stable gathered buffer once and account for that per measured batch.
        if copy and mode == "gather":
            gathered = time.perf_counter()
            torch.stack(parts, out=host)
            gather_s = time.perf_counter() - gathered
        if copy:
            with torch.cuda.stream(copy_stream):
                copy_start.record()
                for _ in range(config.iterations):
                    if mode == "fragmented":
                        for i, source in enumerate(parts):
                            destination[i].copy_(source, non_blocking=True)
                    else:
                        destination.copy_(host, non_blocking=True)
                copy_end.record()
        if compute:
            assert matrices is not None
            with torch.cuda.stream(compute_stream):
                compute_start.record()
                for _ in range(config.iterations):
                    torch.mm(matrices[0], matrices[1], out=matrices[2])
                    work = dist.all_reduce(matrices[2], async_op=True)
                    work.wait()
                compute_end.record()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        # Each sample reports one batch service. Gather is performed once per
        # window, so repeat it conceptually only by selecting iterations=1.
        return {
            "wall_s": elapsed / config.iterations,
            "copy_s": copy_start.elapsed_time(copy_end) / 1000 / config.iterations
            if copy
            else 0.0,
            "compute_s": compute_start.elapsed_time(compute_end)
            / 1000
            / config.iterations
            if compute
            else 0.0,
            "gather_s": gather_s / config.iterations,
        }

    rows = []
    for repetition in range(-config.warmups, config.repetitions):
        copy_only = window(True, False)
        compute_only = window(False, True) if proxy else None
        joint = window(True, True) if proxy else copy_only
        if repetition >= 0:
            rows.append(
                {
                    "repetition": repetition,
                    **joint,
                    "compute_s": joint["compute_s"] if proxy else None,
                    "copy_only_wall_s": copy_only["wall_s"],
                    "compute_only_wall_s": compute_only["wall_s"]
                    if compute_only
                    else None,
                    "joint_wall_s": joint["wall_s"] if proxy else None,
                    "copy_only_components": copy_only,
                    "compute_only_components": compute_only,
                }
            )
    return rows


def _worker(
    rank: int, config: TransferConfig, run_dir: str, contract: dict[str, Any]
) -> None:
    torch = importlib.import_module("torch")
    dist = importlib.import_module("torch.distributed")
    path = Path(run_dir)
    phase = "initialization"
    try:
        if torch.cuda.device_count() != 4:
            raise RuntimeError("transport requires exactly four visible CUDA devices")
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl",
            init_method=(path / "rendezvous").as_uri(),
            rank=rank,
            world_size=4,
            timeout=timedelta(seconds=config.timeout_s),
        )
        from flexmoe.analysis.schema import validate_transfer_contract
        from flexmoe.vllm.analysis_trace import device_record

        devices: list[Any] = [None] * 4
        dist.all_gather_object(devices, device_record(rank))
        contract = validate_transfer_contract(
            {**contract, **hardware_identity(devices)}
        )
        affinity = (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        )
        rows, measurements = [], []
        for experts in config.experts_per_batch:
            for mode in config.modes:
                phase = f"measure-{experts}-{mode}"
                for raw in _measure_case(torch, dist, config, experts, mode):
                    sample = TransferSample(
                        rank,
                        contract,
                        experts,
                        config.expert_bytes,
                        mode,
                        config.contention,
                        raw["repetition"],
                        experts * config.expert_bytes,
                        raw["wall_s"],
                        raw["copy_s"],
                        raw["gather_s"],
                        raw["compute_s"],
                    )
                    rows.append(sample.to_dict())
                    measurements.append(
                        {
                            "rank": rank,
                            "experts_per_batch": experts,
                            "mode": mode,
                            **raw,
                        }
                    )
                torch.cuda.empty_cache()
        atomic_json(
            path / f"worker-{rank}.json",
            diagnostic_artifact(
                "transfer-worker",
                {
                    "status": "complete",
                    "rank": rank,
                    "samples": rows,
                    "measurements": measurements,
                    "cpu_affinity_sha256": digest_json(affinity)
                    if affinity is not None
                    else None,
                    "topology_status": "NUMA/PCIe-unavailable",
                    "device_rank": rank,
                },
            ),
        )
    except BaseException as error:
        atomic_json(
            path / f"worker-{rank}.json",
            diagnostic_artifact(
                "transfer-worker",
                {
                    "status": "failed",
                    "rank": rank,
                    "phase": phase,
                    "error_type": type(error).__name__,
                },
            ),
        )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_transfer(config: TransferConfig, *, project_root: Path, run_dir: Path) -> Path:
    if not run_dir.resolve().is_relative_to(project_root.resolve()):
        raise ValueError("run-dir must remain inside project-root")
    for experts in config.experts_per_batch:
        payload_shape(config, experts)
    if "gather" in config.modes and config.iterations != 1:
        raise ValueError(
            "gather requires iterations=1 to retain per-batch CPU gather cost"
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    fields: dict[str, Any] = {
        "status": "running",
        "samples": [],
        "config": asdict(config),
        "contention_role": "isolated-transfer"
        if config.contention == "isolated"
        else "GEMM-NCCL-proxy-joint-not-transfer-tax",
    }
    children: list[Any] = []
    try:
        atomic_json(
            run_dir / "samples.json", diagnostic_artifact("transfer-samples", fields)
        )
        torch, vllm = importlib.import_module("torch"), importlib.import_module("vllm")
        contract = {
            "commit": subprocess.check_output(
                ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
            ).strip(),
            "tensor_parallel_size": 4,
            "versions": {
                "torch": str(torch.__version__),
                "vllm": str(vllm.__version__),
                "cuda": torch.version.cuda,
                "vllm_commit": os.environ.get("FLEXMOE_VLLM_COMMIT"),
                "python": platform.python_version(),
            },
            "measurement_id": run_dir.name,
            "benchmark_policy_sha256": digest_json(asdict(config)),
        }
        fields["contract"] = contract
        context = multiprocessing.get_context("spawn")
        for rank in range(4):
            child = context.Process(
                target=_worker, args=(rank, config, str(run_dir.resolve()), contract)
            )
            child.start()
            children.append(child)
        wait_owned_workers(children, config.timeout_s)
        workers = [read_json(run_dir / f"worker-{i}.json") for i in range(4)]
        if any(row["status"] != "complete" for row in workers):
            raise RuntimeError("transfer worker did not complete")
        samples = [
            TransferSample.from_dict(sample).to_dict()
            for row in workers
            for sample in row["samples"]
        ]
        if not samples:
            raise RuntimeError("transfer workers returned no samples")
        measured_contract = cast(dict[str, Any], samples[0]["contract"])
        if any(measured_contract.get(k) != v for k, v in contract.items()):
            raise ValueError(
                "worker software/measurement contract differs from coordinator"
            )
        validate_sample_coverage(samples, config, measured_contract)
        fields["contract"] = measured_contract
        fields.update(
            status="complete",
            samples=samples,
            aggregates=aggregate_rows(samples),
            measurements=[sample for row in workers for sample in row["measurements"]],
            worker_metadata=[
                {k: v for k, v in row.items() if k not in ("samples", "measurements")}
                for row in workers
            ],
        )
        atomic_json(
            run_dir / "samples.json", diagnostic_artifact("transfer-samples", fields)
        )
    except BaseException as error:
        fields.update(
            status="failed", error_type=type(error).__name__, phase="coordinator"
        )
        atomic_json(
            run_dir / "samples.json", diagnostic_artifact("transfer-samples", fields)
        )
        raise
    finally:
        # Handles partial spawn failures as well as wait failures; owns only these PIDs.
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            child.join(timeout=5)
            if child.is_alive():
                child.kill()
                child.join(timeout=5)
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--experts-per-batch", default="1,8,32,128,512")
    parser.add_argument("--modes", default="contiguous,fragmented,gather")
    parser.add_argument(
        "--contention", choices=("isolated", "gemm-nccl-proxy"), default="isolated"
    )
    parser.add_argument("--timeout-s", type=float, default=600)
    for name, default in (
        ("expert-bytes", 1572864),
        ("warmups", 2),
        ("repetitions", 5),
        ("iterations", 1),
        ("memory-cap-bytes", 4_000_000_000),
        ("safety-reserve-bytes", 2_000_000_000),
        ("gemm-size", 2048),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    args = vars(parser.parse_args(argv))
    root, run = args.pop("project_root").resolve(), args.pop("run_dir").resolve()
    args["experts_per_batch"] = tuple(
        int(v) for v in args["experts_per_batch"].split(",")
    )
    args["modes"] = tuple(args["modes"].split(","))
    run_transfer(TransferConfig(**args), project_root=root, run_dir=run)
    print(run.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
