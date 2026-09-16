"""Pinned vLLM native TP/DP/EP benchmark with coordinated global timing."""

from __future__ import annotations

import argparse
import importlib
import math
import multiprocessing as mp
import os
import platform
import signal
import socket
import subprocess
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any, cast

from flexmoe.bench.partial_runner import (
    PartialRunConfig,
    _capture_policy,
    _configure_environment,
    atomic_json,
    digest_json,
)

CONFIGS = {"tp4": (4, 1, False), "ep4-dp4": (1, 4, True), "ep4-dp2": (2, 2, True)}


@dataclass(frozen=True)
class ParallelConfig:
    config: str
    model_path: Path
    dataset_path: Path
    dataset_manifest: Path
    batch_size: int = 1024
    context_length: int = 1024
    output_length: int = 512
    repetitions: int = 3
    timeout_s: float = 7200
    max_num_seqs: int = 1024
    max_num_batched_tokens: int = 8192

    def __post_init__(self) -> None:
        if self.config not in CONFIGS:
            raise ValueError("unknown parallel config")
        dp = CONFIGS[self.config][1]
        for name in (
            "batch_size",
            "context_length",
            "output_length",
            "repetitions",
            "max_num_seqs",
            "max_num_batched_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError("positive integer required")
        split_indices(self.batch_size, dp)
        if self.max_num_seqs % dp or self.max_num_batched_tokens % dp:
            raise ValueError("global scheduler budgets must be DP divisible")
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError("token budget smaller than sequence budget")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("positive finite timeout required")


def split_indices(count: int, dp: int) -> list[list[int]]:
    if type(count) is not int or type(dp) is not int or dp < 1 or count < dp:
        raise ValueError("every DP rank requires real requests")
    q, r = divmod(count, dp)
    return [
        list(range(rank * q + min(rank, r), (rank + 1) * q + min(rank + 1, r)))
        for rank in range(dp)
    ]


def summarize_round(
    rows: Sequence[Mapping[str, Any]], *, elapsed_s: float, expected_tokens: int
) -> dict[str, Any]:
    if not rows or sorted(row.get("dp_rank", -1) for row in rows) != list(
        range(len(rows))
    ):
        raise ValueError("missing or duplicated DP partition")
    if not math.isfinite(elapsed_s) or elapsed_s <= 0:
        raise ValueError("invalid global elapsed")
    if any(
        row.get("status", "complete") != "complete"
        or type(row.get("generated_tokens")) is not int
        or row["generated_tokens"] <= 0
        for row in rows
    ):
        raise ValueError("failed or incomplete rank")
    total = sum(row["generated_tokens"] for row in rows)
    if total != expected_tokens:
        raise ValueError("fixed output count differs")
    return {
        "status": "complete",
        "elapsed_s": elapsed_s,
        "generated_tokens": total,
        "tokens_s": total / elapsed_s,
        "timing_scope": "global_wallclock",
    }


def engine_arguments(config: ParallelConfig) -> dict[str, Any]:
    tp, dp, ep = CONFIGS[config.config]
    # DP size is intentionally absent: pinned offline DP reads the VLLM_DP_* env.
    return {
        "model": str(config.model_path),
        "tensor_parallel_size": tp,
        "enable_expert_parallel": ep,
        "dtype": "bfloat16",
        "quantization": None,
        "enforce_eager": False,
        "enable_prefix_caching": False,
        "disable_custom_all_reduce": False,
        "trust_remote_code": False,
        "gpu_memory_utilization": 0.9,
        "max_model_len": config.context_length + config.output_length,
        "max_num_seqs": config.max_num_seqs // dp,
        "max_num_batched_tokens": config.max_num_batched_tokens // dp,
        "enable_chunked_prefill": True,
        "seed": 20260912,
        "cpu_offload_gb": 0,
        "swap_space": 0,
        "distributed_executor_backend": "mp",
        "pipeline_parallel_size": 1,
        "skip_tokenizer_init": True,
        "generation_config": "vllm",
        "worker_extension_cls": "flexmoe.vllm.bridge.FluxMoEWorkerExtension",
    }


def validate_policy(policy: Mapping[str, Any], config: ParallelConfig) -> None:
    args = engine_arguments(config)
    required = {
        "model_config": {
            k: args[k]
            for k in ("enforce_eager", "quantization", "max_model_len", "seed")
        },
        "cache_config": {
            "enable_prefix_caching": False,
            "cpu_offload_gb": 0,
            "swap_space_bytes": 0,
            "gpu_memory_utilization": 0.9,
        },
        "scheduler_config": {
            k: args[k]
            for k in (
                "max_num_seqs",
                "max_num_batched_tokens",
                "enable_chunked_prefill",
            )
        },
        "parallel_config": {
            "tensor_parallel_size": CONFIGS[config.config][0],
            "data_parallel_size": CONFIGS[config.config][1],
            "enable_expert_parallel": CONFIGS[config.config][2],
            "pipeline_parallel_size": 1,
            "enable_eplb": False,
            "disable_custom_all_reduce": False,
        },
    }
    for group, fields in required.items():
        for name, value in fields.items():
            if name not in policy.get(group, {}) or policy[group][name] != value:
                raise ValueError(f"resolved policy differs: {group}.{name}")
    if str(policy.get("model_config", {}).get("dtype")) not in (
        "torch.bfloat16",
        "bfloat16",
    ):
        raise ValueError("resolved dtype differs")
    compilation = policy.get("compilation_config", {})
    if (
        compilation.get("level") in (None, 0, "0")
        or compilation.get("cudagraph_mode") is None
    ):
        raise ValueError("native optimization unavailable")


def latency_summary(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "scope": "arrival-to-finished-offline",
        "count": len(ordered),
        "p50_s": ordered[math.ceil(len(ordered) * 0.5) - 1] if ordered else None,
        "p95_s": ordered[math.ceil(len(ordered) * 0.95) - 1] if ordered else None,
    }


def _rpc_rows(raw: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            rows.append(value)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        else:
            raise TypeError("invalid worker RPC")

    visit(raw)
    return rows


def validate_memory(rows: Sequence[Mapping[str, Any]], tp: int) -> None:
    if len(rows) != tp or len({row.get("rank") for row in rows}) != tp:
        raise ValueError("missing or duplicate TP memory")
    for row in rows:
        for name in (
            "rank",
            "total_gpu_bytes",
            "free_gpu_bytes",
            "kv_cache_allocated_bytes",
            "num_gpu_blocks",
        ):
            value = row.get(name)
            if type(value) is not int or value < (0 if name == "rank" else 1):
                raise ValueError("actual worker memory unavailable")
        if (
            row["free_gpu_bytes"] < 2_000_000_000
            or row.get("kv_cache_accounting_consistent") is not True
        ):
            raise ValueError("unsafe or inconsistent worker memory")


class VLLMBackend:
    def __init__(self, config: ParallelConfig, rank: int, root: Path) -> None:
        self.config = config
        self.vllm = importlib.import_module("vllm")
        torch = importlib.import_module("torch")
        self.versions = {
            "vllm": str(self.vllm.__version__),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "vllm_commit": os.environ.get("FLEXMOE_VLLM_COMMIT"),
        }
        if self.versions["vllm"].split("+")[0] != "0.10.2":
            raise RuntimeError("pinned vLLM 0.10.2 required")
        self.arguments = engine_arguments(config)
        self.engine = self.vllm.LLM(**self.arguments)
        self.sampling = self.vllm.SamplingParams(
            temperature=0,
            top_p=1,
            max_tokens=config.output_length,
            min_tokens=config.output_length,
            ignore_eos=True,
            seed=20260912,
        )

    def metadata(self) -> dict[str, Any]:
        policy = _capture_policy(self.engine, self.arguments)
        parallel = self.engine.llm_engine.vllm_config.parallel_config
        return {
            "resolved_policy": policy,
            "versions": self.versions,
            "devices": _rpc_rows(self.engine.collective_rpc("fluxmoe_analysis_device")),
            "memory": self.memory(),
            "expert_placement": getattr(parallel, "expert_placement_strategy", None),
        }

    def memory(self) -> list[dict[str, Any]]:
        return _rpc_rows(self.engine.collective_rpc("fluxmoe_worker_memory_stats"))

    def generate(self, prompts: Sequence[Sequence[int]]) -> Any:
        return self.engine.generate(
            [{"prompt_token_ids": list(p)} for p in prompts],
            self.sampling,
            use_tqdm=False,
        )


def _worker(
    config: ParallelConfig,
    rank: int,
    root: Path,
    run: Path,
    prompts: Sequence[Sequence[int]],
    indices: list[int],
    port: int,
    connection: Connection,
    factory: Callable[..., Any],
) -> None:
    # Each worker owns a process group, including its vLLM engine descendants.
    os.setsid()
    private = run / "private" / f"dp-{rank}"
    private.mkdir(parents=True, exist_ok=True)
    try:
        _configure_environment(
            PartialRunConfig(
                arm="resident",
                model_path=config.model_path,
                dataset_path=config.dataset_path,
                dataset_manifest=config.dataset_manifest,
            ),
            root,
            (),
        )
        os.environ.update(
            VLLM_DP_RANK=str(rank),
            VLLM_DP_RANK_LOCAL=str(rank),
            VLLM_DP_SIZE=str(CONFIGS[config.config][1]),
            VLLM_DP_MASTER_IP="127.0.0.1",
            VLLM_DP_MASTER_PORT=str(port),
        )
        backend = factory(config, rank, root)
        metadata = backend.metadata()
        metadata.update(dp_rank=rank, indices=indices)
        atomic_json(private / "initialized.json", metadata)
        validate_policy(metadata["resolved_policy"], config)
        validate_memory(metadata["memory"], CONFIGS[config.config][0])
        connection.send(("ready", metadata))
        for iteration in range(config.repetitions + 1):
            if connection.recv() != "prepare":
                raise ValueError("invalid coordination command")
            validate_memory(backend.memory(), CONFIGS[config.config][0])
            connection.send(("prepared", rank))
            if connection.recv() != "generate":
                raise ValueError("invalid generation command")
            outputs = backend.generate(prompts)
            # The small completion signal precedes CPU analysis, telemetry and disk IO.
            connection.send(("generated", rank))
            if connection.recv() != "collect":
                raise ValueError("invalid collection command")
            counts, hashes, latencies = [], [], []
            if len(outputs) != len(prompts):
                raise ValueError("missing request output")
            for prompt, output in zip(prompts, outputs):
                if (
                    list(output.prompt_token_ids) != list(prompt)
                    or len(output.outputs) != 1
                ):
                    raise ValueError("request output identity differs")
                tokens = list(output.outputs[0].token_ids)
                counts.append(len(tokens))
                hashes.append(digest_json(tokens))
                metrics = getattr(output, "metrics", None)
                arrival, finished = (
                    getattr(metrics, "arrival_time", None),
                    getattr(metrics, "finished_time", None),
                )
                if type(arrival) in (int, float) and type(finished) in (int, float):
                    duration = float(cast(float, finished)) - float(cast(float, arrival))
                    if math.isfinite(duration) and duration >= 0:
                        latencies.append(duration)
            row = {
                "dp_rank": rank,
                "indices": indices,
                "status": "complete",
                "request_count": len(counts),
                "output_counts": counts,
                "generated_tokens": sum(counts),
                "output_hashes": hashes,
                "latencies_s": latencies,
                "memory": backend.memory(),
            }
            atomic_json(private / f"round-{iteration}.json", row)
            if any(count != config.output_length for count in counts):
                raise ValueError("fixed output length differs")
            validate_memory(row["memory"], CONFIGS[config.config][0])
            connection.send(("collected", row))
        connection.recv()  # Keep every DP engine alive until all ranks finish.
    except BaseException:  # noqa: BLE001 -- child failures must wake parent and preserve private evidence
        (private / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        try:
            connection.send(("failed", rank))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _receive(
    connections: Sequence[Connection], label: str, deadline: float
) -> list[Any]:
    pending = set(connections)
    rows = []
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("worker deadline exceeded")
        ready = cast(list[Connection], wait(list(pending), timeout=min(remaining, 0.2)))
        for connection in ready:
            try:
                kind, payload = connection.recv()
            except (EOFError, OSError) as error:
                raise RuntimeError("worker exited") from error
            if kind != label:
                raise RuntimeError("worker failed")
            pending.remove(connection)
            rows.append(payload)
    return rows


def _stop_owned(processes: Sequence[Any], connections: Sequence[Connection]) -> None:
    for connection in connections:
        connection.close()
    for process in processes:
        if process.pid is not None:
            try:
                # setsid establishes ownership; never signal the parent's group.
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                if process.is_alive():
                    process.terminate()
    for process in processes:
        process.join(timeout=1)
        if process.pid is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                if process.is_alive():
                    process.kill()
        process.join(timeout=1)


def validate_workers(
    workers: Sequence[Mapping[str, Any]], config: ParallelConfig
) -> str:
    tp, dp, _ = CONFIGS[config.config]
    if sorted(row.get("dp_rank", -1) for row in workers) != list(range(dp)):
        raise ValueError("missing DP worker")
    devices = []
    versions = []
    for row in workers:
        validate_policy(row["resolved_policy"], config)
        validate_memory(row["memory"], tp)
        ranks = {memory["rank"] for memory in row["memory"]}
        if len(row["devices"]) != tp or {d["rank"] for d in row["devices"]} != ranks:
            raise ValueError("device/TP memory mapping differs")
        for device in row["devices"]:
            if (
                not isinstance(device.get("uuid"), str)
                or not device["uuid"]
                or type(device.get("total_memory")) is not int
            ):
                raise ValueError("actual device identity unavailable")
            devices.append(
                {"uuid": device["uuid"], "total_memory": device["total_memory"]}
            )
        versions.append(row["versions"])
    if len({d["uuid"] for d in devices}) != 4 or any(
        v != versions[0] for v in versions
    ):
        raise ValueError("physical device or version mismatch")
    return digest_json(sorted(devices, key=lambda d: d["uuid"]))


def run_parallel(
    config: ParallelConfig,
    *,
    project_root: Path,
    run_dir: Path,
    backend_factory: Callable[..., Any] = VLLMBackend,
) -> dict[str, Any]:
    from flexmoe.datasets.decode_corpus import load_unique_workload

    run_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "parallel-run",
        "diagnostic_only": True,
        "formal_offload_gain": False,
        "deployment_gain_proven": False,
        "config": config.config,
        "mode": "full-resident-native",
        "status": "running",
        "timing_scope": "global_wallclock",
        "workers": [],
        "repetitions": [],
        "partitions": split_indices(config.batch_size, CONFIGS[config.config][1]),
        "requested_policy": {
            k: v for k, v in engine_arguments(config).items() if k != "model"
        },
    }
    processes, connections = [], []
    atomic_json(run_dir / "summary.json", summary)
    try:
        prompts, _, workload = load_unique_workload(
            config.dataset_path,
            config.dataset_manifest,
            context_length=config.context_length,
            request_count=config.batch_size,
            calibration_count=32,
        )
        model_identity = {
            "config_sha256": sha256(
                (config.model_path / "config.json").read_bytes()
            ).hexdigest(),
            "index_sha256": sha256(
                (config.model_path / "model.safetensors.index.json").read_bytes()
            ).hexdigest(),
            "path_sha256": sha256(
                str(config.model_path.resolve()).encode()
            ).hexdigest(),
        }
        commit = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
        ).strip()
        summary["contract"] = {
            **workload,
            "input_sha256": digest_json(prompts),
            "model_identity_sha256": digest_json(model_identity),
            "commit": commit,
            "batch_size": config.batch_size,
            "context_length": config.context_length,
            "output_length": config.output_length,
            "repetitions_requested": config.repetitions,
            "max_num_seqs": config.max_num_seqs,
            "max_num_batched_tokens": config.max_num_batched_tokens,
            "seed": 20260912,
            "warmups": 1,
            "gpu_memory_utilization": 0.9,
            "dtype": "bfloat16",
        }
        atomic_json(run_dir / "summary.json", summary)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        context = mp.get_context("spawn")
        deadline = time.monotonic() + config.timeout_s
        for rank, indices in enumerate(summary["partitions"]):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker,
                args=(
                    config,
                    rank,
                    project_root,
                    run_dir,
                    [prompts[i] for i in indices],
                    indices,
                    port,
                    child,
                    backend_factory,
                ),
            )
            process.start()
            child.close()
            processes.append(process)
            connections.append(parent)
        workers = _receive(connections, "ready", deadline)
        summary["workers"] = sorted(workers, key=lambda row: row["dp_rank"])
        summary["contract"]["hardware_sha256"] = validate_workers(workers, config)
        summary["contract"]["versions"] = workers[0]["versions"]
        atomic_json(run_dir / "summary.json", summary)
        for iteration in range(config.repetitions + 1):
            for connection in connections:
                connection.send("prepare")
            _receive(connections, "prepared", deadline)
            start = time.perf_counter()
            for connection in connections:
                connection.send("generate")
            _receive(connections, "generated", deadline)
            elapsed = time.perf_counter() - start
            for connection in connections:
                connection.send("collect")
            rows = sorted(
                _receive(connections, "collected", deadline),
                key=lambda row: row["dp_rank"],
            )
            if [row["indices"] for row in rows] != summary["partitions"]:
                raise ValueError("partition identity differs")
            result = summarize_round(
                rows,
                elapsed_s=elapsed,
                expected_tokens=config.batch_size * config.output_length,
            )
            result.update(
                iteration=iteration,
                per_rank=rows,
                latency=latency_summary(
                    [v for row in rows for v in row["latencies_s"]]
                ),
            )
            if iteration:
                summary["repetitions"].append(result)
            else:
                summary["warmup"] = result
            atomic_json(run_dir / "summary.json", summary)
        summary["status"] = "complete"
    except BaseException as error:
        summary["status"] = "failed"
        summary["failure_category"] = (
            "timeout"
            if isinstance(error, TimeoutError)
            else "worker-failed"
            if isinstance(error, RuntimeError)
            else "invalid-evidence"
        )
        private = run_dir / "private"
        private.mkdir(exist_ok=True)
        (private / "parent-error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        _stop_owned(processes, connections)
        atomic_json(run_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=CONFIGS, required=True)
    for name in (
        "project-root",
        "run-dir",
        "model-path",
        "dataset-path",
        "dataset-manifest",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name, default in (
        ("repetitions", 3),
        ("output-length", 512),
        ("batch-size", 1024),
        ("context-length", 1024),
        ("timeout-s", 7200),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    args = vars(parser.parse_args(argv))
    root, run = args.pop("project_root").resolve(), args.pop("run_dir").resolve()
    if not run.is_relative_to(root) or run == root:
        parser.error("run-dir must be a fresh directory inside project-root")
    for name in ("model_path", "dataset_path", "dataset_manifest"):
        if not args[name].is_absolute():
            args[name] = root / args[name]
    result = run_parallel(ParallelConfig(**args), project_root=root, run_dir=run)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
