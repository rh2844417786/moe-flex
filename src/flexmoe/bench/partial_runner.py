"""Offline fixed-workload benchmark for resident and partial BF16 host weights.

One process creates one engine. Public artifacts contain hashes and numeric
measurements, never prompts, output IDs, weight paths, or host identifiers.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, cast

ARMS = ("resident", "partial-fixed-kv", "partial-auto-kv")
COUNTERS = ("h2d_bytes", "copy_launches", "offload_forwards", "resident_forwards")
TIMINGS = (
    "sample_count",
    "load_cuda_s",
    "wait_cuda_s",
    "compute_cuda_s",
    "cpu_enqueue_s",
)


@dataclass(frozen=True)
class PartialRunConfig:
    arm: str
    model_path: Path
    dataset_path: Path
    dataset_manifest: Path
    batch_size: int = 256
    context_length: int = 4096
    output_length: int = 256
    tensor_parallel_size: int = 4
    gpu_memory_utilization: float = 0.60
    offload_count: int = 0
    staging_slots: int = 1
    warmups: int = 1
    repetitions: int = 3
    seed: int = 20260905
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    timing_samples: int = 128
    smoke_output_length: int = 8

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError("unknown partial benchmark arm")
        for name in (
            "batch_size",
            "context_length",
            "output_length",
            "tensor_parallel_size",
            "repetitions",
            "max_num_seqs",
            "max_num_batched_tokens",
            "smoke_output_length",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmups", "seed", "offload_count", "timing_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.tensor_parallel_size != 4:
            raise ValueError("this benchmark requires four GPUs")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.staging_slots not in (1, 2):
            raise ValueError("staging_slots must be 1 or 2")
        if self.arm == "resident" and self.offload_count:
            raise ValueError("resident requires offload_count=0")
        if self.offload_count and (
            self.offload_count <= self.staging_slots
            or self.offload_count % self.staging_slots
        ):
            raise ValueError(
                "offload_count must exceed and be divisible by staging_slots"
            )
        if self.max_num_batched_tokens < min(self.batch_size, self.max_num_seqs):
            raise ValueError("max_num_batched_tokens is smaller than max_num_seqs")


@dataclass(frozen=True)
class PartialWorkload:
    prompts: tuple[tuple[int, ...], ...]
    metadata: dict[str, object]


def digest_json(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    """Promote only fsynced, complete JSON; a killed run keeps earlier repetitions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path.name}")
    return cast(dict[str, Any], value)


def load_partial_workload(
    dataset: Path,
    manifest: Path,
    batch_size: int,
    context_length: int,
) -> PartialWorkload:
    from flexmoe.datasets.sharegpt import read_jsonl_zst, verify_subset

    if batch_size < 1 or context_length < 1:
        raise ValueError("batch and context must be positive")
    verify_subset(dataset, manifest)
    source = tuple(
        row.prompt_token_ids
        for row in read_jsonl_zst(dataset)
        if row.context_length == context_length
    )
    if not source:
        raise ValueError("committed dataset has no requests at this context length")
    prompts = tuple(source[index % len(source)] for index in range(batch_size))
    return PartialWorkload(
        prompts,
        {
            "dataset_sha256": read_json(manifest)["sha256"],
            "dataset_manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
            "input_sha256": digest_json(prompts),
            "source_request_count": len(source),
            "unique_selected_request_count": min(len(source), batch_size),
            "repeated_request_count": max(0, batch_size - len(source)),
            "sampling_policy": (
                "repeated-existing-prompts"
                if batch_size > len(source)
                else "existing-prompts"
            ),
        },
    )


def engine_arguments(
    config: PartialRunConfig, fixed_kv_bytes: int | None
) -> dict[str, Any]:
    if config.arm == "partial-fixed-kv":
        if type(fixed_kv_bytes) is not int or fixed_kv_bytes <= 0:
            raise ValueError(
                "partial-fixed-kv requires the completed resident KV budget"
            )
    elif fixed_kv_bytes is not None:
        raise ValueError("only partial-fixed-kv may override the KV budget")
    return {
        "model": str(config.model_path),
        "tensor_parallel_size": config.tensor_parallel_size,
        "dtype": "bfloat16",
        "quantization": None,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "disable_custom_all_reduce": True,
        "trust_remote_code": False,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "kv_cache_memory_bytes": fixed_kv_bytes,
        "max_model_len": config.context_length + config.output_length,
        "max_num_seqs": min(config.batch_size, config.max_num_seqs),
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "enable_chunked_prefill": True,
        "seed": config.seed,
        "cpu_offload_gb": 0,
        "swap_space": 0,
        "distributed_executor_backend": "mp",
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "compilation_config": {"level": 0, "custom_ops": ["all"]},
        "skip_tokenizer_init": True,
        "generation_config": "vllm",
        "worker_extension_cls": "flexmoe.vllm.bridge.FluxMoEWorkerExtension",
    }


def _number(value: object) -> float | None:
    if type(value) not in (int, float):
        return None
    number = float(cast(float, value))
    return number if math.isfinite(number) else None


def summarize_outputs(
    outputs: Sequence[Any],
    *,
    request_count: int,
    output_length: int,
    elapsed_s: float,
) -> dict[str, Any]:
    token_ids = [list(output.outputs[0].token_ids) for output in outputs]
    if len(token_ids) != request_count or any(
        len(row) != output_length for row in token_ids
    ):
        raise RuntimeError(
            "fixed-output protocol violated: request count or output length differs"
        )
    if elapsed_s <= 0 or not math.isfinite(elapsed_s):
        raise RuntimeError("elapsed time must be finite and positive")
    ttfts: list[float] = []
    latencies: list[float] = []
    for output in outputs:
        metrics = getattr(output, "metrics", None)
        arrival = _number(getattr(metrics, "arrival_time", None))
        first = _number(getattr(metrics, "first_token_time", None))
        finished = _number(getattr(metrics, "finished_time", None))
        if arrival is not None and first is not None and first >= arrival:
            ttfts.append(first - arrival)
        if arrival is not None and finished is not None and finished >= arrival:
            latencies.append(finished - arrival)
    generated = request_count * output_length
    return {
        "elapsed_s": elapsed_s,
        "generated_tokens": generated,
        "request_count": request_count,
        "output_sha256": digest_json(token_ids),
        "output_tokens_per_second": generated / elapsed_s,
        "latency_available_requests": len(latencies),
        "ttft_available_requests": len(ttfts),
        "ttft_median_s": median(ttfts) if ttfts else None,
        "request_latency_median_s": median(latencies) if latencies else None,
    }


def worker_rows(raw: object, expected_workers: int) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or len(raw) != expected_workers:
        raise RuntimeError("missing or unexpected number of workers")
    if any(not isinstance(row, dict) for row in raw):
        raise RuntimeError("worker result must be an object")
    rows = cast(list[dict[str, Any]], raw)
    if {row.get("rank") for row in rows} != set(range(expected_workers)):
        raise RuntimeError("workers have missing or duplicate ranks")
    return sorted(rows, key=lambda row: cast(int, row["rank"]))


def worker_deltas(
    before: object, after: object, *, expected_workers: int
) -> dict[str, Any]:
    left, right = (
        worker_rows(before, expected_workers),
        worker_rows(after, expected_workers),
    )
    totals: dict[str, Any] = dict.fromkeys(COUNTERS, 0)
    totals["timing"] = dict.fromkeys(TIMINGS, 0)
    totals["per_rank"] = []
    for old, new in zip(left, right):
        rank_delta: dict[str, Any] = {"rank": new["rank"], "timing": {}}
        for name in COUNTERS:
            if type(old.get(name)) is not int or type(new.get(name)) is not int:
                raise RuntimeError(f"invalid counter {name}")
            delta = new[name] - old[name]
            if delta < 0:
                raise RuntimeError(f"counter {name} moved backwards")
            totals[name] += delta
            rank_delta[name] = delta
        for name in TIMINGS:
            a = _number(old.get("timing", {}).get(name))
            b = _number(new.get("timing", {}).get(name))
            if a is None or b is None or b < a:
                raise RuntimeError(f"invalid timing counter {name}")
            totals["timing"][name] += b - a
            rank_delta["timing"][name] = b - a
        rank_delta["timing"]["sample_count"] = int(rank_delta["timing"]["sample_count"])
        totals["per_rank"].append(rank_delta)
    totals["timing"]["sample_count"] = int(totals["timing"]["sample_count"])
    return totals


def compare_smoke(current: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    if current.get("input_sha256") != reference.get("input_sha256"):
        raise ValueError("reference smoke input differs")
    if not re.fullmatch(r"[0-9a-f]{64}", str(current.get("output_sha256", ""))):
        raise ValueError("smoke output hash missing")
    return bool(
        current["output_sha256"] == reference.get("output_sha256")
        and current.get("generated_tokens") == reference.get("generated_tokens")
    )


def resident_kv_budget(raw: object, expected_workers: int) -> int:
    rows = worker_rows(raw, expected_workers)
    available = [row.get("available_kv_cache_bytes") for row in rows]
    if all(type(value) is int and value > 0 for value in available):
        return min(cast(list[int], available))

    # vLLM 0.10.2 reports the committed KV allocation but may omit the
    # profiling-only available budget.  The committed allocation is the
    # strict fixed-KV reference in that case, provided every rank agrees.
    allocated = [row.get("kv_cache_allocated_bytes") for row in rows]
    if all(type(value) is int and value > 0 for value in allocated):
        return min(cast(list[int], allocated))
    raise RuntimeError("resident KV profiling budget is unavailable")


def validate_fixed_kv(
    current: object, reference: object, expected_workers: int
) -> None:
    for new, old in zip(
        worker_rows(current, expected_workers), worker_rows(reference, expected_workers)
    ):
        if (
            old.get("kv_cache_accounting_consistent") is not True
            or new.get("kv_cache_accounting_consistent") is not True
        ):
            raise RuntimeError(
                "fixed arm actual KV accounting is inconsistent or unavailable"
            )
        for name in ("kv_cache_allocated_bytes", "num_gpu_blocks"):
            if (
                type(old.get(name)) is not int
                or old[name] <= 0
                or new.get(name) != old[name]
            ):
                raise RuntimeError(
                    f"fixed arm actual KV differs or is unavailable: {name}"
                )


def validate_contract(current: Mapping[str, Any], reference: Mapping[str, Any]) -> None:
    differences = sorted(
        key
        for key in current.keys() | reference.keys()
        if current.get(key) != reference.get(key)
    )
    if differences:
        raise ValueError("comparison contract differs: " + ", ".join(differences))


def _resolved_policy(engine: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Capture defaults after vLLM resolves them, then verify critical switches."""
    config = engine.llm_engine.vllm_config
    fields = {
        "model_config": (
            "dtype",
            "enforce_eager",
            "max_model_len",
            "seed",
            "quantization",
            "runner_type",
            "max_seq_len_to_capture",
            "use_mla",
            "is_attention_free",
            "disable_sliding_window",
            "disable_cascade_attn",
            "use_async_output_proc",
        ),
        "cache_config": (
            "cache_dtype",
            "block_size",
            "enable_prefix_caching",
            "gpu_memory_utilization",
            "cpu_offload_gb",
            "swap_space_bytes",
            "mamba_cache_dtype",
            "mamba_ssm_cache_dtype",
            "mamba_block_size",
            "calculate_kv_scales",
            "sliding_window",
        ),
        "scheduler_config": (
            "max_num_seqs",
            "max_num_batched_tokens",
            "enable_chunked_prefill",
            "disable_hybrid_kv_cache_manager",
            "async_scheduling",
            "policy",
            "num_scheduler_steps",
            "max_num_partial_prefills",
            "max_long_partial_prefills",
            "long_prefill_token_threshold",
            "num_lookahead_slots",
            "preemption_mode",
        ),
        "parallel_config": (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "disable_custom_all_reduce",
            "enable_expert_parallel",
            "enable_eplb",
            "distributed_executor_backend",
            "decode_context_parallel_size",
        ),
        "compilation_config": (
            "level",
            "cudagraph_mode",
            "use_inductor",
            "cudagraph_num_of_warmups",
            "custom_ops",
        ),
    }
    policy: dict[str, Any] = {
        "requested": {
            key: value
            for key, value in arguments.items()
            if key not in ("model", "kv_cache_memory_bytes")
        }
    }
    for group, names in fields.items():
        values: dict[str, Any] = {}
        obj = getattr(config, group, None)
        for name in names:
            if hasattr(obj, name):
                value = getattr(obj, name)
                values[name] = (
                    value
                    if type(value) in (int, float, str, bool, type(None))
                    else str(value)
                )
        policy[group] = values
    required = {
        ("model_config", "enforce_eager"): True,
        ("model_config", "max_model_len"): arguments["max_model_len"],
        ("cache_config", "enable_prefix_caching"): False,
        ("cache_config", "gpu_memory_utilization"): arguments["gpu_memory_utilization"],
        ("scheduler_config", "max_num_seqs"): arguments["max_num_seqs"],
        ("scheduler_config", "max_num_batched_tokens"): arguments[
            "max_num_batched_tokens"
        ],
        ("scheduler_config", "enable_chunked_prefill"): True,
        ("parallel_config", "tensor_parallel_size"): arguments["tensor_parallel_size"],
        ("parallel_config", "disable_custom_all_reduce"): True,
        ("parallel_config", "data_parallel_size"): 1,
        ("parallel_config", "pipeline_parallel_size"): 1,
        ("compilation_config", "level"): 0,
    }
    for (group, name), expected in required.items():
        if policy[group].get(name) != expected:
            raise RuntimeError(f"resolved engine policy differs: {group}.{name}")
    if str(policy["model_config"].get("dtype")) not in ("torch.bfloat16", "bfloat16"):
        raise RuntimeError("resolved engine dtype is not bfloat16")
    if getattr(config.compilation_config, "custom_ops", None) != ["all"]:
        raise RuntimeError(
            "resolved engine policy differs: compilation_config.custom_ops"
        )
    return policy


def _configure_environment(
    config: PartialRunConfig, project_root: Path, layers: tuple[int, ...]
) -> None:
    for key in list(os.environ):
        if key.startswith("FLUXMOE_") and key != "FLUXMOE_PROCESS_NAME":
            del os.environ[key]
    os.environ.update(
        {
            "FLUXMOE_ENABLE": "0" if config.arm == "resident" else "1",
            "FLUXMOE_STORAGE_MODE": "partial-host",
            "FLUXMOE_MODEL_PATH": str(config.model_path),
            "FLUXMOE_PARTIAL_OFFLOAD_LAYERS": ",".join(str(layer) for layer in layers),
            "FLUXMOE_PARTIAL_STAGING_SLOTS": str(config.staging_slots),
            "FLUXMOE_PARTIAL_TIMING_SAMPLES": str(config.timing_samples),
            "VLLM_USE_V1": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "VLLM_NO_USAGE_STATS": "1",
        }
    )
    for variable, suffix in {
        "HF_HOME": "huggingface",
        "XDG_CACHE_HOME": "xdg",
        "VLLM_CACHE_ROOT": "vllm",
        "TORCH_HOME": "torch",
        "TORCH_EXTENSIONS_DIR": "torch-extensions",
        "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda",
        "TMPDIR": "tmp",
    }.items():
        target = project_root / "build" / "partial-cache" / suffix
        target.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(target)


def _memory(engine: Any, workers: int) -> list[dict[str, Any]]:
    rows = worker_rows(engine.collective_rpc("fluxmoe_worker_memory_stats"), workers)
    allowed = (
        "rank",
        "total_gpu_bytes",
        "free_gpu_bytes",
        "torch_allocated_bytes",
        "torch_reserved_bytes",
        "available_kv_cache_bytes",
        "model_memory_bytes",
        "kv_cache_allocated_bytes",
        "kv_cache_declared_bytes",
        "num_gpu_blocks",
    )
    clean: list[dict[str, Any]] = []
    for row in rows:
        values = {name: row.get(name) for name in allowed}
        if any(
            value is not None and (type(value) is not int or value < 0)
            for value in values.values()
        ):
            raise RuntimeError("invalid worker memory schema")
        values["kv_cache_accounting_consistent"] = row.get(
            "kv_cache_accounting_consistent"
        )
        clean.append(values)
    return clean


def _snapshot(
    engine: Any, workers: int, *, reset_timing: bool = False, synchronize: bool = True
) -> list[dict[str, Any]]:
    rows = worker_rows(
        engine.collective_rpc(
            "fluxmoe_partial_stats",
            kwargs={"synchronize": synchronize, "reset_timing": reset_timing},
        ),
        workers,
    )
    allowed = (
        "schema_version",
        "rank",
        "total_layers",
        "staging_slots",
        "layer_bytes",
        "host_source_bytes",
        "gpu_staging_bytes",
        "resident_routed_bytes",
        "net_freed_bytes",
        *COUNTERS,
    )
    clean: list[dict[str, Any]] = []
    for row in rows:
        item = {name: row.get(name) for name in allowed}
        if any(type(value) is not int or value < 0 for value in item.values()):
            raise RuntimeError("invalid partial worker counters")
        layers = row.get("offload_layers")
        if not isinstance(layers, list) or any(
            type(value) is not int or value < 0 for value in layers
        ):
            raise RuntimeError("invalid offloaded layer indices")
        item["offload_layers"] = layers
        for name in ("weights_expected", "weights_verified"):
            if name in row:
                if type(row[name]) is not int or row[name] < 0:
                    raise RuntimeError("invalid weight verification count")
                item[name] = row[name]
        timing = row.get("timing")
        if not isinstance(timing, dict):
            raise TypeError("partial timing counters unavailable")
        item["timing"] = {name: timing.get(name) for name in TIMINGS}
        if reset_timing:
            # RPC returns the old snapshot before clearing only timing totals.
            item["timing"] = dict.fromkeys(TIMINGS, 0)
        clean.append(item)
    return clean


def validate_partial_mechanism(
    raw: object,
    layers: tuple[int, ...],
    slots: int,
    workers: int,
) -> None:
    for row in worker_rows(raw, workers):
        if row.get("offload_layers") != list(layers):
            raise RuntimeError("worker offload placement differs")
        if not layers:
            if any(
                row.get(name) != 0
                for name in (
                    "h2d_bytes",
                    "copy_launches",
                    "host_source_bytes",
                    "gpu_staging_bytes",
                    "net_freed_bytes",
                )
            ):
                raise RuntimeError("zero-offload control unexpectedly moves weights")
            continue
        layer_bytes = row.get("layer_bytes")
        if (
            type(layer_bytes) is not int
            or layer_bytes <= 0
            or row.get("staging_slots") != slots
        ):
            raise RuntimeError("partial staging mechanism is unavailable")
        for name, expected in (
            ("host_source_bytes", len(layers) * layer_bytes),
            ("gpu_staging_bytes", slots * layer_bytes),
            ("net_freed_bytes", (len(layers) - slots) * layer_bytes),
        ):
            if row.get(name) != expected:
                raise RuntimeError(f"partial memory accounting differs: {name}")
        expected_weights = 2 * len(layers)
        if (
            row.get("weights_expected") != expected_weights
            or row.get("weights_verified") != expected_weights
        ):
            raise RuntimeError("partial BF16 weight verification is incomplete")
        if any(
            type(row.get(name)) is not int or row[name] <= 0
            for name in ("h2d_bytes", "copy_launches", "offload_forwards")
        ):
            raise RuntimeError(
                "partial offload mechanism has no executed transfer/forward evidence"
            )


def run_benchmark(
    config: PartialRunConfig,
    *,
    project_root: Path,
    run_dir: Path,
    resident_run: Path | None = None,
) -> Path:
    """Execute one immutable point. Exceptions retain atomic numeric progress."""
    from flexmoe.runtime.partial_plan import PartialPlan

    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}", run_dir.name):
        raise ValueError("run ID must be a short path-free identifier")
    run_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "arm": config.arm,
        "status": "running",
        "offload_count": config.offload_count,
        "staging_slots": config.staging_slots,
        "repetitions": [],
        "repetitions_completed": 0,
        "smoke_matches_resident": None,
    }
    atomic_json(run_dir / "summary.json", summary)
    try:
        workload = load_partial_workload(
            config.dataset_path,
            config.dataset_manifest,
            config.batch_size,
            config.context_length,
        )
        model_config = read_json(config.model_path / "config.json")
        text_config = model_config.get("text_config", model_config)
        total_layers = text_config["num_hidden_layers"]
        plan = PartialPlan.evenly_spaced(
            total_layers, config.offload_count, config.staging_slots
        )
        summary["offload_layers"] = list(plan.offload_layers)
        reference = read_json(resident_run / "summary.json") if resident_run else None
        if reference is not None and (
            reference.get("status") != "complete" or reference.get("arm") != "resident"
        ):
            raise ValueError("reference must be a completed resident point")
        if config.arm == "partial-fixed-kv" and reference is None:
            raise ValueError("partial-fixed-kv requires --resident-run")
        kv_bytes = (
            resident_kv_budget(reference["memory"], config.tensor_parallel_size)
            if config.arm == "partial-fixed-kv" and reference
            else None
        )
        summary["requested_kv_cache_bytes"] = kv_bytes
        arguments = engine_arguments(config, kv_bytes)
        commit = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
        ).strip()
        index_path = config.model_path / "model.safetensors.index.json"
        model_identity = {
            "config_sha256": sha256(
                (config.model_path / "config.json").read_bytes()
            ).hexdigest(),
            "index_sha256": sha256(index_path.read_bytes()).hexdigest()
            if index_path.is_file()
            else None,
            "path_sha256": sha256(
                str(config.model_path.resolve()).encode()
            ).hexdigest(),
        }
        _configure_environment(config, project_root, plan.offload_layers)
        vllm = importlib.import_module("vllm")
        torch = importlib.import_module("torch")
        versions = {
            "torch": str(torch.__version__),
            "vllm": str(vllm.__version__),
            "cuda": torch.version.cuda,
            "vllm_commit": os.environ.get("FLEXMOE_VLLM_COMMIT"),
        }
        if versions["vllm"].split("+")[0] != "0.10.2":
            raise RuntimeError("this protocol requires pinned vLLM 0.10.2")
        engine = vllm.LLM(**arguments)
        policy = _resolved_policy(engine, arguments)
        contract = {
            "commit": commit,
            "model_identity_sha256": digest_json(model_identity),
            "model_config_sha256": model_identity["config_sha256"],
            **workload.metadata,
            "batch_size": config.batch_size,
            "context_length": config.context_length,
            "output_length": config.output_length,
            "tensor_parallel_size": config.tensor_parallel_size,
            "dtype": "bfloat16",
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "seed": config.seed,
            "engine_policy_sha256": digest_json(policy),
            "versions": versions,
            "warmups": config.warmups,
            "timing_samples": config.timing_samples,
            "smoke_output_length": min(
                config.smoke_output_length, config.output_length
            ),
        }
        summary.update({"contract": contract, "engine_policy": policy})
        if reference is not None:
            validate_contract(contract, reference["contract"])
            summary["resident_run_id"] = reference["run_id"]
        workers = config.tensor_parallel_size
        _snapshot(engine, workers)
        memory = _memory(engine, workers)
        validate_fixed_kv(memory, memory, workers)
        summary["memory"] = memory
        summary["hardware_budget"] = {
            "total_gpu_bytes_by_rank": [row["total_gpu_bytes"] for row in memory],
            "requested_gpu_bytes_by_rank": [
                int(row["total_gpu_bytes"] * config.gpu_memory_utilization)
                for row in memory
            ],
        }
        if reference is not None:
            if summary["hardware_budget"] != reference.get("hardware_budget"):
                raise ValueError(
                    "comparison contract differs: physical GPU memory budget"
                )
            if config.arm == "partial-fixed-kv":
                validate_fixed_kv(memory, reference["memory"], workers)
        atomic_json(run_dir / "summary.json", summary)
        smoke_prompts = [{"prompt_token_ids": list(workload.prompts[0][:1024])}]
        smoke_length = min(config.smoke_output_length, config.output_length)

        def sampling(length: int) -> Any:
            return vllm.SamplingParams(
                temperature=0.0,
                min_tokens=length,
                max_tokens=length,
                ignore_eos=True,
                seed=config.seed,
                detokenize=False,
            )

        _snapshot(engine, workers)
        started = perf_counter()
        smoke_outputs = engine.generate(
            smoke_prompts, sampling(smoke_length), use_tqdm=False
        )
        engine.collective_rpc("fluxmoe_synchronize")
        smoke_elapsed = perf_counter() - started
        smoke_stats = _snapshot(engine, workers, synchronize=False)
        if config.arm != "resident":
            validate_partial_mechanism(
                smoke_stats, plan.offload_layers, config.staging_slots, workers
            )
            summary["mechanism_validated"] = True
        smoke = summarize_outputs(
            smoke_outputs,
            request_count=1,
            output_length=smoke_length,
            elapsed_s=smoke_elapsed,
        )
        smoke["input_sha256"] = digest_json([smoke_prompts[0]["prompt_token_ids"]])
        summary["smoke"] = smoke
        atomic_json(run_dir / "smoke.json", smoke)
        if reference is not None:
            summary["smoke_matches_resident"] = compare_smoke(smoke, reference["smoke"])
            if not summary["smoke_matches_resident"]:
                raise RuntimeError("batch1 smoke output hash differs from resident")
        prompts = [{"prompt_token_ids": list(row)} for row in workload.prompts]
        params = sampling(config.output_length)
        for _ in range(config.warmups):
            outputs = engine.generate(prompts, params, use_tqdm=False)
            summarize_outputs(
                outputs,
                request_count=config.batch_size,
                output_length=config.output_length,
                elapsed_s=1.0,
            )
        first_hash: str | None = None
        stable = True
        for repetition in range(config.repetitions):
            before = _snapshot(engine, workers, reset_timing=True)
            started = perf_counter()
            outputs = engine.generate(prompts, params, use_tqdm=False)
            engine.collective_rpc("fluxmoe_synchronize")
            elapsed = perf_counter() - started
            after = _snapshot(engine, workers, synchronize=False)
            result = summarize_outputs(
                outputs,
                request_count=config.batch_size,
                output_length=config.output_length,
                elapsed_s=elapsed,
            )
            result.update(
                {
                    "repetition": repetition,
                    "diagnostics": worker_deltas(
                        before, after, expected_workers=workers
                    ),
                }
            )
            if plan.offload_layers and any(
                row["h2d_bytes"] <= 0
                or row["copy_launches"] <= 0
                or row["offload_forwards"] <= 0
                for row in result["diagnostics"]["per_rank"]
            ):
                raise RuntimeError(
                    "measured repetition has no per-rank offload mechanism evidence"
                )
            result["timing_available"] = (
                result["diagnostics"]["timing"]["sample_count"] > 0
            )
            if first_hash is None:
                first_hash = result["output_sha256"]
            stable = stable and result["output_sha256"] == first_hash
            atomic_json(run_dir / f"rep-{repetition:03d}.json", result)
            summary["repetitions"].append(result)
            summary["repetitions_completed"] = repetition + 1
            summary["performance_outputs_stable"] = stable
            atomic_json(run_dir / "summary.json", summary)
        summary["final_memory"] = _memory(engine, workers)
        summary["partial_stats"] = _snapshot(engine, workers)
        throughputs = [
            row["output_tokens_per_second"] for row in summary["repetitions"]
        ]
        summary.update(
            {
                "status": "complete",
                "throughput_median": median(throughputs),
                "throughput_min": min(throughputs),
                "throughput_max": max(throughputs),
            }
        )
        if reference is not None:
            summary["performance_outputs_match_resident"] = [
                row["output_sha256"] for row in summary["repetitions"]
            ] == [row["output_sha256"] for row in reference["repetitions"]]
        atomic_json(run_dir / "summary.json", summary)
    except BaseException as error:
        summary["status"] = "failed"
        summary["error_type"] = type(error).__name__
        atomic_json(run_dir / "summary.json", summary)
        raise
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("benchmarks/data/sharegpt/qwen3next_1024_requests.jsonl.zst"),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("benchmarks/data/sharegpt/dataset_manifest.json"),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resident-run", type=Path)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    for flag, default in (
        ("batch-size", 256),
        ("context-length", 4096),
        ("output-length", 256),
        ("offload-count", 0),
        ("staging-slots", 1),
        ("warmups", 1),
        ("repetitions", 3),
        ("seed", 20260905),
        ("max-num-seqs", 256),
        ("max-num-batched-tokens", 8192),
        ("timing-samples", 128),
    ):
        parser.add_argument(f"--{flag}", type=int, default=default)
    args = parser.parse_args(argv)
    if not args.run_dir.resolve().is_relative_to(args.project_root.resolve()):
        parser.error("run-dir must remain inside project-root")
    values = vars(args).copy()
    for key in ("project_root", "run_dir", "resident_run"):
        values.pop(key)
    run_benchmark(
        PartialRunConfig(**values),
        project_root=args.project_root.resolve(),
        run_dir=args.run_dir.resolve(),
        resident_run=args.resident_run,
    )
    print(args.run_dir.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
