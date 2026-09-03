"""Immutable run-directory creation and lazy vLLM benchmark execution."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Literal, cast

import torch
import yaml  # type: ignore[import-untyped]

from flexmoe.bench.workload import load_workload
from flexmoe.bench.router_trace import router_probes_match
from flexmoe.bench.evidence import validate_mechanism_counters
from flexmoe.errors import UnsupportedModeError
from flexmoe.runtime.telemetry import JsonlTelemetry

BenchVariant = Literal[
    "resident",
    "vllm-o",
    "fluxmoe-fixed",
    "fluxmoe-gpu-compressed",
    "fluxmoe-dynamic",
    "fluxmoe-unbalanced",
    "pagedtensor-resident",
]

_VARIANTS = {
    "resident",
    "vllm-o",
    "fluxmoe-fixed",
    "fluxmoe-gpu-compressed",
    "fluxmoe-dynamic",
    "fluxmoe-unbalanced",
    "pagedtensor-resident",
}
_IMPLEMENTED_VARIANTS = {"resident", "fluxmoe-fixed", "fluxmoe-gpu-compressed"}
_MECHANISM_COUNTER_NAMES = (
    "mapped_bytes",
    "mapping_count",
    "h2d_bytes",
    "decompressed_bytes",
    "weights_verified",
    "weights_expected",
    "startup_gpu_store_upload_bytes",
    "runtime_host_expert_h2d_bytes",
    "runtime_host_copy_launches",
    "gpu_decode_input_bytes",
    "gpu_decode_output_bytes",
    "gpu_decode_launches",
    "gpu_compressed_source_bytes",
    "gpu_compressed_storage_bytes",
    "expert_source_bytes",
)


@dataclass(frozen=True)
class RunConfig:
    schema_version: int
    variant: BenchVariant
    model_path: Path
    dataset_path: Path
    dataset_manifest: Path
    dataset_sha256: str
    batch_size: int
    context_length: int
    output_length: int
    tensor_parallel_size: int
    dtype: str
    greedy: bool
    enforce_eager: bool
    gpu_memory_utilization: float
    warmups: int
    repetitions: int
    seed: int
    gpu_compressed_budget_bytes: int
    host_capacity_bytes: int
    gpu_decode_bytes_per_second: float
    host_h2d_bytes_per_second: float
    storage_mode: Literal["hybrid", "gpu-compressed"]
    gpu_materialization_mode: Literal["expertwise", "batched"]
    minimum_kv_gain_bytes: int
    gpu_safety_margin_bytes: int


def _mapping(path: Path) -> dict[str, object]:
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError("benchmark config must be a YAML mapping")
    return cast(dict[str, object], parsed)


def _int_field(data: Mapping[str, object], name: str, *, minimum: int = 1) -> int:
    value = data.get(name)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}")
    return value


def _float_field(data: Mapping[str, object], name: str) -> float:
    value = data.get(name)
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be numeric")
    return float(cast(int | float, value))


def _string_field(data: Mapping[str, object], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _bool_field(data: Mapping[str, object], name: str) -> bool:
    value = data.get(name)
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def load_run_config(path: Path) -> RunConfig:
    data = _mapping(path)
    variant_value = _string_field(data, "variant")
    if variant_value not in _VARIANTS:
        raise ValueError(f"unsupported benchmark variant: {variant_value}")
    dataset_sha = _string_field(data, "dataset_sha256")
    if len(dataset_sha) != 64:
        raise ValueError("dataset_sha256 must contain 64 characters")
    gpu_utilization = _float_field(data, "gpu_memory_utilization")
    if not 0 < gpu_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be inside (0, 1]")
    config = RunConfig(
        schema_version=_int_field(data, "schema_version"),
        variant=cast(BenchVariant, variant_value),
        model_path=Path(_string_field(data, "model_path")),
        dataset_path=Path(_string_field(data, "dataset_path")),
        dataset_manifest=Path(_string_field(data, "dataset_manifest")),
        dataset_sha256=dataset_sha,
        batch_size=_int_field(data, "batch_size"),
        context_length=_int_field(data, "context_length"),
        output_length=_int_field(data, "output_length"),
        tensor_parallel_size=_int_field(data, "tensor_parallel_size"),
        dtype=_string_field(data, "dtype"),
        greedy=_bool_field(data, "greedy"),
        enforce_eager=_bool_field(data, "enforce_eager"),
        gpu_memory_utilization=gpu_utilization,
        warmups=_int_field(data, "warmups"),
        repetitions=_int_field(data, "repetitions"),
        seed=_int_field(data, "seed", minimum=0),
        gpu_compressed_budget_bytes=_int_field(
            data, "gpu_compressed_budget_bytes", minimum=0
        ),
        host_capacity_bytes=_int_field(data, "host_capacity_bytes"),
        gpu_decode_bytes_per_second=_float_field(data, "gpu_decode_bytes_per_second"),
        host_h2d_bytes_per_second=_float_field(data, "host_h2d_bytes_per_second"),
        storage_mode=cast(
            Literal["hybrid", "gpu-compressed"],
            _string_field(data, "storage_mode"),
        ),
        gpu_materialization_mode=cast(
            Literal["expertwise", "batched"],
            _string_field(data, "gpu_materialization_mode"),
        ),
        minimum_kv_gain_bytes=_int_field(data, "minimum_kv_gain_bytes", minimum=0),
        gpu_safety_margin_bytes=_int_field(data, "gpu_safety_margin_bytes", minimum=0),
    )
    if config.schema_version != 1:
        raise ValueError("unsupported benchmark config schema")
    if config.tensor_parallel_size != 4:
        raise ValueError("formal reproduction requires tensor_parallel_size=4")
    if config.dtype != "bfloat16":
        raise ValueError("formal reproduction requires bfloat16")
    if not config.greedy or not config.enforce_eager:
        raise ValueError("formal reproduction requires greedy eager execution")
    if config.warmups != 3 or config.repetitions != 3:
        raise ValueError("formal reproduction requires 3 warmups and 3 repetitions")
    if config.gpu_decode_bytes_per_second <= 0 or config.host_h2d_bytes_per_second <= 0:
        raise ValueError("backend bandwidths must be positive")
    if config.storage_mode not in {"hybrid", "gpu-compressed"}:
        raise ValueError("storage_mode must be hybrid or gpu-compressed")
    if config.gpu_materialization_mode not in {"expertwise", "batched"}:
        raise ValueError("gpu_materialization_mode must be expertwise or batched")
    if config.variant == "fluxmoe-gpu-compressed":
        if config.storage_mode != "gpu-compressed":
            raise ValueError("fluxmoe-gpu-compressed requires gpu-compressed storage")
        if config.gpu_materialization_mode != "batched":
            raise ValueError("fluxmoe-gpu-compressed requires batched materialization")
    elif config.storage_mode == "gpu-compressed":
        raise ValueError("gpu-compressed storage requires fluxmoe-gpu-compressed")
    return config


def variant_environment(variant: str) -> dict[str, str]:
    environments = {
        "resident": {"FLUXMOE_ENABLE": "0"},
        "vllm-o": {"FLUXMOE_ENABLE": "0", "VLLM_O_ENABLE": "1"},
        "fluxmoe-fixed": {
            "FLUXMOE_ENABLE": "1",
            "FLUXMOE_PLANNER_MODE": "fixed",
            "FLUXMOE_STORAGE_MODE": "hybrid",
            "FLUXMOE_GPU_MATERIALIZATION_MODE": "expertwise",
        },
        "fluxmoe-gpu-compressed": {
            "FLUXMOE_ENABLE": "1",
            "FLUXMOE_PLANNER_MODE": "fixed",
            "FLUXMOE_STORAGE_MODE": "gpu-compressed",
            "FLUXMOE_GPU_MATERIALIZATION_MODE": "batched",
        },
        "fluxmoe-dynamic": {
            "FLUXMOE_ENABLE": "1",
            "FLUXMOE_PLANNER_MODE": "dynamic",
        },
        "fluxmoe-unbalanced": {
            "FLUXMOE_ENABLE": "1",
            "FLUXMOE_PLANNER_MODE": "dynamic-unbalanced",
        },
        "pagedtensor-resident": {
            "FLUXMOE_ENABLE": "1",
            "FLUXMOE_PLANNER_MODE": "pagedtensor-resident",
        },
    }
    try:
        return dict(environments[variant])
    except KeyError as error:
        raise ValueError(f"unsupported benchmark variant: {variant}") from error


def create_run_directory(
    root: Path, git_sha: str, timestamp: datetime | None = None
) -> Path:
    if len(git_sha) != 40 or any(char not in "0123456789abcdef" for char in git_sha):
        raise ValueError("git_sha must be a 40-character lowercase hex digest")
    instant = timestamp or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        raise ValueError("run timestamp must be timezone-aware")
    name = f"{instant.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}-{git_sha[:12]}"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / name
    run_dir.mkdir()
    return run_dir


def _git_sha(project_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _router_trace_manifest(
    root: Path, *, expected_ranks: int, probe_records: int
) -> dict[str, dict[str, int | str]]:
    files = sorted(root.glob("rank-*.jsonl"))
    expected_files = [root / f"rank-{rank}.jsonl" for rank in range(expected_ranks)]
    if files != expected_files:
        raise RuntimeError(f"router trace rank files are incomplete under {root}")
    if probe_records <= 0:
        raise ValueError("router probe must contain at least one record")
    manifest: dict[str, dict[str, int | str]] = {}
    for path in files:
        payload = path.read_bytes()
        if not payload:
            raise RuntimeError(f"router trace is empty: {path}")
        lines = payload.splitlines(keepends=True)
        if len(lines) < probe_records:
            raise RuntimeError(f"router trace has an incomplete probe: {path}")
        probe_payload = b"".join(lines[:probe_records])
        manifest[path.name] = {
            "sha256": sha256(payload).hexdigest(),
            "line_count": len(lines),
            "probe_sha256": sha256(probe_payload).hexdigest(),
            "probe_line_count": probe_records,
        }
    return manifest


def _model_hidden_layers(model_path: Path) -> int:
    parsed = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError("model config must be a JSON object")
    hidden_layers = parsed.get("num_hidden_layers")
    if type(hidden_layers) is not int or hidden_layers <= 0:
        raise ValueError("model config has invalid num_hidden_layers")
    return hidden_layers


def _median_throughput(metrics: Mapping[str, object]) -> float:
    repetitions = metrics.get("repetitions")
    if not isinstance(repetitions, list) or not repetitions:
        raise ValueError("metrics contain no repetitions")
    values: list[float] = []
    for repetition in repetitions:
        if not isinstance(repetition, dict):
            raise TypeError("repetition metrics must be objects")
        value = repetition.get("output_tokens_per_second")
        if type(value) not in {int, float}:
            raise TypeError("output_tokens_per_second must be numeric")
        values.append(float(cast(int | float, value)))
    return median(values)


def compare_reference(
    config: RunConfig,
    metrics: dict[str, object],
    router_manifest: dict[str, dict[str, int | str]],
    reference_run: Path,
) -> tuple[bool, bool, float]:
    reference_config = json.loads(
        (reference_run / "config.json").read_text(encoding="utf-8")
    )
    reference_metrics = json.loads(
        (reference_run / "metrics.json").read_text(encoding="utf-8")
    )
    if not isinstance(reference_config, dict) or not isinstance(
        reference_metrics, dict
    ):
        raise TypeError("reference run artifacts must be JSON objects")
    expected_fields: dict[str, object] = {
        "model_path": str(config.model_path),
        "dataset_sha256": config.dataset_sha256,
        "batch_size": config.batch_size,
        "context_length": config.context_length,
        "output_length": config.output_length,
        "tensor_parallel_size": config.tensor_parallel_size,
        "dtype": config.dtype,
        "seed": config.seed,
    }
    for field_name, expected in expected_fields.items():
        if reference_config.get(field_name) != expected:
            raise ValueError(f"reference run differs in {field_name}")
    if reference_config.get("variant") != "resident":
        raise ValueError("reference run must use the resident variant")

    reference_repetitions = reference_metrics.get("repetitions")
    current_repetitions = metrics.get("repetitions")
    if not isinstance(reference_repetitions, list) or not isinstance(
        current_repetitions, list
    ):
        raise TypeError("run repetitions must be lists")
    reference_tokens = [
        repetition.get("output_token_ids")
        for repetition in reference_repetitions
        if isinstance(repetition, dict)
    ]
    current_tokens = [
        repetition.get("output_token_ids")
        for repetition in current_repetitions
        if isinstance(repetition, dict)
    ]
    output_tokens_match = (
        len(reference_tokens) == len(reference_repetitions)
        and len(current_tokens) == len(current_repetitions)
        and current_tokens == reference_tokens
    )
    metrics["performance_output_tokens_match"] = output_tokens_match
    reference_router = reference_metrics.get("router_trace")
    metrics["router_full_trace_match"] = reference_router == router_manifest
    router_topk_match = router_probes_match(reference_router, router_manifest)
    reference_throughput = _median_throughput(
        cast(dict[str, object], reference_metrics)
    )
    current_throughput = _median_throughput(metrics)
    stressed_delta = (current_throughput - reference_throughput) / (
        reference_throughput
    )
    return output_tokens_match, router_topk_match, stressed_delta


def validate_resident_oom(config: RunConfig, failure_run: Path, git_sha: str) -> Path:
    failure_path = failure_run.resolve()
    failure = json.loads((failure_path / "error.json").read_text(encoding="utf-8"))
    failure_config = json.loads(
        (failure_path / "config.json").read_text(encoding="utf-8")
    )
    if not isinstance(failure, dict) or not isinstance(failure_config, dict):
        raise TypeError("resident failure artifacts must be JSON objects")
    failure_message = str(failure.get("message", "")).lower()
    if "out of memory" not in failure_message and "oom" not in failure_message:
        raise ValueError("resident failure is not an OOM/capacity result")
    for field_name, expected in {
        "git_sha": git_sha,
        "variant": "resident",
        "model_path": str(config.model_path),
        "dataset_sha256": config.dataset_sha256,
        "batch_size": config.batch_size,
        "context_length": config.context_length,
        "output_length": config.output_length,
        "tensor_parallel_size": config.tensor_parallel_size,
    }.items():
        if failure_config.get(field_name) != expected:
            raise ValueError(f"resident OOM differs in {field_name}")
    return failure_path


def _configure_environment(config: RunConfig) -> None:
    values = variant_environment(config.variant)
    values.update(
        {
            "FLUXMOE_MODEL_PATH": str(config.model_path),
            "FLUXMOE_GPU_COMPRESSED_BUDGET_BYTES": str(
                config.gpu_compressed_budget_bytes
            ),
            "FLUXMOE_HOST_CAPACITY_BYTES": str(config.host_capacity_bytes),
            "FLUXMOE_GPU_DECODE_BYTES_PER_SECOND": str(
                config.gpu_decode_bytes_per_second
            ),
            "FLUXMOE_HOST_H2D_BYTES_PER_SECOND": str(config.host_h2d_bytes_per_second),
            "FLUXMOE_STORAGE_MODE": config.storage_mode,
            "FLUXMOE_GPU_MATERIALIZATION_MODE": config.gpu_materialization_mode,
            "FLUXMOE_MINIMUM_KV_GAIN_BYTES": str(config.minimum_kv_gain_bytes),
            "FLUXMOE_GPU_SAFETY_MARGIN_BYTES": str(config.gpu_safety_margin_bytes),
            "FLUXMOE_GPU_MEMORY_UTILIZATION": str(config.gpu_memory_utilization),
        }
    )
    os.environ.update(values)


def _aggregate_worker_counters(
    worker_counters: object, *, expected_workers: int
) -> dict[str, int]:
    if not isinstance(worker_counters, list):
        raise TypeError("vLLM worker mechanism counters must be a list")
    if len(worker_counters) != expected_workers:
        raise RuntimeError(
            "vLLM returned mechanism counters from "
            f"{len(worker_counters)} workers; expected {expected_workers}"
        )
    totals = dict.fromkeys(_MECHANISM_COUNTER_NAMES, 0)
    expected_names = set(_MECHANISM_COUNTER_NAMES)
    for worker_index, counters in enumerate(worker_counters):
        if not isinstance(counters, dict):
            raise TypeError(
                f"mechanism counters from worker {worker_index} are not an object"
            )
        if set(counters) != expected_names:
            raise RuntimeError(
                f"mechanism counters from worker {worker_index} have unexpected fields"
            )
        for name in _MECHANISM_COUNTER_NAMES:
            value = counters[name]
            if type(value) is not int or value < 0:
                raise TypeError(
                    f"mechanism counter {name} from worker {worker_index} "
                    "must be a non-negative int"
                )
            totals[name] += value
    return totals


def _execute_vllm(
    config: RunConfig, *, require_stable_output: bool
) -> dict[str, object]:
    workload = load_workload(
        dataset=config.dataset_path,
        manifest=config.dataset_manifest,
        batch_size=config.batch_size,
        context_length=config.context_length,
    )
    if workload.dataset_sha256 != config.dataset_sha256:
        raise ValueError("configured dataset SHA256 does not match the manifest")
    vllm = importlib.import_module("vllm")
    llm_class = vllm.LLM
    sampling_class = vllm.SamplingParams
    engine = llm_class(
        model=str(config.model_path),
        tensor_parallel_size=config.tensor_parallel_size,
        dtype=config.dtype,
        enforce_eager=config.enforce_eager,
        gpu_memory_utilization=config.gpu_memory_utilization,
        # vLLM custom all-reduce stalls before weight loading on this host's
        # four-GPU partition. Use the standard NCCL collective instead.
        disable_custom_all_reduce=True,
        trust_remote_code=False,
        seed=config.seed,
        worker_extension_cls=(
            "flexmoe.vllm.bridge.FluxMoEWorkerExtension"
            if os.environ.get("FLUXMOE_ENABLE") == "1"
            else ""
        ),
    )
    sampling = sampling_class(
        temperature=0.0,
        min_tokens=config.output_length,
        max_tokens=config.output_length,
        ignore_eos=True,
        seed=config.seed,
    )
    prompts = [
        {"prompt_token_ids": list(request.prompt_token_ids)}
        for request in workload.requests
    ]

    for _ in range(config.warmups):
        engine.generate(prompts, sampling, use_tqdm=False)
    repetitions: list[dict[str, object]] = []
    reference_tokens: list[list[int]] | None = None
    output_tokens_stable = True
    for repetition in range(config.repetitions):
        started = perf_counter()
        outputs = engine.generate(prompts, sampling, use_tqdm=False)
        elapsed = perf_counter() - started
        token_ids = [list(output.outputs[0].token_ids) for output in outputs]
        if len(token_ids) != config.batch_size or any(
            len(tokens) != config.output_length for tokens in token_ids
        ):
            raise RuntimeError(
                "fixed-output protocol requires every request to generate "
                f"exactly {config.output_length} tokens"
            )
        generated = sum(len(tokens) for tokens in token_ids)
        if reference_tokens is None:
            reference_tokens = token_ids
        elif token_ids != reference_tokens:
            output_tokens_stable = False
            if require_stable_output:
                raise RuntimeError("greedy output tokens changed between repetitions")
        repetitions.append(
            {
                "repetition": repetition,
                "elapsed_s": elapsed,
                "generated_tokens": generated,
                "request_count": len(token_ids),
                "output_lengths": [len(tokens) for tokens in token_ids],
                "output_tokens_per_second": generated / elapsed,
                "output_token_ids": token_ids,
            }
        )
    if os.environ.get("FLUXMOE_ENABLE") == "1":
        counters = _aggregate_worker_counters(
            engine.collective_rpc("fluxmoe_mechanism_counters"),
            expected_workers=config.tensor_parallel_size,
        )
    else:
        counters = dict.fromkeys(_MECHANISM_COUNTER_NAMES, 0)
    return {
        "schema_version": 1,
        "variant": config.variant,
        "dataset_sha256": workload.dataset_sha256,
        "repetitions": repetitions,
        "output_tokens_stable": output_tokens_stable,
        "mechanism_counters": counters,
    }


def run_benchmark(
    config: RunConfig,
    *,
    project_root: Path,
    runs_root: Path,
    reference_run: Path | None = None,
    correctness_mode: bool = False,
    correctness_evidence: Path | None = None,
    resident_failure_run: Path | None = None,
    result_path_file: Path | None = None,
) -> Path:
    if config.variant not in _IMPLEMENTED_VARIANTS:
        raise UnsupportedModeError(
            f"benchmark variant {config.variant} is a declared DEV_ONLY contract"
        )
    git_sha = _git_sha(project_root)
    validated_resident_failure = (
        validate_resident_oom(config, resident_failure_run, git_sha)
        if resident_failure_run is not None
        else None
    )
    preflight_source = runs_root / f"preflight-{git_sha}.json"
    if not preflight_source.is_file():
        raise FileNotFoundError(
            f"current checkout has no preflight artifact: {preflight_source}"
        )
    preflight_payload = json.loads(preflight_source.read_text(encoding="utf-8"))
    if (
        not isinstance(preflight_payload, dict)
        or preflight_payload.get("ok") is not True
    ):
        raise RuntimeError(f"preflight did not pass: {preflight_source}")
    run_dir = create_run_directory(runs_root, git_sha)
    if result_path_file is not None:
        result_path_file.parent.mkdir(parents=True, exist_ok=True)
        result_path_file.write_text(str(run_dir.resolve()) + "\n", encoding="utf-8")
    (run_dir / "preflight.json").write_text(
        json.dumps(preflight_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_json = asdict(config)
    for path_field in ("model_path", "dataset_path", "dataset_manifest"):
        config_json[path_field] = str(config_json[path_field])
    _write_json(run_dir / "config.json", {**config_json, "git_sha": git_sha})
    _write_json(
        run_dir / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "git_sha": git_sha,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "vllm_disable_custom_all_reduce": True,
            "process_name": os.environ.get("FLUXMOE_PROCESS_NAME", "wth333"),
        },
    )
    _configure_environment(config)
    os.environ["FLUXMOE_TRACE_ROUTER"] = "1" if correctness_mode else "0"
    os.environ["FLUXMOE_VERIFY_WEIGHTS"] = (
        "1"
        if correctness_mode
        and config.variant
        in {
            "fluxmoe-fixed",
            "fluxmoe-gpu-compressed",
        }
        else "0"
    )
    router_trace_root = run_dir / "router"
    if correctness_mode:
        router_trace_root.mkdir()
        os.environ["FLUXMOE_ROUTER_TRACE_DIR"] = str(router_trace_root)
    else:
        os.environ.pop("FLUXMOE_ROUTER_TRACE_DIR", None)
    telemetry = JsonlTelemetry(
        run_dir / "events.jsonl",
        run_id=run_dir.name,
        git_sha=git_sha,
        rank=0,
        variant=config.variant,
    )
    telemetry.emit(
        "run_start",
        {
            "batch_size": config.batch_size,
            "context_length": config.context_length,
            "output_length": config.output_length,
        },
    )
    try:
        metrics = _execute_vllm(config, require_stable_output=correctness_mode)
        raw_counters = metrics.get("mechanism_counters")
        if not isinstance(raw_counters, dict):
            raise TypeError("benchmark mechanism counters are missing")
        counters = cast(dict[str, int], raw_counters)
        router_manifest = (
            _router_trace_manifest(
                router_trace_root,
                expected_ranks=config.tensor_parallel_size,
                probe_records=2 * _model_hidden_layers(config.model_path),
            )
            if correctness_mode
            else {}
        )
        if router_manifest:
            metrics["router_trace"] = router_manifest
        weights_bit_exact = (
            counters["weights_expected"] > 0
            and counters["weights_verified"] == counters["weights_expected"]
        )
        output_tokens_match = False
        router_topk_match = False
        stressed_delta: float | None = None
        if reference_run is not None:
            output_tokens_match, router_topk_match, stressed_delta = compare_reference(
                config,
                metrics,
                router_manifest,
                reference_run.resolve(),
            )
            metrics["reference_run"] = str(reference_run.resolve())
            metrics["stressed_delta"] = stressed_delta
        if correctness_evidence is not None:
            from flexmoe.bench.report import validate_run_directory

            evidence_path = correctness_evidence.resolve()
            delegated_evidence = validate_run_directory(evidence_path)
            evidence_config = json.loads(
                (evidence_path / "config.json").read_text(encoding="utf-8")
            )
            if not isinstance(evidence_config, dict):
                raise TypeError("correctness evidence config must be an object")
            for field_name, expected in {
                "git_sha": git_sha,
                "variant": config.variant,
                "model_path": str(config.model_path),
                "dataset_sha256": config.dataset_sha256,
                "tensor_parallel_size": config.tensor_parallel_size,
            }.items():
                if evidence_config.get(field_name) != expected:
                    raise ValueError(f"correctness evidence differs in {field_name}")
            metrics["correctness_evidence"] = str(evidence_path)
            output_tokens_match = delegated_evidence.output_tokens_match
            router_topk_match = delegated_evidence.router_topk_match
            weights_bit_exact = delegated_evidence.weights_bit_exact
        delayed_oom = False
        if validated_resident_failure is not None:
            delayed_oom = True
            metrics["resident_failure_run"] = str(validated_resident_failure)
            metrics["delayed_oom"] = True
        correctness = {
            "output_tokens_match": output_tokens_match,
            "router_topk_match": router_topk_match,
            "weights_bit_exact": weights_bit_exact,
        }
        metrics["correctness"] = correctness
        repetitions = cast(list[dict[str, object]], metrics["repetitions"])
        for repetition in repetitions:
            telemetry.emit(
                "repetition",
                {
                    "repetition": cast(int, repetition["repetition"]),
                    "elapsed_s": cast(float, repetition["elapsed_s"]),
                    "generated_tokens": cast(int, repetition["generated_tokens"]),
                    "output_tokens_per_second": cast(
                        float, repetition["output_tokens_per_second"]
                    ),
                },
            )
        telemetry.emit("mechanism", counters)
        telemetry.emit("correctness", correctness)
        telemetry.emit("run_complete", {"repetitions": len(repetitions)})
        _write_json(run_dir / "metrics.json", metrics)
        if config.variant == "resident":
            state_status = "BASELINE_COMPLETE"
        else:
            try:
                validate_mechanism_counters(config.variant, counters)
                mechanism_valid = True
            except ValueError:
                mechanism_valid = False
            state_status = (
                "COMPLETE"
                if (
                    (reference_run is not None or delayed_oom)
                    and output_tokens_match
                    and router_topk_match
                    and weights_bit_exact
                    and mechanism_valid
                )
                else "MEASURED_UNVALIDATED"
            )
        _write_json(
            run_dir / "state.json",
            {"status": state_status, "git_sha": git_sha},
        )
    except Exception as error:
        telemetry.emit(
            "run_error",
            {"error_type": type(error).__name__, "message": str(error)},
        )
        _write_json(
            run_dir / "error.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        _write_json(
            run_dir / "state.json",
            {"status": "FAILED", "git_sha": git_sha},
        )
        raise
    finally:
        telemetry.close()
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--output-length", type=int)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--correctness-mode", action="store_true")
    parser.add_argument("--correctness-evidence", type=Path)
    parser.add_argument("--resident-failure-run", type=Path)
    parser.add_argument("--result-path-file", type=Path)
    arguments = parser.parse_args(argv)
    config = load_run_config(arguments.config)
    if arguments.batch_size is not None:
        config = replace(config, batch_size=arguments.batch_size)
    if arguments.context_length is not None:
        config = replace(config, context_length=arguments.context_length)
    if arguments.output_length is not None:
        if arguments.output_length <= 0:
            parser.error("--output-length must be positive")
        config = replace(config, output_length=arguments.output_length)
    run_dir = run_benchmark(
        config,
        project_root=arguments.project_root.resolve(),
        runs_root=arguments.runs_root.resolve(),
        reference_run=(
            arguments.reference_run.resolve()
            if arguments.reference_run is not None
            else None
        ),
        correctness_mode=arguments.correctness_mode,
        correctness_evidence=(
            arguments.correctness_evidence.resolve()
            if arguments.correctness_evidence is not None
            else None
        ),
        resident_failure_run=(
            arguments.resident_failure_run.resolve()
            if arguments.resident_failure_run is not None
            else None
        ),
        result_path_file=arguments.result_path_file,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BenchVariant",
    "RunConfig",
    "compare_reference",
    "create_run_directory",
    "load_run_config",
    "run_benchmark",
    "validate_resident_oom",
    "variant_environment",
]
