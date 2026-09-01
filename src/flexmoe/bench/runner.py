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
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

import torch
import yaml  # type: ignore[import-untyped]

from flexmoe.bench.workload import load_workload
from flexmoe.errors import UnsupportedModeError
from flexmoe.runtime.telemetry import JsonlTelemetry

BenchVariant = Literal[
    "resident",
    "vllm-o",
    "fluxmoe-fixed",
    "fluxmoe-dynamic",
    "fluxmoe-unbalanced",
    "pagedtensor-resident",
]

_VARIANTS = {
    "resident",
    "vllm-o",
    "fluxmoe-fixed",
    "fluxmoe-dynamic",
    "fluxmoe-unbalanced",
    "pagedtensor-resident",
}
_IMPLEMENTED_VARIANTS = {"resident", "fluxmoe-fixed"}


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
        gpu_decode_bytes_per_second=_float_field(
            data, "gpu_decode_bytes_per_second"
        ),
        host_h2d_bytes_per_second=_float_field(
            data, "host_h2d_bytes_per_second"
        ),
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
    if (
        config.gpu_decode_bytes_per_second <= 0
        or config.host_h2d_bytes_per_second <= 0
    ):
        raise ValueError("backend bandwidths must be positive")
    return config


def variant_environment(variant: str) -> dict[str, str]:
    environments = {
        "resident": {"FLUXMOE_ENABLE": "0"},
        "vllm-o": {"FLUXMOE_ENABLE": "0", "VLLM_O_ENABLE": "1"},
        "fluxmoe-fixed": {
            "FLUXMOE_ENABLE": "1",
            "FLUXMOE_PLANNER_MODE": "fixed",
            "FLUXMOE_VERIFY_WEIGHTS": "1",
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
            "FLUXMOE_HOST_H2D_BYTES_PER_SECOND": str(
                config.host_h2d_bytes_per_second
            ),
        }
    )
    os.environ.update(values)


def _execute_vllm(config: RunConfig) -> dict[str, object]:
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
        trust_remote_code=False,
        seed=config.seed,
    )
    sampling = sampling_class(
        temperature=0.0,
        max_tokens=config.output_length,
    )
    prompts = [
        {"prompt_token_ids": list(request.prompt_token_ids)}
        for request in workload.requests
    ]

    for _ in range(config.warmups):
        engine.generate(prompts, sampling, use_tqdm=False)
    repetitions: list[dict[str, object]] = []
    reference_tokens: list[list[int]] | None = None
    for repetition in range(config.repetitions):
        started = perf_counter()
        outputs = engine.generate(prompts, sampling, use_tqdm=False)
        elapsed = perf_counter() - started
        token_ids = [list(output.outputs[0].token_ids) for output in outputs]
        generated = sum(len(tokens) for tokens in token_ids)
        if reference_tokens is None:
            reference_tokens = token_ids
        elif token_ids != reference_tokens:
            raise RuntimeError("greedy output tokens changed between repetitions")
        repetitions.append(
            {
                "repetition": repetition,
                "elapsed_s": elapsed,
                "generated_tokens": generated,
                "output_tokens_per_second": generated / elapsed,
                "output_token_ids": token_ids,
            }
        )
    return {
        "schema_version": 1,
        "variant": config.variant,
        "dataset_sha256": workload.dataset_sha256,
        "repetitions": repetitions,
    }


def run_benchmark(config: RunConfig, *, project_root: Path, runs_root: Path) -> Path:
    if config.variant not in _IMPLEMENTED_VARIANTS:
        raise UnsupportedModeError(
            f"benchmark variant {config.variant} is a declared DEV_ONLY contract"
        )
    git_sha = _git_sha(project_root)
    run_dir = create_run_directory(runs_root, git_sha)
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
        },
    )
    _configure_environment(config)
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
        metrics = _execute_vllm(config)
        if os.environ.get("FLUXMOE_ENABLE") == "1":
            from flexmoe.vllm.bridge import require_active_registry

            counters = require_active_registry().mechanism_counters()
        else:
            counters = {
                "mapped_bytes": 0,
                "mapping_count": 0,
                "h2d_bytes": 0,
                "decompressed_bytes": 0,
                "weights_verified": 0,
                "weights_expected": 0,
            }
        metrics["mechanism_counters"] = counters
        metrics["correctness"] = {
            "output_tokens_match": False,
            "router_topk_match": False,
            "weights_bit_exact": (
                counters["weights_expected"] > 0
                and counters["weights_verified"] == counters["weights_expected"]
            ),
        }
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
        telemetry.emit("run_complete", {"repetitions": len(repetitions)})
        _write_json(run_dir / "metrics.json", metrics)
        _write_json(
            run_dir / "state.json",
            {"status": "MEASURED_UNVALIDATED", "git_sha": git_sha},
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
    arguments = parser.parse_args(argv)
    config = load_run_config(arguments.config)
    if arguments.batch_size is not None:
        config = replace(config, batch_size=arguments.batch_size)
    if arguments.context_length is not None:
        config = replace(config, context_length=arguments.context_length)
    run_dir = run_benchmark(
        config,
        project_root=arguments.project_root.resolve(),
        runs_root=arguments.runs_root.resolve(),
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BenchVariant",
    "RunConfig",
    "create_run_directory",
    "load_run_config",
    "run_benchmark",
    "variant_environment",
]
