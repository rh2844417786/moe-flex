"""Immutable configuration contracts for FluxMoE runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

RunStatus = Literal["SUPPORTED", "MIXED", "NOT_SUPPORTED", "INCONCLUSIVE"]
Variant = Literal[
    "vllm-resident",
    "vllm-o",
    "fluxmoe-fixed",
    "fluxmoe-host-offload",
    "fluxmoe-routed-host-offload",
    "fluxmoe-gpu-compressed",
    "fluxmoe-dynamic",
    "fluxmoe-dynamic-unbalanced",
    "pagedtensor-resident",
]


@dataclass(frozen=True)
class ModelSpec:
    """Expected on-disk checkpoint identity."""

    path: Path
    architecture: str
    dtype: str
    expected_shards: int


@dataclass(frozen=True)
class PlannerConfig:
    """Residency-controller thresholds."""

    io_bound_threshold: float = 0.9
    compute_bound_threshold: float = 1.0
    decision_interval: int = 300


@dataclass(frozen=True)
class BenchmarkConfig:
    """One benchmark point and system variant."""

    variant: Variant
    batch_size: int
    context_length: int
    output_length: int
    tensor_parallel_size: int = 4
    greedy: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    """Complete immutable run configuration."""

    project_root: Path
    model: ModelSpec
    planner: PlannerConfig
    benchmark: BenchmarkConfig
    gpu_ids: tuple[int, ...]
    vllm_commit: str = "01efc7ef781391e744ed08c3292817a773d654e6"
