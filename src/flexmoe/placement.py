"""Deterministic bandwidth-proportional expert tensor placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from flexmoe.errors import ConfigurationError

BackendName = Literal["gpu_compressed", "host_pinned"]
TensorKind = Literal["w13", "w2"]


@dataclass(frozen=True, order=True)
class TensorSpec:
    """One independently placeable expert tensor."""

    tensor_id: str
    layer_idx: int
    expert_idx: int
    kind: TensorKind
    nbytes: int


@dataclass(frozen=True)
class BackendProfile:
    """Measured effective bandwidth and capacity for one backend."""

    name: BackendName
    bytes_per_second: float
    capacity_bytes: int


@dataclass(frozen=True, order=True)
class Placement:
    """Stable tensor location within one backend."""

    tensor_id: str
    backend: BackendName
    offset: int
    nbytes: int


def assign_tensors(
    tensors: tuple[TensorSpec, ...],
    profiles: tuple[BackendProfile, ...],
    gpu_budget_bytes: int,
) -> tuple[Placement, ...]:
    """Assign tensors to the backend with the earliest projected finish time."""

    if gpu_budget_bytes < 0:
        raise ConfigurationError("gpu_budget_bytes must be non-negative")
    if not profiles:
        raise ConfigurationError("at least one backend profile is required")
    if len({profile.name for profile in profiles}) != len(profiles):
        raise ConfigurationError("backend profile names must be unique")
    if any(
        profile.bytes_per_second <= 0 or profile.capacity_bytes < 0
        for profile in profiles
    ):
        raise ConfigurationError("backend bandwidth must be positive")
    if len({tensor.tensor_id for tensor in tensors}) != len(tensors):
        raise ConfigurationError("tensor IDs must be unique")
    if any(tensor.nbytes <= 0 for tensor in tensors):
        raise ConfigurationError("tensor sizes must be positive")

    ordered_tensors = sorted(
        tensors,
        key=lambda tensor: (
            tensor.layer_idx,
            tensor.kind,
            tensor.expert_idx,
            tensor.tensor_id,
        ),
    )
    ordered_profiles = sorted(profiles, key=lambda profile: profile.name)
    assigned_bytes: dict[BackendName, int] = {
        profile.name: 0 for profile in ordered_profiles
    }
    capacity: dict[BackendName, int] = {
        profile.name: (
            min(profile.capacity_bytes, gpu_budget_bytes)
            if profile.name == "gpu_compressed"
            else profile.capacity_bytes
        )
        for profile in ordered_profiles
    }
    placements: list[Placement] = []

    for tensor in ordered_tensors:
        eligible = [
            profile
            for profile in ordered_profiles
            if assigned_bytes[profile.name] + tensor.nbytes <= capacity[profile.name]
        ]
        if not eligible:
            raise ConfigurationError(
                f"insufficient backend capacity for tensor {tensor.tensor_id}"
            )
        selected = min(
            eligible,
            key=lambda profile: (
                (assigned_bytes[profile.name] + tensor.nbytes)
                / profile.bytes_per_second,
                profile.name,
            ),
        )
        offset = assigned_bytes[selected.name]
        placements.append(
            Placement(
                tensor_id=tensor.tensor_id,
                backend=selected.name,
                offset=offset,
                nbytes=tensor.nbytes,
            )
        )
        assigned_bytes[selected.name] += tensor.nbytes

    return tuple(placements)
