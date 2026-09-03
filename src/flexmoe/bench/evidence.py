"""Shared storage-mode-specific evidence contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CounterRequirement = Literal["zero", "positive", "any"]


@dataclass(frozen=True)
class EvidenceProfile:
    mapped_bytes: CounterRequirement
    runtime_host_expert_h2d_bytes: CounterRequirement
    runtime_host_copy_launches: CounterRequirement
    gpu_decode_output_bytes: CounterRequirement
    gpu_decode_launches: CounterRequirement


_PROFILES = {
    "resident": EvidenceProfile("zero", "zero", "zero", "zero", "zero"),
    "fluxmoe-fixed": EvidenceProfile(
        "positive", "positive", "positive", "positive", "positive"
    ),
    "fluxmoe-gpu-compressed": EvidenceProfile(
        "positive", "zero", "zero", "positive", "positive"
    ),
}


def evidence_profile(variant: str) -> EvidenceProfile:
    try:
        return _PROFILES[variant]
    except KeyError as error:
        raise ValueError(
            f"variant has no implemented evidence profile: {variant}"
        ) from error


def validate_mechanism_counters(variant: str, counters: dict[str, int]) -> None:
    profile = evidence_profile(variant)
    for field_name in profile.__dataclass_fields__:
        value = counters.get(field_name)
        if type(value) is not int or value < 0:
            raise ValueError(f"mechanism counter {field_name} must be non-negative")
        requirement = getattr(profile, field_name)
        if requirement == "zero" and value != 0:
            raise ValueError(f"{variant} requires {field_name}=0")
        if requirement == "positive" and value <= 0:
            raise ValueError(f"{variant} requires {field_name}>0")


__all__ = ["EvidenceProfile", "evidence_profile", "validate_mechanism_counters"]
