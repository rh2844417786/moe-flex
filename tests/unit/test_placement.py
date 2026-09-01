from itertools import pairwise

import pytest

from flexmoe.errors import ConfigurationError
from flexmoe.placement import BackendProfile, TensorSpec, assign_tensors


def test_assignment_is_deterministic_and_respects_gpu_budget() -> None:
    tensors = tuple(
        TensorSpec(f"t{index}", index // 4, index % 4, "w13", 10)
        for index in range(12)
    )
    profiles = (
        BackendProfile("gpu_compressed", 300.0, 40),
        BackendProfile("host_pinned", 100.0, 1_000),
    )

    first = assign_tensors(tensors, profiles, gpu_budget_bytes=40)
    second = assign_tensors(tuple(reversed(tensors)), profiles, gpu_budget_bytes=40)

    assert first == second
    assert len(first) == len(tensors)
    assert sum(
        item.nbytes for item in first if item.backend == "gpu_compressed"
    ) <= 40


def test_assignment_offsets_do_not_overlap() -> None:
    tensors = (
        TensorSpec("a", 0, 0, "w13", 10),
        TensorSpec("b", 0, 0, "w2", 20),
        TensorSpec("c", 0, 1, "w13", 30),
    )
    profiles = (
        BackendProfile("gpu_compressed", 1.0, 100),
        BackendProfile("host_pinned", 1.0, 100),
    )

    placements = assign_tensors(tensors, profiles, gpu_budget_bytes=100)

    for backend in {item.backend for item in placements}:
        selected = sorted(
            (item for item in placements if item.backend == backend),
            key=lambda item: item.offset,
        )
        assert all(
            left.offset + left.nbytes <= right.offset
            for left, right in pairwise(selected)
        )


def test_assignment_rejects_insufficient_capacity() -> None:
    tensors = (TensorSpec("too-large", 0, 0, "w13", 30),)
    profiles = (
        BackendProfile("gpu_compressed", 1.0, 10),
        BackendProfile("host_pinned", 1.0, 10),
    )

    with pytest.raises(ConfigurationError, match="insufficient backend capacity"):
        assign_tensors(tensors, profiles, gpu_budget_bytes=10)
