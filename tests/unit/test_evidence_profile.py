from __future__ import annotations

import pytest

from flexmoe.bench.evidence import validate_mechanism_counters


def _counters() -> dict[str, int]:
    return {
        "mapped_bytes": 1,
        "runtime_host_expert_h2d_bytes": 1,
        "runtime_host_copy_launches": 1,
        "gpu_decode_output_bytes": 1,
        "gpu_decode_launches": 1,
    }


def test_gpu_compressed_profile_requires_decode_without_runtime_host_copy() -> None:
    counters = _counters()
    counters["runtime_host_expert_h2d_bytes"] = 0
    counters["runtime_host_copy_launches"] = 0

    validate_mechanism_counters("fluxmoe-gpu-compressed", counters)

    counters["runtime_host_copy_launches"] = 1
    with pytest.raises(ValueError, match="requires runtime_host_copy_launches=0"):
        validate_mechanism_counters("fluxmoe-gpu-compressed", counters)


def test_resident_profile_rejects_storage_activity() -> None:
    counters = dict.fromkeys(_counters(), 0)
    validate_mechanism_counters("resident", counters)

    counters["mapped_bytes"] = 1
    with pytest.raises(ValueError, match="requires mapped_bytes=0"):
        validate_mechanism_counters("resident", counters)
