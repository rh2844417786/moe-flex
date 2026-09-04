from __future__ import annotations

import pytest

from flexmoe.bench.evidence import validate_mechanism_counters


def _counters() -> dict[str, int]:
    return {
        "mapped_bytes": 1,
        "runtime_host_expert_h2d_bytes": 1,
        "runtime_host_copy_launches": 1,
        "gpu_decode_input_bytes": 1,
        "gpu_decode_output_bytes": 1,
        "gpu_decode_launches": 1,
        "gpu_compressed_source_bytes": 1,
        "gpu_compressed_storage_bytes": 1,
        "startup_gpu_store_upload_bytes": 1,
        "routed_layer_loads": 0,
        "routed_expert_loads": 0,
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


def test_host_offload_requires_host_copies_and_rejects_compression() -> None:
    counters = _counters()
    for name in (
        "gpu_decode_input_bytes",
        "gpu_decode_output_bytes",
        "gpu_decode_launches",
        "gpu_compressed_source_bytes",
        "gpu_compressed_storage_bytes",
        "startup_gpu_store_upload_bytes",
    ):
        counters[name] = 0

    validate_mechanism_counters("fluxmoe-host-offload", counters)

    counters["gpu_compressed_storage_bytes"] = 1
    with pytest.raises(
        ValueError, match="requires gpu_compressed_storage_bytes=0"
    ):
        validate_mechanism_counters("fluxmoe-host-offload", counters)


def test_routed_host_offload_requires_routed_load_evidence() -> None:
    counters = _counters()
    for name in (
        "gpu_decode_input_bytes",
        "gpu_decode_output_bytes",
        "gpu_decode_launches",
        "gpu_compressed_source_bytes",
        "gpu_compressed_storage_bytes",
        "startup_gpu_store_upload_bytes",
    ):
        counters[name] = 0
    counters["routed_layer_loads"] = 2
    counters["routed_expert_loads"] = 7

    validate_mechanism_counters("fluxmoe-routed-host-offload", counters)

    counters["routed_expert_loads"] = 0
    with pytest.raises(ValueError, match="requires routed_expert_loads>0"):
        validate_mechanism_counters("fluxmoe-routed-host-offload", counters)
