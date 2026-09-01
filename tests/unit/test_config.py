from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from flexmoe.config import (
    BenchmarkConfig,
    ModelSpec,
    PlannerConfig,
    RuntimeConfig,
)
from flexmoe.manifest import canonical_json_bytes, sha256_file


def test_runtime_config_is_immutable_and_pins_vllm() -> None:
    model = ModelSpec(Path("/models/qwen"), "Qwen3NextForCausalLM", "bfloat16", 41)
    benchmark = BenchmarkConfig(
        variant="vllm-resident",
        batch_size=32,
        context_length=1024,
        output_length=16,
    )
    config = RuntimeConfig(
        project_root=Path("/project"),
        model=model,
        planner=PlannerConfig(),
        benchmark=benchmark,
        gpu_ids=(0, 1, 2, 3),
    )

    assert config.vllm_commit == "01efc7ef781391e744ed08c3292817a773d654e6"
    assert config.benchmark.tensor_parallel_size == 4
    with pytest.raises(FrozenInstanceError):
        config.gpu_ids = (4, 5, 6, 7)  # type: ignore[misc]


def test_canonical_json_is_independent_of_key_order() -> None:
    assert canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    assert canonical_json_bytes({"a": 1, "b": 2}) == b'{"a":1,"b":2}'


def test_sha256_file_streams_known_payload(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"abc")

    assert sha256_file(payload) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
