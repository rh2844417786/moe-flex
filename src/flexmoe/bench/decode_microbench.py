"""Measure expertwise versus packed decode on one real checkpoint layer."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Protocol, cast

import torch
from safetensors import safe_open

from flexmoe.codec.packed import pack_layer_descriptor
from flexmoe.codec.reference import EncodedBFloat16, encode_bf16_bits
from flexmoe.storage.gpu_compressed import (
    BatchedGpuCompressedStore,
    GpuCompressedStore,
)
from flexmoe.vllm.loader import ExpertLoadAccumulator


class _TensorReader(Protocol):
    def get_tensor(self, name: str) -> torch.Tensor: ...


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return (
        tensor.contiguous()
        .view(torch.int16)
        .numpy()
        .astype("<i2", copy=False)
        .tobytes()
    )


def _checkpoint_tensor(
    model_path: Path,
    weight_map: dict[str, str],
    readers: dict[str, _TensorReader],
    stack: ExitStack,
    name: str,
) -> torch.Tensor:
    try:
        shard = weight_map[name]
    except KeyError as error:
        raise ValueError(f"checkpoint index has no tensor {name}") from error
    reader = readers.get(shard)
    if reader is None:
        reader = cast(
            _TensorReader,
            stack.enter_context(
                safe_open(model_path / shard, framework="pt", device="cpu")
            ),
        )
        readers[shard] = reader
    return reader.get_tensor(name)


def _encode_layer(
    model_path: Path,
    *,
    layer_idx: int,
    expert_count: int,
    tp_rank: int,
    tp_size: int,
) -> tuple[
    dict[str, dict[int, EncodedBFloat16]],
    dict[str, tuple[int, ...]],
    float,
]:
    index = json.loads(
        (model_path / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise ValueError("checkpoint index has no weight map")  # noqa: TRY004 - malformed checkpoint
    weight_map = index["weight_map"]
    encoded: dict[str, dict[int, EncodedBFloat16]] = {"w13": {}, "w2": {}}
    shapes: dict[str, tuple[int, ...]] = {}
    started = perf_counter()
    readers: dict[str, _TensorReader] = {}
    with ExitStack() as stack:
        for expert_idx in range(expert_count):
            prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            accumulator = ExpertLoadAccumulator(
                layer_name=f"model.layers.{layer_idx}.mlp.experts",
                tp_rank=tp_rank,
                tp_size=tp_size,
            )
            for shard_id, suffix in (
                ("w1", "gate_proj.weight"),
                ("w2", "down_proj.weight"),
                ("w3", "up_proj.weight"),
            ):
                accumulator.ingest(
                    shard_id,
                    expert_idx,
                    _checkpoint_tensor(
                        model_path,
                        weight_map,
                        readers,
                        stack,
                        f"{prefix}.{suffix}",
                    ),
                )
            weights = accumulator.finalize_expert(expert_idx)
            for kind, tensor in (("w13", weights.w13), ("w2", weights.w2)):
                shapes.setdefault(kind, tuple(tensor.shape))
                if shapes[kind] != tuple(tensor.shape):
                    raise ValueError(f"inconsistent {kind} expert shape")
                encoded[kind][expert_idx] = encode_bf16_bits(
                    _tensor_bytes(tensor), tuple(tensor.shape)
                )
    return encoded, shapes, perf_counter() - started


def _cuda_duration(operation: Callable[[], None], repetitions: int) -> list[float]:
    durations: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        start.record()  # type: ignore[no-untyped-call]
        operation()
        end.record()  # type: ignore[no-untyped-call]
        end.synchronize()
        durations.append(start.elapsed_time(end) / 1000.0)  # type: ignore[no-untyped-call]
    return durations


def _launch_expertwise(
    store: GpuCompressedStore,
    destination: torch.Tensor,
    stream_handle: int,
    expert_count: int,
) -> None:
    for expert_idx in range(expert_count):
        store.materialize(
            f"expert.{expert_idx}", destination[expert_idx], stream_handle
        )


def _launch_batched(
    store: BatchedGpuCompressedStore,
    layer_kind: str,
    destination: torch.Tensor,
    stream_handle: int,
) -> None:
    store.materialize_batched(layer_kind, destination, stream_handle)


def run_microbenchmark(
    model_path: Path,
    *,
    layer_idx: int,
    expert_count: int,
    tp_rank: int,
    tp_size: int,
    repetitions: int,
    device: int,
) -> dict[str, object]:
    encoded_by_kind, shapes, encode_s = _encode_layer(
        model_path,
        layer_idx=layer_idx,
        expert_count=expert_count,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )
    results: dict[str, object] = {}
    cuda_device = torch.device("cuda", device)
    with torch.cuda.device(cuda_device):
        for kind in ("w13", "w2"):
            expert_shape = shapes[kind]
            destination_shape = (expert_count, *expert_shape)
            packed = pack_layer_descriptor(
                encoded_by_kind[kind], destination_shape=destination_shape
            )
            expert_store = GpuCompressedStore(
                {
                    f"expert.{expert_idx}": value
                    for expert_idx, value in encoded_by_kind[kind].items()
                },
                device=device,
            )
            batched_store = BatchedGpuCompressedStore(
                {f"layer.{layer_idx}.{kind}": packed}, device=device
            )
            expert_destination = torch.empty(
                destination_shape, dtype=torch.bfloat16, device=cuda_device
            )
            batched_destination = torch.empty_like(expert_destination)
            stream = torch.cuda.current_stream(device)

            expertwise = partial(
                _launch_expertwise,
                expert_store,
                expert_destination,
                stream.cuda_stream,
                expert_count,
            )
            batched = partial(
                _launch_batched,
                batched_store,
                f"layer.{layer_idx}.{kind}",
                batched_destination,
                stream.cuda_stream,
            )

            expertwise()
            batched()
            stream.synchronize()
            for expert_idx in range(expert_count):
                expert_store.raise_for_decode_errors(f"expert.{expert_idx}")
            batched_store.raise_for_decode_errors(f"layer.{layer_idx}.{kind}")
            if not torch.equal(expert_destination, batched_destination):
                raise RuntimeError(f"{kind} expertwise and batched outputs differ")
            expertwise_s = _cuda_duration(expertwise, repetitions)
            batched_s = _cuda_duration(batched, repetitions)
            output_bytes = packed.source_bytes
            results[kind] = {
                "source_bytes": output_bytes,
                "encoded_payload_bytes": packed.encoded_payload_bytes,
                "gpu_storage_bytes": packed.gpu_storage_bytes,
                "compression_ratio": packed.gpu_storage_bytes / output_bytes,
                "expertwise_launches": expert_count,
                "batched_launches": 1,
                "expertwise_cuda_s": expertwise_s,
                "batched_cuda_s": batched_s,
                "expertwise_median_cuda_s": median(expertwise_s),
                "batched_median_cuda_s": median(batched_s),
                "speedup": median(expertwise_s) / median(batched_s),
                "batched_output_gbps": output_bytes / median(batched_s) / 1e9,
                "bit_exact": True,
            }
            del expert_store, batched_store, expert_destination, batched_destination
            torch.cuda.empty_cache()
    return {
        "schema_version": 1,
        "process_name": "wth333",
        "model": model_path.name,
        "layer_idx": layer_idx,
        "expert_count": expert_count,
        "tp_rank": tp_rank,
        "tp_size": tp_size,
        "startup_cpu_encode_s": encode_s,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    arguments = parser.parse_args()
    result = run_microbenchmark(
        arguments.model,
        layer_idx=arguments.layer,
        expert_count=arguments.experts,
        tp_rank=arguments.tp_rank,
        tp_size=arguments.tp_size,
        repetitions=arguments.repetitions,
        device=arguments.device,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
