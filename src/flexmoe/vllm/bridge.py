"""Fail-closed bridge called by the minimal pinned vLLM patch."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import prod
from pathlib import Path
from threading import Lock
from typing import Protocol

import torch

from flexmoe.codec.reference import encode_bf16_bits
from flexmoe.errors import ConfigurationError, IntegrityError, UnsupportedModeError
from flexmoe.paged_tensor import PagedTensorRegion
from flexmoe.placement import (
    BackendProfile,
    TensorKind,
    TensorSpec,
    assign_tensors,
)
from flexmoe.runtime.lifecycle import LayerLifecycle, LayerMaterializer
from flexmoe.storage.base import ExpertTensorStore, MaterializationReceipt
from flexmoe.storage.gpu_compressed import GpuCompressedStore
from flexmoe.storage.hierarchy import StorageHierarchy
from flexmoe.storage.host_pinned import PinnedHostStore
from flexmoe.vllm.loader import ExpertLoadAccumulator, ExpertWeights


class _Region(Protocol):
    @property
    def device(self) -> int: ...

    @property
    def virtual_bytes(self) -> int: ...

    @property
    def granularity(self) -> int: ...

    def create_block(self, nbytes: int) -> int: ...

    def map(self, offset: int, block_id: int, nbytes: int) -> None: ...

    def unmap(self, offset: int, nbytes: int) -> None: ...

    def tensor(
        self, offset: int, shape: Sequence[int], dtype: torch.dtype
    ) -> torch.Tensor: ...


RegionFactory = Callable[[int, int], _Region]


def _default_region_factory(device: int, virtual_bytes: int) -> _Region:
    return PagedTensorRegion(device=device, virtual_bytes=virtual_bytes)


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _set_parameter_attr(
    parameter: torch.nn.Parameter, name: str, value: object
) -> None:
    setattr(parameter, name, value)


class _TensorPool:
    def __init__(
        self,
        *,
        device: int,
        total_layers: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        region_factory: RegionFactory,
    ) -> None:
        probe = region_factory(device, 1)
        granularity = probe.granularity
        del probe
        raw_bytes = prod(shape) * torch.empty((), dtype=dtype).element_size()
        self.aligned_bytes = _round_up(raw_bytes, granularity)
        self.region = region_factory(device, total_layers * self.aligned_bytes)
        if self.region.granularity != granularity:
            raise IntegrityError("CUDA VMM granularity changed between reservations")
        self.shape = shape
        self.dtype = dtype
        self.blocks = (
            self.region.create_block(self.aligned_bytes),
            self.region.create_block(self.aligned_bytes),
        )
        self.parameters: dict[int, torch.nn.Parameter] = {}
        self._mapped_blocks: dict[int, int] = {}
        self._free_blocks = list(self.blocks)
        self._mapping_lock = Lock()
        self._mapped_bytes_total = 0
        self._mapping_count = 0

    def create_parameter(self, layer_idx: int) -> torch.nn.Parameter:
        if layer_idx in self.parameters:
            raise IntegrityError(f"duplicate paged parameter for layer {layer_idx}")
        offset = layer_idx * self.aligned_bytes
        scratch = self.blocks[0]
        self.region.map(offset, scratch, self.aligned_bytes)
        try:
            view = self.region.tensor(offset, self.shape, self.dtype)
            parameter = torch.nn.Parameter(view, requires_grad=False)
        finally:
            self.region.unmap(offset, self.aligned_bytes)
        self.parameters[layer_idx] = parameter
        return parameter

    def map_layer(self, layer_idx: int) -> None:
        if layer_idx not in self.parameters:
            raise IntegrityError(f"unknown paged layer {layer_idx}")
        with self._mapping_lock:
            if layer_idx in self._mapped_blocks:
                raise IntegrityError(f"layer {layer_idx} is already mapped")
            if not self._free_blocks:
                raise IntegrityError("two-layer physical window is exhausted")
            block = self._free_blocks.pop(0)
            offset = layer_idx * self.aligned_bytes
            self.region.map(offset, block, self.aligned_bytes)
            self._mapped_blocks[layer_idx] = block
            self._mapped_bytes_total += self.aligned_bytes
            self._mapping_count += 1

    def unmap_layer(self, layer_idx: int) -> None:
        with self._mapping_lock:
            try:
                block = self._mapped_blocks.pop(layer_idx)
            except KeyError as error:
                raise IntegrityError(f"layer {layer_idx} is not mapped") from error
            offset = layer_idx * self.aligned_bytes
            self.region.unmap(offset, self.aligned_bytes)
            self._free_blocks.append(block)
            self._free_blocks.sort()

    def destination(self, layer_idx: int, expert_idx: int) -> torch.Tensor:
        try:
            parameter = self.parameters[layer_idx]
        except KeyError as error:
            raise IntegrityError(f"unknown paged layer {layer_idx}") from error
        if not 0 <= expert_idx < parameter.shape[0]:
            raise IntegrityError(f"expert {expert_idx} is outside the tensor")
        return parameter[expert_idx]

    def mapping_counters(self) -> tuple[int, int]:
        with self._mapping_lock:
            return self._mapped_bytes_total, self._mapping_count


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def layer_index(layer_name: str) -> int:
    match = _LAYER_PATTERN.search(layer_name)
    if match is None:
        raise IntegrityError(f"cannot parse transformer layer from {layer_name}")
    return int(match.group(1))


def _canonical_layer_path(name: str) -> str:
    match = _LAYER_PATTERN.search(name)
    if match is None:
        raise IntegrityError(f"cannot parse transformer layer from {name}")
    start = match.start()
    if name[start] == ".":
        start += 1
    return name[start:]


@dataclass(frozen=True)
class ForwardToken:
    layer_name: str
    layer_idx: int


@dataclass(frozen=True)
class RegistryStorageConfig:
    gpu_compressed_budget_bytes: int
    host_capacity_bytes: int
    gpu_decode_bytes_per_second: float
    host_h2d_bytes_per_second: float

    def __post_init__(self) -> None:
        if self.gpu_compressed_budget_bytes < 0:
            raise ConfigurationError("GPU compressed budget must be non-negative")
        if self.host_capacity_bytes <= 0:
            raise ConfigurationError("host capacity must be positive")
        if self.gpu_decode_bytes_per_second <= 0:
            raise ConfigurationError("GPU decode bandwidth must be positive")
        if self.host_h2d_bytes_per_second <= 0:
            raise ConfigurationError("host HtoD bandwidth must be positive")


def _tensor_id(layer_idx: int, expert_idx: int, kind: TensorKind) -> str:
    return f"layer.{layer_idx}.expert.{expert_idx}.{kind}"


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return (
        tensor.contiguous()
        .view(torch.int16)
        .numpy()
        .astype("<i2", copy=False)
        .tobytes()
    )


class _LayerKindMaterializer(LayerMaterializer):
    def __init__(
        self,
        *,
        kind: TensorKind,
        num_experts: int,
        device: int,
        pool: _TensorPool,
        hierarchy: StorageHierarchy,
        expected_layer_hashes: Mapping[int, str],
        verify_weights: bool,
    ) -> None:
        self._kind = kind
        self._num_experts = num_experts
        self._device = device
        self._pool = pool
        self._hierarchy = hierarchy
        self._expected_layer_hashes = dict(expected_layer_hashes)
        self._verify_weights = verify_weights
        self._verified_layers: set[int] = set()
        self._receipts: list[MaterializationReceipt] = []
        self._lock = Lock()

    def evict(self, layer_idx: int) -> None:
        self._pool.unmap_layer(layer_idx)

    def materialize(self, layer_idx: int, load_stream: int) -> None:
        self._pool.map_layer(layer_idx)
        destinations = {
            _tensor_id(layer_idx, expert_idx, self._kind): self._pool.destination(
                layer_idx, expert_idx
            )
            for expert_idx in range(self._num_experts)
        }
        receipts = self._hierarchy.materialize_layer(
            layer_idx, destinations, stream=load_stream
        )
        verified_layer: int | None = None
        if self._verify_weights and layer_idx not in self._verified_layers:
            external_stream = torch.cuda.ExternalStream(  # type: ignore[no-untyped-call]
                load_stream, device=self._device
            )
            external_stream.synchronize()
            actual = sha256(
                _tensor_bytes(self._pool.parameters[layer_idx])
            ).hexdigest()
            expected = self._expected_layer_hashes[layer_idx]
            if actual != expected:
                raise IntegrityError(
                    f"materialized BF16 bits differ for layer {layer_idx} "
                    f"{self._kind}: expected {expected}, got {actual}"
                )
            verified_layer = layer_idx
        with self._lock:
            if verified_layer is not None:
                self._verified_layers.add(verified_layer)
            self._receipts.extend(receipts)

    def receipts(self) -> tuple[MaterializationReceipt, ...]:
        with self._lock:
            return tuple(self._receipts)

    def verification_counts(self) -> tuple[int, int]:
        with self._lock:
            return len(self._verified_layers), len(self._expected_layer_hashes)


class FluxMoERegistry:
    def __init__(
        self,
        *,
        total_layers: int,
        device: int,
        tp_rank: int,
        tp_size: int,
        num_experts: int,
        region_factory: RegionFactory = _default_region_factory,
        storage_config: RegistryStorageConfig | None = None,
    ) -> None:
        if type(total_layers) is not int or total_layers < 2:
            raise ConfigurationError("total_layers must be at least two")
        if type(device) is not int or device < 0:
            raise ConfigurationError("device must be a non-negative int")
        if type(tp_size) is not int or tp_size <= 0:
            raise ConfigurationError("tp_size must be positive")
        if type(tp_rank) is not int or not 0 <= tp_rank < tp_size:
            raise ConfigurationError("tp_rank must be inside the TP world")
        if type(num_experts) is not int or num_experts <= 0:
            raise ConfigurationError("num_experts must be positive")
        self.total_layers = total_layers
        self.device = device
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.num_experts = num_experts
        self._region_factory = region_factory
        self._storage_config = storage_config
        self._pools: dict[str, _TensorPool] = {}
        self._accumulators: dict[int, ExpertLoadAccumulator] = {}
        self._layer_names: dict[int, str] = {}
        self._completed: dict[tuple[int, int], ExpertWeights] = {}
        self._loading_layers: frozenset[int] | None = None
        self._lifecycle: LayerLifecycle | None = None
        self._materializers: dict[TensorKind, _LayerKindMaterializer] = {}
        self._ready = False
        self._finalized_expert_count = 0
        self._lock = Lock()

    def _pool(
        self, kind: str, shape: tuple[int, ...], dtype: torch.dtype
    ) -> _TensorPool:
        existing = self._pools.get(kind)
        if existing is not None:
            if existing.shape != shape or existing.dtype is not dtype:
                raise IntegrityError(f"inconsistent {kind} tensor geometry")
            return existing
        pool = _TensorPool(
            device=self.device,
            total_layers=self.total_layers,
            shape=shape,
            dtype=dtype,
            region_factory=self._region_factory,
        )
        self._pools[kind] = pool
        return pool

    def register_layer(
        self,
        *,
        layer_name: str,
        w13_shape: tuple[int, ...],
        w2_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
        layer_idx = layer_index(layer_name)
        if layer_idx >= self.total_layers:
            raise IntegrityError(
                f"layer {layer_idx} exceeds configured total {self.total_layers}"
            )
        with self._lock:
            if self._loading_layers is not None:
                raise IntegrityError("cannot register layers after weight loading starts")
            if layer_idx in self._layer_names:
                raise IntegrityError(f"duplicate routed-expert layer {layer_idx}")
            w13 = self._pool("w13", w13_shape, dtype).create_parameter(layer_idx)
            w2 = self._pool("w2", w2_shape, dtype).create_parameter(layer_idx)
            self._layer_names[layer_idx] = layer_name
            self._accumulators[layer_idx] = ExpertLoadAccumulator(
                layer_name=layer_name,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
            )
        return w13, w2

    def ingest(
        self,
        *,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        layer_name = getattr(param, "_fluxmoe_layer_name", None)
        kind = getattr(param, "_fluxmoe_kind", None)
        if not isinstance(layer_name, str) or kind not in {"w13", "w2"}:
            raise IntegrityError("weight loader received an unregistered parameter")
        if kind == "w13" and shard_id not in {"w1", "w3"}:
            raise IntegrityError(f"{kind} parameter received shard {shard_id}")
        if kind == "w2" and shard_id != "w2":
            raise IntegrityError(f"{kind} parameter received shard {shard_id}")
        registered_path = _canonical_layer_path(layer_name)
        loaded_path = _canonical_layer_path(weight_name)
        if not (
            loaded_path == registered_path
            or loaded_path.startswith(f"{registered_path}.")
        ):
            raise IntegrityError(
                f"weight name {weight_name} does not belong to {layer_name}"
            )
        layer_idx = layer_index(layer_name)
        with self._lock:
            if self._loading_layers is None:
                self._loading_layers = frozenset(self._layer_names)
                expected = frozenset(range(self.total_layers))
                if self._loading_layers != expected:
                    raise IntegrityError(
                        "routed-expert layers are not contiguous: "
                        f"expected {sorted(expected)}, "
                        f"got {sorted(self._loading_layers)}"
                    )
            accumulator = self._accumulators[layer_idx]
            accumulator.ingest(shard_id, expert_id, loaded_weight)
            if accumulator.has_complete_expert(expert_id):
                key = (layer_idx, expert_id)
                if key in self._completed:
                    raise IntegrityError(f"duplicate completed expert {key}")
                self._completed[key] = accumulator.finalize_expert(expert_id)
            if len(self._completed) == self.total_layers * self.num_experts:
                self._finalize_storage_locked()

    def _finalize_storage_locked(self) -> None:
        if self._ready:
            raise IntegrityError("FluxMoE storage was finalized twice")
        if self._storage_config is None:
            raise ConfigurationError("FluxMoE storage configuration is missing")
        for accumulator in self._accumulators.values():
            pending = accumulator.pending_keys()
            if pending:
                raise IntegrityError(f"unconsumed expert shards remain: {pending}")

        raw_tensors: dict[str, torch.Tensor] = {}
        tensor_layers: dict[str, int] = {}
        specs: list[TensorSpec] = []
        kinds: tuple[TensorKind, ...] = ("w13", "w2")
        for (layer_idx, expert_idx), weights in sorted(self._completed.items()):
            tensors: dict[TensorKind, torch.Tensor] = {
                "w13": weights.w13,
                "w2": weights.w2,
            }
            for kind in kinds:
                tensor_id = _tensor_id(layer_idx, expert_idx, kind)
                tensor = tensors[kind]
                nbytes = tensor.numel() * tensor.element_size()
                raw_tensors[tensor_id] = tensor
                tensor_layers[tensor_id] = layer_idx
                specs.append(
                    TensorSpec(
                        tensor_id=tensor_id,
                        layer_idx=layer_idx,
                        expert_idx=expert_idx,
                        kind=kind,
                        nbytes=nbytes,
                    )
                )

        config = self._storage_config
        placements = assign_tensors(
            tuple(specs),
            (
                BackendProfile(
                    name="gpu_compressed",
                    bytes_per_second=config.gpu_decode_bytes_per_second,
                    capacity_bytes=config.gpu_compressed_budget_bytes,
                ),
                BackendProfile(
                    name="host_pinned",
                    bytes_per_second=config.host_h2d_bytes_per_second,
                    capacity_bytes=config.host_capacity_bytes,
                ),
            ),
            gpu_budget_bytes=config.gpu_compressed_budget_bytes,
        )
        placement_by_id = {
            placement.tensor_id: placement for placement in placements
        }
        verify_weights = os.environ.get("FLUXMOE_VERIFY_WEIGHTS") == "1"
        expected_layer_hashes: dict[TensorKind, dict[int, str]] = {
            "w13": {},
            "w2": {},
        }
        if verify_weights:
            for kind in kinds:
                for layer_idx in range(self.total_layers):
                    digest = sha256()
                    for expert_idx in range(self.num_experts):
                        digest.update(
                            _tensor_bytes(
                                raw_tensors[
                                    _tensor_id(layer_idx, expert_idx, kind)
                                ]
                            )
                        )
                    expected_layer_hashes[kind][layer_idx] = digest.hexdigest()
        gpu_inputs = {
            tensor_id: encode_bf16_bits(
                _tensor_bytes(tensor),
                tuple(tensor.shape),
            )
            for tensor_id, tensor in raw_tensors.items()
            if placement_by_id[tensor_id].backend == "gpu_compressed"
        }
        host_inputs = {
            tensor_id: tensor
            for tensor_id, tensor in raw_tensors.items()
            if placement_by_id[tensor_id].backend == "host_pinned"
        }
        gpu_store = (
            GpuCompressedStore(gpu_inputs, device=self.device) if gpu_inputs else None
        )
        host_store = PinnedHostStore(host_inputs) if host_inputs else None
        stores: dict[str, ExpertTensorStore] = {}
        for tensor_id in raw_tensors:
            if placement_by_id[tensor_id].backend == "gpu_compressed":
                if gpu_store is None:
                    raise IntegrityError("GPU placement has no compressed store")
                stores[tensor_id] = gpu_store
            else:
                if host_store is None:
                    raise IntegrityError("host placement has no pinned store")
                stores[tensor_id] = host_store

        materializers: dict[str, LayerMaterializer] = {}
        for kind in kinds:
            kind_ids = {
                tensor_id
                for tensor_id in raw_tensors
                if tensor_id.endswith(f".{kind}")
            }
            hierarchy = StorageHierarchy(
                stores={tensor_id: stores[tensor_id] for tensor_id in kind_ids},
                tensor_layers={
                    tensor_id: tensor_layers[tensor_id] for tensor_id in kind_ids
                },
            )
            materializer = _LayerKindMaterializer(
                kind=kind,
                num_experts=self.num_experts,
                device=self.device,
                pool=self._pools[kind],
                hierarchy=hierarchy,
                expected_layer_hashes=expected_layer_hashes[kind],
                verify_weights=verify_weights,
            )
            self._materializers[kind] = materializer
            materializers[kind] = materializer
        self._lifecycle = LayerLifecycle(
            device=self.device,
            total_layers=self.total_layers,
            materializers=materializers,
        )
        self._lifecycle.schedule_next(0)
        self._lifecycle.schedule_next(1)
        self._finalized_expert_count = len(self._completed)
        self._completed.clear()
        self._ready = True

    def completed_experts(self) -> int:
        with self._lock:
            return max(len(self._completed), self._finalized_expert_count)

    def before_forward(
        self, layer_name: str, w13: torch.Tensor, w2: torch.Tensor
    ) -> ForwardToken:
        layer_idx = layer_index(layer_name)
        expected_w13 = self._pools["w13"].parameters[layer_idx]
        expected_w2 = self._pools["w2"].parameters[layer_idx]
        if w13.data_ptr() != expected_w13.data_ptr():
            raise IntegrityError("w13 pointer does not match the paged registry")
        if w2.data_ptr() != expected_w2.data_ptr():
            raise IntegrityError("w2 pointer does not match the paged registry")
        if not self._ready or self._lifecycle is None:
            raise IntegrityError("FluxMoE storage hierarchy is not finalized")
        stream = torch.cuda.current_stream(self.device)
        self._lifecycle.ensure_ready(layer_idx, stream)
        return ForwardToken(layer_name=layer_name, layer_idx=layer_idx)

    def after_forward(self, token: ForwardToken) -> None:
        if not self._ready or self._lifecycle is None:
            raise IntegrityError("FluxMoE storage hierarchy is not finalized")
        if self._layer_names.get(token.layer_idx) != token.layer_name:
            raise IntegrityError("forward token does not match the registered layer")
        stream = torch.cuda.current_stream(self.device)
        self._lifecycle.mark_consumed(token.layer_idx, stream)
        self._lifecycle.schedule_next((token.layer_idx + 2) % self.total_layers)

    def receipts(self) -> tuple[MaterializationReceipt, ...]:
        receipts: list[MaterializationReceipt] = []
        for kind in sorted(self._materializers):
            receipts.extend(self._materializers[kind].receipts())
        return tuple(receipts)

    def mechanism_counters(self) -> dict[str, int]:
        receipts = self.receipts()
        mapped_bytes = 0
        mapping_count = 0
        weights_verified = 0
        weights_expected = 0
        for pool in self._pools.values():
            pool_bytes, pool_count = pool.mapping_counters()
            mapped_bytes += pool_bytes
            mapping_count += pool_count
        for materializer in self._materializers.values():
            verified, expected = materializer.verification_counts()
            weights_verified += verified
            weights_expected += expected
        return {
            "mapped_bytes": mapped_bytes,
            "mapping_count": mapping_count,
            "h2d_bytes": sum(
                receipt.nbytes
                for receipt in receipts
                if receipt.backend == "host_pinned"
            ),
            "decompressed_bytes": sum(
                receipt.nbytes
                for receipt in receipts
                if receipt.backend == "gpu_compressed"
            ),
            "weights_verified": weights_verified,
            "weights_expected": weights_expected,
        }

    def close(self) -> None:
        lifecycle = self._lifecycle
        self._lifecycle = None
        self._ready = False
        if lifecycle is not None:
            lifecycle.close()


_ACTIVE_REGISTRY: FluxMoERegistry | None = None
_REGISTRY_LOCK = Lock()
_ROUTER_TRACE_LOCK = Lock()
_ROUTER_SEQUENCE = 0


def record_router_ids(layer_name: str, topk_ids: torch.Tensor) -> None:
    """Persist exact routed expert IDs as an ordered, rank-local digest."""

    global _ROUTER_SEQUENCE
    trace_root = os.environ.get("FLUXMOE_ROUTER_TRACE_DIR")
    if not trace_root:
        raise IntegrityError("FLUXMOE_ROUTER_TRACE_DIR is required for tracing")
    if topk_ids.ndim != 2 or topk_ids.numel() == 0:
        raise IntegrityError("router Top-k IDs must be a non-empty matrix")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank_value = os.environ.get("RANK", "0")
        try:
            rank = int(rank_value)
        except ValueError as error:
            raise IntegrityError(f"invalid RANK value {rank_value}") from error
    normalized = (
        topk_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    )
    digest = sha256(
        normalized.numpy().astype("<i8", copy=False).tobytes()
    ).hexdigest()
    root = Path(trace_root)
    root.mkdir(parents=True, exist_ok=True)
    with _ROUTER_TRACE_LOCK:
        payload = {
            "schema_version": 1,
            "sequence": _ROUTER_SEQUENCE,
            "rank": rank,
            "layer_name": layer_name,
            "shape": list(normalized.shape),
            "sha256": digest,
        }
        _ROUTER_SEQUENCE += 1
        with (root / f"rank-{rank}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            stream.write("\n")


def reset_router_trace_state() -> None:
    global _ROUTER_SEQUENCE
    with _ROUTER_TRACE_LOCK:
        _ROUTER_SEQUENCE = 0


def install_registry(registry: FluxMoERegistry) -> None:
    global _ACTIVE_REGISTRY
    with _REGISTRY_LOCK:
        if _ACTIVE_REGISTRY is not None and _ACTIVE_REGISTRY is not registry:
            raise IntegrityError("a different FluxMoE registry is already active")
        _ACTIVE_REGISTRY = registry


def reset_registry() -> None:
    global _ACTIVE_REGISTRY
    with _REGISTRY_LOCK:
        registry = _ACTIVE_REGISTRY
        _ACTIVE_REGISTRY = None
    if registry is not None:
        registry.close()


def require_active_registry() -> FluxMoERegistry:
    with _REGISTRY_LOCK:
        if _ACTIVE_REGISTRY is None:
            raise IntegrityError("FluxMoE registry is not initialized")
        return _ACTIVE_REGISTRY


def _total_layers_from_model_path() -> int:
    model_path = os.environ.get("FLUXMOE_MODEL_PATH")
    if not model_path:
        raise ConfigurationError("FLUXMOE_MODEL_PATH is required")
    config_path = Path(model_path) / "config.json"
    try:
        parsed = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"cannot read model config: {error}") from error
    total_layers = parsed.get("num_hidden_layers") if isinstance(parsed, dict) else None
    if type(total_layers) is not int or total_layers < 2:
        raise ConfigurationError("model config has invalid num_hidden_layers")
    return total_layers


def _required_env_int(name: str, *, allow_zero: bool = False) -> int:
    raw = os.environ.get(name)
    if raw is None:
        raise ConfigurationError(f"{name} is required")
    try:
        value = int(raw)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an int") from error
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ConfigurationError(f"{name} must be {qualifier}")
    return value


def _required_env_float(name: str) -> float:
    raw = os.environ.get(name)
    if raw is None:
        raise ConfigurationError(f"{name} is required")
    try:
        value = float(raw)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be numeric") from error
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return value


def _storage_config_from_env() -> RegistryStorageConfig:
    return RegistryStorageConfig(
        gpu_compressed_budget_bytes=_required_env_int(
            "FLUXMOE_GPU_COMPRESSED_BUDGET_BYTES", allow_zero=True
        ),
        host_capacity_bytes=_required_env_int("FLUXMOE_HOST_CAPACITY_BYTES"),
        gpu_decode_bytes_per_second=_required_env_float(
            "FLUXMOE_GPU_DECODE_BYTES_PER_SECOND"
        ),
        host_h2d_bytes_per_second=_required_env_float(
            "FLUXMOE_HOST_H2D_BYTES_PER_SECOND"
        ),
    )


def _registry_for_layer(layer: torch.nn.Module, num_experts: int) -> FluxMoERegistry:
    global _ACTIVE_REGISTRY
    with _REGISTRY_LOCK:
        if _ACTIVE_REGISTRY is None:
            tp_rank = getattr(layer, "tp_rank", None)
            tp_size = getattr(layer, "tp_size", None)
            if type(tp_rank) is not int or type(tp_size) is not int:
                raise UnsupportedModeError("vLLM layer does not expose TP rank/size")
            _ACTIVE_REGISTRY = FluxMoERegistry(
                total_layers=_total_layers_from_model_path(),
                device=torch.cuda.current_device(),
                tp_rank=tp_rank,
                tp_size=tp_size,
                num_experts=num_experts,
                storage_config=_storage_config_from_env(),
            )
        return _ACTIVE_REGISTRY


def maybe_create_weights(
    layer: torch.nn.Module,
    num_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    params_dtype: torch.dtype,
    **extra_weight_attrs: object,
) -> bool:
    if os.environ.get("FLUXMOE_ENABLE") != "1":
        return False
    if params_dtype is not torch.bfloat16:
        raise UnsupportedModeError("FluxMoE supports only BF16 expert weights")
    if getattr(layer, "quant_config", None) is not None:
        raise UnsupportedModeError("FluxMoE does not support quantized FusedMoE")
    if bool(getattr(layer, "use_ep", False)):
        raise UnsupportedModeError("FluxMoE does not support expert parallelism")
    if bool(getattr(layer, "enable_eplb", False)):
        raise UnsupportedModeError("FluxMoE does not support EPLB")
    moe_config = getattr(layer, "moe_config", None)
    if bool(getattr(moe_config, "has_bias", False)):
        raise UnsupportedModeError("FluxMoE does not support expert bias")
    if num_experts <= 0 or hidden_size <= 0 or intermediate_size_per_partition <= 0:
        raise IntegrityError("expert tensor dimensions must be positive")
    layer_name = getattr(layer, "layer_name", None)
    if not isinstance(layer_name, str):
        raise IntegrityError("FusedMoE layer_name is missing")

    registry = _registry_for_layer(layer, num_experts)
    if registry.num_experts != num_experts:
        raise IntegrityError("num_experts changed between FusedMoE layers")
    w13, w2 = registry.register_layer(
        layer_name=layer_name,
        w13_shape=(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
        ),
        w2_shape=(num_experts, hidden_size, intermediate_size_per_partition),
        dtype=params_dtype,
    )
    from flexmoe.vllm.loader import store_expert_weight

    for param, kind in ((w13, "w13"), (w2, "w2")):
        for attribute, value in extra_weight_attrs.items():
            if attribute != "weight_loader":
                setattr(param, attribute, value)
        _set_parameter_attr(param, "weight_loader", store_expert_weight)
        _set_parameter_attr(param, "_fluxmoe_layer_name", layer_name)
        _set_parameter_attr(param, "_fluxmoe_kind", kind)
    layer.register_parameter("w13_weight", w13)
    layer.register_parameter("w2_weight", w2)
    return True


def before_forward(
    layer_name: str, w13: torch.Tensor, w2: torch.Tensor
) -> ForwardToken:
    return require_active_registry().before_forward(layer_name, w13, w2)


def after_forward(token: ForwardToken) -> None:
    require_active_registry().after_forward(token)


__all__ = [
    "FluxMoERegistry",
    "ForwardToken",
    "RegistryStorageConfig",
    "after_forward",
    "before_forward",
    "install_registry",
    "layer_index",
    "maybe_create_weights",
    "record_router_ids",
    "require_active_registry",
    "reset_registry",
    "reset_router_trace_state",
]
