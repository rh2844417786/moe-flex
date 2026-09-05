"""Partial host offload preserving native weights for the resident majority."""

from __future__ import annotations

import importlib
import json
import os
import re
from dataclasses import dataclass
from math import prod
from pathlib import Path

import torch

from flexmoe.errors import IntegrityError
from flexmoe.runtime.partial_plan import PartialPlan
from flexmoe.runtime.partial_staging import CudaStager, Stager
from flexmoe.vllm.loader import ExpertLoadAccumulator, ExpertWeights


def _layer_index(name: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    if match is None:
        raise IntegrityError(f"cannot locate transformer layer in {name}")
    return int(match.group(1))


def validate_execution_config(config: object) -> None:
    model = getattr(config, "model_config", None)
    compilation = getattr(config, "compilation_config", None)
    parallel = getattr(config, "parallel_config", None)
    if getattr(model, "enforce_eager", False) is not True:
        raise ValueError("partial staging requires eager execution")
    if getattr(compilation, "level", None) != 0:
        raise ValueError("partial staging requires compilation level 0")
    ops = getattr(compilation, "custom_ops", None)
    if not isinstance(ops, list) or (
        "all" not in ops or "none" in ops or "-fused_moe" in ops
    ):
        raise ValueError("partial staging requires the fused_moe CUDA hook enabled")
    if getattr(parallel, "pipeline_parallel_size", None) != 1 or (
        getattr(parallel, "data_parallel_size", None) != 1
    ):
        raise ValueError("partial staging currently requires DP=1 and PP=1")


class HostExpertLayer:
    """One contiguous pinned buffer per tensor kind, filled exactly once."""

    def __init__(
        self,
        w13_shape: tuple[int, ...],
        w2_shape: tuple[int, ...],
        *,
        pin_memory: bool,
    ) -> None:
        if (
            len(w13_shape) != 3
            or len(w2_shape) != 3
            or (w13_shape[0] != w2_shape[0] or min(*w13_shape, *w2_shape) <= 0)
        ):
            raise ValueError("invalid expert layer geometry")
        self.w13 = torch.empty(
            w13_shape, dtype=torch.bfloat16, device="cpu", pin_memory=pin_memory
        )
        self.w2 = torch.empty(
            w2_shape, dtype=torch.bfloat16, device="cpu", pin_memory=pin_memory
        )
        self._seen: set[int] = set()

    def add(self, weights: ExpertWeights) -> None:
        expert = weights.expert_id
        if type(expert) is not int or not 0 <= expert < self.w13.shape[0]:
            raise IntegrityError("expert ID outside the host layer")
        if expert in self._seen:
            raise IntegrityError("duplicate expert in host layer")
        for source, target in (
            (weights.w13, self.w13[expert]),
            (weights.w2, self.w2[expert]),
        ):
            if (
                source.device.type != "cpu"
                or source.dtype != torch.bfloat16
                or tuple(source.shape) != tuple(target.shape)
            ):
                raise IntegrityError("expert shard geometry/dtype/device mismatch")
        self.w13[expert].copy_(weights.w13)
        self.w2[expert].copy_(weights.w2)
        self._seen.add(expert)

    def require_complete(self) -> None:
        if len(self._seen) != self.w13.shape[0]:
            raise IntegrityError("host layer is incomplete")


@dataclass(frozen=True)
class PartialForwardToken:
    layer_idx: int
    slot: int
    generation: int
    timing_ticket: object | None


class PartialRegistry:
    def __init__(
        self,
        *,
        plan: PartialPlan,
        device: int,
        tp_rank: int,
        tp_size: int,
        num_experts: int,
        stager: Stager | None = None,
        timing_samples: int = 128,
    ) -> None:
        if tp_size <= 0 or not 0 <= tp_rank < tp_size or num_experts <= 0:
            raise ValueError("invalid partial-offload TP/expert configuration")
        self.plan = plan
        self.device = device
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.num_experts = num_experts
        self._stager = stager
        self._timing_samples = timing_samples
        self._geometry: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        self._names: dict[int, str] = {}
        self._host: dict[int, HostExpertLayer] = {}
        self._accumulators: dict[int, ExpertLoadAccumulator] = {}
        self._slots: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._pointers: dict[int, tuple[int, int]] = {}
        self._stride: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        self._started = False
        self._position = 0
        self._generation = 0
        self._active: PartialForwardToken | None = None
        self._h2d_bytes = 0
        self._copy_launches = 0
        self._offload_forwards = 0
        self._resident_forwards = 0
        self._verified: set[int] = set()

    @property
    def layer_bytes(self) -> int:
        if self._geometry is None:
            return 0
        return sum(prod(shape) * 2 for shape in self._geometry)

    def register_layer(
        self,
        layer_name: str,
        w13_shape: tuple[int, ...],
        w2_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> tuple[torch.nn.Parameter, torch.nn.Parameter] | None:
        idx = _layer_index(layer_name)
        if idx in self._names or idx >= self.plan.total_layers or self._started:
            raise IntegrityError(
                "duplicate, late, or invalid partial layer registration"
            )
        if (
            dtype != torch.bfloat16
            or w13_shape[0] != self.num_experts
            or (w2_shape[0] != self.num_experts)
        ):
            raise IntegrityError("partial offload requires the original BF16 experts")
        geometry = (w13_shape, w2_shape)
        if self._geometry is not None and self._geometry != geometry:
            raise IntegrityError("expert geometry differs between layers")
        self._geometry = geometry
        self._names[idx] = layer_name
        slot = self.plan.slot_for(idx)
        if slot is None:
            return None
        if self._stager is None:
            self._stager = CudaStager(
                self.device, self.plan.staging_slots, self._timing_samples
            )
        if not self._slots:
            self._slots = [
                (self._stager.allocate(w13_shape), self._stager.allocate(w2_shape))
                for _ in range(self.plan.staging_slots)
            ]
            self._stride = (
                tuple(self._slots[0][0].stride()),
                tuple(self._slots[0][1].stride()),
            )
        pair = (
            torch.nn.Parameter(self._slots[slot][0], requires_grad=False),
            torch.nn.Parameter(self._slots[slot][1], requires_grad=False),
        )
        self._pointers[idx] = (pair[0].data_ptr(), pair[1].data_ptr())
        self._host[idx] = HostExpertLayer(
            w13_shape, w2_shape, pin_memory=self._stager.pin_memory
        )
        self._accumulators[idx] = ExpertLoadAccumulator(
            layer_name, self.tp_rank, self.tp_size
        )
        for parameter, kind in zip(pair, ("w13", "w2"), strict=True):
            for name, value in (("_partial_layer_idx", idx), ("_partial_kind", kind)):
                setattr(parameter, name, value)
        return pair

    def ingest(
        self,
        *,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        idx = getattr(param, "_partial_layer_idx", None)
        kind = getattr(param, "_partial_kind", None)
        if type(idx) is not int or idx not in self._host or self._started:
            raise IntegrityError("unregistered or late partial expert shard")
        if _layer_index(weight_name) != idx:
            raise IntegrityError("checkpoint weight belongs to another layer")
        if (
            (kind == "w13" and shard_id not in ("w1", "w3"))
            or (kind == "w2" and shard_id != "w2")
            or kind not in ("w13", "w2")
        ):
            raise IntegrityError("checkpoint shard belongs to another tensor kind")
        if type(expert_id) is not int or not 0 <= expert_id < self.num_experts:
            raise IntegrityError("expert ID outside partial layer")
        accumulator = self._accumulators[idx]
        accumulator.ingest(shard_id, expert_id, loaded_weight)
        if accumulator.has_complete_expert(expert_id):
            self._host[idx].add(accumulator.finalize_expert(expert_id))

    def _upload(self, layer: int, *, after_compute: bool) -> None:
        assert self._stager is not None
        slot = self.plan.slot_for(layer)
        assert slot is not None
        self._stager.enqueue(
            slot, self._host[layer], self._slots[slot], after_compute=after_compute
        )
        self._h2d_bytes += self.layer_bytes
        self._copy_launches += 2

    def _start(self) -> None:
        if self._started:
            return
        if len(self._names) != self.plan.total_layers:
            raise IntegrityError("model layer registration is incomplete")
        for idx in self.plan.offload_layers:
            self._host[idx].require_complete()
            if self._accumulators[idx].pending_keys():
                raise IntegrityError("unconsumed expert shards remain")
        for idx in self.plan.initial_prefetch:
            self._upload(idx, after_compute=False)
        self._started = True

    def before_forward(
        self,
        layer_name: str,
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> PartialForwardToken | None:
        idx = _layer_index(layer_name)
        if self._names.get(idx) != layer_name:
            raise IntegrityError("unregistered partial forward")
        self._start()
        slot = self.plan.slot_for(idx)
        if slot is None:
            self._resident_forwards += 1
            return None
        expected = self.plan.offload_layers[self._position]
        if idx != expected or self._active is not None:
            raise IntegrityError(
                f"partial forward expected layer {expected}, got {idx}"
            )
        if (
            (w13.data_ptr(), w2.data_ptr()) != self._pointers[idx]
            or (tuple(w13.shape), tuple(w2.shape)) != self._geometry
            or (tuple(w13.stride()), tuple(w2.stride())) != self._stride
            or w13.dtype != torch.bfloat16
            or w2.dtype != torch.bfloat16
        ):
            raise IntegrityError("vLLM parameter no longer aliases its staging slot")
        assert self._stager is not None
        token = PartialForwardToken(
            idx, slot, self._generation, self._stager.begin(slot)
        )
        if idx not in self._verified:
            # This runs once per offloaded layer during engine profiling/smoke.
            # Integer comparison preserves NaNs, signed zero and every BF16 bit.
            host = self._host[idx]
            if not all(
                torch.equal(
                    device_value.detach().view(torch.int16).cpu(),
                    host_value.view(torch.int16),
                )
                for device_value, host_value in ((w13, host.w13), (w2, host.w2))
            ):
                raise IntegrityError("partial staging BF16 bits differ from checkpoint")
            self._verified.add(idx)
        self._active = token
        self._offload_forwards += 1
        return token

    def after_forward(self, token: PartialForwardToken | None) -> None:
        if token is None:
            return
        if token is not self._active or token.generation != self._generation:
            raise IntegrityError("stale or mismatched partial forward token")
        assert self._stager is not None
        self._stager.end(token.slot, token.timing_ticket)
        self._active = None
        self._generation += 1
        self._position = (self._position + 1) % len(self.plan.offload_layers)
        self._upload(self.plan.prefetch_after(token.layer_idx), after_compute=True)

    def stats(
        self,
        *,
        synchronize: bool = True,
        reset_timing: bool = False,
    ) -> dict[str, object]:
        if self._active is not None:
            raise IntegrityError("cannot snapshot during an active MoE forward")
        timing: dict[str, int | float] = {
            "sample_count": 0,
            "load_cuda_s": 0.0,
            "wait_cuda_s": 0.0,
            "compute_cuda_s": 0.0,
            "cpu_enqueue_s": 0.0,
        }
        if self._stager is not None:
            if synchronize:
                self._stager.synchronize()
            timing = self._stager.timing(reset=reset_timing)
        count = len(self.plan.offload_layers)
        return {
            "schema_version": 1,
            "rank": self.tp_rank,
            "total_layers": self.plan.total_layers,
            "offload_layers": list(self.plan.offload_layers),
            "staging_slots": self.plan.staging_slots if count else 0,
            "layer_bytes": self.layer_bytes,
            "host_source_bytes": count * self.layer_bytes,
            "gpu_staging_bytes": len(self._slots) * self.layer_bytes,
            "resident_routed_bytes": (self.plan.total_layers - count)
            * self.layer_bytes,
            "net_freed_bytes": (count - len(self._slots)) * self.layer_bytes,
            "h2d_bytes": self._h2d_bytes,
            "copy_launches": self._copy_launches,
            "offload_forwards": self._offload_forwards,
            "resident_forwards": self._resident_forwards,
            "timing": timing,
            "weights_expected": count * 2,
            "weights_verified": len(self._verified) * 2,
        }

    def close(self) -> None:
        if self._stager is not None:
            self._stager.synchronize()


_REGISTRY: PartialRegistry | None = None


def registry_for_layer(layer: torch.nn.Module, num_experts: int) -> PartialRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        vllm_config = importlib.import_module("vllm.config")
        validate_execution_config(vllm_config.get_current_vllm_config())
        model_path = os.environ.get("FLUXMOE_MODEL_PATH", "")
        if not model_path:
            raise ValueError("FLUXMOE_MODEL_PATH is required for partial offload")
        config = json.loads((Path(model_path) / "config.json").read_text())
        total_layers = config["num_hidden_layers"]
        raw_layers = os.environ.get("FLUXMOE_PARTIAL_OFFLOAD_LAYERS")
        if raw_layers is None:
            raise ValueError("FLUXMOE_PARTIAL_OFFLOAD_LAYERS is required")
        layers = tuple(int(value) for value in raw_layers.split(",") if value.strip())
        slots = int(os.environ.get("FLUXMOE_PARTIAL_STAGING_SLOTS", "1"))
        tp_rank = getattr(layer, "tp_rank", None)
        tp_size = getattr(layer, "tp_size", None)
        if type(tp_rank) is not int or type(tp_size) is not int:
            raise IntegrityError("vLLM TP metadata is unavailable")
        _REGISTRY = PartialRegistry(
            plan=PartialPlan(total_layers, layers, slots),
            device=torch.cuda.current_device(),
            tp_rank=tp_rank,
            tp_size=tp_size,
            num_experts=num_experts,
            timing_samples=int(os.environ.get("FLUXMOE_PARTIAL_TIMING_SAMPLES", "128")),
        )
    if _REGISTRY.num_experts != num_experts:
        raise IntegrityError("expert count changed between partial layers")
    return _REGISTRY


def require_registry() -> PartialRegistry:
    if _REGISTRY is None:
        raise IntegrityError("partial offload registry is not initialized")
    return _REGISTRY


def reset_registry() -> None:
    global _REGISTRY
    if _REGISTRY is not None:
        _REGISTRY.close()
    _REGISTRY = None


def store_partial_weight(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    expert_id: int,
    return_success: bool = False,
) -> bool | None:
    require_registry().ingest(
        param=param,
        loaded_weight=loaded_weight,
        weight_name=weight_name,
        shard_id=shard_id,
        expert_id=expert_id,
    )
    return True if return_success else None
