"""TP-only eager expert cache integration for pinned vLLM 0.10.2."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import torch

from flexmoe.errors import IntegrityError
from flexmoe.runtime.expert_cache_policy import ExpertCachePolicy
from flexmoe.runtime.expert_pool import CudaPoolBackend, ExpertPool, PoolBackend
from flexmoe.runtime.expert_profile import ExpertProfile
from flexmoe.vllm.expert_calibration import ModelProfileIdentity, model_profile_identity
from flexmoe.vllm.loader import ExpertLoadAccumulator
from flexmoe.vllm.partial import (
    HostExpertLayer,
    _layer_index,
    validate_execution_config,
)


def logical_fused_experts(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    *,
    num_experts: int,
    activation: str,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    """Keep native logical-E tuning even though pool physical E is different.

    The pinned kernel internally reselects configuration for its last chunk.
    Run each chunk under its own public override; never monkeypatch globals.
    The owning pool fail-stops on exceptions (upstream override lacks finally).
    """
    package = importlib.import_module("vllm.model_executor.layers.fused_moe")
    kernel = importlib.import_module("vllm.model_executor.layers.fused_moe.fused_moe")
    envs = importlib.import_module("vllm.envs")
    if package.get_config() is not None:
        raise RuntimeError("nested fused MoE configuration override is unsupported")
    chunk_size = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("invalid native MoE chunk size")
    dtype = kernel.get_config_dtype_str(dtype=hidden_states.dtype)
    output = (
        torch.empty_like(hidden_states) if hidden_states.shape[0] > chunk_size else None
    )
    for begin in range(0, hidden_states.shape[0], chunk_size):
        end = min(begin + chunk_size, hidden_states.shape[0])
        config = kernel.try_get_optimal_moe_config(
            (num_experts, *w13.shape[1:]),
            (num_experts, *w2.shape[1:]),
            topk_ids.shape[1],
            dtype,
            end - begin,
        )
        with package.override_config(config):
            result = package.fused_experts(
                hidden_states[begin:end],
                w13,
                w2,
                topk_weights[begin:end],
                topk_ids[begin:end],
                inplace=False,
                global_num_experts=num_experts,
                expert_map=expert_map,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
        if output is None:
            return result  # type: ignore[no-any-return]
        output[begin:end].copy_(result)
    if output is None:
        raise ValueError("empty expert-cache forward")
    return output


class ExpertCacheRegistry:
    def __init__(
        self,
        *,
        identity: ModelProfileIdentity,
        profile: ExpertProfile,
        tp_rank: int,
        resident_ratio: float,
        cache_slots: int,
        backend: PoolBackend,
        policy: str = "decayed-lfu",
    ) -> None:
        profile.validate(
            expected_geometry=identity["geometry"],
            expected_model_config_sha256=identity["model_config_sha256"],
            expected_model_identity_sha256=identity["model_identity_sha256"],
            expected_tensor_parallel_size=identity["tensor_parallel_size"],
        )
        if not all(count > 0 for count in profile.forward_counts):
            raise ValueError("startup profile must observe every layer")
        if not 0 <= tp_rank < identity["tensor_parallel_size"]:
            raise ValueError("invalid TP rank")
        if cache_slots < 1:
            raise ValueError("expert-cache requires at least one persistent cache slot")
        self.identity = identity
        self.tp_rank, self.tp_size = tp_rank, identity["tensor_parallel_size"]
        geometry = identity["geometry"]
        self.total_layers, self.num_experts = (
            geometry["total_layers"],
            geometry["num_experts"],
        )
        hidden, intermediate = geometry["hidden_size"], geometry["intermediate_size"]
        self.shapes = (
            (self.num_experts, 2 * intermediate, hidden),
            (self.num_experts, hidden, intermediate),
        )
        self.host = [
            HostExpertLayer(*self.shapes, pin_memory=False)
            for _ in range(self.total_layers)
        ]
        self.pool = ExpertPool(
            ExpertCachePolicy(
                self.total_layers,
                self.num_experts,
                resident_ratio,
                cache_slots,
                policy=policy,
                initial_counts=profile.counts,
            ),
            [(layer.w13, layer.w2) for layer in self.host],
            backend=backend,
            load_residents=False,
        )
        self.names: dict[int, str] = {}
        self.accumulators: dict[int, ExpertLoadAccumulator] = {}
        self.started = False

    def register_layer(
        self,
        layer_name: str,
        w13_shape: tuple[int, ...],
        w2_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
        idx = _layer_index(layer_name)
        if (
            idx in self.names
            or not 0 <= idx < self.total_layers
            or self.started
            or (w13_shape, w2_shape) != self.shapes
            or dtype != torch.bfloat16
        ):
            raise IntegrityError(
                "duplicate/late expert-cache layer or geometry mismatch"
            )
        self.names[idx] = layer_name
        self.accumulators[idx] = ExpertLoadAccumulator(
            layer_name, self.tp_rank, self.tp_size
        )
        # Shape-only placeholders alias the shared ingress. The inserted Top-k
        # branch always supplies the actual global pool to the native kernel.
        pair = tuple(
            torch.nn.Parameter(tensor[self.pool.ingress_start :], requires_grad=False)
            for tensor in (self.pool.w13, self.pool.w2)
        )
        for param, kind in zip(pair, ("w13", "w2"), strict=True):
            for attribute, value in (
                ("_expert_cache_layer", idx),
                ("_expert_cache_kind", kind),
            ):
                setattr(param, attribute, value)
        return pair[0], pair[1]

    def ingest(
        self,
        *,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        idx = getattr(param, "_expert_cache_layer", None)
        kind = getattr(param, "_expert_cache_kind", None)
        if (
            self.started
            or idx not in self.accumulators
            or _layer_index(weight_name) != idx
        ):
            raise IntegrityError("late/unregistered/mismatched expert-cache weight")
        if (
            (kind == "w13" and shard_id not in ("w1", "w3"))
            or (kind == "w2" and shard_id != "w2")
            or kind not in ("w13", "w2")
        ):
            raise IntegrityError("expert-cache checkpoint tensor kind mismatch")
        if type(expert_id) is not int or not 0 <= expert_id < self.num_experts:
            raise IntegrityError("expert ID out of range")
        accumulator = self.accumulators[idx]
        accumulator.ingest(shard_id, expert_id, loaded_weight)
        if accumulator.has_complete_expert(expert_id):
            self.host[idx].add(accumulator.finalize_expert(expert_id))

    def start(self) -> None:
        if self.started:
            return
        if len(self.names) != self.total_layers:
            raise IntegrityError("expert-cache layer registration incomplete")
        for idx, host in enumerate(self.host):
            host.require_complete()
            if self.accumulators[idx].pending_keys():
                raise IntegrityError("expert-cache checkpoint shards incomplete")
        self.pool.reload_residents()
        self.started = True

    def forward(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        activation: str,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        idx = _layer_index(layer_name)
        if self.names.get(idx) != layer_name:
            raise IntegrityError("unregistered expert-cache forward")
        if (
            hidden_states.dtype != torch.bfloat16
            or hidden_states.device != self.pool.backend.device
            or topk_weights.shape != topk_ids.shape
            or hidden_states.shape[0] != topk_ids.shape[0]
            or activation not in ("silu", "gelu")
        ):
            raise ValueError(
                "unsupported expert-cache forward geometry/dtype/activation"
            )
        self.start()
        return self.pool.execute(
            idx,
            topk_ids,
            lambda w13, w2, mapping: logical_fused_experts(
                hidden_states,
                w13,
                w2,
                topk_weights,
                topk_ids,
                mapping,
                num_experts=self.num_experts,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
            ),
        )

    def stats(
        self, *, synchronize: bool = True, reset_timing: bool = False
    ) -> dict[str, object]:
        return {
            **self.pool.stats(synchronize=synchronize, reset_timing=reset_timing),
            "rank": self.tp_rank,
            "tensor_parallel_size": self.tp_size,
            "identity": self.identity,
            "kernel_config_scope": "native logical E geometry per actual native chunk size",
            "transfer_schedule": "demand-critical H2D on compute stream; no speculative prefetch",
        }

    def reconfigure(self, resident_ratio: float) -> dict[str, object]:
        self.start()
        self.pool.reconfigure(resident_ratio)
        return self.stats()

    def close(self) -> None:
        self.pool.backend.synchronize()


_REGISTRY: ExpertCacheRegistry | None = None


def registry_for_layer(layer: torch.nn.Module, num_experts: int) -> ExpertCacheRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        config = importlib.import_module("vllm.config").get_current_vllm_config()
        validate_execution_config(config)
        tp_size, tp_rank = (
            getattr(layer, "tp_size", None),
            getattr(layer, "tp_rank", None),
        )
        if type(tp_size) is not int or type(tp_rank) is not int:
            raise IntegrityError("vLLM TP metadata unavailable")
        identity = model_profile_identity(os.environ["FLUXMOE_MODEL_PATH"], tp_size)
        profile = ExpertProfile.from_dict(
            json.loads(Path(os.environ["FLUXMOE_EXPERT_PROFILE_PATH"]).read_text())
        )
        _REGISTRY = ExpertCacheRegistry(
            identity=identity,
            profile=profile,
            tp_rank=tp_rank,
            resident_ratio=float(os.environ["FLUXMOE_RESIDENT_RATIO"]),
            cache_slots=int(os.environ["FLUXMOE_CACHE_SLOTS"]),
            policy=os.environ.get("FLUXMOE_CACHE_POLICY", "decayed-lfu"),
            backend=CudaPoolBackend(torch.cuda.current_device()),
        )
    if _REGISTRY.num_experts != num_experts:
        raise IntegrityError("expert count differs from startup profile")
    return _REGISTRY


def require_registry() -> ExpertCacheRegistry:
    if _REGISTRY is None:
        raise IntegrityError("expert-cache registry is not initialized")
    return _REGISTRY


def reset_registry() -> None:
    global _REGISTRY
    if _REGISTRY is not None:
        _REGISTRY.close()
    _REGISTRY = None


def store_expert_cache_weight(
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
