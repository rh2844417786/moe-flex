"""TP-aware accumulation of routed-expert checkpoint shards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

import torch

from flexmoe.errors import ConfigurationError, IntegrityError


@dataclass(frozen=True)
class ExpertWeights:
    expert_id: int
    w13: torch.Tensor
    w2: torch.Tensor


class ExpertLoadAccumulator:
    def __init__(self, layer_name: str, tp_rank: int, tp_size: int) -> None:
        if not layer_name:
            raise ConfigurationError("layer_name must not be empty")
        if type(tp_size) is not int or tp_size <= 0:
            raise ConfigurationError("tp_size must be a positive int")
        if type(tp_rank) is not int or not 0 <= tp_rank < tp_size:
            raise ConfigurationError("tp_rank must be inside the TP world")
        self.layer_name = layer_name
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self._parts: dict[tuple[int, str], torch.Tensor] = {}

    def ingest(
        self,
        shard_id: str,
        expert_id: int,
        loaded_weight: torch.Tensor,
    ) -> None:
        if shard_id not in {"w1", "w2", "w3"}:
            raise IntegrityError(f"unexpected shard {shard_id}")
        if type(expert_id) is not int or expert_id < 0:
            raise IntegrityError("expert_id must be a non-negative int")
        if loaded_weight.dtype is not torch.bfloat16:
            raise IntegrityError("expert weights must have BF16 dtype")
        if loaded_weight.ndim != 2:
            raise IntegrityError("expert weights must be two-dimensional")
        if loaded_weight.numel() == 0:
            raise IntegrityError("expert weights must not be empty")
        key = (expert_id, shard_id)
        if key in self._parts:
            raise IntegrityError(f"duplicate expert shard {key}")
        self._parts[key] = (
            loaded_weight.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        )

    def _pop(self, expert_id: int, shard_id: str) -> torch.Tensor:
        try:
            return self._parts.pop((expert_id, shard_id))
        except KeyError as error:
            raise IntegrityError(
                f"missing expert shard {(expert_id, shard_id)}"
            ) from error

    def finalize_w13(self, expert_id: int) -> torch.Tensor:
        w1 = self._pop(expert_id, "w1")
        w3 = self._pop(expert_id, "w3")
        if tuple(w1.shape) != tuple(w3.shape):
            raise IntegrityError("w1 and w3 shapes do not match")
        if w1.shape[0] % self.tp_size:
            raise IntegrityError("w1/w3 output dimension is not TP-divisible")
        rows = w1.shape[0] // self.tp_size
        start = rows * self.tp_rank
        return torch.cat(
            (
                w1.narrow(0, start, rows),
                w3.narrow(0, start, rows),
            ),
            dim=0,
        ).contiguous()

    def finalize_w2(self, expert_id: int) -> torch.Tensor:
        w2 = self._pop(expert_id, "w2")
        if w2.shape[1] % self.tp_size:
            raise IntegrityError("w2 input dimension is not TP-divisible")
        columns = w2.shape[1] // self.tp_size
        return w2.narrow(1, columns * self.tp_rank, columns).contiguous()

    def has_complete_expert(self, expert_id: int) -> bool:
        return all(
            (expert_id, shard_id) in self._parts
            for shard_id in ("w1", "w2", "w3")
        )

    def finalize_expert(self, expert_id: int) -> ExpertWeights:
        if not self.has_complete_expert(expert_id):
            raise IntegrityError(f"expert {expert_id} is incomplete")
        w13 = self.finalize_w13(expert_id)
        w2 = self.finalize_w2(expert_id)
        return ExpertWeights(expert_id=expert_id, w13=w13, w2=w2)

    def pending_keys(self) -> tuple[tuple[int, str], ...]:
        return tuple(sorted(self._parts))


@overload
def store_expert_weight(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    expert_id: int,
    return_success: Literal[False] = False,
) -> None: ...


@overload
def store_expert_weight(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    expert_id: int,
    return_success: Literal[True] = True,
) -> bool: ...


def store_expert_weight(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    expert_id: int,
    return_success: bool = False,
) -> bool | None:
    """Divert routed-expert weights to the active FluxMoE registry."""

    from flexmoe.vllm.bridge import require_active_registry

    registry = require_active_registry()
    registry.ingest(
        param=param,
        loaded_weight=loaded_weight,
        weight_name=weight_name,
        shard_id=shard_id,
        expert_id=expert_id,
    )
    return True if return_success else None


__all__ = ["ExpertLoadAccumulator", "ExpertWeights", "store_expert_weight"]
