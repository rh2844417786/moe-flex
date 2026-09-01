"""Shared contracts for expert tensor materialization backends."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Protocol

import torch

StorageBackend = Literal["gpu_compressed", "host_pinned"]


@dataclass(frozen=True)
class MaterializationReceipt:
    tensor_id: str
    backend: StorageBackend
    nbytes: int
    elapsed_s: float

    def __post_init__(self) -> None:
        if not self.tensor_id:
            raise ValueError("tensor_id must not be empty")
        if self.backend not in {"gpu_compressed", "host_pinned"}:
            raise ValueError(f"unsupported storage backend: {self.backend}")
        if type(self.nbytes) is not int or self.nbytes <= 0:
            raise ValueError("nbytes must be a positive int")
        if not isfinite(self.elapsed_s) or self.elapsed_s <= 0:
            raise ValueError("elapsed_s must be finite and positive")


class ExpertTensorStore(Protocol):
    def materialize(
        self,
        tensor_id: str,
        destination: torch.Tensor,
        stream: int,
    ) -> MaterializationReceipt: ...


__all__ = ["ExpertTensorStore", "MaterializationReceipt", "StorageBackend"]
