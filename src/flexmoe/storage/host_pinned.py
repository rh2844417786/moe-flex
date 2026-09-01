"""Pinned-host BF16 expert tensor storage with asynchronous HtoD copies."""

from __future__ import annotations

from collections.abc import Mapping
from sys import float_info
from time import perf_counter

import torch

from flexmoe.errors import ConfigurationError, IntegrityError
from flexmoe.storage.base import MaterializationReceipt


class PinnedHostStore:
    def __init__(self, tensors: Mapping[str, torch.Tensor]) -> None:
        if not tensors:
            raise ConfigurationError("pinned host tensors must not be empty")
        self._tensors: dict[str, torch.Tensor] = {}
        for tensor_id in sorted(tensors):
            source = tensors[tensor_id]
            if not tensor_id:
                raise ConfigurationError("tensor IDs must not be empty")
            if source.device.type != "cpu":
                raise ConfigurationError(f"{tensor_id} must originate on CPU")
            if source.dtype is not torch.bfloat16:
                raise ConfigurationError(f"{tensor_id} must have BF16 dtype")
            if source.numel() == 0:
                raise ConfigurationError(f"{tensor_id} must not be empty")
            self._tensors[tensor_id] = source.contiguous().pin_memory()

    def source(self, tensor_id: str) -> torch.Tensor:
        try:
            return self._tensors[tensor_id]
        except KeyError as error:
            raise IntegrityError(f"unknown pinned tensor {tensor_id}") from error

    def materialize(
        self,
        tensor_id: str,
        destination: torch.Tensor,
        stream: int,
    ) -> MaterializationReceipt:
        source = self.source(tensor_id)
        if destination.device.type != "cuda":
            raise ValueError("destination must be a CUDA tensor")
        if destination.dtype is not torch.bfloat16:
            raise TypeError("destination must have BF16 dtype")
        if not destination.is_contiguous():
            raise ValueError("destination must be contiguous")
        if tuple(destination.shape) != tuple(source.shape):
            raise ValueError(
                f"destination shape {tuple(destination.shape)} does not match "
                f"source shape {tuple(source.shape)}"
            )
        if type(stream) is not int or stream < 0:
            raise ValueError("stream must be a non-negative int handle")
        device_index = destination.device.index
        if device_index is None:
            raise ValueError("destination CUDA device has no index")
        load_stream = torch.cuda.ExternalStream(  # type: ignore[no-untyped-call]
            stream, device=device_index
        )

        started = perf_counter()
        with torch.cuda.stream(load_stream):
            destination.copy_(source, non_blocking=True)
        elapsed = max(perf_counter() - started, float_info.min)
        return MaterializationReceipt(
            tensor_id=tensor_id,
            backend="host_pinned",
            nbytes=destination.numel() * destination.element_size(),
            elapsed_s=elapsed,
        )


__all__ = ["PinnedHostStore"]
