"""GPU-resident compressed expert tensors decoded into mapped BF16 views."""

from __future__ import annotations

from collections.abc import Mapping
from sys import float_info
from time import perf_counter

import torch

from flexmoe.codec.cuda import CudaEncodedBFloat16, prepare_cuda_encoded
from flexmoe.codec.reference import EncodedBFloat16
from flexmoe.errors import ConfigurationError, IntegrityError
from flexmoe.storage.base import MaterializationReceipt


class GpuCompressedStore:
    def __init__(
        self, encoded_tensors: Mapping[str, EncodedBFloat16], *, device: int
    ) -> None:
        if not encoded_tensors:
            raise ConfigurationError("encoded_tensors must not be empty")
        if type(device) is not int or device < 0:
            raise ConfigurationError("device must be a non-negative int")
        self._encoded: dict[str, CudaEncodedBFloat16] = {}
        for tensor_id in sorted(encoded_tensors):
            if not tensor_id:
                raise ConfigurationError("tensor IDs must not be empty")
            self._encoded[tensor_id] = prepare_cuda_encoded(
                encoded_tensors[tensor_id], device=device
            )

    def materialize(
        self,
        tensor_id: str,
        destination: torch.Tensor,
        stream: int,
    ) -> MaterializationReceipt:
        try:
            encoded = self._encoded[tensor_id]
        except KeyError as error:
            raise IntegrityError(f"unknown compressed tensor {tensor_id}") from error
        started = perf_counter()
        encoded.launch(destination, stream)
        elapsed = max(perf_counter() - started, float_info.min)
        return MaterializationReceipt(
            tensor_id=tensor_id,
            backend="gpu_compressed",
            nbytes=destination.numel() * destination.element_size(),
            elapsed_s=elapsed,
        )

    def raise_for_decode_errors(self, tensor_id: str) -> None:
        try:
            encoded = self._encoded[tensor_id]
        except KeyError as error:
            raise IntegrityError(f"unknown compressed tensor {tensor_id}") from error
        encoded.raise_for_errors()


__all__ = ["GpuCompressedStore"]
