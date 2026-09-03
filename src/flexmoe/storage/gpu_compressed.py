"""GPU-resident compressed expert tensors decoded into mapped BF16 views."""

from __future__ import annotations

from collections.abc import Mapping
from sys import float_info
from time import perf_counter

import torch

from flexmoe.codec.cuda import (
    CudaEncodedBFloat16,
    CudaPackedLayerDescriptor,
    prepare_cuda_encoded,
    prepare_cuda_packed,
)
from flexmoe.codec.packed import PackedLayerDescriptor
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

    @property
    def source_bytes(self) -> int:
        return sum(encoded.element_count * 2 for encoded in self._encoded.values())

    @property
    def storage_bytes(self) -> int:
        return sum(encoded.storage_bytes for encoded in self._encoded.values())

    def input_bytes(self, tensor_id: str) -> int:
        try:
            return self._encoded[tensor_id].input_bytes
        except KeyError as error:
            raise IntegrityError(f"unknown compressed tensor {tensor_id}") from error

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


class BatchedGpuCompressedStore:
    """GPU compressed layer-kind payloads decoded with one launch each."""

    def __init__(
        self,
        descriptors: Mapping[str, PackedLayerDescriptor],
        *,
        device: int,
    ) -> None:
        if not descriptors:
            raise ConfigurationError("packed descriptors must not be empty")
        if type(device) is not int or device < 0:
            raise ConfigurationError("device must be a non-negative int")
        self._encoded: dict[str, CudaPackedLayerDescriptor] = {}
        self._source_bytes = 0
        for layer_kind in sorted(descriptors):
            if not layer_kind:
                raise ConfigurationError("layer-kind IDs must not be empty")
            descriptor = descriptors[layer_kind]
            self._encoded[layer_kind] = prepare_cuda_packed(descriptor, device=device)
            self._source_bytes += descriptor.source_bytes

    @property
    def source_bytes(self) -> int:
        return self._source_bytes

    @property
    def storage_bytes(self) -> int:
        return sum(encoded.storage_bytes for encoded in self._encoded.values())

    def input_bytes(self, layer_kind: str) -> int:
        try:
            return self._encoded[layer_kind].input_bytes
        except KeyError as error:
            raise IntegrityError(f"unknown packed layer-kind {layer_kind}") from error

    def materialize_batched(
        self,
        layer_kind: str,
        destination: torch.Tensor,
        stream: int,
    ) -> MaterializationReceipt:
        try:
            encoded = self._encoded[layer_kind]
        except KeyError as error:
            raise IntegrityError(f"unknown packed layer-kind {layer_kind}") from error
        started = perf_counter()
        encoded.launch(destination, stream)
        elapsed = max(perf_counter() - started, float_info.min)
        return MaterializationReceipt(
            tensor_id=layer_kind,
            backend="gpu_compressed",
            nbytes=destination.numel() * destination.element_size(),
            elapsed_s=elapsed,
        )

    def raise_for_decode_errors(self, layer_kind: str) -> None:
        try:
            encoded = self._encoded[layer_kind]
        except KeyError as error:
            raise IntegrityError(f"unknown packed layer-kind {layer_kind}") from error
        encoded.raise_for_errors()


__all__ = ["BatchedGpuCompressedStore", "GpuCompressedStore"]
