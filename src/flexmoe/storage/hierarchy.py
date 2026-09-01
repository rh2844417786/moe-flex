"""Deterministic routing from tensor IDs to expert storage backends."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from flexmoe.errors import ConfigurationError, IntegrityError
from flexmoe.storage.base import ExpertTensorStore, MaterializationReceipt


class StorageHierarchy:
    def __init__(
        self,
        *,
        stores: Mapping[str, ExpertTensorStore],
        tensor_layers: Mapping[str, int],
    ) -> None:
        if not stores:
            raise ConfigurationError("at least one expert tensor store is required")
        if set(stores) != set(tensor_layers):
            raise ConfigurationError(
                "stores and tensor_layers must contain identical tensor IDs"
            )
        normalized_layers: dict[str, int] = {}
        for tensor_id, layer_idx in tensor_layers.items():
            if not tensor_id:
                raise ConfigurationError("tensor IDs must not be empty")
            if type(layer_idx) is not int or layer_idx < 0:
                raise ConfigurationError("tensor layer indices must be non-negative ints")
            normalized_layers[tensor_id] = layer_idx
        self._stores = dict(stores)
        self._tensor_layers = normalized_layers

    def materialize_layer(
        self,
        layer_idx: int,
        destinations: Mapping[str, torch.Tensor],
        *,
        stream: int,
    ) -> tuple[MaterializationReceipt, ...]:
        if type(layer_idx) is not int or layer_idx < 0:
            raise ValueError("layer_idx must be a non-negative int")
        if type(stream) is not int or stream < 0:
            raise ValueError("stream must be a non-negative int handle")
        expected = {
            tensor_id
            for tensor_id, assigned_layer in self._tensor_layers.items()
            if assigned_layer == layer_idx
        }
        if not expected:
            raise IntegrityError(f"no expert tensors are assigned to layer {layer_idx}")
        if set(destinations) != expected:
            raise IntegrityError(
                "destination tensor IDs do not match the assigned layer: "
                f"expected {sorted(expected)}, got {sorted(destinations)}"
            )

        receipts: list[MaterializationReceipt] = []
        for tensor_id in sorted(expected):
            destination = destinations[tensor_id]
            if not isinstance(destination, torch.Tensor):
                raise TypeError(f"destination {tensor_id} must be a torch.Tensor")
            receipt = self._stores[tensor_id].materialize(
                tensor_id, destination, stream
            )
            if receipt.tensor_id != tensor_id:
                raise IntegrityError(
                    f"store returned receipt for {receipt.tensor_id}, expected {tensor_id}"
                )
            expected_bytes = destination.numel() * destination.element_size()
            if receipt.nbytes != expected_bytes:
                raise IntegrityError(
                    f"receipt byte count for {tensor_id} is {receipt.nbytes}, "
                    f"expected {expected_bytes}"
                )
            receipts.append(receipt)
        return tuple(receipts)


__all__ = ["StorageHierarchy"]
