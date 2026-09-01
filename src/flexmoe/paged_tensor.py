"""Typed Python wrapper for CUDA virtual-memory expert tensor regions."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, cast

import torch


class _NativePagedRegion(Protocol):
    device: int
    base_address: int
    virtual_bytes: int
    granularity: int

    def create_block(self, nbytes: int) -> int: ...

    def map(self, offset: int, block_id: int, nbytes: int) -> None: ...

    def unmap(self, offset: int, nbytes: int) -> None: ...

    def tensor(
        self, offset: int, shape: Sequence[int], dtype_code: int
    ) -> torch.Tensor: ...

    def snapshot(self) -> dict[str, int]: ...


class _NativeModule(Protocol):
    PagedRegion: Callable[[int, int], _NativePagedRegion]


@lru_cache(maxsize=1)
def _native_module() -> _NativeModule:
    try:
        module = importlib.import_module("flexmoe._C")
    except ImportError as error:
        raise RuntimeError(
            "flexmoe._C is unavailable; build the CUDA extension on Linux"
        ) from error
    if getattr(module, "PagedRegion", None) is None:
        raise RuntimeError("flexmoe._C does not expose PagedRegion")
    return cast(_NativeModule, module)


@dataclass(frozen=True)
class PagedRegionSnapshot:
    device: int
    base_address: int
    virtual_bytes: int
    granularity: int
    block_count: int
    mapping_count: int
    mapped_bytes: int


_DTYPE_CODES: dict[torch.dtype, int] = {
    torch.bfloat16: 0,
    torch.float16: 1,
    torch.float32: 2,
    torch.uint8: 3,
    torch.int8: 4,
    torch.int32: 5,
    torch.int64: 6,
}


def _require_plain_int(name: str, value: int, *, positive: bool = False) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{name} must be non-negative")


class PagedTensorRegion:
    """Own a stable CUDA VA reservation and replaceable physical allocations."""

    def __init__(self, device: int, virtual_bytes: int) -> None:
        _require_plain_int("device", device)
        _require_plain_int("virtual_bytes", virtual_bytes, positive=True)
        self._native = _native_module().PagedRegion(device, virtual_bytes)

    @property
    def device(self) -> int:
        return self._native.device

    @property
    def base_address(self) -> int:
        return self._native.base_address

    @property
    def virtual_bytes(self) -> int:
        return self._native.virtual_bytes

    @property
    def granularity(self) -> int:
        return self._native.granularity

    def _validate_page_range(self, offset: int, nbytes: int) -> None:
        _require_plain_int("offset", offset)
        _require_plain_int("nbytes", nbytes, positive=True)
        if offset % self.granularity != 0 or nbytes % self.granularity != 0:
            raise ValueError(
                "offset and nbytes must be aligned to the CUDA VMM granularity"
            )
        if offset + nbytes > self.virtual_bytes:
            raise ValueError("range exceeds the virtual reservation")

    def create_block(self, nbytes: int) -> int:
        self._validate_page_range(0, nbytes)
        return self._native.create_block(nbytes)

    def map(self, offset: int, block_id: int, nbytes: int) -> None:
        self._validate_page_range(offset, nbytes)
        _require_plain_int("block_id", block_id, positive=True)
        self._native.map(offset, block_id, nbytes)

    def unmap(self, offset: int, nbytes: int) -> None:
        self._validate_page_range(offset, nbytes)
        self._native.unmap(offset, nbytes)

    def tensor(
        self, offset: int, shape: Sequence[int], dtype: torch.dtype
    ) -> torch.Tensor:
        _require_plain_int("offset", offset)
        normalized_shape = tuple(shape)
        if not normalized_shape:
            raise ValueError("shape must contain at least one axis")
        for dimension in normalized_shape:
            _require_plain_int("shape dimension", dimension, positive=True)
        try:
            dtype_code = _DTYPE_CODES[dtype]
        except (KeyError, TypeError) as error:
            raise TypeError(f"unsupported dtype: {dtype}") from error
        return self._native.tensor(offset, normalized_shape, dtype_code)

    def snapshot(self) -> PagedRegionSnapshot:
        raw = self._native.snapshot()
        try:
            return PagedRegionSnapshot(
                device=raw["device"],
                base_address=raw["base_address"],
                virtual_bytes=raw["virtual_bytes"],
                granularity=raw["granularity"],
                block_count=raw["block_count"],
                mapping_count=raw["mapping_count"],
                mapped_bytes=raw["mapped_bytes"],
            )
        except KeyError as error:
            raise RuntimeError(f"native snapshot is missing {error.args[0]}") from error


__all__ = ["PagedRegionSnapshot", "PagedTensorRegion"]
