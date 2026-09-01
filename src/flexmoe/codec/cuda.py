"""CUDA preparation and launch helpers for canonical BF16 Huffman data."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from math import prod
from typing import Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray

from flexmoe.codec.reference import (
    CHUNK_ELEMENTS,
    EncodedBFloat16,
    canonical_codes,
)
from flexmoe.errors import IntegrityError


class _NativeCodecModule(Protocol):
    def huffman_decode(
        self,
        sign_mantissa: torch.Tensor,
        exponent_payload: torch.Tensor,
        chunk_byte_offsets: torch.Tensor,
        chunk_bit_lengths: torch.Tensor,
        trie_left: torch.Tensor,
        trie_right: torch.Tensor,
        trie_symbol: torch.Tensor,
        destination: torch.Tensor,
        errors: torch.Tensor,
        chunk_elements: int,
        stream_handle: int,
    ) -> None: ...


@lru_cache(maxsize=1)
def _native_module() -> _NativeCodecModule:
    try:
        module = importlib.import_module("flexmoe._C")
    except ImportError as error:
        raise RuntimeError(
            "flexmoe._C is unavailable; build the CUDA extension on Linux"
        ) from error
    if getattr(module, "huffman_decode", None) is None:
        raise RuntimeError("flexmoe._C does not expose huffman_decode")
    return cast(_NativeCodecModule, module)


def _validate_metadata(encoded: EncodedBFloat16) -> int:
    if not encoded.shape or prod(encoded.shape) != encoded.element_count:
        raise IntegrityError("encoded shape does not match element count")
    if len(encoded.sign_mantissa) != encoded.element_count:
        raise IntegrityError("sign/mantissa byte count mismatch")
    if encoded.chunk_elements != CHUNK_ELEMENTS:
        raise IntegrityError(
            f"CUDA codec requires {CHUNK_ELEMENTS}-element chunks"
        )
    expected_chunks = (
        encoded.element_count + encoded.chunk_elements - 1
    ) // encoded.chunk_elements
    if (
        len(encoded.chunk_byte_offsets) != expected_chunks
        or len(encoded.chunk_bit_lengths) != expected_chunks
    ):
        raise IntegrityError("chunk metadata count mismatch")
    if sum(encoded.chunk_bit_lengths) != encoded.bit_count:
        raise IntegrityError("aggregate bit count mismatch")
    if len(encoded.code_lengths) != 256:
        raise IntegrityError("code lengths must contain 256 values")

    expected_offset = 0
    for offset, bit_length in zip(
        encoded.chunk_byte_offsets,
        encoded.chunk_bit_lengths,
        strict=True,
    ):
        if offset != expected_offset:
            raise IntegrityError("chunk byte offsets are not contiguous")
        if bit_length <= 0:
            raise IntegrityError("chunk bit lengths must be positive")
        expected_offset += (bit_length + 7) // 8
    if expected_offset != len(encoded.exponent_payload):
        raise IntegrityError("exponent payload size does not match chunk metadata")
    return expected_chunks


def build_decode_trie(
    code_lengths: tuple[int, ...],
) -> tuple[NDArray[np.int16], NDArray[np.int16], NDArray[np.int16]]:
    codes = canonical_codes(code_lengths)
    left = [-1]
    right = [-1]
    symbols = [-1]

    for symbol, (code, length) in sorted(codes.items()):
        node = 0
        for shift in range(length - 1, -1, -1):
            if symbols[node] >= 0:
                raise IntegrityError("Huffman symbol is a prefix of another code")
            branch = right if ((code >> shift) & 1) else left
            child = branch[node]
            if child < 0:
                child = len(symbols)
                if child > 510:
                    raise IntegrityError("Huffman decode trie exceeds 511 nodes")
                branch[node] = child
                left.append(-1)
                right.append(-1)
                symbols.append(-1)
            node = child
        if symbols[node] >= 0 or left[node] >= 0 or right[node] >= 0:
            raise IntegrityError("duplicate or non-prefix-free Huffman symbol")
        symbols[node] = symbol

    return (
        np.asarray(left, dtype=np.int16),
        np.asarray(right, dtype=np.int16),
        np.asarray(symbols, dtype=np.int16),
    )


def _uint8_array(payload: bytes) -> NDArray[np.uint8]:
    return np.frombuffer(payload, dtype=np.uint8).copy()


@dataclass
class CudaEncodedBFloat16:
    shape: tuple[int, ...]
    element_count: int
    chunk_elements: int
    sign_mantissa: torch.Tensor
    exponent_payload: torch.Tensor
    chunk_byte_offsets: torch.Tensor
    chunk_bit_lengths: torch.Tensor
    trie_left: torch.Tensor
    trie_right: torch.Tensor
    trie_symbol: torch.Tensor
    errors: torch.Tensor

    @property
    def device(self) -> int:
        index = self.sign_mantissa.device.index
        if index is None:
            raise RuntimeError("CUDA tensor has no device index")
        return index

    def launch(self, destination: torch.Tensor, stream_handle: int) -> None:
        if destination.device.type != "cuda" or destination.device.index != self.device:
            raise ValueError("destination must be on the encoded CUDA device")
        if destination.dtype is not torch.bfloat16:
            raise TypeError("destination must have BF16 dtype")
        if not destination.is_contiguous():
            raise ValueError("destination must be contiguous")
        if tuple(destination.shape) != self.shape:
            raise ValueError(
                f"destination shape {tuple(destination.shape)} does not match {self.shape}"
            )
        if type(stream_handle) is not int or stream_handle < 0:
            raise ValueError("stream_handle must be a non-negative int")
        _native_module().huffman_decode(
            self.sign_mantissa,
            self.exponent_payload,
            self.chunk_byte_offsets,
            self.chunk_bit_lengths,
            self.trie_left,
            self.trie_right,
            self.trie_symbol,
            destination,
            self.errors,
            self.chunk_elements,
            stream_handle,
        )

    def raise_for_errors(self) -> None:
        error_values = self.errors.cpu().tolist()
        failures = [
            (chunk_idx, error_code)
            for chunk_idx, error_code in enumerate(error_values)
            if error_code
        ]
        if failures:
            chunk_idx, error_code = failures[0]
            raise IntegrityError(
                f"CUDA Huffman decode failed in chunk {chunk_idx} "
                f"with error code {error_code}"
            )


def prepare_cuda_encoded(
    encoded: EncodedBFloat16, *, device: int
) -> CudaEncodedBFloat16:
    if type(device) is not int or device < 0:
        raise ValueError("device must be a non-negative int")
    expected_chunks = _validate_metadata(encoded)
    left, right, symbols = build_decode_trie(encoded.code_lengths)
    cuda_device = torch.device("cuda", device)
    with torch.cuda.device(cuda_device):
        prepared = CudaEncodedBFloat16(
            shape=encoded.shape,
            element_count=encoded.element_count,
            chunk_elements=encoded.chunk_elements,
            sign_mantissa=torch.from_numpy(
                _uint8_array(encoded.sign_mantissa)
            ).to(cuda_device),
            exponent_payload=torch.from_numpy(
                _uint8_array(encoded.exponent_payload)
            ).to(cuda_device),
            chunk_byte_offsets=torch.tensor(
                encoded.chunk_byte_offsets, dtype=torch.int64, device=cuda_device
            ),
            chunk_bit_lengths=torch.tensor(
                encoded.chunk_bit_lengths, dtype=torch.int64, device=cuda_device
            ),
            trie_left=torch.from_numpy(left).to(cuda_device),
            trie_right=torch.from_numpy(right).to(cuda_device),
            trie_symbol=torch.from_numpy(symbols).to(cuda_device),
            errors=torch.empty(
                expected_chunks, dtype=torch.int32, device=cuda_device
            ),
        )
        torch.cuda.current_stream(device).synchronize()
    return prepared


def cuda_decode(
    encoded: EncodedBFloat16,
    *,
    device: int,
    destination: torch.Tensor | None = None,
) -> torch.Tensor:
    """Synchronously validate CUDA decode; storage backends use async launch."""

    prepared = prepare_cuda_encoded(encoded, device=device)
    if destination is None:
        destination = torch.empty(
            encoded.shape,
            dtype=torch.bfloat16,
            device=torch.device("cuda", device),
        )
    stream = torch.cuda.current_stream(device)
    prepared.launch(destination, stream.cuda_stream)
    stream.synchronize()
    prepared.raise_for_errors()
    return destination


__all__ = [
    "CudaEncodedBFloat16",
    "build_decode_trie",
    "cuda_decode",
    "prepare_cuda_encoded",
]
