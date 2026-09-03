"""Packed per-layer descriptors assembled from per-expert Huffman streams."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from flexmoe.codec.cuda import build_decode_trie
from flexmoe.codec.reference import (
    CHUNK_ELEMENTS,
    SEGMENTS_PER_CHUNK,
    EncodedBFloat16,
)
from flexmoe.errors import ConfigurationError, IntegrityError


_INT64_MAX = (1 << 63) - 1


def _checked_add(left: int, right: int, name: str) -> int:
    value = left + right
    if left < 0 or right < 0 or value > _INT64_MAX:
        raise IntegrityError(f"{name} exceeds signed 64-bit range")
    return value


@dataclass(frozen=True)
class PackedLayerDescriptor:
    """Immutable data needed for one layer-kind batched decode launch."""

    shape: tuple[int, ...]
    expert_count: int
    expert_elements: int
    chunk_elements: int
    sign_mantissa: bytes
    exponent_payload: bytes
    chunk_byte_offsets: tuple[int, ...]
    chunk_bit_lengths: tuple[int, ...]
    chunk_destination_offsets: tuple[int, ...]
    chunk_element_counts: tuple[int, ...]
    chunk_expert_indices: tuple[int, ...]
    segment_bit_offsets: bytes
    trie_left: tuple[int, ...]
    trie_right: tuple[int, ...]
    trie_symbol: tuple[int, ...]
    expert_trie_offsets: tuple[int, ...]
    expert_trie_node_counts: tuple[int, ...]

    @property
    def element_count(self) -> int:
        return prod(self.shape)

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_byte_offsets)

    @property
    def source_bytes(self) -> int:
        return self.element_count * 2

    @property
    def encoded_payload_bytes(self) -> int:
        return len(self.sign_mantissa) + len(self.exponent_payload)

    @property
    def descriptor_bytes(self) -> int:
        chunk_int64_fields = 5
        expert_int64_fields = 2
        return (
            self.chunk_count * chunk_int64_fields * 8
            + len(self.segment_bit_offsets)
            + self.expert_count * expert_int64_fields * 8
        )

    @property
    def codebook_bytes(self) -> int:
        return (len(self.trie_left) + len(self.trie_right) + len(self.trie_symbol)) * 2

    @property
    def error_buffer_bytes(self) -> int:
        return self.chunk_count * 4

    @property
    def gpu_storage_bytes(self) -> int:
        return (
            self.encoded_payload_bytes
            + self.descriptor_bytes
            + self.codebook_bytes
            + self.error_buffer_bytes
        )


def pack_layer_descriptor(
    encoded_experts: dict[int, EncodedBFloat16],
    *,
    destination_shape: tuple[int, ...],
) -> PackedLayerDescriptor:
    """Pack complete, identically-shaped expert encodings without re-encoding."""

    if not encoded_experts:
        raise ConfigurationError("encoded experts must not be empty")
    if len(destination_shape) < 2 or any(
        dimension <= 0 for dimension in destination_shape
    ):
        raise ConfigurationError("destination shape must contain positive dimensions")
    expert_count = destination_shape[0]
    expected_experts = set(range(expert_count))
    if set(encoded_experts) != expected_experts:
        raise IntegrityError(
            "encoded expert IDs must be contiguous: "
            f"expected {sorted(expected_experts)}, got {sorted(encoded_experts)}"
        )
    expert_shape = destination_shape[1:]
    expert_elements = prod(expert_shape)

    sign_mantissa = bytearray()
    exponent_payload = bytearray()
    chunk_byte_offsets: list[int] = []
    chunk_bit_lengths: list[int] = []
    chunk_destination_offsets: list[int] = []
    chunk_element_counts: list[int] = []
    chunk_expert_indices: list[int] = []
    segment_bit_offsets = bytearray()
    trie_left: list[int] = []
    trie_right: list[int] = []
    trie_symbol: list[int] = []
    expert_trie_offsets: list[int] = []
    expert_trie_node_counts: list[int] = []

    destination_cursor = 0
    for expert_idx in range(expert_count):
        encoded = encoded_experts[expert_idx]
        if encoded.shape != expert_shape or encoded.element_count != expert_elements:
            raise IntegrityError(
                f"expert {expert_idx} shape does not match destination"
            )
        if encoded.chunk_elements != CHUNK_ELEMENTS:
            raise IntegrityError(f"expert {expert_idx} has unsupported chunk size")
        if len(encoded.sign_mantissa) != expert_elements:
            raise IntegrityError(f"expert {expert_idx} sign/mantissa size is invalid")
        expected_chunks = (expert_elements + CHUNK_ELEMENTS - 1) // CHUNK_ELEMENTS
        if (
            len(encoded.chunk_byte_offsets) != expected_chunks
            or len(encoded.chunk_bit_lengths) != expected_chunks
        ):
            raise IntegrityError(f"expert {expert_idx} chunk metadata is incomplete")
        expected_segment_bytes = expected_chunks * SEGMENTS_PER_CHUNK * 4
        if len(encoded.segment_bit_offsets) != expected_segment_bytes:
            raise IntegrityError(f"expert {expert_idx} segment metadata is incomplete")

        payload_base = len(exponent_payload)
        expected_payload_offset = 0
        for local_chunk, (byte_offset, bit_length) in enumerate(
            zip(
                encoded.chunk_byte_offsets,
                encoded.chunk_bit_lengths,
                strict=True,
            )
        ):
            if byte_offset != expected_payload_offset or bit_length <= 0:
                raise IntegrityError(
                    f"expert {expert_idx} chunk payload is not contiguous"
                )
            byte_length = (bit_length + 7) // 8
            expected_payload_offset = _checked_add(
                expected_payload_offset, byte_length, "expert payload offset"
            )
            global_offset = _checked_add(payload_base, byte_offset, "payload offset")
            local_destination = local_chunk * CHUNK_ELEMENTS
            element_count = min(CHUNK_ELEMENTS, expert_elements - local_destination)
            if element_count <= 0:
                raise IntegrityError(f"expert {expert_idx} contains an empty chunk")
            chunk_byte_offsets.append(global_offset)
            chunk_bit_lengths.append(bit_length)
            chunk_destination_offsets.append(
                _checked_add(
                    destination_cursor, local_destination, "destination offset"
                )
            )
            chunk_element_counts.append(element_count)
            chunk_expert_indices.append(expert_idx)
        if expected_payload_offset != len(encoded.exponent_payload):
            raise IntegrityError(
                f"expert {expert_idx} exponent payload size is invalid"
            )
        if sum(encoded.chunk_bit_lengths) != encoded.bit_count:
            raise IntegrityError(f"expert {expert_idx} aggregate bit count is invalid")
        segment_bit_offsets.extend(encoded.segment_bit_offsets)

        left, right, symbol = build_decode_trie(encoded.code_lengths)
        trie_base = len(trie_symbol)
        trie_nodes = len(symbol)
        expert_trie_offsets.append(trie_base)
        expert_trie_node_counts.append(trie_nodes)
        trie_left.extend(int(branch) for branch in left.tolist())
        trie_right.extend(int(branch) for branch in right.tolist())
        trie_symbol.extend(int(value) for value in symbol.tolist())

        sign_mantissa.extend(encoded.sign_mantissa)
        exponent_payload.extend(encoded.exponent_payload)
        destination_cursor = _checked_add(
            destination_cursor, expert_elements, "destination cursor"
        )

    if destination_cursor != prod(destination_shape):
        raise IntegrityError("packed descriptor leaves a destination hole")
    for chunk_idx, expert_idx in enumerate(chunk_expert_indices):
        trie_offset = expert_trie_offsets[expert_idx]
        trie_nodes = expert_trie_node_counts[expert_idx]
        if (
            trie_offset < 0
            or trie_nodes <= 1
            or trie_offset + trie_nodes > len(trie_symbol)
        ):
            raise IntegrityError(f"chunk {chunk_idx} references an invalid codebook")

    return PackedLayerDescriptor(
        shape=destination_shape,
        expert_count=expert_count,
        expert_elements=expert_elements,
        chunk_elements=CHUNK_ELEMENTS,
        sign_mantissa=bytes(sign_mantissa),
        exponent_payload=bytes(exponent_payload),
        chunk_byte_offsets=tuple(chunk_byte_offsets),
        chunk_bit_lengths=tuple(chunk_bit_lengths),
        chunk_destination_offsets=tuple(chunk_destination_offsets),
        chunk_element_counts=tuple(chunk_element_counts),
        chunk_expert_indices=tuple(chunk_expert_indices),
        segment_bit_offsets=bytes(segment_bit_offsets),
        trie_left=tuple(trie_left),
        trie_right=tuple(trie_right),
        trie_symbol=tuple(trie_symbol),
        expert_trie_offsets=tuple(expert_trie_offsets),
        expert_trie_node_counts=tuple(expert_trie_node_counts),
    )


__all__ = ["PackedLayerDescriptor", "pack_layer_descriptor"]
