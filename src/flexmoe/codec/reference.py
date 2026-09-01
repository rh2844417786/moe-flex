"""Canonical CPU Huffman reference for raw BF16 bit patterns."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import prod

from flexmoe.errors import ConfigurationError, IntegrityError

CHUNK_ELEMENTS = 4096


@dataclass(frozen=True)
class EncodedBFloat16:
    """Losslessly encoded BF16 signs/mantissas and Huffman exponents."""

    shape: tuple[int, ...]
    element_count: int
    bit_count: int
    chunk_elements: int
    chunk_byte_offsets: tuple[int, ...]
    chunk_bit_lengths: tuple[int, ...]
    sign_mantissa: bytes
    exponent_payload: bytes
    code_lengths: tuple[int, ...]


@dataclass
class _Node:
    symbol: int | None
    minimum_symbol: int
    left: _Node | None = None
    right: _Node | None = None


def _code_lengths(symbols: list[int]) -> tuple[int, ...]:
    frequencies = [0] * 256
    for symbol in symbols:
        frequencies[symbol] += 1

    heap: list[tuple[int, int, int, _Node]] = []
    node_id = 0
    for symbol, frequency in enumerate(frequencies):
        if frequency:
            heapq.heappush(
                heap,
                (frequency, symbol, node_id, _Node(symbol, symbol)),
            )
            node_id += 1

    lengths = [0] * 256
    if len(heap) == 1:
        lengths[heap[0][3].minimum_symbol] = 1
        return tuple(lengths)

    while len(heap) > 1:
        left_frequency, _, _, left = heapq.heappop(heap)
        right_frequency, _, _, right = heapq.heappop(heap)
        parent = _Node(
            symbol=None,
            minimum_symbol=min(left.minimum_symbol, right.minimum_symbol),
            left=left,
            right=right,
        )
        heapq.heappush(
            heap,
            (
                left_frequency + right_frequency,
                parent.minimum_symbol,
                node_id,
                parent,
            ),
        )
        node_id += 1

    stack = [(heap[0][3], 0)]
    while stack:
        node, depth = stack.pop()
        if node.symbol is not None:
            lengths[node.symbol] = max(1, depth)
            continue
        assert node.left is not None and node.right is not None
        stack.append((node.right, depth + 1))
        stack.append((node.left, depth + 1))
    return tuple(lengths)


def _canonical_codes(code_lengths: tuple[int, ...]) -> dict[int, tuple[int, int]]:
    if len(code_lengths) != 256 or any(length < 0 for length in code_lengths):
        raise IntegrityError("code lengths must contain 256 non-negative values")
    ordered = sorted(
        (length, symbol)
        for symbol, length in enumerate(code_lengths)
        if length > 0
    )
    if not ordered:
        raise IntegrityError("at least one Huffman symbol is required")

    codes: dict[int, tuple[int, int]] = {}
    code = 0
    previous_length = 0
    for length, symbol in ordered:
        code <<= length - previous_length
        if code >= 1 << length:
            raise IntegrityError("oversubscribed canonical Huffman table")
        codes[symbol] = (code, length)
        code += 1
        previous_length = length
    return codes


def _encode_chunk(
    symbols: list[int], codes: dict[int, tuple[int, int]]
) -> tuple[bytes, int]:
    payload = bytearray()
    current_byte = 0
    filled_bits = 0
    bit_count = 0
    for symbol in symbols:
        code, length = codes[symbol]
        bit_count += length
        for shift in range(length - 1, -1, -1):
            current_byte = (current_byte << 1) | ((code >> shift) & 1)
            filled_bits += 1
            if filled_bits == 8:
                payload.append(current_byte)
                current_byte = 0
                filled_bits = 0
    if filled_bits:
        payload.append(current_byte << (8 - filled_bits))
    return bytes(payload), bit_count


def encode_bf16_bits(raw: bytes, shape: tuple[int, ...]) -> EncodedBFloat16:
    """Encode little-endian BF16 words without interpreting float values."""

    if len(raw) % 2:
        raise ConfigurationError("BF16 input must contain an even number of bytes")
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ConfigurationError("shape dimensions must be positive")
    element_count = len(raw) // 2
    if prod(shape) != element_count:
        raise ConfigurationError("shape does not match element count")

    sign_mantissa = bytearray(element_count)
    exponents: list[int] = []
    for index in range(element_count):
        word = raw[2 * index] | (raw[2 * index + 1] << 8)
        mantissa = word & 0x7F
        sign = (word >> 15) & 1
        sign_mantissa[index] = mantissa | (sign << 7)
        exponents.append((word >> 7) & 0xFF)

    lengths = _code_lengths(exponents)
    codes = _canonical_codes(lengths)
    payload = bytearray()
    offsets: list[int] = []
    bit_lengths: list[int] = []
    for start in range(0, element_count, CHUNK_ELEMENTS):
        offsets.append(len(payload))
        chunk, chunk_bits = _encode_chunk(
            exponents[start : start + CHUNK_ELEMENTS], codes
        )
        payload.extend(chunk)
        bit_lengths.append(chunk_bits)

    return EncodedBFloat16(
        shape=shape,
        element_count=element_count,
        bit_count=sum(bit_lengths),
        chunk_elements=CHUNK_ELEMENTS,
        chunk_byte_offsets=tuple(offsets),
        chunk_bit_lengths=tuple(bit_lengths),
        sign_mantissa=bytes(sign_mantissa),
        exponent_payload=bytes(payload),
        code_lengths=lengths,
    )


def _decode_chunk(
    payload: bytes,
    bit_length: int,
    expected_symbols: int,
    reverse_codes: dict[tuple[int, int], int],
    maximum_length: int,
) -> list[int]:
    symbols: list[int] = []
    code = 0
    length = 0
    for bit_index in range(bit_length):
        byte = payload[bit_index // 8]
        bit = (byte >> (7 - (bit_index % 8))) & 1
        code = (code << 1) | bit
        length += 1
        symbol = reverse_codes.get((length, code))
        if symbol is None:
            if length > maximum_length:
                raise IntegrityError("invalid exponent Huffman code")
            continue
        symbols.append(symbol)
        code = 0
        length = 0
    if length != 0:
        raise IntegrityError("truncated exponent Huffman code")
    if len(symbols) != expected_symbols:
        raise IntegrityError(
            "decoded exponent count mismatch: "
            f"expected {expected_symbols}, got {len(symbols)}"
        )
    return symbols


def decode_bf16_bits(encoded: EncodedBFloat16) -> bytes:
    """Decode BF16 words and reject inconsistent or truncated metadata."""

    if not encoded.shape or prod(encoded.shape) != encoded.element_count:
        raise IntegrityError("encoded shape does not match element count")
    if len(encoded.sign_mantissa) != encoded.element_count:
        raise IntegrityError("sign/mantissa byte count mismatch")
    if encoded.chunk_elements <= 0:
        raise IntegrityError("chunk_elements must be positive")
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

    codes = _canonical_codes(encoded.code_lengths)
    reverse_codes = {
        (length, code): symbol for symbol, (code, length) in codes.items()
    }
    maximum_length = max(encoded.code_lengths)
    exponents: list[int] = []
    expected_offset = 0
    for chunk_index, (offset, bit_length) in enumerate(
        zip(
            encoded.chunk_byte_offsets,
            encoded.chunk_bit_lengths,
            strict=True,
        )
    ):
        if offset != expected_offset:
            raise IntegrityError("chunk byte offsets are not contiguous")
        byte_length = (bit_length + 7) // 8
        end = offset + byte_length
        if end > len(encoded.exponent_payload):
            raise IntegrityError("truncated exponent payload")
        expected_symbols = min(
            encoded.chunk_elements,
            encoded.element_count - chunk_index * encoded.chunk_elements,
        )
        exponents.extend(
            _decode_chunk(
                encoded.exponent_payload[offset:end],
                bit_length,
                expected_symbols,
                reverse_codes,
                maximum_length,
            )
        )
        expected_offset = end
    if expected_offset != len(encoded.exponent_payload):
        raise IntegrityError("trailing exponent payload")

    raw = bytearray(encoded.element_count * 2)
    for index, (sign_mantissa, exponent) in enumerate(
        zip(encoded.sign_mantissa, exponents, strict=True)
    ):
        mantissa = sign_mantissa & 0x7F
        sign = (sign_mantissa & 0x80) << 8
        word = sign | (exponent << 7) | mantissa
        raw[2 * index] = word & 0xFF
        raw[2 * index + 1] = word >> 8
    return bytes(raw)
