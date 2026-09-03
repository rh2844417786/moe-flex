from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from flexmoe.codec.packed import pack_layer_descriptor
from flexmoe.codec.reference import decode_bf16_bits, encode_bf16_bits
from flexmoe.errors import IntegrityError


def _encoded(seed: int, elements: int = 5000):
    raw = (
        np.random.default_rng(seed)
        .integers(0, 65_536, elements, dtype=np.uint16)
        .astype("<u2", copy=False)
        .tobytes()
    )
    return raw, encode_bf16_bits(raw, (elements,))


def test_pack_layer_descriptor_preserves_expert_payload_and_layout() -> None:
    raw0, encoded0 = _encoded(1)
    raw1, encoded1 = _encoded(2)

    packed = pack_layer_descriptor(
        {0: encoded0, 1: encoded1}, destination_shape=(2, 5000)
    )

    assert packed.shape == (2, 5000)
    assert packed.expert_count == 2
    assert packed.sign_mantissa == encoded0.sign_mantissa + encoded1.sign_mantissa
    assert packed.exponent_payload == (
        encoded0.exponent_payload + encoded1.exponent_payload
    )
    assert packed.chunk_destination_offsets == (0, 4096, 5000, 9096)
    assert packed.chunk_element_counts == (4096, 904, 4096, 904)
    assert packed.chunk_expert_indices == (0, 0, 1, 1)
    assert packed.chunk_byte_offsets[2] == len(encoded0.exponent_payload)
    assert packed.source_bytes == len(raw0) + len(raw1)
    assert packed.descriptor_bytes == (
        packed.chunk_count * 5 * 8
        + len(packed.segment_bit_offsets)
        + packed.expert_count * 2 * 8
    )
    assert packed.gpu_storage_bytes == (
        packed.encoded_payload_bytes
        + packed.descriptor_bytes
        + packed.codebook_bytes
        + packed.error_buffer_bytes
    )


def test_pack_layer_descriptor_rejects_missing_or_mismatched_experts() -> None:
    _, encoded = _encoded(3)

    with pytest.raises(IntegrityError, match="contiguous"):
        pack_layer_descriptor({1: encoded}, destination_shape=(2, 5000))
    with pytest.raises(IntegrityError, match="shape"):
        pack_layer_descriptor({0: encoded}, destination_shape=(1, 4999))


def test_pack_layer_descriptor_rejects_payload_gaps_and_truncation() -> None:
    _, encoded = _encoded(4)
    gapped = replace(
        encoded,
        chunk_byte_offsets=(0, encoded.chunk_byte_offsets[1] + 1),
    )
    truncated = replace(encoded, exponent_payload=encoded.exponent_payload[:-1])

    with pytest.raises(IntegrityError, match="not contiguous"):
        pack_layer_descriptor({0: gapped}, destination_shape=(1, 5000))
    with pytest.raises(IntegrityError, match="payload size"):
        pack_layer_descriptor({0: truncated}, destination_shape=(1, 5000))


@pytest.mark.parametrize("elements", [1, 4095, 4096, 4097, 20_000])
def test_vectorized_encoder_round_trips_all_chunk_boundaries(elements: int) -> None:
    raw, encoded = _encoded(17, elements)

    assert decode_bf16_bits(encoded) == raw
