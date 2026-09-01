from dataclasses import replace

import numpy as np
import pytest

from flexmoe.codec.reference import decode_bf16_bits, encode_bf16_bits
from flexmoe.errors import ConfigurationError, IntegrityError


def test_bf16_round_trip_preserves_every_bit_pattern() -> None:
    raw = np.arange(0, 65536, dtype="<u2").tobytes()

    encoded = encode_bf16_bits(raw, shape=(65536,))

    assert decode_bf16_bits(encoded) == raw


def test_codec_is_deterministic() -> None:
    raw = np.random.default_rng(7).integers(
        0, 65536, 10_000, dtype=np.uint16
    ).astype("<u2", copy=False).tobytes()

    assert encode_bf16_bits(raw, (10_000,)) == encode_bf16_bits(raw, (10_000,))


def test_single_exponent_uses_one_bit_code() -> None:
    raw = np.full(5000, 0x3F80, dtype="<u2").tobytes()

    encoded = encode_bf16_bits(raw, (5000,))

    assert encoded.code_lengths[0x7F] == 1
    assert sum(length > 0 for length in encoded.code_lengths) == 1
    assert decode_bf16_bits(encoded) == raw


def test_decoder_rejects_truncated_payload() -> None:
    raw = np.arange(4096, dtype="<u2").tobytes()
    encoded = encode_bf16_bits(raw, (4096,))
    truncated = replace(encoded, exponent_payload=encoded.exponent_payload[:-1])

    with pytest.raises(IntegrityError, match="truncated exponent payload"):
        decode_bf16_bits(truncated)


@pytest.mark.parametrize(
    ("raw", "shape", "message"),
    [
        (b"\x00", (1,), "even number of bytes"),
        (b"\x00\x00", (2,), "shape does not match element count"),
        (b"", (), "shape dimensions must be positive"),
    ],
)
def test_encoder_rejects_invalid_raw_shape(
    raw: bytes, shape: tuple[int, ...], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        encode_bf16_bits(raw, shape)
