from __future__ import annotations

import numpy as np

from flexmoe.codec.cuda import build_decode_trie
from flexmoe.codec.reference import encode_bf16_bits


def test_decode_trie_matches_every_encoded_exponent() -> None:
    words = np.arange(65_536, dtype="<u2")
    encoded = encode_bf16_bits(words.tobytes(), (65_536,))
    left, right, symbols = build_decode_trie(encoded.code_lengths)
    decoded: list[int] = []

    for offset, bit_length in zip(
        encoded.chunk_byte_offsets,
        encoded.chunk_bit_lengths,
        strict=True,
    ):
        node = 0
        for bit_idx in range(bit_length):
            byte = encoded.exponent_payload[offset + bit_idx // 8]
            bit = (byte >> (7 - bit_idx % 8)) & 1
            node = int(right[node] if bit else left[node])
            assert node >= 0
            symbol = int(symbols[node])
            if symbol >= 0:
                decoded.append(symbol)
                node = 0
        assert node == 0

    expected = ((words >> 7) & 0xFF).astype(np.uint8).tolist()
    assert decoded == expected
    assert len(symbols) <= 511
