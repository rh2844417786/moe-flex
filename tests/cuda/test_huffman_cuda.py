from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from flexmoe.codec.cuda import cuda_decode
from flexmoe.codec.reference import encode_bf16_bits
from flexmoe.errors import IntegrityError

pytestmark = pytest.mark.cuda


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def test_cuda_decode_matches_reference_for_all_bf16_patterns() -> None:
    _require_cuda()
    raw = np.arange(65_536, dtype="<u2").tobytes()
    encoded = encode_bf16_bits(raw, (65_536,))

    decoded = cuda_decode(encoded, device=0)

    assert decoded.view(torch.int16).cpu().numpy().tobytes() == raw


def test_cuda_decode_writes_exact_mapped_destination() -> None:
    _require_cuda()
    raw = np.random.default_rng(9).integers(
        0, 65_536, 8192, dtype=np.uint16
    ).astype("<u2", copy=False).tobytes()
    encoded = encode_bf16_bits(raw, (128, 64))
    destination = torch.empty((128, 64), dtype=torch.bfloat16, device="cuda:0")

    decoded = cuda_decode(encoded, device=0, destination=destination)

    assert decoded.data_ptr() == destination.data_ptr()
    assert decoded.view(torch.int16).cpu().numpy().tobytes() == raw


def test_cuda_decode_rejects_inconsistent_code_table() -> None:
    _require_cuda()
    raw = np.arange(4096, dtype="<u2").tobytes()
    encoded = encode_bf16_bits(raw, (4096,))
    invalid = replace(encoded, code_lengths=(0,) * 256)

    with pytest.raises(IntegrityError, match="Huffman symbol"):
        cuda_decode(invalid, device=0)
