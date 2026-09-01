from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from flexmoe.vllm.bridge import record_router_ids, reset_router_trace_state


def test_router_trace_records_ordered_exact_id_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLUXMOE_ROUTER_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "3")
    reset_router_trace_state()
    ids = torch.tensor([[1, 7], [2, 6]], dtype=torch.int32)

    record_router_ids("model.layers.0.mlp.experts", ids)
    record_router_ids("model.layers.1.mlp.experts", ids)

    lines = (tmp_path / "rank-3.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [record["sequence"] for record in records] == [0, 1]
    assert records[0]["sha256"] == records[1]["sha256"]
    assert records[0]["shape"] == [2, 2]
