from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from flexmoe.bench.router_trace import router_probes_match
from flexmoe.vllm.bridge import record_router_ids, reset_router_trace_state


def test_router_trace_records_exact_expert_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLUXMOE_ROUTER_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "3")
    reset_router_trace_state()
    ids = torch.tensor([[1, 7], [2, 6]], dtype=torch.int32)
    permuted = torch.tensor([[7, 1], [6, 2]], dtype=torch.int32)
    changed = torch.tensor([[1, 7], [2, 5]], dtype=torch.int32)

    record_router_ids("model.layers.0.mlp.experts", ids)
    record_router_ids("model.layers.1.mlp.experts", permuted)
    record_router_ids("model.layers.2.mlp.experts", changed)

    lines = (tmp_path / "rank-3.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [record["sequence"] for record in records] == [0, 1, 2]
    assert records[0]["sha256"] == records[1]["sha256"]
    assert records[0]["sha256"] != records[2]["sha256"]
    assert records[0]["shape"] == [2, 2]


def _manifest(*, full_sha: str = "a", probe_sha: str = "b") -> dict[str, object]:
    return {
        "rank-0.jsonl": {
            "sha256": full_sha * 64,
            "line_count": 1296,
            "probe_sha256": probe_sha * 64,
            "probe_line_count": 96,
        }
    }


def test_router_probe_match_allows_different_full_trace() -> None:
    assert router_probes_match(_manifest(full_sha="a"), _manifest(full_sha="c"))


def test_router_probe_match_rejects_different_probe() -> None:
    assert not router_probes_match(_manifest(probe_sha="a"), _manifest(probe_sha="b"))


@pytest.mark.parametrize(
    "invalid",
    [
        {},
        {"rank-1.jsonl": _manifest()["rank-0.jsonl"]},
        {"rank-0.jsonl": {}},
        {
            "rank-0.jsonl": {
                "sha256": "a" * 64,
                "line_count": 95,
                "probe_sha256": "b" * 64,
                "probe_line_count": 96,
            }
        },
        {
            "rank-0.jsonl": {
                "sha256": "not-a-sha",
                "line_count": 1296,
                "probe_sha256": "b" * 64,
                "probe_line_count": 96,
            }
        },
    ],
)
def test_router_probe_match_fails_closed(invalid: dict[str, object]) -> None:
    assert not router_probes_match(_manifest(), invalid)
    assert not router_probes_match(invalid, _manifest())
