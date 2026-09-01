from __future__ import annotations

import json
from pathlib import Path

import pytest

from flexmoe.runtime.telemetry import JsonlTelemetry


def test_jsonl_telemetry_flushes_sorted_schema_events(tmp_path: Path) -> None:
    output = tmp_path / "events.jsonl"
    telemetry = JsonlTelemetry(
        output,
        run_id="run-1",
        git_sha="a" * 40,
        rank=2,
        variant="fluxmoe-fixed",
        buffer_events=2,
    )

    telemetry.emit("mapping", {"mapped_bytes": 4096, "layer": 3})
    telemetry.emit("transfer", {"h2d_bytes": 2048, "layer": 3})
    telemetry.close()

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0] == json.dumps(
        json.loads(lines[0]), sort_keys=True, separators=(",", ":")
    )
    first = json.loads(lines[0])
    assert first["schema_version"] == 1
    assert first["rank"] == 2
    assert first["kind"] == "mapping"
    assert first["payload"]["run_id"] == "run-1"
    assert first["payload"]["git_sha"] == "a" * 40


def test_telemetry_rejects_emit_after_close(tmp_path: Path) -> None:
    telemetry = JsonlTelemetry(
        tmp_path / "events.jsonl",
        run_id="run-1",
        git_sha="b" * 40,
        rank=0,
        variant="resident",
    )
    telemetry.close()

    with pytest.raises(RuntimeError, match="closed"):
        telemetry.emit("mapping", {"mapped_bytes": 1})


def test_telemetry_does_not_swallow_open_errors(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        JsonlTelemetry(
            tmp_path,
            run_id="run-1",
            git_sha="c" * 40,
            rank=0,
            variant="resident",
        )
