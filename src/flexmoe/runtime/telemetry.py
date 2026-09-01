"""Versioned, fail-closed JSONL telemetry for FluxMoE runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import monotonic_ns

PayloadValue = int | float | str | bool


@dataclass(frozen=True)
class Event:
    schema_version: int
    monotonic_ns: int
    rank: int
    kind: str
    payload: dict[str, PayloadValue]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported telemetry schema_version")
        if self.monotonic_ns <= 0:
            raise ValueError("monotonic_ns must be positive")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        if not self.kind:
            raise ValueError("event kind must not be empty")
        if not all(isinstance(key, str) and key for key in self.payload):
            raise TypeError("event payload keys must be non-empty strings")
        if not all(
            isinstance(value, (int, float, str, bool))
            for value in self.payload.values()
        ):
            raise TypeError("event payload values must be JSON scalars")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "monotonic_ns": self.monotonic_ns,
            "rank": self.rank,
            "kind": self.kind,
            "payload": self.payload,
        }


class JsonlTelemetry:
    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        git_sha: str,
        rank: int,
        variant: str,
        buffer_events: int = 256,
    ) -> None:
        if not run_id or not variant:
            raise ValueError("run_id and variant must not be empty")
        if len(git_sha) != 40:
            raise ValueError("git_sha must contain 40 characters")
        if rank < 0:
            raise ValueError("rank must be non-negative")
        if buffer_events <= 0:
            raise ValueError("buffer_events must be positive")
        self._path = path
        self._rank = rank
        self._base_payload: dict[str, PayloadValue] = {
            "run_id": run_id,
            "git_sha": git_sha,
            "variant": variant,
        }
        self._buffer_events = buffer_events
        self._buffer: list[str] = []
        self._lock = Lock()
        self._closed = False
        self._stream = path.open("x", encoding="utf-8", newline="\n")

    @property
    def path(self) -> Path:
        return self._path

    def emit(self, kind: str, payload: Mapping[str, PayloadValue]) -> Event:
        if set(payload) & set(self._base_payload):
            raise ValueError("event payload may not override run metadata")
        merged = {**self._base_payload, **dict(payload)}
        event = Event(
            schema_version=1,
            monotonic_ns=monotonic_ns(),
            rank=self._rank,
            kind=kind,
            payload=merged,
        )
        encoded = json.dumps(
            event.as_dict(), sort_keys=True, separators=(",", ":")
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("telemetry writer is closed")
            self._buffer.append(encoded)
            if len(self._buffer) >= self._buffer_events:
                self._flush_locked()
        return event

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        self._stream.write("\n".join(self._buffer))
        self._stream.write("\n")
        self._stream.flush()
        self._buffer.clear()

    def flush(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("telemetry writer is closed")
            self._flush_locked()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._flush_locked()
            self._stream.close()
            self._closed = True


__all__ = ["Event", "JsonlTelemetry", "PayloadValue"]
