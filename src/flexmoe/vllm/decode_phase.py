"""Bounded per-rank wall-clock phase evidence for offline decode runs."""

from __future__ import annotations

import math
from collections.abc import Callable
from time import perf_counter_ns
from typing import Any


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class PhaseTimeline:
    """Record prepared-input to execute-return phases on one worker clock."""

    def __init__(self, clock_ns: Callable[[], int] = perf_counter_ns) -> None:
        self._clock = clock_ns
        self._start_ns: int | None = None
        self._stop_ns: int | None = None
        self._active: tuple[int, str, int | None, int] | None = None
        self._first_decode_ns: int | None = None
        self._last_decode_end_ns: int | None = None
        self._decode_step_ms: list[float] = []
        self._sequence: list[str] = []
        self._step_count = 0

    def _now(self) -> int:
        value = self._clock()
        if type(value) is not int or value < 0:
            raise ValueError("phase clock must return nonnegative integer nanoseconds")
        return value

    def start(self) -> None:
        if self._start_ns is not None:
            raise RuntimeError("phase timeline already started")
        self._start_ns = self._now()

    def record(self, *, step: int, phase: str, actual_batch: int | None) -> None:
        if self._start_ns is None or self._stop_ns is not None:
            raise RuntimeError("phase timeline is not active")
        if self._active is not None:
            raise RuntimeError("previous phase step has not finished")
        if type(step) is not int or step < 0:
            raise ValueError("phase step must be a nonnegative integer")
        if phase not in ("prefill", "decode", "unknown"):
            raise ValueError("phase must be prefill, decode, or unknown")
        if actual_batch is not None and (
            type(actual_batch) is not int or actual_batch <= 0
        ):
            raise ValueError("actual_batch must be a positive integer when present")
        now = self._now()
        if not self._sequence or self._sequence[-1] != phase:
            self._sequence.append(phase)
        if phase == "decode" and self._first_decode_ns is None:
            self._first_decode_ns = now
        self._active = (step, phase, actual_batch, now)

    def finish(self, *, step: int) -> None:
        if self._active is None:
            raise RuntimeError("no active phase step")
        active_step, phase, _, started = self._active
        if step != active_step:
            raise ValueError("phase finish step differs from active step")
        ended = self._now()
        if ended < started:
            raise RuntimeError("phase clock moved backwards")
        if phase == "decode":
            self._last_decode_end_ns = ended
            self._decode_step_ms.append((ended - started) / 1_000_000)
        self._step_count += 1
        self._active = None

    def stop(self) -> dict[str, Any]:
        if self._start_ns is None or self._stop_ns is not None:
            raise RuntimeError("phase timeline is not active")
        if self._active is not None:
            self.finish(step=self._active[0])
        self._stop_ns = self._now()
        complete = (
            self._first_decode_ns is not None
            and self._last_decode_end_ns is not None
            and self._last_decode_end_ns >= self._first_decode_ns
        )
        return {
            "status": "measured" if complete else "incomplete-phase-boundaries",
            "prefill_wall_time_s": (
                (self._first_decode_ns - self._start_ns) / 1_000_000_000
                if complete and self._first_decode_ns is not None
                else None
            ),
            "decode_wall_time_s": (
                (self._last_decode_end_ns - self._first_decode_ns) / 1_000_000_000
                if complete
                and self._last_decode_end_ns is not None
                and self._first_decode_ns is not None
                else None
            ),
            "decode_step_ms_p50": _percentile(self._decode_step_ms, 0.50),
            "decode_step_ms_p95": _percentile(self._decode_step_ms, 0.95),
            "decode_step_samples": len(self._decode_step_ms),
            "observed_step_count": self._step_count,
            "phase_sequence": list(self._sequence),
            "scope": "per-rank monotonic worker wall; never summed across TP ranks",
            "boundaries": {
                "measurement_start_ns": self._start_ns,
                "first_decode_start_ns": self._first_decode_ns,
                "last_decode_end_ns": self._last_decode_end_ns,
                "measurement_stop_ns": self._stop_ns,
            },
        }


__all__ = ["PhaseTimeline"]
