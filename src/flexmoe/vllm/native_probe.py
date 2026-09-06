"""Root-forward CPU metadata counters; never inspect or synchronize GPU tensors."""

from __future__ import annotations

from typing import Any


class NativeProbe:
    def __init__(self, runner: Any, rank: int) -> None:
        self.runner = runner
        self.rank = rank
        self.handle: Any = None
        self.calls = 0
        self.observations = 0
        self.request_sum = 0
        self.peak = 0
        self.requests_available = True
        self.parameter_count = 0
        self.parameters_cuda = False
        self.forward_available = False

    def _forward(self, module: Any, args: Any) -> None:
        self.calls += 1
        value = getattr(getattr(self.runner, "input_batch", None), "num_reqs", None)
        if type(value) is not int or value <= 0:
            self.requests_available = False
            return
        self.observations += 1
        self.request_sum += value
        self.peak = max(self.peak, value)

    def start(self) -> dict[str, Any]:
        if self.handle is not None:
            raise RuntimeError("native probe already started")
        self.calls = self.observations = self.request_sum = self.peak = 0
        self.requests_available = True
        model = getattr(self.runner, "model", None)
        self.forward_available = callable(
            getattr(model, "register_forward_pre_hook", None)
        )
        parameters = list(model.parameters()) if model is not None else []
        self.parameter_count = len(parameters)
        self.parameters_cuda = bool(parameters) and all(p.is_cuda for p in parameters)
        if self.forward_available and model is not None:
            self.handle = model.register_forward_pre_hook(self._forward)
        return self.snapshot()

    def stop(self) -> dict[str, Any]:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        available = self.requests_available and self.observations > 0
        return {
            "rank": self.rank,
            "forward_telemetry_available": self.forward_available,
            "forward_calls": self.calls if self.forward_available else None,
            "request_telemetry_available": available,
            "request_observations": self.observations,
            "request_sum": self.request_sum if available else None,
            "request_mean": self.request_sum / self.observations if available else None,
            "request_peak": self.peak if available else None,
            "parameter_count": self.parameter_count,
            "all_parameters_cuda": self.parameters_cuda,
            "kv_occupancy": None,
            "preemptions": None,
        }
