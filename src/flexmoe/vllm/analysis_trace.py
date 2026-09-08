"""Bounded eager-only expert occupancy capture. No route D2H in the hot path."""

from __future__ import annotations

import os
from typing import Any

import torch

from flexmoe.analysis.schema import TraceEvent


def positive(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def device_record(rank: int) -> dict[str, Any]:
    """Query only inside a worker that already owns its CUDA context."""
    try:
        prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        return {
            "rank": rank,
            "uuid": getattr(prop, "uuid", None),
            "total_memory": prop.total_memory,
        }
    except (RuntimeError, AssertionError, AttributeError):
        return {"rank": rank, "uuid": None, "total_memory": None}


def step_metadata(runner: Any) -> tuple[int | None, str, int | None]:
    """Read only pinned runner CPU metadata; missing evidence stays unknown."""
    batch = getattr(runner, "input_batch", None)
    requests = getattr(batch, "num_reqs", None)
    if batch is None or type(requests) is not int or requests <= 0:
        return None, "unknown", None
    phase = "unknown"
    try:
        prompt = batch.num_prompt_tokens
        computed = batch.num_computed_tokens_cpu
        # Never accidentally consume a CUDA tensor supplied by another runner.
        if isinstance(prompt, torch.Tensor) or isinstance(computed, torch.Tensor):
            raise TypeError("expected CPU arrays")
        pending = [int(computed[i]) < int(prompt[i]) for i in range(requests)]
        phase = "prefill" if all(pending) else "mixed" if any(pending) else "decode"
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    rows = None
    try:
        starts = runner.query_start_loc.np
        if isinstance(starts, torch.Tensor):
            raise TypeError("expected CPU array")
        value = int(starts[requests])
        if value >= requests:
            rows = value
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    return requests, phase, rows


class AnalysisTraceCollector:
    def __init__(
        self,
        model_runner: Any,
        *,
        total_layers: int,
        num_experts: int,
        top_k: int,
        device: str | torch.device = "cuda",
        max_trace_steps: int = 1024,
        trace_budget_bytes: int = 134217728,
        safety_reserve_bytes: int = 2_000_000_000,
    ) -> None:
        self.runner = model_runner
        self.total_layers = positive(total_layers, "total_layers")
        self.num_experts = positive(num_experts, "num_experts")
        self.top_k = positive(top_k, "top_k")
        if self.top_k > self.num_experts:
            raise ValueError("top_k exceeds experts")
        self.max_steps = positive(max_trace_steps, "max_trace_steps")
        self.budget = positive(trace_budget_bytes, "trace budget")
        self.reserve = positive(safety_reserve_bytes, "safety reserve")
        self.device = torch.device(device)
        self.buffer_bytes = self.max_steps * self.total_layers * self.num_experts
        self.buffer: torch.Tensor | None = None
        self.hook: Any = None
        self.observed = 0
        self.metadata: dict[tuple[int, int], tuple[int, int | None, str]] = {}
        self.current: tuple[int | None, str, int | None] = (None, "unknown", None)
        self.last: dict[str, Any] | None = None

    def start(self) -> dict[str, Any]:
        if self.hook is not None:
            return self.status()
        if self.buffer_bytes > self.budget:
            raise ValueError("trace buffer exceeds trace budget")
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            if self.buffer_bytes + self.reserve > free:
                raise RuntimeError("trace buffer violates physical safety reserve")
        model = getattr(self.runner, "model", None)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("root model module unavailable for eager capture")
        self.buffer = torch.zeros(
            (self.max_steps, self.total_layers, self.num_experts),
            dtype=torch.bool,
            device=self.device,
        )
        self.metadata = {}
        self.observed = 0
        self.last = None
        self.hook = model.register_forward_pre_hook(self._root)
        return self.status()

    def _root(self, module: Any, args: Any) -> None:
        self.observed += 1
        self.current = step_metadata(self.runner)

    def record(self, layer: int, topk_ids: torch.Tensor) -> None:
        if self.hook is None or self.observed == 0 or self.observed > self.max_steps:
            return
        if (
            type(layer) is not int
            or not 0 <= layer < self.total_layers
            or topk_ids.ndim != 2
            or topk_ids.shape[1] != self.top_k
            or topk_ids.dtype not in (torch.int32, torch.int64)
            or topk_ids.device.type != self.device.type
        ):
            raise ValueError("invalid route layer, shape, dtype, or device")
        requests, phase, valid_rows = self.current
        rows = topk_ids.shape[0] if valid_rows is None else valid_rows
        if rows <= 0 or rows > topk_ids.shape[0]:
            raise ValueError("scheduled route rows exceed tensor shape")
        if requests is not None and requests > rows:
            requests, phase = None, "unknown"
        key = (self.observed - 1, layer)
        if key in self.metadata:
            raise RuntimeError("duplicate layer event in root step")
        self.metadata[key] = (rows, requests, phase)
        assert self.buffer is not None
        # Idempotent occupancy update: no unique(), item(), cpu(), or synchronize().
        self.buffer[key].index_fill_(0, topk_ids[:rows].reshape(-1).long(), True)

    def status(self) -> dict[str, Any]:
        return {
            "active": self.hook is not None,
            "observed_steps": self.observed,
            "buffer_bytes": self.buffer_bytes,
            "max_trace_steps": self.max_steps,
            "trace_budget_bytes": self.budget,
            "safety_reserve_bytes": self.reserve,
        }

    def stop(self) -> dict[str, Any]:
        if self.hook is None:
            return self.last or {**self.status(), "status": "empty", "events": []}
        self.hook.remove()
        self.hook = None
        assert self.buffer is not None
        try:
            # One bounded D2H, after the controller's end-of-generation sync.
            occupancy = self.buffer.cpu()
            bounded = min(self.observed, self.max_steps)
            complete = 0
            while complete < bounded and all(
                (complete, layer) in self.metadata for layer in range(self.total_layers)
            ):
                complete += 1
            missing = bounded * self.total_layers - len(self.metadata)
            events = []
            for step in range(complete):
                for layer in range(self.total_layers):
                    rows, requests, phase = self.metadata[step, layer]
                    experts = tuple(occupancy[step, layer].nonzero().flatten().tolist())
                    events.append(
                        TraceEvent(
                            step, layer, rows, requests, phase, experts
                        ).to_dict()
                    )
            full = complete == self.observed and complete > 0
            self.last = {
                **self.status(),
                "status": "complete" if full else "incomplete",
                "captured_steps": complete,
                "truncated": self.observed > self.max_steps,
                "missing_layer_events": missing,
                "full_workload": full,
                "events": events,
                "discarded_present_events": len(self.metadata) - len(events),
            }
            return self.last
        finally:
            self.buffer = None


_COLLECTOR: AnalysisTraceCollector | None = None


def trace_rpc(
    action: str, rank: int, model_runner: Any, **limits: Any
) -> dict[str, Any]:
    global _COLLECTOR
    if os.environ.get("FLUXMOE_ANALYSIS_TRACE") != "1":
        raise RuntimeError("explicit FLUXMOE_ANALYSIS_TRACE=1 is required")
    if action not in ("start", "stop", "status"):
        raise ValueError("trace action must be start, stop, or status")
    if action == "start" and (_COLLECTOR is None or _COLLECTOR.hook is None):
        _COLLECTOR = AnalysisTraceCollector(model_runner, **limits)
    if _COLLECTOR is None:
        return {"rank": rank, "active": False, "status": "empty"}
    result = getattr(_COLLECTOR, action)()
    return {"rank": rank, **result}


def record_trace(layer_name: str, topk_ids: torch.Tensor) -> None:
    if _COLLECTOR is not None:
        from flexmoe.vllm.partial import _layer_index

        _COLLECTOR.record(_layer_index(layer_name), topk_ids)
