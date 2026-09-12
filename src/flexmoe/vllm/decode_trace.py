"""Explicit bounded eager histograms and Graph-safe prepared-input observations."""

from __future__ import annotations

import math
import os
from collections import Counter
from typing import Any, cast

import torch

from flexmoe.analysis.decode_counts import (
    summarize_activations,
    validate_activation_row,
)
from flexmoe.vllm.analysis_trace import positive, step_metadata


class DecodeTraceCollector:
    def __init__(
        self,
        model_runner: Any,
        *,
        total_layers: int,
        num_experts: int,
        top_k: int,
        profile: bool = False,
        target_batch: int | None = None,
        capture_steps: int = 256,
        min_capture_steps: int = 64,
        trace_budget_bytes: int = 134217728,
        max_batch: int = 1024,
        safety_reserve_bytes: int = 2_000_000_000,
        device: str | torch.device = "cuda",
    ) -> None:
        self.runner = model_runner
        self.layers = positive(total_layers, "total_layers")
        self.experts = positive(num_experts, "num_experts")
        self.top_k = positive(top_k, "top_k")
        self.max_batch = positive(max_batch, "max_batch")
        self.capacity = positive(capture_steps, "capture_steps")
        self.minimum = positive(min_capture_steps, "min_capture_steps")
        self.budget = positive(trace_budget_bytes, "trace_budget_bytes")
        self.reserve = positive(safety_reserve_bytes, "safety reserve")
        self.target = (
            None if target_batch is None else positive(target_batch, "target_batch")
        )
        if self.top_k > self.experts or self.minimum > self.capacity:
            raise ValueError("invalid top_k or capture window")
        self.profile = profile
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.buffer_bytes = (
            4 * self.capacity * self.layers * self.experts
            + 12 * self.max_batch * self.top_k
            if profile
            else 0
        )
        self.buffer: torch.Tensor | None = None
        self.ones: torch.Tensor | None = None
        self.indices: torch.Tensor | None = None
        self.active = False
        self.observed = 0
        self.phase_counts: Counter[str] = Counter()
        self.batch_counts: Counter[str] = Counter()
        self.skipped = dict.fromkeys(("non_decode", "non_target_batch", "capacity"), 0)
        self.selected: list[tuple[int, int]] = []
        self.present: set[tuple[int, int]] = set()
        self.current: dict[str, Any] | None = None
        self.original: Any = None
        self.original_execute: Any = None
        self.step_events: list[tuple[int, Any, Any]] = []
        self.last: dict[str, Any] | None = None
        self.pool: Any = None

    def start(self) -> dict[str, Any]:
        if self.active:
            return self.status()
        if not callable(getattr(self.runner, "_prepare_inputs", None)):
            raise TypeError("pinned prepared-input observer boundary unavailable")
        if self.buffer_bytes > self.budget:
            raise ValueError("trace buffer exceeds trace budget")
        if self.profile:
            if self.device.type == "cuda":
                free, _ = torch.cuda.mem_get_info(self.device)
                if self.buffer_bytes + self.reserve > free:
                    raise RuntimeError("trace violates physical safety reserve")
            self.buffer = torch.zeros(
                (self.capacity, self.layers, self.experts),
                dtype=torch.int32,
                device=self.device,
            )
            self.ones = torch.ones(
                self.max_batch * self.top_k, dtype=torch.int32, device=self.device
            )
            self.indices = torch.empty(
                (self.max_batch, self.top_k), dtype=torch.int64, device=self.device
            )
        self.original = self.runner._prepare_inputs

        def prepare(*args: Any, **kwargs: Any) -> Any:
            result = self.original(*args, **kwargs)
            self._prepared()
            return result

        self.runner._prepare_inputs = prepare
        if self.profile and self.device.type == "cuda":
            self.original_execute = self.runner.execute_model

            def execute(*args: Any, **kwargs: Any) -> Any:
                self.current = None
                result = self.original_execute(*args, **kwargs)
                if (
                    self.current is not None
                    and self.current.get("start_event") is not None
                ):
                    end = self._event()
                    self.step_events.append(
                        (self.current["step"], self.current["start_event"], end)
                    )
                return result

            self.runner.execute_model = execute
        self.active = True
        return self.status()

    def _event(self) -> Any:
        event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        event.record(torch.cuda.current_stream(self.device))  # type: ignore[no-untyped-call]
        return event

    def _prepared(self) -> None:
        batch, phase, rows = step_metadata(self.runner)
        step = self.observed
        self.observed += 1
        if (
            batch is None
            or batch > self.max_batch
            or (phase == "decode" and rows != batch)
        ):
            phase = "unknown"
        if phase == "decode":
            starts = self.runner.query_start_loc.np
            assert batch is not None
            if int(starts[0]) != 0 or any(
                int(starts[i + 1]) - int(starts[i]) != 1 for i in range(batch)
            ):
                phase = "unknown"
        self.phase_counts[phase] += 1
        if phase == "decode":
            self.batch_counts[str(batch)] += 1
        self.current = None
        if phase != "decode":
            self.skipped["non_decode"] += 1
        elif self.target is not None and batch != self.target:
            self.skipped["non_target_batch"] += 1
        elif self.profile:
            if len(self.selected) == self.capacity:
                self.skipped["capacity"] += 1
                return
            assert batch is not None
            self.current = {
                "step": step,
                "actual_batch": batch,
                "phase": phase,
                "slot": len(self.selected),
            }
            self.selected.append((step, batch))
            if self.device.type == "cuda":
                self.current["start_event"] = self._event()

    def record(self, layer: int, topk_ids: torch.Tensor) -> None:
        if not self.active or self.current is None:
            return
        batch = self.current["actual_batch"]
        if (
            type(layer) is not int
            or not 0 <= layer < self.layers
            or topk_ids.ndim != 2
            or topk_ids.shape[1] != self.top_k
            or topk_ids.shape[0] < batch
            or topk_ids.dtype not in (torch.int32, torch.int64)
            or topk_ids.device != self.device
        ):
            raise ValueError("invalid decode route layer, tensor, or scheduled rows")
        key = (self.current["slot"], layer)
        if key in self.present:
            raise ValueError("duplicate decode layer observation")
        assert (
            self.buffer is not None
            and self.ones is not None
            and self.indices is not None
        )
        # No unique/bincount shape synchronization, no route D2H. Padding excluded.
        self.indices[:batch].copy_(topk_ids[:batch])
        self.buffer[key].scatter_add_(
            0, self.indices[:batch].view(-1), self.ones[: batch * self.top_k]
        )
        self.present.add(key)

    def pool_context(self, layer: int) -> dict[str, Any] | None:
        if not self.active or self.current is None:
            return None
        return {key: self.current[key] for key in ("step", "actual_batch", "phase")}

    def status(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "profile": self.profile,
            "observed_steps": self.observed,
            "buffer_bytes": self.buffer_bytes,
            "trace_budget_bytes": self.budget,
            "capture_steps": self.capacity,
            "min_capture_steps": self.minimum,
            "target_batch": self.target,
            "metadata_source": "pinned-GPUModelRunner._prepare_inputs-after-original",
            "metadata_scope": "prepared-CPU-inputs-including-CUDA-Graph-replay",
            "trace_memory_scope": "fixed histogram/ones/int64-index CUDA payload; bounded event/Python metadata overhead included in physical measurements, not tensor byte estimate",
        }

    def restore(self) -> None:
        if not self.active:
            return
        self.runner._prepare_inputs = self.original
        if self.original_execute is not None:
            self.runner.execute_model = self.original_execute
        self.active = False

    def stop(self) -> dict[str, Any]:
        if not self.active:
            return self.last or {**self.status(), "activation_rows": []}
        self.restore()
        try:
            host = self.buffer.cpu() if self.buffer is not None else None
            rows = []
            complete_rows = []
            complete_steps = 0
            for slot, (step, batch) in enumerate(self.selected):
                full = all(
                    (slot, layer) in self.present for layer in range(self.layers)
                )
                complete_steps += int(full)
                for layer in range(self.layers):
                    if (slot, layer) not in self.present:
                        continue
                    assert host is not None
                    row = validate_activation_row(
                        {
                            "step": step,
                            "layer": layer,
                            "phase": "decode",
                            "actual_batch": batch,
                            "histogram": host[slot, layer].tolist(),
                        },
                        layers=self.layers,
                        experts=self.experts,
                        top_k=self.top_k,
                    )
                    rows.append(row)
                    if full:
                        complete_rows.append(row)
            missing = len(self.selected) * self.layers - len(rows)
            targets = (
                [self.target]
                if self.target is not None
                else sorted({cast(int, row["actual_batch"]) for row in complete_rows})
            )
            summaries = [
                summarize_activations(
                    complete_rows,
                    target_batch=batch,
                    layers=self.layers,
                    experts=self.experts,
                    top_k=self.top_k,
                    min_steps=self.minimum,
                )
                for batch in targets
            ]
            coverage = (
                "not-requested"
                if not self.profile
                else "incomplete-layers"
                if missing
                else "unreached"
                if not complete_steps
                else "complete"
                if complete_steps >= self.minimum
                else "insufficient-window"
            )
            if not self.profile and self.target is not None:
                matched = self.batch_counts[str(self.target)]
                coverage = (
                    "unreached"
                    if not matched
                    else "complete"
                    if matched >= self.minimum
                    else "insufficient-window"
                )
            spans = []
            for step, start, end in self.step_events:
                ready = end.query()
                spans.append(
                    {
                        "step": step,
                        "status": "measured" if ready else "unavailable",
                        "cuda_s": start.elapsed_time(end) / 1000 if ready else None,
                    }
                )
            self.last = {
                **self.status(),
                "coverage_status": coverage,
                "phase_step_counts": dict(self.phase_counts),
                "decode_batch_step_counts": dict(self.batch_counts),
                "skipped_steps": dict(self.skipped),
                "captured_steps": complete_steps,
                "selected_steps": len(self.selected),
                "missing_layer_events": missing,
                "activation_rows": rows,
                "activation_summaries": summaries,
                "model_step_spans": spans,
                "model_step_span_scope": "CUDA-prepared-inputs-through-execute_model-return; instrumented",
                "model_step_span_status": "measured" if spans else "unavailable",
                "timing_eligible": not self.profile,
            }
            return self.last
        finally:
            self.buffer = self.ones = self.indices = None
            self.current = None


class SchedulerObserver:
    """Reversible forwarding decorator; offline v1 rejects custom stat_loggers."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner
        self.original: Any = None
        self.active = False
        self.values: dict[str, list[float]] = {
            key: [0, 0.0, 0.0, 0]
            for key in (
                "kv_cache_usage",
                "running_requests",
                "waiting_requests",
                "preemptions",
            )
        }

    def start(self) -> None:
        self.original = getattr(self.owner, "stat_logger", None)
        if self.original is not None and callable(
            getattr(self.original, "record", None)
        ):
            self.owner.stat_logger = self
            self.active = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)

    def record(
        self, scheduler_stats: Any, iteration_stats: Any, engine_idx: int = 0
    ) -> Any:
        fields: dict[str, Any] = {
            "kv_cache_usage": getattr(scheduler_stats, "kv_cache_usage", None),
            "running_requests": getattr(scheduler_stats, "num_running_reqs", None),
            "waiting_requests": getattr(scheduler_stats, "num_waiting_reqs", None),
            "preemptions": getattr(iteration_stats, "num_preempted_reqs", None),
        }
        for key, value in fields.items():
            state = self.values[key]
            valid = (
                type(value) in (int, float)
                and math.isfinite(value)
                and value >= 0
                and (value <= 1 if key == "kv_cache_usage" else type(value) is int)
            )
            if valid:
                state[0] += 1
                state[1] += value
                state[2] = max(state[2], value)
            else:
                state[3] += 1
        return self.original.record(
            scheduler_stats=scheduler_stats,
            iteration_stats=iteration_stats,
            engine_idx=engine_idx,
        )

    def stop(self) -> dict[str, Any]:
        if self.active:
            self.owner.stat_logger = self.original
            self.active = False
        result: dict[str, Any] = {
            "source": "pinned-LLMEngine.stat_logger.record forwarding decorator",
            "scope": "frontend-engine-output-samples; not aligned one-to-one with worker steps",
            "invalid_samples": {key: int(v[3]) for key, v in self.values.items()},
        }
        for key, (count, total, peak, invalid) in self.values.items():
            status = (
                "measured"
                if count and not invalid
                else "partial"
                if count
                else "unavailable"
            )
            result[key] = {"status": status, "samples": int(count)}
            if not count:
                result[key]["unavailable_reason"] = (
                    "pinned-stat-logger-unavailable"
                    if self.original is None
                    else "no-valid-frontend-stat-samples-in-window"
                )
            if key == "preemptions":
                result[key]["total"] = int(total) if count else None
            else:
                result[key].update(
                    mean=total / count if count else None, peak=peak if count else None
                )
        return result


_COLLECTOR: DecodeTraceCollector | None = None


def decode_rpc(
    action: str, rank: int, model_runner: Any, **limits: Any
) -> dict[str, Any]:
    global _COLLECTOR
    if os.environ.get("FLUXMOE_DECODE_MECHANISM") != "1":
        raise RuntimeError("explicit FLUXMOE_DECODE_MECHANISM=1 required")
    if action not in ("start", "stop", "status"):
        raise ValueError("decode action must be start, stop, or status")
    if action == "start" and _COLLECTOR is not None and _COLLECTOR.active:
        return {"rank": rank, **_COLLECTOR.status()}
    if action == "start" and (_COLLECTOR is None or not _COLLECTOR.active):
        if (
            limits.get("profile")
            and getattr(
                getattr(model_runner, "model_config", None), "enforce_eager", None
            )
            is not True
        ):
            raise ValueError(
                "detailed decode profile requires explicit eager model execution"
            )
        _COLLECTOR = DecodeTraceCollector(model_runner, **limits)
    if _COLLECTOR is None:
        return {"rank": rank, "active": False, "status": "empty"}
    collector = _COLLECTOR
    if action == "start":
        result = collector.start()
        if (
            collector.profile
            and os.environ.get("FLUXMOE_ENABLE") == "1"
            and os.environ.get("FLUXMOE_STORAGE_MODE") == "expert-cache"
        ):
            from flexmoe.runtime.expert_pool import PoolProfileObserver
            from flexmoe.vllm.expert_cache import require_registry

            collector.pool = require_registry().pool
            collector.pool.profile_observer = PoolProfileObserver(
                collector.pool_context,
                capacity=collector.capacity * collector.layers,
                device=collector.device,
            )
    elif action == "stop":
        try:
            # End-of-window synchronization only; never a per-layer timing wait.
            if collector.device.type == "cuda":
                torch.cuda.synchronize(collector.device)
            result = collector.stop()
            if (
                collector.pool is not None
                and collector.pool.profile_observer is not None
            ):
                result["pool_profile"] = collector.pool.profile_observer.snapshot()
        finally:
            collector.restore()
            if collector.pool is not None:
                collector.pool.profile_observer = None
    else:
        result = collector.status()
    return {"rank": rank, **result}


def record_decode(layer_name: str, topk_ids: torch.Tensor) -> None:
    if _COLLECTOR is not None:
        from flexmoe.vllm.partial import _layer_index

        _COLLECTOR.record(_layer_index(layer_name), topk_ids)
