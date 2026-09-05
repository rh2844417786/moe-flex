"""A bounded, model-wide BF16 pool; logical routing is never modified."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from threading import Lock
from time import perf_counter
from typing import Protocol

import torch

from flexmoe.runtime.expert_cache_policy import ExpertCachePolicy
from flexmoe.runtime.partial_staging import should_sample_upload


class PoolBackend(Protocol):
    device: torch.device
    pin_memory: bool

    def begin(self) -> None: ...
    def wait_host(self) -> None: ...
    def uploaded(self) -> None: ...
    def end(self) -> None: ...
    def synchronize(self) -> None: ...
    def mark(self, stage: int) -> None: ...
    def timing(self, reset: bool) -> dict[str, int | float]: ...


class CudaPoolBackend:
    """One ingress, shared across streams with explicit producer dependencies."""

    pin_memory = True

    def __init__(self, device: int, *, timing_capacity: int = 128) -> None:
        if type(timing_capacity) is not int or timing_capacity < 0:
            raise ValueError("CUDA timing capacity must be a nonnegative integer")
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise ValueError("expert cache requires NVIDIA CUDA")
        self.device = torch.device("cuda", device)
        self.timing_capacity = timing_capacity
        self._done: torch.cuda.Event | None = None
        self._uploaded: torch.cuda.Event | None = None
        self._samples: deque[list[torch.cuda.Event]] = deque()
        self._sample: list[torch.cuda.Event] | None = None
        self._forward_serial = 0
        self._totals: dict[str, int | float] = {
            "cuda_sample_count": 0,
            "load_cuda_s": 0.0,
            "compute_cuda_s": 0.0,
            "promotion_cuda_s": 0.0,
        }

    def begin(self) -> None:
        self._collect()
        self._sample = (
            []
            if should_sample_upload(
                self._forward_serial,
                capacity=self.timing_capacity,
                pending_count=len(self._samples),
            )
            else None
        )
        self._forward_serial += 1
        if self._done is not None:
            torch.cuda.current_stream(self.device).wait_event(self._done)

    def wait_host(self) -> None:
        if self._uploaded is not None:
            self._uploaded.synchronize()

    def uploaded(self) -> None:
        if self._uploaded is None:
            self._uploaded = torch.cuda.Event()  # type: ignore[no-untyped-call]
        self._uploaded.record(torch.cuda.current_stream(self.device))  # type: ignore[no-untyped-call]

    def end(self) -> None:
        if self._done is None:
            self._done = torch.cuda.Event()  # type: ignore[no-untyped-call]
        self._done.record(torch.cuda.current_stream(self.device))  # type: ignore[no-untyped-call]

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)

    def mark(self, stage: int) -> None:
        if self._sample is None:
            return
        event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        event.record(torch.cuda.current_stream(self.device))  # type: ignore[no-untyped-call]
        self._sample.append(event)
        if stage == 3:
            self._samples.append(self._sample)
            self._sample = None

    def _collect(self) -> None:
        while self._samples and self._samples[0][-1].query():  # type: ignore[no-untyped-call]
            events = self._samples.popleft()
            self._totals["cuda_sample_count"] += 1
            for i, key in enumerate(
                ("load_cuda_s", "compute_cuda_s", "promotion_cuda_s")
            ):
                self._totals[key] += events[i].elapsed_time(events[i + 1]) / 1000  # type: ignore[no-untyped-call]

    def timing(self, reset: bool) -> dict[str, int | float]:
        self._collect()
        result = dict(self._totals)
        if reset:
            if self._samples:
                raise RuntimeError("synchronize before resetting CUDA timing")
            self._totals = dict.fromkeys(self._totals, 0)
            self._forward_serial = 0
        return result


class ExpertPool:
    def __init__(
        self,
        policy: ExpertCachePolicy,
        sources: Sequence[tuple[torch.Tensor, torch.Tensor]],
        *,
        backend: PoolBackend,
        load_residents: bool = True,
    ) -> None:
        self.policy = policy
        self.backend = backend
        self.sources = tuple(sources)
        self.total_layers = len(policy.resident_ids)
        if len(sources) != self.total_layers or not sources:
            raise ValueError("source layer count differs from policy")
        self.num_experts = sources[0][0].shape[0]
        shapes = tuple(tuple(t.shape) for t in sources[0])
        if any(len(s) != 3 or min(s) <= 0 for s in shapes) or (
            shapes[0] != (self.num_experts, 2 * shapes[1][2], shapes[1][1])
        ):
            raise ValueError("incompatible BF16 expert geometry")
        for pair in sources:
            if any(
                t.device.type != "cpu"
                or t.dtype != torch.bfloat16
                or tuple(t.shape) != shape
                or not t.is_contiguous()
                for t, shape in zip(pair, shapes, strict=True)
            ):
                raise ValueError("CPU source geometry/dtype differs")
        self.resident_slots = sum(map(len, policy.resident_ids))
        self.cache_slots = int(policy.stats()["cache_slots"])
        self.capacity = self.resident_slots + self.cache_slots + self.num_experts
        self.ingress_start = self.capacity - self.num_experts
        self.expert_bytes = sum(t[0].numel() * t.element_size() for t in sources[0])
        self.w13, self.w2 = (
            torch.empty(
                (self.capacity, *shape[1:]), device=backend.device, dtype=torch.bfloat16
            )
            for shape in shapes
        )
        self.gather13, self.gather2 = (
            torch.empty(
                shape, device="cpu", dtype=torch.bfloat16, pin_memory=backend.pin_memory
            )
            for shape in shapes
        )
        self.host_map = torch.empty(
            self.num_experts,
            device="cpu",
            dtype=torch.int32,
            pin_memory=backend.pin_memory,
        )
        self.expert_map = torch.empty_like(self.host_map, device=backend.device)
        self.host_indices = torch.empty(
            self.num_experts,
            device="cpu",
            dtype=torch.int64,
            pin_memory=backend.pin_memory,
        )
        self.device_indices = torch.empty_like(self.host_indices, device=backend.device)
        self._lock = Lock()
        self._active = False
        self._failed = False
        self._h2d_bytes = 0
        self._copy_launches = 0
        self._startup_bytes = 0
        self._promotion_bytes = 0
        self._metadata_h2d_bytes = 0
        self._forwards = [0] * self.total_layers
        self._resident_hits = 0
        self._unique_demands = 0
        self._max_unique = 0
        self._per_layer_unique_demands = [0] * self.total_layers
        self._per_layer_max_unique = [0] * self.total_layers
        self._timing = dict.fromkeys(
            (
                "route_d2h_s",
                "policy_cpu_s",
                "host_reuse_wait_s",
                "host_gather_s",
                "h2d_enqueue_s",
                "compute_enqueue_s",
            ),
            0.0,
        )
        if load_residents:
            self.reload_residents()

    def _gather(self, layer: int, ids: Sequence[int]) -> None:
        for i, expert in enumerate(ids):
            self.host_indices[i] = expert
        n = len(ids)
        for source, dest in zip(
            self.sources[layer], (self.gather13, self.gather2), strict=True
        ):
            torch.index_select(source, 0, self.host_indices[:n], out=dest[:n])

    def reload_residents(self) -> None:
        """Called after checkpoint completeness, before the profiling forward."""
        self.backend.synchronize()
        offset = 0
        for layer, ids in enumerate(self.policy.resident_ids):
            n = len(ids)
            if not n:
                continue
            self._gather(layer, ids)
            for gpu, host in ((self.w13, self.gather13), (self.w2, self.gather2)):
                gpu[offset : offset + n].copy_(host[:n], non_blocking=True)
            # A single reusable pinned gather cannot be refilled until copy ends.
            self.backend.synchronize()
            offset += n
            self._startup_bytes += n * self.expert_bytes

    def execute(
        self,
        layer: int,
        topk_ids: torch.Tensor,
        compute: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("expert pool has an active operation")
        try:
            if self._failed:
                raise RuntimeError("expert pool failed; restart worker")
            self._active = True
            if (
                topk_ids.ndim != 2
                or topk_ids.numel() == 0
                or topk_ids.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("Top-k IDs must be a nonempty integer matrix")
            t = perf_counter()
            # Copy the native IDs, not weights, and deduplicate on CPU. This
            # includes the wait for upstream routing/previous GPU dependencies.
            ids = sorted(set(topk_ids.detach().to(device="cpu").flatten().tolist()))
            self._timing["route_d2h_s"] += perf_counter() - t
            t = perf_counter()
            self.policy.observe(layer, ids)
            mapping = [-1] * self.num_experts
            misses: list[int] = []
            protected: set[tuple[int, int]] = set()
            for expert in ids:
                slot = self.policy.resident_slot(layer, expert)
                if slot is not None:
                    mapping[expert] = slot
                    self._resident_hits += 1
                else:
                    slot = self.policy.lookup(layer, expert)
                    if slot is None:
                        misses.append(expert)
                    else:
                        mapping[expert] = self.resident_slots + slot
                        protected.add((layer, expert))
            admissions: list[tuple[int, int]] = []
            bypasses: list[int] = []
            for expert in misses:
                admission = self.policy.admit(layer, expert, protected)
                if admission is None:
                    bypasses.append(expert)
                else:
                    admissions.append((expert, admission[0]))
                    protected.add((layer, expert))
            ordered = [expert for expert, _ in admissions] + bypasses
            for i, expert in enumerate(ordered):
                mapping[expert] = self.ingress_start + i
            self._timing["policy_cpu_s"] += perf_counter() - t
            self.backend.begin()
            t = perf_counter()
            self.backend.wait_host()
            self._timing["host_reuse_wait_s"] += perf_counter() - t
            t = perf_counter()
            n = len(ordered)
            if n:
                self._gather(layer, ordered)
            for expert, slot in enumerate(mapping):
                self.host_map[expert] = slot
            self._timing["host_gather_s"] += perf_counter() - t
            t = perf_counter()
            self.backend.mark(0)
            if n:
                for gpu, host in ((self.w13, self.gather13), (self.w2, self.gather2)):
                    gpu[self.ingress_start : self.ingress_start + n].copy_(
                        host[:n], non_blocking=True
                    )
                self._h2d_bytes += n * self.expert_bytes
                self._copy_launches += 2
            self.expert_map.copy_(self.host_map, non_blocking=True)
            self._metadata_h2d_bytes += self.num_experts * 4
            count = len(admissions)
            for i, (_, slot) in enumerate(admissions):
                self.host_indices[i] = slot
            if count:
                self.device_indices[:count].copy_(
                    self.host_indices[:count], non_blocking=True
                )
                self._metadata_h2d_bytes += count * 8
            self.backend.uploaded()
            self.backend.mark(1)
            self._timing["h2d_enqueue_s"] += perf_counter() - t
            t = perf_counter()
            result = compute(self.w13, self.w2, self.expert_map)
            self.backend.mark(2)
            self._timing["compute_enqueue_s"] += perf_counter() - t
            if count:
                # Disjoint views, contiguous admission prefix, no GPU gather
                # allocation whose peak could escape dummy-run KV profiling.
                for gpu in (self.w13, self.w2):
                    gpu[self.resident_slots : self.ingress_start].index_copy_(
                        0,
                        self.device_indices[:count],
                        gpu[self.ingress_start : self.ingress_start + count],
                    )
                self._promotion_bytes += count * self.expert_bytes
            self.backend.mark(3)
            self.backend.end()
            self._forwards[layer] += 1
            self._unique_demands += len(ids)
            self._max_unique = max(self._max_unique, len(ids))
            self._per_layer_unique_demands[layer] += len(ids)
            self._per_layer_max_unique[layer] = max(
                self._per_layer_max_unique[layer], len(ids)
            )
            return result
        except Exception:
            self._failed = True
            raise
        finally:
            self._active = False
            self._lock.release()

    def reconfigure(self, resident_ratio: float) -> None:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot reconfigure an active forward")
        try:
            if self._failed:
                raise RuntimeError("expert pool failed; restart worker")
            if (
                isinstance(resident_ratio, bool)
                or not isinstance(resident_ratio, (int, float))
                or not math.isfinite(resident_ratio)
                or not 0 <= resident_ratio <= 1
            ):
                raise ValueError("resident ratio must be finite within [0,1]")
            residents = self.total_layers * math.floor(
                self.num_experts * resident_ratio
            )
            slots = self.ingress_start - residents
            if not 1 <= slots <= self.total_layers * self.num_experts:
                raise ValueError(
                    "fixed physical pool requires >=1 persistent cache slot"
                )
            self.backend.synchronize()
            self.policy.reconfigure(resident_ratio, slots)
            self.resident_slots, self.cache_slots = residents, slots
            try:
                self.reload_residents()
            except Exception:
                self._failed = True
                raise
        finally:
            self._lock.release()

    def stats(
        self, *, synchronize: bool = True, reset_timing: bool = False
    ) -> dict[str, object]:
        if self._active:
            raise RuntimeError("cannot snapshot an active forward")
        if synchronize:
            self.backend.synchronize()
        timing = dict(self._timing)
        cuda_timing = self.backend.timing(reset_timing)
        if reset_timing:
            self._timing = dict.fromkeys(self._timing, 0.0)
        source_bytes = self.total_layers * self.num_experts * self.expert_bytes
        pool_bytes = self.capacity * self.expert_bytes
        return {
            "schema_version": 1,
            "total_layers": self.total_layers,
            "num_experts": self.num_experts,
            "physical_slots": self.capacity,
            "resident_slots": self.resident_slots,
            "cache_slots": self.cache_slots,
            "ingress_slots": self.num_experts,
            "expert_bytes": self.expert_bytes,
            "host_source_bytes": source_bytes,
            "pinned_gather_bytes": self.num_experts * self.expert_bytes,
            "gpu_pool_bytes": pool_bytes,
            "gpu_resident_bytes": self.resident_slots * self.expert_bytes,
            "gpu_cache_bytes": self.cache_slots * self.expert_bytes,
            "gpu_ingress_bytes": self.num_experts * self.expert_bytes,
            "gpu_metadata_bytes": self.num_experts * 12,
            "pinned_metadata_bytes": self.num_experts * 12,
            "net_freed_bytes": source_bytes - pool_bytes - self.num_experts * 12,
            "h2d_bytes": self._h2d_bytes,
            "copy_launches": self._copy_launches,
            "startup_resident_h2d_bytes": self._startup_bytes,
            "promotion_d2d_bytes": self._promotion_bytes,
            "metadata_h2d_bytes": self._metadata_h2d_bytes,
            "forward_counts": list(self._forwards),
            "unique_demands": self._unique_demands,
            "max_unique_per_forward": self._max_unique,
            "per_layer_unique_demands": list(self._per_layer_unique_demands),
            "per_layer_max_unique_per_forward": list(self._per_layer_max_unique),
            "mean_unique_coverage": self._unique_demands
            / max(1, sum(self._forwards))
            / self.num_experts,
            "resident_hits": self._resident_hits,
            "policy": self.policy.stats(),
            "timing": {**timing, **cuda_timing},
            "weights_verified": 0,
            "weight_verification_scope": "CUDA parity tests required; no runtime weight D2H verification",
            "temporary_memory_scope": "native fused kernel workspace/output and CPU route IDs scale with scheduled tokens; no weight-sized GPU gather temporary",
            "memory_accounting_scope": "tensor payload bytes; excludes allocator rounding, CUDA event driver memory and Python policy metadata; native/chunk output workspace must be included in vLLM measured profiling peak",
            "failed": self._failed,
        }
