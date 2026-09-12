"""Unique-input measured decode runs, explicit fixed KV, and separate profiles."""

from __future__ import annotations

import argparse
import math
import os
import platform
import resource
from collections.abc import Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from flexmoe.analysis.schema import diagnostic_artifact
from flexmoe.bench import partial_runner as shared
from flexmoe.bench.analysis_runner import (
    AnalysisBackend,
    atomic_gzip,
    hardware_identity,
)
from flexmoe.bench.expert_cache_runner import ExpertBackend, load_profile
from flexmoe.datasets.decode_corpus import load_unique_workload
from flexmoe.vllm.decode_trace import SchedulerObserver
from flexmoe.vllm.expert_calibration import model_profile_identity


def cpu_memory() -> dict[str, Any]:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "status": "measured",
        "peak_rss_bytes": int(value * (1 if platform.system() == "Darwin" else 1024)),
        "scope": "frontend-process-lifetime-high-water; excludes-live-worker-RSS",
    }


class DecodeMechanismBackend(ExpertBackend):
    def __init__(
        self,
        config: shared.PartialRunConfig,
        *,
        mode: str = "native",
        profile: bool = False,
        profile_path: Path | None = None,
        resident_ratio: float = 0.8,
        cache_slots: int = 512,
        cache_policy: str = "decayed-lfu",
        calibration_count: int = 32,
        target_batch: int | None = None,
        capture_steps: int = 256,
        min_capture_steps: int = 64,
        trace_budget_bytes: int = 134217728,
        kv_bytes: int | None = None,
        selection_offset: int = 0,
        safety_reserve_bytes: int = 2_000_000_000,
    ) -> None:
        if mode not in ("native", "matched-resident", "offload"):
            raise ValueError("unknown decode mechanism mode")
        if mode == "native" and profile:
            raise ValueError(
                "native+profile is unsupported; use matched-resident --profile"
            )
        expected_arm = "partial-auto-kv" if mode == "offload" else "resident"
        if config.arm != expected_arm or config.offload_count:
            raise ValueError(
                "decode mode requires matching arm and no whole-layer offload"
            )
        if mode == "offload" and profile_path is None:
            raise ValueError("offload requires a held-out expert profile")
        for name, value in (("target_batch", target_batch), ("kv_bytes", kv_bytes)):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("capture_steps", capture_steps),
            ("min_capture_steps", min_capture_steps),
            ("trace_budget_bytes", trace_budget_bytes),
            ("safety_reserve_bytes", safety_reserve_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")
        if min_capture_steps > capture_steps:
            raise ValueError("min_capture_steps exceeds capture_steps")
        if type(selection_offset) is not int or selection_offset < 0:
            raise ValueError("selection_offset must be nonnegative")
        if config.timing_samples != 0:
            raise ValueError(
                "new decode runs require timing_samples=0; use --profile for detailed timing"
            )
        super().__init__(
            config,
            profile_path or Path("."),
            resident_ratio,
            cache_slots,
            cache_policy,
            calibration_count,
        )
        self.has_profile_path = profile_path is not None
        self.mode, self.profile = mode, profile
        self.target_batch, self.capture_steps = target_batch, capture_steps
        self.min_capture_steps, self.trace_budget_bytes = (
            min_capture_steps,
            trace_budget_bytes,
        )
        self.kv_bytes, self.selection_offset = kv_bytes, selection_offset
        self.safety_reserve_bytes = safety_reserve_bytes
        self.analysis = AnalysisBackend(
            mode="native" if mode == "native" else "eager",
            safety_reserve_bytes=safety_reserve_bytes,
        )
        self.run_dir: Path | None = None
        self.summary: dict[str, Any] = {}
        self.geometry: dict[str, int] = {}
        self.capture_active = False
        self.scheduler: SchedulerObserver | None = None
        self.repetition = 0

    def prepare(self, config: shared.PartialRunConfig, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.analysis.prepare(config, run_dir)
        self.geometry = self.analysis.geometry

    def _labels(self) -> dict[str, Any]:
        return {
            "comparison_backend": "decode-mechanism",
            "mode": self.mode,
            "profile": self.profile,
            "evidence_kind": "instrumented" if self.profile else "measured",
            "timing_eligible": not self.profile,
        }

    def summary_fields(self) -> dict[str, Any]:
        return diagnostic_artifact(
            "decode-run",
            {
                **self._labels(),
                "physical_safety_reserve_bytes": self.safety_reserve_bytes,
                "storage_backend": "expert-cache"
                if self.mode == "offload"
                else "native",
            },
        )

    def record_fields(self, kind: str) -> dict[str, Any]:
        return diagnostic_artifact(f"decode-{kind}", self._labels())

    def workload(self, config: shared.PartialRunConfig) -> shared.PartialWorkload:
        prompts, _, metadata = load_unique_workload(
            config.dataset_path,
            config.dataset_manifest,
            context_length=config.context_length,
            request_count=config.batch_size,
            calibration_count=self.calibration_count,
            selection_offset=self.selection_offset,
        )
        identity = model_profile_identity(
            config.model_path, config.tensor_parallel_size
        )
        profile_hash = None
        if self.has_profile_path:
            profile = load_profile(
                self.profile_path,
                config.model_path,
                config.tensor_parallel_size,
                cast(Sequence[str], metadata["calibration_input_hashes"]),
            )
            profile_hash = shared.digest_json(profile.to_dict())
        self.settings = {
            "timing_samples": config.timing_samples,
            "resident_ratio": self.resident_ratio,
            "cache_slots": self.cache_slots,
            "cache_policy": self.cache_policy,
            "calibration_count": self.calibration_count,
            "profile_sha256": profile_hash,
            "identity": identity,
        }
        if self.mode == "offload":
            geometry = identity["geometry"]
            layers, experts = geometry["total_layers"], geometry["num_experts"]
            slots = (
                layers * math.floor(experts * self.resident_ratio)
                + self.cache_slots
                + experts
            )
            freed = (layers * experts - slots) * 6 * geometry["hidden_size"] * geometry[
                "intermediate_size"
            ] - experts * 12
            if freed <= 0:
                raise ValueError("cache point does not free routed tensor bytes")
        return shared.PartialWorkload(
            prompts,
            {
                **metadata,
                "input_sha256": shared.digest_json(prompts),
                "prompt_hashes": metadata["selected_input_hashes"],
                "calibration_input_hashes_sha256": shared.digest_json(
                    metadata["calibration_input_hashes"]
                ),
                "unique_selected_request_count": len(prompts),
                "repeated_request_count": 0,
                "evaluation_pool_count": metadata["unique_evaluation_pool_count"],
            },
        )

    def contract_fields(self) -> dict[str, Any]:
        return {
            **self._labels(),
            "expert_cache": self.settings,
            "target_batch": self.target_batch,
            "capture_steps": self.capture_steps,
            "min_capture_steps": self.min_capture_steps,
            "trace_budget_bytes": self.trace_budget_bytes,
            "physical_safety_reserve_bytes": self.safety_reserve_bytes,
            "observation_policy": "prepared-inputs+forwarding-stats; profile-adds-eager-histograms-and-events",
        }

    def deltas(
        self, before: object, after: object, *, expected_workers: int
    ) -> dict[str, Any]:
        result = super().deltas(before, after, expected_workers=expected_workers)
        result["timing_scope"] = (
            "legacy-counter-sums-across-ranks; not TP-global elapsed or net loss; detailed per-layer events live in decode-profile"
        )
        result["cuda_timing_status"] = (
            "unavailable: legacy CUDA timing disabled; profile events are separate"
        )
        return result

    def configure(
        self, config: shared.PartialRunConfig, root: Path, layers: tuple[int, ...]
    ) -> None:
        if self.mode == "offload":
            super().configure(config, root, layers)
        else:
            shared.BenchmarkBackend.configure(self, config, root, ())
        os.environ["FLUXMOE_DECODE_MECHANISM"] = "1"
        if self.profile:
            os.environ["FLUXMOE_EXPERT_CALIBRATION"] = "1"

    def engine_arguments(
        self, config: shared.PartialRunConfig, fixed_kv_bytes: int | None
    ) -> dict[str, Any]:
        if fixed_kv_bytes is not None:
            raise ValueError(
                "decode adapter uses explicit --kv-bytes, not old resident-run override"
            )
        # Keep the old shared gate intact; the explicit experiment applies its override here.
        args = self.analysis.engine_arguments(replace(config, arm="resident"), None)
        args.update(
            kv_cache_memory_bytes=self.kv_bytes,
            disable_log_stats=False,
            max_num_seqs=config.max_num_seqs,
        )
        return args

    def resolved_policy(
        self, engine: Any, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        policy = self.analysis.resolved_policy(engine, arguments)
        if (
            "quantization" not in policy["model_config"]
            or policy["model_config"].get("quantization") is not None
            or policy["cache_config"].get("cpu_offload_gb") != 0
            or policy["cache_config"].get("swap_space_bytes") != 0
        ):
            raise RuntimeError(
                "decode requires actual unquantized BF16 with no built-in offload/swap"
            )
        return policy

    def validate_kv(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        ranked = shared.worker_rows(list(rows), self.config.tensor_parallel_size)
        shared.validate_fixed_kv(ranked, ranked, self.config.tensor_parallel_size)
        pairs = {
            (row["kv_cache_allocated_bytes"], row["num_gpu_blocks"]) for row in ranked
        }
        if len(pairs) != 1:
            raise ValueError("actual KV bytes/blocks differ between ranks")
        actual, blocks = next(iter(pairs))
        rounding = None
        if self.kv_bytes is not None:
            rounding = self.kv_bytes - actual
            if (
                any(
                    row.get("available_kv_cache_bytes") != self.kv_bytes
                    for row in ranked
                )
                or actual % blocks != 0
                or not 0 <= rounding < actual // blocks
            ):
                raise ValueError(
                    "actual KV allocation differs from explicit budget beyond allocator rounding"
                )
        return {
            "requested_bytes": self.kv_bytes,
            "allocated_bytes_per_rank": actual,
            "num_gpu_blocks": blocks,
            "rounding_bytes": rounding,
            "bytes_per_block": actual // blocks if actual % blocks == 0 else None,
            "scope": "worker-actual-storage+declared-bytes+num-blocks; rounding less than one aggregate block",
        }

    def _check_memory(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.analysis._check_memory(rows)
        for row in rows:
            current, peak = (
                row.get("torch_reserved_bytes"),
                row.get("torch_peak_reserved_bytes"),
            )
            if type(current) is not int or type(peak) is not int:
                raise RuntimeError("physical GPU peak reserve evidence unavailable")
            if (
                row["free_gpu_bytes"] - max(0, peak - current)
                < self.safety_reserve_bytes
            ):
                raise RuntimeError(
                    "physical GPU safety reserve exceeded during measured Torch peak"
                )

    def initialized(self, summary: dict[str, Any], engine: Any) -> None:
        self.summary = summary
        devices = shared.worker_rows(
            engine.collective_rpc("fluxmoe_analysis_device"), 4
        )
        hardware = hardware_identity(devices)
        if hardware["hardware_sha256"] is None:
            raise ValueError("four-worker UUID/capacity identity unavailable")
        summary["contract"].update(hardware)
        summary["contract"]["versions"]["python"] = platform.python_version()
        summary["geometry"] = self.geometry
        summary["actual_kv"] = self.validate_kv(summary["memory"])
        summary["hardware_budget"]["interpretation"] = (
            "utilization-is-profiling-not-hard-isolation"
        )
        summary["memory_scope"] = (
            "physical free + Torch current/peak; non-Torch transient peaks unavailable"
        )
        summary["cpu_memory"] = cpu_memory()
        summary["source_hashes"] = {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).parents[1] / "vllm" / "decode_trace.py",
                Path(__file__).parents[1] / "runtime" / "expert_pool.py",
            )
        }
        self._check_memory(summary["memory"])

    def before_measurement(self, engine: Any, workers: int) -> None:
        self._check_memory(shared._memory(engine, workers))
        super().before_measurement(engine, workers)
        self.scheduler = SchedulerObserver(engine.llm_engine)
        self.scheduler.start()
        self.capture_active = True
        starts = shared.worker_rows(
            engine.collective_rpc(
                "fluxmoe_decode_mechanism",
                kwargs={
                    "action": "start",
                    **{
                        key: value
                        for key, value in self.geometry.items()
                        if key != "expert_bytes"
                    },
                    "profile": self.profile,
                    "target_batch": self.target_batch,
                    "capture_steps": self.capture_steps,
                    "min_capture_steps": self.min_capture_steps,
                    "trace_budget_bytes": self.trace_budget_bytes,
                    "max_batch": self.config.max_num_seqs,
                    "safety_reserve_bytes": self.safety_reserve_bytes,
                },
            ),
            workers,
        )
        if any(row.get("active") is not True for row in starts):
            raise RuntimeError("decode observer start worker coverage incomplete")

    def _save_capture(
        self, raw: object, generation_status: str
    ) -> list[dict[str, Any]]:
        assert self.run_dir is not None
        observations = []
        for row in shared.worker_rows(raw, self.config.tensor_parallel_size):
            atomic_gzip(
                self.run_dir
                / f"decode-rep-{self.repetition:03d}-rank-{row['rank']}.json.gz",
                diagnostic_artifact(
                    "decode-profile",
                    {
                        **self._labels(),
                        "generation_status": generation_status,
                        "repetition": self.repetition,
                        "rank": row["rank"],
                        "contract": self.summary["contract"],
                        "geometry": self.geometry,
                        "observation": row,
                    },
                ),
            )
            observations.append(
                {
                    key: value
                    for key, value in row.items()
                    if key
                    not in (
                        "activation_rows",
                        "activation_summaries",
                        "pool_profile",
                        "model_step_spans",
                    )
                }
            )
        return observations

    def measurement_fields(self, engine: Any, workers: int) -> dict[str, Any]:
        raw = engine.collective_rpc(
            "fluxmoe_decode_mechanism", kwargs={"action": "stop"}
        )
        observations = self._save_capture(raw, "complete")
        self.capture_active = False
        assert self.scheduler is not None
        scheduler = self.scheduler.stop()
        self.repetition += 1
        return {
            "generation_status": "complete",
            "timing_eligible": not self.profile,
            "worker_observations": observations,
            "scheduler": scheduler,
            "memory": shared._memory(engine, workers),
            "cpu_memory": cpu_memory(),
            "memory_peak_scope": "measured-generate-after-synchronized-reset",
        }

    def validate_measurement(
        self, result: dict[str, Any], config: shared.PartialRunConfig
    ) -> None:
        self._check_memory(result["memory"])
        actual = self.validate_kv(result["memory"])
        if actual != self.summary["actual_kv"]:
            raise ValueError("KV allocation changed during measured generation")
        super().validate_repetition(result["diagnostics"], config)

    def finalize(self, summary: dict[str, Any]) -> None:
        self._check_memory(summary["final_memory"])
        summary["generation_status"] = "complete"
        summary["cpu_memory"] = cpu_memory()

    def cleanup(self, engine: Any) -> None:
        try:
            if self.capture_active and engine is not None:
                raw = engine.collective_rpc(
                    "fluxmoe_decode_mechanism", kwargs={"action": "stop"}
                )
                observations = self._save_capture(raw, "failed")
                self.summary["failed_observations"] = observations
                self.summary["failure_memory"] = shared._memory(
                    engine, self.config.tensor_parallel_size
                )
        except Exception as error:  # noqa: BLE001 -- preserve original generation failure
            self.summary["capture_failure"] = {
                "status": "failed",
                "error_type": type(error).__name__,
            }
        finally:
            was_active = self.capture_active
            self.capture_active = False
            if self.scheduler is not None:
                scheduler = self.scheduler.stop()
                if was_active:
                    self.summary["failure_scheduler"] = scheduler
            if was_active and self.run_dir is not None:
                self.summary["generation_status"] = "failed"
                self.summary["cpu_memory"] = cpu_memory()
                shared.atomic_json(self.run_dir / "summary.json", self.summary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("native", "matched-resident", "offload"), required=True
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct"),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(
            "benchmarks/data/decode-mechanism/qwen3next_unique_1536x_1k4k.jsonl.zst"
        ),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("benchmarks/data/decode-mechanism/dataset_manifest.json"),
    )
    parser.add_argument("--profile-path", type=Path)
    parser.add_argument(
        "--cache-policy", choices=("decayed-lfu", "lru"), default="decayed-lfu"
    )
    parser.add_argument("--resident-ratio", type=float, default=0.8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--target-batch", type=int)
    parser.add_argument("--kv-bytes", type=int)
    for flag, default in (
        ("batch-size", 1024),
        ("context-length", 1024),
        ("output-length", 512),
        ("max-num-seqs", 1024),
        ("max-num-batched-tokens", 8192),
        ("warmups", 1),
        ("repetitions", 3),
        ("seed", 20260912),
        ("smoke-output-length", 8),
        ("capture-steps", 256),
        ("min-capture-steps", 64),
        ("trace-budget-bytes", 134217728),
        ("safety-reserve-bytes", 2_000_000_000),
        ("selection-offset", 0),
        ("cache-slots", 512),
        ("calibration-count", 32),
    ):
        parser.add_argument("--" + flag, type=int, default=default)
    args = vars(parser.parse_args(argv))
    root, run = args.pop("project_root").resolve(), args.pop("run_dir").resolve()
    if not run.is_relative_to(root):
        parser.error("run-dir must remain inside project-root")
    for key in ("dataset_path", "dataset_manifest", "profile_path"):
        if args.get(key) is not None and not args[key].is_absolute():
            args[key] = root / args[key]
    config_values = {
        key: args.pop(key)
        for key in tuple(args)
        if key in shared.PartialRunConfig.__dataclass_fields__
    }
    config = shared.PartialRunConfig(
        arm="partial-auto-kv" if args["mode"] == "offload" else "resident",
        timing_samples=0,
        **config_values,
    )
    backend = DecodeMechanismBackend(config, **args)
    shared.run_benchmark(config, project_root=root, run_dir=run, backend=backend)
    print(run.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
