"""Diagnostic native/eager timing and a separate, single-window route capture."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import tempfile
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from flexmoe.analysis.schema import (
    DemandTrace,
    TraceEvent,
    diagnostic_artifact,
    parse_diagnostic_artifact,
    validate_contract,
)
from flexmoe.bench.partial_runner import (
    BenchmarkBackend,
    PartialRunConfig,
    PartialWorkload,
    _capture_policy,
    _memory,
    atomic_json,
    digest_json,
    read_json,
    run_benchmark,
    worker_rows,
)


def hardware_identity(devices: Sequence[Any]) -> dict[str, Any]:
    """Same ordered physical UUID/capacity digest for all three producers."""
    try:
        if len(devices) != 4:
            raise ValueError("expected four devices")
        rows = []
        for rank, prop in enumerate(devices):
            if isinstance(prop, Mapping) and prop.get("rank") != rank:
                raise ValueError("ordered worker rank identity differs")
            uuid = (
                prop.get("uuid")
                if isinstance(prop, Mapping)
                else getattr(prop, "uuid", None)
            )
            total = (
                prop.get("total_memory")
                if isinstance(prop, Mapping)
                else getattr(prop, "total_memory", None)
            )
            if (
                not isinstance(uuid, str)
                or not uuid
                or type(total) is not int
                or total <= 0
            ):
                raise ValueError("actual device UUID or capacity unavailable")
            rows.append({"rank": rank, "uuid": uuid, "total_memory": total})
        return {"hardware_sha256": digest_json(rows)}
    except (RuntimeError, AssertionError, ValueError, AttributeError) as error:
        return {
            "hardware_sha256": None,
            "hardware_unavailable_reason": f"CUDA UUID/capacity inventory unavailable: {type(error).__name__}",
        }


def select_workload(
    config: PartialRunConfig,
    *,
    selection_offset: int = 0,
    excluded_hashes: set[str] | None = None,
    synthetic_context: bool = False,
) -> PartialWorkload:
    from flexmoe.datasets.sharegpt import read_jsonl_zst, verify_subset

    if type(selection_offset) is not int or selection_offset < 0:
        raise ValueError("selection_offset must be nonnegative")
    verify_subset(config.dataset_path, config.dataset_manifest)
    records = tuple(read_jsonl_zst(config.dataset_path))
    source = tuple(
        r.prompt_token_ids for r in records if r.context_length == config.context_length
    )
    synthetic = not source
    source_hash = digest_json(source)
    if synthetic:
        if not synthetic_context:
            raise ValueError(
                "context unavailable; explicit synthetic context packing required"
            )
        lengths = [
            r.context_length
            for r in records
            if r.context_length < config.context_length
        ]
        if not lengths:
            raise ValueError("no shorter source bucket for synthetic packing")
        bucket = max(lengths)
        short = tuple(r.prompt_token_ids for r in records if r.context_length == bucket)
        source_hash = digest_json(short)
        # One packed request per source start; deterministic cycling of existing tokens.
        source = tuple(
            tuple(
                token
                for j in range((config.context_length + bucket - 1) // bucket)
                for token in short[(i + j) % len(short)]
            )[: config.context_length]
            for i in range(len(short))
        )
    excluded = excluded_hashes or set()
    pool = tuple(row for row in source if digest_json(row) not in excluded)
    if not pool:
        raise ValueError("prompt exclusion leaves an empty evaluation pool")
    prompts = tuple(
        pool[(selection_offset + i) % len(pool)] for i in range(config.batch_size)
    )
    hashes = tuple(sorted({digest_json(row) for row in prompts}))
    metadata: dict[str, Any] = {
        "dataset_sha256": read_json(config.dataset_manifest)["sha256"],
        "dataset_manifest_sha256": sha256(
            config.dataset_manifest.read_bytes()
        ).hexdigest(),
        "input_sha256": digest_json(prompts),
        "prompt_hashes": hashes,
        "source_request_count": len(source),
        "evaluation_pool_count": len({digest_json(row) for row in pool}),
        "unique_selected_request_count": len(hashes),
        "repeated_request_count": config.batch_size - len(hashes),
        "sampling_policy": "deterministic-packed-existing-tokens"
        if synthetic
        else "existing-prompts",
        "synthetic": synthetic,
        "split_scope": "per-prompt-hash-exclusion" if excluded else "unsplit",
    }
    if synthetic:
        metadata["synthetic_packing"] = (
            f"cyclic-source-start;source_sha256={source_hash}"
        )
    if excluded:
        metadata["calibration_input_hashes_sha256"] = digest_json(sorted(excluded))
    return PartialWorkload(prompts, metadata)


def excluded_prompt_hashes(paths: Sequence[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as stream:
            root = json.load(stream)
        fields = parse_diagnostic_artifact(root, expected_kind="demand-trace")
        trace = DemandTrace.from_dict(fields["trace"])
        result.update(cast(Sequence[str], trace.contract["prompt_hashes"]))
    return result


def atomic_gzip(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
                stream.write(json.dumps(data, sort_keys=True, allow_nan=False).encode())
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class AnalysisBackend(BenchmarkBackend):
    requires_layer_transfers = False

    def __init__(
        self,
        mode: str = "native",
        max_trace_steps: int = 1024,
        trace_budget_bytes: int = 134217728,
        synthetic_context: bool = False,
        *,
        selection_offset: int = 0,
        exclude_trace: Sequence[Path] = (),
        safety_reserve_bytes: int = 2_000_000_000,
    ) -> None:
        if mode not in ("native", "eager", "trace"):
            raise ValueError("unknown analysis mode")
        for name, value in (
            ("max_trace_steps", max_trace_steps),
            ("trace_budget_bytes", trace_budget_bytes),
            ("safety_reserve_bytes", safety_reserve_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")
        self.mode = mode
        self.max_trace_steps = max_trace_steps
        self.trace_budget_bytes = trace_budget_bytes
        self.synthetic_context = synthetic_context
        self.selection_offset = selection_offset
        self.exclude_trace = exclude_trace
        self.safety_reserve_bytes = safety_reserve_bytes
        self.capture_active = False
        self.run_dir: Path | None = None
        self.summary: dict[str, Any] = {}
        self.geometry: dict[str, int] = {}

    def prepare(self, config: PartialRunConfig, run_dir: Path) -> None:
        self.run_dir = run_dir
        raw = read_json(config.model_path / "config.json")
        raw = raw.get("text_config", raw)
        if raw.get("model_type") != "qwen3_next":
            raise ValueError("analysis requires the existing Qwen3-Next model")
        names = {
            "total_layers": "num_hidden_layers",
            "num_experts": "num_experts",
            "top_k": "num_experts_per_tok",
        }
        self.geometry = {name: raw[field] for name, field in names.items()}
        hidden, intermediate = raw["hidden_size"], raw["moe_intermediate_size"]
        if any(
            type(v) is not int or v <= 0
            for v in (*self.geometry.values(), hidden, intermediate)
        ):
            raise ValueError("invalid model geometry")
        if intermediate % 4:
            raise ValueError("expert intermediate size must be TP4 divisible")
        self.geometry["expert_bytes"] = 3 * hidden * (intermediate // 4) * 2

    def summary_fields(self) -> dict[str, Any]:
        return diagnostic_artifact(
            "native-summary",
            {
                "engine_mode": self.mode,
                "source_kind": "native-measured",
                "timing_eligible": self.mode != "trace",
                "measurement_evidence": "measured"
                if self.mode != "trace"
                else "instrumented-not-throughput",
                "physical_safety_reserve_bytes": self.safety_reserve_bytes,
                "selection_offset": self.selection_offset,
            },
        )

    def record_fields(self, kind: str) -> dict[str, Any]:
        return diagnostic_artifact(
            f"analysis-{kind}", {"timing_eligible": self.mode != "trace"}
        )

    def workload(self, config: PartialRunConfig) -> PartialWorkload:
        if self.mode == "trace" and config.repetitions != 1:
            raise ValueError(
                "trace requires repetitions=1: one immutable capture window"
            )
        return select_workload(
            config,
            selection_offset=self.selection_offset,
            excluded_hashes=excluded_prompt_hashes(self.exclude_trace),
            synthetic_context=self.synthetic_context,
        )

    def configure(
        self, config: PartialRunConfig, root: Path, layers: tuple[int, ...]
    ) -> None:
        if config.arm != "resident" or layers:
            raise ValueError("analysis measures unmodified resident weights only")
        super().configure(config, root, layers)
        if self.mode == "trace":
            os.environ.update(
                FLUXMOE_EXPERT_CALIBRATION="1", FLUXMOE_ANALYSIS_TRACE="1"
            )

    def engine_arguments(
        self, config: PartialRunConfig, fixed_kv_bytes: int | None
    ) -> dict[str, Any]:
        args = super().engine_arguments(config, fixed_kv_bytes)
        if self.mode == "native":
            args["enforce_eager"] = False
            args["disable_custom_all_reduce"] = False
            args.pop("compilation_config")
        return args

    def resolved_policy(
        self, engine: Any, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        if self.mode != "native":
            return super().resolved_policy(engine, arguments)
        policy = _capture_policy(engine, arguments)
        required = {
            "model_config": {
                "enforce_eager": False,
                "quantization": None,
                "max_model_len": arguments["max_model_len"],
                "seed": arguments["seed"],
            },
            "cache_config": {
                "enable_prefix_caching": False,
                "cpu_offload_gb": 0,
                "swap_space_bytes": 0,
                "gpu_memory_utilization": arguments["gpu_memory_utilization"],
            },
            "scheduler_config": {
                "max_num_seqs": arguments["max_num_seqs"],
                "max_num_batched_tokens": arguments["max_num_batched_tokens"],
                "enable_chunked_prefill": True,
            },
            "parallel_config": {
                "tensor_parallel_size": 4,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "disable_custom_all_reduce": False,
            },
        }
        for group, fields in required.items():
            for name, expected in fields.items():
                if name not in policy[group] or policy[group][name] != expected:
                    raise RuntimeError(
                        f"resolved native policy differs: {group}.{name}"
                    )
        if str(policy["model_config"].get("dtype")) not in (
            "torch.bfloat16",
            "bfloat16",
        ):
            raise RuntimeError("native dtype must remain bfloat16")
        if any(
            policy["compilation_config"].get(name) is None
            for name in ("level", "cudagraph_mode")
        ):
            raise RuntimeError("resolved native compilation/graph state unavailable")
        return policy

    def initialized(self, summary: dict[str, Any], engine: Any) -> None:
        self.summary = summary
        devices = worker_rows(engine.collective_rpc("fluxmoe_analysis_device"), 4)
        summary["contract"].update(hardware_identity(devices))
        summary["contract"]["versions"]["python"] = platform.python_version()
        summary["contract"] = validate_contract(
            summary["contract"], require_prompt_hashes=True
        )
        summary["geometry"] = self.geometry
        summary["source_hashes"] = {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).parents[1] / "vllm" / "analysis_trace.py",
            )
        }
        summary["hardware_budget"]["interpretation"] = (
            "utilization-is-profiling-not-hard-isolation"
        )
        self._check_memory(summary["memory"])

    def _check_memory(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            if (
                type(row.get("free_gpu_bytes")) is not int
                or row["free_gpu_bytes"] < self.safety_reserve_bytes
            ):
                raise RuntimeError(
                    "physical GPU safety reserve unavailable or exceeded"
                )

    def before_measurement(self, engine: Any, workers: int) -> None:
        self._check_memory(_memory(engine, workers))
        engine.collective_rpc("fluxmoe_reset_memory_peaks")
        if self.mode == "trace":
            self.capture_active = True
            engine.collective_rpc(
                "fluxmoe_analysis_trace",
                kwargs={
                    "action": "start",
                    **{k: v for k, v in self.geometry.items() if k != "expert_bytes"},
                    "max_trace_steps": self.max_trace_steps,
                    "trace_budget_bytes": self.trace_budget_bytes,
                    "safety_reserve_bytes": self.safety_reserve_bytes,
                },
            )

    def _save_capture(
        self, raw: object, generation_status: str
    ) -> list[dict[str, Any]]:
        assert self.run_dir is not None
        clean = []
        for row in worker_rows(raw, 4):
            trace = None
            if row.get("captured_steps", 0) > 0:
                full = row["full_workload"] and generation_status == "complete"
                trace = DemandTrace(
                    self.run_dir.name,
                    row["rank"],
                    self.summary["contract"],
                    **self.geometry,
                    observed_steps=row["observed_steps"],
                    captured_steps=row["captured_steps"],
                    full_workload=full,
                    generated_tokens=(
                        self.summary["contract"]["batch_size"]
                        * self.summary["contract"]["output_length"]
                    )
                    if full
                    else None,
                    events=tuple(TraceEvent.from_dict(e) for e in row["events"]),
                ).to_dict()
            coverage = {k: v for k, v in row.items() if k != "events"}
            fields = {
                "trace": trace,
                "capture": coverage,
                "generation_status": generation_status,
                "timing_eligible": False,
                "missing_evidence": []
                if trace is not None
                else ["no-complete-step-layer-grid"],
            }
            atomic_gzip(
                self.run_dir / f"trace-rank-{row['rank']}.json.gz",
                diagnostic_artifact("demand-trace", fields),
            )
            clean.append(coverage)
        return clean

    def measurement_fields(self, engine: Any, workers: int) -> dict[str, Any]:
        result: dict[str, Any] = {"timing_eligible": self.mode != "trace"}
        if self.mode == "trace":
            raw = engine.collective_rpc(
                "fluxmoe_analysis_trace", kwargs={"action": "stop"}
            )
            self.capture_active = False
            result["capture"] = self._save_capture(raw, "complete")
        result["memory"] = _memory(engine, workers)
        return result

    def validate_measurement(
        self, result: dict[str, Any], config: PartialRunConfig
    ) -> None:
        # The shared runner persists the complete failed sample if this rejects.
        self._check_memory(result["memory"])

    def finalize(self, summary: dict[str, Any]) -> None:
        self._check_memory(summary["final_memory"])
        if self.mode == "trace":
            summary["capture_status"] = (
                "complete"
                if all(
                    row["full_workload"] for row in summary["repetitions"][0]["capture"]
                )
                else "incomplete"
            )

    def cleanup(self, engine: Any) -> None:
        if self.capture_active and engine is not None:
            try:
                raw = engine.collective_rpc(
                    "fluxmoe_analysis_trace", kwargs={"action": "stop"}
                )
                self._save_capture(raw, "failed")
            except Exception as error:  # noqa: BLE001 -- preserve original generation error after best-effort cleanup
                if self.run_dir is not None:
                    atomic_json(
                        self.run_dir / "capture-failure.json",
                        diagnostic_artifact(
                            "capture-failure",
                            {
                                "status": "failed",
                                "error_type": type(error).__name__,
                                "timing_eligible": False,
                            },
                        ),
                    )
            finally:
                self.capture_active = False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "eager", "trace"), required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct"),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("benchmarks/data/sharegpt/qwen3next_1024_requests.jsonl.zst"),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("benchmarks/data/sharegpt/dataset_manifest.json"),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--synthetic-context", action="store_true")
    parser.add_argument("--exclude-trace", type=Path, action="append", default=[])
    for name, default in (
        ("batch-size", 32),
        ("context-length", 1024),
        ("output-length", 128),
        ("max-num-seqs", 512),
        ("max-num-batched-tokens", 8192),
        ("warmups", 1),
        ("repetitions", 3),
        ("seed", 20260905),
        ("smoke-output-length", 8),
        ("timing-samples", 0),
        ("max-trace-steps", 1024),
        ("trace-budget-bytes", 134217728),
        ("safety-reserve-bytes", 2_000_000_000),
        ("selection-offset", 0),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    args = vars(parser.parse_args(argv))
    root = args.pop("project_root").resolve()
    run = args.pop("run_dir").resolve()
    if not run.is_relative_to(root):
        parser.error("run-dir must be inside project-root")
    backend_fields = (
        "mode",
        "max_trace_steps",
        "trace_budget_bytes",
        "synthetic_context",
        "selection_offset",
        "exclude_trace",
        "safety_reserve_bytes",
    )
    backend = AnalysisBackend(**{key: args.pop(key) for key in backend_fields})
    for key in ("dataset_path", "dataset_manifest"):
        if not args[key].is_absolute():
            args[key] = root / args[key]
    run_benchmark(
        PartialRunConfig(arm="resident", **args),
        project_root=root,
        run_dir=run,
        backend=backend,
    )
    print(run.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
