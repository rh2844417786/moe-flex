"""Native resident KV capacity diagnostic, using the shared benchmark loop."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from flexmoe.bench import kv_oracle_evidence as evidence
from flexmoe.bench.partial_runner import (
    BenchmarkBackend,
    PartialRunConfig,
    _memory,
    atomic_json,
    engine_arguments,
    read_json,
    run_benchmark,
)


class OracleBackend(BenchmarkBackend):
    requires_layer_transfers = False

    def __init__(
        self,
        role: str,
        baseline_reference: Mapping[str, Any] | None = None,
        small_add_bytes: int = 3_200_000_000,
        large_add_bytes: int = 18_100_000_000,
        safety_fraction: float = 0.90,
        safety_margin_bytes: int = 2_000_000_000,
    ) -> None:
        if role not in evidence.ROLES:
            raise ValueError("unknown Oracle role")
        evidence.integer(small_add_bytes, 1)
        if evidence.integer(large_add_bytes, 1) <= small_add_bytes:
            raise ValueError("large increment must exceed small")
        if not 0 < evidence.number(safety_fraction) <= 1:
            raise ValueError("invalid safety fraction")
        evidence.integer(safety_margin_bytes)
        self.role = role
        self.reference = baseline_reference
        self.small_add_bytes, self.large_add_bytes = small_add_bytes, large_add_bytes
        self.fraction, self.margin = safety_fraction, safety_margin_bytes
        self.k0: int | None = None
        self.target: int | None = None
        self.initial_kv: int | None = None
        self.last_probe: list[dict[str, Any]] = []
        if role != "r0":
            if baseline_reference is None:
                raise ValueError("O requires completed r0 anchor")
            self.k0 = evidence.validate_run(baseline_reference)
            if baseline_reference["role"] != "r0":
                raise ValueError("only r0 may anchor capacity")
            self.target = self.k0 + (
                small_add_bytes if role == "small" else large_add_bytes
            )
            for raw in [
                baseline_reference["memory"],
                baseline_reference["final_memory"],
                *(rep["memory"] for rep in baseline_reference["repetitions"]),
            ]:
                for row in evidence.rank_rows(raw):
                    non_kv = evidence.estimated_peak(row) - self.k0
                    if (
                        non_kv + self.target + self.margin
                        > row["total_gpu_bytes"] * self.fraction
                    ):
                        raise evidence.SafetyRefusal(
                            "estimated Oracle safety boundary exceeded"
                        )
        elif baseline_reference is not None:
            raise ValueError(
                "r0 must derive fresh automatic KV; anchor checked by suite"
            )

    def summary_fields(self) -> dict[str, Any]:
        return {
            **evidence.LABELS,
            "role": self.role,
            "baseline_kv_bytes": self.k0,
            "small_add_bytes": self.small_add_bytes,
            "large_add_bytes": self.large_add_bytes,
            "safety_fraction": self.fraction,
            "safety_margin_bytes": self.margin,
            "safety_scope": "estimate-and-observed-boundaries-not-hard-cap",
            "nominal_r0_gpu_memory_utilization": 0.60,
            "obeys_nominal_r0_budget": self.role == "r0",
        }

    def contract_fields(self) -> dict[str, Any]:
        return {"comparison_backend": "kv-oracle"}

    def engine_arguments(
        self, config: PartialRunConfig, fixed_kv_bytes: int | None
    ) -> dict[str, Any]:
        if (
            config.arm != "resident"
            or config.offload_count
            or fixed_kv_bytes is not None
        ):
            raise ValueError("Oracle must use native resident engine")
        if config.gpu_memory_utilization != 0.60 or config.warmups < 1:
            raise ValueError("Oracle requires nominal 0.60 and warmup")
        args = engine_arguments(config, None)
        args["kv_cache_memory_bytes"] = self.target
        return args

    def initialized(self, summary: dict[str, Any], engine: Any) -> None:
        self.initial_kv = evidence.validate_memory(
            summary["memory"], self.fraction, self.margin
        )
        if self.target is not None:
            for row in evidence.rank_rows(summary["memory"]):
                block = self.initial_kv / row["num_gpu_blocks"]
                if (
                    self.k0 is None
                    or not self.k0 < self.initial_kv <= self.target
                    or self.target - self.initial_kv >= block
                ):
                    raise ValueError("initial actual KV differs from aligned target")
        # Keep the common field truthful: O's requested bytes are not a 0.60 cap.
        budget = summary["hardware_budget"]
        budget["nominal_r0_gpu_bytes_by_rank"] = budget.pop(
            "requested_gpu_bytes_by_rank"
        )
        if self.reference is not None:
            evidence.validate_anchor(self.reference, summary)
        probes = evidence.rank_rows(
            engine.collective_rpc("fluxmoe_native_probe", kwargs={"action": "start"})
        )
        engine.collective_rpc("fluxmoe_native_probe", kwargs={"action": "stop"})
        if any(
            p.get("all_parameters_cuda") is not True or not p.get("parameter_count")
            for p in probes
        ):
            raise ValueError("native CUDA parameter residency unavailable")

    def before_measurement(self, engine: Any, workers: int) -> None:
        engine.collective_rpc("fluxmoe_reset_memory_peaks")
        engine.collective_rpc("fluxmoe_native_probe", kwargs={"action": "start"})

    def measurement_fields(self, engine: Any, workers: int) -> dict[str, Any]:
        self.last_probe = evidence.rank_rows(
            engine.collective_rpc("fluxmoe_native_probe", kwargs={"action": "stop"})
        )
        memory = _memory(engine, workers)
        if (
            evidence.validate_memory(memory, self.fraction, self.margin)
            != self.initial_kv
        ):
            raise ValueError("actual KV capacity changed during measurement")
        evidence.validate_probes(self.last_probe)
        return {"native_probe": self.last_probe, "memory": memory}

    def finalize(self, summary: dict[str, Any]) -> None:
        summary["native_probe"] = self.last_probe
        evidence.validate_run(summary)
        if self.reference is not None:
            evidence.validate_anchor(self.reference, summary)
            matched = all(
                summary["smoke"][k] == self.reference["smoke"][k]
                for k in ("input_sha256", "output_sha256")
            )
            summary["smoke_matches_resident"] = matched
            if not matched:
                raise ValueError("Oracle smoke differs from r0")


def run_point(
    config: PartialRunConfig,
    *,
    project_root: Path,
    run_dir: Path,
    role: str,
    reference: Mapping[str, Any] | None = None,
    small_add_bytes: int = 3_200_000_000,
    large_add_bytes: int = 18_100_000_000,
    safety_fraction: float = 0.90,
    safety_margin_bytes: int = 2_000_000_000,
) -> Path:
    evidence.identifier(run_dir.name)
    if run_dir.exists():
        raise FileExistsError(run_dir)
    stage = "budget-construction"
    try:
        backend = OracleBackend(
            role,
            reference,
            small_add_bytes,
            large_add_bytes,
            safety_fraction,
            safety_margin_bytes,
        )
        stage = "native-execution"
        return run_benchmark(
            config, project_root=project_root, run_dir=run_dir, backend=backend
        )
    except BaseException as error:
        path = run_dir / "summary.json"
        row = (
            read_json(path)
            if path.is_file()
            else {
                **evidence.LABELS,
                "schema_version": 1,
                "run_id": run_dir.name,
                "role": role,
                "arm": "resident",
                "repetitions": [],
                "repetitions_completed": 0,
            }
        )
        row.update(
            status="failed",
            failure_code=evidence.failure_code(error),
            failure_stage=stage,
        )
        for name, value in (
            ("small_add_bytes", small_add_bytes),
            ("large_add_bytes", large_add_bytes),
            ("safety_fraction", safety_fraction),
            ("safety_margin_bytes", safety_margin_bytes),
        ):
            try:
                evidence.number(value)
            except ValueError:
                continue
            row[name] = value
        atomic_json(path, row)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--role", choices=evidence.ROLES, required=True)
    parser.add_argument("--baseline-reference", type=Path)
    for name in ("model-path", "dataset-path", "dataset-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name, default in {
        "batch-size": 512,
        "context-length": 4096,
        "output-length": 256,
        "warmups": 1,
        "repetitions": 1,
        "max-num-seqs": 512,
        "max-num-batched-tokens": 8192,
        "seed": 20260905,
        "small-add-bytes": 3_200_000_000,
        "large-add-bytes": 18_100_000_000,
        "safety-margin-bytes": 2_000_000_000,
    }.items():
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--safety-fraction", type=float, default=0.90)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    if root != Path("/home/jovyan/wangtonghan/moe-flex"):
        parser.error("server project root required")
    if not args.run_dir.resolve().is_relative_to(root / "runs/kv-oracle"):
        parser.error("run directory must be inside project runs/kv-oracle")
    reference = (
        read_json(args.baseline_reference / "summary.json")
        if args.baseline_reference
        else None
    )
    config = PartialRunConfig(
        "resident",
        args.model_path,
        args.dataset_path,
        args.dataset_manifest,
        batch_size=args.batch_size,
        context_length=args.context_length,
        output_length=args.output_length,
        warmups=args.warmups,
        repetitions=args.repetitions,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        seed=args.seed,
        timing_samples=0,
    )
    run_point(
        config,
        project_root=root,
        run_dir=args.run_dir,
        role=args.role,
        reference=reference,
        small_add_bytes=args.small_add_bytes,
        large_add_bytes=args.large_add_bytes,
        safety_fraction=args.safety_fraction,
        safety_margin_bytes=args.safety_margin_bytes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
