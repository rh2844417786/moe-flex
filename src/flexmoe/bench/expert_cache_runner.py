"""Native held-out calibration and explicit expert-cache benchmark adapter."""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

from flexmoe.bench import expert_cache_evidence as evidence
from flexmoe.bench import partial_runner as shared
from flexmoe.runtime.expert_profile import ExpertProfile
from flexmoe.vllm.expert_calibration import ModelProfileIdentity, model_profile_identity


@dataclass(frozen=True)
class CalibrationSplit:
    calibration: tuple[tuple[int, ...], ...]
    hashes: tuple[str, ...]
    evaluation: shared.PartialWorkload


def split_workload(
    dataset: Path,
    manifest: Path,
    context_length: int,
    batch_size: int,
    calibration_count: int,
) -> CalibrationSplit:
    from flexmoe.datasets.sharegpt import read_jsonl_zst, verify_subset

    if any(
        type(x) is not int or x < 1
        for x in (context_length, batch_size, calibration_count)
    ):
        raise ValueError("split sizes must be positive integers")
    verify_subset(dataset, manifest)
    source = [
        row.prompt_token_ids
        for row in read_jsonl_zst(dataset)
        if row.context_length == context_length
    ]
    unique = {shared.digest_json(row): row for row in source}
    if len(unique) <= calibration_count:
        raise ValueError("held-out split leaves no evaluation pool")
    hashes = tuple(list(unique)[:calibration_count])
    calibration = tuple(unique[key] for key in hashes)
    pool = tuple(row for key, row in unique.items() if key not in hashes)
    prompts = tuple(pool[index % len(pool)] for index in range(batch_size))
    if set(hashes) & {shared.digest_json(row) for row in prompts}:
        raise ValueError("calibration/evaluation input overlap")
    metadata = {
        "dataset_sha256": shared.read_json(manifest)["sha256"],
        "dataset_manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
        "input_sha256": shared.digest_json(prompts),
        "source_request_count": len(source),
        "evaluation_pool_count": len(pool),
        "calibration_count": calibration_count,
        "calibration_input_hashes_sha256": shared.digest_json(hashes),
        "unique_selected_request_count": min(batch_size, len(pool)),
        "repeated_request_count": max(0, batch_size - len(pool)),
        "sampling_policy": "heldout-token-hash-first-unique",
        "split_scope": "full-prompt-token-hashes; source-conversation-disjointness-not-claimed",
    }
    return CalibrationSplit(
        calibration, hashes, shared.PartialWorkload(prompts, metadata)
    )


def _validate_profile(
    profile: ExpertProfile, identity: ModelProfileIdentity, hashes: Sequence[str]
) -> None:
    profile.validate(
        expected_geometry=identity["geometry"],
        expected_model_config_sha256=identity["model_config_sha256"],
        expected_model_identity_sha256=identity["model_identity_sha256"],
        expected_tensor_parallel_size=identity["tensor_parallel_size"],
        expected_calibration_input_hashes=hashes,
    )
    if not all(x > 0 for x in profile.forward_counts):
        raise ValueError("calibration must observe every layer")


def profile_from_workers(
    raw: object, identity: ModelProfileIdentity, hashes: Sequence[str]
) -> ExpertProfile:
    profiles = []
    for row in evidence.ranked(raw, identity["tensor_parallel_size"]):
        profile = ExpertProfile.from_dict(
            {
                "schema_version": row["schema_version"],
                **{key: row[key] for key in identity},
                "calibration_input_hashes": list(hashes),
                "counts": row["counts"],
                "forward_counts": row["forward_counts"],
            }
        )
        _validate_profile(profile, identity, hashes)
        profiles.append(profile)
    layers, experts = (
        identity["geometry"]["total_layers"],
        identity["geometry"]["num_experts"],
    )
    return ExpertProfile.from_dict(
        {
            "schema_version": 1,
            **identity,
            "calibration_input_hashes": list(hashes),
            "counts": [
                [
                    sum(p.counts[layer][expert] for p in profiles)
                    for expert in range(experts)
                ]
                for layer in range(layers)
            ],
            "forward_counts": [
                sum(p.forward_counts[layer] for p in profiles)
                for layer in range(layers)
            ],
        }
    )


def load_profile(
    path: Path, model_path: Path, workers: int, hashes: Sequence[str]
) -> ExpertProfile:
    profile = ExpertProfile.from_dict(shared.read_json(path))
    _validate_profile(profile, model_profile_identity(model_path, workers), hashes)
    return profile


class ExpertBackend(shared.BenchmarkBackend):
    stats_key = "expert_cache_stats"
    requires_layer_transfers = False

    def __init__(
        self,
        config: shared.PartialRunConfig,
        profile_path: Path,
        resident_ratio: float = 0.5,
        cache_slots: int = 256,
        cache_policy: str = "decayed-lfu",
        calibration_count: int = 32,
    ):
        if (
            not 0 <= resident_ratio < 1
            or not 1 <= cache_slots
            or cache_policy not in ("decayed-lfu", "lru")
        ):
            raise ValueError("invalid expert cache settings")
        if config.offload_count:
            raise ValueError("expert cache cannot also offload whole layers")
        self.config, self.profile_path = config, profile_path
        self.resident_ratio, self.cache_slots = resident_ratio, cache_slots
        self.cache_policy, self.calibration_count = cache_policy, calibration_count
        self.settings: dict[str, Any] = {}

    def summary_fields(self) -> dict[str, Any]:
        return {
            "storage_backend": "native"
            if self.config.arm == "resident"
            else "expert-cache"
        }

    def workload(self, config: shared.PartialRunConfig) -> shared.PartialWorkload:
        split = split_workload(
            config.dataset_path,
            config.dataset_manifest,
            config.context_length,
            config.batch_size,
            self.calibration_count,
        )
        profile = load_profile(
            self.profile_path,
            config.model_path,
            config.tensor_parallel_size,
            split.hashes,
        )
        identity = model_profile_identity(
            config.model_path, config.tensor_parallel_size
        )
        self.settings = {
            "resident_ratio": self.resident_ratio,
            "cache_slots": self.cache_slots,
            "cache_policy": self.cache_policy,
            "calibration_count": self.calibration_count,
            "profile_sha256": shared.digest_json(profile.to_dict()),
            "identity": identity,
        }
        geometry = identity["geometry"]
        layers, experts = geometry["total_layers"], geometry["num_experts"]
        slots = layers * int(experts * self.resident_ratio) + self.cache_slots + experts
        if (layers * experts - slots) * 6 * geometry["hidden_size"] * geometry[
            "intermediate_size"
        ] - experts * 12 <= 0:
            raise ValueError("cache point does not free routed tensor bytes")
        return split.evaluation

    def contract_fields(self) -> dict[str, Any]:
        return {"comparison_backend": "expert-cache", "expert_cache": self.settings}

    def smoke_tokens(self, workload: shared.PartialWorkload) -> tuple[int, ...]:
        # Keep the full held-out token hash; truncation can alias calibration inputs.
        return workload.prompts[0]

    def configure(
        self, config: shared.PartialRunConfig, root: Path, layers: tuple[int, ...]
    ) -> None:
        super().configure(config, root, ())
        for key in list(os.environ):
            if key.startswith("FLUXMOE_PARTIAL_"):
                del os.environ[key]
        os.environ.update(
            {
                "FLUXMOE_STORAGE_MODE": "expert-cache",
                "FLUXMOE_EXPERT_PROFILE_PATH": str(self.profile_path.resolve()),
                "FLUXMOE_RESIDENT_RATIO": str(self.resident_ratio),
                "FLUXMOE_CACHE_SLOTS": str(self.cache_slots),
                "FLUXMOE_CACHE_POLICY": self.cache_policy,
            }
        )

    def kv_budget(self, raw: object, workers: int) -> int:
        shared.validate_fixed_kv(raw, raw, workers)
        values = [
            row["kv_cache_allocated_bytes"] for row in evidence.ranked(raw, workers)
        ]
        if len(set(values)) != 1:
            raise ValueError(
                "single fixed KV budget cannot reproduce unequal rank allocations"
            )
        return int(values[0])

    def snapshot(
        self,
        engine: Any,
        workers: int,
        *,
        reset_timing: bool = False,
        synchronize: bool = True,
    ) -> list[dict[str, Any]]:
        if self.config.arm == "resident":
            return [
                {
                    "rank": rank,
                    "storage_backend": "native",
                    **dict.fromkeys(evidence.COUNTERS, 0),
                    "timing": dict.fromkeys(evidence.TIMINGS, 0),
                    "policy": dict.fromkeys(evidence.POLICY_COUNTERS, 0),
                }
                for rank in range(workers)
            ]
        rows = evidence.ranked(
            engine.collective_rpc(
                "fluxmoe_expert_cache_stats",
                kwargs={"synchronize": synchronize, "reset_timing": reset_timing},
            ),
            workers,
        )
        for row in rows:
            evidence.public_stats(row)
            if reset_timing:
                row["timing"] = dict.fromkeys(evidence.TIMINGS, 0)
        return rows

    def validate_initial(self, raw: object, config: shared.PartialRunConfig) -> None:
        if config.arm != "resident":
            evidence.validate_stats(
                raw, self.settings, config.tensor_parallel_size, executed=False
            )

    def validate(
        self, raw: object, config: shared.PartialRunConfig, layers: tuple[int, ...]
    ) -> None:
        evidence.validate_stats(raw, self.settings, config.tensor_parallel_size)

    def deltas(
        self, before: object, after: object, *, expected_workers: int
    ) -> dict[str, Any]:
        return evidence.cache_deltas(before, after, expected_workers=expected_workers)

    def validate_repetition(
        self, diagnostics: dict[str, Any], config: shared.PartialRunConfig
    ) -> None:
        if config.arm == "resident":
            return
        geometry = self.settings["identity"]["geometry"]
        for row in diagnostics["per_rank"]:
            if not all(x > 0 for x in row["forward_counts"]):
                raise ValueError("measured layer forwards missing")
            evidence.validate_demand(
                row, 6 * geometry["hidden_size"] * geometry["intermediate_size"]
            )

    def timing_available(self, diagnostics: dict[str, Any]) -> bool:
        return bool(diagnostics["timing"]["cuda_sample_count"] > 0)

    def before_measurement(self, engine: Any, workers: int) -> None:
        if sorted(engine.collective_rpc("fluxmoe_reset_memory_peaks")) != list(
            range(workers)
        ):
            raise ValueError("memory peak reset worker coverage differs")

    def measurement_fields(self, engine: Any, workers: int) -> dict[str, Any]:
        return {
            "memory": shared._memory(engine, workers),
            "memory_peak_scope": "measured-generate-after-synchronized-reset",
        }


def run_calibration(
    config: shared.PartialRunConfig,
    *,
    project_root: Path,
    run_dir: Path,
    profile_path: Path,
    calibration_count: int,
) -> Path:
    if profile_path.exists():
        raise FileExistsError("calibration profile already exists")
    run_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "status": "running",
        "storage_backend": "native-calibration",
    }
    shared.atomic_json(run_dir / "summary.json", summary)
    try:
        split = split_workload(
            config.dataset_path,
            config.dataset_manifest,
            config.context_length,
            config.batch_size,
            calibration_count,
        )
        identity = model_profile_identity(
            config.model_path, config.tensor_parallel_size
        )
        native = replace(
            config, arm="resident", offload_count=0, batch_size=calibration_count
        )
        shared._configure_environment(native, project_root, ())
        os.environ["FLUXMOE_EXPERT_CALIBRATION"] = "1"
        vllm = importlib.import_module("vllm")
        if vllm.__version__.split("+")[0] != "0.10.2":
            raise ValueError("calibration requires pinned vLLM 0.10.2")
        arguments = shared.engine_arguments(native, None)
        engine = vllm.LLM(**arguments)
        policy = shared._resolved_policy(engine, arguments)
        # Constructor profiling is over before capture starts. No smoke/warmup is captured.
        starts = evidence.ranked(
            engine.collective_rpc(
                "fluxmoe_expert_calibration", kwargs={"action": "start"}
            ),
            config.tensor_parallel_size,
        )
        if any(
            row.get("active") is not True
            or any(row.get(k) != v for k, v in identity.items())
            for row in starts
        ):
            raise ValueError("native calibration start identity differs")
        started = perf_counter()
        try:
            outputs = engine.generate(
                [{"prompt_token_ids": list(p)} for p in split.calibration],
                vllm.SamplingParams(
                    temperature=0.0,
                    min_tokens=config.output_length,
                    max_tokens=config.output_length,
                    ignore_eos=True,
                    seed=config.seed,
                    detokenize=False,
                ),
                use_tqdm=False,
            )
            engine.collective_rpc("fluxmoe_synchronize")
            elapsed = perf_counter() - started
        finally:
            counts = engine.collective_rpc(
                "fluxmoe_expert_calibration", kwargs={"action": "stop"}
            )
        measured = shared.summarize_outputs(
            outputs,
            request_count=calibration_count,
            output_length=config.output_length,
            elapsed_s=elapsed,
        )
        profile = profile_from_workers(counts, identity, split.hashes)
        summary.update(
            {
                "identity": identity,
                "profile_sha256": shared.digest_json(profile.to_dict()),
                "calibration_input_hashes": list(split.hashes),
                "workload": split.evaluation.metadata,
                "capture_scope": "explicit-native-generate-only; excludes-constructor-profiling",
                "rank_profiles": counts,
                "measurement": measured,
                "engine_policy": policy,
                "commit": subprocess.check_output(
                    ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
                ).strip(),
                "status": "complete",
            }
        )
        shared.atomic_json(profile_path, profile.to_dict())
        shared.atomic_json(run_dir / "summary.json", summary)
    except BaseException as error:
        summary.update(status="failed", error_type=type(error).__name__)
        shared.atomic_json(run_dir / "summary.json", summary)
        raise
    return profile_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("point", "calibrate-point"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resident-run", type=Path)
    parser.add_argument("--arm", choices=shared.ARMS, default="resident")
    parser.add_argument("--profile-path", type=Path, required=True)
    parser.add_argument("--resident-ratio", type=float, default=0.5)
    parser.add_argument("--cache-slots", type=int, default=256)
    parser.add_argument(
        "--cache-policy", choices=("decayed-lfu", "lru"), default="decayed-lfu"
    )
    parser.add_argument("--calibration-count", type=int, default=32)
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
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    for flag, default in (
        ("batch-size", 512),
        ("context-length", 4096),
        ("output-length", 256),
        ("warmups", 1),
        ("repetitions", 3),
        ("seed", 20260905),
        ("max-num-seqs", 256),
        ("max-num-batched-tokens", 8192),
        ("timing-samples", 128),
    ):
        parser.add_argument("--" + flag, type=int, default=default)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    if not args.run_dir.resolve().is_relative_to(root) or (
        args.mode == "calibrate-point"
        and not args.profile_path.resolve().is_relative_to(root)
    ):
        parser.error("all outputs must remain inside project-root")
    values = {
        key: value
        for key, value in vars(args).items()
        if key in shared.PartialRunConfig.__dataclass_fields__
    }
    config = shared.PartialRunConfig(**values)
    if args.mode == "calibrate-point":
        run_calibration(
            config,
            project_root=root,
            run_dir=args.run_dir,
            profile_path=args.profile_path,
            calibration_count=args.calibration_count,
        )
    else:
        backend = ExpertBackend(
            config,
            args.profile_path,
            args.resident_ratio,
            args.cache_slots,
            args.cache_policy,
            args.calibration_count,
        )
        shared.run_benchmark(
            config,
            project_root=root,
            run_dir=args.run_dir,
            resident_run=args.resident_run,
            backend=backend,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
