"""Finite, resumable, standard-library host orchestration of H100 experiments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if not __package__:
    spec = importlib.util.spec_from_file_location(
        "_experiment_offline",
        Path(__file__).with_name("__init__.py"),
        submodule_search_locations=[str(Path(__file__).parent)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("analysis package unavailable")
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    __package__ = spec.name

from . import decode_report, decode_suite, parallel_report
from .decode_suite import digest, integer, mapping, number, ranked, read_artifact
from .experiment_state import (
    ExperimentState,
    PointExecutor,
    SafetyStop,
    atomic_json,
    bounded,
    file_digest,
    read_json,
    safe_name,
)

SERVER_ROOT = Path("/home/jovyan/wangtonghan/moe-flex")
PARALLEL = ("tp4", "ep4-dp4", "ep4-dp2")
BATCHES = (1, 16, 50, 100, 200, 500, 1000)
PUBLIC_FILES = (
    "report.json",
    "report.csv",
    "report.md",
    "kv-throughput.svg",
    "batch-coverage.svg",
    "miss-service.svg",
)


def gpu_ids(value: str) -> str:
    parts = value.split(",")
    if (
        len(parts) != 4
        or any(not x.isdigit() or str(int(x)) != x for x in parts)
        or len(set(parts)) != 4
    ):
        raise ValueError("GPU_IDS must explicitly name four distinct numeric GPUs")
    return value


def host_gpu_inventory(ids: str) -> list[dict[str, int | str]]:
    """Bind selected host indices to physical UUIDs without creating CUDA state."""
    selected = [int(value) for value in gpu_ids(ids).split(",")]
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader",
            f"--id={ids}",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    inventory: dict[int, str] = {}
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if (
            len(fields) != 2
            or re.fullmatch(r"0|[1-9][0-9]*", fields[0]) is None
            or re.fullmatch(
                r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
                fields[1],
            )
            is None
        ):
            raise ValueError("selected physical GPU inventory is malformed")
        index, physical_uuid = int(fields[0]), "GPU-" + fields[1][4:].lower()
        if (
            index not in selected
            or index in inventory
            or physical_uuid in inventory.values()
        ):
            raise ValueError("selected physical GPU inventory is ambiguous")
        inventory[index] = physical_uuid
    if set(inventory) != set(selected):
        raise ValueError("selected physical GPU inventory is incomplete")
    return [{"index": index, "uuid": inventory[index]} for index in selected]


def select_kv(raw: Mapping[str, Any], sha: str) -> tuple[int, int] | None:
    """Use only this native run's four-rank actual allocation and reserve gates."""
    try:
        c = mapping(raw.get("contract"))
        if raw.get("mode") != "native" or c.get("commit") != sha:
            return None
        if not decode_suite.validate_run(raw)["timing_eligible"]:
            return None
        if not integer(c.get("physical_safety_reserve_bytes"), 2_000_000_000):
            return None
        allocations, block_bytes = [], []
        for row in ranked(raw.get("memory")):
            actual, blocks = row["kv_cache_allocated_bytes"], row["num_gpu_blocks"]
            if not integer(actual, 1) or not integer(blocks, 2) or actual % blocks:
                return None
            allocations.append(actual)
            block_bytes.append(actual // blocks)
        if len(set(block_bytes)) != 1:
            return None
        alignment = block_bytes[0]
        large = min(allocations) // alignment * alignment
        small = (large // 2) // alignment * alignment
        return (small, large) if 0 < small < large else None
    except (ValueError, TypeError, KeyError):
        return None


def reached_summary(
    raw: Mapping[str, Any], sha: str, context: int, batch: int | None
) -> bool:
    try:
        c = mapping(raw.get("contract"))
        if (
            raw.get("profile") is not True
            or c.get("commit") != sha
            or c.get("context_length") != context
            or c.get("target_batch") != batch
            or c.get("capture_steps") != 256
            or c.get("min_capture_steps") != 64
            or c.get("repetitions_requested") != 3
        ):
            return False
        if decode_suite.validate_run(raw)["reasons"] != ["instrumented-timing"]:
            return False
        for rep in raw["repetitions"]:
            for row in ranked(rep["worker_observations"]):
                if not integer(row.get("captured_steps"), 64):
                    return False
                if batch is not None and (
                    row.get("coverage_status") != "complete"
                    or not integer(
                        mapping(row.get("decode_batch_step_counts")).get(str(batch)), 64
                    )
                ):
                    return False
        return True
    except (ValueError, TypeError, KeyError):
        return False


def representatives(values: Sequence[int]) -> list[int]:
    ordered = sorted(set(values))
    return (
        sorted({ordered[0], ordered[len(ordered) // 2], ordered[-1]}) if ordered else []
    )


def archive_public(root: Path, destination: Path, files: Sequence[Path]) -> None:
    bounded(root, destination)
    if destination.exists():
        raise FileExistsError(destination)
    checked = [bounded(root, path) for path in files]
    if len(set(checked)) != len(checked) or any(not path.is_file() for path in checked):
        raise ValueError("archive requires distinct allowlisted regular files")
    with tarfile.open(destination, "x:gz", dereference=False) as archive:
        for path in checked:
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)


class Suite:
    def __init__(self, state: ExperimentState, executor: PointExecutor):
        self.state, self.executor = state, executor
        self.root, self.config = state.root, state.config
        self.sha = str(self.config["sha"])

    def container(self, *argv: str) -> list[str]:
        return ["bash", "scripts/server/run_container.sh", *argv]

    def preflight(self, output: Path) -> list[str]:
        return self.container(
            "python3",
            "-m",
            "flexmoe.runtime.preflight",
            "check",
            "--project-root",
            str(self.root),
            "--model-path",
            self.config["model_path"],
            "--gpu-ids",
            "0,1,2,3",
            "--output",
            str(output),
        )

    def latest(self, key: str) -> dict[str, Any]:
        attempts = self.state.data["points"].get(key, {}).get("attempts", [])
        return dict(attempts[-1]) if attempts else {}

    def successful(self, key: str) -> bool:
        return bool(self.state.data["points"].get(key, {}).get("status") == "complete")

    def decide(self, key: str, proposal: list[int] | None) -> list[int] | None:
        decisions = self.state.data.setdefault("decisions", {})
        if decisions.get(key) is None:
            decisions[key] = proposal
            self.state.save()
        value = decisions[key]
        return list(value) if isinstance(value, list) else None

    def corpus(self) -> dict[str, Any]:
        return read_json(Path(self.latest("corpus")["artifacts"][0]))

    def verify_identity(
        self,
        raw: Mapping[str, Any],
        context: int,
        count: int,
        *,
        calibration: bool = False,
    ) -> bool:
        corpus = self.corpus()[str(context)]
        c = mapping(raw.get("workload" if calibration else "contract"))
        same_source = all(
            c.get(key) == corpus.get(key)
            for key in ("dataset_sha256", "dataset_manifest_sha256")
        )
        if calibration:
            return bool(
                raw.get("commit") == self.sha
                and same_source
                and raw.get("calibration_input_hashes")
                == corpus["calibration_input_hashes"]
                and c.get("input_sha256") == corpus["evaluation_input_sha256"]
                and c.get("unique_selected_request_count") == count
                and c.get("repeated_request_count") == 0
            )
        return bool(
            c.get("commit") == self.sha
            and same_source
            and c.get("calibration_input_hashes") == corpus["calibration_input_hashes"]
            and c.get("selected_input_hashes")
            == corpus["selected_input_hashes"][:count]
        )

    def validate(self, attempt: dict[str, Any]) -> bool:
        kind = attempt["kind"]
        if kind == "build":
            contents = Path(attempt["artifacts"][0]).read_text().splitlines()
            return f"GIT_SHA={self.sha}" in contents
        if kind == "corpus":
            raw = read_json(Path(attempt["artifacts"][0]))
            return all(
                mapping(raw.get(str(context))).get("no_repetition") is True
                and raw[str(context)]["dataset_sha256"] == self.config["dataset_sha256"]
                and raw[str(context)]["dataset_manifest_sha256"]
                == self.config["manifest_sha256"]
                and len(raw[str(context)]["selected_input_hashes"]) == 1024
                and len(set(raw[str(context)]["selected_input_hashes"])) == 1024
                for context in (1024, 4096)
            )
        source = bounded(self.root, Path(attempt["source"]))
        raw = read_json(bounded(self.root, source / "summary.json"))
        if kind == "parallel":
            return (
                parallel_report.summarize_parallel(source)["status"] == "complete"
                and mapping(raw.get("contract")).get("commit") == self.sha
                and raw.get("config") == attempt["parallel_config"]
                and self.verify_identity(raw, 1024, 1024)
            )
        launcher = read_json(bounded(self.root, source / "launcher.json"))
        if launcher.get("status") != "complete" or launcher.get("exit_code") != 0:
            return False
        context, count = attempt["context"], attempt["count"]
        if not self.verify_identity(
            raw, context, count, calibration=kind == "calibrate"
        ):
            return False
        if kind == "calibrate":
            profile = read_json(bounded(self.root, source / "profile.json"))
            identity = mapping(raw.get("identity"))
            geometry = mapping(profile.get("geometry"))
            counts, forwards = profile.get("counts"), profile.get("forward_counts")
            return (
                raw.get("status") == "complete"
                and raw.get("storage_backend") == "native-calibration"
                and profile.get("schema_version") == 1
                and profile.get("tensor_parallel_size") == 4
                and raw.get("profile_sha256") == digest(profile)
                and profile.get("calibration_input_hashes")
                == self.corpus()[str(context)]["calibration_input_hashes"]
                and all(
                    profile.get(k) == identity.get(k)
                    for k in (
                        "geometry",
                        "model_identity_sha256",
                        "model_config_sha256",
                        "tensor_parallel_size",
                    )
                )
                and isinstance(counts, list)
                and len(counts) == geometry.get("total_layers")
                and all(
                    isinstance(row, list)
                    and len(row) == geometry.get("num_experts")
                    and all(number(n) for n in row)
                    for row in counts
                )
                and isinstance(forwards, list)
                and len(forwards) == len(counts)
                and all(integer(n, 1) for n in forwards)
            )
        c = mapping(raw.get("contract"))
        profile = read_json(Path(attempt["profile_path"]))
        if mapping(c.get("expert_cache")).get("profile_sha256") != digest(profile):
            return False
        if not attempt["profiled"]:
            return bool(decode_suite.validate_run(raw)["timing_eligible"])
        if not reached_summary(raw, self.sha, context, attempt["batch"]):
            return False
        # Complete 4-rank x 3-repeat capture files are required, not only summary flags.
        captures = []
        for rep in range(3):
            for rank in range(4):
                path = bounded(
                    self.root, source / f"decode-rep-{rep:03d}-rank-{rank}.json.gz"
                )
                capture = read_artifact(path)
                if capture.get("rank") != rank or capture.get("repetition") != rep:
                    return False
                item = decode_report.capture_summary(capture, raw)
                if item["activation_status"] != "complete":
                    return False
                if attempt["batch"] is not None and not any(
                    row.get("target_batch") == attempt["batch"]
                    and row.get("status") == "complete"
                    for row in item["activation_summaries"]
                ):
                    return False
                captures.append(str(path))
        attempt["artifacts"] = list(dict.fromkeys([*attempt["artifacts"], *captures]))
        return True

    def decode(
        self,
        key: str,
        stage: str,
        index: int = 0,
        *,
        context: int = 1024,
        batches: Sequence[int] | None = None,
        kv: Sequence[int] = (),
    ) -> None:
        calibration = stage == "baseline" and index in (0, 1)
        profile = (
            None
            if calibration
            else str(Path(self.latest(f"cal-{context}")["source"]) / "profile.json")
        )

        def factory(directory: Path, child_id: str) -> dict[str, Any]:
            plan = decode_suite.build_plan(
                stage=stage,
                run_id=child_id,
                model_path=self.config["model_path"],
                dataset_path=self.config["dataset_path"],
                dataset_manifest=self.config["dataset_manifest"],
                context_length=context,
                profile_path=profile,
                kv_bytes=kv,
                batches=batches,
                timeout_s=self.config["timeout_s"],
            )
            argv = plan["commands"][index]["argv"]
            run_id = argv[argv.index("--run-id") + 1]
            source = bounded(self.root, self.root / "runs/decode-mechanism" / run_id)
            # Refuse all wrapper-owned outputs before spawning, including abandoned paths.
            for path in (
                source,
                source.with_name(source.name + "-launcher"),
                self.root / "docs/results" / f"decode-mechanism-{run_id}",
            ):
                bounded(self.root, path)
                if path.exists():
                    raise FileExistsError(path)
            profiled = "--profile" in argv
            artifacts = [str(source / "summary.json"), str(source / "launcher.json")]
            if calibration:
                artifacts.append(str(source / "profile.json"))
            return {
                "argv": argv,
                "artifacts": artifacts,
                "kind": "calibrate" if calibration else "decode",
                "source": str(source),
                "context": int(argv[argv.index("--context-length") + 1]),
                "count": int(argv[argv.index("--batch-size") + 1]),
                "profiled": profiled,
                "profile_path": profile,
                "batch": int(argv[argv.index("--target-batch") + 1])
                if "--target-batch" in argv
                else None,
            }

        self.executor.run(
            key,
            factory,
            self.validate,
            preflight=self.preflight,
            inputs={
                "stage": stage,
                "index": index,
                "context": context,
                "batches": list(batches) if batches is not None else None,
                "kv": list(kv),
                "profile_sha256": file_digest(Path(profile)) if profile else None,
            },
        )
        # The wrapper's additional preflight can observe a late foreign occupant.
        latest = self.latest(key)
        if latest.get("status") != "complete" and latest.get("source"):
            check = Path(latest["source"] + "-launcher") / "preflight.json"
            if (
                check.exists()
                and read_json(bounded(self.root, check)).get("ok") is not True
            ):
                raise SafetyStop("preflight-failed")

    def pipeline(self) -> None:
        self.executor.recover()
        self.executor.run(
            "build",
            lambda d, i: {
                "argv": ["bash", "scripts/server/build.sh"],
                "artifacts": [str(self.root / "build/image.env")],
                "kind": "build",
            },
            self.validate,
        )
        if not self.successful("build"):
            self.state.skip("corpus", "build-unavailable")
            return
        corpus_code = (
            "import json,sys; from pathlib import Path; "
            "from flexmoe.datasets.decode_corpus import load_unique_workload; "
            "from flexmoe.bench.partial_runner import digest_json; "
            "loaded={c:load_unique_workload(Path(sys.argv[1]),Path(sys.argv[2]),context_length=c,request_count=1024,calibration_count=32) for c in (1024,4096)}; "
            "rows={str(c):dict(v[2],evaluation_input_sha256=digest_json(v[0])) for c,v in loaded.items()}; "
            "Path(sys.argv[3]).write_text(json.dumps(rows))"
        )
        self.executor.run(
            "corpus",
            lambda d, i: {
                "argv": self.container(
                    "python3",
                    "-c",
                    corpus_code,
                    self.config["dataset_path"],
                    self.config["dataset_manifest"],
                    str(d / "corpus.json"),
                ),
                "artifacts": [str(d / "corpus.json")],
                "kind": "corpus",
            },
            self.validate,
            preflight=self.preflight,
        )
        if not self.successful("corpus"):
            for key in (
                *[f"parallel-{x}" for x in PARALLEL],
                "cal-1024",
                "cal-4096",
                "native",
                "coverage-1024",
                "coverage-4096",
                "offload",
                "kv",
                "overhead",
            ):
                self.state.skip(key, "corpus-unavailable")
            return
        for config in PARALLEL:

            def factory(d: Path, i: str, config: str = config) -> dict[str, Any]:
                source = d / config / "parallel-run"
                source.parent.mkdir()
                return {
                    "argv": self.container(
                        "python3",
                        "-m",
                        "flexmoe.bench.parallel_runner",
                        "--config",
                        config,
                        "--run-dir",
                        str(source),
                        "--project-root",
                        str(self.root),
                        "--model-path",
                        self.config["model_path"],
                        "--dataset-path",
                        self.config["dataset_path"],
                        "--dataset-manifest",
                        self.config["dataset_manifest"],
                        "--timeout-s",
                        str(self.config["timeout_s"]),
                    ),
                    "artifacts": [str(source / "summary.json")],
                    "kind": "parallel",
                    "source": str(source),
                    "parallel_config": config,
                }

            self.executor.run(
                f"parallel-{config}", factory, self.validate, preflight=self.preflight
            )
        self.decode("cal-1024", "baseline", 0)
        self.decode("cal-4096", "baseline", 1, context=4096)
        if not self.successful("cal-1024"):
            for key in (
                "native",
                "coverage-1024",
                "coverage-4096",
                "offload",
                "kv",
                "overhead",
            ):
                self.state.skip(key, "calibration-unavailable")
            return
        self.decode("native", "baseline", 2)
        reached = []
        for batch in BATCHES:
            key = f"coverage-1024-{batch}"
            self.decode(key, "coverage", batches=[batch])
            if self.successful(key):
                reached.append(batch)
        selected = self.decide("representatives", representatives(reached)) or []
        # A previously selected point that is now invalid cannot support later work.
        selected = [batch for batch in selected if batch in reached]
        self.state.data["decisions"]["kv_policy"] = (
            "half-and-native-observed-capacity; aligned; no-full-freed-space-exploitation"
        )
        self.state.save()
        if selected and self.successful("cal-4096"):
            for batch in selected:
                self.decode(
                    f"coverage-4096-{batch}", "coverage", context=4096, batches=[batch]
                )
        else:
            self.state.skip(
                "coverage-4096",
                "no-reached-batch" if not selected else "calibration-unavailable",
            )
        native = self.latest("native")
        observed = (
            select_kv(read_json(Path(native["source"]) / "summary.json"), self.sha)
            if self.successful("native")
            else None
        )
        kv = self.decide("kv", list(observed) if observed else None)
        if (
            observed is None
            or kv is not None
            and (kv[0] > observed[0] or kv[1] > observed[1])
        ):
            kv = None
        if kv is None:
            for key in ("offload", "kv", "overhead"):
                self.state.skip(key, "native-memory-unavailable")
            return
        if selected:
            for batch in selected:
                for index in range(3):
                    self.decode(
                        f"offload-{batch}-{index}",
                        "offload",
                        index,
                        batches=[batch],
                        kv=[kv[0]],
                    )
        else:
            self.state.skip("offload", "no-reached-batch")
        for index, arm in enumerate(("a", "b", "c")):
            self.decode(f"kv-{arm}", "kv", index, kv=kv)
        for index in range(2):
            self.decode(f"overhead-{index}", "overhead", index, kv=[kv[0]])

    def export(self) -> Path:
        public = bounded(
            self.root,
            self.root / "docs/results" / f"decode-mechanism-suite-{self.state.run_id}",
        )
        public.mkdir(parents=True, exist_ok=True)
        number_ = 1
        while (public / f"report-{number_:03d}").exists():
            number_ += 1
        output = bounded(self.root, public / f"report-{number_:03d}")
        output.mkdir()
        files: list[Path] = []
        points = []
        for key, point in self.state.data["points"].items():
            attempts = point.get("attempts", [])
            row = {
                "point": key,
                "status": point.get("status", "unavailable"),
                "attempts": len(attempts),
            }
            if attempts:
                row["reason"] = attempts[-1].get("reason")
            elif point.get("reason"):
                row["reason"] = point["reason"]
            points.append(row)
            for attempt in attempts:
                if attempt.get("kind") != "decode":
                    continue
                source = bounded(self.root, Path(attempt["source"]))
                destination = bounded(
                    self.root, output / f"{key}-attempt-{attempt['number']:03d}"
                )
                try:
                    result = decode_report.export_report(source, destination)
                    result.pop("cohort_sha256", None)
                    atomic_json(self.root, destination / "report.json", result)
                    files.extend(
                        destination / name
                        for name in PUBLIC_FILES
                        if (destination / name).is_file()
                    )
                except (OSError, ValueError, TypeError, KeyError):
                    row["report_status"] = "unavailable"
        sources = [
            Path(
                self.latest(f"parallel-{config}").get(
                    "source", self.state.directory / "not-run" / config
                )
            )
            for config in PARALLEL
        ]
        parallel = parallel_report.export_parallel(sources, output)
        files += [output / "parallel-summary.json", output / "parallel-summary.md"]
        comparison: dict[str, Any] = {"status": "unavailable"}
        if all(self.successful(f"kv-{arm}") for arm in ("a", "b", "c")):
            try:
                abc = [
                    Path(self.latest(f"kv-{arm}")["source"]) for arm in ("a", "b", "c")
                ]
                native = (
                    [Path(self.latest("native")["source"])]
                    if self.successful("native")
                    else []
                )
                destination = output / "abc"
                report = decode_report.export_report(
                    abc[0], destination, comparisons=[*abc[1:], *native]
                )
                report.pop("cohort_sha256", None)
                atomic_json(self.root, destination / "report.json", report)
                comparison = report["comparison"]
                files.extend(
                    destination / name
                    for name in PUBLIC_FILES
                    if (destination / name).is_file()
                )
            except (OSError, ValueError, TypeError, KeyError):
                comparison = {"status": "unavailable"}
        summary = {
            "schema_version": 1,
            "status": self.state.data["status"],
            "points": points,
            "decisions": self.state.data.get("decisions", {}),
            "parallel_comparison": parallel,
            "tp4_abc": comparison,
            "ep_offload": "unsupported",
            "diagnostic_only": True,
            "formal_offload_gain": False,
            "deployment_gain_proven": False,
        }
        atomic_json(self.root, output / "summary.json", summary)
        text = [
            "# 一键实验汇总",
            "",
            "三组全常驻并行对比与 TP4 A/B/C 分开解释。EP 卸载不支持。",
            "",
            "本实验是诊断证据；不保证每点成功，也不保证卸载加速。详细采集是 instrumented，不进入正式吞吐。",
            "",
            "KV 小点为当前 native 实际容量的一半并按实际字节 block 对齐；大点不超过四卡最小已观测容量；未尝试吃满释放权重空间。",
            "",
            f"并行比较：{parallel['status']}；TP4 A/B/C：{comparison['status']}。",
            "",
            "并行吞吐使用所有 DP 的总输出 / 父进程同轮墙钟；可用请求延迟见 parallel-summary，缺失保持 null。",
            "",
            "| 实验点 | 状态 | 尝试数 | 原因 |",
            "|---|---|---:|---|",
        ]
        text += [
            f"| {row['point']} | {row['status']} | {row['attempts']} | {row.get('reason') or ''} |"
            for row in points
        ]
        (output / "summary.md").write_text("\n".join(text) + "\n")
        files += [output / "summary.json", output / "summary.md"]
        archive = self.state.directory / f"public-report-{number_:03d}.tar.gz"
        archive_public(self.root, archive, files)
        self.state.data["public_report"] = str(output)
        self.state.data["public_archive"] = str(archive)
        self.state.data["public_allowlist"] = [
            str(path.relative_to(self.root)) for path in files
        ]
        self.state.save()
        return archive


def interrupted(signum: int, frame: object) -> None:
    raise InterruptedError(f"signal {signum}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status"):
        sub = subs.add_parser(name, allow_abbrev=False)
        sub.add_argument("--run-id", required=True)
        sub.add_argument(
            "--project-root", type=Path, default=Path(__file__).resolve().parents[3]
        )
        if name == "run":
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--retry-failed", action="store_true")
            sub.add_argument("--dry-run", action="store_true")
            sub.add_argument("--timeout-s", type=int, default=7200)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    try:
        safe_name(args.run_id)
        if args.command == "status":
            checkpoint = read_json(
                bounded(
                    root, root / "runs/experiment-suite" / args.run_id / "state.json"
                )
            )
            print(
                json.dumps(
                    {
                        "status": checkpoint["status"],
                        "points": {
                            k: v.get("status") for k, v in checkpoint["points"].items()
                        },
                        "report": checkpoint.get("public_report"),
                        "archive": checkpoint.get("public_archive"),
                    },
                    indent=2,
                )
            )
            return 0
        ids = gpu_ids(os.environ.get("GPU_IDS", ""))
        if not 1 <= args.timeout_s <= 86400 or args.retry_failed and not args.resume:
            raise ValueError("positive bounded timeout; retry-failed requires resume")
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "run_id": args.run_id,
                        "gpu_ids": ids,
                        "timeout_s": args.timeout_s,
                        "parallel": list(PARALLEL),
                        "contexts": [1024, 4096],
                        "coverage_1024": BATCHES,
                        "adaptive": "up to 3 reached batches; 3 offload configs per batch; A/B/C; overhead pair",
                        "ep_offload": "unsupported",
                        "executes_gpu": False,
                        "baseline": decode_suite.build_plan(
                            stage="baseline",
                            run_id=args.run_id,
                            timeout_s=args.timeout_s,
                        ),
                    },
                    indent=2,
                )
            )
            return 0
        if root != SERVER_ROOT:
            raise ValueError("execution is restricted to the authorized server project")
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                ".",
                ":(exclude)docs/results/decode-mechanism-*",
            ],
            cwd=root,
            text=True,
        )
        if dirty:
            raise ValueError(
                "commit code before running; public decode reports alone are exempt"
            )
        config = {
            "sha": sha,
            "gpu_ids": ids,
            "gpu_inventory": host_gpu_inventory(ids),
            "timeout_s": args.timeout_s,
            "grace_s": 60,
            "model_path": decode_suite.MODEL,
            "dataset_path": decode_suite.DATA,
            "dataset_manifest": decode_suite.MANIFEST,
            "policy_version": 1,
            "dataset_sha256": file_digest(root / decode_suite.DATA),
            "manifest_sha256": file_digest(root / decode_suite.MANIFEST),
        }
        with ExperimentState(root, args.run_id, config, resume=args.resume) as state:
            executor = PointExecutor(
                state, timeout_s=args.timeout_s, retry_failed=args.retry_failed
            )
            suite = Suite(state, executor)
            old = {
                sig: signal.signal(sig, interrupted)
                for sig in (signal.SIGINT, signal.SIGTERM)
            }
            try:
                suite.pipeline()
                state.data["status"] = (
                    "complete"
                    if all(
                        x.get("status") == "complete"
                        for x in state.data["points"].values()
                    )
                    else "partial"
                )
            except SafetyStop:
                state.data["status"] = "safety-stopped"
            except (InterruptedError, KeyboardInterrupt):
                state.data["status"] = "interrupted"
            finally:
                for sig, handler in old.items():
                    signal.signal(sig, handler)
                state.save()
                archive = suite.export()
                print(f"Report: {state.data['public_report']}\nArchive: {archive}")
                print(
                    f"Resume: GPU_IDS={ids} bash scripts/server/run_all_experiments.sh --run-id {shlex.quote(args.run_id)} --resume --timeout-s {args.timeout_s}"
                )
            return 0 if state.data["status"] == "complete" else 1
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"experiment suite refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
