"""Evidence-gated H100 cross-hardware support classification."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from statistics import median
from typing import cast

from flexmoe.config import RunStatus
from flexmoe.bench.router_trace import router_probes_match


@dataclass(frozen=True)
class RunEvidence:
    output_tokens_per_second: float
    mapped_bytes: int
    h2d_bytes: int
    decompressed_bytes: int
    output_tokens_match: bool
    router_topk_match: bool
    weights_bit_exact: bool


@dataclass(frozen=True)
class SupportClassification:
    status: RunStatus
    reasons: tuple[str, ...]


def classify_support(
    evidence: RunEvidence,
    stressed_delta: float | None = None,
    delayed_oom: bool = False,
) -> SupportClassification:
    missing: list[str] = []
    if not isfinite(evidence.output_tokens_per_second) or (
        evidence.output_tokens_per_second <= 0
    ):
        missing.append("output_tokens_per_second")
    for field_name in ("mapped_bytes", "h2d_bytes", "decompressed_bytes"):
        if getattr(evidence, field_name) <= 0:
            missing.append(field_name)
    for field_name in (
        "output_tokens_match",
        "router_topk_match",
        "weights_bit_exact",
    ):
        if not getattr(evidence, field_name):
            missing.append(field_name)
    if missing:
        return SupportClassification(status="INCONCLUSIVE", reasons=tuple(missing))

    if delayed_oom:
        return SupportClassification(
            status="SUPPORTED",
            reasons=("FluxMoE delayed the capacity/OOM boundary",),
        )
    if stressed_delta is None or not isfinite(stressed_delta):
        return SupportClassification(
            status="INCONCLUSIVE",
            reasons=("stressed_delta",),
        )
    if stressed_delta > 0:
        return SupportClassification(
            status="SUPPORTED",
            reasons=("capacity-stressed throughput improved",),
        )
    if stressed_delta < 0:
        return SupportClassification(
            status="NOT_SUPPORTED",
            reasons=("repeatable capacity-stressed trend was opposite",),
        )
    return SupportClassification(
        status="MIXED",
        reasons=("complete evidence showed a flat stressed trend",),
    )


def _json_object(path: Path) -> dict[str, object]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(parsed, dict):
        raise TypeError(f"JSON artifact {path} must contain an object")
    return cast(dict[str, object], parsed)


def validate_run_directory(run_dir: Path) -> RunEvidence:
    state_path = run_dir / "state.json"
    state = _json_object(state_path)
    if state.get("status") != "COMPLETE":
        raise ValueError(f"run {run_dir} is not COMPLETE")
    required = (
        "config.json",
        "environment.json",
        "events.jsonl",
        "metrics.json",
        "preflight.json",
        "state.json",
    )
    for filename in required:
        path = run_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required run artifact is missing or empty: {path}")

    metrics = _json_object(run_dir / "metrics.json")
    raw_repetitions = metrics.get("repetitions")
    if not isinstance(raw_repetitions, list) or len(raw_repetitions) != 3:
        raise ValueError("metrics must contain exactly three repetitions")
    throughputs: list[float] = []
    for raw_repetition in raw_repetitions:
        if not isinstance(raw_repetition, dict):
            raise TypeError("repetition metrics must be objects")
        throughput = raw_repetition.get("output_tokens_per_second")
        if type(throughput) not in {int, float}:
            raise ValueError("repetition throughput must be numeric")
        throughputs.append(float(cast(int | float, throughput)))

    raw_counters = metrics.get("mechanism_counters")
    raw_correctness = metrics.get("correctness")
    if not isinstance(raw_counters, dict) or not isinstance(raw_correctness, dict):
        raise TypeError("metrics are missing mechanism or correctness evidence")
    counters = cast(dict[str, object], raw_counters)
    correctness = cast(dict[str, object], raw_correctness)

    def counter(name: str) -> int:
        value = counters.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"mechanism counter {name} must be non-negative")
        return value

    def exact_bool(name: str) -> bool:
        value = correctness.get(name)
        if type(value) is not bool:
            raise ValueError(f"correctness field {name} must be a boolean")
        return value

    evidence = RunEvidence(
        output_tokens_per_second=median(throughputs),
        mapped_bytes=counter("mapped_bytes"),
        h2d_bytes=counter("h2d_bytes"),
        decompressed_bytes=counter("decompressed_bytes"),
        output_tokens_match=exact_bool("output_tokens_match"),
        router_topk_match=exact_bool("router_topk_match"),
        weights_bit_exact=exact_bool("weights_bit_exact"),
    )
    delegated_value = metrics.get("correctness_evidence")
    delegated = isinstance(delegated_value, str)
    if delegated:
        delegated_path = Path(cast(str, delegated_value)).resolve()
        if delegated_path == run_dir.resolve():
            raise ValueError("correctness evidence cannot reference the same run")
        delegated_evidence = validate_run_directory(delegated_path)
        if not (
            delegated_evidence.router_topk_match
            and delegated_evidence.weights_bit_exact
        ):
            raise ValueError("delegated correctness evidence is incomplete")
        current_config = _json_object(run_dir / "config.json")
        delegated_config = _json_object(delegated_path / "config.json")
        for field_name in (
            "git_sha",
            "model_path",
            "dataset_sha256",
            "tensor_parallel_size",
        ):
            if current_config.get(field_name) != delegated_config.get(field_name):
                raise ValueError(
                    f"delegated correctness evidence differs in {field_name}"
                )

    if evidence.weights_bit_exact and not delegated:
        expected_weights = counter("weights_expected")
        verified_weights = counter("weights_verified")
        if expected_weights <= 0 or verified_weights != expected_weights:
            raise ValueError("weight verification counters do not prove bit parity")

    raw_router = metrics.get("router_trace")
    if not delegated:
        if not isinstance(raw_router, dict) or not raw_router:
            raise ValueError("metrics contain no router trace manifest")
        router_manifest = cast(dict[str, object], raw_router)
        for filename, raw_entry in router_manifest.items():
            if not isinstance(raw_entry, dict):
                raise TypeError("router trace manifest entries must be objects")
            entry = cast(dict[str, object], raw_entry)
            trace_path = run_dir / "router" / filename
            payload = trace_path.read_bytes()
            if sha256(payload).hexdigest() != entry.get("sha256"):
                raise ValueError(f"router trace SHA256 mismatch: {trace_path}")
            if len(payload.splitlines()) != entry.get("line_count"):
                raise ValueError(f"router trace line count mismatch: {trace_path}")
            probe_line_count = entry.get("probe_line_count")
            lines = payload.splitlines(keepends=True)
            if (
                type(probe_line_count) is not int
                or not 0 < probe_line_count <= len(lines)
            ):
                raise ValueError(f"invalid router probe line count: {trace_path}")
            probe_payload = b"".join(lines[:probe_line_count])
            if sha256(probe_payload).hexdigest() != entry.get("probe_sha256"):
                raise ValueError(f"router probe SHA256 mismatch: {trace_path}")

    reference_value = metrics.get("reference_run")
    failure_value = metrics.get("resident_failure_run")
    if isinstance(reference_value, str):
        reference_metrics = _json_object(Path(reference_value) / "metrics.json")
        current_repetitions = cast(list[object], raw_repetitions)
        reference_repetitions = reference_metrics.get("repetitions")
        if not isinstance(reference_repetitions, list):
            raise TypeError("resident reference repetitions must be a list")

        def output_tokens(repetitions: list[object]) -> list[object]:
            tokens: list[object] = []
            for repetition in repetitions:
                if not isinstance(repetition, dict):
                    raise TypeError("repetition metrics must be objects")
                tokens.append(repetition.get("output_token_ids"))
            return tokens

        if output_tokens(current_repetitions) != output_tokens(
            reference_repetitions
        ):
            raise ValueError("greedy output token IDs differ from resident reference")
        if not delegated and not router_probes_match(
            reference_metrics.get("router_trace"), raw_router
        ):
            raise ValueError("router Top-k trace differs from resident reference")
    elif isinstance(failure_value, str):
        failure = _json_object(Path(failure_value) / "error.json")
        message = str(failure.get("message", "")).lower()
        if "out of memory" not in message and "oom" not in message:
            raise ValueError("resident failure evidence is not an OOM")
        if metrics.get("delayed_oom") is not True or not delegated:
            raise ValueError("delayed OOM requires delegated correctness evidence")
    else:
        raise TypeError("validated FluxMoE run has no resident reference or OOM")
    classification = classify_support(evidence, stressed_delta=0.0)
    if classification.status == "INCONCLUSIVE":
        raise ValueError(
            "run evidence is incomplete: " + ", ".join(classification.reasons)
        )
    return evidence


def latest_complete_run(root: Path) -> Path:
    if not root.is_dir():
        raise ValueError(f"runs root does not exist: {root}")
    for candidate in sorted(root.iterdir(), reverse=True):
        if not candidate.is_dir():
            continue
        try:
            validate_run_directory(candidate)
        except ValueError:
            continue
        return candidate
    raise ValueError(f"no complete validated runs exist under {root}")


def build_report(run_dir: Path, output: Path, figures: Path) -> None:
    evidence = validate_run_directory(run_dir)
    metrics = _json_object(run_dir / "metrics.json")
    raw_delta = metrics.get("stressed_delta")
    stressed_delta = (
        float(cast(int | float, raw_delta))
        if type(raw_delta) in {int, float}
        else None
    )
    delayed_oom = metrics.get("delayed_oom") is True
    classification = classify_support(evidence, stressed_delta, delayed_oom)
    figures.mkdir(parents=True, exist_ok=True)
    evidence_path = figures / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "evidence": asdict(evidence),
                "classification": asdict(classification),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FluxMoE H100 跨硬件复现报告",
        "",
        f"状态：`{classification.status}`",
        "",
        f"- output tokens/s（3 次中位数）：{evidence.output_tokens_per_second:.6f}",
        f"- VMM mapped bytes：{evidence.mapped_bytes}",
        f"- HtoD bytes：{evidence.h2d_bytes}",
        f"- decompressed bytes：{evidence.decompressed_bytes}",
        f"- 原因：{'; '.join(classification.reasons)}",
        "",
        f"<!-- artifact: {run_dir.resolve()} -->",
        f"<!-- artifact: {evidence_path.resolve()} -->",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def validate_report(path: Path) -> tuple[Path, ...]:
    text = path.read_text(encoding="utf-8")
    artifacts = tuple(
        Path(line.removeprefix("<!-- artifact: ").removesuffix(" -->"))
        for line in text.splitlines()
        if line.startswith("<!-- artifact: ") and line.endswith(" -->")
    )
    if not artifacts:
        raise ValueError("report contains no artifact references")
    for artifact in artifacts:
        if not artifact.exists():
            raise ValueError(f"report references missing artifact {artifact}")
    return artifacts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    latest_parser = subparsers.add_parser("latest")
    latest_parser.add_argument("--root", type=Path, default=Path("runs"))
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("run_dir", type=Path)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("run_dir", type=Path)
    build_parser.add_argument("--output", required=True, type=Path)
    build_parser.add_argument("--figures", required=True, type=Path)
    report_parser = subparsers.add_parser("validate-report")
    report_parser.add_argument("report", type=Path)
    arguments = parser.parse_args(argv)

    if arguments.command == "latest":
        print(latest_complete_run(arguments.root))
    elif arguments.command == "validate":
        validate_run_directory(arguments.run_dir)
    elif arguments.command == "build":
        build_report(arguments.run_dir, arguments.output, arguments.figures)
    elif arguments.command == "validate-report":
        validate_report(arguments.report)
    else:
        raise AssertionError(f"unhandled command {arguments.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RunEvidence",
    "SupportClassification",
    "build_report",
    "classify_support",
    "latest_complete_run",
    "validate_report",
    "validate_run_directory",
]
