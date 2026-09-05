"""Compulsory hardware tests; skipped CUDA tests can never pass this gate."""

from __future__ import annotations

import argparse
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from flexmoe.bench.partial_runner import atomic_json


def validate_junit(path: Path) -> dict[str, int]:
    tree = ET.parse(path)
    suites = list(tree.getroot().iter("testsuite"))
    totals = {
        key: sum(int(row.attrib[key]) for row in suites)
        for key in ("tests", "skipped", "failures", "errors")
    }
    if totals["tests"] <= 0 or any(
        totals[key] != 0 for key in ("skipped", "failures", "errors")
    ):
        raise ValueError(
            "CUDA gate needs executed tests with zero skips/failures/errors"
        )
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if not args.output.resolve().is_relative_to(root):
        parser.error("preflight output must stay inside project-root")
    if args.output.exists():
        raise FileExistsError("preflight evidence already exists; choose a fresh suite")
    result: dict[str, Any] = {"schema_version": 1, "status": "failed"}
    try:
        from flexmoe.vllm.expert_calibration import model_profile_identity

        # The wrapper's checkpoint/exclusivity preflight checks this same path.
        result["identity"] = model_profile_identity(args.model_path, 4)
        import pytest
        import torch
        import vllm  # type: ignore[import-not-found]

        if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
            raise ValueError("CUDA gate requires four visible GPUs")
        if (
            any(torch.cuda.get_device_capability(i) != (9, 0) for i in range(4))
            or vllm.__version__.split("+")[0] != "0.10.2"
        ):
            raise ValueError("CUDA gate requires H100 and pinned vLLM")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        junit = args.output.with_suffix(".private.xml")
        exit_code = pytest.main(
            [
                str(root / "tests/cuda/test_expert_cache.py"),
                str(root / "tests/cuda/test_partial_runtime.py"),
                "-q",
                "--junitxml=" + str(junit),
            ]
        )
        result.update(validate_junit(junit))
        if exit_code != 0:
            raise ValueError("pytest returned a failure status")
        result.update(
            status="passed",
            commit=subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip(),
        )
    except Exception as error:
        result["error_type"] = type(error).__name__
        atomic_json(args.output, result)
        raise
    atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
