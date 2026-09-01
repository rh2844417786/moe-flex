from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

COMMIT = "01efc7ef781391e744ed08c3292817a773d654e6"


@pytest.mark.integration
def test_patch_applies_to_exact_vllm_checkout() -> None:
    source_value = os.environ.get("VLLM_SOURCE_DIR")
    if source_value is None:
        pytest.skip("set VLLM_SOURCE_DIR to a pristine pinned vLLM checkout")
    source = Path(source_value).resolve()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head == COMMIT

    project_root = Path(__file__).resolve().parents[2]
    patch = project_root / "patches" / "vllm-v0.10.2.patch"
    apply_check = subprocess.run(
        ["git", "apply", "--unidiff-zero", "--check", str(patch)],
        cwd=source,
        check=False,
        capture_output=True,
        text=True,
    )
    if apply_check.returncode == 0:
        return

    subprocess.run(
        [
            "git",
            "apply",
            "--unidiff-zero",
            "--reverse",
            "--check",
            str(patch),
        ],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    touched = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert touched == ["vllm/model_executor/layers/fused_moe/layer.py"]
