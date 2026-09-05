from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.integration
def test_applied_patch_compiles_and_cache_hook_keeps_native_topk(tmp_path, monkeypatch):
    """Execute only patched method bodies: no vLLM/CUDA claim on CPU."""
    import torch

    from flexmoe.vllm import bridge

    source_value = os.environ.get("VLLM_SOURCE_DIR")
    if source_value is None:
        pytest.skip("set VLLM_SOURCE_DIR to pristine pinned vLLM")
    project = Path(__file__).resolve().parents[2]
    relative = Path("vllm/model_executor/layers/fused_moe/layer.py")
    original = Path(source_value).resolve() / relative
    lock = json.loads((project / "third_party/vllm.lock.json").read_text())
    assert sha256(original.read_bytes()).hexdigest() == lock["upstream_layer_sha256"]
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    shutil.copyfile(original, target)
    subprocess.run(
        ["git", "apply", "--unidiff-zero", str(project / "patches/vllm-v0.10.2.patch")],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    assert sha256(target.read_bytes()).hexdigest() == lock["patched_layer_sha256"]
    tree = ast.parse(target.read_text())
    compile(tree, str(target), "exec")

    def method(class_name):
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
        fn = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "forward_cuda"
        )
        fn.returns = None
        for arg in fn.args.args + fn.args.kwonlyargs:
            arg.annotation = None
        fn.decorator_list = []
        module = ast.Module(body=[fn], type_ignores=[])
        namespace = {
            "os": os,
            "FusedMoE": SimpleNamespace(select_experts=lambda **kw: (weights, ids)),
        }
        exec(compile(module, str(target), "exec"), namespace)  # noqa: S102 - SHA-verified pinned source
        return namespace["forward_cuda"]

    ids = torch.tensor([[3, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.75, 0.25]])
    hidden = torch.zeros((1, 4), dtype=torch.bfloat16)
    layer = SimpleNamespace(
        layer_name="model.layers.0.mlp.experts", w13_weight=None, w2_weight=None
    )
    calls = []

    def record(name, actual_ids):
        assert actual_ids is ids
        calls.append("calibration")

    def cache(name, x, actual_weights, actual_ids, **kwargs):
        assert actual_ids is ids and actual_weights is weights and x is hidden
        calls.append("cache")
        return x + 1

    def legacy(*args):
        raise AssertionError("expert-cache entered legacy registry")

    monkeypatch.setattr(bridge, "record_calibration", record)
    monkeypatch.setattr(bridge, "expert_cache_forward", cache)
    monkeypatch.setattr(bridge, "before_forward", legacy)
    monkeypatch.setenv("FLUXMOE_ENABLE", "1")
    monkeypatch.setenv("FLUXMOE_STORAGE_MODE", "expert-cache")
    monkeypatch.setenv("FLUXMOE_EXPERT_CALIBRATION", "1")
    monkeypatch.delenv("FLUXMOE_TRACE_ROUTER", raising=False)
    monkeypatch.delenv("FLUXMOE_ROUTED_EXPERTS", raising=False)
    inner = method("UnquantizedFusedMoEMethod")
    state = SimpleNamespace(
        topk_indices_dtype=torch.int32,
        rocm_aiter_moe_enabled=False,
        has_bias=False,
        fused_experts=lambda **kw: kw["hidden_states"] + 2,
    )
    native = lambda x, logits: inner(state, layer, x, False, 2, logits, True)
    outer = method("FusedMoE")
    result = outer(SimpleNamespace(forward_native=native), hidden, torch.zeros(1, 4))
    assert torch.equal(result, hidden + 1)
    assert calls == ["calibration", "cache"]
    calls.clear()
    monkeypatch.setenv("FLUXMOE_ENABLE", "0")
    result = outer(SimpleNamespace(forward_native=native), hidden, torch.zeros(1, 4))
    assert torch.equal(result, hidden + 2)
    assert calls == ["calibration"]
