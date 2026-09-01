# FluxMoE H100 Cross-Hardware Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a source-grounded FluxMoE reimplementation that pages routed-expert tensors through stable CUDA virtual addresses, feeds them from compressed GPU memory and pinned host DRAM, adapts residency under memory pressure, and measures whether the expected trends appear on 4x H100 PCIe GPUs.

**Architecture:** Keep FluxMoE as an out-of-tree Python/C++/CUDA package and apply a small, pinned patch to vLLM v0.10.2 at expert-parameter creation and `FusedMoE.forward_cuda`. A fail-closed loader diverts only routed-expert BF16 weights into the storage hierarchy; a two-layer PagedTensor window materializes the current/next layer while telemetry records every mapping, transfer, decompression, planner decision, and benchmark result.

**Tech Stack:** Python 3.11, PyTorch 2.8.0, CUDA 12.x driver/runtime compatible with vLLM v0.10.2, C++17, CUDA Driver VMM API, pybind11 via `torch.utils.cpp_extension`, vLLM v0.10.2, pytest, Ruff, mypy, PyYAML, zstandard, Hugging Face Hub, Docker/GHCR, NVIDIA H100.

**Spec:** `docs/superpowers/specs/2026-09-01-fluxmoe-reproduction-design.md`

## Global Constraints

- Public repository: `rh2844417786/moe-flex`; stable branch `main`; execution branch `repro/fluxmoe`.
- Server checkout: `/home/jovyan/wangtonghan/moe-flex`; server updates use `git pull --ff-only` and do not push back.
- Model source root `/mnt/public_data` is read-only; never create, update, rename, or delete files beneath it.
- All writable server files stay under `/home/jovyan/wangtonghan/moe-flex/{.cache,artifacts,runs,build}`.
- Primary checkpoint: `/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct`, BF16, 41 non-empty shards, `Qwen3NextForCausalLM`.
- Supplemental checkpoint: `/mnt/public_data/modelscope/mistralai/Mixtral-8x7B-v0.1_`; label results as Base-checkpoint supplemental evidence.
- Use only routed experts in FluxMoE; shared experts, router, attention, embeddings, and other non-expert parameters remain resident.
- Pin vLLM tag `v0.10.2` at commit `01efc7ef781391e744ed08c3292817a773d654e6` and PyTorch `2.8.0`.
- Start with vLLM eager execution; enabling CUDA Graph requires a separate passing replay test.
- Formal benchmarks require 4 exclusive H100 GPUs and must reject ranks with unrelated compute processes.
- ShareGPT source is `anon8231489123/ShareGPT_Vicuna_unfiltered@192ab2185289094fc556ec8ce5ce1e8e587154ca`, file `ShareGPT_V3_unfiltered_cleaned_split.json`; sampling seed is `20260901`.
- Correctness is strict: reconstructed BF16 bits, router Top-k IDs, and greedy output tokens must match the resident baseline.
- Performance conclusions are trend-based, not exact L40 number matching; valid statuses are `SUPPORTED`, `MIXED`, `NOT_SUPPORTED`, and `INCONCLUSIVE`.
- Missing telemetry, silent fallback, checkpoint mismatch, unsupported quantization, writable model mount, or unverified CUDA Graph must fail closed.

---

## Planned File Map

### Repository and dependency control

- `README.md`: Chinese project entrypoint, status boundary, Mac/server workflow, and exact commands.
- `LICENSE`: Apache-2.0 text.
- `.gitignore`: excludes raw downloads, builds, runtime artifacts, and results while retaining the fixed benchmark subset.
- `pyproject.toml`: Python package metadata, exact runtime/dev dependencies, pytest/Ruff/mypy configuration.
- `setup.py`: optional CUDA extension build using `torch.utils.cpp_extension.CUDAExtension`.
- `third_party/vllm.lock.json`: pinned vLLM repository, tag, commit, and patch checksum.
- `patches/vllm-v0.10.2.patch`: minimal weight-creation and forward-lifecycle integration.
- `.github/workflows/ci.yml`: CPU unit/static checks.
- `.github/workflows/container.yml`: build and publish the pinned CUDA image to GHCR.
- `docker/Dockerfile`: pinned vLLM image/source, patch application, and FluxMoE extension build.

### Python package

- `src/flexmoe/__init__.py`: version and public exports.
- `src/flexmoe/errors.py`: typed fail-closed exceptions.
- `src/flexmoe/config.py`: immutable runtime, model, planner, dataset, and benchmark configuration.
- `src/flexmoe/manifest.py`: canonical JSON serialization and SHA256 helpers.
- `src/flexmoe/datasets/sharegpt.py`: deterministic ShareGPT download/subset builder and verifier.
- `src/flexmoe/codec/reference.py`: canonical CPU Huffman encoder/decoder for raw BF16 bit patterns.
- `src/flexmoe/placement.py`: deterministic bandwidth-proportional tensor placement.
- `src/flexmoe/planner.py`: residency controller and hard capacity enforcement.
- `src/flexmoe/paged_tensor.py`: Python wrapper around CUDA VMM regions and layer tensor views.
- `src/flexmoe/storage/base.py`: storage protocol and transfer receipt types.
- `src/flexmoe/storage/gpu_compressed.py`: compressed GPU storage and CUDA decompression launch.
- `src/flexmoe/storage/host_pinned.py`: pinned BF16 host storage and asynchronous HtoD launch.
- `src/flexmoe/storage/hierarchy.py`: layer materialization and backend concurrency.
- `src/flexmoe/runtime/lifecycle.py`: two-layer RAW/WAR lifecycle.
- `src/flexmoe/runtime/preflight.py`: model, mount, GPU, process, version, and VMM checks.
- `src/flexmoe/runtime/telemetry.py`: versioned JSONL events and run-state transitions.
- `src/flexmoe/vllm/bridge.py`: vLLM patch-facing API.
- `src/flexmoe/vllm/loader.py`: TP-aware expert weight capture and backend finalization.
- `src/flexmoe/vllm/patch_contract.py`: patch/commit verification.
- `src/flexmoe/bench/workload.py`: benchmark subset reader and exact batch/context selection.
- `src/flexmoe/bench/runner.py`: resident/offload/FluxMoE variant runner.
- `src/flexmoe/bench/report.py`: evidence validation, summary tables, and support classification.

### C++ and CUDA extension

- `csrc/bindings.cpp`: pybind11/torch operator registration.
- `csrc/vmm/paged_region.h`: `PagedRegion` ownership and public methods.
- `csrc/vmm/paged_region.cpp`: CUDA Driver VMM reserve/create/map/unmap/release implementation.
- `csrc/runtime/stream_lifecycle.h`: event and layer-state declarations.
- `csrc/runtime/stream_lifecycle.cu`: compute/load event coordination and state snapshots.
- `csrc/codec/huffman.h`: encoded-buffer metadata and CUDA decode declaration.
- `csrc/codec/huffman_cuda.cu`: BF16 exponent decode and bit reconstruction kernel.

### Data, configurations, scripts, and tests

- `benchmarks/data/sharegpt/source_manifest.json`: source revision, raw hash, tokenizer revision, seed, and generation policy.
- `benchmarks/data/sharegpt/qwen3next_1024_requests.jsonl.zst`: 1024 committed requests.
- `benchmarks/data/sharegpt/dataset_manifest.json`: subset hash and per-length counts.
- `benchmarks/configs/{resident,vllm_o,fluxmoe_fixed,fluxmoe_dynamic,fluxmoe_unbalanced,pagedtensor_resident}.yaml`: exact variants.
- `scripts/prepare_sharegpt.py`: Mac dataset entrypoint.
- `scripts/check_vllm_patch.sh`: clone/checkout/patch validation.
- `scripts/server/{build.sh,run_container.sh,preflight.sh,run_smoke.sh,run_key_matrix.sh}`: server-only operations.
- `tests/fixtures/sharegpt_mini.json`: tiny deterministic dataset fixture.
- `tests/unit/`: CPU tests for every pure-Python contract.
- `tests/cuda/`: VMM, stream, codec, and transfer tests.
- `tests/integration/`: vLLM patch, FusedMoE parity, and benchmark smoke tests.

---

### Task 1: Establish the package, development branch, CI, and public remote

**Files:**
- Create: `README.md`
- Create: `LICENSE`
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/flexmoe/__init__.py`
- Create: `tests/unit/test_package.py`
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: importable package `flexmoe` with `__version__: str = "0.1.0"`.
- Produces: branch `repro/fluxmoe` and public remote `https://github.com/rh2844417786/moe-flex.git`.

- [ ] **Step 1: Create the execution branch and write the failing package test**

```bash
git switch -c repro/fluxmoe
```

```python
# tests/unit/test_package.py
import flexmoe


def test_package_version_is_explicit() -> None:
    assert flexmoe.__version__ == "0.1.0"
```

- [ ] **Step 2: Run the test and verify the package is missing**

Run: `python3 -m pytest tests/unit/test_package.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'flexmoe'`.

- [ ] **Step 3: Add the minimal package and repository controls**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "moe-flex"
version = "0.1.0"
requires-python = ">=3.10,<3.14"
license = {text = "Apache-2.0"}
dependencies = [
  "huggingface-hub>=0.27,<2",
  "numpy>=1.26,<3",
  "pyyaml>=6.0,<7",
  "torch==2.8.0",
  "zstandard>=0.23,<1",
]

[project.optional-dependencies]
dev = ["build>=1.2,<2", "mypy>=1.13,<2", "pytest>=8.3,<9", "ruff>=0.8,<1"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["cuda: requires a CUDA-capable Linux host", "integration: requires pinned vLLM"]

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.mypy]
python_version = "3.10"
strict = true
packages = ["flexmoe"]
```

```python
# src/flexmoe/__init__.py
__version__ = "0.1.0"
```

```gitignore
# .gitignore
.cache/
artifacts/
runs/
build/
dist/
*.egg-info/
*.so
*.dylib
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
benchmarks/data/sharegpt/raw/
```

Create `README.md` in Chinese with the project status `DESIGN APPROVED - IMPLEMENTATION IN PROGRESS`, the H100 conclusion boundary, and links to the spec and plan. Fetch the Apache-2.0 license text from the pinned vLLM repository:

```bash
curl -L https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/LICENSE -o LICENSE
```

Create `.github/workflows/ci.yml` to run on Ubuntu with Python 3.10 and 3.11:

```yaml
name: cpu-ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: ["3.10", "3.11"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - run: python -m pip install -e '.[dev]'
      - run: ruff check .
      - run: mypy src/flexmoe
      - run: pytest tests/unit -q
```

- [ ] **Step 4: Verify package, static checks, and build metadata**

Run: `python3 -m pip install -e '.[dev]'`

Run: `ruff check . && mypy src/flexmoe && pytest tests/unit -q && python3 -m build`

Expected: all commands exit 0; `dist/moe_flex-0.1.0-py3-none-any.whl` exists.

- [ ] **Step 5: Commit the package foundation**

```bash
git add README.md LICENSE .gitignore pyproject.toml src tests .github
git commit -m "chore: initialize moe-flex package"
```

- [ ] **Step 6: Re-authenticate GitHub, create the public repository, and push both branches**

Run: `gh auth status -h github.com`

If it reports the known invalid token, run: `gh auth login -h github.com --web --git-protocol https`

Then run:

```bash
gh repo create rh2844417786/moe-flex --public --source=. --remote=origin
git push -u origin main
git push -u origin repro/fluxmoe
git ls-remote origin refs/heads/main refs/heads/repro/fluxmoe
```

Expected: both remote refs resolve; no force push is used.

---

### Task 2: Build and commit the deterministic ShareGPT benchmark subset

**Files:**
- Create: `src/flexmoe/datasets/__init__.py`
- Create: `src/flexmoe/datasets/sharegpt.py`
- Create: `scripts/prepare_sharegpt.py`
- Create: `tests/fixtures/sharegpt_mini.json`
- Create: `tests/unit/test_sharegpt.py`
- Create: `benchmarks/data/sharegpt/source_manifest.json`
- Create: `benchmarks/data/sharegpt/qwen3next_1024_requests.jsonl.zst`
- Create: `benchmarks/data/sharegpt/dataset_manifest.json`

**Interfaces:**
- Produces: `build_fixed_requests(conversations, tokenizer, lengths, per_length, seed) -> list[PromptRecord]`.
- Produces: `PromptRecord(request_id: str, prompt_token_ids: tuple[int, ...], context_length: int)`.
- Produces: `verify_subset(path: Path, manifest_path: Path) -> None`.

- [ ] **Step 1: Write fixture and failing deterministic-subset tests**

```python
# tests/unit/test_sharegpt.py
from pathlib import Path

from flexmoe.datasets.sharegpt import build_fixed_requests


class TinyTokenizer:
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) % 251 + 1 for char in text]


def test_build_fixed_requests_is_exact_and_deterministic() -> None:
    conversations = [f"conversation-{index}-" * 8 for index in range(20)]
    first = build_fixed_requests(conversations, TinyTokenizer(), (32, 64), 3, 7)
    second = build_fixed_requests(conversations, TinyTokenizer(), (32, 64), 3, 7)
    assert first == second
    assert [len(item.prompt_token_ids) for item in first] == [32] * 3 + [64] * 3
    assert len({item.request_id for item in first}) == 6
```

- [ ] **Step 2: Run the test and verify missing dataset API**

Run: `pytest tests/unit/test_sharegpt.py -q`

Expected: FAIL because `flexmoe.datasets.sharegpt` does not exist.

- [ ] **Step 3: Implement the exact dataset records, packing, download, and serialization APIs**

```python
# src/flexmoe/datasets/sharegpt.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from random import Random
from typing import Iterable, Protocol, Sequence
import json

import zstandard as zstd


class TokenizerLike(Protocol):
    eos_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class PromptRecord:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    context_length: int


def build_fixed_requests(
    conversations: Sequence[str],
    tokenizer: TokenizerLike,
    lengths: tuple[int, ...],
    per_length: int,
    seed: int,
) -> list[PromptRecord]:
    if not conversations or per_length <= 0:
        raise ValueError("conversations and per_length must be non-empty")
    order = list(range(len(conversations)))
    Random(seed).shuffle(order)
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer must define eos_token_id")
    records: list[PromptRecord] = []
    cursor = 0
    for target in lengths:
        for sample_index in range(per_length):
            packed: list[int] = []
            source_ids: list[int] = []
            while len(packed) < target:
                source_index = order[cursor % len(order)]
                cursor += 1
                source_ids.append(source_index)
                packed.extend(tokenizer.encode(conversations[source_index], False))
                packed.append(eos)
            token_ids = tuple(packed[:target])
            digest = sha256(
                f"{seed}:{target}:{sample_index}:{source_ids}".encode()
            ).hexdigest()[:20]
            records.append(PromptRecord(digest, token_ids, target))
    return records


def write_jsonl_zst(records: Iterable[PromptRecord], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    compressor = zstd.ZstdCompressor(level=19)
    with path.open("wb") as raw, compressor.stream_writer(raw) as stream:
        for record in records:
            stream.write((json.dumps(asdict(record), separators=(",", ":")) + "\n").encode())
    return sha256(path.read_bytes()).hexdigest()
```

Implement `load_sharegpt_conversations()` to accept only rows containing a non-empty `conversations` list, apply the Qwen chat template to the ordered turns, and reject rows with malformed roles. Implement `verify_subset()` to decompress every line, assert 1024 unique request IDs, assert counts `{1024:256, 2048:256, 3072:256, 4096:256}`, and compare the compressed file SHA256 with `dataset_manifest.json`.

The CLI in `scripts/prepare_sharegpt.py` must call `huggingface_hub.hf_hub_download()` with the exact dataset revision and `snapshot_download()` with tokenizer/config-only allow patterns.

- [ ] **Step 4: Run unit tests on the fixture**

Run: `pytest tests/unit/test_sharegpt.py -q`

Expected: PASS.

- [ ] **Step 5: Download raw data on Mac, generate the 1024-request subset, and verify it**

```bash
python scripts/prepare_sharegpt.py \
  --cache-dir .cache/sharegpt \
  --output benchmarks/data/sharegpt/qwen3next_1024_requests.jsonl.zst \
  --manifest benchmarks/data/sharegpt/dataset_manifest.json
python -m flexmoe.datasets.sharegpt verify \
  benchmarks/data/sharegpt/qwen3next_1024_requests.jsonl.zst \
  benchmarks/data/sharegpt/dataset_manifest.json
test $(stat -f%z benchmarks/data/sharegpt/qwen3next_1024_requests.jsonl.zst) -lt 100000000
```

Expected: verifier reports 1024 records and four groups of 256; the file is under 100MB. Store raw source SHA256, source revision, tokenizer revision, seed, packing rule, subset SHA256, and counts in the two manifests.

- [ ] **Step 6: Commit only the generator, manifests, fixture, and compressed subset**

```bash
git add src/flexmoe/datasets scripts/prepare_sharegpt.py tests/fixtures tests/unit/test_sharegpt.py benchmarks/data/sharegpt
git commit -m "feat: add deterministic ShareGPT benchmark subset"
```

---

### Task 3: Add immutable configuration, manifests, and fail-closed preflight

**Files:**
- Create: `src/flexmoe/errors.py`
- Create: `src/flexmoe/config.py`
- Create: `src/flexmoe/manifest.py`
- Create: `src/flexmoe/runtime/__init__.py`
- Create: `src/flexmoe/runtime/preflight.py`
- Create: `tests/unit/test_config.py`
- Create: `tests/unit/test_preflight.py`

**Interfaces:**
- Produces: `RuntimeConfig`, `ModelSpec`, `PlannerConfig`, `BenchmarkConfig` frozen dataclasses.
- Produces: `CheckResult(name: str, ok: bool, details: str)` and `PreflightReport(ok: bool, checks: tuple[CheckResult, ...], environment: dict[str, object])`.
- Produces: `run_preflight(config: RuntimeConfig, probe: SystemProbe) -> PreflightReport`.

- [ ] **Step 1: Write failing config and preflight tests**

```python
# tests/unit/test_preflight.py
from pathlib import Path

import pytest

from flexmoe.config import ModelSpec
from flexmoe.errors import PreflightError
from flexmoe.runtime.preflight import validate_checkpoint_files


def test_checkpoint_requires_every_indexed_shard(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        '{"architectures":["Qwen3NextForCausalLM"],"torch_dtype":"bfloat16"}'
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"model-00001-of-00002.safetensors",'
        '"b":"model-00002-of-00002.safetensors"}}'
    )
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"x")
    spec = ModelSpec(tmp_path, "Qwen3NextForCausalLM", "bfloat16", 2)
    with pytest.raises(PreflightError, match="model-00002-of-00002"):
        validate_checkpoint_files(spec)
```

- [ ] **Step 2: Verify tests fail because contracts do not exist**

Run: `pytest tests/unit/test_config.py tests/unit/test_preflight.py -q`

Expected: FAIL on missing modules.

- [ ] **Step 3: Implement exact config and error types**

```python
# src/flexmoe/config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


RunStatus = Literal["SUPPORTED", "MIXED", "NOT_SUPPORTED", "INCONCLUSIVE"]
Variant = Literal[
    "vllm-resident", "vllm-o", "fluxmoe-fixed", "fluxmoe-dynamic",
    "fluxmoe-dynamic-unbalanced", "pagedtensor-resident",
]


@dataclass(frozen=True)
class ModelSpec:
    path: Path
    architecture: str
    dtype: str
    expected_shards: int


@dataclass(frozen=True)
class PlannerConfig:
    io_bound_threshold: float = 0.9
    compute_bound_threshold: float = 1.0
    decision_interval: int = 300


@dataclass(frozen=True)
class BenchmarkConfig:
    variant: Variant
    batch_size: int
    context_length: int
    output_length: int
    tensor_parallel_size: int = 4
    greedy: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    project_root: Path
    model: ModelSpec
    planner: PlannerConfig
    benchmark: BenchmarkConfig
    vllm_commit: str = "01efc7ef781391e744ed08c3292817a773d654e6"
```

Create `FluxMoEError`, `ConfigurationError`, `PreflightError`, `IntegrityError`, and `UnsupportedModeError` in `errors.py`. Implement canonical JSON with sorted keys and compact separators in `manifest.py`.

- [ ] **Step 4: Implement preflight with dependency injection**

Define `SystemProbe` protocol methods `gpu_inventory()`, `compute_processes()`, `mount_options(path)`, `driver_version()`, `torch_version()`, `vllm_commit()`, and `vmm_supported(device)`. Implement checks for exact project/model roots, index/shards, architecture/dtype, source mount `ro`, 4 distinct GPUs, zero unrelated compute processes, pinned vLLM commit, and VMM support. Never test read-only status by writing to `/mnt/public_data`; inspect `/proc/self/mountinfo` through the probe.

```python
@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    details: str


@dataclass(frozen=True)
class PreflightReport:
    ok: bool
    checks: tuple[CheckResult, ...]
    environment: dict[str, object]
```

- [ ] **Step 5: Run CPU tests and type checks**

Run: `pytest tests/unit/test_config.py tests/unit/test_preflight.py -q && mypy src/flexmoe`

Expected: PASS.

- [ ] **Step 6: Commit configuration and preflight**

```bash
git add src/flexmoe tests/unit/test_config.py tests/unit/test_preflight.py
git commit -m "feat: add fail-closed runtime preflight"
```

---

### Task 4: Implement deterministic bandwidth-proportional placement

**Files:**
- Create: `src/flexmoe/placement.py`
- Create: `tests/unit/test_placement.py`

**Interfaces:**
- Produces: `TensorSpec`, `BackendProfile`, `Placement`, and `assign_tensors(tensors: tuple[TensorSpec, ...], profiles: tuple[BackendProfile, ...], gpu_budget_bytes: int) -> tuple[Placement, ...]`.
- Consumes: exact per-rank tensor sizes and measured backend bandwidth/capacity.

- [ ] **Step 1: Write failing placement tests**

```python
from flexmoe.placement import BackendProfile, TensorSpec, assign_tensors


def test_assignment_is_deterministic_and_respects_gpu_budget() -> None:
    tensors = tuple(TensorSpec(f"t{i}", i // 4, i % 4, "w13", 10) for i in range(12))
    profiles = (
        BackendProfile("gpu_compressed", 300.0, 40),
        BackendProfile("host_pinned", 100.0, 1_000),
    )
    first = assign_tensors(tensors, profiles, gpu_budget_bytes=40)
    second = assign_tensors(tuple(reversed(tensors)), profiles, gpu_budget_bytes=40)
    assert first == second
    assert sum(item.nbytes for item in first if item.backend == "gpu_compressed") <= 40
```

- [ ] **Step 2: Verify missing placement API**

Run: `pytest tests/unit/test_placement.py -q`

Expected: FAIL on missing module.

- [ ] **Step 3: Implement exact placement types and stable greedy assignment**

```python
@dataclass(frozen=True, order=True)
class TensorSpec:
    tensor_id: str
    layer_idx: int
    expert_idx: int
    kind: Literal["w13", "w2"]
    nbytes: int


@dataclass(frozen=True)
class BackendProfile:
    name: Literal["gpu_compressed", "host_pinned"]
    bytes_per_second: float
    capacity_bytes: int


@dataclass(frozen=True, order=True)
class Placement:
    tensor_id: str
    backend: Literal["gpu_compressed", "host_pinned"]
    offset: int
    nbytes: int
```

Sort tensors by `(layer_idx, kind, expert_idx, tensor_id)`. At each step, select the eligible backend whose projected finish time `(assigned_bytes + tensor.nbytes) / bytes_per_second` is smallest; break ties by backend name. Enforce both backend capacity and the explicit GPU budget. Raise `ConfigurationError` if total capacity is insufficient.

- [ ] **Step 4: Verify deterministic placement and capacity errors**

Run: `pytest tests/unit/test_placement.py -q`

Expected: PASS.

- [ ] **Step 5: Commit placement**

```bash
git add src/flexmoe/placement.py tests/unit/test_placement.py
git commit -m "feat: add bandwidth-balanced expert placement"
```

---

### Task 5: Implement the budget-aware residency planner

**Files:**
- Create: `src/flexmoe/planner.py`
- Create: `tests/unit/test_planner.py`

**Interfaces:**
- Consumes: `PlannerConfig` from Task 3.
- Produces: `PlannerObservation`, `PlannerDecision`, and `ResidencyPlanner.decide()`.

- [ ] **Step 1: Write failing decrease/hold/increase/capacity tests**

```python
from flexmoe.config import PlannerConfig
from flexmoe.planner import PlannerObservation, ResidencyPlanner


def observation(load_s: float, expert_bytes: int = 400) -> PlannerObservation:
    return PlannerObservation(
        iteration=300, compute_reference_s=1.0, load_s=load_s,
        gpu_capacity_bytes=1_000, kv_bytes=200, fixed_bytes=200,
        expert_gpu_bytes=expert_bytes, step_bytes=100,
    )


def test_planner_actions_and_capacity_precedence() -> None:
    planner = ResidencyPlanner(PlannerConfig())
    assert planner.decide(observation(0.5)).action == "decrease"
    assert planner.decide(observation(1.05)).action == "hold"
    assert planner.decide(observation(1.2)).action == "increase"
    constrained = observation(1.2, expert_bytes=700)
    decision = planner.decide(constrained)
    assert decision.expert_gpu_bytes <= 600
```

- [ ] **Step 2: Verify missing planner API**

Run: `pytest tests/unit/test_planner.py -q`

Expected: FAIL on missing module.

- [ ] **Step 3: Implement immutable planner contracts and decision logic**

```python
@dataclass(frozen=True)
class PlannerObservation:
    iteration: int
    compute_reference_s: float
    load_s: float
    gpu_capacity_bytes: int
    kv_bytes: int
    fixed_bytes: int
    expert_gpu_bytes: int
    step_bytes: int


@dataclass(frozen=True)
class PlannerDecision:
    action: Literal["increase", "hold", "decrease"]
    expert_gpu_bytes: int
    ratio: float
    reason: str
```

Compute `ratio = compute_reference_s / load_s`. Decrease residency when ratio is above 1.0, increase when below 0.9, and hold in the closed interval `[0.9, 1.0]`. Clamp every result to `max(0, gpu_capacity_bytes - kv_bytes - fixed_bytes)` before returning. Reject non-positive times and negative byte counts.

- [ ] **Step 4: Run planner tests including seven repeated decisions and dead-zone stability**

Run: `pytest tests/unit/test_planner.py -q`

Expected: PASS with no oscillation for observations inside the dead zone.

- [ ] **Step 5: Commit planner**

```bash
git add src/flexmoe/planner.py tests/unit/test_planner.py
git commit -m "feat: add budget-aware residency planner"
```

---

### Task 6: Implement the canonical CPU BF16 Huffman reference codec

**Files:**
- Create: `src/flexmoe/codec/__init__.py`
- Create: `src/flexmoe/codec/reference.py`
- Create: `tests/unit/test_codec_reference.py`

**Interfaces:**
- Produces: `EncodedBFloat16` and bit-exact `encode_bf16_bits()` / `decode_bf16_bits()`.
- Produces: canonical code lengths for all 256 exponent symbols with symbol-ID tie breaking.

- [ ] **Step 1: Write failing edge-pattern and random round-trip tests**

```python
import numpy as np

from flexmoe.codec.reference import decode_bf16_bits, encode_bf16_bits


def test_bf16_round_trip_preserves_every_bit_pattern() -> None:
    values = np.arange(0, 65536, dtype=np.uint16).tobytes()
    encoded = encode_bf16_bits(values, shape=(65536,))
    assert decode_bf16_bits(encoded) == values


def test_codec_is_deterministic() -> None:
    raw = np.random.default_rng(7).integers(0, 65536, 10000, dtype=np.uint16).tobytes()
    assert encode_bf16_bits(raw, (10000,)) == encode_bf16_bits(raw, (10000,))
```

- [ ] **Step 2: Verify missing codec API**

Run: `pytest tests/unit/test_codec_reference.py -q`

Expected: FAIL on missing module.

- [ ] **Step 3: Implement the encoded representation and canonical Huffman codec**

```python
@dataclass(frozen=True)
class EncodedBFloat16:
    shape: tuple[int, ...]
    element_count: int
    bit_count: int
    chunk_elements: int
    chunk_byte_offsets: tuple[int, ...]
    chunk_bit_lengths: tuple[int, ...]
    sign_mantissa: bytes
    exponent_payload: bytes
    code_lengths: tuple[int, ...]
```

Interpret each little-endian BF16 word with `mantissa = word & 0x7f`, `sign = (word >> 15) & 1`, `sign_mantissa = mantissa | (sign << 7)`, and `exponent = (word >> 7) & 0xff`. Build a frequency heap keyed by `(frequency, minimum_symbol, node_id)` so ties are deterministic; a single-symbol alphabet receives code length 1 and zero-frequency symbols receive length 0. Convert lengths to canonical codes ordered by `(length, symbol)`. Split elements into fixed 4096-element chunks, encode each chunk with the global canonical table, pad each chunk to the next byte, and record its byte offset and meaningful bit length. Pack bits most-significant-bit first. Decoder must consume exactly `element_count` symbols and exactly each recorded chunk length; reject truncated or trailing non-zero data.

- [ ] **Step 4: Run the exhaustive 65,536-pattern test**

Run: `pytest tests/unit/test_codec_reference.py -q`

Expected: PASS; all BF16 bit patterns round-trip.

- [ ] **Step 5: Commit CPU codec**

```bash
git add src/flexmoe/codec tests/unit/test_codec_reference.py
git commit -m "feat: add bit-exact BF16 Huffman reference"
```

---

### Task 7: Add the pinned CUDA development image and extension build shell

**Files:**
- Create: `third_party/vllm.lock.json`
- Create: `setup.py`
- Create: `csrc/bindings.cpp`
- Create: `docker/Dockerfile`
- Create: `.github/workflows/container.yml`
- Create: `tests/unit/test_vllm_lock.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: importable `flexmoe._C` with `extension_version() -> str`.
- Produces: public GHCR image in `ghcr.io/rh2844417786/moe-flex`, tagged with the current 40-character Git commit SHA.

- [ ] **Step 1: Write failing lock and extension-import tests**

```python
def test_vllm_lock_is_exact() -> None:
    import json
    from pathlib import Path
    lock = json.loads(Path("third_party/vllm.lock.json").read_text())
    assert lock["commit"] == "01efc7ef781391e744ed08c3292817a773d654e6"
    assert lock["tag"] == "v0.10.2"
```

Create `tests/cuda/test_extension_import.py` asserting `flexmoe._C.extension_version() == "0.1.0"`.

- [ ] **Step 2: Verify lock test fails**

Run: `pytest tests/unit/test_vllm_lock.py -q`

Expected: FAIL because the lock file is missing.

- [ ] **Step 3: Add exact vLLM lock and CUDAExtension build**

```json
{
  "repository": "https://github.com/vllm-project/vllm.git",
  "tag": "v0.10.2",
  "commit": "01efc7ef781391e744ed08c3292817a773d654e6",
  "torch": "2.8.0"
}
```

`setup.py` must set `build_cuda = os.environ.get("FLEXMOE_BUILD_CUDA") == "1"` and use `ext_modules=[]` for Mac/CPU installs. When `build_cuda` is true, define `CUDAExtension("flexmoe._C", sources=["csrc/bindings.cpp"], extra_link_args=["-lcuda"], extra_compile_args={"cxx":["-O3","-std=c++17"],"nvcc":["-O3","-std=c++17","-lineinfo"]})`. `bindings.cpp` exposes the fixed version string and `cuda_driver_version()`; the Docker image sets `FLEXMOE_BUILD_CUDA=1` before installation.

- [ ] **Step 4: Add a pinned container and GHCR workflow**

Start `docker/Dockerfile` from `vllm/vllm-openai:v0.10.2`, install `git`, `ninja-build`, and build prerequisites, copy the repository, and run `pip install --no-build-isolation -e .`. Set `TORCH_CUDA_ARCH_LIST=9.0` for H100.

The workflow must use `docker/login-action@v3` with `${{ secrets.GITHUB_TOKEN }}`, `docker/build-push-action@v6`, and push both the commit SHA tag and `repro-fluxmoe` tag on the execution branch.

After the first successful workflow, make the container package public and verify unauthenticated metadata access:

```bash
gh api --method PATCH /user/packages/container/moe-flex/visibility -f visibility=public
docker pull ghcr.io/rh2844417786/moe-flex:repro-fluxmoe
```

- [ ] **Step 5: Verify CPU lock test and container extension import**

Run: `pytest tests/unit/test_vllm_lock.py -q`

Run in an amd64 CUDA build environment: `docker build -t moe-flex:dev -f docker/Dockerfile .`

Run: `docker run --rm --gpus all moe-flex:dev python -c 'import flexmoe._C as c; assert c.extension_version() == "0.1.0"'`

Expected: all exit 0.

- [ ] **Step 6: Commit build foundation**

```bash
git add third_party setup.py csrc/bindings.cpp docker .github/workflows/container.yml pyproject.toml tests
git commit -m "build: add pinned CUDA extension image"
```

---

### Task 8: Implement CUDA virtual-memory regions and stable tensor views

**Files:**
- Create: `csrc/vmm/paged_region.h`
- Create: `csrc/vmm/paged_region.cpp`
- Create: `src/flexmoe/paged_tensor.py`
- Create: `tests/cuda/test_paged_region.py`
- Modify: `csrc/bindings.cpp`
- Modify: `setup.py`

**Interfaces:**
- Produces C++ class `PagedRegion(device, virtual_bytes)` with `create_block`, `map`, `unmap`, `tensor`, `base_address`, and `snapshot`.
- Produces Python `PagedTensorRegion` wrapper with the same lifetime and typed validation.

- [ ] **Step 1: Write failing CUDA pointer-stability and alignment tests**

```python
import pytest
import torch

from flexmoe.paged_tensor import PagedTensorRegion


@pytest.mark.cuda
def test_pointer_is_stable_across_remap() -> None:
    region = PagedTensorRegion(device=0, virtual_bytes=2 << 20)
    block_a = region.create_block(1 << 20)
    block_b = region.create_block(1 << 20)
    region.map(0, block_a, 1 << 20)
    view = region.tensor(0, (512, 1024), torch.bfloat16)
    pointer = view.data_ptr()
    view.fill_(1)
    torch.cuda.synchronize()
    region.unmap(0, 1 << 20)
    region.map(0, block_b, 1 << 20)
    assert region.tensor(0, (512, 1024), torch.bfloat16).data_ptr() == pointer
```

- [ ] **Step 2: Build and verify the test fails on missing class**

Run inside container: `pip install --no-build-isolation -e . && pytest tests/cuda/test_paged_region.py -q`

Expected: FAIL because `PagedRegion` is not bound.

- [ ] **Step 3: Implement VMM ownership and error checking**

`PagedRegion` constructor calls `cuInit(0)`, queries minimum allocation granularity, rounds the virtual size up, and reserves one address range. `create_block` uses `cuMemCreate`; `map` uses `cuMemMap` plus `cuMemSetAccess`; `unmap` uses `cuMemUnmap`. Reject overlapping mappings, unknown block IDs, unaligned offsets/sizes, out-of-range mappings, or tensor views that cross unmapped pages. Destructor synchronizes the owned device, unmaps all live ranges, releases every allocation handle, and frees the address reservation exactly once.

Create CUDA tensors with `torch::from_blob` using the stable address, explicit sizes/strides, a no-op deleter tied to the owning `PagedRegion` Python object, and CUDA TensorOptions.

- [ ] **Step 4: Run VMM tests and a 10,000-remap stress test**

Run: `pytest tests/cuda/test_paged_region.py -q`

Run: `compute-sanitizer --tool memcheck python -m pytest tests/cuda/test_paged_region.py -q`

Expected: PASS; base pointer never changes; sanitizer reports zero errors.

- [ ] **Step 5: Commit VMM**

```bash
git add csrc/vmm csrc/bindings.cpp setup.py src/flexmoe/paged_tensor.py tests/cuda/test_paged_region.py
git commit -m "feat: add stable CUDA paged tensor regions"
```

---

### Task 9: Implement the two-layer RAW/WAR lifecycle

**Files:**
- Create: `csrc/runtime/stream_lifecycle.h`
- Create: `csrc/runtime/stream_lifecycle.cu`
- Create: `src/flexmoe/runtime/lifecycle.py`
- Create: `tests/unit/test_lifecycle_state.py`
- Create: `tests/cuda/test_stream_lifecycle.py`
- Modify: `csrc/bindings.cpp`
- Modify: `setup.py`

**Interfaces:**
- Produces: `LayerLifecycle.ensure_ready(layer_idx, compute_stream)`, `mark_consumed(layer_idx, compute_stream)`, and `schedule_next(layer_idx)`.
- Produces states `UNMAPPED`, `LOADING`, `RESIDENT`, `EVICTING` and snapshots for telemetry.

- [ ] **Step 1: Write failing cyclic-target and illegal-transition tests**

```python
from flexmoe.runtime.lifecycle import LayerState, target_recycle_layer


def test_two_layer_recycle_wraps_across_iterations() -> None:
    assert target_recycle_layer(layer_idx=1, total_layers=48) == 47
    assert target_recycle_layer(layer_idx=0, total_layers=48) == 46


def test_state_enum_is_explicit() -> None:
    assert {state.value for state in LayerState} == {
        "unmapped", "loading", "resident", "evicting"
    }
```

- [ ] **Step 2: Verify missing lifecycle API**

Run: `pytest tests/unit/test_lifecycle_state.py -q`

Expected: FAIL on missing module.

- [ ] **Step 3: Implement the state machine and background host orchestration**

Implement `target_recycle_layer(i, n) = (i - 2) % n` for zero-based indices. Use one background worker per tensor kind. Before host-side `cuMemUnmap`, the worker calls `cudaEventSynchronize(compute_done[target])`; after mapping, it launches copy/decode on its load stream and records `load_done[layer]`. `ensure_ready` calls `cudaStreamWaitEvent(compute_stream, load_done[layer])`. `mark_consumed` records the compute event after the fused MoE operation. Bypass recycle only for the first two layers of the first iteration.

- [ ] **Step 4: Run CPU transition tests and CUDA race stress**

Run: `pytest tests/unit/test_lifecycle_state.py -q`

Run: `compute-sanitizer --tool racecheck python -m pytest tests/cuda/test_stream_lifecycle.py -q`

Expected: PASS; no stale reads or premature reuse in at least 10,000 cyclic transitions.

- [ ] **Step 5: Commit lifecycle**

```bash
git add csrc/runtime csrc/bindings.cpp setup.py src/flexmoe/runtime/lifecycle.py tests
git commit -m "feat: add two-layer expert lifecycle"
```

---

### Task 10: Implement CUDA Huffman decode and both storage backends

**Files:**
- Create: `csrc/codec/huffman.h`
- Create: `csrc/codec/huffman_cuda.cu`
- Create: `src/flexmoe/storage/__init__.py`
- Create: `src/flexmoe/storage/base.py`
- Create: `src/flexmoe/storage/gpu_compressed.py`
- Create: `src/flexmoe/storage/host_pinned.py`
- Create: `src/flexmoe/storage/hierarchy.py`
- Create: `tests/unit/test_storage_contract.py`
- Create: `tests/cuda/test_huffman_cuda.py`
- Create: `tests/cuda/test_storage_hierarchy.py`
- Modify: `csrc/bindings.cpp`
- Modify: `setup.py`

**Interfaces:**
- Produces: `MaterializationReceipt(tensor_id, backend, nbytes, elapsed_s)`.
- Produces protocol `ExpertTensorStore.materialize(tensor_id, destination, stream) -> MaterializationReceipt`.
- Produces: `StorageHierarchy.materialize_layer(layer_idx: int, destinations: Mapping[str, torch.Tensor]) -> tuple[MaterializationReceipt, ...]`.

- [ ] **Step 1: Write failing storage protocol and CUDA bit-parity tests**

```python
@pytest.mark.cuda
def test_cuda_decode_matches_reference_for_all_bf16_patterns() -> None:
    raw = np.arange(65536, dtype=np.uint16).tobytes()
    encoded = encode_bf16_bits(raw, (65536,))
    decoded = cuda_decode(encoded, device=0)
    assert decoded.view(torch.uint16).cpu().numpy().tobytes() == raw
```

Test the host store by asserting `source.is_pinned()` and a non-blocking copy into a CUDA destination. Test the hierarchy with fake stores and assert receipts are stable-sorted by tensor ID.

- [ ] **Step 2: Verify tests fail before CUDA codec/storage exists**

Run: `pytest tests/unit/test_storage_contract.py -q`

Run in container: `pytest tests/cuda/test_huffman_cuda.py tests/cuda/test_storage_hierarchy.py -q`

Expected: FAIL on missing storage and CUDA decode symbols.

- [ ] **Step 3: Implement CUDA canonical decode and storage protocols**

Build a fixed prefix lookup table from the CPU canonical lengths. Each CUDA warp decodes independent byte-aligned chunks whose starting bit offsets are recorded during encoding. For raw byte `sign_mantissa`, each thread reconstructs `mantissa = sign_mantissa & 0x7f`, `sign = (sign_mantissa & 0x80) << 8`, and `uint16 word = sign | (exponent << 7) | mantissa`, then writes directly to the mapped BF16 destination. Validate element count and output size before launch.

```python
@dataclass(frozen=True)
class MaterializationReceipt:
    tensor_id: str
    backend: Literal["gpu_compressed", "host_pinned"]
    nbytes: int
    elapsed_s: float
```

Expose `cuda_decode(encoded: EncodedBFloat16, device: int, destination: torch.Tensor | None = None) -> torch.Tensor`; when supplied, `destination` must be a mapped contiguous BF16 tensor with the exact encoded shape.

`GpuCompressedStore` owns device byte tensors for sign/mantissa, exponent payload, code lengths/table, and chunk offsets. `PinnedHostStore` owns final TP-sharded BF16 tensors created with `pin_memory=True`. Neither backend writes persistent weight payloads to disk.

- [ ] **Step 4: Benchmark and verify both paths**

Run: `pytest tests/unit/test_storage_contract.py -q`

Run: `compute-sanitizer --tool memcheck python -m pytest tests/cuda/test_huffman_cuda.py tests/cuda/test_storage_hierarchy.py -q`

Run: `python -m flexmoe.storage.hierarchy benchmark --bytes 1073741824 --iterations 20`

Expected: bit-exact decode, non-zero reported HtoD/decompression bytes, zero sanitizer errors, and positive measured bandwidth for both backends.

- [ ] **Step 5: Commit storage hierarchy**

```bash
git add csrc/codec csrc/bindings.cpp setup.py src/flexmoe/storage tests
git commit -m "feat: add compressed GPU and pinned host stores"
```

---

### Task 11: Integrate FluxMoE with the pinned vLLM FusedMoE loader and forward path

**Files:**
- Create: `src/flexmoe/vllm/__init__.py`
- Create: `src/flexmoe/vllm/bridge.py`
- Create: `src/flexmoe/vllm/loader.py`
- Create: `src/flexmoe/vllm/patch_contract.py`
- Create: `patches/vllm-v0.10.2.patch`
- Create: `scripts/check_vllm_patch.sh`
- Create: `tests/unit/test_expert_loader.py`
- Create: `tests/integration/test_vllm_patch.py`
- Create: `tests/integration/test_fused_moe_parity.py`
- Modify: `docker/Dockerfile`

**Interfaces:**
- Patch consumes: `maybe_create_weights(layer: torch.nn.Module, num_experts: int, hidden_size: int, intermediate_size_per_partition: int, params_dtype: torch.dtype, **weight_attrs: object) -> bool`, `before_forward(layer_name: str, w13: torch.Tensor, w2: torch.Tensor) -> ForwardToken`, and `after_forward(token: ForwardToken) -> None`.
- Loader consumes the exact vLLM weight-loader call `(param, loaded_weight, weight_name, shard_id, expert_id)`.
- Produces: TP-sharded `w13`/`w2` storage without full resident expert Parameters.

- [ ] **Step 1: Write failing TP-sharding and patch-contract tests**

```python
def test_loader_combines_w1_w3_and_shards_like_vllm() -> None:
    accumulator = ExpertLoadAccumulator(
        layer_name="model.layers.0.mlp.experts", tp_rank=1, tp_size=4
    )
    w1 = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
    w3 = w1 + 100
    accumulator.ingest("w1", 0, w1)
    accumulator.ingest("w3", 0, w3)
    combined = accumulator.finalize_w13(0)
    assert combined.shape == (4, 4)
    assert torch.equal(combined[:2], w1[2:4])
    assert torch.equal(combined[2:], w3[2:4])
```

`test_vllm_patch.py` clones/checks out the lock commit into a temporary directory and runs `git apply --check patches/vllm-v0.10.2.patch`.

- [ ] **Step 2: Verify loader and patch tests fail**

Run: `pytest tests/unit/test_expert_loader.py tests/integration/test_vllm_patch.py -q`

Expected: FAIL because loader and patch are absent.

- [ ] **Step 3: Implement TP-aware accumulation and fail-closed bridge**

```python
@dataclass(frozen=True)
class ForwardToken:
    layer_name: str
    layer_idx: int


class ExpertLoadAccumulator:
    def __init__(self, layer_name: str, tp_rank: int, tp_size: int) -> None:
        self.layer_name = layer_name
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.parts: dict[tuple[int, str], torch.Tensor] = {}

    def ingest(
        self,
        shard_id: str,
        expert_id: int,
        loaded_weight: torch.Tensor,
    ) -> None:
        if shard_id not in {"w1", "w2", "w3"}:
            raise IntegrityError(f"unexpected shard {shard_id}")
        key = (expert_id, shard_id)
        if key in self.parts:
            raise IntegrityError(f"duplicate expert shard {key}")
        self.parts[key] = loaded_weight.contiguous()

    def finalize_w13(self, expert_id: int) -> torch.Tensor:
        w1 = self.parts.pop((expert_id, "w1"))
        w3 = self.parts.pop((expert_id, "w3"))
        rows = w1.shape[0] // self.tp_size
        start = rows * self.tp_rank
        return torch.cat((w1.narrow(0, start, rows), w3.narrow(0, start, rows)))

    def finalize_w2(self, expert_id: int) -> torch.Tensor:
        w2 = self.parts.pop((expert_id, "w2"))
        columns = w2.shape[1] // self.tp_size
        return w2.narrow(1, columns * self.tp_rank, columns)


def store_expert_weight(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    expert_id: int,
    return_success: bool = False,
) -> bool | None:
    registry = require_active_registry()
    registry.ingest(param, loaded_weight.cpu(), weight_name, shard_id, expert_id)
    return True if return_success else None
```

For `w1/w3`, TP-shard output dimension and combine gate/up in vLLM order; for `w2`, TP-shard input dimension. Reject any dtype except BF16, any quant config, EP/EPLB, missing expert callback, duplicate shard, or unexpected shape. Finalize each layer only after every routed expert has `w1/w2/w3`.

`maybe_create_weights` returns `False` only when `FLUXMOE_ENABLE` is absent. When enabled, any unsupported state raises `UnsupportedModeError`; it never falls back. It registers VMM-backed `w13_weight` and `w2_weight` Parameters with `store_expert_weight` as their `weight_loader`.

- [ ] **Step 4: Create the minimal pinned patch**

The patch changes only `vllm/model_executor/layers/fused_moe/layer.py`:

```diff
@@ UnquantizedFusedMoEMethod.create_weights(...):
+        if os.environ.get("FLUXMOE_ENABLE") == "1":
+            from flexmoe.vllm.bridge import maybe_create_weights
+            if maybe_create_weights(layer, num_experts, hidden_size,
+                                    intermediate_size_per_partition,
+                                    params_dtype, **extra_weight_attrs):
+                return
@@ FusedMoE.forward_cuda(...):
-        return self.forward_native(hidden_states, router_logits)
+        if os.environ.get("FLUXMOE_ENABLE") != "1":
+            return self.forward_native(hidden_states, router_logits)
+        from flexmoe.vllm.bridge import after_forward, before_forward
+        token = before_forward(self.layer_name, self.w13_weight, self.w2_weight)
+        try:
+            return self.forward_native(hidden_states, router_logits)
+        finally:
+            after_forward(token)
```

Add the corresponding `import os`. `scripts/check_vllm_patch.sh` must clone the exact commit, run `git apply --check`, apply the patch, and assert the resulting diff touches only that file.

- [ ] **Step 5: Update Docker image to install patched editable vLLM**

Clone vLLM in the image, verify HEAD equals the lock commit, apply the patch, and run `VLLM_USE_PRECOMPILED=1 pip install --no-build-isolation -e /opt/vllm`. Rebuild FluxMoE afterward so it uses the same PyTorch ABI.

- [ ] **Step 6: Run patch, loader, and tiny FusedMoE parity tests**

Run: `pytest tests/unit/test_expert_loader.py tests/integration/test_vllm_patch.py -q`

Run in CUDA container: `pytest tests/integration/test_fused_moe_parity.py -q`

Expected: resident and paged tiny FusedMoE outputs are bit-equal for deterministic BF16 inputs; telemetry shows non-zero mapping and materialization events.

- [ ] **Step 7: Commit vLLM integration**

```bash
git add src/flexmoe/vllm patches scripts/check_vllm_patch.sh tests docker/Dockerfile
git commit -m "feat: integrate paged experts with vLLM"
```

---

### Task 12: Add telemetry, benchmark variants, reports, and server commands

**Files:**
- Create: `src/flexmoe/runtime/telemetry.py`
- Create: `src/flexmoe/bench/__init__.py`
- Create: `src/flexmoe/bench/workload.py`
- Create: `src/flexmoe/bench/runner.py`
- Create: `src/flexmoe/bench/report.py`
- Create: `benchmarks/configs/resident.yaml`
- Create: `benchmarks/configs/vllm_o.yaml`
- Create: `benchmarks/configs/fluxmoe_fixed.yaml`
- Create: `benchmarks/configs/fluxmoe_dynamic.yaml`
- Create: `benchmarks/configs/fluxmoe_unbalanced.yaml`
- Create: `benchmarks/configs/pagedtensor_resident.yaml`
- Create: `scripts/server/build.sh`
- Create: `scripts/server/run_container.sh`
- Create: `scripts/server/preflight.sh`
- Create: `scripts/server/run_smoke.sh`
- Create: `scripts/server/run_key_matrix.sh`
- Create: `tests/unit/test_telemetry.py`
- Create: `tests/unit/test_report.py`

**Interfaces:**
- Produces: schema-versioned `Event`, `JsonlTelemetry`, `RunEvidence`, and `classify_support(evidence: RunEvidence, stressed_delta: float | None = None, delayed_oom: bool = False) -> SupportClassification`.
- Produces: benchmark CLI that writes one immutable `runs/{timestamp}-{git_sha}/` directory.

- [ ] **Step 1: Write failing evidence and classification tests**

```python
def test_missing_mechanism_evidence_is_inconclusive() -> None:
    evidence = RunEvidence(
        output_tokens_per_second=100.0,
        mapped_bytes=0,
        h2d_bytes=0,
        decompressed_bytes=0,
        output_tokens_match=True,
        router_topk_match=True,
        weights_bit_exact=True,
    )
    assert classify_support(evidence).status == "INCONCLUSIVE"
```

Add tests for `SUPPORTED` when strict correctness holds and stressed FluxMoE improves or delays OOM, `MIXED` for valid but inconsistent trends, and `NOT_SUPPORTED` for repeatable opposite trends with complete evidence.

- [ ] **Step 2: Verify telemetry/report APIs are missing**

Run: `pytest tests/unit/test_telemetry.py tests/unit/test_report.py -q`

Expected: FAIL on missing modules.

- [ ] **Step 3: Implement event schema and buffered JSONL writer**

```python
@dataclass(frozen=True)
class Event:
    schema_version: int
    monotonic_ns: int
    rank: int
    kind: str
    payload: dict[str, int | float | str | bool]


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
```

`JsonlTelemetry` writes sorted compact JSON, buffers up to 256 events, flushes on close/error, and includes run ID, git SHA, rank, layer, variant, VMM bytes, HtoD bytes, decompression bytes, load/compute/stall times, KV bytes, expert residency, and planner decisions. It must never swallow I/O errors.

- [ ] **Step 4: Implement workload, runner, exact configs, and report**

`runner.py` loads prompt token IDs, instantiates vLLM with TP=4, BF16, greedy decoding, and `enforce_eager=True`, runs 3 warmups and 3 measured repetitions, and writes environment/config/preflight/events/metrics/summary/log paths. Variant selection controls `FLUXMOE_ENABLE`, backend placement, planner mode, and 12.5% whole-layer offload. `classify_support()` returns `INCONCLUSIVE` before comparing performance whenever strict correctness or mechanism counters are missing; otherwise it applies the trend rules from the spec and returns `SupportClassification` with explicit reasons.

Each YAML must include model path, dataset path/hash, batch, context, output length, TP, eager flag, `gpu_memory_utilization`, warmups, repetitions, and variant. Every compared variant uses the same value. The capacity-stressed H100 regime uses `gpu_memory_utilization: 0.60` (approximately 48GB of an 80GB H100); a supplemental native-H100 diagnostic uses `0.90` and is reported separately. Key-matrix script runs `(32,1024)`, `(128,4096)`, and `(256,4096)` in the 0.60 regime before any full sweep. If the resident baseline cannot initialize under the shared budget, record the initialization OOM as a capacity result rather than changing its budget.

The report CLI exposes four concrete subcommands: `latest --root runs` prints the newest complete run directory; `validate RUN_DIR` checks evidence; `build RUN_DIR --output FILE --figures DIR` generates the report; and `validate-report FILE` checks that every referenced artifact exists and every conclusion is backed by a valid run.

- [ ] **Step 5: Implement safe server container scripts**

`run_container.sh` requires explicit `GPU_IDS`, confirms project root equals `/home/jovyan/wangtonghan/moe-flex`, mounts the project root read-write and `/mnt/public_data:/mnt/public_data:ro`, uses `--ipc=host`, and sets the working directory to the project root. It refuses fewer than 4 GPU IDs for formal configs. `python -m flexmoe.runtime.preflight select-gpus --count 4` prints comma-separated idle H100 IDs and exits non-zero unless four are available.

`preflight.sh` writes `preflight.json`; `run_smoke.sh` runs unit/CUDA/tiny integration tests; `run_key_matrix.sh` runs only after smoke status is successful.

`build.sh` computes `IMAGE_TAG=$(git rev-parse HEAD)` and pulls `ghcr.io/rh2844417786/moe-flex:${IMAGE_TAG}`. It verifies the image label `org.opencontainers.image.revision` equals the checkout SHA and fails instead of silently using `latest` or installing packages into `wth333`.

- [ ] **Step 6: Run CPU telemetry/report tests and shell syntax checks**

Run: `pytest tests/unit/test_telemetry.py tests/unit/test_report.py -q`

Run: `bash -n scripts/server/*.sh`

Expected: PASS.

- [ ] **Step 7: Commit benchmark and server workflow**

```bash
git add src/flexmoe/runtime/telemetry.py src/flexmoe/bench benchmarks/configs scripts/server tests
git commit -m "feat: add evidence-gated benchmark workflow"
git push origin repro/fluxmoe
```

---

### Task 13: Execute server smoke gates and Qwen3-Next correctness

**Files:**
- Runtime output only: `runs/{timestamp}-{git_sha}/`
- Optional small summary update: `docs/results/h100-smoke-summary.md`

**Interfaces:**
- Consumes: public `repro/fluxmoe` branch, GHCR image, read-only model, 4 exclusive GPUs.
- Produces: verified server SHA, CUDA test outputs, full-model parity, and structured evidence.

- [ ] **Step 1: Clone or fast-forward the exact execution branch on the server**

```bash
git clone --branch repro/fluxmoe --single-branch \
  https://github.com/rh2844417786/moe-flex.git \
  /home/jovyan/wangtonghan/moe-flex
git -C /home/jovyan/wangtonghan/moe-flex rev-parse HEAD
```

For an existing checkout use:

```bash
git -C /home/jovyan/wangtonghan/moe-flex pull --ff-only origin repro/fluxmoe
```

Expected: server SHA equals the pushed execution SHA.

- [ ] **Step 2: Build/pull the image and run preflight on four exclusive GPUs**

```bash
cd /home/jovyan/wangtonghan/moe-flex
export GPU_IDS="$(python -m flexmoe.runtime.preflight select-gpus --count 4)"
GPU_IDS="$GPU_IDS" scripts/server/build.sh
GPU_IDS="$GPU_IDS" scripts/server/preflight.sh
```

Expected: 4 H100s, zero unrelated compute processes, model index/41 shards/BF16 architecture valid, `/mnt/public_data` mounted read-only, vLLM commit exact, VMM supported.

- [ ] **Step 3: Run CUDA, sanitizer, and tiny FusedMoE smoke gates**

```bash
GPU_IDS="$GPU_IDS" scripts/server/run_smoke.sh
```

Expected: VMM, RAW/WAR, Huffman, storage, patch, and tiny parity tests pass; mapped/HtoD/decompressed bytes are non-zero.

- [ ] **Step 4: Run full Qwen3-Next greedy parity on a small batch**

Run resident and `fluxmoe-fixed` with batch 4, context 128, output 16, fixed prompt IDs, and TP=4.

Expected: reconstructed expert bits, router Top-k IDs, and generated token IDs match; no OOM/error; FluxMoE telemetry is complete.

- [ ] **Step 5: Record actual status without overstating it**

If all steps pass, write `docs/results/h100-smoke-summary.md` as `MECHANISM VERIFIED - PERFORMANCE NOT YET EVALUATED`. If any required evidence is missing, write `INCONCLUSIVE` with the exact failing command and log path.

- [ ] **Step 6: Commit only the small smoke summary from Mac after reviewing server output**

```bash
git add docs/results/h100-smoke-summary.md
git commit -m "docs: record H100 mechanism smoke evidence"
git push origin repro/fluxmoe
```

---

### Task 14: Run the three key performance points and publish the evidence-bounded report

**Files:**
- Runtime output: `runs/{timestamp}-{git_sha}/`
- Create: `docs/results/h100-cross-hardware-report.md`
- Create: `docs/results/figures/throughput-key-points.png`
- Create: `docs/results/figures/residency-trace.png`
- Modify: `README.md`

**Interfaces:**
- Consumes: six benchmark variants and complete correctness/telemetry evidence.
- Produces: `SUPPORTED`, `MIXED`, `NOT_SUPPORTED`, or `INCONCLUSIVE` H100 cross-hardware conclusion.

- [ ] **Step 1: Run the three key points**

```bash
cd /home/jovyan/wangtonghan/moe-flex
export GPU_IDS="$(python -m flexmoe.runtime.preflight select-gpus --count 4)"
GPU_IDS="$GPU_IDS" scripts/server/run_key_matrix.sh
```

Expected: each variant/key point writes 3 warmups, 3 measured repetitions, and complete environment/config/events/metrics files.

- [ ] **Step 2: Validate evidence before comparing performance**

Run: `export RUN_DIR="$(python -m flexmoe.bench.report latest --root runs)" && python -m flexmoe.bench.report validate "$RUN_DIR"`

Expected: no variant marked valid without bit-exact weights, matching Top-k/tokens, non-zero mechanism counters, exact git/model/data manifests, and exclusive GPU evidence.

- [ ] **Step 3: Generate the report and figures**

Run: `python -m flexmoe.bench.report build "$RUN_DIR" --output docs/results/h100-cross-hardware-report.md --figures docs/results/figures`

The report must state that H100 is not the paper's L40 testbed, show resident/offload/fixed/dynamic/unbalanced/PagedTensor comparisons, identify reasonable small-batch decompression tax, show whether stressed workloads improve or delay the memory bottleneck, and separate reclaimed memory from actual KV-cache growth.

- [ ] **Step 4: Decide whether a full matrix is warranted**

Run the full batch/context sweep only when the three key points are valid and at least one stressed point is informative. Otherwise stop with the evidence-bounded status; do not spend GPU time to fill a decorative matrix.

- [ ] **Step 5: Verify report integrity and repository cleanliness**

Run: `python -m flexmoe.bench.report validate-report docs/results/h100-cross-hardware-report.md`

Run: `ruff check . && mypy src/flexmoe && pytest tests/unit -q`

Run: `git diff --check && git status --short`

Expected: report references existing small artifacts, no raw run/model/cache files are staged, and all CPU checks pass.

- [ ] **Step 6: Commit and publish the final evidence-bounded report**

```bash
git add README.md docs/results
git commit -m "docs: report FluxMoE H100 cross-hardware results"
git push origin repro/fluxmoe
```

The final user-facing statement must distinguish local code, GitHub publication, server checkout, CUDA mechanism tests, full-model correctness, and performance evidence.
