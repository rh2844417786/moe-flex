# Hot Expert Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现可调热专家驻留、跨层共享持久缓存和完整离线 H100 实验入口。

**Architecture:** CPU 纯策略/校准模型与 tensor/stream 运行时分离。连续 GPU 专家池由原生 vLLM fused_experts 的 expert_map 访问；原路由不变。扩展现有 benchmark 的 backend 选择，保留 R/B/C、隐私导出和真实 KV 验证。

**Tech Stack:** Python 3.10+、PyTorch 2.8.0、vLLM 0.10.2 固定 commit、pytest、原有离线 Docker。

**Spec:** docs/superpowers/specs/2026-09-06-hot-expert-cache.md

## Global Constraints

- Worktree: `/Users/a1234/code/flexmoe/.worktrees/repro-fluxmoe`, branch `repro/fluxmoe`; base server commit `74a4b87`.
- No new dependency, model/data download, pruning, quantization, rerouting or extra GPU memory. Server network is limited to user-authorized GitHub sync and fetching the exact pinned Docker Hub base image if absent; Docker build RUN steps remain offline.
- GPU: 4 exclusive H100, TP4, eager, compilation level=0, original BF16 weights and routing.
- Server outputs remain under `/home/jovyan/wangtonghan/moe-flex`; public weights/data are read-only.
- CPU tests are not H100 evidence; all hardware performance remains pending until server runs.
- Use apply_patch for file edits; TDD then targeted tests; preserve unrelated server changes; no push by implementer agents.
- Use Chinese documentation. Preserve old partial-host path and APIs unless an additive, tested extension is required.

---

### Task 1: CPU cache policy and calibration profile

**Files:** Create `src/flexmoe/runtime/expert_cache_policy.py`, `src/flexmoe/runtime/expert_profile.py`, `tests/unit/test_expert_cache_policy.py`, `tests/unit/test_expert_profile.py`.

**Interfaces:**
- `ExpertCachePolicy(total_layers: int, num_experts: int, resident_ratio: float, cache_slots: int, decay_interval: int=64, policy: str='decayed-lfu', initial_counts: Sequence[Sequence[float]] | None=None)`.
- `.resident_ids: tuple[tuple[int,...],...]`, `.observe(layer:int, expert_ids:Sequence[int]) -> None`, `.resident_slot(layer:int, expert:int) -> int | None` (dense prefix layer*num_resident + local rank).
- `.lookup(layer:int,expert:int) -> int | None` returns relative persistent cache slot. `.admit(layer:int,expert:int, protected:set[tuple[int,int]]) -> tuple[int, tuple[int,int] | None] | None` returns relative slot and optional evicted key; mutation is planned synchronously, runtime protects live values until ordered copies complete. No admission if newcomer score is not strictly better than victim; fill empty slots unconditionally. LRU always admits and uses monotonic access serial.
- `.reconfigure(resident_ratio:float, cache_slots:int) -> None` clears persistent assignments, selects resident from current heat, preserves observed heat; caller handles device synchronization and refilling. `.stats() -> dict[str,int|float]` returns policy counters.
- `ExpertProfile` typed JSON model plus `.validate(...)`, `.to_dict()`, `.from_dict(...)`, and deterministic `select_resident_ids(counts, ratio)`. Geometry, model identity and calibration input hashes must be validated by profile consumer; hashes/unique demand counts must be representable without private prompts.

- [ ] Write real behavioral tests, including the hand-derived example below; run RED before implementing.

```python
p = ExpertCachePolicy(2, 4, 0.25, 1, initial_counts=[[9, 3, 2, 1], [8, 2, 1, 0]])
assert p.resident_ids == ((0,), (0,))
assert p.resident_slot(1, 0) == 1
assert p.admit(0, 1, set()) == (0, None)
assert p.admit(1, 3, set()) is None
assert p.lookup(0, 1) == 0
```

- [ ] Implement finite bounded config validation, duplicate-insensitive observation, lazy or bounded decaying heat, misses counted, layer-order-aware same-score victim selection, protected entries, deterministic ties, reconfiguration safety and LRU control.
- [ ] Test repeated IDs add one demand observation, all-protected cache bypasses, equal expert numbers in different layers remain distinct, a changing workload replaces stale hot entries, resident entries never enter eviction, ratio boundaries/NaN and corrupt/mismatched profiles fail.
- [ ] Run `.venv/bin/python -m pytest tests/unit/test_expert_cache_policy.py tests/unit/test_expert_profile.py -q`; `.venv/bin/ruff check` on new files, `.venv/bin/mypy src/flexmoe/runtime/expert_cache_policy.py src/flexmoe/runtime/expert_profile.py`; commit only task files and report RED/GREEN evidence.

### Task 2: Expert-addressed GPU pool, vLLM hook, calibration RPC

**Files:** Create `src/flexmoe/vllm/expert_cache.py`, `src/flexmoe/runtime/expert_pool.py`, `src/flexmoe/vllm/expert_calibration.py`, `tests/unit/test_expert_cache_runtime.py`, `tests/cuda/test_expert_cache.py`; modify `src/flexmoe/vllm/bridge.py`, `patches/vllm-v0.10.2.patch`, `third_party/vllm.lock.json`, `docker/Dockerfile` and relevant patch integration tests.

**Interfaces:** consumes Task 1 policy/profile. Produces storage mode `expert-cache` under `FLUXMOE_ENABLE=1`; env `FLUXMOE_RESIDENT_RATIO`, `FLUXMOE_CACHE_SLOTS`, `FLUXMOE_CACHE_POLICY`, `FLUXMOE_EXPERT_PROFILE_PATH`, `FLUXMOE_MODEL_PATH`. Startup profile is required for real server expert-cache mode. `FLUXMOE_EXPERT_CALIBRATION=1` enables opt-in route capture for resident calibration. Worker RPCs: `fluxmoe_expert_cache_stats(synchronize=True, reset_timing=False)`, `fluxmoe_expert_calibration(action: str)` (start/stop), `fluxmoe_expert_cache_reconfigure(resident_ratio:float)`. Registry `.forward(layer_name, hidden_states, topk_weights, topk_ids, *, activation, apply_router_weight_on_input) -> Tensor` called immediately after native Top-k selection, before backend-specific fused_experts branches.

- [ ] Add tests first using actual CPU tensors and a synchronous tensor backend only in tests; demonstrate cold miss, cached repeat without H2D, cross-layer collision, full expert demand, resident selection and bounded memory accounting.
- [ ] Implement a single contiguous pool: all layer resident slots + persistent cache slots + one layer's `num_experts` ingress slots. CPU BF16 originals and reusable pinned gather ingress are separately accounted. No per-layer full GPU materialization. Assemble one map of length num_experts, logical IDs remain unchanged.

```python
result = fused_experts(hidden_states, pool.w13, pool.w2,
                       topk_weights, topk_ids, inplace=False,
                       global_num_experts=num_experts, expert_map=expert_map,
                       activation=activation,
                       apply_router_weight_on_input=apply_router_weight_on_input)
```

- [ ] Gather only misses on CPU, transfer w13/w2 in two contiguous H2D copies, map misses to ingress; promote admitted misses with device copies ordered before ingress reuse. Pin current live cached keys during planning. Retain entries across model cycles and requests. A single ingress buffer must wait for prior H2D before CPU refill and for consumers before GPU overwrite. Keep map tensor copies and mutations ordered on compute stream.
- [ ] Profile records every layer unique routed expert occurrence after explicit RPC start only, with no dummy/profile runs. Stop returns rank-local counts/forward counts; build schema artifacts in Task 3. Trace collection must work with native resident weights (`FLUXMOE_ENABLE=0`).
- [ ] Handle quiescent reconfigure under fixed physical pool capacity: synchronize, validate new ratio and >=1 persistent slot, clear cache, rebuild resident prefix from heat, reload residents. No KV resize; reject reconfigure while a forward is active. Expose exact per-rank model-pool counters, source/pinned/ingress/cache/resident bytes, unique coverage, hits/misses, admissions/evictions/bypasses, transfers, forward counts and weight-verification scope.
- [ ] Add explicit new-mode branch before legacy before/after_forward wrapper so old registry is never accidentally accessed. Guard unsupported model/config/backend behavior. Patch remains one upstream file. Update lock and Docker patched SHA from applying patch to exact upstream layer source, not guessed constants.
- [ ] CUDA tests use vLLM fused_experts on identical BF16 weights and Top-k to compare mapped pool vs native across 4 TP shard indices, two layers, cache churn and full coverage. Test safe reuse on non-default stream; exact BF16 weight copies and numerically matching outputs, no missing experts. CPU checks run locally; CUDA tests skip cleanly without CUDA.
- [ ] Run targeted tests, full unit suite, static checks and real patch apply/compile check; commit and report RED/GREEN and CUDA-not-run boundary.

### Task 3: Calibration/benchmark workflow and server handoff

**Files:** Modify `src/flexmoe/bench/partial_runner.py`, `src/flexmoe/bench/partial_suite.py` additively or factor shared benchmark helpers where needed; create `src/flexmoe/bench/expert_cache_runner.py`, `src/flexmoe/bench/expert_cache_suite.py`, `scripts/server/run_expert_cache.sh`, `docs/expert-cache-runbook.md`, `docs/server-codex-expert-cache-prompt.md`, `tests/unit/test_expert_cache_bench.py`.

**Interfaces:** consumes Task 2 env and RPC. User entrypoint `bash scripts/server/run_expert_cache.sh calibrate|confirm|export` with `--suite-id`, `--resident-ratio`, `--cache-slots`, `--cache-policy`, `--profile-path`, dataset/model paths and existing equal-budget workload flags. Dataset default stays committed ShareGPT; latest SWE-bench custom dataset option must remain usable without network.

- [ ] Write unit tests for calibration/evaluation split disjointness by actual token hashes, profile model/TP/geometry mismatch rejection, CLI forwarding, new mode env, actual fixed KV equivalence, counter deltas, malformed export and falsely claimed gains.
- [ ] Implement calibration on native resident: deterministically select held-out calibration examples, explicit start/stop RPC around measured calibration workload, exclude dummy profiling, record provenance plus rank counts. Run calibration before profile-required expert cache instantiation. Evaluation excludes every calibration input hash, then repeat only the remaining evaluation pool to reach request count. Abort on empty/overlapping split.
- [ ] Integrate expert backend into existing R/B/C generation and timing mechanics through explicit parameters/hooks; do not copy the entire runner or monkeypatch functions. R is native resident. B uses exact R allocated KV bytes/blocks; C expands KV under the same utilization. Record profile hash, ratio and all requested cache settings, plus actual worker evidence. Engine policy/inputs/model/budget must match between corresponding R/B/C except intentionally declared expert/KV policy.
- [ ] Preserve all counters in private summaries; whitelist export numeric cache evidence, per-rank capacity and hashes without prompts, generated text, weight paths or machine identity. Make invalid/missing evidence explicit; CSV and report ratios derive from actual fixed output tokens and measured seconds. A policy may be mechanically valid but slower; report it plainly.
- [ ] Write server script following existing preflight/container/offline patterns with timeout and separate output dirs; support one calibration then 0.60 confirmation for ratios 0.50/0.75/0.90 using explicit separate suite IDs. Initial smoke stops on correctness failure, no full matrix blind run. LRU ablation uses same capacity/layout. 0.90 only after 0.60 is understood. No destructive cleanup or automatic unbounded retries.
- [ ] Chinese Codex CLI prompt: check clean worktree and SHA, offline build, select 4 exclusive GPUs, run CUDA tests, prepare/validate calibration, smoke/confirmation, diagnose safe local integration errors if needed, export only sanitized results under docs/results, commit/push back to same branch with reproducible executed SHA. Results failures must also be returned; no successful-throughput claim before actual evidence.
- [ ] Run unit/CLI/export tests, full CPU suite, ruff and mypy, bash syntax, wheel build and patch contract checks. Commit only task files and report evidence.

## Completion

- [ ] Independent whole-change review and covering fixes.
- [ ] Fresh local verification, git diff/check/status, explicit push to `origin repro/fluxmoe`, verify remote SHA.
- [ ] Give user the pushed SHA, actual tests passed, H100 pending boundary, and server Codex prompt path.
