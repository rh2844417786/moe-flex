# Offload Feasibility Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付可运行的原生访问采集、缓存回放、四卡传输微基准和吞吐盈亏分析工具，不伪装部署收益。

**Architecture:** 标准库 analysis 包定义单一数据合同、回放和成本逻辑；现有公共 benchmark 通过小的 backend hook 产生兼容数据；独立 CUDA 微基准测传输，host CLI 负责有限编排和脱敏报告。

**Tech Stack:** Python3.10+ stdlib、已有 PyTorch2.8/vLLM0.10.2、pytest、ruff、mypy；无新依赖。

**Spec:** docs/superpowers/specs/2026-09-08-offload-feasibility.md

## Global Constraints

- Worktree `/Users/a1234/code/flexmoe/.worktrees/repro-fluxmoe`, branch `repro/fluxmoe`, pulled base `b52252b`.
- Existing Qwen3-Next-80B-A3B-Instruct, BF16, original routing, four exclusive H100, TP4; no new model/data/dependency downloads.
- Server writes only `/home/jovyan/wangtonghan/moe-flex`; public data read-only; only GitHub sync and exact pinned Docker Hub image fetch may use network.
- No expert-cache/policy/pool/CUDA-kernel rewrites. Preserve old partial, expert-cache, KV Oracle and malformed/smoke-only failure recovery.
- All new analysis outputs `diagnostic_only=true`, `formal_offload_gain=false`, `deployment_gain_proven=false`; sampled timing is not throughput evidence and predictions are not measured gains.
- Normal deployment baseline defaults utilization=0.90 and native vLLM optimization; 0.60/eager/synthetic workload conclusions remain diagnostic.
- TDD, apply_patch edits, scoped commits, no subagent push or subagents-of-subagents. Mac tests are not H100 results.

### Task 1: Standard-library trace, cache replay and cost contracts

**Files:** Create `src/flexmoe/analysis/__init__.py`, `schema.py`, `replay.py`, `cost.py`; `tests/unit/test_analysis_schema.py`, `test_analysis_replay.py`, `test_analysis_cost.py`.

**Interfaces:** These names/fields are binding producers for Tasks 2/3. Use dataclasses with validated constructors and explicit `to_dict`/`from_dict` methods; small helper functions for finite numbers, IDs/hashes and atomic JSON/gzip are allowed in schema.py. Constructors reject bool-as-int and nonfinite floats. Keep modules focused; no new general framework.

```python
@dataclass(frozen=True)
class TraceEvent:
    step: int
    layer: int
    token_rows: int
    requests: int | None
    phase: str  # prefill|decode|mixed|unknown
    experts: tuple[int, ...]  # sorted, distinct IDs local to layer

@dataclass(frozen=True)
class DemandTrace:
    trace_id: str
    rank: int
    contract: dict[str, object]  # standard shared run contract; identity below
    total_layers: int
    num_experts: int
    top_k: int
    expert_bytes: int  # actual BF16 shard bytes for this rank
    observed_steps: int
    captured_steps: int
    full_workload: bool
    generated_tokens: int | None
    events: tuple[TraceEvent, ...]

@dataclass(frozen=True)
class ReplayConfig:
    resident_experts: tuple[tuple[int, ...], ...]  # one row per layer
    cache_slots: int
    staging_experts: int
    policy: str  # lru|decayed-lfu|future
    extra_gpu_bytes: int = 0

@dataclass(frozen=True)
class TransferSample:
    rank: int
    contract: dict[str, object]  # commit, versions, common four-GPU hardware_sha256
    experts_per_batch: int
    expert_bytes: int
    mode: str  # contiguous|fragmented|gather
    contention: str  # isolated|gemm-nccl-proxy
    repetition: int
    payload_bytes: int
    wall_s: float
    copy_s: float
    gather_s: float
    compute_s: float | None

@dataclass(frozen=True)
class TimingPoint:
    point_id: str
    contract: dict[str, object]
    generated_tokens: int
    elapsed_s: tuple[float, ...]
    kv_bytes_by_rank: tuple[int, ...]  # four ranks, actual bytes
    gpu_total_bytes_by_rank: tuple[int, ...]
    source_kind: str  # native-measured|kv-oracle-diagnostic
    engine_mode: str  # native|eager
```

Contract minimum: SHA256 model_identity_sha256/model_config_sha256/dataset_sha256/input_sha256; commit SHA40; tensor_parallel_size=4; positive batch_size/context_length/output_length/max_num_seqs/max_num_batched_tokens; gpu_memory_utilization finite (0,1]; engine_policy_sha256 SHA256; versions object with torch/vllm/cuda/vllm_commit; hardware_sha256 hash of ordered four-card UUID/capacity inventory (null only when explicitly unavailable); prompt_hashes nonempty sequence of per-request hashes (trace requires this). Preserve optional standard shared metadata only by explicit allowlist. Calibration checks same model/config/geometry/TP and disjoint prompt_hashes, not merely differing batch input_sha256. Timing reference pairing checks model/dataset/input/commit/TP/request count/lengths/seed if present/software/physical GPU totals and hardware identity. K utilization may exceed R's and native/eager/engine policy differences remain explicit counterfactual assumptions, never silently equivalent. Missing hardware identity produces insufficient-evidence for calibrated throughput prediction, not a guessed match.

`select_resident(calibration: DemandTrace, evaluation: DemandTrace, offload_fraction: float) -> tuple[tuple[int,...],...]` uses independent per-event frequency, ceil-resident rounding, deterministic expert-ID tie break. `replay_trace(trace: DemandTrace, config: ReplayConfig) -> ReplayResult`; ReplayResult must expose config, trace/model/input provenance, loaded_experts/loaded_bytes, resident_hits/cache_hits/demands, bytes_per_generated_token (null unless full_workload), net_freed_bytes, per-event load counts/bytes, per-layer/phase totals, reuse-gap summary and explicit assumptions. Field names may be expanded but not renamed without controller agreement. Store after-event cache content in test-observable result if needed as meaningful diagnostic, not private test methods.

`analyze_feasibility(traces: Sequence[DemandTrace], replays: Sequence[ReplayResult], samples: Sequence[TransferSample], baseline: TimingPoint, kv_reference: TimingPoint | None) -> dict[str, object]` validates four distinct ranks, geometry/trace/contract compatibility, completed trace normalization, real sample coverage and KV increment <= every rank's net freed bytes. Return rank-wise transfer service estimates, bottleneck rank, time headroom, optimistic full-overlap and serial estimates, prediction ranges/assumptions, missing evidence, and candidate/incomplete/model-no-headroom status; all fixed diagnostic labels. For each missing-expert count use deterministic greedy exact decomposition into measured experts_per_batch sizes <= staging_experts; record chunks/copy count, prefer largest sizes first, and missing evidence if no exact coverage (size1 normally covers remainder). No silent shape extrapolation or padded bytes pretending measured transfer. The two time scenarios are modelling assumptions, not strict hardware bounds. No-K still gives byte/bandwidth requirements, never invented speedup. Never average rank bottlenecks or use trace run elapsed as baseline.

- [ ] **RED / GREEN schema and exact trace semantics.** Add independent literal fixtures: two layers/four experts/top-k1, expert_bytes=100, full trace two steps (each layer present), generated_tokens=4. Reject missing/duplicate layer, step gaps, duplicate/out-of-range expert, too many unique IDs for token_rows*top_k, wrong rank/TP/hash, false full coverage, zero tokens, NaN/bools. Truncated trace keeps window statistics but no per-generated-token cost.
- [ ] **RED / GREEN cache persistence and accounting.** A hand fixture with resident expert0 per layer and demand `[0,1]` then `[1,2]` (with valid token_rows) must count initial resident hit, repeated cache hit, and load each missing expert once per event. Cross-layer identical expert ID is a different key; protect all current hits from within-event traversal effects. With zero cache repeated nonresident visits reload; with sufficient cache only first visits load. Verify LRU/decayed LFU/future deterministic behavior and future marker. Reject leaking calibration. Net freed is `(L*E - resident_count - cache_slots - staging_experts)*expert_bytes-extra_gpu_bytes`, preserving negative result. Add a case where token frequency is high but unique demand count remains one.
- [ ] **RED / GREEN numerical feasibility.** Literal baseline N=1000, times=(1,1,1), K=(2/3,2/3,2/3), actual K delta=200, replays netfree=300; matching four-rank calibrated total transfer service=0.2s gives headroom=1/3, serial predicted tps≈1153.846, optimistic=1500. Rank3 slower than other ranks must determine cost. Repeat at 0.4s -> serial937.5. Missing rank, unmatched shape, mismatched workload/model/SHA, trace-derived timing, deltaKV>netfree or no reference cannot produce a claimed gain. Synthetic/engine-mode differences are visible and diagnostics-only.
- [ ] **Verify / self-review / commit.** Run new focused tests, full-source Ruff/Mypy, existing CPU tests once, diff check. No torch import in analysis core; test via Python -S controlled import. Commit and write detailed task report with RED/GREEN and exact public APIs for downstream implementers.

### Task 2: Native baseline, bounded GPU route capture and four-rank transport measurements

**Files:** Create `src/flexmoe/vllm/analysis_trace.py`, `src/flexmoe/bench/analysis_runner.py`, `src/flexmoe/bench/transfer_microbench.py`; tests `test_analysis_trace.py`, `test_analysis_runner.py`, `test_transfer_microbench.py`. Modify `bridge.py` only for a trace RPC and existing record_calibration dispatch; `partial_runner.py` only for default-preserving engine-policy hook and necessary lifecycle metadata hooks. Pinned vLLM patch/checksums remain unchanged.

**Interfaces:** Consume Task1 schema types; persist `trace-rank-<r>.json.gz` via DemandTrace serialization, transport `samples.json` via TransferSample.to_dict and explicit measurement metadata, native `summary.json` using shared runner. New `AnalysisBackend(mode='native'|'eager'|'trace', max_trace_steps=1024, trace_budget_bytes=134217728, synthetic_context=False)` integrates shared run. CLI `python -m flexmoe.bench.analysis_runner --mode ...` takes existing model/data/run/root paths plus all standard numeric workload/utilization/seed flags. Transport CLI `python -m flexmoe.bench.transfer_microbench --project-root ... --run-dir ... --expert-bytes 1572864 --experts-per-batch 1,8,32,128,512 --modes contiguous,fragmented,gather --contention isolated|gemm-nccl-proxy --warmups 2 --repetitions 5`; optional memory cap/iterations/safety/timeout are explicit and validated. Fixed TP4, CUDA 0..3 inside mapped container.

- [ ] **RED/GREEN new backend with old-default preservation.** Add `BenchmarkBackend.resolved_policy(engine,args)` default to old strict `_resolved_policy`, change one shared call site. Analysis native requests enforce_eager=False and does not force level0 compilation; inspect/record actual vLLM resolved model/cache/scheduler/parallel/compilation settings and validate BF16/TP4/no CPU offload/no prefix caching/fixed workload/native intent. No fallback into eager on error. Eager/trace use existing engine args/resolver unchanged. Ordinary resident explicit-KV guard and old analyzer behavior remain strict. Save trace mode timing_eligible=false and actual profile/memory/versions/source hashes. Generate TimingPoint-compatible native summaries through an explicit conversion helper for Task3.
- [ ] **RED/GREEN bounded collector before hooking.** `AnalysisTraceCollector` owns root hook, step/phase/request metadata and fixed GPU occupancy buffer; record(layer,topk_ids) validates shapes/dtypes from metadata, stores expert presence without per-layer D2H/synchronize, stop copies once then validates canonical events. Root CPU runner metadata only; unavailable phase/request=null/unknown. Buffer limit checked before allocation against free memory and a 2GB reserve, max steps 1024 default; overflow marks truncated and tracks observed steps, incomplete layer coverage cannot be full_workload. Repeated start/stop cleanup, no duplicate hook, no capture outside window. CPU tests may use torch CPU tensors for real collector behavior; mock CUDA boundary only.
- [ ] **RED/GREEN runtime gateway.** In trace backend configure `FLUXMOE_ENABLE=0`, existing `FLUXMOE_EXPERT_CALIBRATION=1`, and new explicit analysis trace flag. Existing `bridge.record_calibration` dispatches to trace only under that flag, otherwise invokes original calibration unchanged. New worker RPC accepts start/stop/status with rank/model_runner and trace limits; avoid changing pinned patch. Start after warmup via before_measurement; serialize stop results after generate and GPU sync, bind shared contract/geometry/expert bytes. Trace buffer and instrumentation contaminate its timing by design, never eligible for throughput. Failed/truncated trace has status and evidence, not silently complete. Native/eager benchmark cannot accidentally leave capture flags enabled.
- [ ] **RED/GREEN workloads and hardware identity.** Reuse committed dataset verification/selection, disclose repeated inputs; implement opt-in deterministic packing from existing token rows for context>available bucket, mark synthetic and record source/input hashes. Native and trace with same flags choose byte-identical prompts. Support explicit selection offset and `--exclude-trace` prompt-hash exclusion so calibration/test remain disjoint even when evaluation repeats a smaller remaining pool; empty pool rejects. Bind the same ordered four-GPU UUID/capacity hash in all rank traces, native summaries and transfer sample contracts; use actual CUDA/device inventory or report null/unavailable, never invent UUIDs. Record versions/commit consistently.
- [ ] **RED/GREEN transport kernels and coordinator.** Pure helpers derive exact payload/shard shapes, validate positive limits and summarize four-rank results with wall bottleneck. CUDA execution uses pinned host tensors, contiguous copy, per-expert fragmented copy, CPU gather then bulk. Independent copy/compute streams, CUDA events and outer wall timestamps distinguish gather/copy/compute/total; barrier before timing, all synchronization before reading events. Optional GEMM+NCCL proxy runs actual four-rank all_reduce with deterministic payload, labelled proxy, reports compute-only and concurrent cost. Process group timeout, fresh port or supported file rendezvous under run dir, try/finally destroy and owned-child termination on failures; no leftover background workers. No new kernels or dependencies. Record rank/device/affinity hashes or unavailable plus memory cap; emit failed state atomically on OOM/error. Tests check real orchestration/aggregation with external CUDA/MP boundaries controlled, not mocks asserting themselves.
- [ ] **Verify / commit / report.** Run focused collector/backend/transport tests and existing touched consumers. Baseline parity under untouched default hooks remains covered. Full-source Ruff/Mypy, CLI help, diff checks. Report pinned-source assumptions and exact raw filenames/metadata conventions for Task3; no H100 success claim. Commit only task files.

### Task 3: Offline CLI, finite server workflow and evidence-bounded handoff

**Files:** Create `src/flexmoe/analysis/cli.py`, `io.py`, `report.py`, `scripts/server/run_offload_analysis.sh`, `docs/offload-analysis-runbook.md`, `docs/server-codex-offload-analysis-prompt.md`; tests `test_analysis_cli.py`, `test_analysis_report.py`. Small README links and explicit formal-analysis marker rejection in `partial_suite.py` as needed; no old Oracle-rule weakening.

**Interfaces:** Consume Tasks1/2 types, serialization helpers and exact measured output conventions from their reports. `python3 -S src/flexmoe/analysis/cli.py` works with no torch by a single controlled package bootstrap (stdlib package __init__ stays empty/light). Provide validate/replay/analyze/plan/export. `io.py` adapts shared native summaries into TimingPoint only after verifying complete count/tokens/seconds, hashes, actual four-rank KV and timing_eligible; reads existing Oracle via its own validation first and keeps over-budget label. report.py builds allowlisted JSON/CSV/Chinese Markdown without raw prompts/paths/logs/large traces. Analysis is never formal gain.

- [ ] **RED/GREEN real stdlib CLI path.** Subprocess tests invoke every help/validate/replay/analyze command via `python -S`, using actual tiny files. Check invalid inputs fail nonzero and no half-success report; complete native/trace/transport fixture flows produce correct cost/headroom; wrong modes, forged timing, malformed summaries, and data leakage remain incomplete/invalid. Do not merely grep source/commands.
- [ ] **RED/GREEN finite plan generation.** plan emits a manifest and shell argv (not auto-executed) for short/generation/medium native scan, concurrency16/32/64/128/256/512, utilization0.90 default, plus optional0.60 and explicit long synthetic stress. Include adjustable custom model/data paths, seed and numeric flags. Trace/calibration and microbench commands follow named selected points, not full Cartesian execution; machine-readable plan identifies phases requiring inspection. Deduplicate configs, fresh IDs, keep same execution SHA before result commits; failure cases represented.
- [ ] **RED/GREEN server execution wrapper.** resident/trace/transport modes call pinned container with offline env caches inside project, exact root, clean code except generated offload-analysis-* outputs, four exclusive-GPU preflight, timeout inside Docker. Forward custom flags accurately. Pure validate/replay/analyze/export may run host stdlib. No installing dependencies, fetching model/data, killing unrelated processes or silently reducing workload. Export all failures, preserve unreadable source/independent smoke and samples; output path only project's docs/results/offload-analysis-* for server wrapper. Test actual script argv and dirty guard with temp git fixtures where needed.
- [ ] **RED/GREEN honest adapters/report.** Wrong software/model/TP/inputs/lengths or four-rank coverage reject; sampled trace time cannot become native measured time. Native/eager mismatch and repeated/synthetic workloads remain visible limitations. Negative net freed KV and missing K-reference produce bytes-only/incomplete output. Existing formal partial exporter/analyzer must explicitly reject new offload-analysis marker BEFORE sanitization. Unit fixtures verify the diagnostic labels survive all branches, failed measurement peaks remain numeric evidence, no machine identity/private path/prompt can escape whitelist.
- [ ] **Docs, verify, commit.** Chinese runbook gives concrete build, native scan, selected calibration/eval trace, transport, replay/analyze/export commands with exact schemas and stable IDs. Server prompt asks for realistic normal-budget resident comparison, independent inputs, failure retention and GitHub return; no automatic rewritten offloader, no prediction-as-success. Note only1K–4K committed inputs, opt-in long packing limits, full captured trace requirement, proxy contention limits, and future real candidate validation. README links. Full CPU suite, Ruff/Mypy, shell syntax, stdlib help and wheel build; commit/report.

## Completion

- [ ] Independent per-task review and scoped fixes, whole-change architecture review.
- [ ] Controller fresh validation, push existing repro/fluxmoe (already user-authorized), verify remote SHA, give server prompt. No PR/main merge.
