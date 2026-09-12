# Decode Mechanism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实测实际decode batch的专家激活次数/覆盖、真实卸载缺失时间与实际KV容量的吞吐收益。

**Architecture:** 复用现有真实ExpertPool和shared runner，小的显式新入口连接有界histogram collector、每层真实加载计时和固定KV对照；标准库host模块负责严格汇总/配对和脱敏交接。旧工具默认行为保持不变。

**Tech Stack:** Python3.10+、已有PyTorch2.8/vLLM0.10.2、zstandard/transformers tokenizer（Mac已有）；无新服务器依赖。

**Spec:** docs/superpowers/specs/2026-09-12-decode-mechanism.md

## Global Constraints

- Worktree `/Users/a1234/code/flexmoe/.worktrees/repro-fluxmoe`, branch `repro/fluxmoe`, pulled base `73939e57baf35ef680884d006068ea37c7074afb`.
- Existing Qwen3-Next-80B-A3B-Instruct, BF16, original routing, four exclusive H100, TP4; no new weights or server dependency downloads.
- Server writes only `/home/jovyan/wangtonghan/moe-flex`; `/mnt/public_data` read-only. Mac dataset preparation is approved. Preserve historical data/results and old runner defaults.
- Actual pure-decode batch is not submitted request count or cumulative generated tokens. No silent repeated inputs or fabricated missing telemetry. Timing must be real runtime evidence, not microbenchmark prediction.
- Separate instrumented profiles from throughput runs; retain native0.90 baseline and matched eager control when needed. All timing/memory units explicit; GB decimal.
- TDD/apply_patch/scoped commits. No subagent push or subagents-of-subagents. H100 execution remains external; controller owns final authorized GitHub push.

### Task 1: Unique corpus and standard-library activation contracts

**Files:** create `src/flexmoe/analysis/decode_counts.py`, `src/flexmoe/datasets/decode_corpus.py`, `scripts/prepare_decode_corpus.py`; tests `tests/unit/test_decode_counts.py`, `tests/unit/test_decode_corpus.py`. Generated new corpus/manifests under `benchmarks/data/decode-mechanism/`; never overwrite ShareGPT files. Controller generates actual corpus after helper is ready; include CLI/API instructions in report.

**Interfaces:** `validate_activation_row(row, *, layers, experts, top_k) -> dict` accepts `{step:int,layer:int,phase:str,actual_batch:int,histogram:list[int]}` with nonnegative unique indices/counts and allowed phases. `summarize_activations(rows, *, target_batch, layers, experts, top_k, min_steps=64) -> dict` counts only complete all-layer puredecode steps with actual_batch=target; outputs `accepted_steps`, `status`, `per_layer` (rows contain `layer`, `histogram_total`, `coverage_mean`, `coverage_p50`, `coverage_p95`, `coverage_max`, `token_count`, `selection_count`), plus phase/batch denominators. A row represents token→expert occurrences, not bool union. `prepare_decode_corpus(source, tokenizer, output, manifest_path, *, lengths=(1024,4096), unique_per_length=1536, seed=20260912) -> dict` reuses fixed packing/source cleaning with bounded deterministic candidate generation and full token-hash dedup, fails on insufficient unique data; manifest compatible with existing verify_subset plus unique_by_context, no_repetition=true, source/packing provenance. CLI accepts explicit local source/tokenizer paths and source-manifest; no network side effects. `load_unique_workload(dataset, manifest, *, context_length, request_count, calibration_count=32, selection_offset=0) -> (prompts, calibration, metadata)` preserves deterministic disjoint hashes and rejects requests beyond unique evaluation pool rather than cycling. The calibration is the first unique rows in file order, matching existing split_workload; selection_offset applies only to the remaining evaluation pool, with no wrap.

- [ ] RED/GREEN histogram behavior with literal two-layer/topk2/batch2 rows; `[2,1,1,0]` selects4 token-expert edges and covers3 experts, not4; accumulate two steps but coverage is per-step, not final union. Reject bools/NaN/negative counts, wrong sum, wrong length, duplicate/missing layer, phase/batch disagreement. Mixed/prefill rows are counted as excluded, never relabeled; unreachable target emits no accepted statistics.

```python
rows = [dict(step=s, layer=l, phase='decode', actual_batch=2,
             histogram=[2,1,1,0] if s == 0 else [0,1,1,2])
        for s in (0,1) for l in (0,1)]
out = summarize_activations(rows, target_batch=2, layers=2, experts=4, top_k=2, min_steps=2)
assert out['accepted_steps'] == 2
assert out['per_layer'][0]['histogram_total'] == [2,2,2,2]
assert out['per_layer'][0]['coverage_mean'] == 3
```

- [ ] RED/GREEN corpus real JSON/zstd roundtrip, deterministic hash/dedup, duplicate token sequences with distinct request IDs rejected/removed, no source wrap presented as new unique input, insufficient pool error, calibration disjointness, request_count1000 only when enough unique prompts. Use small real file fixtures + stub tokenizer external boundary, not network mocks. Keep old prepare_subset behavior unchanged.
- [ ] Provide local-only CLI tested via subprocess, verify counts/hashes and size<100MB before promoting fresh outputs; source/tokenizer provenance explicit. Run focused tests and source Ruff/Mypy, full CPU once, commit/report with exact downstream APIs and RED/GREEN. Do not commit raw source or tokenizer/weights.

### Task 2: Real bounded decode profiling and fixed-KV benchmark adapter

**Files:** create `src/flexmoe/vllm/decode_trace.py`, `src/flexmoe/bench/decode_mechanism_runner.py`; focused modifications `vllm/bridge.py`, `runtime/expert_pool.py`, optionally `vllm/expert_cache.py`, small shared hooks only if necessary. Tests `test_decode_trace.py`, `test_decode_mechanism_runner.py`, `test_decode_pool_timing.py`. Preserve pinned kernel/patch unless controller identifies an unavoidable explicit observer integration requirement.

**Interfaces:** Task1 activation dictionaries and strict unique workload. New worker RPC `fluxmoe_decode_mechanism(action='start'|'stop'|'status', ...)` produces per-rank bounded activation rows and observed actual batch/phase/step spans, plus real pool event observations in profile mode. Explicit env `FLUXMOE_DECODE_MECHANISM=1`; branch existing calibration gateway before old analysis collector, otherwise untouched. New runner CLI `--mode native|matched-resident|offload --profile` (flag defaults false), `--target-batch`, `--capture-steps` default256, `--min-capture-steps` default64, `--trace-budget-bytes` default134217728, `--kv-bytes` optional explicit experiment-only capacity, existing model/data/runtime flags, profile/cache settings. Uses shared fixed-token/smoke/warmup/failure lifecycle and ExpertBackend for genuine offload; actual collected contract is authoritative ABI for Task3.

New saved roots use existing `diagnostic_artifact` with kinds `decode-run`, `decode-profile`, `decode-repetition` and `decode-smoke`; `comparison_backend=decode-mechanism`, `evidence_kind=measured|instrumented` and actual `mode` are explicit. Pool profile rows identify `step`, `layer`, `actual_batch`, `phase`, `resident_hits`, `cache_hits`, `unique_misses`, `first_loads`, `reloads`, `loaded_bytes`, CPU timing mapping and CUDA timing mapping plus per-row measured/unavailable status. Source observers may add fields but must document exact ABI in task report before Task3; never relabel old prediction DTOs as measured rows. Inputs consumed by the runner contain no repetitions introduced by workload loading.

Detailed --profile is allowed only for matched-resident/offload (explicit eager). Reject native+profile rather than silently downgrading native. Native retains Graph-safe CPU prepared-input/scheduler observations without histograms. Task3 coverage commands therefore use matched-resident --profile; native unprofiled scans remain the optimized baseline. This avoids captured Python-selected histogram slots being replayed into the wrong step.

- [ ] RED/GREEN GPU histogram on CPU tensor test boundary: repeated topk IDs count multiplicity; hook start/stop excludes initialization/warmup; masks padded rows; partial/missing layers and targetbatch-unreached reported; captures only selected bounded puredecode steps while lightweight observed-batch counts continue. GPU fixed int32 buffers and post-stop D2H, no hot-path per-layer synchronize added. Preserve older boolean collector tests.
- [ ] Verify pinned vLLM CPU scheduling or stat logger boundary for native Graph actual-batch/KV usage/preemptions; model-only hooks must not silently miss Graph replay. Implement scoped wrapper/logger with reversible lifecycle; captured metadata is actual, unsupported values explicitly unavailable. Save source/evidence scope and use focused runtime doubles matching pinned structures. Controller is auditing this boundary in parallel.
- [ ] Add optional pool profile observer producing each layer's unique misses/loads, resident/cache hits, bytes, first-load/reload from lifecycle seen-state, eviction/bypass observations, CPU preparation waits and actual CUDA spans. Default off -> old counters/timing unchanged. Current on-compute-stream load/promotion is explicitly serialized; do not report fabricated prefetch overlap or sum CPU/GPU overlapping time. Fixed capacity, reset semantics, samples matched to layer/step and honest unmeasured/dropped count. Real events consumed only after ready/end synchronization. Keep profiling from altering route/admission behavior.

```python
# CPU-backed real pool with external CUDA backend double:
# first request for expert1 loads it; repeated cached request is a hit;
# after a real eviction, expert1 loads again and is classified reload, not cold.
# Assert real output parity and unchanged pool budget/loads with observer on/off.
# A supplied event quartet at 0,.002,.005,.006 seconds yields .002 load,
# .003 compute,.001 promotion; CPU gather .001 is not blindly added to GPU span.
```

- [ ] Implement unique-data native/eager/offload adapter, explicit KV only in new entry, actual BF16 policy validation, same four-worker UUID/version/model/input/profile identities. Native mode remains optimized and no trace timing substituted. Physical precheck and postmeasurement validate hook preserve numeric failed sample, independent smoke and incomplete profiles. KV requested must match actual worker bytes/blocks subject to disclosed allocator rounding; unequal ranks/mismatch fail closed. Cold/profile state must not be accidentally reset before every layer.
- [ ] Tests must drive actual shared runner + backend with only GPU/vLLM boundary faked for native, matched resident and offload, including KV propagation, default old gate unchanged, invalid profile/input, unreachable batch, mid-run failure, profile-vs-timing labeling, CPU peak evidence. Run relevant old cache/partial/native tests plus full CPU once, source Ruff/Mypy; commit/report exact saved output kinds and fields for Task3.

### Task 3: Measured comparison, safe server workflow and reports

**Files:** create `src/flexmoe/analysis/decode_suite.py`, `src/flexmoe/analysis/decode_report.py`, `scripts/server/run_decode_mechanism.sh`, `docs/decode-mechanism-runbook.md`, `docs/server-codex-decode-mechanism-prompt.md`; tests `test_decode_suite.py`, `test_decode_report.py`; small README links and formal-analyzer marker protection if required. Consume Tasks1/2 exact output ABI, no duplicate GPU backend.

**Interfaces:** `python -S src/flexmoe/analysis/decode_suite.py` commands plan/validate/summarize/compare/export. Standard-library bootstrap like existing analysis CLI; no torch import. Plan emits finite manual phases, not auto-executed: unique data+calibration, matched-resident --profile coverage batch1/16/50/100/200/500/1000 at1K with native unprofiled baseline then selected4K, limited actual offload pairs, explicit actual-KV sweep with total1024 unique requests/output512/repetitions3/common scheduler limits, profile overhead pair. Server GPU wrapper native|matched-resident|offload|calibrate reuses existing pinned container/clean SHA/offline/projectcache/exclusive4-GPU preflight and inside-container timeout; exact flags forwarded and new-ID paths checked canonically. calibration can call existing native calibration producer against new corpus.

- [ ] RED/GREEN native/eager/offload saved evidence adapters: require fixed output count, actual4-rank memory/KV/policy/identity, complete timing repetitions, real runtime labels. Profile runs can yield activation/timing observations but never headline throughput. A/B must same requested workload/actualKV/mode; B/C same cache/profile/mode/inputs but different actualKV; native0.90 final reference remains separate. If execution mode differs, expose engine tax, do not label it pure miss time. Failed/OOM/unreached/incomplete records never become zero TPS or no-benefit verdict.

```python
# Literal measured timings N=1000: matched resident10s, offload sameKV12s,
# offload largerKV9s, native reference8s; expect offload tax+2s,
# KV recovery3s, matched-net gain but no gain vs native. All are measured
# only if3 eligible repetitions, same contracts and valid actualKV evidence.
```

- [ ] RED/GREEN finite plan/custom args/actual batch gates; truthful cohort identity and no dataset cycling; GPU wrapper path/timeout/argument injection/dirty checks with actual shell recorder fixtures. Raw outputs `runs/decode-mechanism/<id>`, public `docs/results/decode-mechanism-<id>`. Preserve failed collector/pool/repetition rows and independent smoke even for malformed summaries. Existing tools/results unaffected.
- [ ] White-list JSON/CSV/Markdown includes bounded layer/expert histograms, denominators, coverage, batch distribution, miss reasons, per-rank measured service/exposed intervals, KV actual usage/allocated distinction and matched throughput. Unavailable telemetry and safe validation-error categories survive export; no prompts/token IDs/logits/UUID/paths/raw logs. Add simple deterministic SVG charts for batch-vs-coverage, miss-vs-wait, KV-vs-throughput (only actual data, correct units/status), optional standard-lib SVG generation without new dependencies; no fake curves for absent runs.
- [ ] Chinese docs provide exact staged commands, heldout calibration/profile pairing, unchanged same-SHA sequence, actual capacity selection after preflight, native/eager controls, sampling overhead checks, maxbatch-unreached and no speculative prefetch scope, local raw preservation and exact public files to push. Controller prepares/validates actual corpus using Task1 CLI and commits it before final verification/publication; docs must point at the real new paths.
- [ ] Full CPU tests, Ruff source/unit, strict Mypy source, shell syntax, -S command help/fixture roundtrips and wheel build; commit/report. Independent task review + final whole-feature review precede controller final verification and authorized ordinary push, no PR/main merge.
