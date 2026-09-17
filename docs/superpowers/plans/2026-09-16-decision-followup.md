# Decode Offload Decision Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a one-command, fail-closed TP4 H100 experiment that tests the zero-miss overhead and real KV-capacity advantage at low concurrency, and a separate exact-budget access replay; never label replay as real prefetch throughput.

**Architecture:** A host-only controller calls the existing offline Docker benchmark point wrapper sequentially and reads immutable point summaries. It selects actual auto-allocated KV budgets with one-block rounding, then runs measured, fixed-workload comparisons. A bounded instrumented low-batch trace feeds a CPU-only replay; public reports state which gates were reached and which hypotheses remain unmeasured.

**Tech Stack:** Bash, Python 3.10 standard library, existing vLLM 0.10.2 offline image and `flexmoe.bench.decode_mechanism_runner` on the server, pytest on Mac.

**Spec:** `docs/decode-decision.md`, which carries the user-requested zero-miss/KV/replay design and the exact Oracle evidence boundary.

## Global Constraints

- Repository branch: `repro/fluxmoe`; server project: `/home/jovyan/wangtonghan/moe-flex`; weight mount `/mnt/public_data` is read-only.
- `GPU_IDS=0,1,2,3`, TP4, BF16, pinned vLLM container; no new model weights or server network downloads.
- Total GPU utilization `0.90`, safety reserve `2_000_000_000` bytes per GPU, fixed workload and mode/KV for causal comparisons; warmup 1, measured repetitions 3.
- All logs and data stay under the server project; only compact public numeric conclusions can be committed subsequently. Preserve existing result files and fail rather than guessing a missing capacity, trace, rank, or timing boundary.
- If auto-allocated offload KV is not strictly above **both** native and eager-resident auto KV after block rounding, stop the large-KV claim. A zero-miss point is zero miss only if every measured repetition and all four ranks have zero payload bytes, zero cache misses and zero copy launches.

---

### Task 1: Checked decision analysis

**Files:**
- Create: `src/flexmoe/analysis/decode_decision.py`
- Test: `tests/unit/test_decode_decision.py`

**Interfaces:**
- Consumes: immutable `summary.json` and `rep-*.json` from `run_decode_mechanism.sh`.
- Produces: `_actual_kv(summary)`, `zero_miss_gate(summary)`, `capacity_gate(native, eager, offload)`, `compare_pair(resident, offload)`; JSON-safe public report with explicit unmeasured fields.

- [ ] **Step 1: Write failing tests** for all-rank zero-miss proof and KV strict growth:

```python
def test_one_rank_with_a_miss_rejects_zero_miss():
    from flexmoe.analysis.decode_decision import zero_miss_gate
    run = summary()
    run['repetitions'][2]['diagnostics']['per_rank'][3]['policy']['cache_misses'] = 1
    assert zero_miss_gate(run)['status'] == 'not-zero-miss'

def test_equal_kv_does_not_prove_extra_capacity():
    from flexmoe.analysis.decode_decision import capacity_gate
    assert capacity_gate(summary(mode='native'), summary(mode='matched-resident'), summary())['status'] == 'no-extra-capacity'
```

- [ ] **Step 2: Run** `pytest -q tests/unit/test_decode_decision.py` and verify missing interfaces fail.
- [ ] **Step 3: Implement** validated summary readers: demand exact four ranks, three completed measured repetitions, real `actual_kv.allocated_bytes_per_rank`, no ignored failed repetitions; numerical median, scheduler peak/preemptions and latency availability become labeled evidence, not guessed measurements.
- [ ] **Step 4: Re-run** `pytest -q tests/unit/test_decode_decision.py` and confirm both branches pass.

### Task 2: One-command server execution and resumability

**Files:**
- Create: `scripts/server/run_decode_decision.sh`
- Modify: `src/flexmoe/analysis/decode_decision.py`
- Test: `tests/unit/test_decode_decision.py`

**Interfaces:**
- Consumes: `--run-id`, optional `--resume`, existing offline `run_decode_mechanism.sh` point CLI.
- Produces: `runs/decode-decision/<ID>/state.json`, `docs/results/decode-decision-<ID>/report.md`; nonzero exit and retained diagnostics on a failed point.

- [ ] **Step 1: Write failing host-controller tests** asserting selected low-batch point arguments, capped concurrency 32, exact workload, 3 repetitions and rejection outside the authorized server root (`tests/unit/test_decode_decision.py`).
- [ ] **Step 2: Run** that test and verify the missing controller behavior fails.
- [ ] **Step 3: Implement** pinned offline image build for the new commit, profile calibration, native/eager-resident/offload auto-capacity probes, same-KV batch-1 eager resident/offload measured and instrumented contrasts, plus native at its own auto-KV and eager resident/offload at their common feasible KV for the low-batch 4K run. Run offload K1 only if it exceeds both resident auto capacities. Each point has a fresh ID; resume skips only validated complete points and never repeats or overwrites a failed one without an explicit new run ID. Record all point commands and outcomes in project-only state.
- [ ] **Step 4: Run** the controller tests and `bash -n scripts/server/run_decode_decision.sh`.

### Task 3: Exact-budget cache replay and truth labeling

**Files:**
- Create: `src/flexmoe/analysis/decode_replay.py`
- Modify: `src/flexmoe/runtime/expert_cache_policy.py`, `src/flexmoe/runtime/expert_pool.py`
- Test: `tests/unit/test_decode_replay.py`, `tests/unit/test_decode_trace.py`

**Interfaces:**
- Consumes: bounded batch 16/32 per-layer demand IDs and the policy’s true state at the beginning of the contiguous window.
- Produces: validated reproduction of observed current-policy misses; same resident/cache capacity LRU and future-aware cache victim comparison; no throughput/oracle-prefetch claim.

- [ ] **Step 1: Write a failing test** in which a seeded cache starts nonempty, future-aware eviction avoids one miss, and replay rejects a trace gap or a baseline miss mismatch.
- [ ] **Step 2: Run** `pytest -q tests/unit/test_decode_replay.py` and verify failure from absent behavior.
- [ ] **Step 3: Implement** start-state capture and bounded per-layer demand logging only in separate instrumented profiles (not the unprofiled timing pair); restore policy scores/clock/assignments exactly. Replay inside the pinned container (the `python3 -S` host has no torch), respecting protected simultaneous demands, persistent slots and ingress bypasses with fixed capacity. Host reads only the resulting numeric JSON. Mark invalid or partial traces as inconclusive.
- [ ] **Step 4: Re-run** replay and existing policy/trace tests, then broader CPU suite; report what remains to be done for a real router-ahead, H2D-inclusive, end-to-end Oracle prefetch test.

### Task 4: Publish and execute on H100

**Files:**
- Create: `docs/decode-decision.md`
- Test: `tests/unit/test_decode_decision.py`, `tests/unit/test_decode_replay.py`

- [ ] **Step 1: Verify** `git diff --check`, focused pytest and Bash syntax; review only intended paths in staged diff.
- [ ] **Step 2: Commit** and push `repro/fluxmoe` to GitHub; verify the remote hash.
- [ ] **Step 3: With Computer Use, connect to the existing server terminal**, inspect the prompt and exact server repository path, pull with `--ff-only`, confirm hash and idle exclusive GPUs, then run `GPU_IDS=0,1,2,3 bash scripts/server/run_decode_decision.sh`; do not infer success from invocation alone.
- [ ] **Step 4: Verify** point state/report and GPU run logs, retaining any failure diagnostics for follow-up.
