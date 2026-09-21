# TP4 Decode Oracle Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one fail-closed TP4 experiment that measures zero-miss runtime overhead, genuine KV capacity value at concurrency 32, exact-budget cache replay, and real horizon-0/1/2 Oracle H2D prefetch with omission fallback.

**Architecture:** Keep the existing immutable decode point runner and add focused modules for phase metrics, full cache traces, replay, and Oracle ingress transfer. Record an instrumented deterministic route trace first, then run separate uninstrumented Oracle points that execute the complete gather/queue/H2D/map/wait/compute path without changing resident, cache, ingress, KV, routing, or eviction budgets. The host controller remains standard-library-only and publishes conclusions only after checking source, workload, hardware, KV geometry, routes, counters, and output hashes.

**Tech Stack:** Python 3.10–3.13, PyTorch 2.8 CUDA APIs, pinned vLLM 0.10.2, Bash, pytest, ruff, mypy, offline Docker.

**Spec:** `docs/superpowers/specs/2026-09-20-decode-oracle-followup-design.md`

## Global Constraints

- Branch `repro/fluxmoe`; server root `/home/jovyan/wangtonghan/moe-flex`; model `/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct`; existing decode-mechanism dataset only.
- TP4, BF16, four exclusive H100s, `GPU_IDS=0,1,2,3`, pinned vLLM 0.10.2 and existing offline image flow.
- Only batch 1, actual batch 16/32, and a 64-request service load with `max_num_seqs=32`; no EP/EP+DP, high-batch grid, extra policy, extra horizon, predictor, model, or dataset.
- Formal timing comes only from uninstrumented repetitions. Per-rank CPU/GPU intervals remain per-rank and are never summed into a TP-global tax.
- Logical routes come from rank 0 only after all ranks prove identical IDs. Physical bytes/waits remain per rank.
- K1 must exceed both native and eager-resident stable maxima by at least one actual KV block. Missing, failed, rejected, and timed-out evidence remains unavailable rather than zero.
- Cache replay and prefetch Oracle remain separate. Replay cannot support throughput claims; Oracle must execute real host gather, transfer queue, H2D, mapping/materialization, ready wait, promotion, and original expert compute.
- Preserve user-owned untracked `.DS_Store` and `完整实验报告.md`; every `git add` lists exact paths.

---

### Task 1: Worker Phase Boundaries and Request-Latency Evidence

**Files:**
- Create: `src/flexmoe/vllm/decode_phase.py`
- Modify: `src/flexmoe/vllm/decode_trace.py`
- Modify: `src/flexmoe/bench/partial_runner.py`
- Modify: `src/flexmoe/bench/decode_mechanism_runner.py`
- Test: `tests/unit/test_decode_phase.py`
- Test: `tests/unit/test_decode_trace.py`
- Test: `tests/unit/test_decode_mechanism_runner.py`

**Interfaces:**
- Consumes: `step_metadata(model_runner) -> tuple[int | None, str, int | None]`, worker `time.perf_counter_ns()`, and vLLM output metrics.
- Produces: `PhaseTimeline.start()`, `PhaseTimeline.record(step, phase, actual_batch)`, `PhaseTimeline.stop() -> dict[str, object]`; `summarize_request_metrics(outputs, output_length) -> dict[str, object]`.
- The phase artifact contains `prefill_wall_time_s`, `decode_wall_time_s`, `decode_step_ms_p50/p95`, exact scope strings, and unavailable reasons. The request artifact contains TTFT, request latency, derived TPOT percentiles, and leaves ITL unavailable when token timestamps do not exist.

- [ ] **Step 1: Write failing phase-timeline tests**

```python
def test_phase_timeline_separates_prefill_and_decode_without_rank_summing():
    clock = iter((1_000, 2_000, 5_000, 8_000, 13_000))
    timeline = PhaseTimeline(clock_ns=lambda: next(clock))
    timeline.start()
    timeline.record(step=0, phase="prefill", actual_batch=2)
    timeline.record(step=1, phase="decode", actual_batch=2)
    result = timeline.stop()
    assert result["prefill_wall_time_s"] == 4e-6
    assert result["decode_wall_time_s"] == 3e-6
    assert result["phase_sequence"] == ["prefill", "decode"]
    assert result["scope"] == "per-rank monotonic worker wall; never summed across TP ranks"

def test_missing_decode_keeps_phase_metrics_unavailable():
    timeline = PhaseTimeline(clock_ns=iter((10, 20, 30)).__next__)
    timeline.start()
    timeline.record(step=0, phase="prefill", actual_batch=1)
    result = timeline.stop()
    assert result["decode_wall_time_s"] is None
    assert result["status"] == "incomplete-phase-boundaries"
```

- [ ] **Step 2: Run the phase tests and verify RED**

Run: `.venv/bin/pytest -q tests/unit/test_decode_phase.py`

Expected: import failure for `flexmoe.vllm.decode_phase`.

- [ ] **Step 3: Implement `PhaseTimeline` and attach it to `DecodeTraceCollector`**

Implement a fixed-size monotonic record that captures measurement start, first prefill, first decode, decode-step start deltas, last decode, and stop. Use the existing `_prepare_inputs` wrapper so profile-off points also observe phases. Do not allocate CUDA tensors or events when `profile=False`.

```python
class PhaseTimeline:
    def __init__(self, clock_ns: Callable[[], int] = perf_counter_ns) -> None: ...
    def start(self) -> None: ...
    def record(self, *, step: int, phase: str, actual_batch: int | None) -> None: ...
    def stop(self) -> dict[str, object]: ...
```

- [ ] **Step 4: Write failing request-percentile tests**

```python
def test_request_metrics_report_ttft_latency_and_derived_tpot_percentiles():
    outputs = fake_outputs(
        arrival=[0.0, 0.0], first=[1.0, 2.0], finished=[5.0, 8.0], length=5
    )
    result = summarize_request_metrics(outputs, output_length=5)
    assert result["ttft_s"]["p50"] == 1.5
    assert result["request_latency_s"]["p95"] == pytest.approx(7.85)
    assert result["derived_tpot_s"]["values"] == [1.0, 1.5]
    assert result["itl_s"] == {"status": "unavailable", "reason": "per-token timestamps absent"}
```

- [ ] **Step 5: Implement request metrics in `partial_runner.py`**

Add an interpolation percentile helper with hand-checked p50/p95/p99 behavior. Preserve existing median fields for backward compatibility and add `request_metrics`. Derived TPOT is `(finished-first)/(output_length-1)` only when all three timestamps are valid and output length exceeds one; never label it measured ITL.

- [ ] **Step 6: Persist and validate four-rank phase evidence**

Add `phase_timeline` to each worker observation. In `DecodeMechanismBackend.validate_measurement`, require four rank rows and identical phase sequences; expose per-rank values even when aggregate wall time is rejected. Extend `point_metrics()` to retain phase/request percentiles.

- [ ] **Step 7: Run focused tests and commit**

Run:

```bash
.venv/bin/pytest -q \
  tests/unit/test_decode_phase.py \
  tests/unit/test_decode_trace.py \
  tests/unit/test_decode_mechanism_runner.py \
  tests/unit/test_partial_runner.py
```

Commit:

```bash
git add src/flexmoe/vllm/decode_phase.py src/flexmoe/vllm/decode_trace.py \
  src/flexmoe/bench/partial_runner.py src/flexmoe/bench/decode_mechanism_runner.py \
  tests/unit/test_decode_phase.py tests/unit/test_decode_trace.py \
  tests/unit/test_decode_mechanism_runner.py tests/unit/test_partial_runner.py
git commit -m "feat: record decode phase and request latency evidence"
```

---

### Task 2: Exact Expert Cache Trace and Split Host Preparation Timing

**Files:**
- Create: `src/flexmoe/runtime/cache_trace.py`
- Modify: `src/flexmoe/runtime/expert_cache_policy.py`
- Modify: `src/flexmoe/runtime/expert_pool.py`
- Modify: `src/flexmoe/vllm/decode_trace.py`
- Test: `tests/unit/test_cache_trace.py`
- Test: `tests/unit/test_expert_cache_policy.py`
- Test: `tests/unit/test_decode_pool_timing.py`

**Interfaces:**
- Consumes: production `ExpertCachePolicy` and the actual unique expert IDs passed to `ExpertPool.execute`.
- Produces: `CacheTraceRow.to_dict()` with exactly `step_id`, `layer_id`, `actual_batch`, `actual_expert_ids`, `resident_hit_ids`, `cache_hit_ids`, `miss_ids`, `bypass_ids`, `admitted_ids`, `evicted_ids`, `cache_state_before`, `cache_state_after`, and `loaded_bytes`.
- Adds `ExpertCachePolicy.trace_snapshot() -> dict[str, object]` without weight payload and `AdmissionTrace` containing slot/victim information.

- [ ] **Step 1: Write the failing complete-row test**

```python
def test_pool_trace_records_exact_decisions_and_cache_states():
    pool, sources = observed_pool()
    attach_profile(pool, step=7, batch=16)
    invoke(pool, sources, layer=0, ids=[0, 1, 2])
    row = pool.profile_observer.snapshot()["rows"][0]
    assert row["step_id"] == 7
    assert row["actual_expert_ids"] == [0, 1, 2]
    assert row["resident_hit_ids"] == [0]
    assert row["miss_ids"] == [1, 2]
    assert row["admitted_ids"] == [1]
    assert row["bypass_ids"] == [2]
    assert row["cache_state_before"]["slots"] != row["cache_state_after"]["slots"]
    assert row["loaded_bytes"] == 2 * pool.expert_bytes
```

- [ ] **Step 2: Run the trace test and verify RED**

Run: `.venv/bin/pytest -q tests/unit/test_cache_trace.py`

Expected: `cache_trace` module or exact fields missing.

- [ ] **Step 3: Implement immutable trace schema and policy snapshots**

Use tuples internally and JSON lists at serialization. `trace_snapshot()` returns geometry, resident IDs, persistent slots, free slots, policy name, scores, scale, forward counts, decay events, access serial, and per-slot access; it excludes observer counters and weights. Add a traced admission method that returns the existing `(slot, victim)` plus a serializable victim decision without changing production order.

- [ ] **Step 4: Instrument the real decision points in `ExpertPool.execute`**

Capture `cache_state_before` immediately before `policy.observe`; classify resident/cache/miss IDs during existing lookups; capture admission/bypass/victim results from the existing loop; capture `cache_state_after` after all admissions. Do not perform a second policy lookup or admission for tracing.

- [ ] **Step 5: Split CPU timing boundaries**

Replace the ambiguous preparation region with three non-overlapping counters:

```python
"host_reuse_wait_s"   # backend.wait_host only
"weight_gather_cpu_s" # _gather only, zero when no ordered misses
"host_map_cpu_s"      # host_map assignment loop only
```

Retain `host_gather_s = weight_gather_cpu_s + host_map_cpu_s` as a compatibility field and attach exact scope text to the profile artifact.

- [ ] **Step 6: Add rank-0 logical trace validation**

In `decode_trace.stop`, hash each complete row’s `(step_id, layer_id, actual_batch, actual_expert_ids)`. `DecodeMechanismBackend` checks all four ranks have the same ordered hashes. Save rank 0 logical rows once in `decode-logical-trace.json.gz`; save rank-local physical timing/bytes in existing rank profiles.

- [ ] **Step 7: Run focused tests and commit**

Run:

```bash
.venv/bin/pytest -q \
  tests/unit/test_cache_trace.py \
  tests/unit/test_expert_cache_policy.py \
  tests/unit/test_decode_pool_timing.py \
  tests/unit/test_decode_trace.py
```

Commit:

```bash
git add src/flexmoe/runtime/cache_trace.py src/flexmoe/runtime/expert_cache_policy.py \
  src/flexmoe/runtime/expert_pool.py src/flexmoe/vllm/decode_trace.py \
  tests/unit/test_cache_trace.py tests/unit/test_expert_cache_policy.py \
  tests/unit/test_decode_pool_timing.py tests/unit/test_decode_trace.py
git commit -m "feat: capture exact expert cache decisions"
```

---

### Task 3: Exact-Budget Replay, Reuse Distance, and Layer Distributions

**Files:**
- Modify: `src/flexmoe/analysis/decode_replay.py`
- Modify: `src/flexmoe/bench/decode_mechanism_runner.py`
- Test: `tests/unit/test_decode_replay.py`
- Test: `tests/unit/test_decode_mechanism_runner.py`

**Interfaces:**
- Consumes: rank-0 `CacheTraceRow` sequence and its initial policy snapshot.
- Produces: `ReplayResult` dictionaries for `current`, `lru`, and `future-aware`, each containing totals, reuse-distance histogram/percentiles, per-layer miss counts, per-layer loaded bytes, and baseline reproduction status.
- `replay_trace(snapshot, rows, expert_bytes, layers, ingress_slots) -> dict[str, object]` remains the public pure-CPU boundary.

- [ ] **Step 1: Write failing baseline-reproduction tests**

```python
def test_replay_reproduces_every_recorded_transition_before_oracles():
    result = replay_trace(snapshot(), complete_rows(), expert_bytes=24, layers=2, ingress_slots=4)
    assert result["status"] == "baseline-reproduced"
    assert result["current"]["misses"] == 4
    assert result["current"]["bypasses"] == 1
    assert result["current"]["admissions"] == 3
    assert result["current"]["evictions"] == 2
    assert result["current"]["per_layer_misses"] == [1, 3]

def test_replay_stops_on_cache_state_mismatch():
    rows = complete_rows()
    rows[1]["cache_state_after"]["slots"] = [[0, 99]]
    result = replay_trace(snapshot(), rows, expert_bytes=24, layers=2, ingress_slots=4)
    assert result["status"] == "baseline-mismatch"
    assert "lru" not in result
```

- [ ] **Step 2: Run replay tests and verify RED**

Run: `.venv/bin/pytest -q tests/unit/test_decode_replay.py`

Expected: signature mismatch or missing exact transition fields.

- [ ] **Step 3: Implement exact current-policy transition replay**

Validate contiguous `(step, layer)` order, sorted unique IDs, protected same-layer working set, ingress upper bound, and `loaded_bytes == len(miss_ids) * expert_bytes`. Compare all recorded ID classes and both cache states after every layer. Stop at the first mismatch with repetition/rank/step/layer and differing field.

- [ ] **Step 4: Implement reuse-distance accounting**

Maintain last-use index for every `(layer, expert)`. Record first-use separately; for repeated uses record number of layer-forwards since last use. Export literal bucket counts (`1`, `2-4`, `5-16`, `17-64`, `65+`) and interpolated p50/p95/p99 over finite repeated distances.

- [ ] **Step 5: Extend LRU and future-aware replay**

Start from identical resident/cache occupancy and ingress capacity. LRU recency is measured only when the captured policy was LRU; for LFU traces label slot-order initialization as conditional. Future-aware replacement chooses the farthest next use among non-protected persistent entries and bypasses a new item when its next use is no sooner than the best victim’s next use.

- [ ] **Step 6: Persist bounded numeric replay results in the container**

Update `save_cache_replays()` to derive repetition count from `summary.json`, reject anything other than the decision protocol’s three repetitions, and write no raw routes into the public report. The host controller reads only numeric results.

- [ ] **Step 7: Run focused tests and commit**

Run:

```bash
.venv/bin/pytest -q tests/unit/test_decode_replay.py tests/unit/test_decode_mechanism_runner.py
```

Commit:

```bash
git add src/flexmoe/analysis/decode_replay.py src/flexmoe/bench/decode_mechanism_runner.py \
  tests/unit/test_decode_replay.py tests/unit/test_decode_mechanism_runner.py
git commit -m "feat: validate full cache replay transitions"
```

---

### Task 4: Oracle Route Artifact and Fail-Closed Alignment

**Files:**
- Create: `src/flexmoe/runtime/oracle_trace.py`
- Modify: `src/flexmoe/vllm/decode_trace.py`
- Modify: `src/flexmoe/bench/decode_mechanism_runner.py`
- Test: `tests/unit/test_oracle_trace.py`
- Test: `tests/unit/test_decode_mechanism_runner.py`

**Interfaces:**
- Produces `OracleTrace.from_rank0_profile(path, identity)`, `OracleTrace.future(step, layer, horizon) -> tuple[int, ...] | None`, and `OracleTrace.validate_actual(step, layer, ids) -> RouteMatch`.
- `RouteMatch` contains `status`, `predicted_ids`, `actual_ids`, `missing_ids`, and `unused_ids`.
- Serialized artifact carries commit, model/profile/input identity, geometry, selected step range, and logical route SHA-256; no prompt IDs, output IDs, paths, or weights.

- [ ] **Step 1: Write failing trace-alignment tests**

```python
def test_oracle_trace_returns_only_horizon_one_and_two():
    trace = OracleTrace.from_rows(rows_for_two_steps(), identity())
    assert trace.future(step=10, layer=3, horizon=1) == (4, 7)
    assert trace.future(step=10, layer=3, horizon=2) == (1, 9)
    with pytest.raises(ValueError, match="horizon"):
        trace.future(step=10, layer=3, horizon=3)

def test_route_mismatch_is_explicit_and_never_changes_actual_ids():
    trace = OracleTrace.from_rows(rows_for_two_steps(), identity())
    match = trace.validate_actual(step=10, layer=4, ids=(4, 8))
    assert match.missing_ids == (8,)
    assert match.unused_ids == (7,)
    assert match.actual_ids == (4, 8)
```

- [ ] **Step 2: Run trace tests and verify RED**

Run: `.venv/bin/pytest -q tests/unit/test_oracle_trace.py`

- [ ] **Step 3: Implement immutable Oracle trace loading and identity checks**

Accept only batch 16 or 32 traces with complete layers and at least 64 contiguous steps. Require exact commit, model identity, calibration profile hash, input hash, geometry, seed, context/output lengths, and rank-route hash agreement. A formal run may select one contiguous 64-step window; it must persist the selected start/end.

- [ ] **Step 4: Add deterministic forced omission selection**

`OracleTrace.with_forced_omission(step, layer, expert) -> OracleTrace` removes exactly one known demanded expert from one future set while preserving the actual route record. Reject an expert not present in that set and reject more than one omission.

- [ ] **Step 5: Add runner flags and contract fields**

Extend the runner with exact, non-abbreviated flags:

```text
--oracle-trace PATH
--prefetch-horizon {0,1,2}
--force-omission STEP:LAYER:EXPERT
```

Require offload mode, eager execution, profile off for performance, and a trace identity match before engine construction. Horizon 0 does not create a transfer queue but still validates actual routes against the trace.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
.venv/bin/pytest -q tests/unit/test_oracle_trace.py tests/unit/test_decode_mechanism_runner.py
```

Commit:

```bash
git add src/flexmoe/runtime/oracle_trace.py src/flexmoe/vllm/decode_trace.py \
  src/flexmoe/bench/decode_mechanism_runner.py tests/unit/test_oracle_trace.py \
  tests/unit/test_decode_mechanism_runner.py
git commit -m "feat: define fail-closed oracle route traces"
```

---

### Task 5: Bounded Oracle Ingress and Transfer Backend

**Files:**
- Create: `src/flexmoe/runtime/oracle_prefetch.py`
- Modify: `src/flexmoe/runtime/expert_pool.py`
- Test: `tests/unit/test_oracle_prefetch.py`
- Test: `tests/cuda/test_oracle_prefetch.py`

**Interfaces:**
- Produces `OracleIngress(capacity, expert_bytes, transfer_backend)`, `schedule(step, source_layer, target_layer, expert_ids)`, `resolve(step, layer, required_ids)`, and `finish_layer(step, layer)`.
- `PrefetchTicket` records ingress slots, enqueue/ready timestamps, mandatory/unused bytes, route match, and transfer event handle.
- Defines a separate `OracleTransferBackend` protocol: `enqueue(copies) -> object`, `query(handle) -> bool`, `wait(handle) -> None`, `elapsed_ms(handle) -> float | None`, `synchronize() -> None`.

- [ ] **Step 1: Write failing CPU state-machine tests**

```python
def test_prefetch_reuses_only_free_existing_ingress_slots():
    ingress = OracleIngress(capacity=4, expert_bytes=24, transfer_backend=FakeTransfer())
    ticket = ingress.schedule(step=3, source_layer=1, target_layer=2, expert_ids=(4, 5))
    assert ticket.slots == (0, 1)
    assert ingress.free_slots == (2, 3)
    with pytest.raises(IngressCapacityError):
        ingress.schedule(step=3, source_layer=1, target_layer=3, expert_ids=(6, 7, 8))

def test_resolve_waits_only_for_required_not_ready_experts():
    backend = FakeTransfer(ready=False)
    ingress = OracleIngress(capacity=4, expert_bytes=24, transfer_backend=backend)
    ingress.schedule(step=1, source_layer=0, target_layer=1, expert_ids=(2, 3))
    resolved = ingress.resolve(step=1, layer=1, required_ids=(2, 4))
    assert resolved.prefetched_ids == (2,)
    assert resolved.fallback_ids == (4,)
    assert backend.waited_handles == [resolved.ticket.handle]
```

- [ ] **Step 2: Run CPU tests and verify RED**

Run: `.venv/bin/pytest -q tests/unit/test_oracle_prefetch.py`

- [ ] **Step 3: Implement the CPU ownership state machine**

States are `free`, `inflight`, `ready`, and `in_use`; every transition is checked. Keys are `(step, layer, expert)`. Scheduling deduplicates expert IDs, never overwrites non-free slots, and records capacity fallback instead of expanding storage. `finish_layer` releases only entries for the completed layer after compute/promotion dependencies are recorded.

- [ ] **Step 4: Implement real CUDA transfer backend**

Use one `torch.cuda.Stream` per pool. Enqueue destination `copy_(host, non_blocking=True)` on that stream and record a dependency end event. The compute stream calls `wait_event(end)` only for required not-ready tickets. `enable_timing=False` uses dependency events without elapsed-time collection for formal throughput; `enable_timing=True` adds start/end timing events only in the separate profile points. A pinned host gather buffer cannot be refilled until its previous end event has completed; expose that wait separately as transfer-queue/host-buffer wait.

- [ ] **Step 5: Add CUDA parity tests**

```python
@pytest.mark.cuda
def test_cuda_oracle_prefetch_copies_real_bf16_payload_and_waits_before_use():
    source = torch.arange(32, dtype=torch.bfloat16, pin_memory=True)
    destination = torch.empty_like(source, device="cuda")
    backend = CudaOracleTransferBackend(torch.device("cuda", 0))
    ticket = backend.enqueue(((destination, source),))
    backend.wait(ticket)
    torch.cuda.synchronize()
    assert torch.equal(destination.cpu(), source)
    assert backend.elapsed_ms(ticket) >= 0
```

- [ ] **Step 6: Run CPU tests, mark CUDA test for server, and commit**

Run:

```bash
.venv/bin/pytest -q tests/unit/test_oracle_prefetch.py
.venv/bin/pytest -q tests/cuda/test_oracle_prefetch.py --collect-only
```

Commit:

```bash
git add src/flexmoe/runtime/oracle_prefetch.py src/flexmoe/runtime/expert_pool.py \
  tests/unit/test_oracle_prefetch.py tests/cuda/test_oracle_prefetch.py
git commit -m "feat: add bounded CUDA oracle ingress"
```

---

### Task 6: Integrate Horizon 0/1/2 and Forced Fallback into Real Expert Execution

**Files:**
- Modify: `src/flexmoe/runtime/expert_pool.py`
- Modify: `src/flexmoe/vllm/expert_cache.py`
- Modify: `src/flexmoe/vllm/decode_trace.py`
- Test: `tests/unit/test_oracle_pool_runtime.py`
- Test: `tests/unit/test_expert_cache_runtime.py`
- Test: `tests/cuda/test_expert_cache.py`

**Interfaces:**
- `ExpertPool.configure_oracle(trace, horizon, forced_omission=None)` enables the new path before measured generation.
- `ExpertPool.execute` preserves the original compute signature and returns identical expert outputs. It uses prefetched ingress payload only when the `(step, layer, expert)` ticket matches the actual route; all missing IDs use the existing synchronous gather/H2D fallback.
- `stats()` adds a separate `oracle` object without changing legacy counters.

- [ ] **Step 1: Write failing runtime equivalence tests**

```python
@pytest.mark.parametrize("horizon", [0, 1, 2])
def test_oracle_horizons_preserve_outputs_and_policy_transitions(horizon):
    baseline, sources = make_deterministic_pool()
    oracle, _ = make_deterministic_pool()
    oracle.configure_oracle(trace(), horizon=horizon)
    baseline_outputs = run_layers(baseline, sources, routes())
    oracle_outputs = run_layers(oracle, sources, routes())
    assert_tensor_lists_equal(oracle_outputs, baseline_outputs)
    assert oracle.policy.replay_snapshot() == baseline.policy.replay_snapshot()

def test_forced_omission_falls_back_once_without_route_substitution():
    pool, sources = make_deterministic_pool()
    pool.configure_oracle(trace(), horizon=1, forced_omission=(5, 7, 12))
    run_layers(pool, sources, routes())
    stats = pool.stats()["oracle"]
    assert stats["forced_omission_count"] == 1
    assert stats["fallback_on_demand_count"] == 1
    assert stats["actual_route_mismatch_count"] == 0
```

- [ ] **Step 2: Run runtime tests and verify RED**

Run: `.venv/bin/pytest -q tests/unit/test_oracle_pool_runtime.py`

- [ ] **Step 3: Integrate Oracle schedule/resolve without changing cache order**

At layer L, perform the normal actual route D2H and policy operations for L. Before invoking current expert compute, obtain trace IDs for L+horizon and schedule their payload into currently free ingress slots. Do not call policy methods for future experts. At target use, resolve prefetched IDs, synchronously gather fallback IDs, build the same expert map, execute compute, then perform normal admission promotion.

- [ ] **Step 4: Record physical Oracle metrics**

Collect per-rank counters in all points, but collect elapsed-time distributions only in the `*-profile` variants:

```text
lookahead_ms samples
mandatory_loaded_bytes
ready_before_use_count / required_prefetch_count
layer_batch_all_ready_count / eligible_layer_batch_count
exposed_wait_ms samples
unused_prefetch_bytes
too_early_prefetch_bytes
transfer_queue_wait_s
fallback_on_demand_count
capacity_fallback_count
forced_omission_count
route_mismatch_count
```

Expose p50/p95/p99 only when timing-enabled samples exist. Bytes loaded for prediction but not used before release count as unused; entries released because their step/layer expired count as too early. Formal output tokens/s, TTFT and derived TPOT come only from the matching profile-off horizon point.

- [ ] **Step 5: Validate smoke/output hash and four-rank fallback**

The runner requires horizon outputs equal the horizon-0 token hash. The forced-omission point requires exactly one omission and one fallback on every rank, zero route substitution, and the same output hash; it is `timing_eligible=False`.

- [ ] **Step 6: Run focused CPU tests and commit**

Run:

```bash
.venv/bin/pytest -q \
  tests/unit/test_oracle_pool_runtime.py \
  tests/unit/test_expert_cache_runtime.py \
  tests/unit/test_decode_trace.py
```

Commit:

```bash
git add src/flexmoe/runtime/expert_pool.py src/flexmoe/vllm/expert_cache.py \
  src/flexmoe/vllm/decode_trace.py tests/unit/test_oracle_pool_runtime.py \
  tests/unit/test_expert_cache_runtime.py tests/cuda/test_expert_cache.py
git commit -m "feat: execute constrained oracle prefetch horizons"
```

---

### Task 7: Exact Experiment Matrix and Resume-Safe Controller

**Files:**
- Modify: `src/flexmoe/analysis/decode_decision.py`
- Modify: `scripts/server/run_decode_decision.sh`
- Modify: `scripts/server/run_decode_mechanism.sh`
- Test: `tests/unit/test_decode_decision.py`
- Test: `tests/unit/test_decode_suite.py`

**Interfaces:**
- Consumes completed point summaries and the rank-0 Oracle trace artifact.
- Produces an immutable point graph with only the approved configurations, selected K0/K1/K_pair, legacy `32_815_054_848`-byte points, and `state.json` recovery.

- [ ] **Step 1: Write a failing exact-matrix test**

```python
def test_decision_plan_contains_only_approved_points():
    plan = build_decision_plan(run_id="r", capacities=capacities(), trace_paths=traces())
    assert [p["label"] for p in plan] == [
        "cal-1024", "cal-4096",
        "native-auto", "eager-auto", "offload-auto",
        "zero-resident", "zero-offload", "zero-offload-profile",
        "legacy-native", "legacy-eager-resident", "legacy-offload",
        "mechanism-native-32", "mechanism-eager-32", "mechanism-offload-32",
        "service-native-k0", "service-offload-k0", "service-eager-kpair", "service-offload-kpair",
        "trace-16", "trace-32",
        "oracle-16-h0", "oracle-16-h1", "oracle-16-h2",
        "oracle-16-h0-profile", "oracle-16-h1-profile", "oracle-16-h2-profile",
        "oracle-32-h0", "oracle-32-h1", "oracle-32-h2",
        "oracle-32-h0-profile", "oracle-32-h1-profile", "oracle-32-h2-profile",
        "oracle-forced-omission",
    ]
    assert not any(p.get("batch") not in (1, 16, 32, 64) for p in plan)
```

When K1 exceeds `K_resident_ceiling`, insert exactly one `service-offload-k1` after `service-offload-k0`; otherwise persist a skipped point with the capacity-gate reason and do not invoke the runner. If offload cannot stably run native `K0`, mark `service-offload-k0` unavailable and do not claim a K0/K1 service comparison. `K_pair=min(K_eager_resident_max,K_offload_max)` remains the separate same-eager-mode budget.

Point workloads are fixed: `zero-*` use request count/target batch 1; `legacy-*` and `mechanism-*-32` use request count/target batch 32; `trace-16` and all `oracle-16-*` use 16; `trace-32`, all `oracle-32-*`, and forced omission use 32; `service-*` use 64 requests with `max_num_seqs=32`. No controller branch may synthesize another request count.

- [ ] **Step 2: Run controller tests and verify RED**

Run: `.venv/bin/pytest -q tests/unit/test_decode_decision.py`

- [ ] **Step 3: Implement staged plan construction**

The controller cannot know capacities or traces at initial construction. Implement `next_points(state)` as a deterministic state machine: calibration → capacity → matched service → trace/replay → Oracle → forced omission → export. Persist every derived budget/trace window before launching dependent points.

- [ ] **Step 4: Add fixed-actual-batch evidence gates**

For `mechanism-*-32`, accept only repetitions whose four-rank `decode_batch_step_counts` contain one key equal to 32 and no other decode batch. If the scheduler does not reach a stable target, preserve service metrics but set `fixed_actual_batch_status=rejected`. The native mechanism point uses `K0`; eager resident/offload use `K_pair`, so the report labels the native comparison as capacity-confounded unless `K0 == K_pair`; the strict offload-management tax is eager resident versus eager offload at `K_pair`.

- [ ] **Step 5: Add scheduler metric availability semantics**

`kv_admission_blocked` is measured only when the observer reports a direct allocation-block reason. Waiting-only evidence is labeled `unattributed-waiting`. Swap disabled by policy is `policy-disabled`, not a measured 0. Recomputed tokens, ITL, and tail fields retain unavailable reasons.

- [ ] **Step 6: Preserve immutable failure/resume behavior**

Never overwrite a point directory. Resume validates current commit, exact argv, completed output and identity. A `running` or `failed` point blocks reuse; the user must choose a new suite ID. Build logs use attempt-specific names so resume never overwrites prior diagnostics.

- [ ] **Step 7: Run controller/wrapper tests and commit**

Run:

```bash
.venv/bin/pytest -q tests/unit/test_decode_decision.py tests/unit/test_decode_suite.py
bash -n scripts/server/run_decode_decision.sh scripts/server/run_decode_mechanism.sh
```

Commit:

```bash
git add src/flexmoe/analysis/decode_decision.py scripts/server/run_decode_decision.sh \
  scripts/server/run_decode_mechanism.sh tests/unit/test_decode_decision.py \
  tests/unit/test_decode_suite.py
git commit -m "feat: orchestrate bounded decode oracle experiment"
```

---

### Task 8: Complete Public Report and Export Conflict Repair

**Files:**
- Create: `src/flexmoe/analysis/decode_oracle_report.py`
- Modify: `src/flexmoe/analysis/decode_decision.py`
- Modify: `src/flexmoe/analysis/decode_report.py`
- Test: `tests/unit/test_decode_oracle_report.py`
- Test: `tests/unit/test_decode_report.py`

**Interfaces:**
- `build_oracle_report(state, point_artifacts) -> tuple[dict[str, object], str]` returns the public JSON and full Markdown.
- `repair_summary_conflict(summary, raw_csv) -> dict[str, object]` changes only derived export status and records provenance; it never edits or triggers rerun of source point data.

- [ ] **Step 1: Write failing full-report tests**

```python
def test_report_embeds_all_eight_required_sections_and_failures():
    public, markdown = build_oracle_report(complete_state(), artifacts())
    for heading in (
        "无采样零 miss 固定开销",
        "执行模式分解",
        "K0 与 K1 稳定容量",
        "最大并发 32 服务收益",
        "同预算缓存重放",
        "真实 Oracle horizon 0/1/2",
        "强制漏预测正确性",
        "失败、超时与不可用指标",
    ):
        assert heading in markdown
    assert public["oracle"]["horizon_2"]["timing_eligible"] is True
    assert public["failed_points"][0]["status"] == "timeout"
```

- [ ] **Step 2: Run report tests and verify RED**

Run: `.venv/bin/pytest -q tests/unit/test_decode_oracle_report.py`

- [ ] **Step 3: Implement evidence-preserving public schema**

Embed point identities, repetition medians/ranges, four-rank counters, phase boundaries, K0/K1 safety reserve, scheduler availability reasons, replay totals/distributions, Oracle readiness/wait/bytes and output hashes. Include all zero-miss rank/repetition rows. Do not copy prompt IDs, output token IDs, full route traces, hostnames, model paths, or raw private logs.

- [ ] **Step 4: Implement fixed causal language**

The report chooses only these conclusions:

```text
runtime-first
current-load-does-not-trigger-kv-gain
cache-policy-first
prefetch-predictor-has-direct-value
prediction-accuracy-not-priority-under-current-budget
inconclusive
```

Each conclusion lists the exact gates that passed and failed. No numeric threshold is invented beyond the spec’s strict capacity, exact replay, and correctness gates; “显著” is reported as measured ratios and intervals for user judgment.

- [ ] **Step 5: Repair summary/CSV export conflicts without rerun**

When a summary says unavailable but the immutable raw CSV contains three valid measured repetitions with matching identity, rebuild only the derived report, record `repair_source_sha256`, and retain both original status and repaired status. Any identity/count mismatch remains inconclusive.

- [ ] **Step 6: Run report tests and commit**

Run:

```bash
.venv/bin/pytest -q tests/unit/test_decode_oracle_report.py tests/unit/test_decode_report.py
```

Commit:

```bash
git add src/flexmoe/analysis/decode_oracle_report.py \
  src/flexmoe/analysis/decode_decision.py src/flexmoe/analysis/decode_report.py \
  tests/unit/test_decode_oracle_report.py tests/unit/test_decode_report.py
git commit -m "feat: publish complete decode oracle evidence"
```

---

### Task 9: Final Verification, Documentation, Review, and GitHub Publication

**Files:**
- Modify: `docs/decode-decision.md`
- Modify: `README.md`
- Verify: all files changed by Tasks 1–8

**Interfaces:**
- Produces a one-line server command and an explicit result-return contract through GitHub.

- [ ] **Step 1: Update operator documentation**

Document the exact command:

```bash
cd /home/jovyan/wangtonghan/moe-flex
git switch repro/fluxmoe
git pull --ff-only origin repro/fluxmoe
GPU_IDS=0,1,2,3 bash scripts/server/run_decode_decision.sh
```

Document output roots, new-run-ID recovery, private raw traces under `runs/`, public aggregate reports under `docs/results/`, and the requirement to commit only public results back to GitHub.

- [ ] **Step 2: Run formatting, lint, typing, and script checks**

Run:

```bash
.venv/bin/ruff format --check src tests
.venv/bin/ruff check src tests
.venv/bin/mypy src/flexmoe
bash -n scripts/server/run_decode_decision.sh scripts/server/run_decode_mechanism.sh
git diff --check
```

- [ ] **Step 3: Run the complete local unit suite**

Run outside the restricted socket sandbox because existing parallel tests bind loopback sockets:

```bash
.venv/bin/pytest -q tests/unit
```

Expected: all unit tests pass; CUDA tests remain collection-only on Mac.

- [ ] **Step 4: Review the complete diff against the spec**

Check every spec section maps to code/tests, no forbidden batch/policy/horizon exists, formal throughput excludes profiles, public reports contain no routes/tokens/paths, and exact files are staged. Request a read-only code review focused on CUDA dependency ordering, policy equivalence, identity gates, recovery, and claim ceilings.

- [ ] **Step 5: Fix all Critical/Important review findings and rerun verification**

Repeat Steps 2–4 after fixes. Do not claim completion from reviewer text alone; inspect the final diff and fresh command outputs.

- [ ] **Step 6: Commit documentation and push**

```bash
git add README.md docs/decode-decision.md
git commit -m "docs: document constrained decode oracle run"
git status --short --branch
git push origin repro/fluxmoe
git ls-remote origin refs/heads/repro/fluxmoe
```

Confirm the remote SHA equals local HEAD. Leave the existing untracked `.DS_Store` and `完整实验报告.md` untouched.

- [ ] **Step 7: Hand off server execution**

Return the pushed SHA, exact one-line command, expected public report path, and failure-diagnostic paths. Do not claim H100 validation until the server result commit is pulled and reviewed.
