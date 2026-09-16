# One Click Parallel and Offload Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 无需服务器Codex，一条命令运行TP4与两种EP/DP常驻对比以及既有TP4卸载实验，保留失败、可恢复并汇总。

**Architecture:** 新的原生offline并行runner独立于TP4-only卸载路径；标准库主机总控复用旧decode plan/validate/export，负责流程、资源归属和结果汇总。不是新的卸载后端。

**Tech Stack:** 已有Python3.10+ / torch2.8 / pinned vLLM0.10.2 / Docker / stdlib host runner。

**Spec:** docs/superpowers/specs/2026-09-16-oneclick-experiments.md

## Global Constraints

- Worktree `/Users/a1234/code/flexmoe/.worktrees/repro-fluxmoe`, existing branch `repro/fluxmoe`, baseline `fdd9d1b9ef5b652bae96f36b1620c28f32006a36`.
- Server project only `/home/jovyan/wangtonghan/moe-flex`; existing `/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct` read-only; four explicitly selected exclusive H100 GPUs.
- Existing vLLM0.10.2 pinned commit `01efc7ef781391e744ed08c3292817a773d654e6`, fixed image/build/patch, BF16/original routing, no new server weights/dependencies.
- Native configs: TP4 DP1 EPoff, TP1 DP4 EP4, TP2 DP2 EP4. Existing TP4 offload remains separate; EP offload unsupported, no silent mode fallback.
- Default global1024 unique requests, input1024/output512, seed20260912, warmup1/reps3, utilization0.90; global sequence/token scheduler ceilings1024/8192 divided byDP.
- TDD/apply_patch/scoped commits; preserve all historical results/default runners. No subagent push/subagents. Controller owns final ordinary GitHub push, no main merge/PR/GPU execution.

### Task 1: Native parallel benchmark and evidence adapter

**Files:** create `src/flexmoe/bench/parallel_runner.py`, `src/flexmoe/analysis/parallel_report.py`, `tests/unit/test_parallel_runner.py`, `tests/unit/test_parallel_report.py`. Only minimal worker RPC extension if existing actual memory/device RPC cannot identify DP/TP mappings; document any ABI change.

**Interfaces:** GPU CLI `python -m flexmoe.bench.parallel_runner --config tp4|ep4-dp4|ep4-dp2 --run-dir DIR --project-root ROOT --model-path MODEL --dataset-path DATA --dataset-manifest MANIFEST [--repetitions 3 --output-length 512 --batch-size 1024 --context-length 1024 --timeout-s 7200]`. Raw root `parallel-run`, `summary.json` with config/mode full-resident-native, explicit workload/versions/projectSHA/model/data/hardware identity, requested+resolved policy, DP partitions, per-rank memory and per-repetition global measured elapsed/outputcounts/TPS/status; private perrank artifacts survive partial failure. `parallel_report.py` is stdlib-safe like existing offline analysis and exports public `summarize_parallel(source: Path) -> dict`, `compare_parallel(sources: Sequence[Path]) -> dict`, `export_parallel(sources: Sequence[Path], output: Path) -> dict` plus direct `python -S ... --help`/CLI. Task2 consumes raw status and these functions; provide exact saved ABI in report.

- [ ] Inspect cached pinned official example via `git -C .cache/expert-cache-vllm-source show HEAD:examples/offline_inference/data_parallel.py` (blob now cached). Follow its VLLM_DP environment/bootstrap, automatic visible-device handling and EP flag; do not use latest-docs API. Set environment before importing vLLM in spawned workers. No parent CUDA context. Native TP4 also uses this runner for comparable timing.
- [ ] RED/GREEN literal config/split/globalaggregation: five inputs acrossDP2 split3+2 without duplication; output10 each produces50tokens, coordinatedwallclock2s gives25TPS regardless localTPS. Reject missing/duplicated DP partitions, wrong fixedcounts, invalid actualresolved TP/DP/EP ordtype, failedrank, nonfinite time, missing memory/KV; no gains from partial output. Reject impossible globalbudget divisibility and requests<DP rather than adding fake placeholders.
```python
assert split_indices(5, 2) == [[0, 1, 2], [3, 4]]
# Real reduction must use this group's parent elapsed, not local rates.
assert summarize_round([{"dp_rank":0,"generated_tokens":30},
                        {"dp_rank":1,"generated_tokens":20}], elapsed_s=2,
                       expected_tokens=50)["tokens_s"] == 25
```
- [ ] Use existing unique workload loader and fixedtoken SamplingParams; first32 calibration excluded; same fullselected inputhash across configs, partition locally. AllDP warmups/reps execute simultaneously with bounded coordination/childsupervision. Parent timing excludes initialization/telemetry/diskwrite and includes coordinated generation; any childerror cancels only ownedchildren. Save incomplete numeric evidence atomically and safe categories, rawerrors only private. Latency from actual request metrics ifavailable: label arrival-to-finished offline latency, missing metrics unavailable, no fakeTTFT; retain counts and P50/P95, not zero. Observe actualworkerKV/memory through existingextension, preserve DP/TP IDs and UUID uniqueness; no 4x logical count.
- [ ] Preserve native optimization policy (enforce_eagerFalse, no forcedcompilation0) and `FLUXMOE_ENABLE=0`; reject resolved offload/quantization/expert mode mismatch. Ensure priorcustomhooks cannotactivate accidentally; setofflineprojectcacheenv and same-worker extension. EP expertlocalplacement support should be observed from actualconfig/worker metadata where available; unsupportedbackend returns failed/unsupported, no install/fallback.
- [ ] Tests run parent coordination/fileoutput against strictfakeGPU external boundary and real process timeout/failure seams, not only purehelpers. Comparepublic requires complete3reps, same model/workload/version/hardware/projectSHA andtotalbudget; per-config KV maydiffer andmustbevisible. Repetitionoutputhash differences areauditonly, fixedcounts andvalidhashes stillchecked. PublicJSON/MD numericwhitelist excludes tokenIDs/outputhashes/UUID/path/errorstrings; one unqualifiedrowstaysfailed, notzeroTPS. `global_wallclock` and latency scope explicit. Run focusedtests/Ruff/Mypy/fullCPUonce, scopedcommit+fullreport withexactABI.

### Task 2: Resumable one command orchestration and server handoff

**Files:** create `src/flexmoe/analysis/experiment_autorun.py`, optionally focused `experiment_state.py` for state/lifecycle, `scripts/server/run_all_experiments.sh`, `docs/oneclick-experiments.md`, `tests/unit/test_experiment_autorun.py`, `tests/unit/test_experiment_launcher.py`; minimal changes `scripts/server/run_container.sh` for opt-inownedCID/labels and README/serverprompt link. Exclude `.superpowers` and `docs/results` in `.dockerignore` so private audit scratch and large previous reports are not copied into the build image; existing project bind mount still exposes runtime reports. Consume Task1finalABI; no second GPU/backend implementation.

**Interfaces:** host directstdlib CLI bootstrap, `run --run-id ID [--resume --retry-failed]` and `status --run-id ID`, `--dry-run` for noGPU planpreview. Wrapper autoID timestamp+unique suffix, foregrounddefault (user canusetmux), optionsprint exactresume; final one-line includesgitpull and GPU_IDS. Outputraw `runs/experiment-suite/<id>` (state/logs/pointattempts), public `docs/results/decode-mechanism-suite-<id>` to remain withinexistingclean-codeallowlist, allowlistarchiveunderrawroot. All childrunids begin suiteid andattemptnumber; allwrittenpaths canonicalprojectbounded.

- [ ] RED/GREEN actualfilesystem/state lifecycle: lockcontention rejected, config/SHA/GPU mismatch refusesresume, validcompletedpoints skipped, missing/corrupt output invalidatesreuse; failedpoints notblindlyretried unlessflag, interruptedrunningpoint preservesoldattempt andresumeswithnewattempt. Atomicstatewrites, no symlink/traversal/oldoutputoverwrite, exactargvnot shell eval. SIGINT/SIGTERM andsubprocesstimeout persist state and clean onlyowncontainer/CID; ifcleanupcannotproveGPUreleased abortnextstages. Never kill foreignprocesses/containers.
```python
# Exercise actual executor with a harmless local recorder subprocess.
# First invocation completespointA theninterruptsB. Resume must runB once
# undera freshattemptpath, retainA and oldB, and rejectchangedSHA.
```
- [ ] Add opt-in `FLEXMOE_CONTAINER_CID_DIR` and ownership token to existingrun_container without changinglegacydefaults; create a fresh CID file for every Docker invocation in that canonical project directory, labels bind suite+SHA; container lifecycle host timeout cleans exact CID after inspect ownership. The decode wrapper launches separate preflight and engine containers, so a single reused cidfile is invalid. UseexistingfourGPUpreflight atstartandeachpoint; clear errorifnoGPU_IDS, duplicates/occupiedcards. NoautoGPUselection, no globaldockerprune/kill, no changeswth333orNFS.
- [ ] Finite sequential pipeline: buildonce(sameSHA), verifyunique corpus, threeparallelconfigs, per-contextcalibrations, oldnativeTP4, seven1Kcoveragepoints, selectatmost3reachedrepresentativesfor4Kandlimitedoffload, A/B/C, overheadpair, finalreports. Reuse `decode_suite.build_plan` actual signature (inspect) /existing CLI instead ofduplicatedargbuilders. Profilefailures persist. Failureindependentstagescontinue; missingcalibration/native/memory skipsdependentstages. Eachpointtimeoutdefault7200 andboundedoutergrace; no implicit infinite retry. Entire suite state includes explicit EP-offload unsupported.
- [ ] KV automaticdecision strictlyfromsuccessfulcurrentSHA TP4nativeactualfourrank allocation/memory: smallestallocated roundedblock; smallhalf aligned; largeatmostobservednativecapacity; discloseconservative selection/no fullfreedspace exploitation. Existingperpointphysical2GBreserve authoritative. Reject malformed/missingfree/peak/blocks; no guessed16/32GiB constants. Representativebatch lowest/middle/highest fromcomplete profiles only; noreachedpoints skip, notclaimtarget reached.
- [ ] Exportpartialresults even aftertimeout, sanitize knownnumeric/status fields only, reuse decode/parallel exporters. UnifiedChinese summarylists attempted/skipped/failed, parallelcomparison andTP4A/B/C separately, availableKV/throughput/latency, diagnosticscope. Token/privatehash/rawlogsstayserver. Archiveonlyexplicitallowlistedpublicfiles, nofollowing symlinks; defaultnogitmutation/autopush. Docs show one-line launch/resume/status/reportupload; guarantee unattendedexecution notguaranteedspeedup. Shellrecorder tests actualargv/preflight/timeout/CIDownership; direct-S no torch import; finalCPU/Ruff/Mypy/shell/wheel andscopedcommit/report.
