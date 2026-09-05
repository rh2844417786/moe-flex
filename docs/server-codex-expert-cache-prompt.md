# 发给服务器 Codex 的执行指令

请在现有 `/home/jovyan/wangtonghan/moe-flex` checkout 完成 BF16 热专家常驻与共享缓存的离线验证和实验，并将脱敏代码/结果经同一 `repro/fluxmoe` 分支回传。按以下顺序实际执行，记录每条执行命令、退出码、开始/结束时间和执行 SHA；Mac 的 CPU tests 不构成 H100 证据。

1. 只读检查 `pwd`、`git status --short`、`git branch --show-current`、`git rev-parse HEAD` 和 remote。确认分支为 `repro/fluxmoe`。保留已有改动；若工作区不干净先查明归属，不 reset/checkout 覆盖。干净时执行 `git pull --ff-only origin repro/fluxmoe`，记录更新后的 SHA。这是同步代码，不授权服务器安装依赖或下载模型/数据。网络受限时使用已经同步好的 clean SHA 并如实记录，禁止为运行流程访问模型 hub、PyPI、apt 或重新下载 vLLM。
2. 读取 `docs/expert-cache-runbook.md`，核对已有 Qwen3-Next-80B-A3B BF16 checkpoint 与现有数据。默认使用随 Git 提交的 ShareGPT；如本次任务明确沿用已有 SWE-bench token 数据，找出已有 dataset/manifest 的确切路径，校准和所有 arms 都传同一对参数。禁止凭空改数据或混用 profile。
3. 用 `nvidia-smi` 核对四张当时空闲、允许独占的 H100，设置实际 `GPU_IDS`。不得杀其他人的进程或改已有容器。没有四张空闲卡就报告资源不足，不冒用有负载的卡。所有写入，包括 cache/tmp/logs/profile、结果及调试代码，都留在 `/home/jovyan/wangtonghan/moe-flex` 内。权重只读挂载。
4. 运行 `bash scripts/server/build.sh`。优先使用已有 pinned base image；无缓存时允许现有脚本从 Docker Hub 获取该确切 digest，随后所有 build RUN 步骤仍为 `--network=none`。Docker Hub 是用户明确允许的例外，不能因此访问 PyPI、apt、HF、GHCR 或 CDN。若此前设置了 `FLEXMOE_OFFLINE_BUILD=1` 而镜像缺失，可对这次 build 使用 `FLEXMOE_OFFLINE_BUILD=0`，不因允许的镜像获取停住。核对 `build/image.env` 的 SHA 和 image revision 标签等于 clean checkout SHA。每次代码修复都必须先跑对应检查、commit，再 rebuild；不能以相同 HEAD 的旧镜像运行未提交修改。
5. 以下校准/确认入口自动运行 exclusivity preflight 和强制 CUDA tests：`tests/cuda/test_expert_cache.py`（四 TP shard BF16 bits、fused parity、cache churn/full demand、无 H2D 重复命中及独立异步延迟消费者）和 `tests/cuda/test_partial_runtime.py`（原有 BF16/fused tests）。检查该 suite 的 `cuda-preflight.json` 与私有 JUnit：tests>0 且 skipped/failures/errors=0 才能继续。不能只看 pytest exit 0。若 async test 无法保持关键 overlap 窗口，不得降低断言来获得通过；保留失败并诊断。

先运行一次原生 resident 校准：

```bash
bash scripts/server/run_expert_cache.sh calibrate \
  --suite-id hot-cal-u060-v1 --gpu-memory-utilization 0.60 \
  --calibration-count 32 --context-length 4096 --output-length 256 --timeout-s 7200
profile_path=/home/jovyan/wangtonghan/moe-flex/runs/expert-cache/hot-cal-u060-v1/profile.json
```

确认 calibration summary complete、4 ranks identity 相同、每层 forward_count>0、profile 的几何为 Qwen3-Next/TP4 实际本地 expert width。start capture 在 engine constructor/profiling 完成之后，stop 在指定 generate 完成之后；不允许把 dummy profiling 当热度校准。评估必须排除所有 32 个完整 token 哈希；重复仅发生于剩余 evaluation pool。profile 是哈希和计数，不含原始 prompt。

随后执行短 sanity，检查每个真实模型的 batch=1 smoke 输出是否与 R 完全相同，以及缓存实际 profile/policy/ratio/layout、per-layer demand 和 KV accounting：

```bash
bash scripts/server/run_expert_cache.sh confirm \
  --suite-id hot-sanity-u060-r050-v1 --profile-path "${profile_path}" \
  --resident-ratio 0.50 --cache-slots 256 --cache-policy decayed-lfu \
  --gpu-memory-utilization 0.60 --batch-size 1 --context-length 4096 \
  --output-length 8 --warmups 0 --repetitions 1 --timeout-s 1800
```

任何 correctness、identity、mechanism、KV accounting 失败/OOM/timeout 都先停止后续扩展，读本 suite 私有日志诊断。允许修复安全、范围内的本地集成错误，但不得弱化 CUDA tests、改路由、改权重、量化、下载依赖或偷换工作量。先提交修复再重建，用全新 suite ID 重跑受影响的校准与 R/B/C；不能把不同 SHA 的结果拼成一组。

sanity 通过后运行默认固定工作量：512 requests、4096 输入、256 输出、1 warmup、3 repetitions，四卡 TP4，utilization=0.60。

```bash
bash scripts/server/run_expert_cache.sh confirm \
  --suite-id hot-confirm-u060-r050-v1 --profile-path "${profile_path}" \
  --resident-ratio 0.50 --cache-slots 256 --cache-policy decayed-lfu \
  --gpu-memory-utilization 0.60 --timeout-s 7200
```

读完该组数值再决定候选；不要无条件循环完整矩阵。若可以安全继续，分别用独立 ID `hot-confirm-u060-r075-v1`、`hot-confirm-u060-r090-v1` 和对应 `--resident-ratio 0.75`、`0.90` 重跑完整 R/B/C，其他参数完全不变。常驻比例 0.90 与显存 utilization 0.90 是不同变量。对有意义的候选，再使用例如 `hot-confirm-u060-r075-lru-v1`、`--cache-policy lru` 进行同容量/同 profile 的 matched control。不得给 offload arm 单独提高 utilization、减少输出、减少 batch、换数据或放宽调度策略。

只有 0.60 的 capacity、逐层 unique coverage、cold cache hit ratio、实际 H2D、host wait/gather/policy、CUDA load/compute/promotion、端到端吞吐已经解释清楚，才考虑 utilization=0.90 的新完整 R/B/C。先检查 0.60 失败/负结果是否已经足够否定当前路径；不追求填满矩阵。稳定收益出现后需要 fresh suite 再确认。

解释结果时，R 是原生常驻；B 按 R 的实际 KV bytes/blocks 固定；C 在与 R 相同 GPU budget 下扩 KV。检查 B/R、C/B、C/R 和真实 KV 增量，报告每轮实际 token 数与秒数。每轮 per-rank allocator peaks 在计时前同步重置；检查真实显存数据并说明 Torch peak 不含全部非 Torch/NCCL 内存。保留完整嵌套 counters、逐层 delta/coverage、真实 kernel configurations 和 runtime weights_verified=0；不得伪造全模型 GPU 权重验证。H2D=0 在全命中时合法，但命中率不等于吞吐收益。高批量输出哈希与 batch1 correctness scope 分开报告。

无论正结果、负结果、失败还是超时，都导出每个已尝试的 suite，用新目录：

```bash
bash scripts/server/run_expert_cache.sh export \
  --suite-dir runs/expert-cache/hot-confirm-u060-r050-v1 \
  --output-dir docs/results/hot-confirm-u060-r050-v1
```

校准 suite 也可同样导出，public JSON/CSV 保留 hash 和逐 rank counts。只将 exporter 生成的脱敏 `summary.json`、`summary.csv`、`evidence.csv`、中文 `report.md` 与必要代码修复提交。审阅 diff，禁止提交权重、原始 prompts/generated text、private XML、stdout/stderr、机器身份或原始路径日志。结果在 report 中标注实际执行 SHA、完整命令和范围内的失败类型；引用运行 ID 定位服务器私有日志，不复制日志内容。

公开报告的可复现命令必须是脱敏模板：保留实际数值参数，以 `$MODEL_PATH`、`$PROFILE_PATH`、`$DATASET_PATH`、`$DATASET_MANIFEST` 代替私有路径，不粘贴展开后的原始命令日志。实际路径与原始完整命令仅保存在服务器私有 runs 中。

最后执行相关 CPU/CLI/export checks、`git diff --check`，commit 并 push 同一 `origin repro/fluxmoe`，核对 remote SHA。此回传已有用户授权，不需再次确认。保留 dirty work，禁止 force-push。返回：代码/结果提交 SHA，CUDA 实际执行项与 gate，profile hash，所有 suite ID 与退出状态，R/B/C 的每轮/中位吞吐与 KV/peak GB，是否存在稳定收益及正确性边界，失败原因和下一步。没有硬件证据时明确说尚未验证，不把命令启动或缓存命中称为论文复现成功。

可选 `fluxmoe_expert_cache_reconfigure` 仅作用于 quiescent engine 的固定物理 pool，重划 resident/cache 并同步重载；不改 active KV 或 resize pool。它不属于上述静态比较，也不是自动 memory controller；本轮默认不调用。
