# 原始 BF16 热专家常驻与共享缓存

这是显式启用的实验后端，目标为已有的 Qwen3-Next-80B-A3B、原始 BF16、TP=4 和独占的四张 H100。Mac 单元测试不能证明 CUDA 正确、实际 KV 收益或吞吐改善。原有 `scripts/server/run_partial_offload.sh` 和历史 partial-host 结果继续可用。

每层固定一组校准热专家常驻；全模型共享持久缓存，另有一整层逻辑专家数的 ingress。全部权重池在 vLLM KV profiling 前分配。缓存策略支持 `decayed-lfu` 与同容量的 `lru`。运行时按原始路由计算，不量化、不重路由、不丢专家、不下载权重。

## 固定实验约束

- 全部输出在 `/home/jovyan/wangtonghan/moe-flex` 下；checkpoint 只读使用已有 `/mnt/public_data`。
- R 为原生 GPU resident（`FLUXMOE_ENABLE=0`）；B/C 为 `FLUXMOE_STORAGE_MODE=expert-cache`。沿用旧 arm 名 `resident` / `partial-fixed-kv` / `partial-auto-kv`，JSON 的 `storage_backend` 明确真实后端。
- R/B/C 必须使用同一个模型、profile、校准排除集、batch/context/output、seed、软件 SHA、TP、GPU 总容量及 utilization。B 使用 R 已实际分配的 KV bytes，逐 rank 验证实际 bytes 与 block 数相同；C 在同 utilization 下重新 profile KV。
- 输出长度固定，greedy、ignore EOS、无 prefix cache、eager、相同 chunked prefill。计时、smoke、warmup 和原子逐轮保存复用 `partial_runner.run_benchmark`。默认 512 requests、4096 输入、256 输出、1 warmup、3 repetitions。
- 先原生校准，不能用尚无 profile 的缓存 engine 校准自身。固定选择数据顺序前 32 个不同完整 token 哈希；评估先移除全部校准哈希及重复，再重复剩余池达到 batch size。空评估池直接拒绝。仅声明完整 prompt token 哈希隔离，不声明源对话/文档独立。

## 离线启动

完整可直接交给服务器 Codex 的指令见 [server-codex-expert-cache-prompt.md](server-codex-expert-cache-prompt.md)。先检查 clean checkout、记录 SHA，构建镜像，再人工选择当时空闲且可独占的四张 H100；不得杀其他任务。优先复用缓存的固定 base digest；若缺少，允许现有 `build.sh` 从 Docker Hub 获取该确切 digest。所有 Docker build RUN 步骤仍使用 `--network=none`，不访问 PyPI、apt、HF、GHCR 或 CDN。

```bash
cd /home/jovyan/wangtonghan/moe-flex
git status --short
git rev-parse HEAD
bash scripts/server/build.sh
export GPU_IDS=4,5,6,7  # 必须按现场空闲状态替换
```

若已确认 base image 缓存齐备，可设置 `FLEXMOE_OFFLINE_BUILD=1`；若该选项因缺少 base image 拒绝，允许清除此项限制并让 `build.sh` 仅获取固定 Docker Hub 镜像，然后继续离线 build。

`calibrate` 和 `confirm` 自动运行 GPU exclusivity preflight 与完整 `tests/cuda/test_expert_cache.py`、`tests/cuda/test_partial_runtime.py`。包括非默认 stream 延迟消费者测试以及原有 fused BF16 parity。JUnit 必须实际执行至少一项且 skips/failures/errors 全为零；仅 pytest exit 0 不够。私有 XML 与数值 gate 放在该 suite。每个 engine 前重新检查 GPU 占用。脚本拒绝 dirty checkout，镜像与 checkout SHA 由现有容器启动器检查。

```bash
bash scripts/server/run_expert_cache.sh calibrate \
  --suite-id hot-cal-u060-v1 --gpu-memory-utilization 0.60 \
  --calibration-count 32 --context-length 4096 --output-length 256

# 后续所有点复用这个已完成 profile；不要覆盖或手改它。
profile_path=/home/jovyan/wangtonghan/moe-flex/runs/expert-cache/hot-cal-u060-v1/profile.json

bash scripts/server/run_expert_cache.sh confirm \
  --suite-id hot-sanity-u060-r050-v1 --profile-path "${profile_path}" \
  --resident-ratio 0.50 --cache-slots 256 --cache-policy decayed-lfu \
  --gpu-memory-utilization 0.60 --batch-size 1 --context-length 4096 \
  --output-length 8 --warmups 0 --repetitions 1 --timeout-s 1800

# 仅 sanity 成功且检查数值后运行。每条命令均是新的完整 R/B/C。
bash scripts/server/run_expert_cache.sh confirm \
  --suite-id hot-confirm-u060-r050-v1 --profile-path "${profile_path}" \
  --resident-ratio 0.50 --cache-slots 256 --cache-policy decayed-lfu \
  --gpu-memory-utilization 0.60 --timeout-s 7200
```

sanity 的 batch1 smoke 失败、CUDA 错误、OOM、invalid mechanism 或超时会停止该 suite，不盲跑矩阵。负吞吐结果仍需导出。理解 0.50 的每层需求覆盖、cold cache hit ratio 与各项 CPU/CUDA 时间之后，才分别用新 suite ID 尝试 `--resident-ratio 0.75`、`0.90`；它们是常驻比例，与 utilization 0.60 不同。保持 cache slots=256 以及同一 workload。针对有意义的比例另跑 `--cache-policy lru`，除 policy 外容量、profile、输入与预算不变。

只有 0.60 全部证据已经解释清楚后，才可另开 utilization=0.90 的完整 R/B/C；不能只给缓存 arm 更多显存。短 sanity 不证明三轮吞吐收益。出现 stable gain 应在新 suite 再确认，而不是继续覆盖旧结果。

## 已有 SWE-bench 或其他已提交 token 数据

保持上一条工作流的自定义数据转发。校准和全部比较命令同时传：

```bash
--dataset-path /home/jovyan/wangtonghan/moe-flex/benchmarks/data/<existing>.jsonl.zst \
--dataset-manifest /home/jovyan/wangtonghan/moe-flex/benchmarks/data/<existing-manifest>.json
```

这些文件必须已经存在并符合当前 `verify_subset` / `PromptRecord` 格式；不访问网络。不要混用 ShareGPT profile 和 SWE-bench profile；输入哈希验证会拒绝不匹配。`--model-path` 同样全程转发，校准 producer 与 runtime 共用 `model_profile_identity`，目标仍限 Qwen3-Next BF16 TP4。

## 如何解释证据

`runs/expert-cache/<suite>/public/summary.json` 保存脱敏比较与完整嵌套数值；`summary.csv` 为吞吐/KV 摘要；`evidence.csv` 为逐层、逐 rank、逐轮的长表；`report.md` 为中文解释。源 summary、校准 provenance、stdout/stderr 和 private JUnit 保留在服务器 runs。

计数只对累计量做差：resident hits、cold hits/misses、admission/eviction/bypass、各类 H2D/D2D、逐层 forwards/demands 与计时。占用量、最大覆盖、策略 hit ratio 等 gauge 单独记录 snapshot；measured cold hit ratio 用该轮 hits 与 misses 重算。H2D=0 可以有效，需同时满足 demand、layout、identity 与 executed forward 检查。校准/startup/warmup 不混入 measured transfer delta。CUDA 计时为抽样，不能替代端到端输出吞吐。

每轮计时前，所有 worker 同步并 reset allocator peaks；结束计时后读取该轮 `torch_peak_allocated_bytes` / `torch_peak_reserved_bytes` 与实际 KV。起始 memory 的 peak 是此前 allocator 生命周期峰值，包含 profiling；per-repetition peak 的明确 scope 为 `measured-generate-after-synchronized-reset`。Torch allocator peak 不等于整个设备或 NCCL/non-Torch 总峰值，仍需结合 total/free/current 与真实 OOM 行为。

`weights_verified=0` 是事实：runtime 不做大权重 D2H 校验。CUDA 测试覆盖小型专家的 BF16 位级映射与 fused output；真实模型的严格 gate 是相同 full-prompt、batch=1 greedy 输出哈希。性能批量哈希单独报告，不一致时不得声称高批量输出等价。profile SHA 为规范化 `ExpertProfile.to_dict()` 的 canonical JSON SHA；模型身份是 config/index/路径 SHA，未重读所有 checkpoint weight bytes。

逐层 unique coverage 接近全部专家时，共享小缓存可能无法消除冷工作集。应先看实际 miss bytes、route D2H、policy CPU、host wait/gather、load/compute/promotion CUDA、实测 KV 增量与 C/R；负结果也必须回传，不以更改 R/B/C 预算掩盖。

## 导出与可选重配置

```bash
bash scripts/server/run_expert_cache.sh export \
  --suite-dir runs/expert-cache/hot-confirm-u060-r050-v1 \
  --output-dir docs/results/hot-confirm-u060-r050-v1
```

导出目录必须是新的；允许在 Mac 直接用 `python3 src/flexmoe/bench/expert_cache_suite.py export ...` 检查已有脱敏结果。导出缺失/损坏记录会保留失败/不可比较状态，不能升级为成功。

运行时另有 `fluxmoe_expert_cache_reconfigure(resident_ratio)` RPC：仅在 engine quiescent、无 active forward 时使用，同步后在固定物理 pool 内重划 resident/cache，重新上传 resident 并清空 cache。它不调整 active KV，不 resize pool，也不是自动控制器。正式 R/B/C 不调用它；若研究该 API，必须单独 suite 和明确 before/after，不混入上述静态比较。
