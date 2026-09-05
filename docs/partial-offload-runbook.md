# 部分 BF16 卸载：离线吞吐实验

本路径用于 Qwen3-Next-80B-A3B-Instruct、4 张独占 H100、TP=4。现有权重只读，服务器写入均留在 `/home/jovyan/wangtonghan/moe-flex`。Mac 的单元测试验证执行合同；正确性和吞吐结论必须由 H100 实际运行产生。

## 一次准备

在服务器更新同一分支并构建当前 commit 的镜像。构建沿用已钉住的 vLLM 0.10.2 Docker Hub 镜像和仓库内 wheel，Docker build 的 RUN 阶段强制离线。模型、数据和 tokenization 均无下载步骤。

```bash
cd /home/jovyan/wangtonghan/moe-flex
git pull --ff-only origin repro/fluxmoe
FLEXMOE_OFFLINE_BUILD=1 bash scripts/server/build.sh
GPU_IDS=0,1,2,3 bash scripts/server/preflight.sh
GPU_IDS=0,1,2,3 bash scripts/server/run_container.sh \
  python3 -m pytest tests/cuda/test_partial_runtime.py -q
```

`GPU_IDS` 替换为已分配的 4 张独占卡。`FLEXMOE_OFFLINE_BUILD=1` 强制使用本机已存在的固定 digest 基础镜像，缺失时立即报错；不设置该变量时，只有缺失基础镜像才尝试 Docker Hub。构建各步骤始终使用 `--network=none`。不要在服务器执行 pip 或模型下载命令。启动脚本会在每个模型点前重新执行 preflight，检查 GPU 独占与可用性。

## 短扫描，再确认一个候选

```bash
GPU_IDS=0,1,2,3 bash scripts/server/run_partial_offload.sh scan \
  --suite-id partial-scan-u060 --gpu-memory-utilization 0.60 \
  --batch-size 256 --context-length 4096 --output-length 32 --timeout-s 1800

GPU_IDS=0,1,2,3 bash scripts/server/run_partial_offload.sh confirm \
  --suite-id partial-confirm-u060 --gpu-memory-utilization 0.60 \
  --scan-dir runs/partial/partial-scan-u060 \
  --batch-size 512 --context-length 4096 --output-length 256 \
  --repetitions 3 --warmups 1 --timeout-s 1800 --reverse-on-gain
```

扫描执行 resident、零卸载 hook 对照、`2/1、4/1、4/2、6/1、6/2、8/1、8/2` 七种“卸载层数 / staging slots”组合。每个点使用新容器和新 engine，初始化不计入吞吐；超时上限包含初始化。扫描只做 1 次测量，用于选择候选。正式确认从扫描中选最快且 smoke、实际 KV 统计有效的非零卸载组合。也可以将 `--scan-dir ...` 换为 `--offload-count 4 --staging-slots 2` 手工指定一个有效组合。

正式确认顺序为 R/B/C：

| Arm 参数 | 权重策略 | KV 设置 |
|---|---|---|
| `resident` | 原生 GPU BF16 | vLLM 按 utilization 自动分配 |
| `partial-fixed-kv` | 选定少数层保存 CPU BF16，固定 GPU slots | 采用 R 各 rank 的最小 profiling 预算；必须再次核对实际分配字节和块数与 R 完全相同 |
| `partial-auto-kv` | 与 B 相同的卸载层与 slots | vLLM 按相同 utilization 自动扩大 KV |

每组同时固定模型身份、commit、PyTorch/vLLM 版本、TP、显存预算、输入哈希、请求数、输入/输出长度、seed、调度上限和编译策略。使用 BF16、greedy、`ignore_eos=True`、`min_tokens=max_tokens`、禁用 prefix caching、eager、NCCL all-reduce、编译 level=0 和 `custom_ops=["all"]`。不额外使用其他 GPU 存权重。

默认 `max_num_seqs=256`、`max_num_batched_tokens=8192`，可通过同名 CLI 参数统一调整。batch 表示一次提交的请求总数，不等于每时刻的实际活跃请求数。每种 context 的现有数据只有 256 条，因此 batch=512 会循环使用原有 prompts；结果明确记录 `repeated-existing-prompts`、重复数量、数据 manifest SHA 和最终输入 SHA。禁用前缀缓存后，重复输入不会造成跨请求缓存命中收益。

`--reverse-on-gain` 只在 C 的三轮最小吞吐高于 R 的三轮最大吞吐，且 C 实际 KV 增加时，再独立执行 C/B/R 的反向顺序。反向 B 使用首轮 R 的固定 KV 请求值，分析时再与反向 R 的实际容量核对；若 profiling 漂移导致容量不一致，该对比自动失效。utilization=0.60 的结果只说明该显存预算下的行为；0.90 需要新 suite ID 独立重跑。不要混合两种 utilization 的结果。

## 单点排查

```bash
GPU_IDS=0,1,2,3 bash scripts/server/run_partial_offload.sh point --timeout-s 1800 \
  --arm resident --run-dir runs/partial/manual-r \
  --batch-size 256 --context-length 4096 --output-length 128 --repetitions 3

GPU_IDS=0,1,2,3 bash scripts/server/run_partial_offload.sh point --timeout-s 1800 \
  --arm partial-fixed-kv --offload-count 4 --staging-slots 2 \
  --resident-run runs/partial/manual-r --run-dir runs/partial/manual-b \
  --batch-size 256 --context-length 4096 --output-length 128 --repetitions 3

GPU_IDS=0,1,2,3 bash scripts/server/run_partial_offload.sh point --timeout-s 1800 \
  --arm partial-auto-kv --offload-count 4 --staging-slots 2 \
  --resident-run runs/partial/manual-r --run-dir runs/partial/manual-c \
  --batch-size 256 --context-length 4096 --output-length 128 --repetitions 3
```

已存在的 run/suite 目录拒绝覆盖。失败点不自动重试；使用新 ID 修正配置后再运行。超时发生在容器内部，退出后容器会清理其分布式 worker。suite 的 `logs/*.stdout.log` 和 `logs/*.stderr.log` 为私有诊断；`summary.json` 保存状态和已完成测量，每个 `rep-*.json` 独立原子写入。SIGKILL 即使打断当前测量，也不会损坏此前的 repetition 文件。

## 验收与回传

每个点先用相同首条 prompt 的前 1024 token、batch=1、固定最多 8 个输出 token 做 smoke。B/C 必须与 R 的实际输出 token 哈希一致，并通过首次使用的 BF16 权重逐位核验。性能批量的输出哈希另外保存；vLLM 调度导致批量输出变化时会明确报告，不把 batch=1 证据扩大为批量输出等价。

计时前清空抽样时序，整数计数保留累计值并做前后差；计时在 generate 和公共 GPU 同步之间，GPU telemetry 提取在计时外。结果包含每 rank 与汇总 H2D 字节、copy launches、offload/resident forwards、load/wait/compute CUDA 时间和 CPU enqueue 时间。抽样 CUDA 时间只用于解释机制，sample_count=0 表示不可用，不表示实际没有等待。vLLM v1 未提供 RequestOutput.metrics 时，TTFT 和请求时延为 null。

每个 suite 自动产生 `runs/partial/<suite-id>/public/{summary.json,summary.csv,report.md}`。比较报告给出 B/R、C/B、C/R 和三轮范围；真实 KV 统计缺失、预算/输入/策略不同、smoke 不匹配、失败或超时均不能标记目标达到。三轮稳定增益先证明当前固定工作负载，反向顺序复测再检查执行顺序影响；没有交互时延硬门槛。

导出到 Git 可跟踪的单独目录：

```bash
python3 src/flexmoe/bench/partial_suite.py export \
  --project-root /home/jovyan/wangtonghan/moe-flex \
  --suite-dir runs/partial/partial-confirm-u060 \
  --output-dir docs/results/partial-confirm-u060

git add docs/results/partial-confirm-u060
git commit -m "results: report partial BF16 offload throughput at utilization 0.60"
git push origin repro/fluxmoe
```

导出采用逐字段、逐类型白名单，只保留数值、commit/数据/模型哈希、软件版本和相对 run ID；不回传 prompts、token IDs、生成文本、权重路径、机器身份或原始日志。模型身份哈希覆盖 config、权重索引和本地路径哈希，不声称重算了 80B 模型全部权重的内容哈希。禁止直接 `git add runs`。回传提交之后如需新实验，重新构建对应 commit 的镜像，以保持镜像与 checkout 一致。
