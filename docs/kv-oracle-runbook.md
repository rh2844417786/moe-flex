# KV Oracle 容量收益诊断

第一阶段只回答：在现有原始 BF16 模型完整 GPU 驻留时，额外 KV 容量能带来多少时间收益空间，是否值得后续讨论小比例卸载。此入口不执行真实卸载，不改变缓存、pool、eviction、路由或 CUDA 内核。

**O 组显式 KV 申请绕过 r0 的名义 0.60 自动预算，可能实际使用更多 GPU 显存。** 所有结果均为 `comparison_backend=kv-oracle`、`diagnostic_only=true`、`formal_offload_gain=false`。它是乐观容量参考，不是公平 R/B/C 比较，也不是严格数学上界。现有正式 analyzer/exporter 拒绝这类输入。

## 环境和权限

服务器项目固定为 `/home/jovyan/wangtonghan/moe-flex`；所有 runs、缓存、日志、临时文件和结果写在该项目内。使用已有 Qwen3-Next-80B-A3B-Instruct 原始 BF16 权重及已提交 ShareGPT 数据。四张 H100 必须空闲且得到独占授权；没有四张符合条件的卡就停止，不杀他人进程。运行时 TP4、eager、编译 level=0、自定义算子开启、prefix caching 关闭、CPU offload=0、固定输出 token 数。

只允许 GitHub 同步和缺少缓存时获取以下唯一 Docker Hub 基础镜像：

`vllm/vllm-openai:v0.10.2@sha256:607442e407b0fea97f8a132a78b787c121a996dd4de181fa08e8da06e71ec2db`

所有 Docker RUN 均 `--network=none`，模型、数据、依赖均不得下载。网络运行容器复用项目已有 NCCL 配置，但模型加载及工作负载保持离线。缺少依赖或原生模型不能启动时保留失败证据，不安装或替换模型补救。Mac CPU tests 不构成 H100 性能证据。

## 同步和构建

下面变量按服务器已有位置设置；不向公开命令嵌入个人权重目录。先确认原 checkout 没有未提交源码、没有正在运行的实验。

```bash
cd "$PROJECT_ROOT"
git status --short
git switch repro/fluxmoe
git pull --ff-only origin repro/fluxmoe
git rev-parse HEAD
bash scripts/server/build.sh
export GPU_IDS=0,1,2,3
```

`PROJECT_ROOT` 必须指向上述固定服务器项目；`MODEL_PATH` 指向现有只读模型。镜像 revision 必须等于执行 SHA。需要修代码时先提交、重新构建，再用新 suite ID 从 r0 开始；不在同一实验序列中间改代码。

## 有限实验

| 预设 | requests | input tokens | output tokens | 用途 |
|---|---:|---:|---:|---|
| control | 32 | 4096 | 64 | 小负载控制 |
| main | 512 | 4096 | 256 | 默认诊断 |
| long | 512 | 4096 | 1024 | 必要时手动追加 |

三组均请求 `max_num_seqs=512`（引擎参数按 requests 数裁剪）、`max_num_batched_tokens=8192`、seed=20260905。源数据不足时循环使用已有 prompts；源条数、实际唯一条数、重复条数和输入 SHA 均记录。main/long 的输入 SHA 可以相同，但输出长度和引擎上下文不同，不能互借 anchor。

每个点启动新进程/容器，顺序默认 r0→small→large：

- r0：完整 GPU 驻留、0.60 自动 KV。K0 取本次完整实验各 rank 的 `kv_cache_allocated_bytes`，不是 `available_kv_cache_bytes`。
- small：仍完整 GPU 驻留，KV 申请为 K0+3.2 GB/卡。
- large：仍完整 GPU 驻留，KV 申请为 K0+18.1 GB/卡。

只有四 rank K0 相同才允许单个 override。实际 KV 大小、声明大小和 blocks 必须一致；block 对齐允许少于一个实际分配 block 的向下误差，不把申请值当成分配值。

```bash
ORACLE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/server/run_kv_oracle.sh screen --preset control \
  --suite-id "${ORACLE_TAG}-control" --model-path "$MODEL_PATH"
bash scripts/server/run_kv_oracle.sh screen --preset main \
  --suite-id "${ORACLE_TAG}-main" --model-path "$MODEL_PATH"
```

screen 为 1 warmup+1 repetition，只筛查。检查两个独立 suite 的 `docs/results/kv-oracle-<ID>/README.md` 和 `results.json`，选择一个有意义的负载再确认。不要把两种预设拼成一组比较。若 main 已说明容量没有明显收益，不机械追加 long。

```bash
bash scripts/server/run_kv_oracle.sh confirm --preset main \
  --suite-id "${ORACLE_TAG}-main-forward" --model-path "$MODEL_PATH"
bash scripts/server/run_kv_oracle.sh confirm --preset main --order reverse \
  --anchor "$PROJECT_ROOT/runs/kv-oracle/${ORACLE_TAG}-main/r0" \
  --suite-id "${ORACLE_TAG}-main-reverse" --model-path "$MODEL_PATH"
```

confirm 为 1 warmup+3 repetitions。reverse 顺序 large→small→r0；anchor 必须是此前同模型、数据、输入、输出长度、引擎设置、软件版本和 SHA 的完整 r0。允许 anchor 的 repetition 数不同，**只借它的 K0**；所有速度比都用反序末尾新测的 r0。新 r0 K0 与 anchor 不同则本组无效。正反序均检查，不能用旧基线速度代替。

自动导出会产生未提交 `docs/results/kv-oracle-*`。启动器只对这个生成结果命名空间豁免干净检查，其余源码/配置/未跟踪文件依然拒绝；原始 runs 已被忽略。**完成同 SHA 的全部 screen/confirm 后再提交导出结果**，否则提交改变执行 SHA，旧 anchor 不再可用。不要把可执行代码放进该豁免目录。

## 安全和测量

默认 fraction=0.90，额外 margin=2.0 GB/卡。用 r0 各次 current/peak 估算非 KV 开销：取边界 GPU 已用量和“Torch peak reserved + 当前非 Torch 开销”的较大值，减实际 K0，再加目标 KV 和 margin。超预算直接拒绝；不自动降低 target。

初始化及每轮后重复验证边界显存、Torch allocator peak、KV 和 rank 覆盖。该估算**不是硬件隔离或强制显存上限**；未采样的非 Torch 瞬态峰值仍可能导致 OOM。若安全拒绝，不自动放宽；只有明确改变 `--safety-fraction`/`--safety-margin-bytes` 后，以新 ID 重跑，报告修改后的参数。

计时前 GPU 同步、reset allocator peak、启用轻量 root-model forward probe，结束时同步后停止 probe；不计初始化和 warmup。probe 只读 CPU 侧 `model_runner.input_batch.num_reqs`，记录实际 forward 次数、观测次数、请求数总和/均值/峰值。这个量是当前批次成员数，不是等待队列长度。没有支持时为 unavailable/null；KV occupancy/preemption 永远 unavailable，不用提交请求数或 token rows 伪造。参数驻留检查在计时外，不做 per-layer GPU 同步或路由 ID D2H。

原始 `summary.json` 保存引擎策略、执行 SHA、模型/输入 hash、每轮固定 tokens、端到端秒数、输出 hash、rank KV/current/peak 和探针。公开结果保留白名单数字、hash、相对 run ID；显存标量统一十进制 GB（原始 bytes 同时保留），不上传 prompts、权重、原始日志或机器身份。正确性范围为 batch1 smoke，性能批次 hash 供审阅，不承诺全批次数值等价。

## 失败、导出和判断

每点 Docker 内默认超时 7200s，可显式设置 `--timeout-s`。失败/OOM/超时/安全拒绝立即停止后续点，保留已有完整轮次和独立 smoke。不会缩小输入或隐藏重试。若 large 失败，仍报告合法 r0/small 的诊断数字，但完整三组判断保持 incomplete。缺证据不等于没有收益。

```bash
bash scripts/server/run_kv_oracle.sh export \
  --source "$PROJECT_ROOT/runs/kv-oracle/${ORACLE_TAG}-main" \
  --output "$PROJECT_ROOT/docs/results/kv-oracle-${ORACLE_TAG}-main-reexport"
```

导出目录必须全新，运行目录不可复用；不删除原失败文件。读 JSON/CSV 和中文结论，检查逐轮时间收益空间、吞吐比、中位数及范围。至少三轮且范围不重叠才称可重复信号。工程止损线：small/large 均低于10%且无机制改善，可停止该负载吞吐路线；10–20%只考虑低成本改动；small至少20%并伴 forward减少或有效请求批次增长，才讨论后续小比例卸载；只有large有信号不能证明5层/1槽可行。机制未知不能授权强继续结论。这些阈值不是统计学定理，不能拿不同 B 调度开销直接推算 C。

先向用户报告三组状态、正反序一致性、实际 KV/峰值/参数、时间收益空间、未知指标与继续/停止理由，再讨论真实卸载。此脚本不会自动执行后续卸载。实验完毕，只提交本轮代码和人工核对的脱敏结果；原 `expert_cache_suite` 的服务器 smoke 失败导出路径继续保留。
