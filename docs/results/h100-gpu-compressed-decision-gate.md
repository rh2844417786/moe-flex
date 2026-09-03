# H100 GPU-Compressed 正式决策门报告

## 结论

commit `8033eafb6878abb646de964ee616d9a7d86c38c4` 上的四卡正式决策门已完成。
GPU-compressed 路径通过了运行证据校验和独立 smoke 正确性门禁，但没有超过同 commit、
同协议的 resident 基线，因此吞吐优化结论为
`NO_CONFIRMED_IMPROVEMENT`，按实验设计停止扩展完整 key matrix。

GPU-compressed 的三次实测吞吐为 `6.3133`、`6.3615` 和
`6.3735 tokens/s`；resident 为 `68.7696`、`68.7928` 和
`68.7538 tokens/s`。GPU-compressed 最小值没有超过 resident 最大值，
其中位吞吐仅为 resident 的约 `9.25%`，慢约 `10.81` 倍。

## 实验配置

- GPU：4 张 NVIDIA H100 80GB PCIe，tensor parallel size 为 4；
- 模型：Qwen3-Next-80B-A3B-Instruct，BF16；
- 推理框架：vLLM v0.10.2；
- batch size：128；
- context length：4096；
- 固定输出：每请求 4 tokens，忽略 EOS；
- 每个变体：3 次 warmup，随后 3 次正式测量；
- 执行模式：eager、greedy decoding；
- GPU 显存利用率上限：0.60；
- GPU-compressed materialization：batched；
- 实验进程标识：`wth333`。

## 吞吐结果

| 变体 | Rep 0 | Rep 1 | Rep 2 | 中位数 | 范围 |
| --- | ---: | ---: | ---: | ---: | ---: |
| resident | 68.7696 | 68.7928 | 68.7538 | 68.7696 | 68.7538-68.7928 |
| GPU-compressed | 6.3133 | 6.3615 | 6.3735 | 6.3615 | 6.3133-6.3735 |

单位均为四卡聚合 output tokens/s。每次 repetition 都实际生成
`128 x 4 = 512 tokens`，没有通过减少请求数或提前遇到 EOS 获得表面性能。

决策门要求 GPU-compressed 三次测量的最小值严格高于 resident 最大值：

```text
6.3133 tokens/s < 68.7928 tokens/s
```

所以该门禁明确失败，相对中位数变化为 `-90.75%`。

## 显存与存储

| 指标 | resident | GPU-compressed |
| --- | ---: | ---: |
| 模型显存/卡 | 约 37.22 GiB | 约 26.01 GiB |
| 可用 KV cache/卡 | 约 6.80 GiB | 约 16.50 GiB |
| KV token capacity | 约 148k | 约 360k |

GPU-compressed 将聚合专家源权重 `154.62 GB` 编码为 `106.33 GB`，存储率约
`68.77%`。与 resident 相比，它把每卡 KV cache 提高约 `2.43` 倍，说明内存受限
目标有效，但这部分容量收益尚未转化为端到端吞吐收益。

首次构建压缩权重约需 `1748-1780 s/rank`。这是当前在线初始化成本，未计入三次
正式吞吐计时，但会显著影响冷启动。

## 机制计数与瓶颈

GPU-compressed 正式运行的四卡聚合软件计数如下：

| 指标 | 数值 |
| --- | ---: |
| 运行时 host expert H2D | 0 bytes |
| 运行时 host copy launches | 0 |
| 启动时 compressed store upload | 106.33 GB |
| GPU decode input | 22.57 TB |
| GPU decode output | 33.87 TB |
| GPU decode launches | 84,112 |
| VMM mapping count | 84,112 |

这些是 4 个 TP rank 在 warmup 和正式测量期间累计的软件逻辑计数，不是 CUPTI 或
Nsight 测得的物理链路流量。结果证明本轮已经完全移除热路径中的 host expert H2D，
但压缩权重仍在每次 materialization 时被反复解码。累计 decode input 约为启动时
compressed store 大小的 `212` 倍；大量 decode 和映射操作占据了本可用于 MoE 计算
的执行窗口。

单层 microbenchmark 中，batched decoder 的 w13/w2 合计时间已经从第一版的
`259.44 ms` 降至 `15.70 ms`，提升 `16.52` 倍且 BF16 bit-exact。然而端到端结果
表明这一带宽仍不足以隐藏每层、每轮的解压成本。当前正式 runner 尚未输出设计要求的
CUDA event `decode_cuda_s`、`compute_cuda_s` 和
`compute_wait_for_load_cuda_s`，因此不能声称已经直接测得 stall ratio；瓶颈判断来自
端到端差值和机制计数，后续应补齐 event timing 后再细分 decode 与调度等待。

## 正确性与证据边界

- 独立 GPU-compressed smoke：恢复权重 BF16 bit-exact、输出 token parity 和 router
  parity 均为 true；
- 正式运行 validator：通过；
- 正式运行每次输出长度固定为 4，工作量一致；
- 性能批次不同 repetition 的 token ID 不完全一致，记录为
  `output_tokens_stable=false`；
- 性能运行本身的跨变体 token/router 全轨迹不完全一致，正式正确性结论来自同 commit
  的独立确定性 smoke，不扩展为生产批次逐 token、逐路由完全一致；
- compute-sanitizer 对进程中的首个自定义 CUDA kernel 仍报告
  `cuKernelGetFunction invalid handle`，普通 CUDA 测试和正式运行正确；该工具链风险
  尚未关闭。

本次本地证据目录：

```text
runs/20260903T072013Z-8033eafb6878  # resident
runs/20260903T072450Z-8033eafb6878  # GPU-compressed
```

运行产物包含模型输出和机器环境元数据，不加入 Git；本报告只提交汇总后的非敏感结果。

## 后续实验计划

1. 在生命周期热路径加入 CUDA event telemetry，记录每层 decode、compute、
   compute-wait-for-load 和 overlap ratio。
2. 减少 `84,112` 次 decode/mapping：优先实现跨 token 的专家缓存或驻留策略，而不是
   继续只优化单次 kernel。
3. 将压缩权重编码结果持久化，避免每次 engine 启动重复约 30 分钟的 CPU 编码。
4. 用 Nsight Systems 或 CUPTI 校准软件计数，并确认 decode、VMM 操作与 MoE compute
   的实际重叠程度。
5. 重新执行 batch 128/context 4096/output 4 决策门。只有 GPU-compressed 的三次
   最小值超过 resident 最大值后，才恢复 output 128 的三个正式关键点。
