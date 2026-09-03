# H100 KV Cache 容量与吞吐对照报告

## 结论

在 resident、无解压路径上，仅提高 vLLM 的 GPU memory utilization，就能在高
batch、长输出场景提高吞吐。`gpu_memory_utilization=0.90` 将实际 KV capacity 从
`148,240` 提高到 `667,216 tokens`，约 `4.50x`；同一工作量下中位吞吐从
`789.99` 提高到 `1,283.88 tokens/s`，约 `1.63x`，相对提升 `62.52%`。

该结果验证了 KV 容量扩展本身对受压调度场景有收益，但它不改变此前 GPU-compressed
路径的结论：GPU-compressed 的主要开销仍是重复专家解压与映射。此次测试使用 resident
路径，目的是隔离 KV 容量变量。

## 实验配置

- GPU：4 张 NVIDIA H100 80GB，tensor parallel size 为 4；
- 模型：Qwen3-Next-80B-A3B-Instruct，BF16；
- 推理框架：vLLM v0.10.2；
- batch size：256；
- context length：4096；
- 固定输出：每请求 128 tokens，忽略 EOS；
- 每组：3 次 warmup，随后 3 次正式测量；
- 执行模式：resident、eager、greedy decoding；
- 唯一自变量：`gpu_memory_utilization`，从 `0.60` 改为 `0.90`；
- 实验进程标识：`wth333`。

两组每次都实际生成 `256 x 128 = 32,768 tokens`，没有通过减少请求数或提前遇到
EOS 改变工作量。两组的 mechanism counters 均为零：没有 GPU decode、VMM mapping
或运行时专家 H2D。

## KV capacity 与吞吐

| GPU memory utilization | KV cache/卡 | KV token capacity | Rep 0 | Rep 1 | Rep 2 | 中位数 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.60 | 约 6.80 GiB | 148,240 | 785.15 | 790.54 | 789.99 | 789.99 |
| 0.90 | 约 30.55 GiB | 667,216 | 1,288.73 | 1,283.88 | 1,256.92 | 1,283.88 |

吞吐单位为四卡聚合 output tokens/s。容量和中位吞吐变化为：

```text
KV capacity ratio = 667,216 / 148,240 = 4.50x
throughput ratio  = 1,283.88 / 789.99 = 1.63x
throughput delta   = +62.52%
```

## 解释边界

本实验点的总输入规模为 `256 x 4096`，低容量组仍能完成全部请求，说明 scheduler
通过分批/排队处理了超出单批 KV 池的压力。扩容后的吞吐提升与更高的同时驻留请求数
一致，但当前 runner 没有记录 peak active requests、scheduler admission 和
KV occupancy，因此不能把提升精确归因到某一个调度计数，也不能据此宣称所有 256
请求始终同时驻留。

同时，较高 utilization 组的吞吐范围为 `1,256.92-1,288.73 tokens/s`，比低容量组
的 `785.15-790.54 tokens/s` 略有更大波动。后续应补齐 admission、occupancy、TTFT
和 P99 延迟，确认扩容收益没有由排队延迟转移而抵消。

## 证据目录

```text
runs/20260903T122936Z-4879f821893b  # utilization 0.60
runs/20260903T123713Z-4879f821893b  # utilization 0.90
```

两组 run 的状态均为 `BASELINE_COMPLETE`，commit 均为
`4879f821893b9aa93c718c2802e09a0998ef7fee`。运行产物未加入 Git。

## 后续计划

1. 在 runner 中加入 peak active requests、peak batched tokens、scheduler admission
   和 KV occupancy telemetry。
2. 对 `0.60`、`0.75`、`0.90` 做同一 workload 的容量曲线，并记录 TTFT/P99。
3. 在完成 decoder 缓存和预取优化后，用同样容量曲线重新测试 GPU-compressed，区分
   KV 容量收益和专家解压开销。
