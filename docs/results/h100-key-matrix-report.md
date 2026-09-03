# FluxMoE H100 关键矩阵实验报告

## 结论

本轮实验结论为 **MIXED**：FluxMoE 的专家权重分页、GPU 压缩解压、
pinned-host H2D 路径和显存容量扩展均已实际运行并通过当前正确性门禁；但当前实现
没有复现论文中的吞吐量收益。三个关键点上，`fluxmoe-fixed` 均显著慢于原生
`resident`。

本报告只评价 commit `de821d7fa2d17c1314d7d1530e95b5e293459f22` 上已经完成的
`resident` 与 `fluxmoe-fixed` 对照，不把尚未运行的变体或推测结果计入结论。

## 实验配置

- GPU：4 张 NVIDIA H100 80GB PCIe，tensor parallel size 为 4；
- 模型：Qwen3-Next-80B-A3B-Instruct，BF16；
- 推理框架：vLLM v0.10.2；
- 执行模式：eager、greedy decoding；
- 显存预算：`gpu_memory_utilization=0.60`；
- 每个关键点：3 次 warmup，随后 3 次正式测量；
- 输出上限：每请求 128 tokens；
- NCCL：标准 NCCL collective，`NCCL_P2P_LEVEL=NVL`，NVLS 禁用。

实验使用固定数据集清单和固定随机种子。模型权重只读，运行结果没有加入 Git。

## 正确性与机制门禁

同一 commit 的确定性 smoke 已完成，状态为 `COMPLETE`：

- 384/384 个专家权重验证通过，恢复后的 BF16 权重逐位一致；
- 确定性单请求 greedy token 一致；
- 两次全层确定性 router Top-k probe 一致；
- 映射、H2D 和 GPU 解压计数均为非零；
- smoke 中 `output_tokens_stable=true`。

性能批次的完整 router trace 和输出 token 序列在独立运行之间不完全一致，三个关键点
均记录了 `router_full_trace_match=false`、`output_tokens_stable=false` 和
`performance_output_tokens_match=false`。因此，本报告只声明确定性 smoke 范围内的
正确性，不声明生产批次逐 token、逐路由完全一致。

## 显存容量

| 指标 | resident | fluxmoe-fixed | 变化 |
| --- | ---: | ---: | ---: |
| 每卡模型权重 | 37.215 GiB | 约 1.262 GiB | 减少约 96.6% |
| 每卡可用 KV cache | 6.80 GiB | 41.18 GiB | 增加约 6.06 倍 |
| KV token capacity | 148,240 | 899,504 | 增加约 6.07 倍 |

这证明当前实现确实把绝大部分专家权重移出了常驻 GPU 权重区，并将释放的显存转化为
KV cache 容量。

## 吞吐量

以下数值为 4 卡聚合 output tokens/s；结果列采用三次正式测量的中位数，括号中为
三次测量的最小值至最大值。

| Batch / Context | resident | fluxmoe-fixed | 相对变化 | 慢于 resident |
| --- | ---: | ---: | ---: | ---: |
| 32 / 1024 | 309.07 (308.08-312.75) | 3.60 (3.55-3.61) | -98.83% | 85.8 倍 |
| 128 / 4096 | 720.70 (652.67-728.46) | 11.71 (11.17-11.96) | -98.37% | 61.5 倍 |
| 256 / 4096 | 759.52 (749.34-772.27) | 18.55 (18.50-18.86) | -97.56% | 40.9 倍 |

更大的 batch 能摊薄一部分加载开销，`fluxmoe-fixed` 从 3.60 tokens/s 提升到
18.55 tokens/s，但三个关键点均没有接近 resident，也没有出现性能交叉点。因此，
当前证据不支持吞吐量收益。

## 数据移动计数

| Batch / Context | 累计 H2D | 累计解压输出 | 映射次数 |
| --- | ---: | ---: | ---: |
| 32 / 1024 | 120.86 TB | 210.32 GB | 300,683 |
| 128 / 4096 | 148.64 TB | 258.70 GB | 369,805 |
| 256 / 4096 | 179.20 TB | 311.85 GB | 445,838 |

这些数值是 4 个 TP rank 在初始化、3 次 warmup 和 3 次正式测量期间的累计软件
逻辑计数，不是模型大小、峰值显存，也不是 CUPTI 或 Nsight 测得的硬件链路流量。

`H2D` 表示 pinned CPU 内存复制到 GPU 显存的 BF16 专家权重字节数。相同权重每次
重新加载都会再次计数。`累计解压输出` 表示从 GPU 压缩区恢复到临时 BF16 权重窗口
的未压缩目标字节数；相同权重每次重新解压也会再次计数。

当前每卡 GPU 压缩预算只有 64 MiB，绝大多数专家权重走 host-pinned 路径。上百 TB
的累计 H2D 表明专家权重被反复加载，缺少足够的跨 token 复用和有效的传输/计算重叠。
这是当前吞吐量显著下降的主要证据。

## 证据边界

1. H100 不是论文使用的 L40 测试平台，本报告不比较绝对吞吐量是否逐项等同于论文。
2. vLLM 会将超过 KV 容量的请求排队执行。`resident` 在 batch 128 和 256 上没有 OOM，
   不能证明所有请求同时占用 KV cache，也不能作为严格容量边界。
3. H2D 和解压字节数来自软件 receipt 累加，后续需要使用 CUPTI 或 Nsight Systems
   校准真实 PCIe/HBM 流量和传输计算重叠比例。
4. 本轮没有运行 `vllm-o`、`fluxmoe-dynamic`、unbalanced 或
   `pagedtensor-resident`，不能据此比较这些变体。
5. 性能批次输出不完全稳定。虽然吞吐量按实际生成 token 数归一化，但跨变体对照仍
   不是严格相同输出工作量。

## 后续实验计划

1. 为每次 repetition 原子写入 checkpoint，降低长时间实验中断后的数据损失风险。
2. 固定性能测试的实际解码长度，消除 EOS 导致的跨轮次输出工作量差异。
3. 实现专家权重跨 token 缓存、next-layer 异步预取、双缓冲以及 CUDA event 生命周期，
   优先降低重复 H2D 并提高传输/计算重叠。
4. 重新运行三个关键点。只有性能明显改善后，才运行 dynamic、unbalanced、
   `vllm-o` 和 `pagedtensor-resident` 对照。
5. 增加 scheduler admission barrier，构造请求同时进入 KV 阶段的容量边界实验，避免
   把 vLLM 排队误判为并发容量。
6. 使用 Nsight Systems 或 CUPTI 校准软件计数并记录 PCIe 带宽、GPU 利用率和每层
   load/compute/stall 时间。
7. 只有三个关键点有效且至少一个压力点具有信息量时，再运行完整 batch/context 矩阵。

## 最终判定

- 专家权重分页和存储层级：**SUPPORTED**；
- BF16 权重恢复与确定性 smoke 正确性：**SUPPORTED**；
- GPU 显存释放与 KV cache 容量增长：**SUPPORTED**；
- 当前实现的吞吐量收益：**NOT SUPPORTED**；
- 对 FluxMoE 论文整体的复现结论：**MIXED**。
