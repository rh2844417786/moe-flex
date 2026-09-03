# H100 Flex-MoE 当前实验瓶颈报告

## 结论

当前 Flex-MoE 实现的主要瓶颈不是 KV cache，也不是 NCCL，而是专家权重的逐层、全量、
重复物化。它确实把专家权重移出 GPU，并将 KV cache capacity 从 resident 的
`148,240 tokens` 提高到 `899,504 tokens`，约 `6.07x`；但在高 batch、长输出压力点上，
权重搬运成本远高于 KV 扩容带来的调度收益。

当前 commit `4a6deb05c1ee8b9bcb7d4c431ae91f381b84f76e` 的正式压力运行结果为：

- Flex-MoE：中位 `21.3503 tokens/s`；
- 同点 resident：中位 `789.9871 tokens/s`；
- Flex-MoE 仅为 resident 的约 `2.70%`，慢约 `37.0x`；
- 两组均为 batch `256`、context `4096`、每请求固定输出 `128 tokens`；
- 每轮均生成 `32,768 tokens`。

因此，当前版本不能证明“通过降低专家权重显存、增加 KV cache 后提升端到端吞吐”。

## 实验配置

- GPU：4 张 NVIDIA H100 80GB，tensor parallel size 为 4；
- 模型：Qwen3-Next-80B-A3B-Instruct，BF16；
- 推理框架：vLLM v0.10.2；
- `gpu_memory_utilization=0.60`；
- Flex-MoE：`storage_mode=hybrid`、`gpu_materialization_mode=expertwise`、GPU 压缩预算
  `64 MiB`；
- resident：原生常驻专家权重；
- 执行模式：eager、greedy decoding；
- 实验进程标识：`wth333`。

## 显存结果

| 指标 | resident | Flex-MoE |
| --- | ---: | ---: |
| 模型权重显存/卡 | 约 37.22 GiB | 约 1.26 GiB |
| 可用 KV cache/卡 | 约 6.80 GiB | 约 41.18 GiB |
| KV token capacity | 148,240 | 899,504 |

Flex-MoE 释放了约 `35.95 GiB/卡` 的专家权重空间，KV capacity 增加约 `6.07x`。这部分
容量收益是真实的，但 vLLM 仍可能通过 scheduler 分批/排队处理超过单批 KV 池的请求，
因此 capacity 增加不会自动等价为同倍吞吐增加。

## 权重物化路径

“物化”是将 CPU 或压缩存储中的专家权重准备成 GPU 上 fused-MoE kernel 可以直接访问的
BF16 tensor，包括 VMM map、数据搬运或 GPU 解压。

当前路径在 [bridge.py](/home/jovyan/wangtonghan/moe-flex/src/flexmoe/vllm/bridge.py:264)
为每一层建立临时权重窗口；expertwise 分支遍历该层的每个 expert（第 278-282 行），
然后调用 [hierarchy.py](/home/jovyan/wangtonghan/moe-flex/src/flexmoe/storage/hierarchy.py:36)
对该层全部 tensor 逐个 materialize（第 60-67 行）。计算完成后，生命周期在
[lifecycle.py](/home/jovyan/wangtonghan/moe-flex/src/flexmoe/runtime/lifecycle.py:215)
等待 compute event、回收旧层并加载下一层。

这意味着当前实现不是只搬 router 选中的 Top-K 专家，而是接近“每层所有专家都搬一次”；
下一次经过该层时又重复搬运。这里的“反复”包括 warmup、prefill、decode step 以及后续
重新进入同一层的每一轮。

## 机制计数

当前 Flex-MoE 正式运行累计计数：

| 指标 | 数值 |
| --- | ---: |
| runtime host expert H2D | `179.20 TB` |
| runtime host copy launches | `227,967,616` |
| GPU decode input | `233.95 GB` |
| GPU decode output | `311.79 GB` |
| GPU decode launches | `297,344` |
| VMM mapping count | `445,837` |

`179.20 TB` 的 H2D 是软件 receipt 累计值，不是 CUPTI 测得的物理链路流量；它仍然足以
说明权重在热路径被大量重复提交。以本次 `32,768` 个输出 token 计，累计 H2D 约为
`5.82 GiB/token`，而 resident 没有对应的运行时专家 H2D。

运行时 GPU 利用率抽样约为 `10%~27%`，四个 worker 各自约占一个 CPU 核心，说明 GPU
大部分时间在等待 CPU 调度、VMM 操作和 pinned-host 数据搬运。NCCL 初始化成功；日志中
出现的 torch dynamo shape recompilation warning 不是主要耗时来源。

## 性能证据

Flex-MoE 三次正式测量：

```text
20.7421 tokens/s  (1579.78 s, 32768 tokens)
21.3503 tokens/s  (1534.78 s, 32768 tokens)
21.8928 tokens/s  (1496.75 s, 32768 tokens)
```

resident 同点三次测量：

```text
785.1534 tokens/s
790.5421 tokens/s
789.9871 tokens/s
```

Flex-MoE run 状态为 `MEASURED_UNVALIDATED`：本次性能 run 的 reference pointer 指向
前一 commit 的 resident 目录，因此不把它当作跨 commit 的 correctness evidence。本文
只使用其固定工作量、吞吐和机制计数作为瓶颈证据；确定性正确性仍以同 storage mode 的
独立 smoke 为准。

## 根因判断

1. **加载粒度过大**：当前 expertwise 物化遍历整层全部专家，未利用 router Top-K 稀疏性。
2. **复用不足**：两层窗口保证 pointer 稳定，但没有跨 token 的专家权重缓存；每次回收后
   再次经过该层都要重新 H2D。
3. **搬运与计算重叠不足**：GPU 利用率低、CPU 高负载，表明异步 copy 提交和生命周期
   等待仍让 compute stream 等待 load stream。
4. **容量收益被排队成本掩盖**：KV capacity 增大使更多请求有机会驻留，但不能抵消每层
   全专家搬运的固定成本。

## 修复优先级

1. 改为 routed-expert-aware materialization，只加载当前 batch 实际需要的专家集合，并
   合并同层专家的 copy；这是收益最大的结构性修复。
2. 增加跨 token/跨请求的专家缓存和命中率 telemetry，避免同一专家在短时间内重复 H2D。
3. 实现 next-layer 预取、双缓冲和 CUDA event timing，分别记录 load、compute 和
   compute-wait-for-load，确认实际 overlap/stall ratio。
4. 将压缩编码结果持久化，避免每次 engine 启动重复数分钟的 CPU 编码。
5. 修复性能 run 的同 commit reference evidence，再以相同 workload 重测 resident 与
   Flex-MoE；只有 Flex-MoE 三次最小吞吐超过 resident 最大吞吐，才能声明容量扩容带来
   端到端收益。

## 证据目录

```text
runs/20260903T124926Z-4a6deb05c1ee  # Flex-MoE 当前压力运行
runs/20260903T122936Z-4879f821893b  # resident 同点对照
```

运行产物未加入 Git；本报告只提交汇总后的非敏感信息。
