# H100 无压缩专家卸载吞吐实验

## 结论

在 4 张 H100 80GB、TP=4 上，将全部 MoE 专家权重以原始 BF16 保存在 pinned host
memory，并在运行时通过 VMM/H2D 逐层调入 GPU，确实显著增加了 KV cache；但当前实现的
端到端吞吐远低于 resident 基线。

- KV token capacity 从 `148,240` 增至 `900,320`，约为 `6.07x`；
- 每卡模型显存从约 `37.22 GiB` 降至约 `1.22 GiB`；
- 每卡可用 KV cache 从约 `6.80 GiB` 增至约 `41.22 GiB`；
- resident 吞吐中位数为 `799.02 tokens/s`；
- 无压缩 host-offload 吞吐中位数为 `19.36 tokens/s`；
- host-offload 仅为 resident 的 `2.42%`，吞吐下降 `97.58%`，约慢 `41.27x`。

因此，本次实验不支持“仅通过无压缩专家调入调出释放显存、扩大 KV cache，就能提高当前
实现的端到端吞吐”这一假设。容量收益存在，但被全专家逐层重复 H2D 的开销完全掩盖。

## 实验约束

- 代码提交：`5e666b83842ce5dbdfb1144dc93ac73cbe52254b`；
- GPU：4 张 NVIDIA H100 80GB，使用 GPU 4、5、6、7；
- 模型：Qwen3-Next-80B-A3B-Instruct，BF16；
- 框架：vLLM v0.10.2；
- workload：batch `256`、context `4096`、每请求输出 `128 tokens`；
- 每轮固定生成 `32,768 tokens`；
- 3 次 warmup，3 次正式测量；
- `gpu_memory_utilization=0.60`、eager、greedy decoding；
- resident 与 host-offload 使用同一提交、同一数据、同一配置；
- 实验进程标识为 `wth333`。

host-offload 的关键配置为：

```yaml
variant: fluxmoe-host-offload
gpu_compressed_budget_bytes: 0
storage_mode: hybrid
gpu_materialization_mode: expertwise
```

## 吞吐结果

| repetition | resident | 无压缩 host-offload |
| ---: | ---: | ---: |
| 1 | 794.1154 tokens/s | 21.1286 tokens/s |
| 2 | 799.8705 tokens/s | 18.7955 tokens/s |
| 3 | 799.0197 tokens/s | 19.3590 tokens/s |
| 中位数 | 799.0197 tokens/s | 19.3590 tokens/s |

host-offload 三次正式测量分别耗时 `1550.88 s`、`1743.39 s` 和 `1692.65 s`，正式测量
合计约 `83.12 min`。resident 三次分别约 `41.26 s`、`40.97 s` 和 `41.01 s`。

## 显存与 KV 容量

| 指标 | resident | 无压缩 host-offload |
| --- | ---: | ---: |
| 模型权重显存/卡 | 约 37.22 GiB | 约 1.22 GiB |
| 可用 KV cache/卡 | 约 6.80 GiB | 约 41.22 GiB |
| KV token capacity | 148,240 | 900,320 |

这说明专家卸载本身达到了释放显存并扩大 KV cache 的目标。resident 在该压力点需要受较小
KV 池约束进行调度，host-offload 则能提供约 `6.07x` 的 KV token capacity，但更大的并发
驻留没有抵消权重搬运成本。

## 无压缩证据

正式 host-offload 运行的机制计数如下：

| 计数器 | 数值 |
| --- | ---: |
| expert source bytes | 154,618,822,656 |
| runtime host expert H2D bytes | 179,516,479,635,456 |
| runtime host copy launches | 228,266,496 |
| VMM mapped bytes | 179,518,358,683,648 |
| VMM mapping count | 445,838 |
| GPU compressed source bytes | 0 |
| GPU compressed storage bytes | 0 |
| startup GPU store upload bytes | 0 |
| GPU decode input/output bytes | 0 / 0 |
| GPU decode launches | 0 |
| decompressed bytes | 0 |

累计 H2D 为 `179.52 TB`（十进制，约 `163.27 TiB`）。这是软件提交计数，不是 CUPTI
测得的 PCIe 物理链路流量；它覆盖初始化 profiling、3 次 warmup 和 3 次正式测量。平均
每次 host copy 仅约 `768 KiB`，大量小拷贝进一步放大了调度开销。

所有压缩存储、上传、解码和解压计数均严格为零，证明本实验没有使用 Huffman 或其他专家
权重压缩路径。

## 正确性与证据状态

独立 correctness smoke 使用 batch `1`、context `1024`、output `4`，与同提交 resident
逐 token 对比，结果为：

- 输出 token 一致；
- router Top-K probe 一致；
- 专家权重 `384/384` 位级校验通过；
- host H2D 和 VMM mapping 为正；
- 全部压缩、GPU decode 和 decompression 计数为零。

正式运行复用该同提交、同变体的 correctness evidence，并通过 artifact validator，状态为
`COMPLETE`。压力 workload 下完整输出 token IDs 与 resident 不一致；这是当前 vLLM
Qwen3-Next hybrid scheduler 在高 batch 下已知的 batch packing 非不变量。两组仍使用固定
请求数和固定输出长度，因此吞吐比较的生成 token 数均严格为 `32,768`，但不能把压力点的
完整输出当作逐 token 正确性证据。

## 瓶颈判断

当前 expertwise 路径为每个 MoE 层物化整层全部专家，而不是只调入 router 实际选中的
Top-K 专家。相同层在后续 prefill/decode step 中再次执行时，全部专家又会重复 H2D。

本次零压缩实验相对旧的 64 MiB GPU 压缩预算路径没有减少这种全量搬运，反而使原先极小的
GPU 压缩驻留部分也改为原始 BF16 H2D。因此 KV capacity 略增至 `900,320 tokens`，吞吐
中位数却从旧路径约 `21.35 tokens/s` 降至 `19.36 tokens/s`。下一步应优先实现 routed
expert materialization、跨 token 专家缓存和合并 H2D，而不是继续单独增大 KV cache。

## 本地证据标识

```text
20260904T044044Z-5e666b83842c  # resident correctness
20260904T044411Z-5e666b83842c  # uncompressed host-offload correctness
20260904T045928Z-5e666b83842c  # resident throughput
20260904T050717Z-5e666b83842c  # uncompressed host-offload throughput
```

运行产物未加入 Git；本文只提交汇总后的非敏感结果。
