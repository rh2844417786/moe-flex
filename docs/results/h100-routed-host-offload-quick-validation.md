# H100 命中专家按需调入快速验证

## 结论

在 4 张 H100 80GB、TP=4 上，新增的 `fluxmoe-routed-host-offload` 路径已经能够在
router Top-K 产生后，仅将本次前向实际命中的去重专家从 pinned host memory 调入 GPU。
该路径保持原始 BF16，不使用压缩、解压或 GPU decode。

本次 batch `1`、context `1024`、output `4` 的快速正确性实验得到：

- 输出 token 与 resident 完全一致；
- router Top-K probe 与 resident 完全一致；
- `131,918/131,918` 个实际访问的专家权重通过 BF16 位级校验；
- 平均每次 layer-kind 调入约 `78.25/512` 个专家；
- 实际 H2D 为 `637.99 GB`，相对同次数全专家调入的 `4.17 TB` 理论值减少 `84.72%`；
- resident 中位吞吐为 `2.4262 output tokens/s`；
- routed host-offload 中位吞吐为 `0.2498 output tokens/s`，约为 resident 的 `10.30%`。

因此，“仅传输命中专家”已经显著降低传输量并保持结果正确，但当前同步调入路径仍未使
短请求吞吐超过 resident。下一阶段需要做专家缓存、合并 H2D 和预取，而不是重新引入压缩。

## 实验配置

- 代码提交：`3bc14bc76f321c884f887c92000b07ddab848abe`；
- GPU：4 张 NVIDIA H100 80GB，使用 GPU 4、5、6、7；
- 模型：Qwen3-Next-80B-A3B-Instruct，BF16；
- 推理框架：vLLM v0.10.2；
- tensor parallel size：`4`；
- batch：`1`；
- context：`1024`；
- output：每请求固定 `4 tokens`；
- 3 次 warmup，3 次正式测量；
- `gpu_memory_utilization=0.60`、eager、greedy decoding；
- 实验进程标识：`wth333`。

固定数据集没有 128-token 请求，因此没有临时生成或改写数据，而是使用其中最短的
1024-token 请求。这个选择保持了数据工件可追溯，但 prefill 会命中较多专家；本结果是
机制和正确性快速验证，不是容量压力吞吐结论。

## 实现路径

新路径在 fused-MoE 完成 Top-K 选择之后提取去重 expert IDs，然后分别为 `w13` 和 `w2`
建立当前层 VMM 映射，只向命中专家对应的 tensor 子区间提交 pinned-host BF16 H2D。kernel
完成后同步当前层计算并回收映射。未命中专家不提交 H2D。

首次实机验证发现，vLLM profile 可能通过辅助 CUDA stream 访问复用后的专家窗口，原来的
单 stream 校验栅栏不足，导致 D2H 哈希读取与跨 stream 工作竞争。修复后在正确性校验前
执行 device fence，并新增“非默认 stream + 同一物理块跨层复用 + 非零 expert ID”的 CUDA
回归测试。修复后的完整 profile 和正式采样均通过位级校验。

## 传输证据

| 计数器 | 数值 |
| --- | ---: |
| expert source bytes | `154,618,822,656` |
| routed layer-kind loads | `10,368` |
| routed expert loads | `811,246` |
| runtime host copy launches | `811,246` |
| runtime host expert H2D bytes | `637,989,814,272` |
| VMM mapped bytes | `4,174,708,211,712` |
| VMM mapping count | `10,368` |
| GPU compressed source/storage bytes | `0 / 0` |
| startup GPU store upload bytes | `0` |
| GPU decode input/output bytes | `0 / 0` |
| GPU decode launches | `0` |
| decompressed bytes | `0` |

平均命中专家数与传输比例为：

```text
average selected experts = 811,246 / 10,368 = 78.2459
selected fraction        = 78.2459 / 512 = 15.28%
H2D reduction            = 1 - 637,989,814,272 / 4,174,708,211,712
                         = 84.72%
```

这里的 H2D 是软件 receipt 累计值，覆盖 vLLM profiling、warmup 和正式测量，不是 CUPTI
测得的 PCIe 物理链路流量。`VMM mapped bytes` 表示完整虚拟层窗口的累计映射大小；它不是
实际传输量，可用于计算同次数全专家物化的上界。

## KV 容量

| 指标 | resident | routed host-offload |
| --- | ---: | ---: |
| 模型权重显存/卡 | 约 `37.22 GiB` | 约 `1.22 GiB` |
| 可用 KV cache/卡 | 约 `6.80-7.17 GiB` | 约 `41.23 GiB` |
| KV token capacity | `148,240-156,400` | `900,320` |

routed 路径将 KV token capacity 提高到 resident 的约 `5.76-6.07x`。这证明按需调入专家
确实释放了权重显存并扩大 KV cache；本次 batch=1 并不是 KV 容量压力点，因此不能用本轮
吞吐判断大 batch 下的最终容量收益。

## 吞吐结果

| repetition | resident | routed host-offload |
| ---: | ---: | ---: |
| 1 | `3.0459` | `0.2498` |
| 2 | `2.4262` | `0.2256` |
| 3 | `2.2002` | `0.2645` |
| 中位数 | `2.4262` | `0.2498` |

单位为四卡聚合 output tokens/s。routed 中位吞吐约为 resident 的 `10.30%`，慢约 `9.71x`。
短请求下，逐层同步 VMM map、多个小 H2D copy 和回收的固定成本无法被更大 KV cache 摊薄。

## 时长与边界

- resident 对照端到端约 `3.1 min`；
- 修复后的 routed 验证端到端约 `13.3 min`；
- routed 权重加载与 pinned-host 存储构建约 `8.3 min`；
- routed profile、KV 初始化、warmup 和正式测量约 `4.2 min`。

目标是将快速验证控制在约 10 分钟；本次因保留 3 次 warmup、3 次正式测量和完整位级校验，
实际 routed 时长超出约 3 分钟。后续快速 gate 应增加明确的 `1 warmup + 1 repetition` 模式，
但正式结果仍保持 `3 + 3`。

## 后续优化

1. 增加同层跨 decode step 的专家缓存，用命中率和淘汰计数验证复用收益。
2. 将连续 expert ID 合并为较大的 H2D 区间，减少 copy launch 数。
3. 使用 next-layer router 结果做双缓冲预取，并记录 compute-wait-for-load 时间。
4. 在 batch `256`、context `4096`、output `128` 的容量压力点重测，与已有全专家
   host-offload 的 `19.36 tokens/s` 和 resident 的约 `799 tokens/s` 做正式对照。

## 本地证据标识

```text
20260904T102209Z-9842cb642be8  # resident quick reference
20260904T104316Z-3bc14bc76f32  # routed host-offload quick validation
```

运行产物未加入 Git；本文只提交汇总后的非敏感结果。
