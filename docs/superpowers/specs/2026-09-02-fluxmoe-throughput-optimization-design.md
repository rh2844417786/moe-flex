# FluxMoE 吞吐优先优化设计

## 1. 背景与问题

当前 H100 关键矩阵已经证明 PagedTensor、GPU Huffman 解压、pinned-host H2D
和两层滑动窗口可以运行，但 `fluxmoe-fixed` 在三个关键点上分别比 resident 慢
85.8、61.5 和 40.9 倍。本设计只处理导致该吞吐差距的主数据路径，不在同一阶段
引入动态 residency planner。

现有实现有两个直接瓶颈：

1. `gpu_compressed_budget_bytes` 只有 64 MiB，绝大多数专家权重来自 pinned host，
   初始化、warmup 和测量期间累计产生 120.86–179.20 TB 的软件逻辑 H2D；
2. 存储对象和物化调用按单个专家权重张量组织。每层每个 tensor kind 都要经过
   多次 Python 调度和 CUDA kernel 或 H2D copy，而物理窗口和 vLLM kernel 实际消费
   的是整层连续权重。

论文在 Qwen3-Next-80B-A3B-Instruct 的 performance-bound 配置中使用压缩 GPU
后端保存全部 routed-expert 权重。本阶段首先实现这一可独立归因的路径，以确认消除
PCIe 和物化调用碎片后吞吐是否真实提高。

## 2. 目标与优先级

本阶段优先级从高到低为：

1. 在相同模型、数据、TP、batch、context 和生成 token 数下，提高
   `fluxmoe` 的 output tokens/s；
2. 保持 BF16 bit-exact、单请求 greedy token parity 和 router probe；
3. 保留比 resident 更多的 KV cache，而不追求当前 41.18 GiB/卡；
4. 保存足够的原始证据，使吞吐变化可归因到数据路径，而不是请求长度或排队差异。

本阶段不把“代码更快”“microbenchmark 更快”或“单次运行更快”作为成功。正式结论
必须来自服务器上三次 warmup 后的三次独立测量。

## 3. 范围

### 3.1 本阶段包含

- 新增全压缩 GPU 的明确实验变体；
- 将 GPU 压缩存储从逐专家对象改为按层、按 tensor kind 打包；
- 每层 `w13` 和 `w2` 各使用一次批量解压 launch；
- 保留并核验两层 VMM 滑动窗口和下一层异步预取；
- 固定实际生成 token 数；
- 增加压缩占用、物化 launch、加载和等待时间 telemetry；
- 重新运行 smoke 和三个关键吞吐点。

### 3.2 本阶段不包含

- host/GPU 混合放置的四流并行重构；
- runtime residency level `alpha` 调节；
- 跨 token 的 routed-expert 子集预测或缓存；
- CUDA Graph；
- vLLM-O、unbalanced、Mixtral 或完整 batch/context 矩阵；
- 为取得好看结果而改变模型、权重精度或模型输出。

这些内容只有在全压缩 GPU 主路径达到本设计的吞吐门槛后才进入下一份设计。

## 4. 方案选择

### 4.1 未选择：只提高 64 MiB 配置

仅增大 `gpu_compressed_budget_bytes` 可以判断 PCIe 是否为主要瓶颈，但仍保留逐专家
Python/CUDA 调用和大量独立压缩对象。它可以作为一次诊断点，不能成为最终优化。

### 4.2 采用：全压缩 GPU + 按层批量物化

将每个 MoE 层的全部专家按 tensor kind 打包为两个逻辑对象：

- `layer.<i>.all.w13`；
- `layer.<i>.all.w2`。

每个对象覆盖对应 vLLM Parameter 的完整连续视图。进入两层物理窗口时，两个
tensor-kind worker 分别向自己的 load stream 发起一次 Huffman decode。当前层计算
消费已映射的完整 Parameter，下一层解压与当前层计算重叠。

该方案同时消除 H2D 和逐专家 launch 碎片，而且没有改变 fused MoE kernel 的输入
布局、参数地址或数学结果，因此是本阶段风险最低且最容易归因的架构改造。

### 4.3 延后：一次性完成混合后端和动态 planner

混合后端需要每个 tensor kind 至少两个 backend stream、跨流 join event、按层
带宽比例切片以及在线显存安全控制。若与按层打包同时引入，性能变化无法区分来自
GPU residency、launch 合并、流并行还是 planner。本阶段不采用。

## 5. 运行变体与配置语义

新增实验变体 `fluxmoe-gpu-compressed`，保留历史 `fluxmoe-fixed` 配置和报告，不覆盖
既有 64 MiB 结果。

`fluxmoe-gpu-compressed` 必须满足：

- routed-expert backing store 全部是 GPU compressed；
- `gpu_compressed_budget_bytes` 表示允许接纳的未压缩源张量总字节上限；
- 配置预算必须不小于全部 routed-expert 未压缩字节数，否则启动失败；
- 实际 GPU 占用以编码后的 payload、sign/mantissa、chunk metadata、codebook 和
  error buffer 的总和记录，不能用未压缩预算冒充；
- 不创建 pinned-host expert store；
- 分配或初始化 OOM 时直接失败，不回退到 host。

变体环境仍使用 `FLUXMOE_ENABLE=1` 和 fixed planner mode，但额外设置显式 storage
mode，避免通过一个恰好很大的数字猜测运行语义。

## 6. 按层压缩存储

### 6.1 打包

权重加载仍按 vLLM 提供的专家 shard 完成。所有专家收集完成后，registry 按层处理：

1. 按 expert ID 从 0 到 `num_experts - 1` 拼接 `w13`；
2. 用相同顺序拼接 `w2`；
3. 对整层连续 BF16 bits 分别执行 canonical Huffman 编码；
4. 将两个编码对象准备到当前 TP rank 的 GPU；
5. 释放该层临时 CPU 打包对象；
6. 完成全部层后清理逐专家加载暂存。

打包顺序是正确性合同的一部分。缺少、重复或乱序 expert ID 必须在编码前失败。

### 6.2 物化

按层 materializer 不再构造 `num_experts` 个 destination。它直接取得当前层完整
Parameter 视图，并调用一次对应 store：

```text
w13 worker: decode(layer.i.all.w13 -> paged_w13[i], w13_load_stream)
w2 worker:  decode(layer.i.all.w2  -> paged_w2[i],  w2_load_stream)
```

每个 tensor kind 每层每次物化恰好产生一个 decode launch。两种 tensor kind 可并行，
同一个 tensor kind 内保持顺序以复用现有生命周期和 CUDA event 合同。

### 6.3 Huffman kernel 边界

第一轮不改变编码格式和 CUDA 解码算法，只扩大一次 launch 覆盖的连续张量范围。
如果按层打包后仍达不到吞吐门槛，再依据 decode bandwidth 和 stall telemetry 单独设计
并行 Huffman kernel；不能在没有证据时同时重写格式和调度。

## 7. VMM 与加载流水线

保留每个 tensor kind 两个物理 layer block：当前层和下一层。生命周期仍为：

```text
map next layer -> async decode -> record load event
                                  |
current compute -> record compute event -> safe unmap/recycle
```

必须维持以下不变量：

- 每个逻辑 Parameter 的 data pointer 在进程生命周期内不变；
- layer `i` 计算前等待其 load event；
- layer `i-1` 的物理块只能在 compute event 完成后复用；
- `w13` 与 `w2` 都 ready 后才能进入 fused MoE；
- worker 异常必须传回推理线程并终止运行；
- close 必须排空任务、同步 load stream 并释放压缩存储。

本阶段不增加物理窗口层数。全压缩 backing store 不等于解压后的 dense 权重永久驻留；
解压目标仍只有两层，因此仍保留显著的 KV cache 空间。

## 8. 固定工作量与基线

旧报告 commit `de821d7fa2d17c1314d7d1530e95b5e293459f22` 的三个关键点是不可覆盖的
历史基线。新运行使用相同模型、数据 SHA、TP=4、BF16、eager、seed、warmup 和
repetition，仅允许以下有记录的变化：

- 运行变体改为 `fluxmoe-gpu-compressed`；
- storage mode 和 GPU compressed budget 改变；
- sampling 强制生成恰好 128 tokens，忽略 EOS；
- 新增 telemetry。

每次 repetition 必须记录请求数、每请求生成长度和总生成 token 数。任何请求没有
生成 128 tokens 时，该 repetition 不进入吞吐比较。

三点保持为：

1. batch 32、context 1024；
2. batch 128、context 4096；
3. batch 256、context 4096。

resident 需要使用同一固定长度协议重新运行，作为公平的当前基线；旧 resident 数字
只用于观察历史变化，不能与新协议直接形成论文级比例。

同一优化 commit 还必须使用原有 64 MiB GPU 压缩预算重跑 `fluxmoe-fixed`，形成
host-heavy 控制组。该控制组与 `fluxmoe-gpu-compressed` 使用完全相同的固定长度协议，
用于区分“消除 PCIe”带来的收益。随后再运行一个保持全 GPU 压缩、但仍采用逐专家
物化的诊断模式，与按层批量物化比较，用于隔离 launch 合并的收益。诊断模式只进入
原始结果和消融表，不作为最终方案。

## 9. Telemetry 与可归因性

每个 TP rank 至少记录：

- `expert_source_bytes`：全部专家未压缩 BF16 字节；
- `gpu_compressed_source_bytes`：被 GPU 压缩存储覆盖的未压缩源字节；
- `gpu_compressed_storage_bytes`：全部 GPU 编码对象实际占用；
- `compression_ratio`；
- `gpu_decode_output_bytes`；
- `gpu_decode_launches`；
- `host_h2d_bytes` 和 `host_copy_launches`；
- 每层 load CUDA elapsed time；
- compute 等待 load event 的 stall time；
- VMM mapping count 和 mapped bytes；
- vLLM 报告的 model/KV cache memory；
- 请求级 TTFT 和端到端 latency 的 P50/P95/P99；
- 每次 repetition 的 elapsed time、generated tokens 和 output tokens/s。

聚合结果必须同时保留 rank-local 数值，避免只看总和掩盖单 rank 慢尾。

全压缩 GPU 运行的机制门禁为：

- `host_h2d_bytes == 0`；
- `host_copy_launches == 0`；
- `gpu_decode_output_bytes > 0`；
- 每个 rank 的 `gpu_decode_launches` 等于实际 materialized layer-kind 次数；
- `gpu_compressed_source_bytes == expert_source_bytes`；
- 映射和物理块复用计数非零。

## 10. 正确性与失败处理

Smoke 继续使用单请求、context 1024，并至少覆盖 prefill 和多个 decode step。必须通过：

- 所有层 `w13`、`w2` 恢复后 BF16 bits 与原始 checkpoint 一致；
- resident 与全压缩 GPU 的 greedy output token IDs 一致；
- resident 与全压缩 GPU 的确定性 router expert-set probe 一致；
- VMM pointer stability；
- CUDA memcheck 和 racecheck；
- 所有机制计数满足第 9 节门禁。

以下情况必须写入结构化错误并停止：

- GPU 无法容纳完整压缩 expert store；
- 存在任何 host expert placement；
- 打包形状、expert 数量或顺序错误；
- 解压错误、权重 hash 不一致或 CUDA lifecycle 错误；
- 固定输出长度未满足；
- 服务器 GPU 非独占；
- 原始运行产物缺失。

不允许失败后静默运行 resident，也不允许借用历史 smoke 为不同代码 commit 提供正确性。

## 11. 吞吐验收

每个点取三次正式测量的中位数，同时报告最小值和最大值。

### 11.1 确认提升

三个关键点上，新版本三次测量的最小值都必须高于同一 commit、同一固定长度协议下
64 MiB host-heavy 控制组对应点的最大值。只有满足这一条件才声明“吞吐提高”，从而
排除运行波动。与 commit `de821d7` 历史结果的比较仅作为方向性参考。

### 11.2 有意义提升

- 至少两个关键点的中位数达到同协议 host-heavy 控制组的 2 倍；
- 第三个关键点的中位数至少达到该控制组的 1.25 倍；
- 任一点不能因错误、OOM 或请求减少而获得表面加速。

达到 11.1 但未达到 11.2 时，结论写为 `IMPROVED_BUT_INSUFFICIENT`，继续依据
load/stall 证据优化，不将其写成论文性能复现。

### 11.3 论文趋势

只有以下条件同时满足，才把本阶段升级为论文吞吐趋势得到支持：

- 固定工作量协议下，高 batch/长 context 的相对表现优于小 batch；
- KV cache 容量仍高于同协议 resident；
- 与 resident 的差距显著缩小，或在真实 KV 压力下超过 resident；
- TTFT、错误、OOM 和排队没有抵消 output throughput 收益。

本设计不预设必须达到论文的 3.0 倍，也不允许把单次或 microbenchmark 加速替代端到端
结果。

## 12. 测试策略

### 12.1 MacBook

- 层打包顺序、形状和 tensor ID；
- 缺失或重复 expert 的 fail-closed 行为；
- 全压缩模式配置解析和禁止 host fallback；
- 压缩实际占用统计；
- launch/copy 计数聚合；
- 固定输出长度配置；
- 旧 `fluxmoe-fixed` 配置仍可解析，旧报告不被改写；
- Ruff、mypy 和全部 CPU 单元测试。

### 12.2 H100 服务器

- 大型按层 BF16 张量的 CUDA bit-exact round-trip；
- 逐专家与按层解压结果一致性；
- 按层解压 microbenchmark，记录有效 GB/s 和 launch 数；
- VMM 两层窗口 pointer、map/unmap、RAW/WAR 压力测试；
- compute-sanitizer memcheck/racecheck；
- 完整 Qwen3-Next smoke；
- 固定长度 resident 与 `fluxmoe-gpu-compressed` 三个关键点；
- 同一固定长度协议下的 64 MiB host-heavy 控制组，以及全 GPU 逐专家诊断组。

Mac 测试只能证明配置和控制逻辑，不作为吞吐提升证据。

## 13. 运行产物与 Git 边界

服务器结果写入：

`/home/jovyan/wangtonghan/moe-flex/runs/<timestamp>-<git-sha>/`

每次运行保存 environment、config、state、metrics、events、rank-local counters、stdout、
stderr 和 summary。大型原始产物留在服务器项目目录；Git 只提交代码、配置、校验和、
小型汇总和最终报告。最终报告必须引用精确 git SHA 和原始运行目录。

## 14. 分阶段停止条件

执行顺序为：

1. 全 GPU 压缩配置诊断；
2. 按层打包与批量解压；
3. CUDA microbenchmark 和正确性 smoke；
4. 三个关键吞吐点；
5. 根据 telemetry 决定下一步。

如果第 4 步未达到 11.1，停止扩展矩阵，优先检查 decode bandwidth、launch overhead、
load/compute overlap 和 scheduler 排队。只有本阶段达到 11.2，才设计 host/GPU 四流
并行和 dynamic planner。
