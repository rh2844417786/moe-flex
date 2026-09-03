# FluxMoE 吞吐优先优化设计

> 初版：2026-09-02；证据审计修订：2026-09-03。

## 1. 背景与问题

当前 H100 关键矩阵已经证明 PagedTensor、GPU Huffman 解压、pinned-host H2D
和两层滑动窗口可以运行，但 `fluxmoe-fixed` 在三个关键点上分别比 resident 慢
85.8、61.5 和 40.9 倍。本设计只处理导致该吞吐差距的主数据路径，不在同一阶段
引入 dynamic residency planner。

现有实现有三个直接瓶颈：

1. `gpu_compressed_budget_bytes` 只有 64 MiB，绝大多数专家权重来自 pinned host，
   初始化、warmup 和测量期间累计产生 120.86–179.20 TB 的软件逻辑 H2D；
2. 存储对象和物化调用按单个专家权重张量组织。每层每个 tensor kind 都要经过
   数百次 Python 调度和 CUDA kernel 或 H2D copy；
3. 当前 receipt 的 `elapsed_s` 只测量异步操作的 CPU 提交时间，无法判断真实 decode
   bandwidth、load/compute overlap 或 GPU stall。

论文在 Qwen3-Next-80B-A3B-Instruct 的 performance-bound 配置中使用压缩 GPU
后端保存全部 routed-expert 权重。本阶段首先实现这一可独立归因的路径，以确认消除
运行时 PCIe 和物化调用碎片后，端到端吞吐是否真实提高。

## 2. 目标与优先级

本阶段优先级从高到低为：

1. 在相同模型、数据、TP、batch、context 和实际生成 token 数下，提高
   `fluxmoe` 的 output tokens/s；
2. 保持 BF16 bit-exact、单请求 greedy token parity 和 router expert-set probe；
3. 保留比同协议 resident 更多的 KV cache，而不追求当前 41.18 GiB/卡；
4. 保存足够的原始证据，使吞吐变化可归因到数据路径，而不是输出长度、排队、
   初始化上传或测量噪声。

本阶段不把“代码看起来更快”“microbenchmark 更快”或“单次运行更快”作为成功。
正式结论来自同一协议下三次 warmup 后的三次 measured repetition；当变化接近阈值
或波动过大时，再用新的 engine 进程复核。

## 3. 范围

### 3.1 本阶段包含

- 新增全压缩 GPU 的明确实验变体；
- 保留逐专家 Huffman bitstream 和 codebook，构建按层 batched descriptor；
- 每层 `w13` 和 `w2` 各使用一次 batched decode launch；
- 保留并核验两层 VMM 滑动窗口和下一层异步预取；
- 在 GPU 分配前生成实际 encoded-size manifest 并执行显存准入；
- 将启动上传与运行时专家 H2D 分开计数；
- 使用 CUDA event 记录 decode、compute 和 stall，而不在热路径全局同步；
- 使用 storage-mode-specific 正确性和机制门禁；
- 固定实际生成 token 数；
- 在一次 engine 生命周期中依次运行三个关键点，减少重复编码和初始化；
- 重新运行 smoke、decode microbenchmark 和三个关键吞吐点。

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
Python/CUDA 调用和大量独立 kernel launch。它只作为诊断控制，不作为最终优化。

### 4.2 未选择：将整个 layer-kind 交给现有 Python encoder

Qwen3-Next 在 TP=4 时，每 rank 的 routed-expert BF16 源权重约 36 GiB；每层
`w13` 约 0.5 GiB，`w2` 约 0.25 GiB。现有 reference encoder 会为每个元素构造
Python 数据并逐 bit 编码。直接把 0.5 GiB 张量交给它会形成巨型 Python 列表和长时间
单线程编码，并且重新生成 layer-level codebook 会改变压缩率，无法纯粹归因 launch
合并的收益。因此不采用“重新整层编码”。

### 4.3 采用：逐专家编码 + 按层 descriptor + batched decode

保留现有每个专家独立产生的 Huffman bitstream 和 canonical codebook。初始化阶段按层、
按 tensor kind 汇总这些编码对象的数据与描述符，但不重新编码：

- 拼接 sign/mantissa payload；
- 拼接 exponent payload；
- 保留每个 expert 的 trie/codebook；
- 将 chunk 映射到 expert、源 byte offset 和目标 element offset；
- 由一个 CUDA grid 覆盖该 layer-kind 的全部 chunks。

每个 chunk block 根据 descriptor 选取原专家的 codebook，并写入整层 Parameter 中该
专家对应的连续目标区间。这样保持原压缩表示和数学结果，同时把每个 layer-kind 的
数百次 Python/CUDA 调用合并为一次 batched launch。

### 4.4 延后：一次性完成混合后端和 dynamic planner

混合后端需要每个 tensor kind 至少两个 backend stream、跨流 join event、按层
带宽比例切片以及在线显存安全控制。若与 batched decode 同时引入，性能变化无法区分
来自 GPU residency、launch 合并、流并行还是 planner。本阶段不采用。

## 5. 运行变体与配置语义

新增正式变体 `fluxmoe-gpu-compressed`，保留历史 `fluxmoe-fixed` 配置和报告，不覆盖
既有 64 MiB 结果。

新增显式配置字段：

- `storage_mode: hybrid | gpu-compressed`；
- `gpu_materialization_mode: expertwise | batched`；
- `minimum_kv_gain_bytes: 1073741824`；
- `gpu_safety_margin_bytes: 536870912`。

正式方案使用：

```yaml
variant: fluxmoe-gpu-compressed
storage_mode: gpu-compressed
gpu_materialization_mode: batched
```

`fluxmoe-gpu-compressed` 必须满足：

- routed-expert backing store 全部是 GPU compressed；
- 现有 `gpu_compressed_budget_bytes` 只用于 hybrid placement；
- GPU-compressed mode 忽略 hybrid source budget，并强制接纳全部 routed experts；
- 实际 GPU 占用按 payload、descriptor、chunk metadata、codebook/trie 和 error buffer
  的真实字节总和记录；
- 不创建 pinned-host expert store；
- 分配或初始化失败时直接退出，不回退到 host。

`expertwise` 只用于同一编码表示下的 launch-overhead microbenchmark。正式端到端运行
固定使用 `batched`，避免把诊断模式误写为最终结果。

## 6. 编码、descriptor 与 GPU 存储

### 6.1 逐专家编码

权重加载仍按 vLLM 提供的专家 shard 完成。每层必须严格收齐 expert ID
`0..num_experts-1` 的 `w13` 和 `w2`。沿用当前逐专家 `encode_bf16_bits()`，避免一次
创建整层 Python exponent 列表。

每个专家编码完成后立即加入对应 layer-kind builder。builder 只保留最终 bytes 和
紧凑 metadata，不保留临时 Python symbol 列表。缺少、重复、乱序或形状不一致必须在
GPU 分配前失败。

### 6.2 PackedLayerDescriptor

每个 layer-kind 形成一个不可变 descriptor，至少包含：

- `expert_count` 和完整 destination shape；
- 拼接后的 sign/mantissa 与 exponent payload；
- 每个 expert 的 destination element base；
- 每个 expert 的 chunk begin/count；
- 每个 chunk 的 payload byte offset 和 bit length；
- 每个 chunk 对应的 expert/codebook index；
- 每个 expert 的 trie begin/node count；
- 解码 error slot。

所有 offsets 在 CPU 侧使用 checked 64-bit arithmetic。任何范围越界、重叠、目标空洞、
chunk 数不一致或 codebook 越界都必须在上传前失败。

### 6.3 Batched GPU store

GPU store 为每个 layer-kind 保存 descriptor 对应的连续 CUDA tensors，并提供两种方法：

```text
materialize_batched(layer_kind, destination, stream)
materialize_expertwise_for_benchmark(layer_kind, destination, stream)
```

两者读取完全相同的 payload、codebook 和目标布局。后者只用于同进程 microbenchmark，
从而把 launch 数变化与压缩率变化分开。

正式物化时：

```text
w13 worker: batched_decode(layer.i.w13 -> paged_w13[i], w13_load_stream)
w2 worker:  batched_decode(layer.i.w2  -> paged_w2[i],  w2_load_stream)
```

每个 tensor kind 每层每次物化恰好产生一个 decode launch。

## 7. 显存准入与启动流程

### 7.1 两阶段初始化

初始化分为：

1. CPU encode/manifest 阶段：完成所有逐专家编码和 layer descriptor，计算真实存储量；
2. GPU admission/upload 阶段：通过容量门禁后才创建并上传完整 GPU compressed store。

CPU manifest 至少记录：

- `expert_source_bytes`；
- `encoded_payload_bytes`；
- `descriptor_bytes`；
- `metadata_bytes`；
- `codebook_bytes`；
- `error_buffer_bytes`；
- `gpu_compressed_source_bytes`；
- `gpu_compressed_storage_bytes`；
- `compression_ratio`。

### 7.2 容量门禁

resident 必须先完成同 commit memory profile。GPU 上传前同时检查物理空闲显存和
`gpu_memory_utilization` 预算。以检查时已经存在的非专家权重与两层 VMM physical
window 为基准，不得重复扣减已分配对象：

```text
minimum_kv_cache_bytes = resident_kv_cache_bytes + minimum_kv_gain_bytes

physical gate:
  gpu_compressed_storage_bytes
  + minimum_kv_cache_bytes
  + resident_peak_activation_bytes
  + gpu_safety_margin_bytes
  <= current_free_gpu_bytes

configured-budget gate:
  current_process_non_kv_bytes
  + gpu_compressed_storage_bytes
  + resident_peak_activation_bytes
  + minimum_kv_cache_bytes
  + gpu_safety_margin_bytes
  <= total_gpu_bytes * gpu_memory_utilization
```

`current_process_non_kv_bytes` 由 worker 启动时的 free-memory snapshot 与 admission 时的
snapshot 推导；GPU 必须独占，期间外部占用变化直接使 admission 失败。resident profile
缺少 KV 或 peak activation 字段时不得猜测默认值。engine 初始化完成后还必须验证
vLLM 实际 KV cache bytes 不低于 `minimum_kv_cache_bytes`，否则本次运行标记为无效并
停止性能测试。

Qwen3-Next 的静态几何只用于预估：每 rank 专家源权重约 36 GiB，两层 dense window
约 1.5 GiB。最终准入只使用当前 checkpoint 实际编码与设备测量值，不能用论文约 20%
压缩率替代。

### 7.3 启动上传与运行时 H2D

压缩 payload 和 descriptor 在启动时必然从 CPU 上传 GPU，必须单独记录：

- `startup_gpu_store_upload_bytes`；
- `startup_gpu_store_upload_s`。

CPU 编码耗时另记为 `startup_cpu_encode_s`，不得计入端到端 generation throughput，
但必须报告，防止优化后形成不可接受的初始化成本。

运行时从 pinned host 物化 BF16 expert 的流量记录为：

- `runtime_host_expert_h2d_bytes`；
- `runtime_host_copy_launches`。

全压缩 GPU 模式要求运行时两项都为零，而不是宣称整个生命周期没有 H2D。

## 8. VMM、加载流水线与 CUDA 计时

### 8.1 生命周期

保留每个 tensor kind 两个物理 layer block：当前层和下一层。生命周期仍为：

```text
map next layer -> async batched decode -> record load-done event
                                           |
current compute -> record compute-done event -> safe unmap/recycle
```

必须维持：

- 每个逻辑 Parameter 的 data pointer 在进程生命周期内不变；
- layer `i` 计算前等待其 load-done event；
- layer `i-1` 的物理块只能在 compute-done event 完成后复用；
- `w13` 与 `w2` 都 ready 后才能进入 fused MoE；
- worker 异常必须传回推理线程并终止运行；
- close 必须排空任务、同步 load stream 并释放 compressed store。

### 8.2 CUDA event telemetry

不能使用包围异步 API 的 host `perf_counter()` 充当 GPU duration。每个 materialization
在其 load stream 上记录 `load_start` 和 `load_end` event。

为测量 compute stream 的真实等待：

1. 在 compute stream 等待 load-done 前记录 `compute_arrival`；
2. 插入对 load-done 的 stream wait；
3. wait 后记录 `compute_resume`；
4. 在 fused MoE 前后记录 `compute_start` 和 `compute_end`。

event 只在结果已经自然完成、物理 slot 即将复用或运行结束时读取 elapsed time，热路径
不得调用 `torch.cuda.synchronize()` 或 device-wide synchronize。另行记录 CPU condition
等待，不能与 GPU stream stall 混为一个指标。

每层、每 rank 至少得到：

- `decode_cuda_s`；
- `compute_cuda_s`；
- `compute_wait_for_load_cuda_s`；
- `scheduler_cpu_wait_s`；
- `overlap_ratio`；
- `stall_ratio`。

定义：

```text
stall_ratio = compute_wait_for_load_cuda_s
              / (compute_wait_for_load_cuda_s + compute_cuda_s)
overlap_ratio = 1 - min(1, compute_wait_for_load_cuda_s / decode_cuda_s)
```

当 `decode_cuda_s == 0` 时 overlap ratio 不定义，必须写为 `null`，不能写成 100%。

## 9. Storage-mode-specific 证据门禁

runner、state writer 和 report classifier 必须共享一份显式 evidence profile，不能各自
硬编码不同规则。

### 9.1 Resident

- mapping、runtime expert H2D 和 decode 均为零；
- 固定工作量和基础正确性成立；
- 状态为 `BASELINE_COMPLETE`。

### 9.2 Hybrid 64 MiB 控制组

- mapping、运行时 expert H2D、decode 和物理块复用均非零；
- correctness evidence 来自同 commit、同 storage mode 的 smoke；
- 不允许借用旧 commit 的 smoke。

### 9.3 GPU compressed 正式组

- mapping、decode 和物理块复用非零；
- `runtime_host_expert_h2d_bytes == 0`；
- `runtime_host_copy_launches == 0`；
- `gpu_compressed_source_bytes == expert_source_bytes`；
- 每 rank 的 batched decode launches 等于实际 materialized layer-kind 次数；
- correctness evidence 来自同 commit 的 `fluxmoe-gpu-compressed` smoke。

新增 variant 时必须同时更新配置 Literal、runner 支持集合、环境变量映射、smoke pointer、
key-matrix 脚本、state transition、report validator 和单元测试。

## 10. 固定工作量与运行组织

### 10.1 采样协议

所有性能组使用：

```text
temperature = 0
ignore_eos = true
min_tokens = 128
max_tokens = 128
```

每次 repetition 必须记录请求数、每请求生成长度和总生成 token 数。任何请求没有生成
128 tokens 时，该 repetition 无效。

### 10.2 对照组

旧报告 commit `de821d7fa2d17c1314d7d1530e95b5e293459f22` 只保留为历史方向参考。
同一优化 commit 必须运行：

1. `resident`：固定长度当前基线；
2. `fluxmoe-fixed`：64 MiB host-heavy 控制组；
3. `fluxmoe-gpu-compressed`：全 GPU、batched decode 正式组。

expertwise 与 batched decode 的 launch 差异在同一 compressed store 的 CUDA
microbenchmark 中比较，不再为 expertwise 单独重复完整模型初始化。

### 10.3 一次 engine 覆盖 smoke 和三个点

每个正式 variant 只初始化一次 engine。该 engine 先执行自身 storage mode 的确定性
smoke；smoke 失败立即退出，成功后关闭 router trace 和逐层 bit-hash 开销，再依次运行：

1. batch 32、context 1024；
2. batch 128、context 4096；
3. batch 256、context 4096。

每点执行 3 次 warmup 和 3 次 measured repetition。每点分别在 warmup 前、warmup 后和
measurement 后记录机制计数 snapshot；吞吐报告使用 measurement 区间，机制报告同时
列出 warmup 与 measurement delta。进入下一点前，必须确认上一个点的请求全部完成、
KV blocks 已释放，不能把前一工作负载累计量归给下一点。

这三次 measured repetition 位于同一 engine，不称为三个独立进程。若任一点变异系数
超过 5%，或吞吐变化距离验收阈值不足 10%，再新建 engine 进程复核该点。

resident 必须最先运行以生成同 commit reference。host-heavy 与 GPU-compressed 的首次
顺序预先固定并记录；若触发新 engine 复核，则反转二者顺序，以检查时钟或热状态偏差。

## 11. Telemetry 与原始证据

每个 TP rank 至少记录：

- 第 7 节全部 source、encoded、metadata、upload 和 runtime H2D 字段；
- `gpu_decode_output_bytes` 和 `gpu_decode_input_bytes`；
- `gpu_decode_launches`；
- 第 8 节全部 CUDA/CPU timing；
- VMM mapping count、mapped bytes 和 recycle count；
- vLLM model/KV cache memory；
- peak active requests、peak batched tokens 和 KV occupancy；
- 请求级 TTFT、queue time、端到端 latency 的 P50/P95/P99；
- 每次 repetition 的 elapsed time、generated tokens 和 output tokens/s。

聚合结果必须同时保存 rank-local 数值，避免总和掩盖单 rank 慢尾。所有软件计数与
Nsight/CUPTI 硬件流量明确分栏；没有硬件 trace 时不得把软件逻辑字节称为 PCIe 实测量。

## 12. 正确性与失败处理

每个非 resident storage mode 都运行自己的同 commit smoke。Smoke 使用单请求、
context 1024、固定短输出，并至少覆盖 prefill 和多个 decode step。必须通过：

- 所有层 `w13`、`w2` 恢复后的 BF16 bits 与 checkpoint 一致；
- expertwise 与 batched decode 的输出 bits 一致；
- resident 与正式变体的 greedy output token IDs 一致；
- resident 与正式变体的确定性 router expert-set probe 一致；
- VMM pointer stability；
- CUDA memcheck 和 racecheck；
- 对应 storage mode 的机制门禁。

以下情况写入结构化错误并停止：

- encoded-size manifest 或 GPU capacity admission 失败；
- GPU compressed 模式出现任何运行时 host expert placement；
- descriptor offset、shape、expert 数量或 chunk/codebook 映射错误；
- 解压错误、权重 hash 不一致或 CUDA lifecycle 错误；
- 固定输出长度未满足；
- 服务器 GPU 非独占；
- 原始运行产物缺失。

不允许失败后静默运行 resident，也不允许跨 commit 或跨 storage mode 借用 smoke。

## 13. 吞吐验收与状态

项目的 reproduction status 仍只使用：

- `SUPPORTED`；
- `MIXED`；
- `NOT_SUPPORTED`；
- `INCONCLUSIVE`。

另增不替代 reproduction status 的 `throughput_outcome`：

- `CONFIRMED_MEANINGFUL_IMPROVEMENT`；
- `CONFIRMED_BUT_INSUFFICIENT`；
- `NO_CONFIRMED_IMPROVEMENT`；
- `INVALID_EVIDENCE`。

### 13.1 确认提高

三个关键点上，正式组三次测量的最小值都必须高于同 commit、同固定长度协议下
host-heavy 控制组对应点的最大值。满足后才确认吞吐提高。

### 13.2 有意义提高

- 至少两个关键点的中位数达到 host-heavy 控制组的 2 倍；
- 第三个关键点的中位数至少达到控制组的 1.25 倍；
- 任一点不能因错误、OOM、请求减少、较短输出或更低实际并发获得表面加速。

只满足 13.1 时，`throughput_outcome=CONFIRMED_BUT_INSUFFICIENT`，reproduction
status 最多为 `MIXED`。没有满足 13.1 时，
`throughput_outcome=NO_CONFIRMED_IMPROVEMENT`，停止扩展矩阵并依据 timing evidence
定位下一瓶颈。达到 13.2 也只证明本阶段优化有效；只有 13.3 的容量压力证据齐全时，
才能据此提高论文 reproduction status。

### 13.3 论文趋势边界

高 batch/长 context 是否超过 resident，只有在 peak active requests、KV occupancy
和 scheduler admission 证明 resident 确实进入容量压力后才有论文级解释。若 vLLM
只是将请求排队，则本阶段只能声明“当前实现吞吐提高”，不能声明论文的 KV-capacity
吞吐趋势得到支持。

若 admission evidence 充分，还必须满足：

- KV cache 容量高于同协议 resident；
- 高 batch/长 context 的相对表现优于小 batch；
- TTFT、P99、错误、OOM 和排队没有抵消 output throughput 收益。

本设计不预设必须达到论文的 3.0 倍，也不允许用 microbenchmark 替代端到端结果。

## 14. 性能决策门

完整 key matrix 前先执行同 store microbenchmark 和一个代表性短运行：

1. 对相同 layer-kind payload 比较 expertwise 与 batched decode；
2. 测量 batched decode 的 input/output GB/s、launch 数和 CUDA duration；
3. 在 batch 128、context 4096、固定短输出上测量 load/compute/stall；
4. 计算每层维持无 stall 所需 bandwidth，而不是使用配置中的假定带宽。

若 `stall_ratio > 10%`，或实测 decode bandwidth
低于由前层 compute window 推导出的 required bandwidth，则不运行三个昂贵关键点，
优先优化 decoder。当前 kernel 每个 chunk 的 Huffman traversal 由单线程完成，因此
kernel 内部并行解码是明确候选，但必须由该决策门触发。

## 15. 测试策略

### 15.1 MacBook

- per-expert encoder 行为保持不变；
- PackedLayerDescriptor 的拼接顺序、64-bit offsets、shape 和 tensor ID；
- 多 codebook、多 chunk、最后一个短 chunk 和目标无空洞；
- descriptor 越界、重叠、缺失和重复 expert 的 fail-closed 行为；
- storage mode 与 materialization mode 配置解析；
- encoded-size manifest 和 capacity decision；
- startup/runtime H2D、launch 和 rank-local counter 聚合；
- variant-specific evidence profile 和状态映射；
- 固定输出长度及无效 repetition；
- multi-point counter snapshot/delta；
- Ruff、mypy 和全部 CPU 单元测试。

### 15.2 H100 服务器

- batched multi-codebook CUDA decode 的 bit-exact round-trip；
- expertwise 与 batched decode 对相同 payload 的逐位一致；
- 大型 layer-kind microbenchmark 和真实 CUDA event duration；
- VMM 两层窗口 pointer、map/unmap、RAW/WAR 压力测试；
- compute-sanitizer memcheck/racecheck；
- host-heavy 与 GPU-compressed 各自完整 Qwen3-Next smoke；
- 固定长度 resident、host-heavy 和 GPU-compressed 三个关键点；
- 必要时对临界结果执行新的 engine 进程复核。

Mac 测试只能证明数据结构和控制逻辑，不作为吞吐提高证据。

## 16. 运行产物与 Git 边界

服务器结果写入：

`/home/jovyan/wangtonghan/moe-flex/runs/<timestamp>-<git-sha>-<variant>/`

一次 multi-point engine 使用一个顶层运行目录，并按阶段隔离产物：

```text
runs/<timestamp>-<git-sha>-<variant>/
  environment.json
  config.json
  encoded-size-manifest.json
  smoke/
  points/b32-c1024/
  points/b128-c4096/
  points/b256-c4096/
  rank-counters/
  state.json
  summary.json
```

每点目录保存 warmup/measurement snapshots、metrics、events、stdout 和 stderr。大型原始
产物留在服务器项目目录；Git 只提交代码、配置、校验和、小型汇总和最终报告。最终
报告必须引用精确 git SHA 和原始运行目录。

## 17. 分阶段停止条件

执行顺序为：

1. 配置、evidence profile、固定工作量和 multi-point runner；
2. packed descriptor、batched CUDA decode 和存储计数；
3. encoded-size manifest、显存准入和 CUDA event telemetry；
4. CPU/CUDA 测试、sanitizer 和两个 storage-mode smoke；
5. 性能决策门；
6. 三个关键吞吐点；
7. 根据 evidence 决定是否进入 hybrid 四流和 dynamic planner。

任一步缺失原始证据都停止。第 5 步显示 decode/load 仍不能被 compute window 隐藏时，
先优化 kernel，不执行第 6 步。只有第 6 步达到 13.2，才设计 host/GPU 四流并行和
dynamic planner。
