# TP4 低并发卸载、KV 扩容与受约束 Oracle 验证设计

**状态：** 用户已批准
**基线提交：** `55de519d8468a9ba11817b5f4acb076d079be2ce`  
**执行环境：** 服务器 `/home/jovyan/wangtonghan/moe-flex`，分支 `repro/fluxmoe`，4 × H100，TP4  
**唯一模型：** `/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct`  
**唯一数据：** 仓库现有 decode-mechanism 数据集

## 1. 目标与证据上限

本轮回答三个因果问题：

1. 在没有专家缺失和权重 payload 搬运时，卸载管理路径本身还剩多少固定开销？
2. 部分专家卸载是否能在相同物理显存与安全余量下提供 resident 无法获得的稳定 KV 容量；该增量能否改善最大并发 32 的目标服务负载？
3. 对实际 batch 16/32 的真实专家访问，加载流量有多少可由缓存替换避免，有多少只能由有限提前窗口的真实 H2D 预取隐藏？

本轮不得把以下证据混用：

- instrumented repetition 不作为正式吞吐；
- cache replay 只说明可避免的 miss/bytes，不说明真实预取吞吐；
- 各 TP rank 的时间不相加为全局时间；
- rank 0 的逻辑路由集合不乘以四；
- 缺失、失败和超时不记为 0；
- K1 未严格超过 native 与 eager 两种 resident 上限时，不宣称卸载具有独有 KV 容量。

## 2. 固定范围

只允许：

- TP4；
- batch 1 的零 miss 对照；
- actual batch 16 和 32 的 trace、replay 与 Oracle；
- 最大并发 32 的服务负载；
- 当前缓存策略、LRU、同容量未来知情缓存；
- Router 后按需、提前 1 层、提前 2 层；
- 当前模型、当前数据集、BF16、原始 Router 选择。

明确禁止：

- EP、EP+DP；
- 新的高 batch 点或完整 batch 网格；
- 新缓存策略；
- horizon 0/1/2 之外的预取窗口；
- 真实预测器训练、recall 或推理开销；
- 新模型、新数据集；
- 重跑已有高 batch 覆盖饱和点；
- 为制造 resident 失败而选择脱离目标业务的负载。

## 3. 当前正式基线

现有正式结果是全批墙钟吞吐，包含 prefill：

| 配置 | 实际 KV/GPU | 吞吐中位数 | 平均 actual decode batch |
|---|---:|---:|---:|
| Native，原生优化、全驻留 | 30.56 GiB | 7,946 tokens/s | 约 884 |
| A，eager、全驻留 | 15.28 GiB | 4,744 tokens/s | 约 491 |
| B，eager、部分卸载 | 15.28 GiB | 362 tokens/s | 约 491 |
| C，eager、部分卸载 | 30.56 GiB | 543 tokens/s | 约 884 |

C 相对 B 提升 50.12%，抢占从每轮 129–130 次降至 0，但 C 只有 Native 的 6.83%。目标场景仍是 actual decode batch 不超过 32；高 batch 失败不能代替低 batch 验证。

旧报告的 30.56 GiB 对应实际分配 `32,815,054,848` bytes/GPU。新实验同时使用动态测得的容量上限，旧值只用于明确的历史同 KV 分解点，不代替 K0/K1 探测。

## 4. 架构

一键控制器按以下阶段串行执行，任何必要门槛失败都停止后续因果结论，但保留已完成结果：

1. 环境、代码、模型、数据和四卡预检；
2. 1K/4K 校准；
3. batch 1 零 miss 无采样/有采样对照；
4. 历史 30.56 GiB 的 Native/eager resident/offload 低 batch 分解；
5. native resident、eager resident、offload 的稳定 auto-KV 探测；
6. K0/K1 最大并发 32 服务比较；
7. batch 16/32 的完整策略 trace；
8. 同 trace、同预算 cache replay；
9. Router 后、提前 1 层、提前 2 层的真实 Oracle；
10. 强制漏掉一个真实专家的补载正确性检查；
11. 生成一份不删减失败和 unavailable 指标的 Markdown/JSON 报告。

所有正式对照均校验：Git SHA、source hashes、模型身份、profile 身份、数据选择、硬件 UUID、显存预算、执行策略、KV block 几何、请求/输出长度、seed、smoke 输出哈希和重复数。

## 5. 实验一：零 miss 固定开销与执行模式分解

### 5.1 点位

在相同请求、相同 KV、相同 eager 模式下运行：

1. eager 全驻留、无采样；
2. 真实专家全部就绪但经过完整卸载管理路径、无采样；
3. 同一卸载管理路径、有采样。

“专家全部就绪”由四个 rank、三次 repetition 同时满足以下条件证明：

- `cache_misses == 0`；
- `h2d_bytes == 0`；
- `copy_launches == 0`。

不得伪造命中、跳过专家、删除映射、元数据或执行依赖。

另外，在目标低 batch 负载补齐实际 KV `32,815,054,848` bytes/GPU 的：

- Native；
- eager 全驻留；
- eager 卸载。

由 Native → eager resident 估计执行模式差异，由 eager resident → eager offload 估计卸载管理、准备、搬运和同步的总成本。若固定 KV 对任一模式违反实测容量或安全余量，该点标记不可运行，不降低安全门槛。

### 5.2 phase 和时间边界

每个 worker 记录单调时钟上的以下边界：

- measurement start；
- first prefill model step start/end；
- first decode model step start；
- final decode model step end；
- synchronized measurement end。

据此返回明确 scope 的：

- `prefill_wall_time`：measurement start 至 first decode start；
- `decode_wall_time`：first decode start 至 final decode end；
- `decode_step_ms_p50/p95`。

只有四 rank phase 序列完整且边界一致时才报告聚合墙钟；否则逐 rank 返回并将聚合标为 unavailable。

`host_gather` 的代码边界固定记录为：

- 起点：`wait_host()` 返回以后、开始 miss 权重 `index_select` 或构建 host expert map 之前；
- 终点：miss gather（若有）与全专家 host map 填充完成以后、H2D enqueue/event mark 之前。

报告必须说明其实际包含：条件性权重 gather，以及无条件 `host_map` 构建；不包含 H2D enqueue、专家计算和 promotion。额外拆分 `weight_gather_cpu_s` 与 `host_map_cpu_s`，避免继续用一个区间解释两种成本。

### 5.3 判定

若无采样、四 rank 零 miss 的卸载路径相对同 KV eager resident 仍明显变慢，报告固定运行时开销，并将主结论写为：

> 应优先修复卸载运行时，暂不进入真实预测器训练。

本轮仍可完成受约束 Oracle 上界，用于判断剩余等待是否可隐藏，但不得把它解释为预测器已经可用。

## 6. 实验二：真实 KV 增量与最大并发 32

### 6.1 稳定容量定义

分别探测：

- `K_native_max`；
- `K_eager_resident_max`；
- `K_offload_max`。

服务基线容量定义为：

`K0 = K_native_max`。

全驻留容量天花板另记为：

`K_resident_ceiling = max(K_native_max, K_eager_resident_max)`。

offload 上限定义为：

`K1 = K_offload_max`。

只有 `K1 - K_resident_ceiling` 至少为一个实际 KV block，且三次正式 workload 都保持物理安全余量时，独有容量门槛才通过。容量计算包括专家 cache、ingress、执行 workspace、trace/Oracle 固定缓冲和其他可见运行时分配；不能只通过初始化。

同 eager 机制代价比较使用双方均可稳定运行的 `K_pair = min(K_eager_resident_max, K_offload_max)`，避免将一方不可行的 KV 强加给另一方。

### 6.2 负载

使用当前数据集中业务合理的 4K 上下文，请求总量固定为 64，`max_num_seqs=32`，输出长度固定。只做：

- native resident，KV=`K0`；
- 若 offload 能稳定运行 `K0`，eager offload，KV=`K0`；
- 若独有容量门槛通过，同一 eager offload，KV=`K1`；
- eager resident，KV=`K_pair`；
- eager offload，KV=`K_pair`；

前三点构成服务容量收益比较；后两点构成同 eager 模式的机制代价比较。另使用请求数 32、目标 actual decode batch 32 的点验证固定 batch 机制差异；若实际 batch 不稳定，拒绝固定 batch 结论但保留服务结果。

机制比较只接受观测到的 decode batch 分布与目标固定 batch 一致的 repetition；否则拒绝“固定 actual batch”结论，但保留服务负载结果。服务收益比较固定请求和最大并发，允许 scheduler 自然改变 actual batch。

### 6.3 调度与请求指标

记录：

- 实际 KV bytes 和 blocks；
- KV usage peak；
- admission blocked count/duration；
- preemptions；
- swap outs；
- recomputed tokens；
- actual decode batch 直方图；
- output tokens/s；
- TTFT；
- TPOT 或 ITL；
- 请求 latency、TTFT、TPOT/ITL 的 p50/p95/p99。

只在 vLLM 暴露了可验证的 KV 阻塞原因时标记 `kv_admission_blocked`。如果只能看到 waiting queue，不将其自动归因于 KV。禁用 swap 是配置事实；运行时 swap 计数不可见时，报告 `policy-disabled`，不伪造测量值 0。recomputed tokens、ITL 等不可获得时原样报告 unavailable。

若 native 在业务上下文下持续推进 32 请求，且没有 KV 准入阻塞、抢占或重算，结论必须写为：

> 当前目标负载没有触发 KV 扩容收益机制。

## 7. 实验三 A：完整 trace 与同预算 cache replay

### 7.1 采集选择

优先复用现有 batch 16 的逐步明细。只有缺少完整字段或不能复现当前策略时，才重新采 batch 16。batch 32 同理。不得采其他 batch。

逻辑路由集合以 rank 0 为准；其余 rank 只保留物理搬运、等待和就绪时间。每个 rank 必须验证实际 route IDs 与 rank 0 逻辑集合一致；不一致时拒绝全局逻辑 trace。

### 7.2 每个 step/layer 的字段

每行包含：

- `step_id`；
- `layer_id`；
- `actual_batch`；
- `actual_expert_ids`；
- `resident_hit_ids`；
- `cache_hit_ids`；
- `miss_ids`；
- `bypass_ids`；
- `admitted_ids`；
- `evicted_ids`；
- `cache_state_before`；
- `cache_state_after`；
- `loaded_bytes`。

cache state 至少保存固定驻留身份、每个 persistent slot 的 `(layer, expert)`、free slots、策略分数/时钟以及容量；不得包含权重 payload。

### 7.3 replay

在完全相同的 trace、固定驻留集合、persistent cache 容量、ingress 容量和同层 protected working set 下比较：

1. 当前策略；
2. LRU；
3. 未来知情缓存。

未来知情缓存只改变 victim/admission 决策，不增加 cache/ingress/KV，不重排访问，不违反同层 working set。必须先逐行重现当前策略的 miss、bypass、admission、eviction 和 loaded bytes；任一不符即停止解释其他策略。

每种策略返回：

- 总 miss、loaded bytes、bypass、admission、eviction；
- 重用距离直方图与分位数；
- 各层 miss 分布；
- 各层 loaded bytes 分布。

## 8. 实验三 B：真实有限提前窗口 Oracle

### 8.1 Oracle 输入

先用独立 instrumented run 记录同一确定性工作负载的 rank 0 逻辑专家集合。正式 Oracle run 在每个 worker 装载相同 trace；实际使用时再次比较当前 Router 的真实 IDs：

- 完全一致：允许使用预取结果；
- 发现未预测专家：走原有按需补载；
- 预测了未使用专家：计入无效/过早预取 bytes；
- route 序列或 step/layer 无法对齐：拒绝该 repetition 的 Oracle 性能结论。

### 8.2 仅三个点

1. horizon 0：Router 后才知道专家，原按需路径；
2. horizon 1：在 layer L 的 Router 后，提供同 step 的 layer L+1 正确集合；
3. horizon 2：在 layer L 的 Router 后，提供同 step 的 layer L+2 正确集合。

不增加其他 horizon。首个 1/2 层没有足够 lookahead 时继续按需加载并单独标记 warmup，不伪造提前量。

### 8.3 物理实现

Oracle 复用现有 ingress 容量，不新增 GPU cache、ingress 或 KV：

- 未来专家先放入 ingress 中的空闲槽；
- 不提前调用 cache policy 的 `observe/lookup/admit`，以免改变驱逐顺序；
- 专家真正被使用时才按原策略执行 lookup、admission、eviction 与 promotion；
- ingress 容量不足时不覆盖仍在使用或 in-flight 的槽，剩余专家回退为按需加载；
- 使用独立 transfer stream 执行真实 H2D；
- pinned host gather buffer 只有在上一次 transfer 完成后才能复用；
- compute stream 在真正需要但尚未 ready 的专家上等待对应 event；
- 映射、promotion 和原始专家 compute 完整执行。

完整路径是：

`调度 → host gather → transfer queue → H2D → 必要映射/物化 → ready wait → 原始专家执行`。

不得删除或模拟其中任一步。

### 8.4 指标

每 rank 记录并在合法范围内聚合：

- `lookahead_ms`；
- mandatory loaded bytes；
- `ready_before_use_ratio`；
- `layer_batch_all_ready_ratio`；
- `exposed_wait_ms_p50/p95/p99`；
- unused/too-early bytes；
- transfer queue wait、H2D event 时间和 ready wait；
- output tokens/s、TTFT、TPOT/ITL 及尾延迟。

局部 CPU/GPU 区间不相加为全局净损失；端到端吞吐只来自无采样正式 repetition。

### 8.5 强制漏预测

使用单独正确性 point，在一个确定的 `(step, layer)` 从 Oracle 集合删除一个实际所需专家。验证：

- `forced_omission_count == 1`；
- `fallback_on_demand_count == 1`；
- 实际 Router ID 未被替换或跳过；
- 四 rank 都完成补载；
- smoke/output token hash 与 horizon 0 对照一致。

该 point 只用于正确性，不纳入吞吐比较。

## 9. 导出与报告

最终 Markdown 和 JSON 直接嵌入以下八部分，不以外链代替核心数字：

1. 无采样零 miss 路径固定开销；
2. Native、同 KV eager resident、同 KV offload 的分解；
3. K0/K1 实际稳定容量及安全余量；
4. 最大并发 32 下新增 KV 的准入、batch、吞吐和延迟影响；
5. 当前策略、LRU、未来知情缓存 replay；
6. horizon 0/1/2 真实 Oracle；
7. 强制漏预测正确性；
8. 所有失败、超时、拒绝点和 unavailable 指标。

当 `summary unavailable` 与原始配对 CSV 的 `measured` 冲突时，只修复导出逻辑并保留原始 CSV，不重跑有效测量。

## 10. 验证和发布

Mac 侧必须完成：

- 新行为的失败测试先行；
- CPU 单元测试覆盖完整 trace、replay、容量门槛、route 对齐、ingress 容量、event/回退状态机和报告拒绝路径；
- Bash 参数、恢复和不可覆盖语义；
- ruff、mypy、完整单元测试；
- 只读代码审查；
- 仅提交本轮代码、测试和文档，不加入用户已有的未跟踪报告。

推送到 `repro/fluxmoe` 后，服务器只需快进拉取并运行更新后的一键脚本。服务器结果随后通过 GitHub 回传；Mac 端不依赖直接服务器访问。
