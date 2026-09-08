# 四卡 H100 专家卸载吞吐可行性分析

用户已批准按聊天方案完成分析编码并推送 GitHub。本轮交付可执行的分析工具，不重写专家缓存、不自动执行真实卸载，不把 Mac 测试或理论预测写成 H100 加速结论。

## 目标与边界

判断已有 Qwen3-Next-80B-A3B-Instruct、原始 BF16、原路由、4×H100 80GB / TP4 上，合理优化后的 CPU 专家权重卸载是否有净吞吐空间。正式部署基线使用默认 utilization=0.90；0.60 与 eager 仅用于诊断。分别保存模型/数据/代码/引擎策略身份、实际 KV、显存及失败信息。全驻留的原生优化不能为方便卸载而被人为禁用。

所有分析输出固定 `diagnostic_only=true`、`formal_offload_gain=false`、`deployment_gain_proven=false`。native 基线吞吐可以是 measured，trace 不能作为吞吐样本，回放/成本曲线是 predicted。只有后续真实同资源卸载实验才可以证明部署收益。本轮工具给出候选与所缺证据，不自动得出“部署已获利”。

继续在 `/Users/a1234/code/flexmoe/.worktrees/repro-fluxmoe` 的 `repro/fluxmoe` 工作，服务器根目录固定 `/home/jovyan/wangtonghan/moe-flex`。只允许已有 GitHub 同步与固定 Docker Hub 基础镜像获取；无新依赖、模型或数据下载。服务器所有写入在项目内，`/mnt/public_data` 只读。四张独占 H100 不满足则停止，不杀其他人的进程。保留现有 partial、expert-cache、KV Oracle 路径及历史失败恢复；仅添加最小默认行为不变的接入点。

## 1. 轨迹与回放合同

新增标准库包 `flexmoe.analysis`，独立于 PyTorch。`schema.py` 定义 `TraceEvent`、`DemandTrace`、`ReplayConfig`、`ReplayResult`、`TransferSample`、`TimingPoint` 的唯一数据合同及 JSON 读写/验证。字段定义见实现计划 Task 1；其他模块必须消费这些类型，不重新发明等价结构。

一个 demand trace 对应一个 rank、一个明确采样窗口；专家键是 `(layer, expert)`。每层每步只保留去重专家集合、实际 token rows、可用请求数及 prefill/decode/mixed/unknown 分类，不包含 token IDs、prompts 或 logits。记录 observed/captured step 数、截断/缺层、输出总数与 full_workload 标识。只有完整、逐步逐层覆盖的轨迹才能按整批 generated_tokens 归一化。部分轨迹可给窗口缺失统计，不给整批预测。四 rank 在身份、几何与事件覆盖上必须校验。

固定驻留专家从独立校准轨迹的每步专家访问频率选择；除整批 input hash 不同，还必须通过逐请求 prompt hash 集合证明校准/测试无交集，缺该证据则拒绝自动校准配对。驻留比例按层取整，输出实际驻留数。全局缓冲支持确定性 LRU、衰减 LFU，以及知道未来访问的理想参考。理想参考不是可部署算法，也不声称是带预取/并行依赖系统的严格全局最优。

每个事件的 hit/miss 根据事件开始前的缓存集合判断；同批所需专家不可因遍历顺序被重复加载或错误计为 miss。本层专家加载一次服务该事件全部 token。缓存持续跨层、步、请求保留。默认 staging_experts=512，cache_slots 扫描 0/64/256/512，卸载比例扫描 0.05/0.10/0.20/0.30/0.50。净释放显存必须扣除固定驻留、全局缓存、加载缓冲、显式额外工作区/元数据；不把负值裁成虚假收益。若用户把 staging 缩到小于一次需求，记录需额外分块计算的未建模成本。

## 2. 成本与结论合同

以相同请求、输出 token、模型、TP、软件/代码身份及声明的资源配置配对全驻留 R 与更多 KV 的参考 K。允许明确记录的 native/eager 诊断差异，但绝不能混成已验证部署比较。KV 参考只用于其真实容量，超过净释放量则拒绝配对，不按容量比例直接缩放吞吐。无匹配 K 时只输出搬运服务需求、带宽阈值和缺证据项。

每 rank 按微基准覆盖的真实传输形状/模式估算总搬运服务时间，并显示四卡瓶颈而非 rank 平均值。允许将一次缺失数明确分解成已测形状（不超过 staging）的多个 copy，记录分解和发起次数；禁止静默外推未测批量或把四卡带宽简单乘四。测量、trace 与 baseline 保存相同四卡有序身份 hash 及软件版本；身份不可得时标缺证据。提供串行情景估计与乐观完全重叠服务参考两种明确假设的区间（不是实际性能严格上下界）；若有并行计算/通信代理测量，作为额外敏感性结果，不作为真实 MoE 重叠保证。

K 参考可以来自同四卡/模型/输入/代码上的更高 utilization 原生测量，因此 utilization/引擎模式差异不是物理资源相同的替代品，也不要求它们对 K 相等；差异必须标明是超基线预算的 counterfactual。实际 K 增量仍不可超过预测净释放量。若无法安全测量更大 K，不伪造参考点或吞吐外推。

对于同一 N，时间收益空间为 T_R-T_K。所有吞吐估计由 N/估计时间重算。带宽、秒数、bytes、整数、哈希、rank 覆盖、工作量与证据级别严格验证；NaN/Infinity、bool 冒充整数、负数、缺失/截断、错模型/批量/长度、错 rank、采样测速混用都 fail closed 或标 incomplete。没有固定 20% 成功门槛；不得把失败、OOM、没有匹配参考当成“没有收益”。

## 3. 原生测速与采集

复用 `run_benchmark` 的工作负载、固定输出、smoke、warmup、同步计时和失败持久化。新增 backend 专用策略解析 hook（默认保持旧 eager 严格规则）；analysis 的 `native` 模式不强制关闭 vLLM 默认编译/Graph，保存实际配置；`eager` 为归因模式。无法运行 native 时保留失败，不静默换 eager 并称最佳部署基线。

trace 单独运行，强制 eager 且不是吞吐证据。复用现有 vLLM 校准入口分派到新的 collector，不改变 pinned patch 或其校验和。root forward hook 标识 step 并读取可用 CPU runner 元数据；仅在 GPU 固定大小缓冲中记录每层专家集合，采样期间不复制路由 IDs 到 CPU、不做每层同步；stop 后统一 D2H、验证并落盘。无可靠 phase/request 数据则标 unknown/null。默认最多 1024 steps、128 MiB/rank 捕获预算；允许显式调整，超限记录截断，不静默假装全量。GPU buffer 计入采样显存安全预检。warmup 不入 trace，hooks 可重复安全启停并清理。

当前数据有 1K/2K/3K/4K 桶；默认 short=(1K,128)、generation=(1K,2048)、medium=(4K,512)。并发候选 16/32/64/128/256/512，以原生扫描发现拐点，选少量点采集。16K/32K 只通过显式 synthetic packing 生成已有 token 的容量压力样本，标 synthetic，不当真实长请求。重复输入数及校准/测试分离必须可追溯。

## 4. 四卡传输微基准

独立 CUDA/torch.distributed 程序，四 rank 同时执行。固定形状 deterministic BF16/byte payload 只用于 DMA/计算代理，不作为新模型权重。覆盖单次专家数 1/8/32/128/512 的 pinned contiguous、fragmented、CPU gather+bulk；支持 isolated 以及 GEMM+NCCL 代理并行场景，报告代理属性。

每次测量前 rank barrier/GPU 同步；分别记录 CPU gather、CUDA copy、计算、端到端 wall 和总字节。收尾同步后读取 CUDA events。按瓶颈 rank 时间计算 aggregate，计算/通信竞争不得重叠计数两次。记录可用 CPU affinity/NUMA/PCIe 拓扑为本地诊断，公开仅脱敏身份 hash/数值或 unavailable。资源上限、warmup/repetitions、明确超时、错误清理、原子失败状态和阶段码必须可执行；不依赖额外包、外网或新 CUDA 内核。

## 5. CLI、报告与服务器交接

提供 `python3 -S src/flexmoe/analysis/cli.py` 的标准库入口：validate、replay、analyze、plan、export；GPU 入口通过 `scripts/server/run_offload_analysis.sh` 的 resident/trace/transport 运行现有 pinned Docker，每点 timeout 在容器内。路径为 `runs/offload-analysis/<id>` 与 `docs/results/offload-analysis-<id>`，ID 全新，保留缺失/损坏 summary 的独立 smoke/失败记录。只对生成的该结果命名空间豁免同 SHA 序列的 dirty check，其他源码改动仍阻止执行。

plan 输出有限的分阶段命令和待测条件，不自动暴力启动完整笛卡尔积；所有参数/自定义路径必须真实传入。replay/analyze 可在 Mac 或服务器离线执行；服务器路径检查在执行/写入 GPU 工作流处强制，纯标准库读分析支持临时目录测试。

公开报告包含原生基线、轨迹覆盖与校准来源、每输出 token 的加载需求（仅全量轨迹）、per-rank 带宽/成本区间、净 KV、盈亏窗口、缺证据与候选；全部 bytes/GB 按十进制标注。只输出白名单字段，不上传原始日志、prompts、权重或大体积原始轨迹。对既有 KV Oracle 数据的转换必须先调用其校验，保留其 over-budget/diagnostic 边界。任何 summary/report 不能被旧正式 analyzer 清洗成正式卸载证据。

中文 runbook 和 server Codex prompt 指导：拉取干净分支并构建、四卡独占、正常预算原生扫描、少量独立校准/测试 trace、并行微基准、离线分析与负结果导出、回传 GitHub。只有真实同资源卸载验证才讨论是否值得深入；本轮不自动启动重写的卸载实现。
