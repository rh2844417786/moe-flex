# 热专家常驻与跨层共享缓存

用户已批准实现、GitHub push 和服务器实验交接。此文记录本轮实现合同，不重复要求批准。

## 目标与边界

在 Qwen3-Next-80B-A3B-Instruct BF16、4 张独占 H100、TP4、相同显存预算和相同请求工作量下，检验部分专家卸载换取 KV 后的净吞吐收益。保留原 Top-k、路由权重、全部专家计算，不剪枝、不量化、不用其他 GPU 隐藏权重。

首轮只证明运行机制和当前负载收益，不预设 50% 最优。Mac 只执行 CPU 测试；CUDA 正确性和端到端性能必须由服务器验证。服务器无正常互联网：仅使用已获用户允许的 GitHub 同步及固定 Docker Hub 基础镜像获取；优先复用缓存镜像与仓库 wheel，Docker 构建 RUN 步骤始终离线，不增加在线安装步骤。模型与公共数据只读，所有输出位于服务器项目 `/home/jovyan/wangtonghan/moe-flex` 内。

## 存储与计算

- 每层保留 `floor(num_experts * resident_ratio)` 个校准热专家。初始比例可配置。
- 每个 TP rank 一个连续 GPU 专家权重池，包含所有层常驻专家、跨层持久缓存、需求加载周转区。每条缓存 key 是 `(layer, expert)`，存的是本 rank 的 BF16 shard。
- 共享缓存用后保留，跨请求和 token 复用；周转区在总预算内预留，不能算作免费显存。
- 采用带衰减的访问频率、层距离同分决策和准入门槛；LRU 为对照。统计每层每次模型前向的唯一专家访问，而不是批内 token 原始次数。
- 主机保存完整 CPU BF16 shard 副本，加载时一次汇集所有缺失专家到连续 pinned ingress，再进行两次 H2D（w13、w2）；使用 GPU scatter 晋升到缓存，避免每专家两次小 H2D。CPU 副本、pinned ingress、GPU ingress、缓存、常驻和元数据都单独记账。
- GPU 周转区能容纳一层全部专家，因此任意路由覆盖都能完整计算，不依赖缓存至少容纳本批所有需求。不得通过丢弃专家或改变路由处理溢出。
- 复用 vLLM 0.10.2 原生 `fused_experts` 和 `expert_map`：逻辑专家 ID/Top-k 权重不变，映射到物理池 slot。`global_num_experts` 保持单层原始专家数，避免按整个池大小排序。不得先把整个专家层重新物化后计算。
- pinned ingress 在上一轮 H2D 完成前不得覆盖；GPU ingress/cache slot 在上一轮消费者结束前不得覆盖。正确的 stream/event 顺序优先于未经实测的重叠优化。
- eager、编译 level=0、BF16、TP-only；不支持的量化、EP、EPLB、bias、PP、DP、ROCm 路径 fail closed。旧 partial-host 路径不删除。

## 校准与在线调整

- 原生 resident engine 收集真实路由，校准数据与测量数据按 token 序列哈希严格不相交。不要把 engine profiling/dummy run 计入校准。
- 校准产物记录 schema、模型 config/权重 index 身份、TP、层/专家数量、校准输入哈希、每层访问计数和 forward 数。读入校验完整性及有限非负数，不匹配即失败。
- 排名使用校准热度初始化；共享缓存根据在线真实路由更新衰减热度。统计 misses 的访问，避免只给命中项加热导致自我强化。
- 常驻比例可在新 engine 启动时变化，重新 profiling 并分配 KV。
- 同一个 engine 内只允许在显式 quiescent RPC（请求批次结束并同步）调整比例/热专家集合；GPU 总池大小不变，常驻区与动态缓存相互交换容量，KV 不变。拒绝超出固定池容量的比例。没有“运行中自动从活跃 KV 偷显存”的实现，也不声称已实现自动控制器。

## 实验与证据

校准一次 -> GPU 小尺寸真实性测试 -> resident R -> 固定 R 实际 KV 的缓存 B -> 自动扩 KV 的缓存 C。每个实验点使用新进程/容器、同 commit/模型/输入/采样/TP/编译/显存预算。保留 0.60 优先、0.90 独立验证。支持一个原生 resident/零卸载开销对照，LRU 与 LFU 共用相同执行内核。

回传 whitelist JSON/CSV/中文报告：实际 per-rank KV 字节/blocks、模型/专家池字节、常驻命中/缓存命中/缺失/淘汰/准入/绕过、逐层 unique coverage、H2D bytes/copy 数、CPU 调度时间、同步/加载时间可用性、实际固定 tokens/elapsed、3 轮吞吐、smoke 和批量输出一致性范围。使用累计计数的 measured delta，排除初始化、校准和 warmup；不能把抽样时间当成端到端。

输出是否逐 token 相等与机制正确性分开报告；新增 CUDA 测试必须比较原生 fused_experts 与同路由映射池输出，并测试 TP shard、跨层命中、槽复用、全专家覆盖。单位 GB 使用十进制，原始 stdout/stderr 不自动进 Git。

## 本轮决策

1. 在线比例调整先以安全批次边界 RPC 提供，不做自动控制器；代价是不能在单个进行中的请求内自动交换 KV/专家预算。
2. 首版有一次路由需求 D2H 和必要的 pinned ingress 复用等待，明确记录成本；不声称已达到 GPU-only 元数据管理。
3. 先按需加载，不预测未来层路由；层执行距离只辅助淘汰同分项，不把猜测当作真实需求。
