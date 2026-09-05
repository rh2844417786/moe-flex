# 部分 BF16 卸载与 KV 吞吐 Implementation Plan

> 按当前任务内执行和测试驱动流程推进；运行时与实验工具两个独立模块可并行开发。

**Goal:** 在相同 GPU、显存预算、模型精度和请求工作量下，验证少量专家权重卸载是否通过增加实际 KV 容量提高端到端输出吞吐。

**Architecture:** 大部分 MoE 层维持 vLLM 原生 BF16 GPU Parameter。选定少数、均匀分布的层保存为连续 pinned CPU BF16 权重，使用固定 GPU staging slots、CUDA stream/event 进行预取；计算中无 VMM map/unmap 和 Huffman 解压。CPU 路径只保留固定大小的聚合计数。

**Tech Stack:** PyTorch 2.8.0、vLLM 0.10.2、现有单文件 vLLM hook、Python 3.10+。

**Spec:** 本文按用户 2026-09-05 的澄清替代先前全压缩 GPU 优化主线：工作负载自主设计、离线吞吐优先、BF16 和原路由保留、由用户在服务器执行，通过 GitHub 回传脱敏数值。

## 固定约束

- 主模型只读：`/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct`。
- 服务器项目：`/home/jovyan/wangtonghan/moe-flex`；所有写入留在该目录。
- 4 张独占 H100，TP=4；先双方 utilization=0.60，再验证 0.90。
- Mac 只编写、CPU 测试和发布；CUDA 正确性及性能必须由服务器结果验证。
- 不下载新模型、不改精度、不更改 router、不使用其他空闲 GPU 隐藏权重。
- 一条 GPU staging slot 的物理权重可供多个逻辑层使用；只有计算完成事件之后才能覆盖。
- staging_slots 支持 1 或 2；卸载层数必须大于 slots 且可被 slots 整除，保证跨迭代静态 slot 映射正确。
- 零卸载作为新 hook 的开销/正确性对照；正数卸载必须给出扣除 staging 后的净权重回收量。

## 任务 1：静态计划与调度合同

Files: `src/flexmoe/runtime/partial_plan.py`、`tests/unit/test_partial_plan.py`。

- [ ] 手工固定 8 层、4 层卸载、2 slots 的调度期望，验证跨轮次 slot 使用、净回收量、无效配置。
- [ ] 先运行失败测试，再实现 `PartialPlan(total_layers, offload_layers, staging_slots)`。
- [ ] 对 `PartialPlan.evenly_spaced(total_layers=8, offload_count=4, staging_slots=2)`，期望卸载层为 `(1, 3, 5, 7)`。
- [ ] 返回 `slot_for(layer)`、`prefetch_after(layer)`；例如 layer 1 计算后为其 slot 预取 layer 5。

## 任务 2：连续 CPU store 与固定 staging 运行时

Files: `src/flexmoe/vllm/partial.py`、`src/flexmoe/vllm/bridge.py`、`tests/unit/test_partial_runtime.py`、`tests/cuda/test_partial_runtime.py`。

- [ ] 先测试部分卸载层与普通 resident 层参数创建分流，真实 CPU store 的 shard 拼接和完整性错误。
- [ ] 为卸载层各自分配连续 pinned `w13`/`w2`，GPU slots 用普通 `torch.empty`；不分配全模型 GPU experts 再拷走。
- [ ] 完成权重加载后预取前 slots 个卸载层；每个层只发起两次连续 copy。
- [ ] 计算流等待 ready event；计算后记录 consumed event，加载流等待 consumed 后预取后续卸载层。
- [ ] 暴露 `FLUXMOE_STORAGE_MODE=partial-host`、`FLUXMOE_PARTIAL_OFFLOAD_LAYERS=逗号分隔层号`、`FLUXMOE_PARTIAL_STAGING_SLOTS=1或2`。
- [ ] 新 worker RPC：`fluxmoe_partial_stats` 返回 rank-local 聚合计数与抽样时序；`fluxmoe_worker_memory_stats` 返回原生 KV 预算与实际 GPU 内存信息。
- [ ] 验证无 GIL 持锁的 C++ 同步调用，计数不保留逐 copy Python receipts。
- [ ] CUDA 测试覆盖多轮循环、并发加载、slot 覆盖顺序以及 BF16 权重逐位一致。

## 任务 3：实验执行器与三组因果对照

Files: `src/flexmoe/bench/partial_runner.py`、`tests/unit/test_partial_runner.py`。

- [ ] 专用执行器保留旧 runner 和历史数据。支持 resident 与 partial-host；固定 greedy 输出长度，禁用 prefix caching，记录完整有效配置。
- [ ] Resident worker 同样启用内存统计 RPC，以获取基线 KV 字节数。
- [ ] R: resident 原 KV；B: 部分卸载但通过 `kv_cache_memory_bytes` 固定为 R 的 KV；C: 同样部分卸载并让 vLLM 自动扩大 KV。
- [ ] 固定输出长度、请求数和输入 ID 的 SHA；任何失败立即标记失败，不把异常作为加速。
- [ ] 单请求正确性与性能分开记录；性能前后在所有 worker 同步并取 snapshot，计时不含初始化。
- [ ] 每 repetition 原子保存脱敏数值；不把累计 warmup 计数混入 measured delta。

## 任务 4：离线实验生成、回传与收益判断

Files: `src/flexmoe/bench/partial_suite.py`、`tests/unit/test_partial_suite.py`、`scripts/server/run_partial_offload.sh`。

- [ ] 从已有 tokenized ShareGPT 子集确定性生成请求，保留来源 SHA；更高请求数只复制既有请求并明确记录重复；不发生网络请求。
- [ ] 先用小型诊断点扫描 offload_count=2/4/6/8 与 slots=1/2 的有效组合，再生成 R/B/C 正式确认点。
- [ ] 执行耗时设上限，失败保留 stderr 和状态；长慢点不无止境重跑。
- [ ] 输出数值 JSON、CSV 和 Markdown，校验模型/数据/精度/并发/输出/总预算相同。
- [ ] 报告 B/R 卸载代价、C/B KV 收益、C/R 净收益。C/R>1 且重复测量有稳定增益后才标记达到目标。
- [ ] 只有允许列表中的数值字段进入 Git 回传包；排除 prompts、token IDs、文本、机器身份和权重路径。

## 任务 5：集成、验证与交付

- [ ] Mac 跑原有与新增 unit tests、Ruff 和 mypy；CUDA tests 写好并在服务器脚本运行。
- [ ] 检查现有补丁的调用点，确保固定 staging 参数不被后处理复制而脱离 slot；添加地址检查。
- [ ] 更新中文运行说明，明确新增路径尚待 H100 验证、服务器无普通网络、GitHub 回传协议。
- [ ] 复核 diff 只包含本功能，提交并正常 push 到 `repro/fluxmoe`；若网络阻断，保留提交并说明准确阻断原因。

## 性能证据与验收

三组使用同一 commit、4 GPU、同一 utilization 和同一 fixed-token workload。KV 字节来自 worker，不以“空闲显存”替代已分配 KV。固定工作量的端到端 throughput 为主要指标，GPU event 及 CPU enqueue 时间用于诊断，不拿 event 时间替代端到端时间。

先报告净增益及三轮范围；有明确收益后复跑正反执行顺序。0.60 的成功只证明受限显存场景，0.90 另行报告。离线吞吐验收不设置交互延迟硬门槛，但保留可测时延。任何本地测试或发布完成都不代表 H100 吞吐目标已经达到。

## 本轮实现记录

- 已实现静态部分卸载计划、连续 CPU BF16 存储、固定 GPU staging 和跨轮次预取。
- 已实现 resident/部分卸载分流、原始指针/步长检查、首次 BF16 位级校验、worker KV 实际存储去重。
- 已实现有界 CUDA 抽样时序、R/B/C 专用执行器、固定工作量、超时、原子存档和白名单回传。
- CUDA 测试已包含非默认 stream、延迟消费、TP 分片和多轮 slot 复用；Mac 只收集并跳过这些测试，实际执行待 H100。
- 每个点启动时重新 preflight；基础镜像已缓存时支持完全离线构建。
- 原有 Huffman/VMM/全量卸载与服务器路由按需卸载代码作为历史和对照保留。
