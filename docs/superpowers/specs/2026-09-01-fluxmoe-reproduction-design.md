# FluxMoE H100 跨硬件复现设计

## 1. 目标与结论边界

本项目在 MacBook 上完成代码、测试、数据集和容器配置，在服务器上通过 GitHub 拉取后执行。目标是独立实现 FluxMoE 的三个核心机制，并在 4 张独占 NVIDIA H100 80GB PCIe GPU 上检查效果是否符合论文预期：

1. PagedTensor：稳定虚拟地址与按层物理内存重映射；
2. 专家存储层级：压缩 GPU 存储与 pinned host DRAM 并行供给；
3. budget-aware residency planner：根据计算、加载和显存压力调整专家驻留率。

结果属于“H100 跨硬件复现”，不要求匹配论文 L40 测试床的绝对吞吐、3.0 倍加速、3% 管理开销、5.3GB 回收量或精确调整次数。性能判断关注机制是否真实启用、相对趋势是否合理以及高内存压力下是否产生预期收益。权重重建、路由和模型输出正确性仍是硬性门禁。

## 2. 已确认约束

- 公开 GitHub 仓库：`rh2844417786/moe-flex`；
- 稳定分支：`main`；
- 复现开发与服务器执行分支：`repro/fluxmoe`；
- 服务器 checkout：`/home/jovyan/wangtonghan/moe-flex`；
- 服务器后续更新只使用 `git pull --ff-only`；
- 服务器暂不向 GitHub 反向推送；
- 正式实验可获得至少 4 张无其他进程占用的 H100；
- 创建独立容器，不修改已运行的 `wth333` 容器；
- `/mnt/public_data` 在本项目中只读；
- 所有可写缓存、构建产物、日志和结果位于服务器项目目录；
- 只使用服务器已经存在的模型权重；
- ShareGPT 基准子集在 MacBook 上生成并随 Git 提交；
- 最终状态使用 `SUPPORTED`、`MIXED`、`NOT SUPPORTED` 或 `INCONCLUSIVE`。

## 3. 模型与数据

### 3.1 主模型

严格主线使用现有 BF16 checkpoint：

`/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct`

启动预检必须验证：

- `config.json` 的架构为 `Qwen3NextForCausalLM`；
- 权重 index 存在；
- 41 个权重 shard 全部存在且非空；
- checkpoint 为 BF16，而不是 AWQ、BNB、GPTQ 或 FP8；
- checkpoint 总大小与清单规模一致；
- 文件可读但源目录不可写。

### 3.2 补充模型

`/mnt/public_data/modelscope/mistralai/Mixtral-8x7B-v0.1_` 只用于跨 checkpoint 验证。它是 Base checkpoint，不是论文使用的 Instruct checkpoint，因此不得写成论文 Exp#2 的严格原配置复现。

### 3.3 ShareGPT 子集

MacBook 下载以下 Apache-2.0 数据快照：

- repository：`anon8231489123/ShareGPT_Vicuna_unfiltered`；
- revision：`192ab2185289094fc556ec8ce5ce1e8e587154ca`；
- file：`ShareGPT_V3_unfiltered_cleaned_split.json`；
- tokenizer：只下载 `Qwen/Qwen3-Next-80B-A3B-Instruct` revision `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7` 的 tokenizer/config 文件，不下载模型权重；
- sampling seed：`20260901`。

下载后记录原始文件 SHA256。生成器按固定顺序抽样并应用 Qwen chat template。若单条对话短于目标 context，生成器按抽样顺序拼接后续对话并插入 EOS；达到目标后精确截断到指定 token 数。最终生成 1024 条请求：

- 输入长度 1024：256 条；
- 输入长度 2048：256 条；
- 输入长度 3072：256 条；
- 输入长度 4096：256 条。

仓库只提交确定性基准子集、生成脚本、来源清单和校验和，不提交完整原始数据集。单个提交文件必须小于 GitHub 100MB 限制。

## 4. 技术路线

采用“外置 C++/CUDA 扩展 + 固定 vLLM 极小补丁”。不维护完整 vLLM fork，也不把纯 Python 原型作为性能复现结果。

- vLLM：固定 `v0.10.2`，tag commit `01efc7ef781391e744ed08c3292817a773d654e6`；
- PyTorch：2.8 系列；
- CUDA：使用与 vLLM v0.10.2 和 PyTorch 2.8 匹配的工具链；
- 第一版强制 eager mode；
- CUDA Graph 只有通过独立重放测试后才能显式启用；
- 论文主线只使用 tensor parallelism，Qwen3-Next 正式实验为 TP=4；
- routed experts 进入分页系统；shared expert、router、attention 和其他非专家权重继续常驻 GPU。

项目模块边界：

```text
vLLM adapter
    -> PagedTensor CUDA VMM
    -> compressed-GPU backend
    -> pinned-host backend
    -> bandwidth-balanced placement
    -> budget-aware residency planner
    -> telemetry and experiments
```

## 5. PagedTensor

PagedTensor 通过 CUDA Driver VMM API 实现：

- `cuMemGetAllocationGranularity`；
- `cuMemAddressReserve`；
- `cuMemCreate`；
- `cuMemMap`；
- `cuMemSetAccess`；
- `cuMemUnmap`；
- `cuMemRelease`；
- `cuMemAddressFree`。

对每个 TP rank：

- 为 `w13_weight` 和 `w2_weight` 分别保留连续虚拟地址空间；
- PyTorch Parameter 的 data pointer 在整个进程生命周期内保持不变；
- 每种权重类型维护 `2L` 个物理块，总计 `4L`；
- 当前层和下一层构成两层滑动驻留窗口；
- 当前层计算期间异步物化下一层；
- 只有在对应 compute event 完成后才能解除映射和复用物理块。

两个加载流分别处理 `w13` 和 `w2`。RAW event 防止 kernel 读取未完成权重，WAR event 防止仍被 kernel 使用的物理块被提前复用。

## 6. 运行时权重准备与存储层级

服务器项目盘剩余空间不足以持久化完整压缩 checkpoint。因此原始 safetensors 保持在只读 NFS，压缩和分层在进程启动阶段完成：

1. 逐个读取 safetensors tensor；
2. 非专家权重按 vLLM 正常方式加载到 GPU；
3. routed-expert 参数不创建完整常驻 GPU Parameter；
4. BF16 指数由 CPU canonical Huffman 编码；
5. sign 和 mantissa 原样保存；
6. compressed-GPU 部分直接进入 GPU 压缩存储；
7. host-offload 部分直接进入 pinned host DRAM；
8. 项目 `artifacts/` 只保存小型 codebook、placement manifest、校验和和环境记录。

GPU Huffman 解压 kernel 必须逐位重建 BF16，包括普通值、零、负零、subnormal、Inf 和 NaN。GPU 后端解压与 host 后端 HtoD 在不同 CUDA stream 上并行执行。

每个 rank 在启动阶段测量：

- compressed-GPU 解压有效带宽；
- pinned-host 到本地 GPU 的 HtoD 有效带宽。

placement 按有效带宽比例分配 expert tensor，同时记录离散分配造成的实际偏差。所有分配必须由 manifest 决定，不能依赖不可复现的迭代顺序。

## 7. Residency planner

planner 维护当前 expert residency level，并使用以下输入：

- resident 配置下的参考计算时间；
- 当前 placement 推导和 CUDA event 实测的加载时间；
- 当前 GPU expert 存储、物理块、非专家权重和 runtime reserve；
- vLLM 报告的 KV cache 占用和总显存预算。

默认控制策略：

- 加载明显快于计算时，允许降低 expert residency 以释放显存；
- 加载明显慢于计算时，提高 expert residency；
- 计算/加载接近平衡时保持不变；
- 显存安全约束高于性能调优目标；
- 迁移跨层交错进行，避免一次集中迁移单层形成 I/O 峰值。

每次决策写入 JSONL，至少包含 iteration、residency、placement、迁移字节、GPU/host expert bytes、KV cache、加载时间、计算时间、stall 和吞吐。

由于 vLLM v0.10.2 的 KV cache 池通常在启动时确定，运行中释放显存不等于已动态扩大 KV cache。报告必须区分“回收了可用 GPU 内存”和“KV cache 实际增加”。

## 8. Fail-closed 条件

出现以下任一情况立即退出，不允许静默退回普通 vLLM 后仍标记为 FluxMoE：

- 模型 shard、index、dtype 或配置不匹配；
- vLLM commit 或补丁上下文不匹配；
- CUDA VMM/UVA 不可用；
- 地址、物理块或映射大小不满足 granularity；
- 权重重建不是 bit-exact；
- router Top-k 或 greedy 输出发生变化；
- 正式实验 GPU 上存在其他计算进程；
- 请求启用未经验证的 CUDA Graph；
- 缺失必要 telemetry；
- `/mnt/public_data` 被检测为写入目标。

失败的运行写入结构化错误状态；证据不全的运行标记为 `INCONCLUSIVE`。

## 9. 测试策略

### 9.1 MacBook

- Huffman CPU 参考 round-trip；
- planner 状态机和显存安全约束；
- placement 确定性；
- model/data manifest 校验；
- ShareGPT 子集生成确定性；
- lint、类型检查和纯 CPU 单元测试。

MacBook 测试不证明 CUDA 扩展可编译或论文机制已经复现。

### 9.2 服务器 CUDA

- VMM pointer stability；
- 多轮 map/unmap 和物理块复用；
- RAW/WAR 双流压力测试；
- `compute-sanitizer` 越界、竞态和 use-after-unmap 检查；
- CUDA Huffman 解压对比 CPU reference；
- HtoD 和 decompression 带宽基准；
- 小型 FusedMoE 常驻权重/分页权重输出一致性；
- 完整 Qwen3-Next 权重、router Top-k 和 greedy token 一致性。

## 10. 对照组与实验顺序

对照组：

1. `vllm-resident`：原生 vLLM，全权重常驻；
2. `vllm-o`：最后 12.5% expert layers 从 host 预取；
3. `fluxmoe-fixed`：专家压缩驻留 GPU，按层解压；
4. `fluxmoe-dynamic`：带宽均衡层级与动态驻留；
5. `fluxmoe-dynamic-unbalanced`：planner 相同，但集中迁移单层；
6. `pagedtensor-resident`：不压缩、不卸载，仅测 PagedTensor 管理开销。

先运行三个关键点，每点 warmup 3 次并正式运行 3 次：

1. batch 32、context 1024；
2. batch 128、context 4096；
3. batch 256、context 4096。

关键点效果合理后，再决定是否运行 batch `32/64/128/256` 与 context `1024/2048/3072/4096` 的完整矩阵。动态 planner 实验使用长解码，运行到显存压力触发多次调整为止，不要求精确复现论文的调整次数。

## 11. 指标与效果判定

记录：

- output tokens/s；
- TTFT；
- TPOT P50/P95/P99；
- GPU/host expert bytes；
- HtoD 和 decompression bytes；
- 每层 load/compute/stall；
- KV cache 和峰值显存；
- GPU utilization；
- planner 决策轨迹。

正确性硬门禁：

- BF16 权重 bit-exact；
- router Top-k 一致；
- greedy 输出 token 一致；
- 非零 VMM 映射、HtoD、解压和物理块复用证据。

性能效果按趋势判断：

- 高 batch/长 context 下，FluxMoE 相对原生 vLLM 有改善，或明显推迟内存瓶颈；
- dynamic placement 确实释放非零 GPU 内存；
- bandwidth-balanced 版本比集中迁移版本更平稳；
- 小 batch 因解压开销低于 resident baseline 属于合理结果；
- 不设置精确加速倍数、管理开销或回收字节阈值。

## 12. 运行产物

服务器运行目录：

`/home/jovyan/wangtonghan/moe-flex/runs/<timestamp>-<git-sha>/`

每次运行包含：

- `environment.json`；
- `config.yaml`；
- `preflight.json`；
- `events.jsonl`；
- `metrics.json`；
- `summary.csv`；
- stdout/stderr 日志；
- 图表与结论 Markdown。

`.cache/`、大型 `artifacts/` 和原始 `runs/` 默认加入 `.gitignore`。仓库只提交代码、配置、数据子集、校验和、小型摘要和选定图表。

## 13. GitHub 与服务器工作流

首次发布：

1. 在 `rh2844417786` 账号下创建公开仓库 `moe-flex`；
2. 本地 `origin` 指向该仓库；
3. `main` 保存稳定规格与稳定代码；
4. `repro/fluxmoe` 保存开发、测试和服务器执行版本；
5. 不强制推送；
6. 推送前检查远端是否有新的服务器或其他任务提交。

服务器首次下载：

```bash
git clone --branch repro/fluxmoe --single-branch \
  https://github.com/rh2844417786/moe-flex.git \
  /home/jovyan/wangtonghan/moe-flex
```

后续更新：

```bash
git -C /home/jovyan/wangtonghan/moe-flex \
  pull --ff-only origin repro/fluxmoe
```

服务器不需要 GitHub 写权限。独立容器把项目目录以读写方式挂载，把 `/mnt/public_data` 以只读方式挂载，并只暴露正式实验选定的 4 张 GPU。

## 14. 状态边界

- 代码与 Mac 单测完成：只能称为本地实现或脚手架完成；
- GitHub 推送完成：只能称为已发布代码；
- 服务器成功 pull：只能证明服务器 checkout 对齐；
- CUDA 单测完成：只能证明底层机制测试通过；
- 完整模型正确性和 telemetry 门禁通过：可以称为机制复现；
- H100 对照实验呈现合理效果：可以称为 H100 跨硬件复现得到支持；
- 任一关键证据缺失：状态保持 `INCONCLUSIVE`。
