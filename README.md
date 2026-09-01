# moe-flex

> 状态：**DESIGN APPROVED - IMPLEMENTATION IN PROGRESS**

`moe-flex` 是 FluxMoE 的独立复现工程。代码在 MacBook 上开发并发布到
GitHub，服务器从公开仓库拉取后，在 4 张独占 NVIDIA H100 80GB PCIe GPU
上执行。

## 结论边界

- 目标是验证 PagedTensor、专家存储层级和动态驻留规划器是否真实工作、效果趋势
  是否符合论文预期。
- H100 不是论文的 L40 测试床，因此不要求复刻绝对吞吐或精确加速倍数。
- Mac 单元测试、GitHub 推送和服务器 checkout 都不等同于完成论文复现。
- 缺失正确性、传输、解压、映射或完整 telemetry 证据的运行统一标记为
  `INCONCLUSIVE`。

## 文档

- [设计规格](docs/superpowers/specs/2026-09-01-fluxmoe-reproduction-design.md)
- [实施计划](docs/superpowers/plans/2026-09-01-fluxmoe-reproduction.md)

## 固定环境

- vLLM：`v0.10.2` / `01efc7ef781391e744ed08c3292817a773d654e6`
- PyTorch：`2.8.0`
- 主模型：`/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct`（只读）
- 服务器项目：`/home/jovyan/wangtonghan/moe-flex`

当前分支只开始实现 CPU 可验证的工程骨架。CUDA 与完整模型结果必须等待服务器
实际运行后再更新状态。
