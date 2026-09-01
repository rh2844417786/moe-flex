# moe-flex

> 状态：**IMPLEMENTED AND CUDA-COMPILED - H100 VALIDATION PENDING**

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

## 已实现边界

- 已实现 CUDA VMM 稳定地址、双层 RAW/WAR 生命周期、BF16 canonical
  Huffman CUDA 解码、GPU 压缩存储、pinned-host HtoD 和固定 vLLM 单文件 hook。
- Python 3.10/3.11 CPU CI 与公开 GHCR 的 `linux/amd64` CUDA 编译已通过。
- 当前可执行对照是 `resident` 与 `fluxmoe-fixed`。`vllm-o`、dynamic、
  unbalanced 和 pagedtensor-resident 仅保留为 `DEV_ONLY` 配置合同，runner 会
  显式拒绝，避免错误标注实验。
- 尚未执行 H100 CUDA 数值测试、完整 Qwen3-Next parity 或性能矩阵，所以现在
  仍不能称为论文复现完成。

## Mac 验证

```bash
python3 -m pip install -e '.[dev]'
ruff check .
mypy src/flexmoe
pytest tests/unit -q
```

固定的 1024 条 ShareGPT 请求已随 Git 提交，服务器无需另外下载数据集。

## 服务器执行

以下命令只创建新 checkout/新容器，不修改已有的 `wth333` 容器；模型目录始终
以只读方式挂载，所有结果写入
`/home/jovyan/wangtonghan/moe-flex/runs/`。

```bash
git clone --branch repro/fluxmoe --single-branch \
  https://github.com/rh2844417786/moe-flex.git \
  /home/jovyan/wangtonghan/moe-flex
cd /home/jovyan/wangtonghan/moe-flex

# 必须替换为当时确实空闲且独占的 4 张 H100。
export GPU_IDS=4,5,6,7
scripts/server/build.sh
scripts/server/preflight.sh
scripts/server/run_smoke.sh
scripts/server/run_key_matrix.sh
```

若已有 checkout，只允许快进更新：

```bash
git pull --ff-only origin repro/fluxmoe
```

任何 preflight、CUDA/sanitizer、正确性或机制计数失败都会阻止性能矩阵。吞吐结果
在 router Top-k、greedy token、bit-exact 权重以及非零映射/HtoD/解压证据齐全前
保持 `INCONCLUSIVE`。
