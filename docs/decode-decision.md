# 卸载是否值得继续：TP4 低并发验证

在已提交的 `repro/fluxmoe` 分支、服务器 `/home/jovyan/wangtonghan/moe-flex` 下，确认 GPU 0–3 由本实验独占后执行：

```bash
GPU_IDS=0,1,2,3 bash scripts/server/run_decode_decision.sh
```

脚本先使用服务器已缓存的固定 Docker Hub 基础镜像执行**离线构建**，并校验新镜像标签的 Git SHA 与当前提交相同；基础镜像不存在时直接失败，不联网拉取。随后只读取现有的 `/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct` 权重与仓库内已提交数据。每个点用既有的离线容器预检和固定 BF16/TP4 协议；显存比例 0.90，物理安全余量每卡 2,000,000,000 bytes。失败时停止，完整日志在 `runs/decode-decision/<ID>/`、各点原始记录在 `runs/decode-mechanism/<ID>-<point>/`，不会清理这些目录。完成后的数字、四 rank × 三重复的零 miss 判据及未测量边界直接写入 `docs/results/decode-decision-<ID>/report.md`。

仅在**已完成一个点、下一个点尚未开始**的间隙中断时，可以传入实际 ID：

```bash
GPU_IDS=0,1,2,3 bash scripts/server/run_decode_decision.sh --run-id <ID> --resume
```

处于 `running` 或 `failed` 的点均不能在原路径覆盖；先确认容器与失败原因，再用新 ID 重新开始。脚本不会将原始 token 路由和隐藏数据提交 Git；公开报告只含聚合数字。提交报告前应先人工检查异常及原始记录。

## 执行顺序与判断

1. 在 1K/4K 上各建一个互斥校准集的专家热度 profile；在相同硬件、预算、4K/batch1 工作量下分别测 native 全驻留、eager 全驻留与 0.8/512 卸载的**自动分配 KV**。卸载必须超过两种全驻留模式中的更高实测容量、且增量至少达到一个真实 KV block，才称为新增容量。
2. 以 eager resident/offload 两者自动容量的较小者作为**同模式配对 KV**，在 1K/batch1 下进行无采样 eager resident vs 0.9/512 offload 三重复。真实零 miss 必须由四 rank × 三重复的 cache miss、payload bytes、copy launches 同时为零证明。另跑两个独立 instrumented 点以观察 CUDA step p50/p95 与包含映射构建的 `host_gather` CPU 区间；instrumented 吞吐不进入净收益对照。
3. 在 4K、64 个不同请求、最大并发 32、输出 256 token 下，测 native 在自身自动容量下的强参考，以及**同 KV** eager 全驻留/卸载配对；容量门槛通过时再测同卸载路径的额外 KV。新增 KV 的吞吐比也必须通过代码版本、硬件、工作量、缓存策略及输出 smoke 的一致性校验才报告。报告实际 decode batch、KV 使用峰值、等待请求峰值、抢占、TTFT 与请求延迟的可用字段；缺少的准入阻塞时间、ITL、独立 prefill/decode 墙钟不得猜测。
4. 在同一 eager 配对 KV 下采 batch16/32 的连续低并发轨迹，保存首次采集前的**真实策略状态**（固定驻留、槽位、热度和时钟）及逐层去重需求。缓存重放在已加载 PyTorch 的离线实验容器内完成，宿主的 `python3 -S` 只读取无原始路由的数字结果。仅在每个 rank、每次采样的原策略能逐层复现真实 miss 和 bytes 时，报告同预算 LRU 和未来知情淘汰的 miss/bytes 对照。LFU 运行未保存此前 LRU 访问顺序，LRU 以相同缓存占用、按槽位顺序初始化，因而其结果是条件性对照。

最后一步仅是缓存替换上界的离线重放，**不是**未来专家预取。权重真实异步传输、Router 后/提前一层/提前两层可用窗口、全层就绪、强制漏预测补载、暴露等待及端到端 Oracle 吞吐，必须在零 miss 固定开销与真实 KV 净增量的门槛检查之后另实现并实测；报告会明确记为未测量。
