# 卸载是否值得继续：TP4 低并发验证

确认 GPU 0–3 由本实验独占后，使用下面的完整快进更新与执行序列：

```bash
cd /home/jovyan/wangtonghan/moe-flex && git switch repro/fluxmoe && git pull --ff-only origin repro/fluxmoe && GPU_IDS=0,1,2,3 bash scripts/server/run_decode_decision.sh
```

脚本先使用服务器已缓存的固定 Docker Hub 基础镜像执行**离线构建**，并校验新镜像标签的 Git SHA 与当前提交相同；基础镜像不存在时直接失败，不联网拉取。随后只读取现有的 `/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct` 权重与仓库内已提交数据。每个点用既有的离线容器预检和固定 BF16/TP4 协议；显存比例 0.90，物理安全余量每卡 2,000,000,000 bytes。失败时停止，完整日志在 `runs/decode-decision/<ID>/`、各点原始记录在 `runs/decode-mechanism/<ID>-<point>/`，不会清理这些目录。完成后的全部公开数字、四 rank × 三重复的零 miss 判据、缓存 replay、真实 Oracle、正确性门槛及 unavailable 原因直接写入：

```text
docs/results/decode-decision-<ID>/report.md
docs/results/decode-decision-<ID>/report.json
```

仅在**已完成一个点、下一个点尚未开始**的间隙中断时，可以传入实际 ID：

```bash
GPU_IDS=0,1,2,3 bash scripts/server/run_decode_decision.sh --run-id <ID> --resume
```

处于 `running` 或 `failed` 的点均不能在原路径覆盖；先确认容器与失败原因，再用新 ID 重新开始。脚本不会将原始 token 路由和隐藏数据提交 Git；公开报告只含聚合数字。提交报告前应先人工检查异常及原始记录。

## 执行顺序与判断

1. 在 1K/4K 上各建一个互斥校准集的专家热度 profile；在相同硬件、预算、4K/batch1 工作量下分别测 native 全驻留、eager 全驻留与 0.8/512 卸载的**自动分配 KV**。卸载必须超过两种全驻留模式中的更高实测容量、且增量至少达到一个真实 KV block，才称为新增容量。
2. 以 eager resident/offload 两者自动容量的较小者作为**同模式配对 KV**，在 1K/batch1 下进行无采样 eager resident vs 0.9/512 offload 三重复。真实零 miss 必须由四 rank × 三重复的 cache miss、payload bytes、copy launches 同时为零证明。另跑一个独立的 `zero-offload-profile` instrumented 点，以观察 CUDA step p50/p95 与拆分后的 weight gather / host map CPU 区间；instrumented 吞吐不进入净收益对照。
3. 在 4K、64 个不同请求、最大并发 32、输出 256 token 下，测 native 在自身自动容量下的强参考，以及**同 KV** eager 全驻留/卸载配对；容量门槛通过时再测同卸载路径的额外 KV。新增 KV 的吞吐比也必须通过代码版本、硬件、工作量、缓存策略及输出 smoke 的一致性校验才报告。报告实际 decode batch、KV 使用峰值、等待请求峰值、抢占、TTFT 与请求延迟的可用字段；缺少的准入阻塞时间、ITL、独立 prefill/decode 墙钟不得猜测。
4. 在同一 eager 配对 KV 下采 batch16/32 的连续低并发轨迹，保存首次采集前的**真实策略状态**（固定驻留、槽位、热度和时钟）及逐层去重需求。缓存重放在已加载 PyTorch 的离线实验容器内完成，宿主的 `python3 -S` 只读取无原始路由的数字结果。仅在每个 rank、每次采样的原策略逐行复现真实 hit/miss、bypass、admission、eviction、前后状态和 bytes 时，报告同预算 LRU 和未来知情淘汰。LFU 运行未保存此前 LRU 顺序时，LRU 结果明确标为条件性对照。
5. 使用同一确定性 trace 分别运行 Router 后按需、提前 1 层和提前 2 层。预取复用现有 ingress，不增加 cache、ingress 或 KV，不提前改变 cache policy；真实执行 host gather、独立 transfer queue、H2D、映射、ready wait、promotion 和专家计算。每个 horizon 的正式吞吐来自 profile-off 点，等待分位数来自对应 profile 点。
6. 单独执行强制漏预测正确性点：从一个真实未来集合中删除一个需要的专家，要求四 rank 都恰好按需补载一次、Router IDs 不变、输出哈希与 horizon 0 一致。该点不进入吞吐比较。

cache replay 与真实 Oracle 始终分开：前者只给出可避免的加载流量，后者才给出真实 H2D 等待和端到端吞吐。instrumented repetition 不作为正式 throughput，局部 rank/CPU/GPU 时间不相加。

## 结果回传

服务器完成后只提交公开聚合报告，不提交 `runs/` 下的逐步专家 IDs、prompt/output token 或私有日志。先从程序输出复制实际 ID，替换下面示例中的值；禁止使用通配符，以免把旧结果或用户文件一起暂存：

```bash
RUN_ID=20260921T000000-example
git add "docs/results/decode-decision-${RUN_ID}/report.md" \
        "docs/results/decode-decision-${RUN_ID}/report.json"
git commit -m "results: add constrained decode oracle experiment"
git push origin repro/fluxmoe
```

若失败，控制器也会把已完成点、失败点和后续 `unreached` 点写入同一公开 `report.md/report.json`；同时保留 `runs/decode-decision/<ID>/state.json`、对应 point 的 stdout/stderr 和原始不可覆盖目录。回传时仍只提交该实际 ID 下的两份公开报告。不要删除、覆盖或用同一失败 ID 重跑。
