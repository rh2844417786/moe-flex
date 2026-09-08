# 交给服务器 Codex 的执行说明

在 `/home/jovyan/wangtonghan/moe-flex` 的 `repro/fluxmoe` 运行本仓库新增的专家卸载可行性诊断。先完整阅读 `docs/offload-analysis-runbook.md` 与本说明，然后 `git pull --ff-only origin repro/fluxmoe`，记录代码 SHA，确认源码干净，按现有 `scripts/server/build.sh` 构建 pinned 镜像。用户已授权 GitHub 同步；无须为正常执行步骤重复询问。不要切 main/合并 PR。

只使用现有模型 `/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct`、原始 BF16、TP4，以及提交的数据与 manifest。文件 `qwen3next_1024_requests.jsonl.zst` 的 1024 是记录总数，内部含 1K/2K/3K/4K 桶，不要推导不存在的 4096 文件名。允许已有 Git 同步与固定基础镜像获取，不装新包，不下载模型/数据。项目外只读，所有缓存/日志/结果留项目内。

1. 检查四张指定 H100 独占和物理余量。使用 `GPU_IDS=0,1,2,3` 与 `scripts/server/run_offload_analysis.sh`；不满足则保存失败、停止，不杀他人进程、不悄悄降负载。每点新 ID，不能提前 mkdir runner 的 run-dir。超时在 Docker 内；worker 清理仅涉及自己的进程。
2. 用 stdlib CLI `plan` 写在 `runs/offload-analysis`，阅读有限分阶段清单。先 normal-budget utilization=.90/native 的 short/generation/medium 并发扫描（16/32/64/128/256/512），按实际资源逐点执行/保留失败。找性能或 KV 压力拐点，选少量负载；不要自动遍历完整比例×缓存×采样×模式笛卡尔积。0.60/eager 只在明确诊断需要时追加，不替代现实预算基线。
3. 对所选点采独立 calibration，然后 matched-native 与 eval trace 使用相同输入排除、seed、offset、长度、batch、调度设置及执行 SHA。检查每请求 prompt hash 无校准/测试交集。trace 必须 repetitions=1，强制 eager/timing_eligible=false；不能把 trace 时间当 native 吞吐。逐 rank 检查 observed/captured/truncated/missing/full_workload；长输出超 1024 steps 时以新 ID 显式增大有界捕获预算，不能按截断轨迹外推整批 bytes。
4. 读取 actual trace geometry 的 expert_bytes，四 rank 同时测 isolated contiguous/fragmented/gather 的真实 1/8/32/128/512 形状。可选单独 GEMM+NCCL proxy；保留 copy-only/compute-only/joint 窗口和 CPU gather。joint 已含代理计算，不得再次加入 K 时间预测。保留 worker checkpoint 中已完成样本；失败 remains failed，不汇总成完整成功。
5. 先 replay 与无 K analyze，查看净释放是否足够；只在物理余量安全时选择一个更高 utilization 的 measured native K（同输入/模型/TP/软件/代码/硬件）。K 是超基线预算 counterfactual，实际 KV 增量须在每 rank replay 净释放范围内。无法安全取得 K 就报告 bytes/bandwidth 和 missing evidence，不捏造 reference，不解释为没有收益。旧 Oracle 只能经自身 validator，再保留 over-budget/diagnostic 与缺身份边界。
6. 选择有限比例/缓存/策略回放，导出中文 JSON/CSV/Markdown 白名单报告。明确 repeated/synthetic 数据的外推限制、contiguous 预打包假设、CPU gather/NUMA/PCIe 限制、proxy 不证明真实 overlap。公开区分 native measured、trace observed、replay/cost predicted；所有 artifact 三个诊断标志固定。不得自动启动改写的真实 offloader，不将候选或乐观预测写成部署加速成功。
7. 保持测量序列同一执行 SHA，完成后再提交结果。仅提交经导出器生成与内容复核的 `docs/results/offload-analysis-*/report.json`、`report.csv`、`report.md`；明确逐文件 git add，禁止上传 raw plan/trace/log/prompt/UUID/绝对路径，即使它们误写进 docs/results。失败和 unreadable summary 的独立 smoke 也要报告。push 既有 `repro/fluxmoe` 并核对远端 SHA。

最后在聊天给出：执行 SHA/结果提交 SHA、实际 H100/TP/软件/数据/模型身份、normal .90 native 主要吞吐与实测 KV、选点理由、完整捕获比例、各传输策略 per-rank GB/s、净释放 GB、真实 K 时间窗口、预测情景和缺证据、失败/OOM/超时，以及是否值得后续真实同资源卸载验证。GB 一律十进制。没有硬件证据的项目写“未测”，不要以已创建命令或 CPU 单测当服务器完成。
