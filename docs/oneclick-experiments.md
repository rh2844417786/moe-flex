# 一行启动与断点恢复

在服务器前台执行；需要脱离终端时，先进入已有 tmux 会话。四张卡必须由你显式指定且独占。默认不自动选择卡、不终止其他任务、不下载权重或依赖、不修改 Git、不自动 push。

```bash
cd /home/jovyan/wangtonghan/moe-flex && git switch repro/fluxmoe && git pull --ff-only origin repro/fluxmoe && GPU_IDS=0,1,2,3 bash scripts/server/run_all_experiments.sh
```

主机需要 Python 3.10–3.13 标准库、Git、Docker 及已有 GPU 运行权限。沿用固定 vLLM 0.10.2 镜像/补丁、已有唯一输入集和只读 Qwen3-Next 权重。镜像构建沿用 build.sh 的既有基础镜像拉取规则，其余安装离线。运行账户必须可读容器生成的原始报告并可写项目内缓存/输出；权限错误记为 artifact-permissions，不自动 chown 或更改全局权限。

启动时自动打印唯一 suite ID，默认每点 7200 秒，加 60 秒外层宽限；预检另有最多 300 秒加 60 秒宽限。每个 GPU 点先做现有四卡预检，decode 包装器还会再做一次自己的预检。这会增加启动成本，不进入 runner 测量计时。预检占卡或本次容器清理无法确认时停止后续 GPU 工作。

只预览有限计划，不运行 GPU、不创建检查点：

```bash
GPU_IDS=0,1,2,3 bash scripts/server/run_all_experiments.sh --dry-run
```

程序自动顺序执行：同 SHA 构建一次、唯一语料校验、三组全常驻并行配置、1K/4K 各自校准、旧 TP4 native、1K 七点覆盖、至多三个已达代表点的 4K 覆盖与九个有限卸载采样点、A/B/C 三点、采样开销两点。配置失败保留为失败，不静默改成其他并行方式。EP 卸载不支持。

并行配置为 TP4/DP1/EP关、TP1/DP4/EP4、TP2/DP2/EP4。都用 BF16、全权重常驻、1024 个唯一输入、1K 输入/512 输出、一次预热、三次重复、seed 20260912、利用率 0.90；全局请求/调度上限按 DP 分配。总吞吐是总输出除以父进程同轮墙钟，不能相加本地 TPS。

覆盖点只从完整 instrumented 采集中选最低、中间、最高实际已达 batch；这些采集不进入正式吞吐。KV 预算只用本次旧 TP4 native 的四 rank 实际分配、block 字节及物理内存证据，小点取一半并向下对齐，大点不超过最小已观测容量，仍遵守每点至少 2 GB 物理余量。没有有效证据就跳过依赖阶段；不猜测固定 16/32 GB，不尝试吃满权重释放空间。

## 恢复与状态

把 `ID` 替换成首次启动打印的 suite ID。保持原 SHA、相同 GPU_IDS、相同 timeout。恢复前不要再次拉取新代码。成功点只有原始文件完整、校验和及语义验证通过才复用。旧目录永不覆盖；中断或失效输出使用新 attempt，已失败的点默认不重试。

```bash
GPU_IDS=0,1,2,3 bash scripts/server/run_all_experiments.sh --run-id ID --resume
GPU_IDS=0,1,2,3 bash scripts/server/run_all_experiments.sh --run-id ID --resume --retry-failed
bash scripts/server/run_all_experiments.sh status --run-id ID
```

`--retry-failed` 每个失败点最多增加一次尝试，不循环调参。已形成的代表 batch 和 KV 决策冻结；依赖 profile 或参数改变不会复用旧成功点。若原代表点失效或新 native 不再支持冻结预算，则跳过相应依赖阶段。运行中 SIGINT/SIGTERM 会保存状态并清理本次已记录 CID；断电后恢复先检查旧尝试的归属。只允许删除 owner、suite、SHA 三标签都匹配的精确 CID，任何归属/daemon 状态不确定都安全停止；不会按进程名、容器名前缀或全局清理操作误杀别人。

## 输出与回传

私有检查点、尝试日志和归属记录位于 `runs/experiment-suite/ID/`。decode 原始证据继续位于 `runs/decode-mechanism/ID-aNNN-...`；实际校准目录显式传给下游。并行原始证据保存在对应 attempt 的 config/parallel-run 下。原始 token、输入/输出哈希、UUID、路径和日志只留服务器。

每次退出生成一个新 `docs/results/decode-mechanism-suite-ID/report-NNN/`，含中文汇总、并行对比、各 decode 尝试的白名单报告以及有证据时的独立 TP4 A/B/C 对比。缺失吞吐保持 null，失败不记为零。报告明确诊断范围，不把仪器化采集当正式吞吐，不保证卸载加速，也不保证全部硬件点成功。

主要回传方式沿用服务器 push 到 GitHub、本地 Mac pull。先完成本次所需的所有 resume/retry，再提交结果：提交会改变 SHA，此后原 suite 将拒绝继续恢复。将下面的 `ID` 和 `NNN` 替换成程序打印的本次公开目录；只添加这个明确目录，不使用 `git add .`：

```bash
# 服务器：人工审核该公开目录后回传
git add -- docs/results/decode-mechanism-suite-ID/report-NNN
git commit -m "results: publish experiment suite ID report NNN"
git push origin repro/fluxmoe

# Mac：在本项目 repro/fluxmoe 检出中获取结果
git switch repro/fluxmoe
git pull --ff-only origin repro/fluxmoe
```

程序也打印白名单归档的精确路径，例如 `runs/experiment-suite/ID/public-report-001.tar.gz`。如已有可用 SSH/SCP，可选择在本地直接回传这个文件；将 `YOUR_SERVER` 替换为已有 SSH 主机别名，编号使用程序实际打印的编号：

```bash
scp YOUR_SERVER:/home/jovyan/wangtonghan/moe-flex/runs/experiment-suite/ID/public-report-001.tar.gz ./
```

归档只包含本次明确列出的公开 JSON/CSV/Markdown/SVG，不递归打包 runs、日志或原始 summary。状态文件记录 public_allowlist；可审核后手动上传归档。程序不自动提交、推送或发送文件。

Mac CPU/子进程/壳脚本测试不等于 H100、真实 Docker 或 vLLM EP 已经运行成功。真实 GPU 配置支持情况、吞吐、KV 容量和请求延迟，以服务器保存的验证结果为准。
