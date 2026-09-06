# 给服务器 Codex CLI 的执行指令

请在已有服务器项目 `/home/jovyan/wangtonghan/moe-flex` 完成第一阶段原生 KV Oracle 容量诊断。先完整阅读 `docs/kv-oracle-runbook.md` 和 `docs/superpowers/specs/2026-09-06-kv-oracle.md`，遵守仓库 AGENTS。已授权同步 `repro/fluxmoe`、构建和执行下面有限实验，不需要再次询问常规步骤；没有四张空闲且明确授权独占的 H100 则停止，不杀他人进程。

只使用已有 Qwen3-Next-80B-A3B-Instruct 原始 BF16 权重、提交在仓库的 ShareGPT 数据和现有依赖。禁止下载模型、数据或新依赖。仅 GitHub 同步及 runbook 中固定 digest 的 Docker Hub 镜像允许联网获取；Docker RUN 和模型/数据操作必须离线。所有输出和临时文件放项目内，公开命令使用 PROJECT_ROOT、MODEL_PATH 变量。

先只读检查 git状态、分支、GPU归属和现有路径。保留服务器已提交结果与旧 expert-cache smoke 导出修复；`git pull --ff-only origin repro/fluxmoe` 后记录执行SHA，运行 `bash scripts/server/build.sh`，验证镜像revision一致。代码修复必须提交后重新构建；修改SHA后用新suite重新测r0，不沿用旧anchor。

使用 `scripts/server/run_kv_oracle.sh`：分别独立ID执行 control、main 的 screen（各1 warmup+1 repetition），先验证原生模型可用并检查报告。只有需要澄清输入/输出长度效应时手动加long；不默认跑无关专家缓存CUDA测试或全矩阵。选一个有意义负载执行3轮forward confirm和3轮reverse confirm；reverse提供同SHA同配置完整r0 anchor，只借实际K0，必须用末尾新r0速度且核对K0一致。

保持完整GPU驻留：r0名义0.60自动KV、small=实际K0+3.2GB/卡、large=实际K0+18.1GB/卡，TP4/eager/level0/prefix-off，固定input/output tokens。K0必须来自完整r0真实分配，不是available profiling预算。O组可能超过名义0.60预算，始终diagnostic_only=true、formal_offload_gain=false；禁止当成公平卸载比较、严格上界或性能承诺。

默认安全fraction0.90、margin2.0GB/卡。检查估算非KV+目标KV+margin，以及初始化/每轮实际GPU边界和Torch峰值。该检查不是硬显存上限，不能捕获全部非Torch瞬态。失败、OOM、超时、安全拒绝均停止后续点并导出，保留完成轮次与smoke，不自动降低KV或输入，不隐藏重试。large失败时保留合法small诊断，但全组判断incomplete。若需要改变安全参数必须明确报告并用新ID，不能静默放宽。

确认root-forward probe只覆盖测量期。报告每rank前向次数、runner.input_batch.num_reqs观测次数/均值/峰值、CUDA参数驻留、KV/current/peak、固定tokens/seconds、输入/输出hash。请求数不是等待队列；没有支持的KV occupancy/preemption/请求指标标unavailable/null，不猜测。不要添加路由ID D2H或逐层GPU同步。

自动公开导出目录 `docs/results/kv-oracle-*` 可以留未提交，启动器只豁免该生成目录；其他脏源码仍阻止启动。所有同SHA点及正反序完成后再提交结果，防止中途提交改变SHA使anchor失效。失败/负结果同样导出，公开仅白名单数字/hash/相对run ID，显存统一十进制GB，不上传原始日志、prompts、权重或机器身份。

最终中文报告须列各组状态、耗时中位数/范围、吞吐比和每轮时间收益空间、实际KV增量、峰值与安全边界、正反序一致性、机制证据/未知字段及工程判断。三轮且范围不重叠才称可重复信号；small>=20%且机制改善才考虑后续小比例卸载，10–20%只建议低成本改动，双方<10%且无机制改善建议停止该负载路线。large单独有收益不证明5层/1槽可行，缺证据不等于无收益。

完成后提交并推送现有分支上的代码修复与人工检查过的脱敏结果，核对远端SHA。先把诊断结果交给用户，**不要自动启动真实卸载，不替换缓存、pool、eviction、路由或CUDA内核**。
