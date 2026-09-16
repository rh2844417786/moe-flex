# 一键并行与卸载实验设计

用户已确认：无需服务器 Codex，以一行命令完成三组全常驻并行方式对比，并继续原有 TP4 覆盖、卸载、KV 实验。本版不移植 EP 卸载。Mac 编码测试，GitHub 交付，H100 运行由用户启动。

## 固定边界

- 已有分支 `repro/fluxmoe`；服务器项目仅 `/home/jovyan/wangtonghan/moe-flex`，已有权重 `/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct` 只读。
- 四张用户显式指定且独占的 H100；TP4 DP1 EP关闭、TP1 DP4 EP4、TP2 DP2 EP4，全部 BF16、全权重GPU常驻、原始路由、正常优化模式。EP不支持/初始化失败要如实失败，禁止静默退回TP或eager。
- 固定现有vLLM0.10.2容器/补丁、已有离线数据，不下载依赖/权重，不升级框架。构建沿用项目脚本；镜像缺失可使用其既有Docker Hub流程，其他安装仍离线。
- 并行比较使用同一份1024唯一输入、1K输入/512输出、seed20260912、warmup1、repetitions3、gpu_memory_utilization0.90。global max_num_seqs1024、max_num_batched_tokens8192按DP数均分，以固定总上限而非每副本放大负载。
- 同时处理全局请求、实际DP分片、DP内TP副本必须区分。全局TPS来自总输出tokens除以父进程同一轮协调开始至所有DP完成的墙钟，禁止相加本地TPS。记录每DP/TP worker实际内存/KV、实际policy和版本；请求延迟可测则记录口径和分位数，不可用则null，禁止伪造TTFT/ITL。

## 功能

1. 原生并行runner使用固定版本官方offline DP流程，spawn DP进程，各DP读取无重复输入的互斥分片，设置VLLM_DP_RANK/LOCAL/SIZE/MASTER_IP/PORT。进程同时运行，warmup及三轮正式生成同步，失败/超时终止仅本次创建的子进程。全局样本必须全部分片输出固定长度且身份一致才有效。允许BF16输出变化，保留哈希审计；公共报告不含token、输出hash、UUID、绝对路径、日志原文。
2. 主机标准库总控入口，无torch依赖，顺序执行三组并行对比和现有decode实验。构建一次、验证数据、校准1K/4K、native、1K覆盖七点、选定4K至多三点、有限卸载采样、A/B/C、采样开销，最终统一汇总。直接复用现有plan/validate/export，不另写卸载算法。
3. 自动决策保守且可审计：只有完整1K覆盖点才进入候选，选最低/中间/最高至多三个代表点；KV小点取TP4 native四卡最小实际KV的一半并按block向下取整，大点不超过该最小已观测容量。保留至少现有2GB物理安全余量及每点检查，不将释放权重全部转换为未经验证的KV预算。无安全证据则跳过依赖阶段并给理由。
4. 每步限时、状态原子保存、日志单独保存。失败不成为零吞吐；无依赖的后续点可继续，预检发现GPU占用/清理失败则安全停止。无需人工再挑参数，不能自动修改代码或盲目反复重试。
5. 支持同SHA/同配置/同GPU显式resume，成功点必须有有效artifact方可复用。新尝试使用新目录，旧结果不覆盖；失败默认保留，可显式retry-failed。并发启动用项目内互斥锁拒绝；路径校验拒绝穿越与symlink逃逸。默认不自动git commit/push；生成白名单结果归档与清晰的回传命令。
6. 拥有的Docker容器必须可识别并可在外层timeout/中断后清理。只对本次CID且归属标签匹配的容器操作，不按名称通配、不kill他人进程；保留旧run_container调用行为。

## 交付与验收

一行命令形态：`cd /home/jovyan/wangtonghan/moe-flex && git pull --ff-only origin repro/fluxmoe && GPU_IDS=0,1,2,3 bash scripts/server/run_all_experiments.sh`。脚本自动生成run ID，输出日志/状态/汇总路径；resume命令携带ID。

本地以真实文件/子进程/壳脚本测试状态、超时、恢复、数据分片、吞吐归约、结果脱敏；GPU与Docker/vLLM仅在外部边界使用测试替身。全量CPU、类型/静态、shell、wheel和独立审核通过后普通push。测试不代表三种配置已在H100跑通；不能保证硬件实验每点成功。
