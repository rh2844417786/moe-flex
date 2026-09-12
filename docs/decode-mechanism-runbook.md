# 解码机制实验：分阶段服务器操作

本轮只执行已批准的真实解码观察、现有 expert-cache 后端和显式 KV 对照。Mac 的 CPU fixture 只验证接口，不能作为 H100 曲线或科学结果。历史 `runs`、旧数据包和正式分析入口不变。

## 固定环境与数据

服务器目录只能是 `/home/jovyan/wangtonghan/moe-flex`；模型只读取已有 `/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct`。使用四张独占 H100、TP4、BF16、原始路由。每个新 engine 都先执行独占检查；不终止其他人的进程。包装脚本复用既有 image SHA 检查、项目内缓存和离线环境，`timeout` 在 Docker 内。禁止服务器下载新依赖、模型或 tokenizer；所有写入留在项目内。

```bash
cd /home/jovyan/wangtonghan/moe-flex
git switch repro/fluxmoe
git pull --ff-only origin repro/fluxmoe
git rev-parse HEAD
bash scripts/server/build.sh
```

同一组实验期间保持这个 SHA，不修代码后接着混跑。需要修复时，先保存失败证据，提交、重建，使用新 ID 重新组成同 SHA 对照。只允许 `docs/results/decode-mechanism-*` 新生成报告不影响干净源码检查。

实际提交的数据是：

- `benchmarks/data/decode-mechanism/qwen3next_unique_1536x_1k4k.jsonl.zst`
- `benchmarks/data/decode-mechanism/dataset_manifest.json`

数据 SHA-256 为 `aace2e1afe7b2566a6cf424f97f69d10cf6e4c9cff8c0432a0df6da94469b198`，3,072 条，1K/4K 各 1,536 个唯一完整 token 输入。每桶前 32 条单独校准，随后 1,504 条用于测试；1,000 和 1,024 请求都无需循环。不同长度分别校验；这些是带来源的打包压力输入，不声称自然独立会话或线上代表性。

```bash
GPU_IDS=0,1,2,3 bash scripts/server/run_container.sh python3 -c 'from pathlib import Path; from flexmoe.datasets.decode_corpus import verify_decode_corpus; verify_decode_corpus(Path("benchmarks/data/decode-mechanism/qwen3next_unique_1536x_1k4k.jsonl.zst"), Path("benchmarks/data/decode-mechanism/dataset_manifest.json")); print("unique corpus verified")'
```

## 1. 校准和正常 native 参考

先选择一个从未使用的实验前缀。下列前缀只执行一次；重跑需整体更换。`plan` 只打印有限 `commands[].argv` 和可逐条执行的 `shell`，不会自动跑后续阶段。默认计划参数可通过 `--model-path`、`--dataset-path`、`--dataset-manifest`、`--profile-path`、`--output-length`、`--repetitions`、`--seed`、共同调度上限和 `--timeout-s` 完整传入；包含空格的路径保留为单个参数。

```bash
python3 -S src/flexmoe/analysis/decode_suite.py plan --stage baseline --run-id dm-20260912
```

按打印顺序执行三条命令：1K 校准、4K 校准、1K native。它们分别写入 `runs/decode-mechanism/dm-20260912-cal-1024`、`dm-20260912-cal-4096`、`dm-20260912-native-1024`。两个校准 profile 均位于各自目录的 `profile.json`，绝不能交换长度。校准复用既有 native producer，输出不是正式吞吐证据。

默认 native 是正常优化模式、预算 0.90，包含低开销实际 batch/scheduler 观察。native 不支持 `--profile`，该组合显式失败；详细路由记录必须用 matched-resident/eager 或 offload/eager。0.60/eager 只能作为明确的诊断对照。

```bash
python3 -S src/flexmoe/analysis/decode_suite.py validate --source runs/decode-mechanism/dm-20260912-native-1024
python3 -S src/flexmoe/analysis/decode_suite.py summarize --source runs/decode-mechanism/dm-20260912-native-1024
```

## 2. 实际 decode batch 覆盖

```bash
python3 -S src/flexmoe/analysis/decode_suite.py plan --stage coverage --run-id dm-20260912
```

逐条执行 1/16/50/100/200/500/1000 七个 matched-resident `--profile --target-batch` 点。每点默认采集 256 个匹配纯 decode 步，最少 64 个完整层网格；详细窗口满后生成继续。提交请求数不等于实际 batch；看 `worker_observations.decode_batch_step_counts`、`coverage_status`、phase 和 skipped 计数。`unreached`、缺层、采集不足都保留，不能按提交数量重标或记为零吞吐。

4K 只补最多三个已选择代表点：用实际1K结果决定参数，再显式提供 `--context-length 4096 --batches`。例如以下命令读取操作者选择，不预设高 batch 可达：

```bash
read -r -p '输入最多三个已选择的4K实际batch目标（空格分隔）: ' -a DECODE_BATCHES
python3 -S src/flexmoe/analysis/decode_suite.py plan --stage coverage --run-id dm-20260912 --context-length 4096 --batches "${DECODE_BATCHES[@]}"
```

直方图是 token→expert 选择次数；`token_count` 是每层采样窗口处理行数，`selection_count = token_count * top_k`。`selection_probability` 是选择边占比，其和为1；若研究每 token 选择概率，用原始 histogram 除以 token_count，其和为 top_k。TP 四 rank 路由逻辑复制，不能把计数乘4。共享专家不在 routed-expert 覆盖中。

## 3. 有限实际卸载点与显式 KV 阶段

先检查 native 和 matched-resident 原始 `memory` / `final_memory` 四行、`actual_kv`、physical free 和 Torch peak。`allocated_bytes_per_rank` 是四 rank 一致的单个标量，`requested_bytes` 可能因 block rounding 稍大；不能用利用率乘显存估算成实际 KV。每卡默认保留 2,000,000,000 bytes；非 Torch 瞬态峰值明确不可用。

选择共同可运行的小 KV 点后，先跑低/中/最高可达 batch 的有限卸载诊断；计划最多接受三个选定 batch。三组实际配置为 resident_ratio0.8/cache512、0.8/cache2048、0.9/cache512，policy=decayed-lfu。不同配置都重新检查真实 pool bytes 和 KV，不从比例臆算释放量。

```bash
read -r -p '输入已检查过的小KV预算bytes: ' DECODE_KV_SMALL
read -r -p '输入最多三个已达到的低/中/高batch（空格分隔）: ' -a DECODE_BATCHES
python3 -S src/flexmoe/analysis/decode_suite.py plan --stage offload --run-id dm-20260912 --kv-bytes "$DECODE_KV_SMALL" --batches "${DECODE_BATCHES[@]}"
```

当前路径为按需 H2D、同计算流串行服务；没有猜测性预取。pool 的 first-load/reload 按整个 engine 生命周期计，包括 prefill/warmup。缓存全命中可以有零专家 payload H2D；它不等于卸载实现不存在。payload/metadata/compute/promotion、CPU route D2H/reuse/gather/enqueue 分开看。CUDA load 服务、CPU 等待和各 rank 时间不可相加成 TP 净损失；净开销来自独立整批墙钟对照。

随后选择一个安全的更大预算作为下一阶段输入，不在预检前猜定。若预算不可行、发生 OOM 或无安全点，保存失败，不强求曲线。

```bash
read -r -p '输入完成安全检查后的更大KV预算bytes: ' DECODE_KV_LARGE
python3 -S src/flexmoe/analysis/decode_suite.py plan --stage kv --run-id dm-20260912 --kv-bytes "$DECODE_KV_SMALL" "$DECODE_KV_LARGE"
```

此阶段 A=matched-resident 小KV，B=offload 同KV，C=同B策略大KV。默认总工作量固定为1,024个唯一请求、输出512、重复3次、max_num_seqs1024、max_num_batched_tokens8192。A/B同实际KV与执行策略；B/C冻结cache/profile/输入顺序/seed/调度上限，仅改变实际KV。KV研究没有 `--target-batch`，不能把它和 batch覆盖试验混为一项。

采集开销单独配对，使用同KV和同配置的一次普通运行与一次 instrumented 运行：

```bash
python3 -S src/flexmoe/analysis/decode_suite.py plan --stage overhead --run-id dm-20260912 --kv-bytes "$DECODE_KV_SMALL"
```

仅用于检查采样扰动。profile elapsed/sample_tokens_s 是诊断时间，永不参与主吞吐比较。所有正式 elapsed 都包含 prefill；本地 model-step span 不是整批 elapsed。

## 4. 比较、导出和回传

以下变量由前一阶段的真实输入值决定，路径与生成 argv 一致：

```bash
DECODE_A="runs/decode-mechanism/dm-20260912-a-1024-${DECODE_KV_SMALL}"
DECODE_B="runs/decode-mechanism/dm-20260912-bc-1024-${DECODE_KV_SMALL}"
DECODE_C="runs/decode-mechanism/dm-20260912-bc-1024-${DECODE_KV_LARGE}"
python3 -S src/flexmoe/analysis/decode_suite.py compare --a "$DECODE_A" --b "$DECODE_B" --c "$DECODE_C" --native runs/decode-mechanism/dm-20260912-native-1024
python3 -S src/flexmoe/analysis/decode_suite.py export --source "$DECODE_A" --b "$DECODE_B" --c "$DECODE_C" --native runs/decode-mechanism/dm-20260912-native-1024 --output docs/results/decode-mechanism-dm-20260912-comparison
```

比较严格要求实际身份、vLLM commit、固定输出、四rank显存、同执行策略、完整三次重复和成功采集。缺失 version 或采集最终化失败时，即使 generation_status=complete，也返回 ineligible；数值 rejected 样本和 smoke 仍保存。`offload_overhead_s` 是实现总开销，不叫纯 miss 时间；native0.90 独立参考。native 与 A 的实际KV不同，则 native→matched 时间差明确带KV混杂，不报告纯 engine mode tax。B/C输出ratio只适用于本工作负载，不证明所有部署收益。

“固定输出”指每请求固定token数量；BF16多请求的整批输出hash在重复间变化只记为 `output_variation=varied` 审计，不抹掉有效测速。原始hash留在本地，公开只保存变化状态和计数，smoke一致性仍严格检查。比较JSON的 `arms` 保留A/B/C及可选native每次重复的四rank实际batch分布、phase/步骤分母，以及独立frontend的running/waiting、KV usage、preemptions和samples/status。`admission_evidence` 单独标B→C批量增加、未观察到增加或不足，以及scheduler/因果解释指标是否齐全；Markdown给出rank0一份逻辑计数的简表。未增batch或指标缺失不会使有效墙钟对照变成零吞吐，也不能把观察到的批量变化解释成全部时间收益的原因。

每次 GPU 包装调用自动保存原始 `runs/decode-mechanism/<id>`、独立 `<id>-launcher` 日志，并导出该点的公开目录。公开白名单只有 `report.json`、`report.csv`、`report.md`，有合格实际数据时另有 `kv-throughput.svg`、`batch-coverage.svg`、`miss-service.svg`。不会为缺失点补零、补假曲线。scheduler KV usage 是block使用比例采样，未换算精确 occupied bytes；mean是样本均值，frontend样本不和worker步骤强制对齐。

原始 prompt/token IDs、UUID、绝对路径、日志和输出哈希不进入公开报告。原始失败结果留在服务器；不要清理 `failed-rep-*`、部分 gzip 或独立 smoke。仅在整组结束后提交真实生成的明确目录：

```bash
git add -- docs/results/decode-mechanism-dm-20260912-comparison
git diff --cached --stat
git commit -m 'results: record measured decode mechanism comparison'
git push origin repro/fluxmoe
```

覆盖/缺失点报告需要回传时，逐个添加对应 `docs/results/decode-mechanism-<实际ID>`，不要添加整个 `runs` 或 launcher 日志。仅Mac单元测试通过时，交接说明必须写“尚未执行H100实验”。
