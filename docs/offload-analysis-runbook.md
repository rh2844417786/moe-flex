# 四卡专家卸载可行性分析运行手册

本工具判断 CPU 专家权重搬运是否可能被实际 KV 收益抵消。native 基线是实测；trace 是带采样开销的需求观测；回放和成本结果是预测。所有新产物固定 `diagnostic_only=true`、`formal_offload_gain=false`、`deployment_gain_proven=false`。Mac/CPU 单测不是 H100 测量，本轮不执行改写的真实卸载器。

Python 版本遵循项目约束 3.10–3.13。服务器示例的 `python3` 应为受支持版本；Mac 自带的 Python 3.9 不满足条件，可直接使用已有 `.venv/bin/python -S`（仍不加载 site-packages），无须安装新依赖。

## 准备与资源边界

服务器唯一根目录 `/home/jovyan/wangtonghan/moe-flex`；模型默认 `/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct`。四张独占 H100 80GB、TP4、原 BF16 权重和路由。选卡不满足则停止，不杀其他进程，不减小已选工作负载冒充成功。镜像固定构建配置与执行代码 SHA，允许已有 Git 同步和固定 Docker Hub 基础镜像获取，不新增依赖，不下载模型/数据。

```bash
cd /home/jovyan/wangtonghan/moe-flex
git pull --ff-only origin repro/fluxmoe
git status --short
bash scripts/server/build.sh
export GPU_IDS=0,1,2,3
python3 -S src/flexmoe/analysis/cli.py --help
```

每次 GPU 点由 wrapper 先执行容器内四卡 preflight，再启动计时/采样/传输程序；`timeout --signal=TERM --kill-after=30` 在 Docker 内。离线缓存全部放在项目 `build/partial-cache`，`/mnt/public_data` 只读。记录实际每 rank KV、blocks、总显存、当前与峰值；utilization 是 vLLM profiling 设置，不是硬隔离。默认物理余量 2,000,000,000 bytes；无法满足则失败，不能只看 utilization。

每点需要新 ID。`runs/offload-analysis/<id>` 由 runner 自己创建；不要预建该目录。日志在独立 `<id>-launcher` 目录。超时/预检/worker 失败保存 `launcher.json`，不覆盖损坏的 `summary.json`，独立 `smoke.json` 和 worker 已完成样本照常导出。失败退出码保持非零。整个同代码测量序列先执行完，再提交结果；仅 `docs/results/offload-analysis-*` 生成文件豁免 dirty guard，源码变化须先提交、重建镜像。

## 输入与有限扫描

唯一已提交的 `benchmarks/data/sharegpt/qwen3next_1024_requests.jsonl.zst` 中，文件名 1024 指总记录数；包含 1K/2K/3K/4K 各 256 行，不按上下文长度更换文件名。1K 有 254 个唯一输入，其余桶各 256。高并发会重复输入，不能视作代表性服务流量。排除前 32 个唯一校准输入后，1K/其他桶最多 222/224 个唯一输入，重复数与实际 prompt hash 都保留。16K/32K 只能通过 `--synthetic-context` 拼接已有 token，必须标注 synthetic 容量压力样本。

```bash
python3 -S src/flexmoe/analysis/cli.py plan \
  --id-prefix sep08a --output runs/offload-analysis/sep08a-plan.json \
  --seed 20260908 --selected-batch 64 --selected-context 1024 --selected-output 128
```

`plan` 仅写可检查的 JSON 命令清单（argv 和 shell），不执行。首阶段为默认 utilization 0.90/native 的 short=(1024,128)、generation=(1024,2048)、medium=(4096,512)，各并发 16/32/64/128/256/512，共 18 点。逐阶段阅读结果，选少量点进入采样，不启动比例×缓存×负载×模式的全笛卡尔积。`--include-low-budget` 加 0.60 诊断；`--include-long` 加须先检查的 synthetic 16K/32K 点。`--model-path`、`--dataset-path`、`--dataset-manifest` 和 seed、warmup、repetition、超时、调度预算均实际传入；自定义数据仍须通过 manifest 校验。

计划中的 `--model-path`、`--timeout-s`、`--warmups`、`--repetitions`、`--safety-reserve-bytes` 同样传入 transport 阶段；模型路径仅用于该阶段 preflight。计划默认 warmups=1、repetitions=3，所有适用阶段共用，trace 仍固定 repetitions=1。直接运行 transport 可显式选择其他 warmup/repetition 值（下例为 2/5），不受已生成计划影响。

一个正常预算原生点：

```bash
bash scripts/server/run_offload_analysis.sh resident --run-id sep08a-short-b64 \
  --timeout-s 7200 --gpu-memory-utilization 0.90 \
  --batch-size 64 --context-length 1024 --output-length 128 \
  --max-num-seqs 64 --max-num-batched-tokens 4096 --seed 20260908 \
  --warmups 1 --repetitions 3
```

`resident --engine-mode eager` 仅归因诊断；native 失败保留失败，不自动 fallback 到 eager。短/长生成/中长度均以 0.90/native 作为现实预算比较起点。

## 独立校准、相同输入原生点和完整 eval trace

选择扫描中的一个工作负载后，先运行小校准，再让 matched-native 与 evaluation 使用完全相同的排除、offset、seed、长度和请求数。旧扫描未排除校准输入，不能直接与排除后的 eval 配对。

```bash
bash scripts/server/run_offload_analysis.sh trace --run-id sep08a-calibration \
  --batch-size 32 --context-length 1024 --output-length 128 --max-num-seqs 32 \
  --max-num-batched-tokens 4096 --seed 20260908 --gpu-memory-utilization .90 \
  --warmups 1 --repetitions 1 --max-trace-steps 1024

bash scripts/server/run_offload_analysis.sh resident --run-id sep08a-matched-native \
  --batch-size 64 --context-length 1024 --output-length 128 --max-num-seqs 64 \
  --max-num-batched-tokens 4096 --seed 20260908 --gpu-memory-utilization .90 \
  --warmups 1 --repetitions 3 \
  --exclude-trace runs/offload-analysis/sep08a-calibration/trace-rank-0.json.gz

bash scripts/server/run_offload_analysis.sh trace --run-id sep08a-evaluation \
  --batch-size 64 --context-length 1024 --output-length 128 --max-num-seqs 64 \
  --max-num-batched-tokens 4096 --seed 20260908 --gpu-memory-utilization .90 \
  --warmups 1 --repetitions 1 --max-trace-steps 1024 \
  --exclude-trace runs/offload-analysis/sep08a-calibration/trace-rank-0.json.gz
```

trace 强制 eager 且 `timing_eligible=false`，不能将 summary 中受采样污染的吞吐写成 native 实测。四 rank `observed_steps`、`captured_steps`、`missing_layer_events`、`truncated`、`full_workload` 都需检查。长输出可能超过 1024 步；以新 ID 显式增大 `--max-trace-steps` 和必要的 `--trace-budget-bytes`（默认 134217728 bytes/rank），仍须满足显存预算。无完整 step 时 `trace=null`；部分 trace 只报告窗口，不能按整个输出 token 数外推。

## 并行四卡传输观测

先从实际 trace 的 `expert_bytes` 读取 BF16/TP4 shard 大小，下例 1572864 必须与当前模型几何核实；不得凭模型名猜测。每 rank 同时测量 pinned CPU→GPU 传输，1/8/32/128/512 是每次 copy 的专家数。

```bash
bash scripts/server/run_offload_analysis.sh transport --run-id sep08a-transport \
  --timeout-s 7200 --expert-bytes 1572864 --experts-per-batch 1,8,32,128,512 \
  --modes contiguous,fragmented,gather --contention isolated \
  --warmups 2 --repetitions 5 --iterations 1 \
  --memory-cap-bytes 4000000000 --safety-reserve-bytes 2000000000

bash scripts/server/run_offload_analysis.sh transport --run-id sep08a-proxy \
  --expert-bytes 1572864 --experts-per-batch 1,8,32,128,512 \
  --modes contiguous,fragmented,gather --contention gemm-nccl-proxy \
  --warmups 2 --repetitions 5 --iterations 1 --gemm-size 2048
```

`contiguous` 假设可用连续/预打包来源，不包含任意专家子集的额外 packing 或 overfetch。`fragmented` 计每专家 launch；`gather` 包含已测 CPU 准备，只支持 iterations=1。公开 affinity hash 或 unavailable；当前 NUMA/PCIe 拓扑 unavailable，当前 CPU gather 不代表最优实现。

wrapper 的 `--timeout-s` 同时设置容器内外层命令超时和 transport producer 的 NCCL/owned-worker deadline。两者使用同一个选定秒数；容器命令的计时也包含初始化，可能先到期。native/trace 没有传输 producer 的内部 timeout 参数。

代理模式单独保存 copy-only、compute-only、joint wall 及组件。joint wall 已含 GEMM/NCCL，不能再作为额外 copy 税加到 K 时间。核心与 CLI 都返回 `incremental-transfer-calibration`，保留观测、禁止吞吐预测。失败/被杀 worker 每次已完成观测原子 checkpoint；总文件保持 failed、不将部分行冒充完整四 rank 数据。各 mode/contention/完整 transport contract 分组，不跨策略平均。

## 回放、K 参考与分析

以下命令使用 bash 数组明确四 rank 路径；纯标准库部分可在 Mac 离线运行。全部原始文件留本地，不放公开目录。

```bash
trace_paths=()
cal_paths=()
for rank in 0 1 2 3; do
  trace_paths+=("runs/offload-analysis/sep08a-evaluation/trace-rank-${rank}.json.gz")
  cal_paths+=("runs/offload-analysis/sep08a-calibration/trace-rank-${rank}.json.gz")
done
python3 -S src/flexmoe/analysis/cli.py validate "${trace_paths[@]}" \
  runs/offload-analysis/sep08a-matched-native/summary.json \
  runs/offload-analysis/sep08a-transport/samples.json

python3 -S src/flexmoe/analysis/cli.py replay --traces "${trace_paths[@]}" \
  --calibration "${cal_paths[@]}" --offload-fraction .20 --cache-slots 0 \
  --staging-experts 512 --extra-gpu-bytes 0 --policy lru \
  --output runs/offload-analysis/sep08a-offline/replay.json

python3 -S src/flexmoe/analysis/cli.py analyze --traces "${trace_paths[@]}" \
  --replay runs/offload-analysis/sep08a-offline/replay.json \
  --samples runs/offload-analysis/sep08a-transport/samples.json \
  --baseline runs/offload-analysis/sep08a-matched-native/summary.json \
  --output runs/offload-analysis/sep08a-offline/analysis.json
```

无 K 时 `predictions=null`、`matching-kv-reference`，仍保存实际加载字节、每生成 token 字节（仅完整轨迹）、净释放 bytes、每 rank 带宽需求和传输时间。净释放扣除驻留、cache、staging、额外工作区，可为负，不截成零。若 staging 小于一次需求，保留 `staging-overflow-compute-cost`，不能遗漏额外分块计算成本。

检查 replay 净释放与实测 native 剩余显存后，才选择一个**物理安全**的更高 utilization 原生 K 点。沿用 matched-native 的所有模型/数据/input SHA/排除/代码/TP/版本/硬件/长度参数，仅修改明确记录的 utilization，使用新 ID `sep08a-kref`。不预设 .92/.95 一定安全；不能安全测就保持缺 K。K 实际四 rank KV 增量须均为正且不超过各 rank 净释放，不能按容量比例插值吞吐。K 超出 .90 基线预算是 counterfactual，不是同资源真实卸载。

```bash
# 已安全测得并验证 sep08a-kref 后，增加以下参数重跑 analyze：
# --kv-reference runs/offload-analysis/sep08a-kref/summary.json
```

候选回放可逐项选择比例 .05/.10/.20/.30/.50 与 cache 0/64/256/512；每个输出新文件名，先缩小范围再扫描。`--policy decayed-lfu` 为确定性频率衰减；`future` 是已知未来轨迹的理想参考，未部署、不是整个系统严格最优。成本先重新执行 canonical replay 核验保存结果，再按实际测量形状精确分解，不外推未测 shape。串行情景采用逐事件跨 rank 最大值求和，乐观重叠采用 rank 总服务最大值；均非硬件严格上下界。

既有 KV Oracle 先走 `kv_oracle_evidence.validate_run`；缺 hardware/prompt 身份时保存 `oracle-adaptation` 的 over-budget/diagnostic/incomplete 与缺证据项，不补造字段强行配对。失败、缺 K、OOM 不等价于无收益。

## 原始合同与公开交付

唯一 JSON 根 metadata 为 schema_version=1（严格整数）、artifact_kind、三个固定布尔诊断标志；payload 字段直接在根部，不另套通用 data。DTO 的 `to_dict`/`from_dict` 不自行加根标签。

|原始文件|artifact_kind|主要 payload|
|---|---|---|
|summary.json|native-summary|共享 runner summary，engine_mode、timing_eligible、contract、memory、repetitions|
|smoke.json / rep-XXX.json|analysis-smoke / analysis-repetition|共享固定 token 输出统计与标签|
|trace-rank-r.json.gz|demand-trace|trace DTO 或 null、capture、generation_status、timing_eligible=false|
|samples.json|transfer-samples|status、config、contract、samples、measurements、部分 worker 证据|
|replay.json|replay-suite|replays DTO 数组、校准 input hashes|
|analysis.json|analysis-suite|每传输组 analyses，必要时 legacy_reference|
|launcher.json|launcher-failure|失败阶段和退出码；原始文件不覆盖|

native→TimingPoint 检查固定请求/token/秒数/吞吐一致、完整 repetitions、smoke、实际四 rank KV/blocks/物理容量/峰值、输入 hash、resolved policy hash/模式、BF16/TP4/版本与硬件身份。合法 eager 可供诊断，不能当 native 最佳部署基线。validate 非零表示证据不能用于该项分析；export 仍保留失败和独立 smoke。

```bash
python3 -S src/flexmoe/analysis/cli.py export \
  --source runs/offload-analysis/sep08a-offline \
  --output docs/results/offload-analysis-sep08a-offline
```

GPU wrapper 已为每点自动产生独立公开 report；手动重复导出须用新目录。`report.json` 为固定诊断根+白名单 records，`report.csv` 为逐字段 metric/value，`report.md` 为中文说明、覆盖/成本/缺证据表。bytes 与 GB 均按十进制（1 GB=1,000,000,000 bytes）。嵌套字符串只接受固定枚举、hash、受限版本号；不上传原始 prompt/token IDs/logits/UUID/路径/命令/大 trace。公开 run ID 用 hash；未识别原因记计数，详细异常只在本地。

公开回放保留 `per_layer_totals`、固定四类 phase 的 `per_phase_totals`、五字段 `reuse_gap_summary`、经校验的 `calibration_input_hashes`，以及从 resident_experts 推导的 `resident_count_by_layer`/`resident_count_total`（不公开专家 ID）。传输报告保留逐 shape/mode/repetition 的 aggregates、`cpu_affinity_sha256`（不可得为 null）、拓扑 unavailable 状态；合同保留字面 TP4/BF16。回放的部分窗口、未建模分块、未来参考等已知 caveat 枚举在三个公开格式中完整保留。

**只提交导出器生成并复核的 `report.json/report.csv/report.md`**，不要 `git add runs`，不要上传原始 plan/replay/analyze、日志或 trace；即使手动把它们写到 docs/results，也不属于公开白名单。同 SHA 序列完成后，按明确文件路径提交结果并 push `repro/fluxmoe`，报告执行 SHA 与结果提交 SHA。只有后续真实同资源卸载的 correctness、实际搬运、吞吐/TTFT/错误/OOM 对照才可证明部署收益。
