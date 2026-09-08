# 专家卸载可行性诊断

所有结果仅用于诊断；未证明真实卸载或部署吞吐收益。

单位：bytes；GB = bytes / 1,000,000,000（十进制）。

|记录|类型|状态|
|---|---|---|
|0|analysis-repetition|unknown|
|1|analysis-repetition|unknown|
|2|analysis-repetition|unknown|
|3|analysis-smoke|unknown|
|4|unknown|unreadable|

完整数值见 report.json 与逐字段 CSV。原始轨迹、日志、路径、提示词、UUID、命令不公开。

原生基线为实测，trace 为采样观测，回放成本为预测。代理 joint wall 含 GEMM/NCCL，禁止重复加入 K 时间。未配对 K 或完整轨迹时保留字节/带宽观测，不解释为无收益。

## 记录 0

|指标|值|
|---|---|
|record_index|0|
|elapsed_s|90.02796962996945|
|generated_tokens|262144|
|memory[0].free_gpu_bytes|3412131840|
|memory[0].kv_cache_accounting_consistent|True|
|memory[0].kv_cache_allocated_bytes|33924710400|
|memory[0].kv_cache_declared_bytes|33924710400|
|memory[0].num_gpu_blocks|10150|
|memory[0].rank|0|
|memory[0].torch_allocated_bytes|74010250752|
|memory[0].torch_peak_allocated_bytes|74729128448|
|memory[0].torch_peak_reserved_bytes|75537317888|
|memory[0].torch_reserved_bytes|75537317888|
|memory[0].total_gpu_bytes|85017493504|
|memory[1].free_gpu_bytes|3410034688|
|memory[1].kv_cache_accounting_consistent|True|
|memory[1].kv_cache_allocated_bytes|33924710400|
|memory[1].kv_cache_declared_bytes|33924710400|
|memory[1].num_gpu_blocks|10150|
|memory[1].rank|1|
|memory[1].torch_allocated_bytes|74010250752|
|memory[1].torch_peak_allocated_bytes|74729128448|
|memory[1].torch_peak_reserved_bytes|75537317888|
|memory[1].torch_reserved_bytes|75537317888|
|memory[1].total_gpu_bytes|85017493504|
|memory[2].free_gpu_bytes|3412131840|
|memory[2].kv_cache_accounting_consistent|True|
|memory[2].kv_cache_allocated_bytes|33924710400|
|memory[2].kv_cache_declared_bytes|33924710400|
|memory[2].num_gpu_blocks|10150|
|memory[2].rank|2|
|memory[2].torch_allocated_bytes|74010250752|
|memory[2].torch_peak_allocated_bytes|74729128448|
|memory[2].torch_peak_reserved_bytes|75537317888|
|memory[2].torch_reserved_bytes|75537317888|
|memory[2].total_gpu_bytes|85017493504|
|memory[3].free_gpu_bytes|3412131840|
|memory[3].kv_cache_accounting_consistent|True|
|memory[3].kv_cache_allocated_bytes|33924710400|
|memory[3].kv_cache_declared_bytes|33924710400|
|memory[3].num_gpu_blocks|10150|
|memory[3].rank|3|
|memory[3].torch_allocated_bytes|74010250752|
|memory[3].torch_peak_allocated_bytes|74729128448|
|memory[3].torch_peak_reserved_bytes|75537317888|
|memory[3].torch_reserved_bytes|75537317888|
|memory[3].total_gpu_bytes|85017493504|
|output_tokens_per_second|2911.8061984231927|
|repetition|0|
|request_count|512|
|timing_eligible|True|
|artifact_kind|analysis-repetition|

## 记录 1

|指标|值|
|---|---|
|record_index|1|
|elapsed_s|89.89740017615259|
|generated_tokens|262144|
|memory[0].free_gpu_bytes|3412131840|
|memory[0].kv_cache_accounting_consistent|True|
|memory[0].kv_cache_allocated_bytes|33924710400|
|memory[0].kv_cache_declared_bytes|33924710400|
|memory[0].num_gpu_blocks|10150|
|memory[0].rank|0|
|memory[0].torch_allocated_bytes|74010250752|
|memory[0].torch_peak_allocated_bytes|74729128448|
|memory[0].torch_peak_reserved_bytes|75537317888|
|memory[0].torch_reserved_bytes|75537317888|
|memory[0].total_gpu_bytes|85017493504|
|memory[1].free_gpu_bytes|3410034688|
|memory[1].kv_cache_accounting_consistent|True|
|memory[1].kv_cache_allocated_bytes|33924710400|
|memory[1].kv_cache_declared_bytes|33924710400|
|memory[1].num_gpu_blocks|10150|
|memory[1].rank|1|
|memory[1].torch_allocated_bytes|74010250752|
|memory[1].torch_peak_allocated_bytes|74729128448|
|memory[1].torch_peak_reserved_bytes|75537317888|
|memory[1].torch_reserved_bytes|75537317888|
|memory[1].total_gpu_bytes|85017493504|
|memory[2].free_gpu_bytes|3412131840|
|memory[2].kv_cache_accounting_consistent|True|
|memory[2].kv_cache_allocated_bytes|33924710400|
|memory[2].kv_cache_declared_bytes|33924710400|
|memory[2].num_gpu_blocks|10150|
|memory[2].rank|2|
|memory[2].torch_allocated_bytes|74010250752|
|memory[2].torch_peak_allocated_bytes|74729128448|
|memory[2].torch_peak_reserved_bytes|75537317888|
|memory[2].torch_reserved_bytes|75537317888|
|memory[2].total_gpu_bytes|85017493504|
|memory[3].free_gpu_bytes|3412131840|
|memory[3].kv_cache_accounting_consistent|True|
|memory[3].kv_cache_allocated_bytes|33924710400|
|memory[3].kv_cache_declared_bytes|33924710400|
|memory[3].num_gpu_blocks|10150|
|memory[3].rank|3|
|memory[3].torch_allocated_bytes|74010250752|
|memory[3].torch_peak_allocated_bytes|74729128448|
|memory[3].torch_peak_reserved_bytes|75537317888|
|memory[3].torch_reserved_bytes|75537317888|
|memory[3].total_gpu_bytes|85017493504|
|output_tokens_per_second|2916.0353857434457|
|repetition|1|
|request_count|512|
|timing_eligible|True|
|artifact_kind|analysis-repetition|

## 记录 2

|指标|值|
|---|---|
|record_index|2|
|elapsed_s|89.96918665803969|
|generated_tokens|262144|
|memory[0].free_gpu_bytes|3412131840|
|memory[0].kv_cache_accounting_consistent|True|
|memory[0].kv_cache_allocated_bytes|33924710400|
|memory[0].kv_cache_declared_bytes|33924710400|
|memory[0].num_gpu_blocks|10150|
|memory[0].rank|0|
|memory[0].torch_allocated_bytes|74010250752|
|memory[0].torch_peak_allocated_bytes|74729128448|
|memory[0].torch_peak_reserved_bytes|75537317888|
|memory[0].torch_reserved_bytes|75537317888|
|memory[0].total_gpu_bytes|85017493504|
|memory[1].free_gpu_bytes|3410034688|
|memory[1].kv_cache_accounting_consistent|True|
|memory[1].kv_cache_allocated_bytes|33924710400|
|memory[1].kv_cache_declared_bytes|33924710400|
|memory[1].num_gpu_blocks|10150|
|memory[1].rank|1|
|memory[1].torch_allocated_bytes|74010250752|
|memory[1].torch_peak_allocated_bytes|74729128448|
|memory[1].torch_peak_reserved_bytes|75537317888|
|memory[1].torch_reserved_bytes|75537317888|
|memory[1].total_gpu_bytes|85017493504|
|memory[2].free_gpu_bytes|3412131840|
|memory[2].kv_cache_accounting_consistent|True|
|memory[2].kv_cache_allocated_bytes|33924710400|
|memory[2].kv_cache_declared_bytes|33924710400|
|memory[2].num_gpu_blocks|10150|
|memory[2].rank|2|
|memory[2].torch_allocated_bytes|74010250752|
|memory[2].torch_peak_allocated_bytes|74729128448|
|memory[2].torch_peak_reserved_bytes|75537317888|
|memory[2].torch_reserved_bytes|75537317888|
|memory[2].total_gpu_bytes|85017493504|
|memory[3].free_gpu_bytes|3412131840|
|memory[3].kv_cache_accounting_consistent|True|
|memory[3].kv_cache_allocated_bytes|33924710400|
|memory[3].kv_cache_declared_bytes|33924710400|
|memory[3].num_gpu_blocks|10150|
|memory[3].rank|3|
|memory[3].torch_allocated_bytes|74010250752|
|memory[3].torch_peak_allocated_bytes|74729128448|
|memory[3].torch_peak_reserved_bytes|75537317888|
|memory[3].torch_reserved_bytes|75537317888|
|memory[3].total_gpu_bytes|85017493504|
|output_tokens_per_second|2913.7086789099553|
|repetition|2|
|request_count|512|
|timing_eligible|True|
|artifact_kind|analysis-repetition|

## 记录 3

|指标|值|
|---|---|
|record_index|3|
|elapsed_s|14.867260031867772|
|generated_tokens|8|
|input_sha256|f9ef3113fd157e161e0b373cbc8d824dfc6f5beb54b87dbda290a8ad54e69419|
|output_tokens_per_second|0.5380951152298479|
|request_count|1|
|timing_eligible|True|
|artifact_kind|analysis-smoke|

## 记录 4

|指标|值|
|---|---|
|record_index|4|
|status|unreadable|
|artifact_kind|unknown|

连续拷贝假设权重已连续/预打包；未计任意路由子集的额外打包或 overfetch。fragmented 计每次 copy launch；gather 计当前 CPU 准备，受 CPU/affinity/拓扑限制。现有数据仅 1K–4K；重复与 synthetic 长上下文仅为压力样本。
