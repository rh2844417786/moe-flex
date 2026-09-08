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
|elapsed_s|46.48389980988577|
|generated_tokens|131072|
|memory[0].free_gpu_bytes|3527475200|
|memory[0].kv_cache_accounting_consistent|True|
|memory[0].kv_cache_allocated_bytes|34472853504|
|memory[0].kv_cache_declared_bytes|34472853504|
|memory[0].num_gpu_blocks|10314|
|memory[0].rank|0|
|memory[0].torch_allocated_bytes|74562651136|
|memory[0].torch_peak_allocated_bytes|75044282368|
|memory[0].torch_peak_reserved_bytes|75629592576|
|memory[0].torch_reserved_bytes|75629592576|
|memory[0].total_gpu_bytes|85017493504|
|memory[1].free_gpu_bytes|3527475200|
|memory[1].kv_cache_accounting_consistent|True|
|memory[1].kv_cache_allocated_bytes|34472853504|
|memory[1].kv_cache_declared_bytes|34472853504|
|memory[1].num_gpu_blocks|10314|
|memory[1].rank|1|
|memory[1].torch_allocated_bytes|74562651136|
|memory[1].torch_peak_allocated_bytes|75044282368|
|memory[1].torch_peak_reserved_bytes|75629592576|
|memory[1].torch_reserved_bytes|75629592576|
|memory[1].total_gpu_bytes|85017493504|
|memory[2].free_gpu_bytes|3527475200|
|memory[2].kv_cache_accounting_consistent|True|
|memory[2].kv_cache_allocated_bytes|34472853504|
|memory[2].kv_cache_declared_bytes|34472853504|
|memory[2].num_gpu_blocks|10314|
|memory[2].rank|2|
|memory[2].torch_allocated_bytes|74562651136|
|memory[2].torch_peak_allocated_bytes|75044282368|
|memory[2].torch_peak_reserved_bytes|75629592576|
|memory[2].torch_reserved_bytes|75629592576|
|memory[2].total_gpu_bytes|85017493504|
|memory[3].free_gpu_bytes|3527475200|
|memory[3].kv_cache_accounting_consistent|True|
|memory[3].kv_cache_allocated_bytes|34472853504|
|memory[3].kv_cache_declared_bytes|34472853504|
|memory[3].num_gpu_blocks|10314|
|memory[3].rank|3|
|memory[3].torch_allocated_bytes|74562651136|
|memory[3].torch_peak_allocated_bytes|75044282368|
|memory[3].torch_peak_reserved_bytes|75629592576|
|memory[3].torch_reserved_bytes|75629592576|
|memory[3].total_gpu_bytes|85017493504|
|output_tokens_per_second|2819.728992964674|
|repetition|0|
|request_count|256|
|timing_eligible|True|
|artifact_kind|analysis-repetition|

## 记录 1

|指标|值|
|---|---|
|record_index|1|
|elapsed_s|46.814905245788395|
|generated_tokens|131072|
|memory[0].free_gpu_bytes|3527475200|
|memory[0].kv_cache_accounting_consistent|True|
|memory[0].kv_cache_allocated_bytes|34472853504|
|memory[0].kv_cache_declared_bytes|34472853504|
|memory[0].num_gpu_blocks|10314|
|memory[0].rank|0|
|memory[0].torch_allocated_bytes|74562651136|
|memory[0].torch_peak_allocated_bytes|75044282368|
|memory[0].torch_peak_reserved_bytes|75629592576|
|memory[0].torch_reserved_bytes|75629592576|
|memory[0].total_gpu_bytes|85017493504|
|memory[1].free_gpu_bytes|3527475200|
|memory[1].kv_cache_accounting_consistent|True|
|memory[1].kv_cache_allocated_bytes|34472853504|
|memory[1].kv_cache_declared_bytes|34472853504|
|memory[1].num_gpu_blocks|10314|
|memory[1].rank|1|
|memory[1].torch_allocated_bytes|74562651136|
|memory[1].torch_peak_allocated_bytes|75044282368|
|memory[1].torch_peak_reserved_bytes|75629592576|
|memory[1].torch_reserved_bytes|75629592576|
|memory[1].total_gpu_bytes|85017493504|
|memory[2].free_gpu_bytes|3527475200|
|memory[2].kv_cache_accounting_consistent|True|
|memory[2].kv_cache_allocated_bytes|34472853504|
|memory[2].kv_cache_declared_bytes|34472853504|
|memory[2].num_gpu_blocks|10314|
|memory[2].rank|2|
|memory[2].torch_allocated_bytes|74562651136|
|memory[2].torch_peak_allocated_bytes|75044282368|
|memory[2].torch_peak_reserved_bytes|75629592576|
|memory[2].torch_reserved_bytes|75629592576|
|memory[2].total_gpu_bytes|85017493504|
|memory[3].free_gpu_bytes|3527475200|
|memory[3].kv_cache_accounting_consistent|True|
|memory[3].kv_cache_allocated_bytes|34472853504|
|memory[3].kv_cache_declared_bytes|34472853504|
|memory[3].num_gpu_blocks|10314|
|memory[3].rank|3|
|memory[3].torch_allocated_bytes|74562651136|
|memory[3].torch_peak_allocated_bytes|75044282368|
|memory[3].torch_peak_reserved_bytes|75629592576|
|memory[3].torch_reserved_bytes|75629592576|
|memory[3].total_gpu_bytes|85017493504|
|output_tokens_per_second|2799.792060068126|
|repetition|1|
|request_count|256|
|timing_eligible|True|
|artifact_kind|analysis-repetition|

## 记录 2

|指标|值|
|---|---|
|record_index|2|
|elapsed_s|46.89099462190643|
|generated_tokens|131072|
|memory[0].free_gpu_bytes|3527475200|
|memory[0].kv_cache_accounting_consistent|True|
|memory[0].kv_cache_allocated_bytes|34472853504|
|memory[0].kv_cache_declared_bytes|34472853504|
|memory[0].num_gpu_blocks|10314|
|memory[0].rank|0|
|memory[0].torch_allocated_bytes|74562651136|
|memory[0].torch_peak_allocated_bytes|75044282368|
|memory[0].torch_peak_reserved_bytes|75629592576|
|memory[0].torch_reserved_bytes|75629592576|
|memory[0].total_gpu_bytes|85017493504|
|memory[1].free_gpu_bytes|3527475200|
|memory[1].kv_cache_accounting_consistent|True|
|memory[1].kv_cache_allocated_bytes|34472853504|
|memory[1].kv_cache_declared_bytes|34472853504|
|memory[1].num_gpu_blocks|10314|
|memory[1].rank|1|
|memory[1].torch_allocated_bytes|74562651136|
|memory[1].torch_peak_allocated_bytes|75044282368|
|memory[1].torch_peak_reserved_bytes|75629592576|
|memory[1].torch_reserved_bytes|75629592576|
|memory[1].total_gpu_bytes|85017493504|
|memory[2].free_gpu_bytes|3527475200|
|memory[2].kv_cache_accounting_consistent|True|
|memory[2].kv_cache_allocated_bytes|34472853504|
|memory[2].kv_cache_declared_bytes|34472853504|
|memory[2].num_gpu_blocks|10314|
|memory[2].rank|2|
|memory[2].torch_allocated_bytes|74562651136|
|memory[2].torch_peak_allocated_bytes|75044282368|
|memory[2].torch_peak_reserved_bytes|75629592576|
|memory[2].torch_reserved_bytes|75629592576|
|memory[2].total_gpu_bytes|85017493504|
|memory[3].free_gpu_bytes|3527475200|
|memory[3].kv_cache_accounting_consistent|True|
|memory[3].kv_cache_allocated_bytes|34472853504|
|memory[3].kv_cache_declared_bytes|34472853504|
|memory[3].num_gpu_blocks|10314|
|memory[3].rank|3|
|memory[3].torch_allocated_bytes|74562651136|
|memory[3].torch_peak_allocated_bytes|75044282368|
|memory[3].torch_peak_reserved_bytes|75629592576|
|memory[3].torch_reserved_bytes|75629592576|
|memory[3].total_gpu_bytes|85017493504|
|output_tokens_per_second|2795.2488757567553|
|repetition|2|
|request_count|256|
|timing_eligible|True|
|artifact_kind|analysis-repetition|

## 记录 3

|指标|值|
|---|---|
|record_index|3|
|elapsed_s|15.59401024878025|
|generated_tokens|8|
|input_sha256|35cab39063c02de054fab846c35c0bab5b08987ec62125418dad8c1c23173583|
|output_tokens_per_second|0.5130174902011336|
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
