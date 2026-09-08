# 专家卸载可行性诊断

所有结果仅用于诊断；未证明真实卸载或部署吞吐收益。

单位：bytes；GB = bytes / 1,000,000,000（十进制）。

|记录|类型|状态|
|---|---|---|
|0|launcher-failure|failed|
|1|native-summary|failed|

完整数值见 report.json 与逐字段 CSV。原始轨迹、日志、路径、提示词、UUID、命令不公开。

原生基线为实测，trace 为采样观测，回放成本为预测。代理 joint wall 含 GEMM/NCCL，禁止重复加入 K 时间。未配对 K 或完整轨迹时保留字节/带宽观测，不解释为无收益。

## 记录 0

|指标|值|
|---|---|
|record_index|0|
|status|failed|
|phase|launcher|
|exit_code|1|
|artifact_kind|launcher-failure|

## 记录 1

|指标|值|
|---|---|
|record_index|1|
|contract.batch_size|256|
|contract.commit|2161d80517315fe9835ee3cf32c98f8e6b0343a7|
|contract.context_length|4096|
|contract.dataset_manifest_sha256|d1fa27a78c422ed77b6e44e7c1b642c8cdf3d1a5b1068d7ee0c4996079f8a75b|
|contract.dataset_sha256|ae5e2428733c21f39efdd1c6ba45d9dac78e4f00d43a0a930151034eda178445|
|contract.dtype|bfloat16|
|contract.engine_policy_sha256|77ede705679762971105bf2c740f88c930363f645ab30af8018cd0b56aefc0e9|
|contract.evaluation_pool_count|224|
|contract.gpu_memory_utilization|0.95|
|contract.hardware_sha256|9791dbc6d311596ecffb3eb44b417020cb7e142e79a148bc73feb7a0573b4bd1|
|contract.input_sha256|efbd829639331de5e1da036e49fbb898ed7c09035d00fb1dfcb632aff3eaf4bc|
|contract.max_num_batched_tokens|4096|
|contract.max_num_seqs|256|
|contract.model_config_sha256|2d483c7cabad7c8704478ed4038fa7e7b2eff840bc00a118eccbe38e2b488303|
|contract.model_identity_sha256|f197bd16d71c7da5e1472aacaeb3bd213c41d7688695ac857cf30665fe951a59|
|contract.output_length|512|
|contract.repeated_request_count|32|
|contract.seed|20260908|
|contract.source_request_count|256|
|contract.synthetic|False|
|contract.tensor_parallel_size|4|
|contract.unique_selected_request_count|224|
|contract.versions.cuda|12.8|
|contract.versions.python|3.12.11|
|contract.versions.torch|2.8.0+cu128|
|contract.versions.vllm|0.10.2|
|contract.versions.vllm_commit|01efc7ef781391e744ed08c3292817a773d654e6|
|engine_mode|native|
|geometry.expert_bytes|1572864|
|geometry.num_experts|512|
|geometry.top_k|10|
|geometry.total_layers|48|
|measurement_evidence|measured|
|memory[0].free_gpu_bytes|2483093504|
|memory[0].kv_cache_accounting_consistent|True|
|memory[0].kv_cache_allocated_bytes|38724304896|
|memory[0].kv_cache_declared_bytes|38724304896|
|memory[0].num_gpu_blocks|11586|
|memory[0].rank|0|
|memory[0].torch_allocated_bytes|78815649792|
|memory[0].torch_peak_allocated_bytes|79363344384|
|memory[0].torch_peak_reserved_bytes|79614181376|
|memory[0].torch_reserved_bytes|79614181376|
|memory[0].total_gpu_bytes|85017493504|
|memory[1].free_gpu_bytes|2483093504|
|memory[1].kv_cache_accounting_consistent|True|
|memory[1].kv_cache_allocated_bytes|38724304896|
|memory[1].kv_cache_declared_bytes|38724304896|
|memory[1].num_gpu_blocks|11586|
|memory[1].rank|1|
|memory[1].torch_allocated_bytes|78815649792|
|memory[1].torch_peak_allocated_bytes|79363344384|
|memory[1].torch_peak_reserved_bytes|79614181376|
|memory[1].torch_reserved_bytes|79614181376|
|memory[1].total_gpu_bytes|85017493504|
|memory[2].free_gpu_bytes|2483093504|
|memory[2].kv_cache_accounting_consistent|True|
|memory[2].kv_cache_allocated_bytes|38724304896|
|memory[2].kv_cache_declared_bytes|38724304896|
|memory[2].num_gpu_blocks|11586|
|memory[2].rank|2|
|memory[2].torch_allocated_bytes|78815649792|
|memory[2].torch_peak_allocated_bytes|79363344384|
|memory[2].torch_peak_reserved_bytes|79614181376|
|memory[2].torch_reserved_bytes|79614181376|
|memory[2].total_gpu_bytes|85017493504|
|memory[3].free_gpu_bytes|2483093504|
|memory[3].kv_cache_accounting_consistent|True|
|memory[3].kv_cache_allocated_bytes|38724304896|
|memory[3].kv_cache_declared_bytes|38724304896|
|memory[3].num_gpu_blocks|11586|
|memory[3].rank|3|
|memory[3].torch_allocated_bytes|78815649792|
|memory[3].torch_peak_allocated_bytes|79363344384|
|memory[3].torch_peak_reserved_bytes|79614181376|
|memory[3].torch_reserved_bytes|79614181376|
|memory[3].total_gpu_bytes|85017493504|
|physical_safety_reserve_bytes|2000000000|
|repetitions_completed|0|
|selection_offset|0|
|source_kind|native-measured|
|status|failed|
|timing_eligible|True|
|artifact_kind|native-summary|
|run_identity_sha256|c9565a3439f52269bf2a161c1f01ed2858ddde92db5bb9e9b300fb8718b39e84|

连续拷贝假设权重已连续/预打包；未计任意路由子集的额外打包或 overfetch。fragmented 计每次 copy launch；gather 计当前 CPU 准备，受 CPU/affinity/拓扑限制。现有数据仅 1K–4K；重复与 synthetic 长上下文仅为压力样本。
