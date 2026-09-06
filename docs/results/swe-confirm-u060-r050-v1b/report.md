# BF16 热专家常驻与共享缓存离线结果

Suite 状态：finished；CUDA 前置检查：passed。
Suite 执行 SHA：4e62dcc0a2ce74dadc56f32f2e0efb820ce7048d。

R=原生 resident；B=专家缓存且实际 KV 等于 R；C=相同专家缓存、相同显存利用率下自动 KV。
实际吞吐由固定输出 token 数 / 测得秒数核对；未完成或损坏结果不产生增益结论。容量采用十进制 GB。

## 01-resident：complete
输出 token/s 中位数 255.803，范围 [255.803, 255.803]。
各 rank 实际 KV 合计 35.923427 GB。
常驻比例 0.5，共享 cache slots=256，policy=decayed-lfu，profile SHA=e102f6959c1e9a64d67294b1800b9f093fb41907a0ca4aa42aa030b7c2dad3fb。
执行 SHA：4e62dcc0a2ce74dadc56f32f2e0efb820ce7048d。以下为完整 R/B/C 的脱敏重跑模板，私有路径需在服务器设定变量：

```bash
bash scripts/server/run_expert_cache.sh confirm --suite-id "$NEW_SUITE_ID" --model-path "$MODEL_PATH" --profile-path "$PROFILE_PATH" --dataset-path "$DATASET_PATH" --dataset-manifest "$DATASET_MANIFEST" --batch-size 32 --context-length 4096 --output-length 64 --gpu-memory-utilization 0.6 --warmups 0 --seed 20260905 --max-num-seqs 32 --max-num-batched-tokens 8192 --timing-samples 128 --repetitions 1 --resident-ratio 0.5 --cache-slots 256 --cache-policy decayed-lfu --calibration-count 32
```

第 0 轮：H2D=0 bytes；resident hits=0；cold cache hit ratio=None。

<details><summary>逐 rank / 逐层数值、kernel 选择与显存证据（bytes）</summary>

```json
{
  "memory": [
    {
      "rank": 0,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 32923254784,
      "torch_allocated_bytes": 49011060736,
      "torch_reserved_bytes": 49977229312,
      "torch_peak_allocated_bytes": 49578542592,
      "torch_peak_reserved_bytes": 49977229312,
      "available_kv_cache_bytes": null,
      "model_memory_bytes": 39959379968,
      "kv_cache_allocated_bytes": 8980856832,
      "kv_cache_declared_bytes": 8980856832,
      "num_gpu_blocks": 2687,
      "kv_cache_accounting_consistent": true
    },
    {
      "rank": 1,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 32923254784,
      "torch_allocated_bytes": 49011060736,
      "torch_reserved_bytes": 49977229312,
      "torch_peak_allocated_bytes": 49578542592,
      "torch_peak_reserved_bytes": 49977229312,
      "available_kv_cache_bytes": null,
      "model_memory_bytes": 39959379968,
      "kv_cache_allocated_bytes": 8980856832,
      "kv_cache_declared_bytes": 8980856832,
      "num_gpu_blocks": 2687,
      "kv_cache_accounting_consistent": true
    },
    {
      "rank": 2,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 32923254784,
      "torch_allocated_bytes": 49011060736,
      "torch_reserved_bytes": 49977229312,
      "torch_peak_allocated_bytes": 49578542592,
      "torch_peak_reserved_bytes": 49977229312,
      "available_kv_cache_bytes": null,
      "model_memory_bytes": 39959379968,
      "kv_cache_allocated_bytes": 8980856832,
      "kv_cache_declared_bytes": 8980856832,
      "num_gpu_blocks": 2687,
      "kv_cache_accounting_consistent": true
    },
    {
      "rank": 3,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 32923254784,
      "torch_allocated_bytes": 49011060736,
      "torch_reserved_bytes": 49977229312,
      "torch_peak_allocated_bytes": 49578542592,
      "torch_peak_reserved_bytes": 49977229312,
      "available_kv_cache_bytes": null,
      "model_memory_bytes": 39959379968,
      "kv_cache_allocated_bytes": 8980856832,
      "kv_cache_declared_bytes": 8980856832,
      "num_gpu_blocks": 2687,
      "kv_cache_accounting_consistent": true
    }
  ],
  "final_memory": [
    {
      "rank": 0,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 29983047680,
      "torch_allocated_bytes": 49011074560,
      "torch_reserved_bytes": 49977229312,
      "torch_peak_allocated_bytes": 49578562560,
      "torch_peak_reserved_bytes": 49977229312,
      "available_kv_cache_bytes": null,
      "model_memory_bytes": 39959379968,
      "kv_cache_allocated_bytes": 8980856832,
      "kv_cache_declared_bytes": 8980856832,
      "num_gpu_blocks": 2687,
      "kv_cache_accounting_consistent": true
    },
    {
      "rank": 1,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 29983047680,
      "torch_allocated_bytes": 49011074560,
      "torch_reserved_bytes": 49977229312,
      "torch_peak_allocated_bytes": 49578562560,
      "torch_peak_reserved_bytes": 49977229312,
      "available_kv_cache_bytes": null,
      "model_memory_bytes": 39959379968,
      "kv_cache_allocated_bytes": 8980856832,
      "kv_cache_declared_bytes": 8980856832,
      "num_gpu_blocks": 2687,
      "kv_cache_accounting_consistent": true
    },
    {
      "rank": 2,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 29983047680,
      "torch_allocated_bytes": 49011074560,
      "torch_reserved_bytes": 49977229312,
      "torch_peak_allocated_bytes": 49578562560,
      "torch_peak_reserved_bytes": 49977229312,
      "available_kv_cache_bytes": null,
      "model_memory_bytes": 39959379968,
      "kv_cache_allocated_bytes": 8980856832,
      "kv_cache_declared_bytes": 8980856832,
      "num_gpu_blocks": 2687,
      "kv_cache_accounting_consistent": true
    },
    {
      "rank": 3,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 29983047680,
      "torch_allocated_bytes": 49011074560,
      "torch_reserved_bytes": 49977229312,
      "torch_peak_allocated_bytes": 49578562560,
      "torch_peak_reserved_bytes": 49977229312,
      "available_kv_cache_bytes": null,
      "model_memory_bytes": 39959379968,
      "kv_cache_allocated_bytes": 8980856832,
      "kv_cache_declared_bytes": 8980856832,
      "num_gpu_blocks": 2687,
      "kv_cache_accounting_consistent": true
    }
  ],
  "expert_cache_stats": [
    {
      "storage_backend": "native",
      "rank": 0,
      "h2d_bytes": 0,
      "copy_launches": 0,
      "startup_resident_h2d_bytes": 0,
      "promotion_d2d_bytes": 0,
      "metadata_h2d_bytes": 0,
      "unique_demands": 0,
      "resident_hits": 0,
      "timing": {
        "route_d2h_s": 0,
        "policy_cpu_s": 0,
        "host_reuse_wait_s": 0,
        "host_gather_s": 0,
        "h2d_enqueue_s": 0,
        "compute_enqueue_s": 0,
        "cuda_sample_count": 0,
        "load_cuda_s": 0,
        "compute_cuda_s": 0,
        "promotion_cuda_s": 0
      },
      "policy": {
        "observations": 0,
        "unique_demands": 0,
        "completed_forwards": 0,
        "decay_events": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "admissions": 0,
        "evictions": 0,
        "cache_bypasses": 0,
        "reconfigurations": 0
      }
    },
    {
      "storage_backend": "native",
      "rank": 1,
      "h2d_bytes": 0,
      "copy_launches": 0,
      "startup_resident_h2d_bytes": 0,
      "promotion_d2d_bytes": 0,
      "metadata_h2d_bytes": 0,
      "unique_demands": 0,
      "resident_hits": 0,
      "timing": {
        "route_d2h_s": 0,
        "policy_cpu_s": 0,
        "host_reuse_wait_s": 0,
        "host_gather_s": 0,
        "h2d_enqueue_s": 0,
        "compute_enqueue_s": 0,
        "cuda_sample_count": 0,
        "load_cuda_s": 0,
        "compute_cuda_s": 0,
        "promotion_cuda_s": 0
      },
      "policy": {
        "observations": 0,
        "unique_demands": 0,
        "completed_forwards": 0,
        "decay_events": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "admissions": 0,
        "evictions": 0,
        "cache_bypasses": 0,
        "reconfigurations": 0
      }
    },
    {
      "storage_backend": "native",
      "rank": 2,
      "h2d_bytes": 0,
      "copy_launches": 0,
      "startup_resident_h2d_bytes": 0,
      "promotion_d2d_bytes": 0,
      "metadata_h2d_bytes": 0,
      "unique_demands": 0,
      "resident_hits": 0,
      "timing": {
        "route_d2h_s": 0,
        "policy_cpu_s": 0,
        "host_reuse_wait_s": 0,
        "host_gather_s": 0,
        "h2d_enqueue_s": 0,
        "compute_enqueue_s": 0,
        "cuda_sample_count": 0,
        "load_cuda_s": 0,
        "compute_cuda_s": 0,
        "promotion_cuda_s": 0
      },
      "policy": {
        "observations": 0,
        "unique_demands": 0,
        "completed_forwards": 0,
        "decay_events": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "admissions": 0,
        "evictions": 0,
        "cache_bypasses": 0,
        "reconfigurations": 0
      }
    },
    {
      "storage_backend": "native",
      "rank": 3,
      "h2d_bytes": 0,
      "copy_launches": 0,
      "startup_resident_h2d_bytes": 0,
      "promotion_d2d_bytes": 0,
      "metadata_h2d_bytes": 0,
      "unique_demands": 0,
      "resident_hits": 0,
      "timing": {
        "route_d2h_s": 0,
        "policy_cpu_s": 0,
        "host_reuse_wait_s": 0,
        "host_gather_s": 0,
        "h2d_enqueue_s": 0,
        "compute_enqueue_s": 0,
        "cuda_sample_count": 0,
        "load_cuda_s": 0,
        "compute_cuda_s": 0,
        "promotion_cuda_s": 0
      },
      "policy": {
        "observations": 0,
        "unique_demands": 0,
        "completed_forwards": 0,
        "decay_events": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "admissions": 0,
        "evictions": 0,
        "cache_bypasses": 0,
        "reconfigurations": 0
      }
    }
  ],
  "measured": [
    {
      "repetition": 0,
      "diagnostics": {
        "h2d_bytes": 0,
        "copy_launches": 0,
        "startup_resident_h2d_bytes": 0,
        "promotion_d2d_bytes": 0,
        "metadata_h2d_bytes": 0,
        "unique_demands": 0,
        "resident_hits": 0,
        "timing": {
          "route_d2h_s": 0,
          "policy_cpu_s": 0,
          "host_reuse_wait_s": 0,
          "host_gather_s": 0,
          "h2d_enqueue_s": 0,
          "compute_enqueue_s": 0,
          "cuda_sample_count": 0,
          "load_cuda_s": 0,
          "compute_cuda_s": 0,
          "promotion_cuda_s": 0
        },
        "policy": {
          "observations": 0,
          "unique_demands": 0,
          "completed_forwards": 0,
          "decay_events": 0,
          "cache_hits": 0,
          "cache_misses": 0,
          "admissions": 0,
          "evictions": 0,
          "cache_bypasses": 0,
          "reconfigurations": 0,
          "cache_hit_ratio": null
        },
        "per_rank": [
          {
            "h2d_bytes": 0,
            "copy_launches": 0,
            "startup_resident_h2d_bytes": 0,
            "promotion_d2d_bytes": 0,
            "metadata_h2d_bytes": 0,
            "unique_demands": 0,
            "resident_hits": 0,
            "timing": {
              "route_d2h_s": 0,
              "policy_cpu_s": 0,
              "host_reuse_wait_s": 0,
              "host_gather_s": 0,
              "h2d_enqueue_s": 0,
              "compute_enqueue_s": 0,
              "cuda_sample_count": 0,
              "load_cuda_s": 0,
              "compute_cuda_s": 0,
              "promotion_cuda_s": 0
            },
            "policy": {
              "observations": 0,
              "unique_demands": 0,
              "completed_forwards": 0,
              "decay_events": 0,
              "cache_hits": 0,
              "cache_misses": 0,
              "admissions": 0,
              "evictions": 0,
              "cache_bypasses": 0,
              "reconfigurations": 0,
              "cache_hit_ratio": null
            },
            "rank": 0
          },
          {
            "h2d_bytes": 0,
            "copy_launches": 0,
            "startup_resident_h2d_bytes": 0,
            "promotion_d2d_bytes": 0,
            "metadata_h2d_bytes": 0,
            "unique_demands": 0,
            "resident_hits": 0,
            "timing": {
              "route_d2h_s": 0,
              "policy_cpu_s": 0,
              "host_reuse_wait_s": 0,
              "host_gather_s": 0,
              "h2d_enqueue_s": 0,
              "compute_enqueue_s": 0,
              "cuda_sample_count": 0,
              "load_cuda_s": 0,
              "compute_cuda_s": 0,
              "promotion_cuda_s": 0
            },
            "policy": {
              "observations": 0,
              "unique_demands": 0,
              "completed_forwards": 0,
              "decay_events": 0,
              "cache_hits": 0,
              "cache_misses": 0,
              "admissions": 0,
              "evictions": 0,
              "cache_bypasses": 0,
              "reconfigurations": 0,
              "cache_hit_ratio": null
            },
            "rank": 1
          },
          {
            "h2d_bytes": 0,
            "copy_launches": 0,
            "startup_resident_h2d_bytes": 0,
            "promotion_d2d_bytes": 0,
            "metadata_h2d_bytes": 0,
            "unique_demands": 0,
            "resident_hits": 0,
            "timing": {
              "route_d2h_s": 0,
              "policy_cpu_s": 0,
              "host_reuse_wait_s": 0,
              "host_gather_s": 0,
              "h2d_enqueue_s": 0,
              "compute_enqueue_s": 0,
              "cuda_sample_count": 0,
              "load_cuda_s": 0,
              "compute_cuda_s": 0,
              "promotion_cuda_s": 0
            },
            "policy": {
              "observations": 0,
              "unique_demands": 0,
              "completed_forwards": 0,
              "decay_events": 0,
              "cache_hits": 0,
              "cache_misses": 0,
              "admissions": 0,
              "evictions": 0,
              "cache_bypasses": 0,
              "reconfigurations": 0,
              "cache_hit_ratio": null
            },
            "rank": 2
          },
          {
            "h2d_bytes": 0,
            "copy_launches": 0,
            "startup_resident_h2d_bytes": 0,
            "promotion_d2d_bytes": 0,
            "metadata_h2d_bytes": 0,
            "unique_demands": 0,
            "resident_hits": 0,
            "timing": {
              "route_d2h_s": 0,
              "policy_cpu_s": 0,
              "host_reuse_wait_s": 0,
              "host_gather_s": 0,
              "h2d_enqueue_s": 0,
              "compute_enqueue_s": 0,
              "cuda_sample_count": 0,
              "load_cuda_s": 0,
              "compute_cuda_s": 0,
              "promotion_cuda_s": 0
            },
            "policy": {
              "observations": 0,
              "unique_demands": 0,
              "completed_forwards": 0,
              "decay_events": 0,
              "cache_hits": 0,
              "cache_misses": 0,
              "admissions": 0,
              "evictions": 0,
              "cache_bypasses": 0,
              "reconfigurations": 0,
              "cache_hit_ratio": null
            },
            "rank": 3
          }
        ]
      },
      "memory": [
        {
          "rank": 0,
          "total_gpu_bytes": 85017493504,
          "free_gpu_bytes": 29983047680,
          "torch_allocated_bytes": 49011074560,
          "torch_reserved_bytes": 49977229312,
          "torch_peak_allocated_bytes": 49578562560,
          "torch_peak_reserved_bytes": 49977229312,
          "kv_cache_allocated_bytes": 8980856832,
          "num_gpu_blocks": 2687,
          "kv_cache_declared_bytes": 8980856832,
          "kv_cache_accounting_consistent": true
        },
        {
          "rank": 1,
          "total_gpu_bytes": 85017493504,
          "free_gpu_bytes": 29983047680,
          "torch_allocated_bytes": 49011074560,
          "torch_reserved_bytes": 49977229312,
          "torch_peak_allocated_bytes": 49578562560,
          "torch_peak_reserved_bytes": 49977229312,
          "kv_cache_allocated_bytes": 8980856832,
          "num_gpu_blocks": 2687,
          "kv_cache_declared_bytes": 8980856832,
          "kv_cache_accounting_consistent": true
        },
        {
          "rank": 2,
          "total_gpu_bytes": 85017493504,
          "free_gpu_bytes": 29983047680,
          "torch_allocated_bytes": 49011074560,
          "torch_reserved_bytes": 49977229312,
          "torch_peak_allocated_bytes": 49578562560,
          "torch_peak_reserved_bytes": 49977229312,
          "kv_cache_allocated_bytes": 8980856832,
          "num_gpu_blocks": 2687,
          "kv_cache_declared_bytes": 8980856832,
          "kv_cache_accounting_consistent": true
        },
        {
          "rank": 3,
          "total_gpu_bytes": 85017493504,
          "free_gpu_bytes": 29983047680,
          "torch_allocated_bytes": 49011074560,
          "torch_reserved_bytes": 49977229312,
          "torch_peak_allocated_bytes": 49578562560,
          "torch_peak_reserved_bytes": 49977229312,
          "kv_cache_allocated_bytes": 8980856832,
          "num_gpu_blocks": 2687,
          "kv_cache_declared_bytes": 8980856832,
          "kv_cache_accounting_consistent": true
        }
      ],
      "memory_peak_scope": "measured-generate-after-synchronized-reset"
    }
  ]
}
```

</details>

## 02-partial-fixed-kv：complete
输出 token/s 中位数 18.744，范围 [18.744, 18.744]。
各 rank 实际 KV 合计 35.923427 GB。
常驻比例 0.5，共享 cache slots=256，policy=decayed-lfu，profile SHA=e102f6959c1e9a64d67294b1800b9f093fb41907a0ca4aa42aa030b7c2dad3fb。
执行 SHA：4e62dcc0a2ce74dadc56f32f2e0efb820ce7048d。以下为完整 R/B/C 的脱敏重跑模板，私有路径需在服务器设定变量：

```bash
bash scripts/server/run_expert_cache.sh confirm --suite-id "$NEW_SUITE_ID" --model-path "$MODEL_PATH" --profile-path "$PROFILE_PATH" --dataset-path "$DATASET_PATH" --dataset-manifest "$DATASET_MANIFEST" --batch-size 32 --context-length 4096 --output-length 64 --gpu-memory-utilization 0.6 --warmups 0 --seed 20260905 --max-num-seqs 32 --max-num-batched-tokens 8192 --timing-samples 128 --repetitions 1 --resident-ratio 0.5 --cache-slots 256 --cache-policy decayed-lfu --calibration-count 32
```

第 0 轮：H2D=1357413089280 bytes；resident hits=2340532；cold cache hit ratio=0.04735517484987637。

<details><summary>逐 rank / 逐层数值、kernel 选择与显存证据（bytes）</summary>

```json
{
  "memory": [
    {
      "rank": 0,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 51308986368,
      "torch_allocated_bytes": 30891673600,
      "torch_reserved_bytes": 31589400576,
      "torch_peak_allocated_bytes": 31492721152,
      "torch_peak_reserved_bytes": 31589400576,
      "available_kv_cache_bytes": 8980856832,
      "model_memory_bytes": 21839992832,
      "kv_cache_allocated_bytes": 8980856832,
      "kv_cache_declared_bytes": 8980856832,
      "num_gpu_blocks": 2687,
      "kv_cache_accounting_consistent": true
    },
    {
      "rank": 1,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 51308986368,
      "torch_allocated_bytes": 30891673600,
      "torch_reserved_bytes": 31589400576,
      "torch_peak_allocated_bytes": 31492721152,
      "torch_peak_reserved_bytes": 31589400576,
      "available_kv_cache_bytes": 8980856832,
      "model_memory_bytes": 21839992832,
      "kv_cache_allocated_bytes": 8980856832,
      "kv_cache_declared_bytes": 8980856832,
      "num_gpu_blocks": 2687,
      "kv_cache_accounting_consistent": true
    },
    {
      "rank": 2,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 51308986368,
      "torch_allocated_bytes": 30891673600,
      "torch_reserved_bytes": 31589400576,
      "torch_peak_allocated_bytes": 31492721152,
      "torch_peak_reserved_bytes": 31589400576,
      "available_kv_cache_bytes": 8980856832,
      "model_memory_bytes": 21839992832,
      "kv_cache_allocated_bytes": 8980856832,
      "kv_cache_declared_bytes": 8980856832,
      "num_gpu_blocks": 2687,
      "kv_cache_accounting_consistent": true
    },
    {
      "rank": 3,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 51308986368,
      "torch_allocated_bytes": 30891673600,
      "torch_reserved_bytes": 31589400576,
      "torch_peak_allocated_bytes": 31492721152,
      "torch_peak_reserved_bytes": 31589400576,
      "available_kv_cache_bytes": 8980856832,
      "model_memory_bytes": 21839992832,
      "kv_cache_allocated_bytes": 8980856832,
      "kv_cache_declared_bytes": 8980856832,
      "num_gpu_blocks": 2687,
      "kv_cache_accounting_consistent": true
    }
  ],
  "final_memory": [
    {
      "rank": 0,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 48368779264,
      "torch_allocated_bytes": 30891687424,
      "torch_reserved_bytes": 31589400576,
      "torch_peak_allocated_bytes": 31492741120,
      "torch_peak_reserved_bytes": 31589400576,
      "available_kv_cache_bytes": 8980856832,
      "model_memory_bytes": 21839992832,
      "kv_cache_allocated_bytes": 8980856832,
      "kv_cache_declared_bytes": 8980856832,
      "num_gpu_blocks": 2687,
      "kv_cache_accounting_consistent": true
    },
    {
      "rank": 1,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 48368779264,
      "torch_allocated_bytes": 30891687424,
      "torch_reserved_bytes": 31589400576,
      "torch_peak_allocated_bytes": 31492741120,
      "torch_peak_reserved_bytes": 31589400576,
      "available_kv_cache_bytes": 8980856832,
      "model_memory_bytes": 21839992832,
      "kv_cache_allocated_bytes": 8980856832,
      "kv_cache_declared_bytes": 8980856832,
      "num_gpu_blocks": 2687,
      "kv_cache_accounting_consistent": true
    },
    {
      "rank": 2,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 48368779264,
      "torch_allocated_bytes": 30891687424,
      "torch_reserved_bytes": 31589400576,
      "torch_peak_allocated_bytes": 31492741120,
      "torch_peak_reserved_bytes": 31589400576,
      "available_kv_cache_bytes": 8980856832,
      "model_memory_bytes": 21839992832,
      "kv_cache_allocated_bytes": 8980856832,
      "kv_cache_declared_bytes": 8980856832,
      "num_gpu_blocks": 2687,
      "kv_cache_accounting_consistent": true
    },
    {
      "rank": 3,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 48368779264,
      "torch_allocated_bytes": 30891687424,
      "torch_reserved_bytes": 31589400576,
      "torch_peak_allocated_bytes": 31492741120,
      "torch_peak_reserved_bytes": 31589400576,
      "available_kv_cache_bytes": 8980856832,
      "model_memory_bytes": 21839992832,
      "kv_cache_allocated_bytes": 8980856832,
      "kv_cache_declared_bytes": 8980856832,
      "num_gpu_blocks": 2687,
      "kv_cache_accounting_consistent": true
    }
  ],
  "expert_cache_stats": [
    {
      "schema_version": 1,
      "rank": 0,
      "tensor_parallel_size": 4,
      "total_layers": 48,
      "num_experts": 512,
      "physical_slots": 13056,
      "resident_slots": 12288,
      "cache_slots": 256,
      "ingress_slots": 512,
      "expert_bytes": 1572864,
      "host_source_bytes": 38654705664,
      "pinned_gather_bytes": 805306368,
      "gpu_pool_bytes": 20535312384,
      "gpu_resident_bytes": 19327352832,
      "gpu_cache_bytes": 402653184,
      "gpu_ingress_bytes": 805306368,
      "gpu_metadata_bytes": 6144,
      "pinned_metadata_bytes": 6144,
      "net_freed_bytes": 18119387136,
      "weights_verified": 0,
      "cuda_timing_capacity": 128,
      "h2d_bytes": 354315927552,
      "copy_launches": 8410,
      "startup_resident_h2d_bytes": 19327352832,
      "promotion_d2d_bytes": 1793064960,
      "metadata_h2d_bytes": 8954784,
      "unique_demands": 837563,
      "resident_hits": 601404,
      "max_unique_per_forward": 509,
      "mean_unique_coverage": 0.3745112716059982,
      "resident_ratio": 0.5,
      "policy": {
        "observations": 4368,
        "unique_demands": 837563,
        "completed_forwards": 91,
        "decay_events": 1,
        "cache_hits": 10891,
        "cache_misses": 225268,
        "admissions": 1140,
        "evictions": 884,
        "cache_bypasses": 224128,
        "reconfigurations": 0,
        "resident_experts": 12288,
        "cache_slots": 256,
        "cache_entries": 256,
        "cache_hit_ratio": 0.04611723457501091
      },
      "timing": {
        "route_d2h_s": 16.271816925145686,
        "policy_cpu_s": 1.4802591656334698,
        "host_reuse_wait_s": 0.025003787595778704,
        "host_gather_s": 63.958307629451156,
        "h2d_enqueue_s": 0.324088420253247,
        "compute_enqueue_s": 2.30659540835768,
        "cuda_sample_count": 124,
        "load_cuda_s": 0.20435520068556068,
        "compute_cuda_s": 0.037352383933961376,
        "promotion_cuda_s": 0.0005906240055337543
      },
      "forward_counts": [
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91
      ],
      "per_layer_unique_demands": [
        21138,
        16941,
        21537,
        15599,
        16854,
        18495,
        20346,
        20046,
        17285,
        15577,
        15503,
        16773,
        14282,
        15270,
        16944,
        18331,
        16716,
        18464,
        19940,
        20062,
        16953,
        15325,
        15398,
        16674,
        14738,
        15495,
        17059,
        18689,
        17148,
        18921,
        20215,
        20348,
        18026,
        16737,
        16304,
        17541,
        15750,
        16202,
        17386,
        18547,
        18166,
        18030,
        17684,
        16712,
        16982,
        16528,
        16246,
        17656
      ],
      "per_layer_max_unique_per_forward": [
        500,
        437,
        509,
        436,
        434,
        468,
        500,
        494,
        472,
        419,
        420,
        441,
        410,
        407,
        450,
        478,
        445,
        473,
        502,
        497,
        461,
        423,
        427,
        448,
        409,
        425,
        462,
        484,
        462,
        485,
        499,
        502,
        474,
        453,
        449,
        463,
        434,
        437,
        459,
        474,
        469,
        450,
        445,
        423,
        431,
        424,
        416,
        442
      ],
      "identity": {
        "geometry": {
          "total_layers": 48,
          "num_experts": 512,
          "hidden_size": 2048,
          "intermediate_size": 128
        },
        "tensor_parallel_size": 4,
        "model_config_sha256": "2d483c7cabad7c8704478ed4038fa7e7b2eff840bc00a118eccbe38e2b488303",
        "model_identity_sha256": "24f6a89ba0e0435caffbb3d056425fb1fe980fca09d5a8749dfb231bd24880d8"
      },
      "profile_sha256": "e102f6959c1e9a64d67294b1800b9f093fb41907a0ca4aa42aa030b7c2dad3fb",
      "cache_policy": "decayed-lfu",
      "failed": false,
      "weight_verification_scope": "CUDA parity tests required; no runtime weight D2H verification",
      "temporary_memory_scope": "native fused kernel workspace/output and CPU route IDs scale with scheduled tokens; no weight-sized GPU gather temporary",
      "memory_accounting_scope": "tensor payload bytes; excludes allocator rounding, CUDA event driver memory and Python policy metadata; native/chunk output workspace must be included in vLLM measured profiling peak",
      "kernel_config_scope": "native logical E geometry per actual native chunk size",
      "transfer_schedule": "demand-critical H2D on compute stream; no speculative prefetch",
      "kernel_config_records": {
        "capacity": 64,
        "unrecorded_selections": 0,
        "records": [
          {
            "chunk_tokens": 8192,
            "top_k": 10,
            "selections": 816,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 32,
            "top_k": 10,
            "selections": 2304,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 4096,
            "top_k": 10,
            "selections": 96,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 1,
            "top_k": 10,
            "selections": 336,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 32,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 4337,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 31,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 30,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 28,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 26,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 24,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 22,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 20,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 18,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 16,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 14,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 12,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 10,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 8,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 6,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 4,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 2,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 64,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 8
            }
          }
        ]
      }
    },
    {
      "schema_version": 1,
      "rank": 1,
      "tensor_parallel_size": 4,
      "total_layers": 48,
      "num_experts": 512,
      "physical_slots": 13056,
      "resident_slots": 12288,
      "cache_slots": 256,
      "ingress_slots": 512,
      "expert_bytes": 1572864,
      "host_source_bytes": 38654705664,
      "pinned_gather_bytes": 805306368,
      "gpu_pool_bytes": 20535312384,
      "gpu_resident_bytes": 19327352832,
      "gpu_cache_bytes": 402653184,
      "gpu_ingress_bytes": 805306368,
      "gpu_metadata_bytes": 6144,
      "pinned_metadata_bytes": 6144,
      "net_freed_bytes": 18119387136,
      "weights_verified": 0,
      "cuda_timing_capacity": 128,
      "h2d_bytes": 354315927552,
      "copy_launches": 8410,
      "startup_resident_h2d_bytes": 19327352832,
      "promotion_d2d_bytes": 1793064960,
      "metadata_h2d_bytes": 8954784,
      "unique_demands": 837563,
      "resident_hits": 601404,
      "max_unique_per_forward": 509,
      "mean_unique_coverage": 0.3745112716059982,
      "resident_ratio": 0.5,
      "policy": {
        "observations": 4368,
        "unique_demands": 837563,
        "completed_forwards": 91,
        "decay_events": 1,
        "cache_hits": 10891,
        "cache_misses": 225268,
        "admissions": 1140,
        "evictions": 884,
        "cache_bypasses": 224128,
        "reconfigurations": 0,
        "resident_experts": 12288,
        "cache_slots": 256,
        "cache_entries": 256,
        "cache_hit_ratio": 0.04611723457501091
      },
      "timing": {
        "route_d2h_s": 10.843114785384387,
        "policy_cpu_s": 1.4549122997559607,
        "host_reuse_wait_s": 0.028148752637207508,
        "host_gather_s": 71.57460931548849,
        "h2d_enqueue_s": 0.3672126685269177,
        "compute_enqueue_s": 2.4505934733897448,
        "cuda_sample_count": 124,
        "load_cuda_s": 0.20770844838768246,
        "compute_cuda_s": 0.03794223997741937,
        "promotion_cuda_s": 0.000601472005946562
      },
      "forward_counts": [
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91
      ],
      "per_layer_unique_demands": [
        21138,
        16941,
        21537,
        15599,
        16854,
        18495,
        20346,
        20046,
        17285,
        15577,
        15503,
        16773,
        14282,
        15270,
        16944,
        18331,
        16716,
        18464,
        19940,
        20062,
        16953,
        15325,
        15398,
        16674,
        14738,
        15495,
        17059,
        18689,
        17148,
        18921,
        20215,
        20348,
        18026,
        16737,
        16304,
        17541,
        15750,
        16202,
        17386,
        18547,
        18166,
        18030,
        17684,
        16712,
        16982,
        16528,
        16246,
        17656
      ],
      "per_layer_max_unique_per_forward": [
        500,
        437,
        509,
        436,
        434,
        468,
        500,
        494,
        472,
        419,
        420,
        441,
        410,
        407,
        450,
        478,
        445,
        473,
        502,
        497,
        461,
        423,
        427,
        448,
        409,
        425,
        462,
        484,
        462,
        485,
        499,
        502,
        474,
        453,
        449,
        463,
        434,
        437,
        459,
        474,
        469,
        450,
        445,
        423,
        431,
        424,
        416,
        442
      ],
      "identity": {
        "geometry": {
          "total_layers": 48,
          "num_experts": 512,
          "hidden_size": 2048,
          "intermediate_size": 128
        },
        "tensor_parallel_size": 4,
        "model_config_sha256": "2d483c7cabad7c8704478ed4038fa7e7b2eff840bc00a118eccbe38e2b488303",
        "model_identity_sha256": "24f6a89ba0e0435caffbb3d056425fb1fe980fca09d5a8749dfb231bd24880d8"
      },
      "profile_sha256": "e102f6959c1e9a64d67294b1800b9f093fb41907a0ca4aa42aa030b7c2dad3fb",
      "cache_policy": "decayed-lfu",
      "failed": false,
      "weight_verification_scope": "CUDA parity tests required; no runtime weight D2H verification",
      "temporary_memory_scope": "native fused kernel workspace/output and CPU route IDs scale with scheduled tokens; no weight-sized GPU gather temporary",
      "memory_accounting_scope": "tensor payload bytes; excludes allocator rounding, CUDA event driver memory and Python policy metadata; native/chunk output workspace must be included in vLLM measured profiling peak",
      "kernel_config_scope": "native logical E geometry per actual native chunk size",
      "transfer_schedule": "demand-critical H2D on compute stream; no speculative prefetch",
      "kernel_config_records": {
        "capacity": 64,
        "unrecorded_selections": 0,
        "records": [
          {
            "chunk_tokens": 8192,
            "top_k": 10,
            "selections": 816,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 32,
            "top_k": 10,
            "selections": 2304,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 4096,
            "top_k": 10,
            "selections": 96,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 1,
            "top_k": 10,
            "selections": 336,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 32,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 4337,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 31,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 30,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 28,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 26,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 24,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 22,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 20,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 18,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 16,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 14,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 12,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 10,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 8,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 6,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 4,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 2,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 64,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 8
            }
          }
        ]
      }
    },
    {
      "schema_version": 1,
      "rank": 2,
      "tensor_parallel_size": 4,
      "total_layers": 48,
      "num_experts": 512,
      "physical_slots": 13056,
      "resident_slots": 12288,
      "cache_slots": 256,
      "ingress_slots": 512,
      "expert_bytes": 1572864,
      "host_source_bytes": 38654705664,
      "pinned_gather_bytes": 805306368,
      "gpu_pool_bytes": 20535312384,
      "gpu_resident_bytes": 19327352832,
      "gpu_cache_bytes": 402653184,
      "gpu_ingress_bytes": 805306368,
      "gpu_metadata_bytes": 6144,
      "pinned_metadata_bytes": 6144,
      "net_freed_bytes": 18119387136,
      "weights_verified": 0,
      "cuda_timing_capacity": 128,
      "h2d_bytes": 354315927552,
      "copy_launches": 8410,
      "startup_resident_h2d_bytes": 19327352832,
      "promotion_d2d_bytes": 1793064960,
      "metadata_h2d_bytes": 8954784,
      "unique_demands": 837563,
      "resident_hits": 601404,
      "max_unique_per_forward": 509,
      "mean_unique_coverage": 0.3745112716059982,
      "resident_ratio": 0.5,
      "policy": {
        "observations": 4368,
        "unique_demands": 837563,
        "completed_forwards": 91,
        "decay_events": 1,
        "cache_hits": 10891,
        "cache_misses": 225268,
        "admissions": 1140,
        "evictions": 884,
        "cache_bypasses": 224128,
        "reconfigurations": 0,
        "resident_experts": 12288,
        "cache_slots": 256,
        "cache_entries": 256,
        "cache_hit_ratio": 0.04611723457501091
      },
      "timing": {
        "route_d2h_s": 11.001282427459955,
        "policy_cpu_s": 1.5709920241497457,
        "host_reuse_wait_s": 0.02558308094739914,
        "host_gather_s": 73.67714391089976,
        "h2d_enqueue_s": 0.3794814180582762,
        "compute_enqueue_s": 2.4611232820898294,
        "cuda_sample_count": 124,
        "load_cuda_s": 0.20642249701917162,
        "compute_cuda_s": 0.03789123209565878,
        "promotion_cuda_s": 0.0005939520080573859
      },
      "forward_counts": [
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91
      ],
      "per_layer_unique_demands": [
        21138,
        16941,
        21537,
        15599,
        16854,
        18495,
        20346,
        20046,
        17285,
        15577,
        15503,
        16773,
        14282,
        15270,
        16944,
        18331,
        16716,
        18464,
        19940,
        20062,
        16953,
        15325,
        15398,
        16674,
        14738,
        15495,
        17059,
        18689,
        17148,
        18921,
        20215,
        20348,
        18026,
        16737,
        16304,
        17541,
        15750,
        16202,
        17386,
        18547,
        18166,
        18030,
        17684,
        16712,
        16982,
        16528,
        16246,
        17656
      ],
      "per_layer_max_unique_per_forward": [
        500,
        437,
        509,
        436,
        434,
        468,
        500,
        494,
        472,
        419,
        420,
        441,
        410,
        407,
        450,
        478,
        445,
        473,
        502,
        497,
        461,
        423,
        427,
        448,
        409,
        425,
        462,
        484,
        462,
        485,
        499,
        502,
        474,
        453,
        449,
        463,
        434,
        437,
        459,
        474,
        469,
        450,
        445,
        423,
        431,
        424,
        416,
        442
      ],
      "identity": {
        "geometry": {
          "total_layers": 48,
          "num_experts": 512,
          "hidden_size": 2048,
          "intermediate_size": 128
        },
        "tensor_parallel_size": 4,
        "model_config_sha256": "2d483c7cabad7c8704478ed4038fa7e7b2eff840bc00a118eccbe38e2b488303",
        "model_identity_sha256": "24f6a89ba0e0435caffbb3d056425fb1fe980fca09d5a8749dfb231bd24880d8"
      },
      "profile_sha256": "e102f6959c1e9a64d67294b1800b9f093fb41907a0ca4aa42aa030b7c2dad3fb",
      "cache_policy": "decayed-lfu",
      "failed": false,
      "weight_verification_scope": "CUDA parity tests required; no runtime weight D2H verification",
      "temporary_memory_scope": "native fused kernel workspace/output and CPU route IDs scale with scheduled tokens; no weight-sized GPU gather temporary",
      "memory_accounting_scope": "tensor payload bytes; excludes allocator rounding, CUDA event driver memory and Python policy metadata; native/chunk output workspace must be included in vLLM measured profiling peak",
      "kernel_config_scope": "native logical E geometry per actual native chunk size",
      "transfer_schedule": "demand-critical H2D on compute stream; no speculative prefetch",
      "kernel_config_records": {
        "capacity": 64,
        "unrecorded_selections": 0,
        "records": [
          {
            "chunk_tokens": 8192,
            "top_k": 10,
            "selections": 816,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 32,
            "top_k": 10,
            "selections": 2304,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 4096,
            "top_k": 10,
            "selections": 96,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 1,
            "top_k": 10,
            "selections": 336,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 32,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 4337,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 31,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 30,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 28,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 26,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 24,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 22,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 20,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 18,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 16,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 14,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 12,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 10,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 8,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 6,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 4,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 2,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 64,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 8
            }
          }
        ]
      }
    },
    {
      "schema_version": 1,
      "rank": 3,
      "tensor_parallel_size": 4,
      "total_layers": 48,
      "num_experts": 512,
      "physical_slots": 13056,
      "resident_slots": 12288,
      "cache_slots": 256,
      "ingress_slots": 512,
      "expert_bytes": 1572864,
      "host_source_bytes": 38654705664,
      "pinned_gather_bytes": 805306368,
      "gpu_pool_bytes": 20535312384,
      "gpu_resident_bytes": 19327352832,
      "gpu_cache_bytes": 402653184,
      "gpu_ingress_bytes": 805306368,
      "gpu_metadata_bytes": 6144,
      "pinned_metadata_bytes": 6144,
      "net_freed_bytes": 18119387136,
      "weights_verified": 0,
      "cuda_timing_capacity": 128,
      "h2d_bytes": 354315927552,
      "copy_launches": 8410,
      "startup_resident_h2d_bytes": 19327352832,
      "promotion_d2d_bytes": 1793064960,
      "metadata_h2d_bytes": 8954784,
      "unique_demands": 837563,
      "resident_hits": 601404,
      "max_unique_per_forward": 509,
      "mean_unique_coverage": 0.3745112716059982,
      "resident_ratio": 0.5,
      "policy": {
        "observations": 4368,
        "unique_demands": 837563,
        "completed_forwards": 91,
        "decay_events": 1,
        "cache_hits": 10891,
        "cache_misses": 225268,
        "admissions": 1140,
        "evictions": 884,
        "cache_bypasses": 224128,
        "reconfigurations": 0,
        "resident_experts": 12288,
        "cache_slots": 256,
        "cache_entries": 256,
        "cache_hit_ratio": 0.04611723457501091
      },
      "timing": {
        "route_d2h_s": 17.809400094673038,
        "policy_cpu_s": 1.4231303138658404,
        "host_reuse_wait_s": 0.02318060351535678,
        "host_gather_s": 61.61580672580749,
        "h2d_enqueue_s": 0.31764781568199396,
        "compute_enqueue_s": 2.2976515060290694,
        "cuda_sample_count": 124,
        "load_cuda_s": 0.20785852880775935,
        "compute_cuda_s": 0.03713347193598746,
        "promotion_cuda_s": 0.0005778240084182472
      },
      "forward_counts": [
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91
      ],
      "per_layer_unique_demands": [
        21138,
        16941,
        21537,
        15599,
        16854,
        18495,
        20346,
        20046,
        17285,
        15577,
        15503,
        16773,
        14282,
        15270,
        16944,
        18331,
        16716,
        18464,
        19940,
        20062,
        16953,
        15325,
        15398,
        16674,
        14738,
        15495,
        17059,
        18689,
        17148,
        18921,
        20215,
        20348,
        18026,
        16737,
        16304,
        17541,
        15750,
        16202,
        17386,
        18547,
        18166,
        18030,
        17684,
        16712,
        16982,
        16528,
        16246,
        17656
      ],
      "per_layer_max_unique_per_forward": [
        500,
        437,
        509,
        436,
        434,
        468,
        500,
        494,
        472,
        419,
        420,
        441,
        410,
        407,
        450,
        478,
        445,
        473,
        502,
        497,
        461,
        423,
        427,
        448,
        409,
        425,
        462,
        484,
        462,
        485,
        499,
        502,
        474,
        453,
        449,
        463,
        434,
        437,
        459,
        474,
        469,
        450,
        445,
        423,
        431,
        424,
        416,
        442
      ],
      "identity": {
        "geometry": {
          "total_layers": 48,
          "num_experts": 512,
          "hidden_size": 2048,
          "intermediate_size": 128
        },
        "tensor_parallel_size": 4,
        "model_config_sha256": "2d483c7cabad7c8704478ed4038fa7e7b2eff840bc00a118eccbe38e2b488303",
        "model_identity_sha256": "24f6a89ba0e0435caffbb3d056425fb1fe980fca09d5a8749dfb231bd24880d8"
      },
      "profile_sha256": "e102f6959c1e9a64d67294b1800b9f093fb41907a0ca4aa42aa030b7c2dad3fb",
      "cache_policy": "decayed-lfu",
      "failed": false,
      "weight_verification_scope": "CUDA parity tests required; no runtime weight D2H verification",
      "temporary_memory_scope": "native fused kernel workspace/output and CPU route IDs scale with scheduled tokens; no weight-sized GPU gather temporary",
      "memory_accounting_scope": "tensor payload bytes; excludes allocator rounding, CUDA event driver memory and Python policy metadata; native/chunk output workspace must be included in vLLM measured profiling peak",
      "kernel_config_scope": "native logical E geometry per actual native chunk size",
      "transfer_schedule": "demand-critical H2D on compute stream; no speculative prefetch",
      "kernel_config_records": {
        "capacity": 64,
        "unrecorded_selections": 0,
        "records": [
          {
            "chunk_tokens": 8192,
            "top_k": 10,
            "selections": 816,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 32,
            "top_k": 10,
            "selections": 2304,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 4096,
            "top_k": 10,
            "selections": 96,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 1,
            "top_k": 10,
            "selections": 336,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 32,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 4337,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 31,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 30,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 28,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 26,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 24,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 22,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 20,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 18,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 16,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 14,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 12,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 10,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 8,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 6,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 4,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 2,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 64,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 8
            }
          }
        ]
      }
    }
  ],
  "measured": [
    {
      "repetition": 0,
      "diagnostics": {
        "h2d_bytes": 1357413089280,
        "copy_launches": 30640,
        "startup_resident_h2d_bytes": 0,
        "promotion_d2d_bytes": 1629487104,
        "metadata_h2d_bytes": 31465568,
        "unique_demands": 3246452,
        "resident_hits": 2340532,
        "timing": {
          "route_d2h_s": 55.925614232663065,
          "policy_cpu_s": 5.929293803405017,
          "host_reuse_wait_s": 0.10191622469574213,
          "host_gather_s": 270.8258675816469,
          "h2d_enqueue_s": 1.3884303225204349,
          "compute_enqueue_s": 9.515963669866323,
          "cuda_sample_count": 496,
          "load_cuda_s": 0.826344674900174,
          "compute_cuda_s": 0.150319327943027,
          "promotion_cuda_s": 0.0023638720279559496
        },
        "policy": {
          "observations": 15360,
          "unique_demands": 3246452,
          "completed_forwards": 320,
          "decay_events": 4,
          "cache_hits": 42900,
          "cache_misses": 863020,
          "admissions": 1036,
          "evictions": 1036,
          "cache_bypasses": 861984,
          "reconfigurations": 0,
          "cache_hit_ratio": 0.04735517484987637
        },
        "per_rank": [
          {
            "h2d_bytes": 339353272320,
            "copy_launches": 7660,
            "startup_resident_h2d_bytes": 0,
            "promotion_d2d_bytes": 407371776,
            "metadata_h2d_bytes": 7866392,
            "unique_demands": 811613,
            "resident_hits": 585133,
            "timing": {
              "route_d2h_s": 16.271816925145686,
              "policy_cpu_s": 1.4802591656334698,
              "host_reuse_wait_s": 0.025003787595778704,
              "host_gather_s": 63.958307629451156,
              "h2d_enqueue_s": 0.324088420253247,
              "compute_enqueue_s": 2.30659540835768,
              "cuda_sample_count": 124,
              "load_cuda_s": 0.20435520068556068,
              "compute_cuda_s": 0.037352383933961376,
              "promotion_cuda_s": 0.0005906240055337543
            },
            "policy": {
              "observations": 3840,
              "unique_demands": 811613,
              "completed_forwards": 80,
              "decay_events": 1,
              "cache_hits": 10725,
              "cache_misses": 215755,
              "admissions": 259,
              "evictions": 259,
              "cache_bypasses": 215496,
              "reconfigurations": 0,
              "cache_hit_ratio": 0.04735517484987637
            },
            "rank": 0,
            "forward_counts": [
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80
            ],
            "per_layer_unique_demands": [
              20542,
              16405,
              20930,
              15090,
              16338,
              17947,
              19758,
              19465,
              16745,
              15092,
              15021,
              16264,
              13812,
              14793,
              16407,
              17768,
              16199,
              17905,
              19346,
              19474,
              16408,
              14834,
              14889,
              16150,
              14238,
              14975,
              16504,
              18106,
              16599,
              18351,
              19622,
              19762,
              17460,
              16201,
              15768,
              16998,
              15212,
              15679,
              16828,
              17982,
              17607,
              17480,
              17145,
              16190,
              16459,
              16011,
              15738,
              17116
            ],
            "per_layer_mean_unique_coverage": [
              0.501513671875,
              0.4005126953125,
              0.510986328125,
              0.368408203125,
              0.398876953125,
              0.4381591796875,
              0.482373046875,
              0.4752197265625,
              0.4088134765625,
              0.36845703125,
              0.3667236328125,
              0.3970703125,
              0.33720703125,
              0.3611572265625,
              0.4005615234375,
              0.4337890625,
              0.3954833984375,
              0.4371337890625,
              0.472314453125,
              0.475439453125,
              0.4005859375,
              0.362158203125,
              0.3635009765625,
              0.394287109375,
              0.347607421875,
              0.3656005859375,
              0.4029296875,
              0.442041015625,
              0.4052490234375,
              0.4480224609375,
              0.479052734375,
              0.482470703125,
              0.42626953125,
              0.3955322265625,
              0.3849609375,
              0.414990234375,
              0.37138671875,
              0.3827880859375,
              0.41083984375,
              0.439013671875,
              0.4298583984375,
              0.4267578125,
              0.4185791015625,
              0.395263671875,
              0.4018310546875,
              0.3908935546875,
              0.384228515625,
              0.41787109375
            ],
            "gauge_snapshot": {
              "max_unique_per_forward": 509,
              "mean_unique_coverage": 0.3745112716059982,
              "per_layer_max_unique_per_forward": [
                500,
                437,
                509,
                436,
                434,
                468,
                500,
                494,
                472,
                419,
                420,
                441,
                410,
                407,
                450,
                478,
                445,
                473,
                502,
                497,
                461,
                423,
                427,
                448,
                409,
                425,
                462,
                484,
                462,
                485,
                499,
                502,
                474,
                453,
                449,
                463,
                434,
                437,
                459,
                474,
                469,
                450,
                445,
                423,
                431,
                424,
                416,
                442
              ]
            },
            "policy_snapshot": {
              "resident_experts": 12288,
              "cache_slots": 256,
              "cache_entries": 256,
              "cache_hit_ratio": 0.04611723457501091
            },
            "kernel_config_records": {
              "capacity": 64,
              "unrecorded_selections": 0,
              "records": [
                {
                  "chunk_tokens": 8192,
                  "top_k": 10,
                  "selections": 816,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 32,
                  "top_k": 10,
                  "selections": 2304,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 4096,
                  "top_k": 10,
                  "selections": 96,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 1,
                  "top_k": 10,
                  "selections": 336,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 32,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 4337,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 31,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 30,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 28,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 26,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 24,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 22,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 20,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 18,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 16,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 14,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 12,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 10,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 8,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 6,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 4,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 2,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 64,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 8
                  }
                }
              ]
            }
          },
          {
            "h2d_bytes": 339353272320,
            "copy_launches": 7660,
            "startup_resident_h2d_bytes": 0,
            "promotion_d2d_bytes": 407371776,
            "metadata_h2d_bytes": 7866392,
            "unique_demands": 811613,
            "resident_hits": 585133,
            "timing": {
              "route_d2h_s": 10.843114785384387,
              "policy_cpu_s": 1.4549122997559607,
              "host_reuse_wait_s": 0.028148752637207508,
              "host_gather_s": 71.57460931548849,
              "h2d_enqueue_s": 0.3672126685269177,
              "compute_enqueue_s": 2.4505934733897448,
              "cuda_sample_count": 124,
              "load_cuda_s": 0.20770844838768246,
              "compute_cuda_s": 0.03794223997741937,
              "promotion_cuda_s": 0.000601472005946562
            },
            "policy": {
              "observations": 3840,
              "unique_demands": 811613,
              "completed_forwards": 80,
              "decay_events": 1,
              "cache_hits": 10725,
              "cache_misses": 215755,
              "admissions": 259,
              "evictions": 259,
              "cache_bypasses": 215496,
              "reconfigurations": 0,
              "cache_hit_ratio": 0.04735517484987637
            },
            "rank": 1,
            "forward_counts": [
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80
            ],
            "per_layer_unique_demands": [
              20542,
              16405,
              20930,
              15090,
              16338,
              17947,
              19758,
              19465,
              16745,
              15092,
              15021,
              16264,
              13812,
              14793,
              16407,
              17768,
              16199,
              17905,
              19346,
              19474,
              16408,
              14834,
              14889,
              16150,
              14238,
              14975,
              16504,
              18106,
              16599,
              18351,
              19622,
              19762,
              17460,
              16201,
              15768,
              16998,
              15212,
              15679,
              16828,
              17982,
              17607,
              17480,
              17145,
              16190,
              16459,
              16011,
              15738,
              17116
            ],
            "per_layer_mean_unique_coverage": [
              0.501513671875,
              0.4005126953125,
              0.510986328125,
              0.368408203125,
              0.398876953125,
              0.4381591796875,
              0.482373046875,
              0.4752197265625,
              0.4088134765625,
              0.36845703125,
              0.3667236328125,
              0.3970703125,
              0.33720703125,
              0.3611572265625,
              0.4005615234375,
              0.4337890625,
              0.3954833984375,
              0.4371337890625,
              0.472314453125,
              0.475439453125,
              0.4005859375,
              0.362158203125,
              0.3635009765625,
              0.394287109375,
              0.347607421875,
              0.3656005859375,
              0.4029296875,
              0.442041015625,
              0.4052490234375,
              0.4480224609375,
              0.479052734375,
              0.482470703125,
              0.42626953125,
              0.3955322265625,
              0.3849609375,
              0.414990234375,
              0.37138671875,
              0.3827880859375,
              0.41083984375,
              0.439013671875,
              0.4298583984375,
              0.4267578125,
              0.4185791015625,
              0.395263671875,
              0.4018310546875,
              0.3908935546875,
              0.384228515625,
              0.41787109375
            ],
            "gauge_snapshot": {
              "max_unique_per_forward": 509,
              "mean_unique_coverage": 0.3745112716059982,
              "per_layer_max_unique_per_forward": [
                500,
                437,
                509,
                436,
                434,
                468,
                500,
                494,
                472,
                419,
                420,
                441,
                410,
                407,
                450,
                478,
                445,
                473,
                502,
                497,
                461,
                423,
                427,
                448,
                409,
                425,
                462,
                484,
                462,
                485,
                499,
                502,
                474,
                453,
                449,
                463,
                434,
                437,
                459,
                474,
                469,
                450,
                445,
                423,
                431,
                424,
                416,
                442
              ]
            },
            "policy_snapshot": {
              "resident_experts": 12288,
              "cache_slots": 256,
              "cache_entries": 256,
              "cache_hit_ratio": 0.04611723457501091
            },
            "kernel_config_records": {
              "capacity": 64,
              "unrecorded_selections": 0,
              "records": [
                {
                  "chunk_tokens": 8192,
                  "top_k": 10,
                  "selections": 816,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 32,
                  "top_k": 10,
                  "selections": 2304,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 4096,
                  "top_k": 10,
                  "selections": 96,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 1,
                  "top_k": 10,
                  "selections": 336,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 32,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 4337,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 31,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 30,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 28,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 26,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 24,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 22,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 20,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 18,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 16,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 14,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 12,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 10,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 8,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 6,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 4,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 2,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 64,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 8
                  }
                }
              ]
            }
          },
          {
            "h2d_bytes": 339353272320,
            "copy_launches": 7660,
            "startup_resident_h2d_bytes": 0,
            "promotion_d2d_bytes": 407371776,
            "metadata_h2d_bytes": 7866392,
            "unique_demands": 811613,
            "resident_hits": 585133,
            "timing": {
              "route_d2h_s": 11.001282427459955,
              "policy_cpu_s": 1.5709920241497457,
              "host_reuse_wait_s": 0.02558308094739914,
              "host_gather_s": 73.67714391089976,
              "h2d_enqueue_s": 0.3794814180582762,
              "compute_enqueue_s": 2.4611232820898294,
              "cuda_sample_count": 124,
              "load_cuda_s": 0.20642249701917162,
              "compute_cuda_s": 0.03789123209565878,
              "promotion_cuda_s": 0.0005939520080573859
            },
            "policy": {
              "observations": 3840,
              "unique_demands": 811613,
              "completed_forwards": 80,
              "decay_events": 1,
              "cache_hits": 10725,
              "cache_misses": 215755,
              "admissions": 259,
              "evictions": 259,
              "cache_bypasses": 215496,
              "reconfigurations": 0,
              "cache_hit_ratio": 0.04735517484987637
            },
            "rank": 2,
            "forward_counts": [
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80
            ],
            "per_layer_unique_demands": [
              20542,
              16405,
              20930,
              15090,
              16338,
              17947,
              19758,
              19465,
              16745,
              15092,
              15021,
              16264,
              13812,
              14793,
              16407,
              17768,
              16199,
              17905,
              19346,
              19474,
              16408,
              14834,
              14889,
              16150,
              14238,
              14975,
              16504,
              18106,
              16599,
              18351,
              19622,
              19762,
              17460,
              16201,
              15768,
              16998,
              15212,
              15679,
              16828,
              17982,
              17607,
              17480,
              17145,
              16190,
              16459,
              16011,
              15738,
              17116
            ],
            "per_layer_mean_unique_coverage": [
              0.501513671875,
              0.4005126953125,
              0.510986328125,
              0.368408203125,
              0.398876953125,
              0.4381591796875,
              0.482373046875,
              0.4752197265625,
              0.4088134765625,
              0.36845703125,
              0.3667236328125,
              0.3970703125,
              0.33720703125,
              0.3611572265625,
              0.4005615234375,
              0.4337890625,
              0.3954833984375,
              0.4371337890625,
              0.472314453125,
              0.475439453125,
              0.4005859375,
              0.362158203125,
              0.3635009765625,
              0.394287109375,
              0.347607421875,
              0.3656005859375,
              0.4029296875,
              0.442041015625,
              0.4052490234375,
              0.4480224609375,
              0.479052734375,
              0.482470703125,
              0.42626953125,
              0.3955322265625,
              0.3849609375,
              0.414990234375,
              0.37138671875,
              0.3827880859375,
              0.41083984375,
              0.439013671875,
              0.4298583984375,
              0.4267578125,
              0.4185791015625,
              0.395263671875,
              0.4018310546875,
              0.3908935546875,
              0.384228515625,
              0.41787109375
            ],
            "gauge_snapshot": {
              "max_unique_per_forward": 509,
              "mean_unique_coverage": 0.3745112716059982,
              "per_layer_max_unique_per_forward": [
                500,
                437,
                509,
                436,
                434,
                468,
                500,
                494,
                472,
                419,
                420,
                441,
                410,
                407,
                450,
                478,
                445,
                473,
                502,
                497,
                461,
                423,
                427,
                448,
                409,
                425,
                462,
                484,
                462,
                485,
                499,
                502,
                474,
                453,
                449,
                463,
                434,
                437,
                459,
                474,
                469,
                450,
                445,
                423,
                431,
                424,
                416,
                442
              ]
            },
            "policy_snapshot": {
              "resident_experts": 12288,
              "cache_slots": 256,
              "cache_entries": 256,
              "cache_hit_ratio": 0.04611723457501091
            },
            "kernel_config_records": {
              "capacity": 64,
              "unrecorded_selections": 0,
              "records": [
                {
                  "chunk_tokens": 8192,
                  "top_k": 10,
                  "selections": 816,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 32,
                  "top_k": 10,
                  "selections": 2304,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 4096,
                  "top_k": 10,
                  "selections": 96,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 1,
                  "top_k": 10,
                  "selections": 336,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 32,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 4337,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 31,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 30,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 28,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 26,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 24,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 22,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 20,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 18,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 16,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 14,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 12,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 10,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 8,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 6,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 4,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 2,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 64,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 8
                  }
                }
              ]
            }
          },
          {
            "h2d_bytes": 339353272320,
            "copy_launches": 7660,
            "startup_resident_h2d_bytes": 0,
            "promotion_d2d_bytes": 407371776,
            "metadata_h2d_bytes": 7866392,
            "unique_demands": 811613,
            "resident_hits": 585133,
            "timing": {
              "route_d2h_s": 17.809400094673038,
              "policy_cpu_s": 1.4231303138658404,
              "host_reuse_wait_s": 0.02318060351535678,
              "host_gather_s": 61.61580672580749,
              "h2d_enqueue_s": 0.31764781568199396,
              "compute_enqueue_s": 2.2976515060290694,
              "cuda_sample_count": 124,
              "load_cuda_s": 0.20785852880775935,
              "compute_cuda_s": 0.03713347193598746,
              "promotion_cuda_s": 0.0005778240084182472
            },
            "policy": {
              "observations": 3840,
              "unique_demands": 811613,
              "completed_forwards": 80,
              "decay_events": 1,
              "cache_hits": 10725,
              "cache_misses": 215755,
              "admissions": 259,
              "evictions": 259,
              "cache_bypasses": 215496,
              "reconfigurations": 0,
              "cache_hit_ratio": 0.04735517484987637
            },
            "rank": 3,
            "forward_counts": [
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80
            ],
            "per_layer_unique_demands": [
              20542,
              16405,
              20930,
              15090,
              16338,
              17947,
              19758,
              19465,
              16745,
              15092,
              15021,
              16264,
              13812,
              14793,
              16407,
              17768,
              16199,
              17905,
              19346,
              19474,
              16408,
              14834,
              14889,
              16150,
              14238,
              14975,
              16504,
              18106,
              16599,
              18351,
              19622,
              19762,
              17460,
              16201,
              15768,
              16998,
              15212,
              15679,
              16828,
              17982,
              17607,
              17480,
              17145,
              16190,
              16459,
              16011,
              15738,
              17116
            ],
            "per_layer_mean_unique_coverage": [
              0.501513671875,
              0.4005126953125,
              0.510986328125,
              0.368408203125,
              0.398876953125,
              0.4381591796875,
              0.482373046875,
              0.4752197265625,
              0.4088134765625,
              0.36845703125,
              0.3667236328125,
              0.3970703125,
              0.33720703125,
              0.3611572265625,
              0.4005615234375,
              0.4337890625,
              0.3954833984375,
              0.4371337890625,
              0.472314453125,
              0.475439453125,
              0.4005859375,
              0.362158203125,
              0.3635009765625,
              0.394287109375,
              0.347607421875,
              0.3656005859375,
              0.4029296875,
              0.442041015625,
              0.4052490234375,
              0.4480224609375,
              0.479052734375,
              0.482470703125,
              0.42626953125,
              0.3955322265625,
              0.3849609375,
              0.414990234375,
              0.37138671875,
              0.3827880859375,
              0.41083984375,
              0.439013671875,
              0.4298583984375,
              0.4267578125,
              0.4185791015625,
              0.395263671875,
              0.4018310546875,
              0.3908935546875,
              0.384228515625,
              0.41787109375
            ],
            "gauge_snapshot": {
              "max_unique_per_forward": 509,
              "mean_unique_coverage": 0.3745112716059982,
              "per_layer_max_unique_per_forward": [
                500,
                437,
                509,
                436,
                434,
                468,
                500,
                494,
                472,
                419,
                420,
                441,
                410,
                407,
                450,
                478,
                445,
                473,
                502,
                497,
                461,
                423,
                427,
                448,
                409,
                425,
                462,
                484,
                462,
                485,
                499,
                502,
                474,
                453,
                449,
                463,
                434,
                437,
                459,
                474,
                469,
                450,
                445,
                423,
                431,
                424,
                416,
                442
              ]
            },
            "policy_snapshot": {
              "resident_experts": 12288,
              "cache_slots": 256,
              "cache_entries": 256,
              "cache_hit_ratio": 0.04611723457501091
            },
            "kernel_config_records": {
              "capacity": 64,
              "unrecorded_selections": 0,
              "records": [
                {
                  "chunk_tokens": 8192,
                  "top_k": 10,
                  "selections": 816,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 32,
                  "top_k": 10,
                  "selections": 2304,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 4096,
                  "top_k": 10,
                  "selections": 96,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 1,
                  "top_k": 10,
                  "selections": 336,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 32,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 4337,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 31,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 30,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 28,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 26,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 24,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 22,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 20,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 18,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 16,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 14,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 12,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 10,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 8,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 6,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 4,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 2,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 64,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 8
                  }
                }
              ]
            }
          }
        ]
      },
      "memory": [
        {
          "rank": 0,
          "total_gpu_bytes": 85017493504,
          "free_gpu_bytes": 48368779264,
          "torch_allocated_bytes": 30891687424,
          "torch_reserved_bytes": 31589400576,
          "torch_peak_allocated_bytes": 31492741120,
          "torch_peak_reserved_bytes": 31589400576,
          "kv_cache_allocated_bytes": 8980856832,
          "num_gpu_blocks": 2687,
          "kv_cache_declared_bytes": 8980856832,
          "kv_cache_accounting_consistent": true
        },
        {
          "rank": 1,
          "total_gpu_bytes": 85017493504,
          "free_gpu_bytes": 48368779264,
          "torch_allocated_bytes": 30891687424,
          "torch_reserved_bytes": 31589400576,
          "torch_peak_allocated_bytes": 31492741120,
          "torch_peak_reserved_bytes": 31589400576,
          "kv_cache_allocated_bytes": 8980856832,
          "num_gpu_blocks": 2687,
          "kv_cache_declared_bytes": 8980856832,
          "kv_cache_accounting_consistent": true
        },
        {
          "rank": 2,
          "total_gpu_bytes": 85017493504,
          "free_gpu_bytes": 48368779264,
          "torch_allocated_bytes": 30891687424,
          "torch_reserved_bytes": 31589400576,
          "torch_peak_allocated_bytes": 31492741120,
          "torch_peak_reserved_bytes": 31589400576,
          "kv_cache_allocated_bytes": 8980856832,
          "num_gpu_blocks": 2687,
          "kv_cache_declared_bytes": 8980856832,
          "kv_cache_accounting_consistent": true
        },
        {
          "rank": 3,
          "total_gpu_bytes": 85017493504,
          "free_gpu_bytes": 48368779264,
          "torch_allocated_bytes": 30891687424,
          "torch_reserved_bytes": 31589400576,
          "torch_peak_allocated_bytes": 31492741120,
          "torch_peak_reserved_bytes": 31589400576,
          "kv_cache_allocated_bytes": 8980856832,
          "num_gpu_blocks": 2687,
          "kv_cache_declared_bytes": 8980856832,
          "kv_cache_accounting_consistent": true
        }
      ],
      "memory_peak_scope": "measured-generate-after-synchronized-reset"
    }
  ]
}
```

</details>

## 03-partial-auto-kv：complete
输出 token/s 中位数 17.417，范围 [17.417, 17.417]。
各 rank 实际 KV 合计 108.264948 GB。
常驻比例 0.5，共享 cache slots=256，policy=decayed-lfu，profile SHA=e102f6959c1e9a64d67294b1800b9f093fb41907a0ca4aa42aa030b7c2dad3fb。
执行 SHA：4e62dcc0a2ce74dadc56f32f2e0efb820ce7048d。以下为完整 R/B/C 的脱敏重跑模板，私有路径需在服务器设定变量：

```bash
bash scripts/server/run_expert_cache.sh confirm --suite-id "$NEW_SUITE_ID" --model-path "$MODEL_PATH" --profile-path "$PROFILE_PATH" --dataset-path "$DATASET_PATH" --dataset-manifest "$DATASET_MANIFEST" --batch-size 32 --context-length 4096 --output-length 64 --gpu-memory-utilization 0.6 --warmups 0 --seed 20260905 --max-num-seqs 32 --max-num-batched-tokens 8192 --timing-samples 128 --repetitions 1 --resident-ratio 0.5 --cache-slots 256 --cache-policy decayed-lfu --calibration-count 32
```

第 0 轮：H2D=1372160262144 bytes；resident hits=2348836；cold cache hit ratio=0.046578420486636296。

<details><summary>逐 rank / 逐层数值、kernel 选择与显存证据（bytes）</summary>

```json
{
  "memory": [
    {
      "rank": 0,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 33214758912,
      "torch_allocated_bytes": 48985901056,
      "torch_reserved_bytes": 49683628032,
      "torch_peak_allocated_bytes": 49586948608,
      "torch_peak_reserved_bytes": 49683628032,
      "available_kv_cache_bytes": null,
      "model_memory_bytes": 21839992832,
      "kv_cache_allocated_bytes": 27066236928,
      "kv_cache_declared_bytes": 27066236928,
      "num_gpu_blocks": 8098,
      "kv_cache_accounting_consistent": true
    },
    {
      "rank": 1,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 33214758912,
      "torch_allocated_bytes": 48985901056,
      "torch_reserved_bytes": 49683628032,
      "torch_peak_allocated_bytes": 49586948608,
      "torch_peak_reserved_bytes": 49683628032,
      "available_kv_cache_bytes": null,
      "model_memory_bytes": 21839992832,
      "kv_cache_allocated_bytes": 27066236928,
      "kv_cache_declared_bytes": 27066236928,
      "num_gpu_blocks": 8098,
      "kv_cache_accounting_consistent": true
    },
    {
      "rank": 2,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 33214758912,
      "torch_allocated_bytes": 48985901056,
      "torch_reserved_bytes": 49683628032,
      "torch_peak_allocated_bytes": 49586948608,
      "torch_peak_reserved_bytes": 49683628032,
      "available_kv_cache_bytes": null,
      "model_memory_bytes": 21839992832,
      "kv_cache_allocated_bytes": 27066236928,
      "kv_cache_declared_bytes": 27066236928,
      "num_gpu_blocks": 8098,
      "kv_cache_accounting_consistent": true
    },
    {
      "rank": 3,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 33214758912,
      "torch_allocated_bytes": 48985901056,
      "torch_reserved_bytes": 49683628032,
      "torch_peak_allocated_bytes": 49586948608,
      "torch_peak_reserved_bytes": 49683628032,
      "available_kv_cache_bytes": null,
      "model_memory_bytes": 21839992832,
      "kv_cache_allocated_bytes": 27066236928,
      "kv_cache_declared_bytes": 27066236928,
      "num_gpu_blocks": 8098,
      "kv_cache_accounting_consistent": true
    }
  ],
  "final_memory": [
    {
      "rank": 0,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 30274551808,
      "torch_allocated_bytes": 48985914880,
      "torch_reserved_bytes": 49683628032,
      "torch_peak_allocated_bytes": 49586968576,
      "torch_peak_reserved_bytes": 49683628032,
      "available_kv_cache_bytes": null,
      "model_memory_bytes": 21839992832,
      "kv_cache_allocated_bytes": 27066236928,
      "kv_cache_declared_bytes": 27066236928,
      "num_gpu_blocks": 8098,
      "kv_cache_accounting_consistent": true
    },
    {
      "rank": 1,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 30274551808,
      "torch_allocated_bytes": 48985914880,
      "torch_reserved_bytes": 49683628032,
      "torch_peak_allocated_bytes": 49586968576,
      "torch_peak_reserved_bytes": 49683628032,
      "available_kv_cache_bytes": null,
      "model_memory_bytes": 21839992832,
      "kv_cache_allocated_bytes": 27066236928,
      "kv_cache_declared_bytes": 27066236928,
      "num_gpu_blocks": 8098,
      "kv_cache_accounting_consistent": true
    },
    {
      "rank": 2,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 30274551808,
      "torch_allocated_bytes": 48985914880,
      "torch_reserved_bytes": 49683628032,
      "torch_peak_allocated_bytes": 49586968576,
      "torch_peak_reserved_bytes": 49683628032,
      "available_kv_cache_bytes": null,
      "model_memory_bytes": 21839992832,
      "kv_cache_allocated_bytes": 27066236928,
      "kv_cache_declared_bytes": 27066236928,
      "num_gpu_blocks": 8098,
      "kv_cache_accounting_consistent": true
    },
    {
      "rank": 3,
      "total_gpu_bytes": 85017493504,
      "free_gpu_bytes": 30274551808,
      "torch_allocated_bytes": 48985914880,
      "torch_reserved_bytes": 49683628032,
      "torch_peak_allocated_bytes": 49586968576,
      "torch_peak_reserved_bytes": 49683628032,
      "available_kv_cache_bytes": null,
      "model_memory_bytes": 21839992832,
      "kv_cache_allocated_bytes": 27066236928,
      "kv_cache_declared_bytes": 27066236928,
      "num_gpu_blocks": 8098,
      "kv_cache_accounting_consistent": true
    }
  ],
  "expert_cache_stats": [
    {
      "schema_version": 1,
      "rank": 0,
      "tensor_parallel_size": 4,
      "total_layers": 48,
      "num_experts": 512,
      "physical_slots": 13056,
      "resident_slots": 12288,
      "cache_slots": 256,
      "ingress_slots": 512,
      "expert_bytes": 1572864,
      "host_source_bytes": 38654705664,
      "pinned_gather_bytes": 805306368,
      "gpu_pool_bytes": 20535312384,
      "gpu_resident_bytes": 19327352832,
      "gpu_cache_bytes": 402653184,
      "gpu_ingress_bytes": 805306368,
      "gpu_metadata_bytes": 6144,
      "pinned_metadata_bytes": 6144,
      "net_freed_bytes": 18119387136,
      "weights_verified": 0,
      "cuda_timing_capacity": 128,
      "h2d_bytes": 357813977088,
      "copy_launches": 8308,
      "startup_resident_h2d_bytes": 19327352832,
      "promotion_d2d_bytes": 1675100160,
      "metadata_h2d_bytes": 8954184,
      "unique_demands": 841839,
      "resident_hits": 603410,
      "max_unique_per_forward": 509,
      "mean_unique_coverage": 0.37642325935782966,
      "resident_ratio": 0.5,
      "policy": {
        "observations": 4368,
        "unique_demands": 841839,
        "completed_forwards": 91,
        "decay_events": 1,
        "cache_hits": 10937,
        "cache_misses": 227492,
        "admissions": 1065,
        "evictions": 809,
        "cache_bypasses": 226427,
        "reconfigurations": 0,
        "resident_experts": 12288,
        "cache_slots": 256,
        "cache_entries": 256,
        "cache_hit_ratio": 0.04587109789497083
      },
      "timing": {
        "route_d2h_s": 8.566066913306713,
        "policy_cpu_s": 1.558783849235624,
        "host_reuse_wait_s": 0.02542046131566167,
        "host_gather_s": 80.1332723768428,
        "h2d_enqueue_s": 0.38467802107334137,
        "compute_enqueue_s": 2.569735643453896,
        "cuda_sample_count": 124,
        "load_cuda_s": 0.20754265635088082,
        "compute_cuda_s": 0.03787894401699306,
        "promotion_cuda_s": 0.0006469439940992742
      },
      "forward_counts": [
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91
      ],
      "per_layer_unique_demands": [
        20991,
        16907,
        21516,
        15645,
        16966,
        18575,
        20358,
        20116,
        17356,
        15640,
        15524,
        16843,
        14395,
        15365,
        17101,
        18463,
        16731,
        18529,
        20060,
        20185,
        17125,
        15436,
        15504,
        16860,
        14906,
        15634,
        17267,
        18857,
        17305,
        18973,
        20332,
        20447,
        18090,
        16834,
        16458,
        17672,
        15803,
        16219,
        17560,
        18688,
        18191,
        18042,
        17750,
        16828,
        17020,
        16652,
        16344,
        17776
      ],
      "per_layer_max_unique_per_forward": [
        500,
        437,
        509,
        432,
        435,
        467,
        500,
        494,
        473,
        418,
        420,
        441,
        410,
        409,
        448,
        477,
        445,
        473,
        500,
        497,
        462,
        422,
        428,
        448,
        409,
        425,
        459,
        485,
        460,
        486,
        499,
        502,
        475,
        450,
        444,
        461,
        434,
        443,
        459,
        474,
        474,
        449,
        448,
        429,
        432,
        426,
        415,
        442
      ],
      "identity": {
        "geometry": {
          "total_layers": 48,
          "num_experts": 512,
          "hidden_size": 2048,
          "intermediate_size": 128
        },
        "tensor_parallel_size": 4,
        "model_config_sha256": "2d483c7cabad7c8704478ed4038fa7e7b2eff840bc00a118eccbe38e2b488303",
        "model_identity_sha256": "24f6a89ba0e0435caffbb3d056425fb1fe980fca09d5a8749dfb231bd24880d8"
      },
      "profile_sha256": "e102f6959c1e9a64d67294b1800b9f093fb41907a0ca4aa42aa030b7c2dad3fb",
      "cache_policy": "decayed-lfu",
      "failed": false,
      "weight_verification_scope": "CUDA parity tests required; no runtime weight D2H verification",
      "temporary_memory_scope": "native fused kernel workspace/output and CPU route IDs scale with scheduled tokens; no weight-sized GPU gather temporary",
      "memory_accounting_scope": "tensor payload bytes; excludes allocator rounding, CUDA event driver memory and Python policy metadata; native/chunk output workspace must be included in vLLM measured profiling peak",
      "kernel_config_scope": "native logical E geometry per actual native chunk size",
      "transfer_schedule": "demand-critical H2D on compute stream; no speculative prefetch",
      "kernel_config_records": {
        "capacity": 64,
        "unrecorded_selections": 0,
        "records": [
          {
            "chunk_tokens": 8192,
            "top_k": 10,
            "selections": 816,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 32,
            "top_k": 10,
            "selections": 2304,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 4096,
            "top_k": 10,
            "selections": 96,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 1,
            "top_k": 10,
            "selections": 336,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 32,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 4337,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 31,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 30,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 28,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 26,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 24,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 22,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 20,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 18,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 16,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 14,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 12,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 10,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 8,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 6,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 4,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 2,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 64,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 8
            }
          }
        ]
      }
    },
    {
      "schema_version": 1,
      "rank": 1,
      "tensor_parallel_size": 4,
      "total_layers": 48,
      "num_experts": 512,
      "physical_slots": 13056,
      "resident_slots": 12288,
      "cache_slots": 256,
      "ingress_slots": 512,
      "expert_bytes": 1572864,
      "host_source_bytes": 38654705664,
      "pinned_gather_bytes": 805306368,
      "gpu_pool_bytes": 20535312384,
      "gpu_resident_bytes": 19327352832,
      "gpu_cache_bytes": 402653184,
      "gpu_ingress_bytes": 805306368,
      "gpu_metadata_bytes": 6144,
      "pinned_metadata_bytes": 6144,
      "net_freed_bytes": 18119387136,
      "weights_verified": 0,
      "cuda_timing_capacity": 128,
      "h2d_bytes": 357813977088,
      "copy_launches": 8308,
      "startup_resident_h2d_bytes": 19327352832,
      "promotion_d2d_bytes": 1675100160,
      "metadata_h2d_bytes": 8954184,
      "unique_demands": 841839,
      "resident_hits": 603410,
      "max_unique_per_forward": 509,
      "mean_unique_coverage": 0.37642325935782966,
      "resident_ratio": 0.5,
      "policy": {
        "observations": 4368,
        "unique_demands": 841839,
        "completed_forwards": 91,
        "decay_events": 1,
        "cache_hits": 10937,
        "cache_misses": 227492,
        "admissions": 1065,
        "evictions": 809,
        "cache_bypasses": 226427,
        "reconfigurations": 0,
        "resident_experts": 12288,
        "cache_slots": 256,
        "cache_entries": 256,
        "cache_hit_ratio": 0.04587109789497083
      },
      "timing": {
        "route_d2h_s": 18.302965915761888,
        "policy_cpu_s": 1.3735566856339574,
        "host_reuse_wait_s": 0.022596919909119606,
        "host_gather_s": 63.46918825665489,
        "h2d_enqueue_s": 0.326983782928437,
        "compute_enqueue_s": 2.368891968857497,
        "cuda_sample_count": 124,
        "load_cuda_s": 0.20979795343987653,
        "compute_cuda_s": 0.03681839986145496,
        "promotion_cuda_s": 0.0005654400046914815
      },
      "forward_counts": [
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91
      ],
      "per_layer_unique_demands": [
        20991,
        16907,
        21516,
        15645,
        16966,
        18575,
        20358,
        20116,
        17356,
        15640,
        15524,
        16843,
        14395,
        15365,
        17101,
        18463,
        16731,
        18529,
        20060,
        20185,
        17125,
        15436,
        15504,
        16860,
        14906,
        15634,
        17267,
        18857,
        17305,
        18973,
        20332,
        20447,
        18090,
        16834,
        16458,
        17672,
        15803,
        16219,
        17560,
        18688,
        18191,
        18042,
        17750,
        16828,
        17020,
        16652,
        16344,
        17776
      ],
      "per_layer_max_unique_per_forward": [
        500,
        437,
        509,
        432,
        435,
        467,
        500,
        494,
        473,
        418,
        420,
        441,
        410,
        409,
        448,
        477,
        445,
        473,
        500,
        497,
        462,
        422,
        428,
        448,
        409,
        425,
        459,
        485,
        460,
        486,
        499,
        502,
        475,
        450,
        444,
        461,
        434,
        443,
        459,
        474,
        474,
        449,
        448,
        429,
        432,
        426,
        415,
        442
      ],
      "identity": {
        "geometry": {
          "total_layers": 48,
          "num_experts": 512,
          "hidden_size": 2048,
          "intermediate_size": 128
        },
        "tensor_parallel_size": 4,
        "model_config_sha256": "2d483c7cabad7c8704478ed4038fa7e7b2eff840bc00a118eccbe38e2b488303",
        "model_identity_sha256": "24f6a89ba0e0435caffbb3d056425fb1fe980fca09d5a8749dfb231bd24880d8"
      },
      "profile_sha256": "e102f6959c1e9a64d67294b1800b9f093fb41907a0ca4aa42aa030b7c2dad3fb",
      "cache_policy": "decayed-lfu",
      "failed": false,
      "weight_verification_scope": "CUDA parity tests required; no runtime weight D2H verification",
      "temporary_memory_scope": "native fused kernel workspace/output and CPU route IDs scale with scheduled tokens; no weight-sized GPU gather temporary",
      "memory_accounting_scope": "tensor payload bytes; excludes allocator rounding, CUDA event driver memory and Python policy metadata; native/chunk output workspace must be included in vLLM measured profiling peak",
      "kernel_config_scope": "native logical E geometry per actual native chunk size",
      "transfer_schedule": "demand-critical H2D on compute stream; no speculative prefetch",
      "kernel_config_records": {
        "capacity": 64,
        "unrecorded_selections": 0,
        "records": [
          {
            "chunk_tokens": 8192,
            "top_k": 10,
            "selections": 816,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 32,
            "top_k": 10,
            "selections": 2304,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 4096,
            "top_k": 10,
            "selections": 96,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 1,
            "top_k": 10,
            "selections": 336,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 32,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 4337,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 31,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 30,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 28,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 26,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 24,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 22,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 20,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 18,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 16,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 14,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 12,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 10,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 8,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 6,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 4,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 2,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 64,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 8
            }
          }
        ]
      }
    },
    {
      "schema_version": 1,
      "rank": 2,
      "tensor_parallel_size": 4,
      "total_layers": 48,
      "num_experts": 512,
      "physical_slots": 13056,
      "resident_slots": 12288,
      "cache_slots": 256,
      "ingress_slots": 512,
      "expert_bytes": 1572864,
      "host_source_bytes": 38654705664,
      "pinned_gather_bytes": 805306368,
      "gpu_pool_bytes": 20535312384,
      "gpu_resident_bytes": 19327352832,
      "gpu_cache_bytes": 402653184,
      "gpu_ingress_bytes": 805306368,
      "gpu_metadata_bytes": 6144,
      "pinned_metadata_bytes": 6144,
      "net_freed_bytes": 18119387136,
      "weights_verified": 0,
      "cuda_timing_capacity": 128,
      "h2d_bytes": 357813977088,
      "copy_launches": 8308,
      "startup_resident_h2d_bytes": 19327352832,
      "promotion_d2d_bytes": 1675100160,
      "metadata_h2d_bytes": 8954184,
      "unique_demands": 841839,
      "resident_hits": 603410,
      "max_unique_per_forward": 509,
      "mean_unique_coverage": 0.37642325935782966,
      "resident_ratio": 0.5,
      "policy": {
        "observations": 4368,
        "unique_demands": 841839,
        "completed_forwards": 91,
        "decay_events": 1,
        "cache_hits": 10937,
        "cache_misses": 227492,
        "admissions": 1065,
        "evictions": 809,
        "cache_bypasses": 226427,
        "reconfigurations": 0,
        "resident_experts": 12288,
        "cache_slots": 256,
        "cache_entries": 256,
        "cache_hit_ratio": 0.04587109789497083
      },
      "timing": {
        "route_d2h_s": 7.5610654153861105,
        "policy_cpu_s": 1.450863336212933,
        "host_reuse_wait_s": 0.02602496324107051,
        "host_gather_s": 83.59698324231431,
        "h2d_enqueue_s": 0.39969751238822937,
        "compute_enqueue_s": 2.539919016417116,
        "cuda_sample_count": 124,
        "load_cuda_s": 0.21342339316941805,
        "compute_cuda_s": 0.0372399039119482,
        "promotion_cuda_s": 0.0006304320055060091
      },
      "forward_counts": [
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91
      ],
      "per_layer_unique_demands": [
        20991,
        16907,
        21516,
        15645,
        16966,
        18575,
        20358,
        20116,
        17356,
        15640,
        15524,
        16843,
        14395,
        15365,
        17101,
        18463,
        16731,
        18529,
        20060,
        20185,
        17125,
        15436,
        15504,
        16860,
        14906,
        15634,
        17267,
        18857,
        17305,
        18973,
        20332,
        20447,
        18090,
        16834,
        16458,
        17672,
        15803,
        16219,
        17560,
        18688,
        18191,
        18042,
        17750,
        16828,
        17020,
        16652,
        16344,
        17776
      ],
      "per_layer_max_unique_per_forward": [
        500,
        437,
        509,
        432,
        435,
        467,
        500,
        494,
        473,
        418,
        420,
        441,
        410,
        409,
        448,
        477,
        445,
        473,
        500,
        497,
        462,
        422,
        428,
        448,
        409,
        425,
        459,
        485,
        460,
        486,
        499,
        502,
        475,
        450,
        444,
        461,
        434,
        443,
        459,
        474,
        474,
        449,
        448,
        429,
        432,
        426,
        415,
        442
      ],
      "identity": {
        "geometry": {
          "total_layers": 48,
          "num_experts": 512,
          "hidden_size": 2048,
          "intermediate_size": 128
        },
        "tensor_parallel_size": 4,
        "model_config_sha256": "2d483c7cabad7c8704478ed4038fa7e7b2eff840bc00a118eccbe38e2b488303",
        "model_identity_sha256": "24f6a89ba0e0435caffbb3d056425fb1fe980fca09d5a8749dfb231bd24880d8"
      },
      "profile_sha256": "e102f6959c1e9a64d67294b1800b9f093fb41907a0ca4aa42aa030b7c2dad3fb",
      "cache_policy": "decayed-lfu",
      "failed": false,
      "weight_verification_scope": "CUDA parity tests required; no runtime weight D2H verification",
      "temporary_memory_scope": "native fused kernel workspace/output and CPU route IDs scale with scheduled tokens; no weight-sized GPU gather temporary",
      "memory_accounting_scope": "tensor payload bytes; excludes allocator rounding, CUDA event driver memory and Python policy metadata; native/chunk output workspace must be included in vLLM measured profiling peak",
      "kernel_config_scope": "native logical E geometry per actual native chunk size",
      "transfer_schedule": "demand-critical H2D on compute stream; no speculative prefetch",
      "kernel_config_records": {
        "capacity": 64,
        "unrecorded_selections": 0,
        "records": [
          {
            "chunk_tokens": 8192,
            "top_k": 10,
            "selections": 816,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 32,
            "top_k": 10,
            "selections": 2304,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 4096,
            "top_k": 10,
            "selections": 96,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 1,
            "top_k": 10,
            "selections": 336,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 32,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 4337,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 31,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 30,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 28,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 26,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 24,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 22,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 20,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 18,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 16,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 14,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 12,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 10,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 8,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 6,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 4,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 2,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 64,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 8
            }
          }
        ]
      }
    },
    {
      "schema_version": 1,
      "rank": 3,
      "tensor_parallel_size": 4,
      "total_layers": 48,
      "num_experts": 512,
      "physical_slots": 13056,
      "resident_slots": 12288,
      "cache_slots": 256,
      "ingress_slots": 512,
      "expert_bytes": 1572864,
      "host_source_bytes": 38654705664,
      "pinned_gather_bytes": 805306368,
      "gpu_pool_bytes": 20535312384,
      "gpu_resident_bytes": 19327352832,
      "gpu_cache_bytes": 402653184,
      "gpu_ingress_bytes": 805306368,
      "gpu_metadata_bytes": 6144,
      "pinned_metadata_bytes": 6144,
      "net_freed_bytes": 18119387136,
      "weights_verified": 0,
      "cuda_timing_capacity": 128,
      "h2d_bytes": 357813977088,
      "copy_launches": 8308,
      "startup_resident_h2d_bytes": 19327352832,
      "promotion_d2d_bytes": 1675100160,
      "metadata_h2d_bytes": 8954184,
      "unique_demands": 841839,
      "resident_hits": 603410,
      "max_unique_per_forward": 509,
      "mean_unique_coverage": 0.37642325935782966,
      "resident_ratio": 0.5,
      "policy": {
        "observations": 4368,
        "unique_demands": 841839,
        "completed_forwards": 91,
        "decay_events": 1,
        "cache_hits": 10937,
        "cache_misses": 227492,
        "admissions": 1065,
        "evictions": 809,
        "cache_bypasses": 226427,
        "reconfigurations": 0,
        "resident_experts": 12288,
        "cache_slots": 256,
        "cache_entries": 256,
        "cache_hit_ratio": 0.04587109789497083
      },
      "timing": {
        "route_d2h_s": 7.588375614956021,
        "policy_cpu_s": 1.445556317921728,
        "host_reuse_wait_s": 0.025636155623942614,
        "host_gather_s": 83.47532714251429,
        "h2d_enqueue_s": 0.39012086344882846,
        "compute_enqueue_s": 2.5011281175538898,
        "cuda_sample_count": 124,
        "load_cuda_s": 0.2136820497196167,
        "compute_cuda_s": 0.03726412799209358,
        "promotion_cuda_s": 0.0006402880067471417
      },
      "forward_counts": [
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91,
        91
      ],
      "per_layer_unique_demands": [
        20991,
        16907,
        21516,
        15645,
        16966,
        18575,
        20358,
        20116,
        17356,
        15640,
        15524,
        16843,
        14395,
        15365,
        17101,
        18463,
        16731,
        18529,
        20060,
        20185,
        17125,
        15436,
        15504,
        16860,
        14906,
        15634,
        17267,
        18857,
        17305,
        18973,
        20332,
        20447,
        18090,
        16834,
        16458,
        17672,
        15803,
        16219,
        17560,
        18688,
        18191,
        18042,
        17750,
        16828,
        17020,
        16652,
        16344,
        17776
      ],
      "per_layer_max_unique_per_forward": [
        500,
        437,
        509,
        432,
        435,
        467,
        500,
        494,
        473,
        418,
        420,
        441,
        410,
        409,
        448,
        477,
        445,
        473,
        500,
        497,
        462,
        422,
        428,
        448,
        409,
        425,
        459,
        485,
        460,
        486,
        499,
        502,
        475,
        450,
        444,
        461,
        434,
        443,
        459,
        474,
        474,
        449,
        448,
        429,
        432,
        426,
        415,
        442
      ],
      "identity": {
        "geometry": {
          "total_layers": 48,
          "num_experts": 512,
          "hidden_size": 2048,
          "intermediate_size": 128
        },
        "tensor_parallel_size": 4,
        "model_config_sha256": "2d483c7cabad7c8704478ed4038fa7e7b2eff840bc00a118eccbe38e2b488303",
        "model_identity_sha256": "24f6a89ba0e0435caffbb3d056425fb1fe980fca09d5a8749dfb231bd24880d8"
      },
      "profile_sha256": "e102f6959c1e9a64d67294b1800b9f093fb41907a0ca4aa42aa030b7c2dad3fb",
      "cache_policy": "decayed-lfu",
      "failed": false,
      "weight_verification_scope": "CUDA parity tests required; no runtime weight D2H verification",
      "temporary_memory_scope": "native fused kernel workspace/output and CPU route IDs scale with scheduled tokens; no weight-sized GPU gather temporary",
      "memory_accounting_scope": "tensor payload bytes; excludes allocator rounding, CUDA event driver memory and Python policy metadata; native/chunk output workspace must be included in vLLM measured profiling peak",
      "kernel_config_scope": "native logical E geometry per actual native chunk size",
      "transfer_schedule": "demand-critical H2D on compute stream; no speculative prefetch",
      "kernel_config_records": {
        "capacity": 64,
        "unrecorded_selections": 0,
        "records": [
          {
            "chunk_tokens": 8192,
            "top_k": 10,
            "selections": 816,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 32,
            "top_k": 10,
            "selections": 2304,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 4096,
            "top_k": 10,
            "selections": 96,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 1,
            "top_k": 10,
            "selections": 336,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 32,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 4337,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 128,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 64,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 31,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 30,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 8
            }
          },
          {
            "chunk_tokens": 28,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 26,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 24,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 22,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 20,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 18,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 16,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 14,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 12,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 10,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 8,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 3,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 6,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 4,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 16,
              "BLOCK_SIZE_N": 128,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 4
            }
          },
          {
            "chunk_tokens": 2,
            "top_k": 10,
            "selections": 48,
            "w13_shape": [
              512,
              256,
              2048
            ],
            "w2_shape": [
              512,
              2048,
              128
            ],
            "config": {
              "BLOCK_SIZE_K": 64,
              "BLOCK_SIZE_M": 32,
              "BLOCK_SIZE_N": 64,
              "GROUP_SIZE_M": 1,
              "num_stages": 4,
              "num_warps": 8
            }
          }
        ]
      }
    }
  ],
  "measured": [
    {
      "repetition": 0,
      "diagnostics": {
        "h2d_bytes": 1372160262144,
        "copy_launches": 30624,
        "startup_resident_h2d_bytes": 0,
        "promotion_d2d_bytes": 1667235840,
        "metadata_h2d_bytes": 31465760,
        "unique_demands": 3263852,
        "resident_hits": 2348836,
        "timing": {
          "route_d2h_s": 42.01847385941073,
          "policy_cpu_s": 5.828760189004242,
          "host_reuse_wait_s": 0.0996785000897944,
          "host_gather_s": 310.6747710183263,
          "h2d_enqueue_s": 1.5014801798388362,
          "compute_enqueue_s": 9.979674746282399,
          "cuda_sample_count": 496,
          "load_cuda_s": 0.8444460526797921,
          "compute_cuda_s": 0.1492013757824898,
          "promotion_cuda_s": 0.0024831040110439065
        },
        "policy": {
          "observations": 15360,
          "unique_demands": 3263852,
          "completed_forwards": 320,
          "decay_events": 4,
          "cache_hits": 42620,
          "cache_misses": 872396,
          "admissions": 1060,
          "evictions": 1060,
          "cache_bypasses": 871336,
          "reconfigurations": 0,
          "cache_hit_ratio": 0.046578420486636296
        },
        "per_rank": [
          {
            "h2d_bytes": 343040065536,
            "copy_launches": 7656,
            "startup_resident_h2d_bytes": 0,
            "promotion_d2d_bytes": 416808960,
            "metadata_h2d_bytes": 7866440,
            "unique_demands": 815963,
            "resident_hits": 587209,
            "timing": {
              "route_d2h_s": 8.566066913306713,
              "policy_cpu_s": 1.558783849235624,
              "host_reuse_wait_s": 0.02542046131566167,
              "host_gather_s": 80.1332723768428,
              "h2d_enqueue_s": 0.38467802107334137,
              "compute_enqueue_s": 2.569735643453896,
              "cuda_sample_count": 124,
              "load_cuda_s": 0.20754265635088082,
              "compute_cuda_s": 0.03787894401699306,
              "promotion_cuda_s": 0.0006469439940992742
            },
            "policy": {
              "observations": 3840,
              "unique_demands": 815963,
              "completed_forwards": 80,
              "decay_events": 1,
              "cache_hits": 10655,
              "cache_misses": 218099,
              "admissions": 265,
              "evictions": 265,
              "cache_bypasses": 217834,
              "reconfigurations": 0,
              "cache_hit_ratio": 0.046578420486636296
            },
            "rank": 0,
            "forward_counts": [
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80
            ],
            "per_layer_unique_demands": [
              20395,
              16372,
              20910,
              15138,
              16450,
              18027,
              19770,
              19538,
              16820,
              15156,
              15042,
              16337,
              13924,
              14890,
              16567,
              17899,
              16215,
              17971,
              19465,
              19600,
              16581,
              14946,
              14997,
              16335,
              14406,
              15117,
              16717,
              18278,
              16760,
              18406,
              19738,
              19862,
              17528,
              16306,
              15922,
              17128,
              15269,
              15697,
              17002,
              18127,
              17634,
              17491,
              17212,
              16307,
              16499,
              16137,
              15837,
              17238
            ],
            "per_layer_mean_unique_coverage": [
              0.4979248046875,
              0.39970703125,
              0.510498046875,
              0.369580078125,
              0.401611328125,
              0.4401123046875,
              0.482666015625,
              0.477001953125,
              0.41064453125,
              0.37001953125,
              0.367236328125,
              0.3988525390625,
              0.33994140625,
              0.363525390625,
              0.4044677734375,
              0.4369873046875,
              0.3958740234375,
              0.4387451171875,
              0.4752197265625,
              0.478515625,
              0.4048095703125,
              0.364892578125,
              0.3661376953125,
              0.3988037109375,
              0.351708984375,
              0.3690673828125,
              0.4081298828125,
              0.446240234375,
              0.4091796875,
              0.449365234375,
              0.481884765625,
              0.484912109375,
              0.4279296875,
              0.398095703125,
              0.388720703125,
              0.4181640625,
              0.3727783203125,
              0.3832275390625,
              0.415087890625,
              0.4425537109375,
              0.430517578125,
              0.4270263671875,
              0.42021484375,
              0.3981201171875,
              0.4028076171875,
              0.3939697265625,
              0.3866455078125,
              0.420849609375
            ],
            "gauge_snapshot": {
              "max_unique_per_forward": 509,
              "mean_unique_coverage": 0.37642325935782966,
              "per_layer_max_unique_per_forward": [
                500,
                437,
                509,
                432,
                435,
                467,
                500,
                494,
                473,
                418,
                420,
                441,
                410,
                409,
                448,
                477,
                445,
                473,
                500,
                497,
                462,
                422,
                428,
                448,
                409,
                425,
                459,
                485,
                460,
                486,
                499,
                502,
                475,
                450,
                444,
                461,
                434,
                443,
                459,
                474,
                474,
                449,
                448,
                429,
                432,
                426,
                415,
                442
              ]
            },
            "policy_snapshot": {
              "resident_experts": 12288,
              "cache_slots": 256,
              "cache_entries": 256,
              "cache_hit_ratio": 0.04587109789497083
            },
            "kernel_config_records": {
              "capacity": 64,
              "unrecorded_selections": 0,
              "records": [
                {
                  "chunk_tokens": 8192,
                  "top_k": 10,
                  "selections": 816,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 32,
                  "top_k": 10,
                  "selections": 2304,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 4096,
                  "top_k": 10,
                  "selections": 96,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 1,
                  "top_k": 10,
                  "selections": 336,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 32,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 4337,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 31,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 30,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 28,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 26,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 24,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 22,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 20,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 18,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 16,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 14,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 12,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 10,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 8,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 6,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 4,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 2,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 64,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 8
                  }
                }
              ]
            }
          },
          {
            "h2d_bytes": 343040065536,
            "copy_launches": 7656,
            "startup_resident_h2d_bytes": 0,
            "promotion_d2d_bytes": 416808960,
            "metadata_h2d_bytes": 7866440,
            "unique_demands": 815963,
            "resident_hits": 587209,
            "timing": {
              "route_d2h_s": 18.302965915761888,
              "policy_cpu_s": 1.3735566856339574,
              "host_reuse_wait_s": 0.022596919909119606,
              "host_gather_s": 63.46918825665489,
              "h2d_enqueue_s": 0.326983782928437,
              "compute_enqueue_s": 2.368891968857497,
              "cuda_sample_count": 124,
              "load_cuda_s": 0.20979795343987653,
              "compute_cuda_s": 0.03681839986145496,
              "promotion_cuda_s": 0.0005654400046914815
            },
            "policy": {
              "observations": 3840,
              "unique_demands": 815963,
              "completed_forwards": 80,
              "decay_events": 1,
              "cache_hits": 10655,
              "cache_misses": 218099,
              "admissions": 265,
              "evictions": 265,
              "cache_bypasses": 217834,
              "reconfigurations": 0,
              "cache_hit_ratio": 0.046578420486636296
            },
            "rank": 1,
            "forward_counts": [
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80
            ],
            "per_layer_unique_demands": [
              20395,
              16372,
              20910,
              15138,
              16450,
              18027,
              19770,
              19538,
              16820,
              15156,
              15042,
              16337,
              13924,
              14890,
              16567,
              17899,
              16215,
              17971,
              19465,
              19600,
              16581,
              14946,
              14997,
              16335,
              14406,
              15117,
              16717,
              18278,
              16760,
              18406,
              19738,
              19862,
              17528,
              16306,
              15922,
              17128,
              15269,
              15697,
              17002,
              18127,
              17634,
              17491,
              17212,
              16307,
              16499,
              16137,
              15837,
              17238
            ],
            "per_layer_mean_unique_coverage": [
              0.4979248046875,
              0.39970703125,
              0.510498046875,
              0.369580078125,
              0.401611328125,
              0.4401123046875,
              0.482666015625,
              0.477001953125,
              0.41064453125,
              0.37001953125,
              0.367236328125,
              0.3988525390625,
              0.33994140625,
              0.363525390625,
              0.4044677734375,
              0.4369873046875,
              0.3958740234375,
              0.4387451171875,
              0.4752197265625,
              0.478515625,
              0.4048095703125,
              0.364892578125,
              0.3661376953125,
              0.3988037109375,
              0.351708984375,
              0.3690673828125,
              0.4081298828125,
              0.446240234375,
              0.4091796875,
              0.449365234375,
              0.481884765625,
              0.484912109375,
              0.4279296875,
              0.398095703125,
              0.388720703125,
              0.4181640625,
              0.3727783203125,
              0.3832275390625,
              0.415087890625,
              0.4425537109375,
              0.430517578125,
              0.4270263671875,
              0.42021484375,
              0.3981201171875,
              0.4028076171875,
              0.3939697265625,
              0.3866455078125,
              0.420849609375
            ],
            "gauge_snapshot": {
              "max_unique_per_forward": 509,
              "mean_unique_coverage": 0.37642325935782966,
              "per_layer_max_unique_per_forward": [
                500,
                437,
                509,
                432,
                435,
                467,
                500,
                494,
                473,
                418,
                420,
                441,
                410,
                409,
                448,
                477,
                445,
                473,
                500,
                497,
                462,
                422,
                428,
                448,
                409,
                425,
                459,
                485,
                460,
                486,
                499,
                502,
                475,
                450,
                444,
                461,
                434,
                443,
                459,
                474,
                474,
                449,
                448,
                429,
                432,
                426,
                415,
                442
              ]
            },
            "policy_snapshot": {
              "resident_experts": 12288,
              "cache_slots": 256,
              "cache_entries": 256,
              "cache_hit_ratio": 0.04587109789497083
            },
            "kernel_config_records": {
              "capacity": 64,
              "unrecorded_selections": 0,
              "records": [
                {
                  "chunk_tokens": 8192,
                  "top_k": 10,
                  "selections": 816,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 32,
                  "top_k": 10,
                  "selections": 2304,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 4096,
                  "top_k": 10,
                  "selections": 96,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 1,
                  "top_k": 10,
                  "selections": 336,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 32,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 4337,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 31,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 30,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 28,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 26,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 24,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 22,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 20,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 18,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 16,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 14,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 12,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 10,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 8,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 6,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 4,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 2,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 64,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 8
                  }
                }
              ]
            }
          },
          {
            "h2d_bytes": 343040065536,
            "copy_launches": 7656,
            "startup_resident_h2d_bytes": 0,
            "promotion_d2d_bytes": 416808960,
            "metadata_h2d_bytes": 7866440,
            "unique_demands": 815963,
            "resident_hits": 587209,
            "timing": {
              "route_d2h_s": 7.5610654153861105,
              "policy_cpu_s": 1.450863336212933,
              "host_reuse_wait_s": 0.02602496324107051,
              "host_gather_s": 83.59698324231431,
              "h2d_enqueue_s": 0.39969751238822937,
              "compute_enqueue_s": 2.539919016417116,
              "cuda_sample_count": 124,
              "load_cuda_s": 0.21342339316941805,
              "compute_cuda_s": 0.0372399039119482,
              "promotion_cuda_s": 0.0006304320055060091
            },
            "policy": {
              "observations": 3840,
              "unique_demands": 815963,
              "completed_forwards": 80,
              "decay_events": 1,
              "cache_hits": 10655,
              "cache_misses": 218099,
              "admissions": 265,
              "evictions": 265,
              "cache_bypasses": 217834,
              "reconfigurations": 0,
              "cache_hit_ratio": 0.046578420486636296
            },
            "rank": 2,
            "forward_counts": [
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80
            ],
            "per_layer_unique_demands": [
              20395,
              16372,
              20910,
              15138,
              16450,
              18027,
              19770,
              19538,
              16820,
              15156,
              15042,
              16337,
              13924,
              14890,
              16567,
              17899,
              16215,
              17971,
              19465,
              19600,
              16581,
              14946,
              14997,
              16335,
              14406,
              15117,
              16717,
              18278,
              16760,
              18406,
              19738,
              19862,
              17528,
              16306,
              15922,
              17128,
              15269,
              15697,
              17002,
              18127,
              17634,
              17491,
              17212,
              16307,
              16499,
              16137,
              15837,
              17238
            ],
            "per_layer_mean_unique_coverage": [
              0.4979248046875,
              0.39970703125,
              0.510498046875,
              0.369580078125,
              0.401611328125,
              0.4401123046875,
              0.482666015625,
              0.477001953125,
              0.41064453125,
              0.37001953125,
              0.367236328125,
              0.3988525390625,
              0.33994140625,
              0.363525390625,
              0.4044677734375,
              0.4369873046875,
              0.3958740234375,
              0.4387451171875,
              0.4752197265625,
              0.478515625,
              0.4048095703125,
              0.364892578125,
              0.3661376953125,
              0.3988037109375,
              0.351708984375,
              0.3690673828125,
              0.4081298828125,
              0.446240234375,
              0.4091796875,
              0.449365234375,
              0.481884765625,
              0.484912109375,
              0.4279296875,
              0.398095703125,
              0.388720703125,
              0.4181640625,
              0.3727783203125,
              0.3832275390625,
              0.415087890625,
              0.4425537109375,
              0.430517578125,
              0.4270263671875,
              0.42021484375,
              0.3981201171875,
              0.4028076171875,
              0.3939697265625,
              0.3866455078125,
              0.420849609375
            ],
            "gauge_snapshot": {
              "max_unique_per_forward": 509,
              "mean_unique_coverage": 0.37642325935782966,
              "per_layer_max_unique_per_forward": [
                500,
                437,
                509,
                432,
                435,
                467,
                500,
                494,
                473,
                418,
                420,
                441,
                410,
                409,
                448,
                477,
                445,
                473,
                500,
                497,
                462,
                422,
                428,
                448,
                409,
                425,
                459,
                485,
                460,
                486,
                499,
                502,
                475,
                450,
                444,
                461,
                434,
                443,
                459,
                474,
                474,
                449,
                448,
                429,
                432,
                426,
                415,
                442
              ]
            },
            "policy_snapshot": {
              "resident_experts": 12288,
              "cache_slots": 256,
              "cache_entries": 256,
              "cache_hit_ratio": 0.04587109789497083
            },
            "kernel_config_records": {
              "capacity": 64,
              "unrecorded_selections": 0,
              "records": [
                {
                  "chunk_tokens": 8192,
                  "top_k": 10,
                  "selections": 816,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 32,
                  "top_k": 10,
                  "selections": 2304,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 4096,
                  "top_k": 10,
                  "selections": 96,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 1,
                  "top_k": 10,
                  "selections": 336,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 32,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 4337,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 31,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 30,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 28,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 26,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 24,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 22,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 20,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 18,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 16,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 14,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 12,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 10,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 8,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 6,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 4,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 2,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 64,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 8
                  }
                }
              ]
            }
          },
          {
            "h2d_bytes": 343040065536,
            "copy_launches": 7656,
            "startup_resident_h2d_bytes": 0,
            "promotion_d2d_bytes": 416808960,
            "metadata_h2d_bytes": 7866440,
            "unique_demands": 815963,
            "resident_hits": 587209,
            "timing": {
              "route_d2h_s": 7.588375614956021,
              "policy_cpu_s": 1.445556317921728,
              "host_reuse_wait_s": 0.025636155623942614,
              "host_gather_s": 83.47532714251429,
              "h2d_enqueue_s": 0.39012086344882846,
              "compute_enqueue_s": 2.5011281175538898,
              "cuda_sample_count": 124,
              "load_cuda_s": 0.2136820497196167,
              "compute_cuda_s": 0.03726412799209358,
              "promotion_cuda_s": 0.0006402880067471417
            },
            "policy": {
              "observations": 3840,
              "unique_demands": 815963,
              "completed_forwards": 80,
              "decay_events": 1,
              "cache_hits": 10655,
              "cache_misses": 218099,
              "admissions": 265,
              "evictions": 265,
              "cache_bypasses": 217834,
              "reconfigurations": 0,
              "cache_hit_ratio": 0.046578420486636296
            },
            "rank": 3,
            "forward_counts": [
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80,
              80
            ],
            "per_layer_unique_demands": [
              20395,
              16372,
              20910,
              15138,
              16450,
              18027,
              19770,
              19538,
              16820,
              15156,
              15042,
              16337,
              13924,
              14890,
              16567,
              17899,
              16215,
              17971,
              19465,
              19600,
              16581,
              14946,
              14997,
              16335,
              14406,
              15117,
              16717,
              18278,
              16760,
              18406,
              19738,
              19862,
              17528,
              16306,
              15922,
              17128,
              15269,
              15697,
              17002,
              18127,
              17634,
              17491,
              17212,
              16307,
              16499,
              16137,
              15837,
              17238
            ],
            "per_layer_mean_unique_coverage": [
              0.4979248046875,
              0.39970703125,
              0.510498046875,
              0.369580078125,
              0.401611328125,
              0.4401123046875,
              0.482666015625,
              0.477001953125,
              0.41064453125,
              0.37001953125,
              0.367236328125,
              0.3988525390625,
              0.33994140625,
              0.363525390625,
              0.4044677734375,
              0.4369873046875,
              0.3958740234375,
              0.4387451171875,
              0.4752197265625,
              0.478515625,
              0.4048095703125,
              0.364892578125,
              0.3661376953125,
              0.3988037109375,
              0.351708984375,
              0.3690673828125,
              0.4081298828125,
              0.446240234375,
              0.4091796875,
              0.449365234375,
              0.481884765625,
              0.484912109375,
              0.4279296875,
              0.398095703125,
              0.388720703125,
              0.4181640625,
              0.3727783203125,
              0.3832275390625,
              0.415087890625,
              0.4425537109375,
              0.430517578125,
              0.4270263671875,
              0.42021484375,
              0.3981201171875,
              0.4028076171875,
              0.3939697265625,
              0.3866455078125,
              0.420849609375
            ],
            "gauge_snapshot": {
              "max_unique_per_forward": 509,
              "mean_unique_coverage": 0.37642325935782966,
              "per_layer_max_unique_per_forward": [
                500,
                437,
                509,
                432,
                435,
                467,
                500,
                494,
                473,
                418,
                420,
                441,
                410,
                409,
                448,
                477,
                445,
                473,
                500,
                497,
                462,
                422,
                428,
                448,
                409,
                425,
                459,
                485,
                460,
                486,
                499,
                502,
                475,
                450,
                444,
                461,
                434,
                443,
                459,
                474,
                474,
                449,
                448,
                429,
                432,
                426,
                415,
                442
              ]
            },
            "policy_snapshot": {
              "resident_experts": 12288,
              "cache_slots": 256,
              "cache_entries": 256,
              "cache_hit_ratio": 0.04587109789497083
            },
            "kernel_config_records": {
              "capacity": 64,
              "unrecorded_selections": 0,
              "records": [
                {
                  "chunk_tokens": 8192,
                  "top_k": 10,
                  "selections": 816,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 32,
                  "top_k": 10,
                  "selections": 2304,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 4096,
                  "top_k": 10,
                  "selections": 96,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 1,
                  "top_k": 10,
                  "selections": 336,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 32,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 4337,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 64,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 31,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 30,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 8
                  }
                },
                {
                  "chunk_tokens": 28,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 26,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 24,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 22,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 20,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 18,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 16,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 14,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 12,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 10,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 8,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 3,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 6,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 4,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 4
                  }
                },
                {
                  "chunk_tokens": 2,
                  "top_k": 10,
                  "selections": 48,
                  "w13_shape": [
                    512,
                    256,
                    2048
                  ],
                  "w2_shape": [
                    512,
                    2048,
                    128
                  ],
                  "config": {
                    "BLOCK_SIZE_K": 64,
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 64,
                    "GROUP_SIZE_M": 1,
                    "num_stages": 4,
                    "num_warps": 8
                  }
                }
              ]
            }
          }
        ]
      },
      "memory": [
        {
          "rank": 0,
          "total_gpu_bytes": 85017493504,
          "free_gpu_bytes": 30274551808,
          "torch_allocated_bytes": 48985914880,
          "torch_reserved_bytes": 49683628032,
          "torch_peak_allocated_bytes": 49586968576,
          "torch_peak_reserved_bytes": 49683628032,
          "kv_cache_allocated_bytes": 27066236928,
          "num_gpu_blocks": 8098,
          "kv_cache_declared_bytes": 27066236928,
          "kv_cache_accounting_consistent": true
        },
        {
          "rank": 1,
          "total_gpu_bytes": 85017493504,
          "free_gpu_bytes": 30274551808,
          "torch_allocated_bytes": 48985914880,
          "torch_reserved_bytes": 49683628032,
          "torch_peak_allocated_bytes": 49586968576,
          "torch_peak_reserved_bytes": 49683628032,
          "kv_cache_allocated_bytes": 27066236928,
          "num_gpu_blocks": 8098,
          "kv_cache_declared_bytes": 27066236928,
          "kv_cache_accounting_consistent": true
        },
        {
          "rank": 2,
          "total_gpu_bytes": 85017493504,
          "free_gpu_bytes": 30274551808,
          "torch_allocated_bytes": 48985914880,
          "torch_reserved_bytes": 49683628032,
          "torch_peak_allocated_bytes": 49586968576,
          "torch_peak_reserved_bytes": 49683628032,
          "kv_cache_allocated_bytes": 27066236928,
          "num_gpu_blocks": 8098,
          "kv_cache_declared_bytes": 27066236928,
          "kv_cache_accounting_consistent": true
        },
        {
          "rank": 3,
          "total_gpu_bytes": 85017493504,
          "free_gpu_bytes": 30274551808,
          "torch_allocated_bytes": 48985914880,
          "torch_reserved_bytes": 49683628032,
          "torch_peak_allocated_bytes": 49586968576,
          "torch_peak_reserved_bytes": 49683628032,
          "kv_cache_allocated_bytes": 27066236928,
          "num_gpu_blocks": 8098,
          "kv_cache_declared_bytes": 27066236928,
          "kv_cache_accounting_consistent": true
        }
      ],
      "memory_peak_scope": "measured-generate-after-synchronized-reset"
    }
  ]
}
```

</details>

比较：no-stable-net-gain。
B/R=0.0733；C/B=0.9292；C/R=0.0681；三轮稳定净收益=False。性能批量输出哈希一致=False。

每层 forward / unique demand 增量与覆盖率、状态最大值、嵌套 policy / timing、实际 kernel 选择和每 rank 容量均保存在 summary.json 与 evidence.csv。覆盖率分母为该层 measured forward 数乘逻辑专家数；最大值是累计状态快照，不做相减。
H2D=0 可以是全命中；命中率分母只含 cold cache hits+misses。命中率或正 H2D 均不能替代端到端收益。吞吐变慢同样是有效负结果。
正确性只由独立 CUDA parity tests 与 batch=1 smoke 各自覆盖；性能批量输出哈希单列。runtime weights_verified=0，未执行运行时大权重 D2H 验证。
模型身份绑定 config/index/本地路径哈希，未逐字节散列全 checkpoint。校准和评估按完整 prompt token 哈希分离，未声称源会话独立。
原始 prompt、生成文本、权重路径、机器身份及日志不在此导出包内。
