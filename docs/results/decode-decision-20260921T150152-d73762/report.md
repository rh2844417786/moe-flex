# TP4 低并发卸载与受约束 Oracle 报告

- Run ID: `20260921T150152-d73762`
- Commit: `21f1b7a2fd002374889ed86cf02e6b3da0c498ff`
- Suite status: `failed`
- 正式吞吐只来自 profile-off 的三次 repetition；profile-on 数值仅用于机制诊断。
- TP 时间采用四个 rank 中的最大墙钟或逐 rank 展示，绝不把四个 rank 的时间相加。
- `—`、`unavailable`、失败和跳过都不解释为数值 0。吞吐为包含 prefill 的整批墙钟。

## 结论门槛

- 结论：`inconclusive`
- 完整门槛：`{"oracle": {"16": {"h1_over_h0": null, "h2_over_h0": null, "output_gate": "unavailable"}, "32": {"h1_over_h0": null, "h2_over_h0": null, "output_gate": "unavailable"}}, "replay": {"16": {"future_aware_reduces_misses": false, "status": "unavailable"}, "32": {"future_aware_reduces_misses": false, "status": "unavailable"}}, "runtime": {"offload_over_resident": null, "zero_miss": "unavailable"}, "service_native_k0": {"kv_admission_blocked": {}, "no_kv_pressure_proven": false, "preemptions_total": null, "recomputed_tokens": {}}}`
- 未使用人为显著性阈值；报告保留比值、范围和证据门槛供判断。

## 无采样零 miss 固定开销

- 零 miss 门槛：`unavailable`
- 同 KV eager offload/resident 正式吞吐比：—
- host preparation 边界：`wait_host()` 返回后开始；包含条件性权重 gather 和无条件 host map 构建；在 H2D enqueue 前结束。instrumented 结果分别记录 `weight_gather_cpu_s`、`host_map_cpu_s`，兼容字段 `host_gather_s` 是两者之和。

| repetition | rank | misses | payload bytes | copy launches |
|---:|---:|---:|---:|---:|

### 零 miss 配对的 phase 与请求延迟

| 点 | 状态 | KV/GPU bytes | output tokens/s | range | prefill wall s | decode wall s | actual decode batch mean |
|---|---|---:|---:|---|---:|---:|---:|
| zero-resident | unavailable | — | — | —–— | — | — | — |
| zero-offload | unavailable | — | — | —–— | — | — | — |

### 零 miss instrumented 逐 rank 时间（不作为正式吞吐）

| rep | rank | CUDA step p50/p95 ms | host reuse wait sum s | weight gather sum s | host map sum s | host gather兼容和 s | payload bytes |
|---:|---:|---|---:|---:|---:|---:|---:|

## 执行模式分解

同一历史 KV 点用于 Native → eager resident → eager offload 分解；机制点只有固定 actual batch 门槛通过时才支持固定-batch 解释。

| 点 | 状态 | KV/GPU bytes | output tokens/s | samples | prefill wall s | decode wall s |
|---|---|---:|---:|---|---:|---:|
| legacy-native | unavailable | — | — | — | — | — |
| legacy-eager-resident | unavailable | — | — | — | — | — |
| legacy-offload | unavailable | — | — | — | — | — |
| mechanism-native-32 | unavailable | — | — | — | — | — |
| mechanism-eager-32 | unavailable | — | — | — | — | — |
| mechanism-offload-32 | unavailable | — | — | — | — | — |

- 固定 actual batch 门槛：`{}`
- 若 Native 的 K0 与 eager 的 K_pair 不同，Native 对比带容量混杂；严格卸载管理税只看 eager resident 与 eager offload 的 K_pair 配对。

## K0 与 K1 稳定容量

- 容量结论：`unavailable`
- 完整容量证据：`{}`
- 选择：`{}`
- K1 只有严格超过 Native 与 eager-resident 两种上限至少一个实际 KV block，才视为卸载独有容量。

## 最大并发 32 服务收益

- K1/K0 同卸载路径干预：`unavailable`；大/小 KV 吞吐比=—
- 同 K_pair eager offload/resident：`unavailable`；吞吐比=—

| 点 | 状态 | KV/GPU bytes | blocks × bytes/block | output tokens/s | batch mean | batch histogram | KV peak | preemptions | waiting peak | TTFT median s | latency median s |
|---|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|
| service-native-k0 | unavailable | — | — × — | — | — | — | — | — | — | — | — |
| service-offload-k0 | unavailable | — | — × — | — | — | — | — | — | — | — | — |
| service-offload-k1 | unavailable | — | — × — | — | — | — | — | — | — | — | — |
| service-eager-kpair | unavailable | — | — × — | — | — | — | — | — | — | — | — |
| service-offload-kpair | unavailable | — | — × — | — | — | — | — | — | — | — | — |

### 服务指标可用性

- `service-native-k0`: KV admission={"status": "unavailable"}; swap={"status": "unavailable"}; recompute={"status": "unavailable"}
  - physical safety headroom: []
- `service-offload-k0`: KV admission={"status": "unavailable"}; swap={"status": "unavailable"}; recompute={"status": "unavailable"}
  - physical safety headroom: []
- `service-offload-k1`: KV admission={"status": "unavailable"}; swap={"status": "unavailable"}; recompute={"status": "unavailable"}
  - physical safety headroom: []
- `service-eager-kpair`: KV admission={"status": "unavailable"}; swap={"status": "unavailable"}; recompute={"status": "unavailable"}
  - physical safety headroom: []
- `service-offload-kpair`: KV admission={"status": "unavailable"}; swap={"status": "unavailable"}; recompute={"status": "unavailable"}
  - physical safety headroom: []

## 同预算缓存重放

重放只改变 eviction/admission，不提前加载、不增加 cache/ingress/KV，也不支持吞吐结论。只有逐行重现生产策略后才展示 LRU 与 future-aware。

| batch | 状态 | current miss | LRU miss | future-aware miss | future-aware saved bytes |
|---:|---|---:|---:|---:|---:|
| 16 | unavailable | — | — | — | — |

### batch 16 的逐 rank/repetition replay

| 32 | unavailable | — | — | — | — |

### batch 32 的逐 rank/repetition replay


## 真实 Oracle horizon 0/1/2

正式点完整执行 host gather、transfer queue、真实 H2D、映射/物化、ready wait、promotion 和原专家计算；profile 点只提供局部机制计时。

| 点 | 状态 | timing eligible | output tokens/s | range | prefill wall s | decode wall s |
|---|---|---:|---:|---|---:|---:|

### Oracle 逐 repetition/rank 物理证据

| 点 | rep | rank | mandatory bytes | prefetched bytes | unused bytes | lookahead ms | ready-before-use | all-ready layer batch | H2D p50/p95/p99 ms | host-buffer wait p50/p95/p99 ms | exposed wait p50/p95/p99 ms | fallback experts |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|---:|

- too-early prefetch bytes：`unavailable`；当前 bounded ingress 不驱逐/过期预取项，无法把“过早”与“最终未使用”拆成两个测量量。`unused_prefetch_bytes` 单独保留为实测。
- transfer queue wait：`unavailable`；当前只记录 H2D event duration、host-buffer reuse wait 与 exposed compute wait，不把它们相加成全局净损失。

### Oracle 请求级延迟


## 强制漏预测正确性

- 强制遗漏：`unavailable`
- batch 16 输出门槛：`unavailable`
- batch 32（含 forced omission）输出门槛：`unavailable`
- 输出哈希证据：`{"batch_16": {}, "batch_32": {}}`
- 只有 forced omission 计数与 on-demand fallback 计数均为 1，且三次 greedy 输出与 horizon 0 完全一致，才通过正确性门槛。

## 失败、超时与不可用指标

- Suite status=`failed`; suite error=KeyboardInterrupt; report error=—
- 无失败或跳过点。

### 所有点

| 点 | 状态 | 模式 | KV bytes/GPU | output tokens/s | timing eligible |
|---|---|---|---:|---:|---:|

## 完整聚合数值证据

以下 JSON 与同目录 `report.json` 内容一致；它直接嵌入报告，便于离线查看，不含 prompt/output token IDs、完整专家 route、主机名、模型路径或私有日志。

```json
{
  "artifact_kind": "decode-oracle-decision-report",
  "capacity": {},
  "commit": "21f1b7a2fd002374889ed86cf02e6b3da0c498ff",
  "decision": "inconclusive",
  "decision_gates": {
    "oracle": {
      "16": {
        "h1_over_h0": null,
        "h2_over_h0": null,
        "output_gate": "unavailable"
      },
      "32": {
        "h1_over_h0": null,
        "h2_over_h0": null,
        "output_gate": "unavailable"
      }
    },
    "replay": {
      "16": {
        "future_aware_reduces_misses": false,
        "status": "unavailable"
      },
      "32": {
        "future_aware_reduces_misses": false,
        "status": "unavailable"
      }
    },
    "runtime": {
      "offload_over_resident": null,
      "zero_miss": "unavailable"
    },
    "service_native_k0": {
      "kv_admission_blocked": {},
      "no_kv_pressure_proven": false,
      "preemptions_total": null,
      "recomputed_tokens": {}
    }
  },
  "decision_scope": "ratios and evidence gates are reported without inventing a significance threshold",
  "decisions": [
    "inconclusive"
  ],
  "failed_points": [],
  "forced_omission": null,
  "mode_decomposition": {
    "fixed_batch_gates": {},
    "instrumented_timing": {
      "trace_16": {},
      "trace_32": {},
      "zero_offload": {}
    },
    "zero_pair": {}
  },
  "oracle": {},
  "oracle_output_gates": {
    "batch_16": {},
    "batch_32": {}
  },
  "points": [],
  "replay": {
    "batch_16": {},
    "batch_32": {}
  },
  "report_error_type": null,
  "run_id": "20260921T150152-d73762",
  "schema_version": 1,
  "selected_capacities": {},
  "service": {
    "kv_intervention": {},
    "low_pair": {}
  },
  "status": "failed",
  "suite_error_type": "KeyboardInterrupt",
  "zero_miss": {}
}
```
