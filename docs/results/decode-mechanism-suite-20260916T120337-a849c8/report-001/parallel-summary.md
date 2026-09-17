# Native parallel comparison

Comparison: ineligible. Timing: global_wallclock.
Latency: arrival-to-finished offline; missing values remain unavailable.

| Config | Status | Median output tokens/s |
| --- | --- | ---: |
| tp4 | complete | 8082.893540944995 |
| ep4-dp4 | failed | unavailable |
| ep4-dp2 | failed | unavailable |

Per-GPU KV allocations and repetition latency counts/P50/P95 are retained in parallel-summary.json.
Native full-resident diagnostic; no offload or deployment gain claim.
