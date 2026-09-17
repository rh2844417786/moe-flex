# 一键实验汇总

三组全常驻并行对比与 TP4 A/B/C 分开解释。EP 卸载不支持。

本实验是诊断证据；不保证每点成功，也不保证卸载加速。详细采集是 instrumented，不进入正式吞吐。

KV 小点为当前 native 实际容量的一半并按实际字节 block 对齐；大点不超过四卡最小已观测容量；未尝试吃满释放权重空间。

并行比较：ineligible；TP4 A/B/C：unavailable。

并行吞吐使用所有 DP 的总输出 / 父进程同轮墙钟；可用请求延迟见 parallel-summary，缺失保持 null。

| 实验点 | 状态 | 尝试数 | 原因 |
|---|---|---:|---|
| build | complete | 1 |  |
| corpus | complete | 1 |  |
| parallel-tp4 | complete | 1 |  |
| parallel-ep4-dp4 | failed | 1 | child-failed |
| parallel-ep4-dp2 | failed | 1 | child-failed |
| cal-1024 | complete | 1 |  |
| cal-4096 | complete | 1 |  |
| native | complete | 1 |  |
| coverage-1024-1 | complete | 1 |  |
| coverage-1024-16 | complete | 1 |  |
| coverage-1024-50 | complete | 1 |  |
| coverage-1024-100 | complete | 1 |  |
| coverage-1024-200 | complete | 1 |  |
| coverage-1024-500 | complete | 1 |  |
| coverage-1024-1000 | complete | 1 |  |
| coverage-4096-1 | complete | 1 |  |
| coverage-4096-100 | complete | 1 |  |
| coverage-4096-1000 | failed | 1 | invalid-evidence |
| offload-1-0 | complete | 1 |  |
| offload-1-1 | complete | 1 |  |
| offload-1-2 | complete | 1 |  |
| offload-100-0 | complete | 1 |  |
| offload-100-1 | complete | 1 |  |
| offload-100-2 | complete | 1 |  |
| offload-1000-0 | failed | 1 | invalid-evidence |
| offload-1000-1 | failed | 1 | invalid-evidence |
| offload-1000-2 | failed | 1 | invalid-evidence |
| kv-a | complete | 1 |  |
| kv-b | complete | 1 |  |
| kv-c | complete | 1 |  |
| overhead-0 | complete | 1 |  |
| overhead-1 | complete | 1 |  |
