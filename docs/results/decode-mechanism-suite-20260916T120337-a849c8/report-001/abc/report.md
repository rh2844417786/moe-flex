# 解码机制实测报告

模式：matched-resident；证据：measured；计时资格：eligible。

验证原因：通过。

整批输出重复变化审计：varied；变化不作为吞吐资格门槛，固定输出长度、有效hash、smoke与失败校验仍保留。

时间为包含 prefill 的整批生成墙钟；采集运行不进入正式吞吐。专家覆盖只针对 routed experts。TP 各 rank 的逻辑选择计数不能相加。

KV allocated 是实际分配量；scheduler usage 是采样占用比例，不能冒充精确已使用字节。CPU 等待和 GPU 本地服务间隔不可直接相加；本地 load span 不代表 TP 全局净损失。

保存 3 个数值样本、12 个 rank 采集。失败状态及数值见 report.json / report.csv。

三次以上有效重复的吞吐中位数：4744.08 output tokens/s。

配对比较状态：measured。

offload_overhead_s: 1339.25

kv_recovery_s: 484.039

matched_net_ratio: 0.114436

native_reference_ratio: 0.0683195

B→C 实际batch观察：observed-increase；指标可用性：available；frontend scheduler：measured。

下表 worker 使用 rank0 一份逻辑计数，mean 按纯decode步骤加权；scheduler 使用独立frontend样本，二者不逐步对齐。batch未增加或scheduler不可用不会抹掉有效墙钟结果，指标齐全也不证明全部时间变化由准入导致。

| arm | rep | worker mean / decode steps / status | KV usage mean / samples / status | running mean / samples / status | waiting mean / samples / status | preemptions total / samples / status | output audit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0 | 491.096 / 934 / measured | 0.802044 / 1092 / measured | 479.32 / 1092 / measured | 235.546 / 1092 / measured | 129 / 1092 / measured | varied |
| A | 1 | 491.202 / 934 / measured | 0.802041 / 1092 / measured | 479.32 / 1092 / measured | 235.662 / 1092 / measured | 130 / 1092 / measured | varied |
| A | 2 | 491.202 / 934 / measured | 0.802041 / 1092 / measured | 479.32 / 1092 / measured | 235.673 / 1092 / measured | 130 / 1092 / measured | varied |
| B | 0 | 491.202 / 934 / measured | 0.802041 / 1092 / measured | 479.32 / 1092 / measured | 236.149 / 1092 / measured | 130 / 1092 / measured | varied |
| B | 1 | 490.868 / 934 / measured | 0.802039 / 1092 / measured | 479.32 / 1092 / measured | 235.723 / 1092 / measured | 129 / 1092 / measured | varied |
| B | 2 | 491.202 / 934 / measured | 0.802041 / 1092 / measured | 479.32 / 1092 / measured | 236.149 / 1092 / measured | 130 / 1092 / measured | varied |
| C | 0 | 884.235 / 511 / measured | 0.673631 / 650 / measured | 805.231 / 650 / measured | 104.168 / 650 / measured | 0 / 650 / measured | varied |
| C | 1 | 884.235 / 511 / measured | 0.673631 / 650 / measured | 805.231 / 650 / measured | 104.168 / 650 / measured | 0 / 650 / measured | varied |
| C | 2 | 883.489 / 511 / measured | 0.673633 / 650 / measured | 805.231 / 650 / measured | 103.586 / 650 / measured | 0 / 650 / measured | varied |
| native | 0 | 883.489 / 511 / measured | 0.673633 / 650 / measured | 805.231 / 650 / measured | 102.538 / 650 / measured | 0 / 650 / measured | varied |
| native | 1 | 884.235 / 511 / measured | 0.673631 / 650 / measured | 805.231 / 650 / measured | 103.405 / 650 / measured | 0 / 650 / measured | varied |
| native | 2 | 884.235 / 511 / measured | 0.673631 / 650 / measured | 805.231 / 650 / measured | 103.028 / 650 / measured | 0 / 650 / measured | varied |
