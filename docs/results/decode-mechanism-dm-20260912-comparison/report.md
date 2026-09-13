# 解码机制实测报告

模式：matched-resident；证据：measured；计时资格：eligible。

验证原因：通过。

整批输出重复变化审计：varied；变化不作为吞吐资格门槛，固定输出长度、有效hash、smoke与失败校验仍保留。

时间为包含 prefill 的整批生成墙钟；采集运行不进入正式吞吐。专家覆盖只针对 routed experts。TP 各 rank 的逻辑选择计数不能相加。

KV allocated 是实际分配量；scheduler usage 是采样占用比例，不能冒充精确已使用字节。CPU 等待和 GPU 本地服务间隔不可直接相加；本地 load span 不代表 TP 全局净损失。

保存 3 个数值样本、12 个 rank 采集。失败状态及数值见 report.json / report.csv。

三次以上有效重复的吞吐中位数：4785.9 output tokens/s。

配对比较状态：invalid-comparison。

B→C 实际batch观察：insufficient；指标可用性：incomplete；frontend scheduler：unavailable。

下表 worker 使用 rank0 一份逻辑计数，mean 按纯decode步骤加权；scheduler 使用独立frontend样本，二者不逐步对齐。batch未增加或scheduler不可用不会抹掉有效墙钟结果，指标齐全也不证明全部时间变化由准入导致。

| arm | rep | worker mean / decode steps / status | KV usage mean / samples / status | running mean / samples / status | waiting mean / samples / status | preemptions total / samples / status | output audit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0 | 490.249 / 930 / measured | 0.768103 / 1089 / measured | 480.641 / 1089 / measured | 223.428 / 1089 / measured | 132 / 1089 / measured | varied |
| A | 1 | 490.249 / 930 / measured | 0.768103 / 1089 / measured | 480.641 / 1089 / measured | 223.613 / 1089 / measured | 132 / 1089 / measured | varied |
| A | 2 | 490.249 / 930 / measured | 0.768103 / 1089 / measured | 480.641 / 1089 / measured | 223.542 / 1089 / measured | 132 / 1089 / measured | varied |
| native | 0 | 884.235 / 511 / measured | 0.673631 / 650 / measured | 805.231 / 650 / measured | 103.285 / 650 / measured | 0 / 650 / measured | varied |
| native | 1 | 884.235 / 511 / measured | 0.673631 / 650 / measured | 805.231 / 650 / measured | 103.203 / 650 / measured | 0 / 650 / measured | varied |
| native | 2 | 884.235 / 511 / measured | 0.673631 / 650 / measured | 805.231 / 650 / measured | 103.203 / 650 / measured | 0 / 650 / measured | varied |
