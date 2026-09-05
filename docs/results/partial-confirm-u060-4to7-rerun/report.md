# 部分 BF16 专家卸载离线吞吐结果

主指标为相同固定 token 工作量的端到端输出吞吐；所有容量以十进制 GB 展示。

| Run | Arm | 状态 | 卸载层 / slots | 输出 token/s 中位数 [最小, 最大] | 实际 KV 总 GB |
|---|---|---|---|---|---|
| 01-resident-o0-s1 | resident | complete | 0 / 1 | 1239.272 [1232.508, 1243.322] | 35.883319 |
| 02-partial-fixed-kv-o6-s2 | partial-fixed-kv | complete | 6 / 2 | 980.666 [973.016, 981.389] | 35.883319 |
| 03-partial-auto-kv-o6-s2 | partial-auto-kv | complete | 6 / 2 | 1243.272 [1231.397, 1254.880] | 48.731259 |

第 1 组：no-stable-net-gain。
B/R=0.7913；C/B=1.2678；C/R=1.0032。三轮稳定净收益=False。
性能批量输出哈希一致=False。

正确性结论仅覆盖相同输入的 batch=1 greedy smoke 输出哈希。性能批量的输出哈希另行比较；不一致时不声称性能批量输出等价。
缺失 RequestOutput.metrics 的时延记为不可用。CUDA 抽样计时只解释机制，不能替代端到端吞吐。短扫描用于选候选，不能证明三轮稳定收益。
模型身份由 config、权重索引和模型路径的哈希识别；未重读并计算全部权重内容哈希。初始化和 warmup 不进入 measured counter delta。
原始 stdout/stderr 留在服务器私有 runs 目录；本包仅含白名单数值、哈希、版本、相对 run ID 和结果文字。
