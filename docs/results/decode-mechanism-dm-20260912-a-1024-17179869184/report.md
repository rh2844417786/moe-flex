# 解码机制实测报告

模式：matched-resident；证据：measured；计时资格：eligible。

验证原因：通过。

整批输出重复变化审计：varied；变化不作为吞吐资格门槛，固定输出长度、有效hash、smoke与失败校验仍保留。

时间为包含 prefill 的整批生成墙钟；采集运行不进入正式吞吐。专家覆盖只针对 routed experts。TP 各 rank 的逻辑选择计数不能相加。

KV allocated 是实际分配量；scheduler usage 是采样占用比例，不能冒充精确已使用字节。CPU 等待和 GPU 本地服务间隔不可直接相加；本地 load span 不代表 TP 全局净损失。

保存 3 个数值样本、12 个 rank 采集。失败状态及数值见 report.json / report.csv。

三次以上有效重复的吞吐中位数：4785.9 output tokens/s。
