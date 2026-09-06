# BF16 热专家常驻与共享缓存离线结果

Suite 状态：finished；CUDA 前置检查：passed。
Suite 执行 SHA：813196a4938d5fa04d6b489a7acaf3b44885b8d3。

R=原生 resident；B=专家缓存且实际 KV 等于 R；C=相同专家缓存、相同显存利用率下自动 KV。
实际吞吐由固定输出 token 数 / 测得秒数核对；未完成或损坏结果不产生增益结论。容量采用十进制 GB。

原生校准状态：complete。逐 rank 校准计数与完整输入哈希保存在同包 JSON/CSV。

每层 forward / unique demand 增量与覆盖率、状态最大值、嵌套 policy / timing、实际 kernel 选择和每 rank 容量均保存在 summary.json 与 evidence.csv。覆盖率分母为该层 measured forward 数乘逻辑专家数；最大值是累计状态快照，不做相减。
H2D=0 可以是全命中；命中率分母只含 cold cache hits+misses。命中率或正 H2D 均不能替代端到端收益。吞吐变慢同样是有效负结果。
正确性只由独立 CUDA parity tests 与 batch=1 smoke 各自覆盖；性能批量输出哈希单列。runtime weights_verified=0，未执行运行时大权重 D2H 验证。
模型身份绑定 config/index/本地路径哈希，未逐字节散列全 checkpoint。校准和评估按完整 prompt token 哈希分离，未声称源会话独立。
原始 prompt、生成文本、权重路径、机器身份及日志不在此导出包内。
