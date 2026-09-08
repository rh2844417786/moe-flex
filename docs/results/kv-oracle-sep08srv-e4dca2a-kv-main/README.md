# KV Oracle 容量诊断

仅诊断：diagnostic_only=true；formal_offload_gain=false。
O组显式KV可能超过r0名义0.60预算，不能算公平卸载收益或严格数学上界。
安全检查为估算加观测边界，不是硬显存上限；非Torch瞬态峰值可能未捕获。
表内显存使用十进制GB。请求观测是运行时批次成员数，不是等待队列；不可用字段为null。

状态：incomplete。
证据不足或尚未重复确认；不据此判断没有收益，也不授权卸载。

- r0 状态：failed；失败码：runtime-error。
- small 状态：failed；失败码：malformed-or-missing。
- large 状态：failed；失败码：malformed-or-missing。

输出正确性范围仅为batch1 smoke；性能批次哈希保留供检查。
正序和反序都要检查；screen单轮不提供稳定结论。
