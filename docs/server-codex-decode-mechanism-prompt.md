# 交给服务器 Codex 的执行说明

请在 `/home/jovyan/wangtonghan/moe-flex` 的 `repro/fluxmoe` 分支执行已批准的解码机制实测，先阅读 `docs/decode-mechanism-runbook.md` 与绑定设计 `docs/superpowers/specs/2026-09-12-decode-mechanism.md`。Mac 已完成的 CPU 测试只验证实现，不是 H100 性能证据。

使用 `git pull --ff-only origin repro/fluxmoe` 获取已提交实现，记录 SHA，按既有 `scripts/server/build.sh` 构建/核对固定镜像，然后整个实验组保持同SHA。GPU仅用四张独占H100、已有 Qwen3-Next-80B-A3B-Instruct 权重、TP4 BF16；不下载权重、依赖或tokenizer，不写项目外，不改/mnt/public_data，不终止别人的任务。所有GPU点经 `run_decode_mechanism.sh`，复用独占预检、离线项目缓存和容器内timeout。重复尝试用新ID，保留所有失败/部分采集/smoke/日志。

按runbook的有限手动阶段推进：验证已提交1K/4K各1536唯一输入；每种长度分别用前32条校准；跑native0.90普通参考；matched-resident/eager详细profile做1K实际batch1/16/50/100/200/500/1000扫描；4K只选最多三个代表点。检查实际puredecode batch、至少64个完整层网格和256步窗口；未达到标unreached，不把总请求数当batch。详细profile只作instrumented观察。

从真实四rankKV和物理内存证据选共同安全容量，再通过计划显式输入KV预算。有限卸载点用已达到的低/中/最高batch、三种批准缓存配置；检查真实per-rank pool大小和首次/重载/miss/CUDA本地服务。全命中窗口可零payload H2D。无预取，不计算虚构overlap；CPU等待和GPU区间不相加，rank时间不当TP全局损失。

KV阶段固定1024唯一测试请求、输出512、3次重复、共同调度上限。A=matched-resident小KV，B=同KV卸载，C=相同缓存/profile的更大实际KV。候选预算必须来自当前阶段真实检查；没有安全点就保存失败。native0.90参考独立，另做相同KV的profile开销配对。generation_status=complete但observer/capture/memory/measurement失败的样本仍不合格，保留其数值。

用 `decode_suite.py compare` 严格核对身份、版本、工作量、实际KV、执行模式和重复，再用 `export --source A --b B --c C --native N --output ...` 生成白名单JSON/CSV/Markdown与仅基于合格实际数据的SVG。报告应说明本工作负载的实测收益/退化/无结论，以及所有缺失指标；不得把缺失结果补零或把旧离线预测当实测。检查 `docs/results/decode-mechanism-*` 无prompt、token IDs、UUID、路径、日志；只提交本次明确的公开目录，普通push到repro/fluxmoe，不创建PR或合入main。回报最终代码SHA、实际执行命令、完成/失败点、A/B/C/native实测对照、缺失证据和公开结果路径。
