# 解码批量、专家缺失与 KV 收益实测

用户已在聊天逐项澄清并批准本方案，包含编码、测试、GitHub 推送及服务器说明；允许扩充去重请求，但只使用服务器已有模型权重。本文记录批准内容，不增加重复审批或自动服务器运行。

## 目标

测量不同**实际纯 decode 批量**下每层各专家的 token 选择次数与单步覆盖，解释实际缓存缺失并测量其时间代价，再测量实际 KV 容量扩大是否通过增加并发带来吞吐收益。累计输出 token 数和提交请求数不能冒充实际 decode batch。当前离线预测不替代真实卸载数据。

## 固定边界

- 工作树 `/Users/a1234/code/flexmoe/.worktrees/repro-fluxmoe`，分支 `repro/fluxmoe`，拉取基点 `73939e57baf35ef680884d006068ea37c7074afb`；不改主工作树、历史实验结果或旧工具默认行为。
- 服务器 `/home/jovyan/wangtonghan/moe-flex`，四张独占 H100、TP4、已有 Qwen3-Next-80B-A3B-Instruct、BF16、原始路由；不下载新权重。
- Mac 可准备去重请求，优先复用固定版本缓存。服务器只通过 GitHub 同步和既有固定 Docker Hub 镜像获取；无新服务器依赖下载。所有服务器写入在项目内，`/mnt/public_data` 只读。
- 默认正常部署预算0.90、物理安全余量2,000,000,000 bytes/card；所有权重、KV、专家缓存、ingress、trace及工作区都计入实测显存。0.60/eager等明确为归因对照，不替代正常原生基线。
- 本轮不改写缓存算法、MoE CUDA kernel或增加猜测性预取。现有同计算流按需加载如实标注；没有 overlap 就不报告虚构隐藏时间。Mac检查与H100执行始终分开。

## 1. 数据与激活统计

新增独立数据包，1K、4K每桶至少1536个唯一token输入；原1024条数据不覆盖。固定source/tokenizer revision、文件hash、seed、packing与去重规则。优先已有 `.cache/sharegpt`，不足时才下载固定版本原始数据与tokenizer元数据，不下载权重。不同长度分别验证去重；校准32条与测试按完整输入hash隔离。禁止静默循环复制来凑1000或固定总请求数；不足显式失败。按token打包的压力输入不声称自然独立会话或代表性线上流量。

候选实际batch：1/16/50/100/200/500/1000。先1K覆盖扫描，4K只补代表点，不自动全笛卡尔积。默认采集目标256个纯decode步，至少64个完整步骤才足以汇总；收集256个匹配步即停止新增详细数据，生成继续。观察上限、跳过prefill/mixed/其他batch的数量、缺层/截断都保存，未达到目标batch标unreached，不按请求数重标。

每条激活记录为step、layer、phase、actual_batch、长度为num_experts的整数histogram；记录的是token→expert选择次数。一个专家被20个token选中计20，唯一专家覆盖计1，真实权重加载另计。纯decode要求CPU调度metadata证明一行一个请求、排除padding；每层histogram总和必须等于actual_batch乘top_k。保存分步计数而非逐token专家ID，不保存prompt、token IDs、logits。完整层网格用于每步覆盖分布；汇总含每层累计histogram、选择概率、coverage均值/分位数/范围、实际batch分布、phase样本数和计量分母。

采集单独运行并标instrumented，不当正式throughput。新采集入口与旧布尔occupancy入口隔离，保持旧消费者兼容。GPU固定有界int计数，结束后统一拷回；不新增每层CUDA同步或逐token D2H。

详细histogram/profile使用明确的matched-resident/eager或offload/eager模式；native优化模式只保留Graph-safe轻量实际batch/scheduler统计和正式测速。禁止将native+profile静默降级成eager：该组合必须提示改用matched-resident。这样不会把Graph捕获时绑定的采样槽反复重放并误标为多个步骤，但详细路由分布的适用范围是eager观察，不声称逐位重现Graph运行路由。

## 2. 真实卸载缺失与时间

复用真实ExpertPool、ExpertBackend和原MoE kernel。新显式mechanism模式记录每层每步resident命中、cache命中、去重miss、实际加载字节/次数、first-load与reload，及eviction/bypass等已知状态。首次/重载以本次engine生命周期（含prefill/warmup）的已加载集合判断，采样开始不将已见专家重置成冷加载。预取就绪状态独立于miss类别；当前无预取明确not-applicable。

真实CUDA事件记录同层load（区分专家payload与metadata如有可用事件）、compute和promotion，以及可测model-step span；CPU准备、route D2H同步、host reuse等待、policy、gather、enqueue分开。当前同流H2D在计算关键路径上，可报告其串行暴露服务时间；不能把host enqueue当DMA，不能把CPU和GPU重叠区间相加成总损失，不能把跨GPU时间加总。计时容量/采样覆盖和未测部分显式记录，不按少量事件外推整批。结束同步后读事件，不在每层为计时强制同步。独立整批墙钟对照是净减速依据。

固定KV与执行模式的resident/offload对照隔离卸载开销；另保留native optimized baseline。若需要eager，明确将native→eager的引擎损失与eager resident→offload的开销分开，不全归咎于专家miss。首轮真实cache配置有限：resident_ratio0.8/cache512，resident_ratio0.8/cache2048，resident_ratio0.9/cache512，policy=decayed-lfu；选覆盖扫描的低/中/最高可达batch，不全扫。配置改变前后净释放与实际KV都重测。

## 3. KV与最终吞吐

新的明确实验入口允许合法显式KV容量（以每rank实测allocated/blocks核对），不能放宽旧普通resident的KV override禁令。原生resident、匹配eager resident、真实卸载均保存真实policy、输入/模型/版本/SHA/四卡identity、profile/cache参数、实际内存和固定输出token数。

固定总工作量默认1024个唯一请求、输入长度、输出512、seed；共同max_num_seqs及max_num_batched_tokens足够大并固定，让容量改变可容纳的实际batch，而非人为改变调度上限。先预检共同可运行容量，分阶段取多个实际KV点；超安全预算/未达到批量/没有有效容量点保留失败，不能为出曲线强行OOM或捏造吞吐。A为resident指定KV，B为真实卸载同KV，C为同B策略增KV；native0.90优化参考独立保留。B/C必须同resident/cache/profile设置，仅KV改变；重复至少3次，同输入顺序，样本配对核验。吞吐只用低开销正式run；详细profile的时间不混入正式比较。

实际batch/phase/forward统计应覆盖native Graph和eager，而非只依赖可能被Graph绕过的model hook；从pinned运行时CPU调度入口或stat logger读取。记录可取得的真实scheduler KV usage/preemptions；确实无可用接口则明确unavailable及原因，绝不能把allocated KV冒充实际occupancy。所缺指标会出现在报告，不用null字段假装完成因果解释。

## 4. 结果与交接

原始新结果 `runs/decode-mechanism/<id>`，白名单公开结果 `docs/results/decode-mechanism-<id>`；全新ID、原子失败记录、独立smoke/failed-rep保留。公开逐层专家计数/覆盖、batch histogram、真实miss/timing/KV/吞吐及对照比值；允许有界计数直方图，不泄露文本/UUID/路径/日志。缺失/校验失败必须有安全分类原因，不能只吞成unreadable。

标准库host入口提供plan/validate/summarize/compare/export，GPU入口由固定root/offline/preflight/timeout wrapper执行。旧结果保持不变。结果区分measured、instrumented、predicted；新主比较不调用旧成本模型生成时间结论。只有通过身份/固定工作量/实际KV/执行模式/重复与失败检查的真实数据才能输出本工作负载的measured net ratio；不自动宣称所有部署都有收益。

中文runbook/server prompt给出完整分阶段命令、图表生成/数据回传说明、无重复输入gate、校准隔离、native/eager关系、实际batch不可达、计时覆盖与安全预算。不自动运行服务器。完工须独立审查、完整CPU测试、Ruff/Mypy、shell与wheel检查、推送既有分支并核对SHA。
