# N4 · AgentX 的来历、定位、局限与批评

调研日期:2026-11-10。口径说明:每条事实后的方括号是来源编号,见文末"来源"。标"二手"的是转述或聚合站;标"自述"的是 SemiAnalysis 自己的说法。

---

## 1. 版本沿革

### 1.1 InferenceMAX v1(2025-10)

- 2025-10-09 前后发布,文章里的成绩快照取自 2025-10-07 的夜间运行;Apache 2.0;用 GitHub Actions 把 GPU 服务器挂成 runner,每晚重跑 [S3][S4]。
- 立项理由(自述):硬件一年一代,软件每天在变,固定时点的基准很快过时 [S3]。
- 首发硬件:GB200 NVL72、B200、H200、H100、MI355X、MI325X、MI300X;当时承诺"两个月内"加入 Google TPU 和 AWS Trainium [S3]。
- 首发模型:Llama 3.3 70B、DeepSeek R1(670B MoE)、gpt-oss 120B;引擎 vLLM、SGLang、TensorRT-LLM,每个模型先只定一个默认引擎,理由之一是避免"SGLang vs vLLM 基准战" [S3]。
- 三个固定形状场景:1k/1k(聊天)、1k/8k(推理)、8k/1k(摘要);每次请求输入长度在标称值的 80%–100% 间随机 [S3]。
- 负载生成:去掉 vLLM 依赖的 vLLM benchmark serving 脚本;请求率设为无穷大,只限制最大并发(即闭环定并发)[S3]。
- **当时就承认的缺口**:用随机 token 序列、故意避开前缀缓存,理由是前缀缓存"因负载而异",需要先调查真实请求模式才能选出有代表性的前缀比例;并说以后会改用 ShareGPT 之类数据集 [S3]。
- 指标框架从 v1 就定了:吞吐(tok/s/GPU)对交互速度(tok/s/user)的帕累托前沿,再用 SemiAnalysis 自家 AI Cloud TCO 模型换算成每百万 token 成本、用 Datacenter 模型换算成每 MW token;功耗当时按 TDP 加总估算,不是实测 [S3]。
- v1 的规则漏洞实例:没写清是否允许预热,NVIDIA 的 SGLang DeepSeek 提交带了预热,AMD 发现后询问;三方商定暂时禁止预热、把 DeepSeek 测试时长最多拉长 5 倍。SemiAnalysis 文中承认是自己没讲清规则 [S3]。
- 自评:v1"远不完美",以后要改进负载 [S3]。
- 媒体:Tom's Hardware 2025-10 的报道基本转述官方说法("尽量中立"),没有提出独立质疑 [S5]。

### 1.2 InferenceX v2(2026-02,改名)

- 2026-02 发布,同时由 InferenceMAX 改名 InferenceX [S1][S6]。HN 上的转帖日期为 2026-02-16 [S20]。
- 新增:DeepSeek MoE 的大规模分离式推理(disagg prefill)加 wide-EP,覆盖 NVIDIA 近 4 年全部 6 个 SKU 和 AMD 近 3 年全部 SKU;首次测 GB300 NVL72 和 B300;首次由第三方测 MI355X 多节点 disagg+wideEP;完整一轮用近 1000 块 GPU [S6]。
- 当时预告:年内加入 TPUv7 Ironwood 和 Trainium3;DeepSeek v4 等模型 day-0 支持 [S6]。
- **v2 文中自己写的局限**:数据应理解为"基线"而非真实推理的完整代表,因为用随机数据且关闭了前缀缓存;换句话说真实性能"至少这么好" [S6]。
- v2 文中已讨论合成数据对 MTP(多 token 预测)接受率的影响,AgentX 文章回指了这一节 [S2][S6]。
- 分离式配置下"每 GPU 输出吞吐"按 decode GPU 数归一、而不是按总 GPU 数 [S6];这一口径在 issue #299 被外部用户质疑(见 4.10)。

### 1.3 中间阶段(2026-03 至 2026-08)

- 2026-03-27:1k8k 场景对所有模型弃用,理由是把集群时间让给"更高优先级的真实智能体编程基准和新前沿模型" [S9]。
- 2026-07-17:1k1k 场景弃用(同样理由);最后一个 1k1k 配置于 2026-10-01 退役 [S9]。
- 8k1k 仍是"主要的固定序列场景",状态为 Active;但文档写明新模型今后大概率只上 agentic coding [S9]。
- AgentX 的 MVP 数据集至少迭代了 5 版:`cc-traces-weka-042026`(MVP v0.1)、`cc-traces-weka-no-subagents-051226`(v0.2)、`with-subagents-052726`、`with-subagents-060226`(及 `-256k`)、最终 `cc-traces-weka-062126` [S13][S2]。
- MVP 期间数据集变动对结果影响很大:AMD 工程师在 issue #1632 里离线算出,v0.1 数据集理论前缀命中率约 96.8%,v0.2 无子智能体数据集约 85.8%;加上 `MAX_CONTEXT=131072` 后分别为约 96.0–96.2% 和约 65.6–69.9%。维护者回复:MVP 阶段允许破坏性变更,1.0 之后不再有 [S13]。
- `agentx-harness` 仓库(ai-dynamo/aiperf 的 fork)创建于 2026-05-11;v1.0.0 标签打在 2026-08-03,说明是"最后一批破坏性变更" [S12]。

### 1.4 InferenceX v3 + AgentX 1.0(2026-08-24)

- 新增的是 AgentX 场景本身,与原有固定序列场景并存 [S2]。
- 发布时硬件:MI355X、GB300 NVL72、GB200 NVL72、B300、B200、MI325、MI300X、H200、RTX Pro Server;文中预告 Rubin"本月稍晚",TPU 和 MI455X UALoE72"今年稍晚" [S2]。
- 数据口径补充(已知事实之外):
  - 母语料 8,271 个会话、341 万请求、约 6,133 亿 token,仪表板显示命中率 99%;"300 万美元"是这批语料按列表价估算的 token 花费,不是基准输出 [S2][S8]。
  - 母语料来自 Claude Code、OpenAI Codex 等;公开的 393 会话子集是 Claude Code [S2][S7]。
  - 入选条件:至少 20 个请求、Claude Code 2.1.139 或更新、并发子智能体不超过 10 个;数据集构建于 2026-06-21 [S8]。
  - 后处理:去掉完全重复的请求、Claude Code 的安全监控(auto mode)和标题生成请求、重建输入超过 990k token 的请求 [S2][S8]。
  - 256k 变体保留 68,266 个请求,输入中位数 88,768、输出中位数 376 [S8]。
  - 175 个会话含子智能体,共 1,697 个子智能体组,每会话中位 4 个,单组墙钟时间中位 2.27 分钟,256k 变体里 p95 为 18.5 分钟 [S2][S8]。
  - 回合间隔中位 3.84 秒,约 10% 超过 1 分钟(推测是在等人)[S2]。
  - 重建 token 数与供应商实际计数之比:393 个会话、135,282 个请求上中位数 1.004 [S8]。
  - vLLM 博客引用:每会话中位 43 回合,前缀缓存命中率 96% 以上 [S10]。
- 为拿到子智能体结构,SemiAnalysis 称与 Anthropic 合作在 Claude Code 里加了请求头 `x-claude-code-agent-id` 和 `x-claude-code-parent-agent-id`(claude-code issue #49207、#66761)[S7]。
- 追踪格式是 WEKA trace format,由 Callan Fox 的 kv-cache-tester 项目提出 [S2]。

### 1.5 1.0 之后的小版本

- 维护方针(自述):近期只对 v1.0.x 做小修,大的改动留给 v1.1;同一模型的所有提交始终跑同一个 minor 版本,以保证可比 [S2]。
- 截至 2026-11-10,发布页只有 1.0.x,最新 v1.0.13(2026-11-03);没有看到 v1.1 发布 [S12]。
- 各小版本要点 [S12]:
  - 1.0.1–1.0.5(2026-08-07 至 08-12):加运行后有效性校验(TTFT/ITL 观测须覆盖剖析窗口的 98%,后对低并发放宽到 95%);加 `full_response_inter_token_latency` 等整响应口径指标;修并发读数据集的 mmap 竞争。均声明不影响性能结果。
  - 1.0.6(10-01):新增 `--scenario agentx`,带标准 InferenceX 默认值;保留旧的 `inferencex-agentx-mvp`。
  - 1.0.7(10-12):发布到 PyPI(`pip install agentx-harness`),新增 `agentx` 命令。
  - 1.0.8–1.0.11(10-14 至 10-15):降低实时统计开销;修"到时截止后子智能体挂起";修子智能体在客户端 worker 间迁移。
  - 1.0.12(10-19):修预热基线标注,使服务端计数器总量不含预热。
  - 1.0.13(11-03):并入上游 aiperf 修复,声明调度、请求内容与顺序、客户端指标不变。
- 1.0 之后的规则变化(不在 harness 版本里,而在仓库规则里):
  - 流式间隔上限:每个新增或重跑的 AgentX 点,有效 stream interval 不得超过 20 token。原因是指标在客户端测,短于间隔的回复会一次到达,TTFT 吞掉解码时间;间隔 100 时 MiniMax M3 负载约 30% 的请求受影响。由 PR #2685(2026-08-21 合并)引入,数值 20 来自 NVIDIA TRT-LLM 工程师建议 [S11][S14]。
  - 主横轴改为"E2E normalized interactivity":对每个请求算 `r_i = E2EL_i / OSL_i`,取第 q 百分位后求倒数,默认 P90;近似 `1/(TPOT + TTFT/OSL)`。以它算出的前沿是"North Star"前沿,其他横轴视图只显示同时在 North Star 前沿上的点 [S9]。1.0 文章里这个指标还标为"实验性、不完美" [S2]。**改为默认的具体日期没查到。**
  - 草稿模型精度规则("serve the draft as it ships")、禁止整数量化 all-reduce(新模型)、Kimi K3.1 Pro 首周精度政策等 [S9][S11]。
- 模型轮换很快:DeepSeek V4 Pro 和 MiniMax M3 于 2026-10-29 全部退役,Kimi K3 于 2026-11-03 退役;在跑的是 DeepSeek-V4.1-Flash(2026-09-10 加入)、GLM-5.2/5.3/5.3-Flash、Qwen3.8-Flash-Next、Muse Spark Open、MiniMax-M3.1-Flash、Kimi-K3.1-Pro 等 [S9][S1]。

### 1.6 公布的路线图

- 快速跟进:SSD/NVMe KV 卸载 [S2];仓库文档写的是"暂定 InferenceX v3.5 或 v4" [S9]。届时可能放宽每流 5 分钟空闲上限以测更长 TTL [S2]。
- 新数据集:更大、更多样、更新,覆盖更多模型和 harness;保留系统指令、用户/助手消息、工具调用、工具结果之间的边界,以便评估"负载感知"的服务技术(例如把低复用的工具流量导到专用 prefill worker、长工具调用期间保留前缀、子智能体 fork 时预取前缀)。当前格式只有扁平的哈希块序列,做不了这些 [S2]。
- 新指标:要定义一个把 TPS 和 TTFT 合起来、并考虑"任务端到端完成速度"的指标 [S2]。
- 功耗:采集细粒度和粗粒度功耗遥测 [S2];2026-11 起多节点运行默认采集实测功耗(PowerX)[S1]。
- 硬件时间线:
  - Vera Rubin NVL72:2026-09-14 发首批 AgentX 结果(预发布 TRT-LLM 软件);2026-09 列为"正式支持" [S15][S1]。
  - TPUv7x Ironwood、RTX PRO 6000 Server:2026-09 列为正式支持 [S1]。但 2026-09 的 TPU 文章只跑了 Qwen3.5 的 8k1k,AgentX 的 TPU 结果"今年稍晚"(二手,来自批评文章的转述)[S17]。
  - MI455 UALoE72、Rubin NVL8:Coming Soon;另有 4 家未具名厂商的 5 款芯片 Coming Soon [S1]。Rubin 文章称 SambaNova 和 Trainium"很快" [S15]。
- 多模态、非编程类智能体、更长会话:路线图里只有"更多样的追踪、更多 harness"这类笼统说法,没有具体承诺(见"没查到")。

---

## 2. 智能体流量为什么不同

### 2.1 AgentX 要抓住的性质(自述)

- 四条:多回合(几十到几百次交互)、长上下文、高前缀复用(第 n 回合输入 = 第 n-1 回合输入 + 输出,缓存占比随 n 趋近 1)、子智能体突发扇出(造成尖峰式的 KV 分配与淘汰)[S7]。
- 固定序列基准做不到的事:一条提示、一次 prefill、一段固定续写、然后丢弃;不测跨回合缓存存活、重复分词、会话亲和、卸载抖动、调度停顿期间的传输进度、长期所有权记账 [S2]。
- 固定序列与智能体对照表(官方):请求形状固定 vs 每回合变;独立到达 vs 由前一请求完成触发;前缀复用可选 vs 大段共享前缀增长;无工具时间 vs 不均匀停顿;请求级并发 vs 主/子智能体分支重叠;缓存寿命短 vs 长期状态争用 HBM 和卸载容量 [S16]。
- 并发的含义变了:是"存活的智能体客户端数",不是固定请求批;并发 50 时在途 HTTP 请求可以超过 50 [S16][S8]。
- 官方仍说固定序列数据有用:它剥掉会话层行为,反映底层芯片和内核性能,是读 AgentX 结果的基线 [S7][S2]。
- 一个说明"测量形状决定结论"的例子:某 AITER GEMM 改动在固定序列低并发下有 2.3% 增益,在智能体追踪上被淹没 [S2]。

### 2.2 其他追踪研究:支持的证据

- **TraceLab(华盛顿大学,arXiv 2606.30560,2026-06)**:约 4,300 个编程智能体会话、约 35 万 LLM 步、43 万次工具调用,来自作者团队日常使用 Claude Code 和 Codex。结论与 AgentX 画像一致:长自主循环、长上下文短输出、工具调用重尾、前缀命中率"高但不完美" [S22]。搜索摘要给出的全局命中率 95.7%、未命中主要发生在等人输入的停顿(二手,未在原文核对)[S22]。
- **CacheWise(arXiv 2606.16824,2026-06)**:编程智能体是长时间闭环会话,LLM 生成与外部工具调用交替;反复复用大前缀,对 KV 缓存形成持续压力,常规服务策略处理不好 [S23]。这也支持"闭环"是该负载的本来面目。
- **vLLM 团队实测**:在 AgentX 上,按队列深度、在跑 token 数或 KV 利用率做负载均衡,都不如简单的会话粘性路由,原因是回合间隔短、下一回合到来时前缀还在原 GPU 上 [S10]。
- **OpenAI 披露(经 vLLM 博客转引)**:截至 2026-06,企业客户中 Codex 产生了 Codex+ChatGPT 合计输出 token 的 64% [S10]。
- **Artificial Analysis** 2026-11-03 的 3.0.0 版也加入了多回合智能体负载并报告缓存命中率,说明"单请求固定长度不够"已是行业共识 [S19]。
- **MLPerf Inference v6.1(2026-09-16)** 加入 Edge Agentic 基准(上下文增长、反复推理),并称数据中心版智能体基准即将推出 [S24][S25]。

### 2.3 其他追踪研究:不一致或需要保留的地方

- **Qwen-Bailian / "KVCache in the Wild"(USENIX ATC'25,arXiv 2506.02634)**——AgentX 脱敏方法借鉴的对象:
  - 方法差别:Qwen-Bailian 用 16 token 一块、加盐 SipHash-2-4、再映射为顺序整数;哈希的是套完 chat template 之后引擎实际消费的 token。AgentX 用 64 token 一块的会话内链式哈希,且哈希的是客户端代理看到的内容,供应商侧模板、分词器、服务端工具都看不到 [S21][S8]。
  - 结论差别:该论文发现复用在请求间是偏斜的,**单回合请求之间的复用与多回合同样重要**;达到理想命中率所需缓存容量"适中" [S21]。AgentX 的哈希作用域是会话内,每次重放还加唯一 cache-bust 标记,跨会话复用(例如所有会话共用的系统提示和工具定义)被有意排除 [S2][S8]。这是 AgentX 相对生产环境**低估**缓存收益的一处。
  - 该追踪是 2 小时采样、来自生产集群的 To-C、To-B、Thinking、Coder 四类流量,带真实到达时间戳,官方回放器是"忠实于时间戳"的开环回放 [S21]。
  - SemiAnalysis 自述试过 SWE-bench、Qwen-Bailian 和 HuggingFace 上的 Claude Code 追踪,认为它们缺少子智能体、1M 上下文、压缩(compaction)等新特征,才决定自采 [S2]。
- **BurstGPT(arXiv 2401.17644)**:Azure OpenAI 区域服务 213 天、1,031 万条追踪,强调请求并发的突发性和服务间差异 [S26]。AgentX 的突发只来自会话内部(子智能体扇出、工具间隔),没有人群层面的到达突发和昼夜波动。
- **ServeGen(阿里,arXiv 2505.09999)**:主张按"每个客户端"组合来生成负载,称朴素的负载生成会造成 50% 的容量低配 [S27]。AgentX 的 393 个会话来自一家公司的员工,客户端异质性无从体现。
- **命中率的敏感性**:issue #1632 表明同一来源、不同批次的 Claude Code 追踪,理论命中率可从约 97% 掉到约 86%,原因是更长的追踪和更多的上下文压缩/重置 [S13]。"95%+ 命中率"是这一批数据的性质,不是智能体负载的常数。
- **TTL**:AgentX 每流空闲上限 5 分钟,理由是 Anthropic 默认缓存 TTL 为 5 分钟 [S2];TraceLab 指出未命中恰恰集中在人为停顿之后 [S22]。把长间隔截到 5 分钟会让这类未命中的形态与真实情况不同。

---

## 3. 对比表素材

| 维度 | AgentX 1.0 | InferenceX 固定场景(8k1k 等) | MLPerf Inference(数据中心 LLM) | Artificial Analysis 性能基准 | AIPerf / GenAI-Perf 合成模式 | vLLM bench / GuideLLM / LLMPerf |
|---|---|---|---|---|---|---|
| 负载来源 | 393 条真实 Claude Code 会话的形状,内容合成 [S8] | 随机 token,固定输入/输出长度 [S3] | 固定数据集加参考实现;v6.1 新增端到端 RAG(数据中心)和 Edge Agentic(20 条编程轨迹回放 + BFCL v4)[S24][S25] | 多样化真实提示;1k/10k/100k 单请求加多回合智能体任务(v3.0.0,2026-11-03)[S19] | 按均值/方差生成合成提示,也能回放追踪(Mooncake、WEKA 格式)[S2][S18] | 合成或公开数据集(见脚注)|
| 多回合 | 是,DAG(主链、子智能体、辅助请求)[S8] | 否 | 数据中心 LLM 场景为单请求;Edge Agentic 是单流多回合 [S25] | 是(3.0.0 起),真实执行任务 [S19] | 取决于模式 | 部分工具有多回合脚本(脚注)|
| 缓存建模 | 核心:前缀哈希保留复用关系;cache-bust 防跨会话命中;预热建立缓存 [S8] | 明确关闭/避开前缀缓存 [S3][S6] | 无专门建模(未核对最新规则)| 报告供应商 API 返回的缓存命中率 [S19] | 可配置 | 一般不建模 |
| 到达模型 | 闭环,定并发会话数 [S8] | 闭环,定并发请求数 [S3] | Server 场景为泊松开环到达,Offline 为批量(MLPerf 既有设计,本次未重新核对)| 持续并发负载 [S19] | 两者皆可 | 两者皆可(脚注)|
| 谁来跑 | SemiAnalysis 的 CI 在其掌握或受赠的硬件上跑;配方由厂商工程师写并签核 [S4][S11] | 同左 | 提交方自己跑,MLCommons 同行评审 [S25] | AA 以普通开发者身份调公开 API [S19] | 用户自己 | 用户自己 |
| 开放程度 | harness、数据集、配方、CI 日志公开(Apache 2.0);前端 AGPL-3.0;文章版权保留 [S1][S16] | 同左 | 规则、参考实现和结果公开 | 方法公开,提示集和原始数据不全公开 | 开源工具 | 开源工具 |
| 硬件覆盖 | NVIDIA(H100 到 Vera Rubin)、AMD MI300X–MI355X、TPUv7x(AgentX 结果状态不明)[S1] | 同左 | 各提交方自选,v6.1 有 30 家组织、486 项结果 [S24] | 不直接测硬件,测 API 端点 | 任意 | 任意 |
| 测什么 | 每芯片吞吐、P90 交互速度、TTFT、每美元/每 MW token [S16] | 同左 | 吞吐(满足延迟约束),带准确率门槛 | 输出速度、首 token/首答 token 时间、端到端时间、错误率 [S19] | 延迟与吞吐分布 | 延迟与吞吐分布 |
| 是否检验输出质量 | AgentX 本身不检验;同一配置另跑 GSM8K 等 eval 作门槛 [S16][S11] | eval 门槛 | 是,准确率必须达到参考值的一定比例 [S25] | 另有 Endpoint Accuracy Index [S19] | 否 | 否 |
| 更新频率 | 连续(配方变更触发)[S11] | 连续 | 半年一轮 | 每 6 小时 [S19] | 不适用 | 不适用 |

脚注(本次没有重新抓取文档,凭既有了解,写报告前应核对):vLLM 自带 `vllm bench serve`,可用 random、sonnet、ShareGPT、BurstGPT 等数据集,另有多回合基准脚本;GuideLLM(vllm-project 下)支持 synchronous、concurrent、constant、poisson、sweep 等速率模式和合成/数据集输入;LLMPerf(Ray 项目)用按均值和标准差采样的合成提示加并发请求数。GenAI-Perf 已由 AIPerf 接替。

对比时值得写进正文的几点:

- AgentX 与 MLPerf 在投机解码规则上有交集:InferenceX 的草稿精度规则明说"与 MLPerf Inference Rules 附录 C 同一原则"(参考 MTP 头须按原精度使用、禁止操纵接受率),但不采用其量化边缘例外、允许模型清单和接受率方法 [S11]。
- AA 与 AgentX 是互补关系:AA 测的是"你买到的 API 实际多快",黑盒、不知道硬件;AgentX 测的是"给定硬件和开源栈能做到多快",白盒、但不是任何一家的生产配置。AA 还有一套"诚信条款",禁止供应商识别其流量并特殊对待 [S19];AgentX 没有这个问题的对应物,因为厂商本来就知道自己在被测并亲自调参。
- AA 的输出速度含 TTFT、多回合时不含工具时间 [S19];AgentX 的"E2E normalized interactivity"思路相近 [S9]。两者数值不可直接比。
- AIPerf 上游文档里的 `inferencex-agentx-mvp` 场景与官方 1.0 口径不完全一样:默认 1800 秒(最短 900 秒)、全系统空闲上限 10 秒、起点比例 0.0–1.0 [S18];官方 1.0 是 1 小时、每流空闲上限 5 分钟、起点 25%–75% [S2][S8]。自己复现时用哪个场景要说清。

---

## 4. 局限与批评(逐条注明是谁说的)

### 4.1 单一追踪来源

- **事实**:393 个会话全部是 SemiAnalysis 内部员工自愿上传的 Claude Code 会话 [S7][S8]。
- **SemiAnalysis 自己承认**:请求分布随 harness 而变(举例 Pi 注入上下文极少、Claude Code 相反),也随模型的分词器而变;辩护理由是"全球相当大一部分智能体编程流量走 Claude Code" [S2]。
- **筛选偏差**(本笔记的推论,依据入选规则):至少 20 个请求、并发子智能体不超过 10 个、Claude Code 版本下限,这些规则把短会话和极端扇出的会话排除在外 [S8]。
- **追踪来自 Claude 模型,重放给别的模型**:工具调用节奏、输出长度(中位 444)、何时开子智能体,都是 Claude 加 Claude Code 的行为;被测的开源模型在同样任务下会走出不同的轨迹。官方只做了 token 数的校准(比值中位 1.004),没有行为层面的校准 [S8]。此点没有找到外部人士明确提出,是本笔记的观察。
- **来源说法不一致**:AIPerf 上游文档说语料是 Callan Fox(kv-cache-tester)采集的 Weka 智能体编程追踪 [S18];SemiAnalysis 说是自家代理采集、只是采用了 Callan Fox 提出的 WEKA 格式 [S2]。两种说法都记下,以 SemiAnalysis 自述为准的可能性更大。

### 4.2 合成 token 内容

- **官方承认**:合成内容下,投机解码的接受/拒绝数会异常,因为草稿模型没在合成数据上训练过 [S2]。
- **SPEED-Bench 论文(NVIDIA 等,arXiv 2604.09557)**:投机解码性能本质上依赖数据;合成输入会**高估**真实吞吐 [S28]。
- **官方补救**:强制接受长度。对每个(模型、草稿方法、草稿长度、思考模式)组合,在 SPEED-Bench 编程类数据上测平均接受长度,写入仓库里的"golden AL"文件,运行时让引擎按这个值接受 [S2][S8][S11]。
- **补救本身的局限**:
  - 用的是平均值,真实接受长度随上下文位置和内容波动;提交方可自选草稿长度但不能换接受目标 [S11]。
  - 参考值统一用 vLLM 生成 [S13 之 issue #1651]。
  - 仓库规则明说"合成 AgentX 接受率不能作为任何证据",草稿精度降低带来的质量损失会转移到接受率上,而目标模型的 eval 测不到 [S11]。
  - AMD 工程师在 issue #3841 中质疑:两个 ROCm 目标侧优化(INT4 quick reduce、a4w4 MoE)因为是进程级开关、会波及草稿模型而被规则拒绝;他们用 GSM8K 实测接受长度(3.912 对 3.891,差 0.55%,小于基线自身波动)要求作为证据被采纳 [S13]。该 issue 已关闭,最终裁定没查到。
- **对其他内容相关技术的影响**(官方只提了投机解码;以下是本笔记的推论,没找到外部批评原文):凡依赖 token 内容的方法——基于注意力稀疏性或压缩的 KV 方法、内容感知的缓存淘汰、提示压缩、采样参数对输出长度的影响——在合成内容上的表现都不能代表真实表现。重放时锁定 `ignore_eos:true`,输出长度被强制为追踪记录的值 [S18],所以"模型自己决定何时停"这件事也不在测量范围内。
- Artificial Analysis 在其方法说明里也写:提示多样性重要,因为投机解码等技术让输出速度随输出类型而变 [S19]。

### 4.3 闭环重放 vs 开环到达

- **官方承认**:AgentX 是闭环基准,不同配置在同一小时内完成的请求数不同,遇到的负载组合有自然差异,低并发时最明显 [S2][S8]。官方文章自己链接了一篇题为"Open vs Closed Loop: A Benchmarking Crime"的博客 [S29]。
- **那篇博客的论点**:闭环下被测系统越快、客户端发得越多,快的客户端贡献更多数据点,延迟分布被拉向乐观;闭环也无法精确控制施加的负载,不同系统间不好比 [S29]。
- **官方辩护**:智能体不可能预先发出所有回合,后一步依赖前一步结果;闭环保留了"服务器越快、会话越快产生后续工作"的关系 [S16]。CacheWise 论文也把编程智能体描述为闭环会话 [S23]。
- **外部复现者的实例**:issue #1580 中,一位用户复现 DeepSeek-R1 H100 分离式结果时 decode 更快(1k/1k 并发 64 下输出吞吐 2,639–2,940 对官方 1,564.8 tok/s),在闭环下更快的 decode 把请求更快地喂回 prefill,放大了 TTFT 排队;改成开环并用相近到达率后 TTFT 才接近官方值 [S13]。这是针对固定场景的,但机制对 AgentX 同样成立。
- **本笔记的补充**:会话内部闭环是合理的,但"会话的到达"在真实服务里是开环的(新用户不会因为服务器慢就不来)。AgentX 用定并发加会话回收,没有会话级到达过程,因此测不出过载、排队崩溃和准入控制行为。

### 4.4 一小时窗口

- 只统计预热后的一小时;每流空闲上限 5 分钟是为了"能在 1 小时内跑完" [S2]。
- 预热从每个会话 25%–75% 处切入,加 10 个预热请求,预热请求不带回合间隔、最多输出 1 个 token [S2][S8]。
- 后果(推论):超过 1 小时的会话尾部、跨小时的缓存淘汰与碎片化、DRAM 卸载层填满后的稳态,都观察不全;子智能体组 p95 时长 18.5 分钟,一小时里这类长分支样本有限 [S8]。
- 低并发时样本少、收敛差,这点官方自己说了 [S2]。
- harness 需要专门的有效性校验(TTFT/ITL 观测覆盖 98%/95% 的窗口),说明窗口尾部稀疏是实际遇到的问题 [S12]。

### 4.5 393 个会话的代表性

- 母语料 8,271 个会话,公开的是"有代表性的子集",占约 4.8%;挑选标准只公布了硬性过滤条件,没有公布抽样方法或与母语料分布的对照 [S2][S8]。
- 并发可以超过 393:会话被回收重用,每次重放加不同的 cache-bust 标记 [S2]。高并发点上同一批形状被反复使用。
- 时间戳不等于因果:HTTP 时间戳看不出主智能体在等子智能体还是在做别的事,AIPerf 两个约束都保留;官方称这是"小局限",希望后续版本改进 [S2]。
- 供应商侧不可见内容:聊天模板、专有分词器、服务端工具、加密的推理内容、图片和文档的 token 展开,都用占位符和按模型校准的填充近似 [S2][S8]。

### 4.6 厂商针对基准调优("benchmaxxing")

- **批评者**:Hacker News 用户 latchkey(2026-08-26)称 AMD 和 NVIDIA 把最好的工程师放在为基准调参上,目的是"在图上好看以卖芯片";同一用户 2026-02-14 称这类第三方基准"都是厂商高度调过的" [S20]。措辞激烈,且该用户与 AMD 云生态有关联(未核实),应当作一方意见。
- **GitHub issue #99(patrick-toulme,2025-10-12)**:B200 的配方开了 AMD 配方没有的编译/融合标志和性能标志,比较的其实是引擎和图编译器而非纯硬件,应当明说 [S13]。
- **SemiAnalysis 自己的话**:"About"页写,静态基准的参与者常提交"专为基准构建"的镜像,不反映真实性能——以此作为 InferenceX 存在的理由 [S4]。
- **事实层面**:配方由厂商工程师编写,CODEOWNERS 里 NVIDIA 和 AMD 各有 5 名具名工程师负责各自的主配置和配方目录 [S30];规则允许每个提交逐点选择是否开 KV 卸载、是否开投机解码、用哪种并行方式,以取最好的帕累托点 [S9]。一条曲线上的不同点可以用不同的优化组合 [S2]。
- **被规则挡下来的实际做法**(说明调优压力真实存在):用 `--hf-overrides` 跳过部分层的 indexer;对草稿头做低于出厂精度的量化(`SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE`、`quark_mxfp4`、`SGLANG_GLM_NEXTN_MOE_PTPC`);给引擎打补丁;整数量化 all-reduce(规则称在 80 万 token 以上长上下文评测中有准确率回退);把 stream interval 设到 100 [S11][S9]。
- **厂商专有引擎**:官方自己提醒,AMD 在 Kimi K3 上的好成绩依赖 ATOM,"只存在于厂商运行时下的曲线,描述不了(用上游开源栈的)客户能部署什么" [S31]。AMD 则于 2026-07 要求删除看板上"ATOM 尚未服务生产 token"的脚注,称已有商用部署(阿里妈妈)[S13]。
- **"工程关注度"混淆**:官方承认 Hopper 在 Kimi K3 上表现差,部分原因是没人给它写调优内核,"代际差距部分是工程关注度的差距" [S31]。

### 4.7 商业关系与利益冲突

- **事实**:
  - 硬件由被测厂商提供。致谢里写明 AMD(Lisa Su、Anush Elangovan)提供 MI355X 等 GPU,NVIDIA(Jensen Huang、Ian Buck)通过 OCI 提供 GB200 NVL72 机柜和 B200,v2 时提供了近 1000 块 GPU 的访问;Crusoe、CoreWeave、Nebius、TensorWave、Oracle、TogetherAI 提供算力 [S1][S3][S6]。
  - 发布文章里并列引用 Jensen Huang、Lisa Su 等被测方高管的背书 [S3]。
  - TCO 和功耗换算用的是 SemiAnalysis 自己出售的 AI Cloud TCO 模型和 Datacenter 模型;文章里有销售链接,AgentX 文章后半段在付费墙后 [S3][S15][S2]。
  - SemiAnalysis 称自己为 AMD 软件团队提供了"多年"的持续反馈与建议 [S2]。
  - 负载生成器 AIPerf 出自 NVIDIA;官方说另维护 fork 是为了"更加厂商中立" [S2]。
- **批评者**:
  - latchkey(HN):称其为"影响力服务""付费的托" [S20]。无具体证据随附于所见评论。
  - Ben Pouladian(BEP Research,2026-09,X 长文,经 youmind 转载,二手):针对的是 2026-09-07 的 TPU 文章而非 AgentX,但论点指向裁判角色——"基准机构是这场比赛的裁判,这次裁判把半场比分当终场公布"。他指出该文把 NVIDIA 一侧的 FP4、MTP、72 芯片机柜、分离式服务四项都关掉后称为"苹果对苹果",而用 SemiAnalysis 自己看板上各方最佳配置比较,GB300 NVL72 在 100 tok/s/user 处每美元 token 是 TPU7x 的 9.7 倍 [S17]。Pouladian 一贯看多 NVIDIA,也应视为一方意见。
  - Reddit r/LocalLLaMA 帖子(2026-05,标题称"NVIDIA/SemiAnalysis 误导性营销"):质疑拿 NVL72 的 72 块 GPU 与 8 块 GPU 比。回帖指出发帖人误读——纵轴已按每 GPU 归一,且多数 NVL72 运行并不用满整柜 [S32]。
  - Remio 分析(2026-07 后):CoreWeave 的 Rubin"10 倍"用的是 2025 年的 GB200 软件基线;SemiAnalysis 为对齐 CoreWeave 口径调整了自己的数据(原口径是算 prefill 加 decode 全部 GPU 的功耗但只报输出 token)[S33]。
- **文风**:官方文章使用"framemog""Jensen 在藏实力"之类措辞,Rubin 文章标题为"每美元性能高 67 倍" [S6][S15]。同一篇文章里也写了:67 倍是在 170 TPS、对手曲线接近其最快端点处取的;在多数供应商实际服务的 60–100 TPS 区间是 1.4–3 倍;对 GB300 SGLang 同一点是 5.56 倍 [S15]。标题数字与正文限定之间的落差是读者应当留意的。

### 4.8 没有机柜级硬件的人能否复现

- **能做的**:harness 可 pip 安装,对任何 OpenAI 兼容端点跑;数据集在 HuggingFace 公开;有独立运行文档 [S1][S12][S4]。
- **做不到的**:
  - 完整矩阵约 2 MW、1000 多块芯片 [S2];单点有效运行最短 900 秒、标准 1 小时,加上数据集重建、预热和排空 [S18][S4]。
  - 中位输入 14.2 万 token、上限 1M 的上下文,本身就要求大显存多卡;256k 变体是为上下文不超过 256k 的模型准备的,不是为小硬件准备的 [S8]。
  - "官方"结果只认 SemiAnalysisAI/InferenceX 仓库;fork 和第三方结果必须标"Unofficial",InferenceX/AgentX 有商标 [S1][S9]。
  - GitHub Actions 产物只保留 90 天 [S4];issue #1580 的用户想找数月前运行的服务端日志时已过期 [S13]。
- **外部复现记录**:
  - issue #1580:同镜像、同配置,复现者的 decode 明显快于官方数字,怀疑是网络/HCA 环境差异;维护者的回答是去看板找对应运行的日志,并说"很高兴你测出更好的结果" [S13]。
  - issue #4339(2026-10):同一配方在两台同型号节点上相差 12% [S13 之 issue 列表]。说明节点间差异可以达到两位数百分比。
  - issue #3229(2026-09):用户发现 AIPerf 输出的 P90 每用户输出速度约 217 tok/s,而看板显示 13.53;芯片数显示 16 而实际 4。维护者解释 16 是并发数;用户追问后,所见评论里没有进一步答复 [S13]。说明看板口径与 harness 原始输出之间的换算对外部读者不透明。
  - issue #3861(2026-10):用户问结果是统计整个过程还是稳定期;回复是"代码开源,自己看" [S13]。
- "被广泛复现和验证"是 SemiAnalysis 的说法,给出的例证是 Google Cloud、Azure 的博客和各家背书语录 [S6][S15];没有找到独立第三方对 AgentX 数字的系统性复现报告。

### 4.9 测的是服务效率,不是任务成功

- 官方多处明说:不评判智能体是否修好了 bug;合成内容不支持模型质量评估;任务质量要另做评测,两种分数要分开 [S16][S8]。
- 质量把关靠旁路 eval:配置须通过 eval(GSM8K 等),CI 会复核 [S11]。Pouladian 指出 GSM8K 太容易,FP4 与 FP8 在其上分数几乎相同(0.9685 对 0.9700),不足以裁定精度影响;并指出 TPU 预览结果当时没有任何准确率结果 [S17]。
- 规则里提到更强的评测 KVV 和 TB 4.0,用于放行 FP4 KV 缓存等更激进的量化;仓库规则也承认整数量化 all-reduce 在长上下文评测和实际生产中出现过质量下降 [S9]。
- 官方自己说智能体场景下人们更在乎"任务端到端完成速度",新指标还在定义中 [S2]。Pouladian 的说法是"买方的比较终点是完成一个任务花多少钱",而看板上没有任务数字 [S17]。
- vLLM 博客用 AgentX 数字算出"相对 Opus 5 API 价格 14.6–106 倍的成本优势",并注明这是服务成本比较而非模型质量比较 [S10]。这种跨模型的成本对比正是"不测任务成功"最容易被误读的地方。
- SemiAnalysis 另有一个测任务的基准 GDPx(2026-11 初出现在 HN),与 AgentX 是两回事 [S20]。

### 4.10 排名对交互速度阈值和 token 记账方式的敏感性

- **阈值**:
  - 官方一贯主张看整条曲线、在同一交互速度下比较 [S3][S16]。
  - 同一组数据,阈值不同结论不同:Rubin 对 GB300 在 100 TPS 是 2.09 倍(对较强引擎)、150 TPS 是 7.2 倍、170 TPS 对 TRT-LLM 是 62.9 倍、200 TPS 回落到 2.72 倍;最强的 GB300 引擎也随阈值换(75 TPS 是 TRT-LLM,更高是 SGLang)[S15]。
  - 这些等交互速度的数是在实测点之间**插值**的 [S15];v2 文章也承认不是每块 GPU 在每个交互速度上都有数据点 [S6]。
  - Kimi K3 上 MI355X ATOM 胜过 GB300 NVL72 只在端到端延迟 40–60 秒那一段 [S31]。
  - vLLM 博客选的是 P90 交互速度高于 50 tok/s/user 的最高吞吐点 [S10]。
- **横轴定义本身在变**:1.0 文章用 1/P90 ITL 的交互速度并要求与 TTFT 一起看;Rubin 文章用"整响应 ITL"口径;现行规则以 E2E normalized interactivity 为 North Star,并用它筛选其他视图上能出现的点 [S2][S15][S9]。官方列出的该指标的缺点:数值比常规交互速度低、不可直接比;TTFT 按输出长度摊销,短输出受罚更重;分不清退化来自 TTFT 还是解码;仍依赖追踪分布 [S9]。
- **token 记账**:
  - 头条纵轴常是"总 token"(输入 + 输出 + 缓存命中的输入)每 GPU 秒或每美元。vLLM 的 83K、70K、11.8K TPGS 都注明含缓存 token [S10]。在命中率 96%+ 的负载下,总 token 的绝大部分是没有被重新计算的缓存读取。
  - 官方方法页的主指标是"每芯片输出吞吐" [S16]。同一运行,两种纵轴数值相差两个数量级,引用时必须说清。
  - 分离式配置的每 GPU 输入/输出吞吐按各自角色的 GPU 数归一,不按总数;issue #299 的用户认为这样与聚合式配置不可比,维护者不同意修改 [S13]。总吞吐每 GPU 则按全部 GPU 计 [S17]。
  - token 数的来源有两种:客户端用本地分词器重算,或信任服务端 usage 字段(`--use-server-token-count`);分词器版本或模板不同会造成偏差 [S18]。
  - 成本换算依赖 SemiAnalysis TCO 模型里的每芯片小时价格(例如 GB300 NVL72 为 2.31 美元、TPU7x 为 1.21 美元,二手转述)[S17];用户可在计算器里改成自己的价格 [S15]。
  - DRAM 卸载的 3 TB 上限、按 GPU 占比分配,都是人为规则;官方承认本地生成器对所有 runner 都套了 3 TB 上限,还没实现"标准化 DRAM 系统不设上限"的例外 [S2]。
- **软件版本的时效性**:Pouladian 记录,TPU 文章里 B200 的数字在文章发布次日的新运行中变了 17%(8,903 到 10,389 tok/s/chip)[S17]。连续基准的数字带日期才有意义。

---

## 5. SemiAnalysis 的回应、缓解措施与治理

### 5.1 对各项批评的回应

| 批评 | 回应或缓解 | 来源 |
|---|---|---|
| 单一来源 | 承认;下一版数据集覆盖更多模型和 harness | [S2] |
| 合成内容 | 强制 golden 接受长度;推动 vLLM、SGLang、TRT-LLM、ATOM 都加上强制接受控制 | [S2][S8] |
| 闭环 | 承认并解释为智能体的本性;建议报整条曲线、用足够高的并发 | [S16] |
| 时间戳非因果 | 承认,保留两个约束,称后续改进 | [S2] |
| 不比纯硬件 | 明确说比的是硬件加软件的整体体验,不强制功能对等,因为现实中用户会用上手头所有优化 | [S13 之 issue #99] |
| 基准特调 | 配方须出现在 vLLM recipes 或 SGLang cookbook 的已合并文档中;禁止给引擎打补丁(除非有书面豁免);禁止减少模型 FLOPs 的改动;草稿按出厂精度;镜像须来自上游主分支;先交 vLLM/SGLang 再交专有引擎 | [S11][S9] |
| 指标被单点优化 | North Star 前沿规则:只在某个次要延迟指标上占优的点不能进入公布的前沿 | [S9] |
| 可审计性 | 看板每个点链接到产生它的 GitHub Actions 运行;配方、镜像、命令行固定在仓库里;失败和不稳定的配置也应报告 | [S4][S16] |
| 不测质量 | 明确划界;旁路 eval 门槛 | [S16][S11] |
| DRAM 军备竞赛 | 3 TB 上限、按 GPU 占比;TCO 按服务器总 DDR5 成本归一 | [S9] |
| 利益冲突 | 没有找到正面回应的专门声明;通用说法是"独立、厂商中立、可复现",并举出 AMD 在若干配置上胜出的结果 | [S4][S3][S2] |

### 5.2 治理

- **谁能提交**:任何人可提 PR;fork 的 PR 不能自己打标签,须由维护者对当前 head SHA 打标签才触发跑分,每次 push 后要重新打 [S11]。
- **评审流程**:(1) PR 跑完整 sweep 且全绿,含 eval;(2) 若改动文件的 CODEOWNER 是厂商工程师,须由一名 CODEOWNER 按最新模板贴出 PR Review Checklist 签核;(3) 在 Slack 上找核心维护者最终批准;(4) 维护者评论 `/use <run_id>` 指定复用哪次运行的结果,合并后主分支不重跑,而是把那次 PR 运行的产物入库 [S11]。
- **自动复核**:CI 机器人(用 Claude)独立复核签核里的每一项,包括 CODEOWNER 身份、sweep 和 eval 是否全绿、上游配方链接、是否用上游镜像、有无改架构的手法、草稿精度等;目前是"建议性"的,不发布提交状态 [S11]。
- **变更留痕**:改变任何基准运行内容的 PR 必须在 `perf-changelog.yaml` 末尾追加条目,历史条目不得修改 [S11]。
- **AI 披露**:每个 PR 须写明用了哪个模型的哪个版本 [S11]。
- **评审时限**:目标是 `max(Slack 上被提醒后 1 天, 厂商响应时间)`,以"公平"为由 [S9]。
- **规则怎么改**:欢迎合作伙伴和社区提 PR 增删检查项,原则是"删一条准则应当和加一条一样容易" [S14]。规划文档 MODELS.md 的 CODEOWNER 里 NVIDIA 和 AMD 各有一人,加核心团队 [S30]。
- **争议处理**:没有成文的申诉或仲裁程序。实际做法是 GitHub issue 加 Slack 加邮件(inferencex@semianalysis.com)[S13];issue 30 天无活动自动关闭 [S13]。历史上的例子:v1 的预热争议由 AMD、NVIDIA、SemiAnalysis 三方商定 [S3];issue #3841 的草稿精度争议走公开 issue [S13]。
- **引擎配额**:模型到主引擎的映射在只指定一个主引擎的模型之间对半分给 vLLM 和 SGLang,理由是不偏袒任何一方;依据是"一线 AI 实验室的反馈",称 TRT-LLM、ATOM 这类专有引擎并不总能提供其 AgentX 负载所需的全部功能 [S9]。
- **仓库规模**(2026-11-10):约 2,019 星、367 个 fork、141 个未关闭 issue,开启了 Discussions [S30]。

---

## 6. 厂商引用 AgentX 数字时,读者应当追问的问题

1. **哪个纵轴?** 是每芯片输出 token,还是含缓存的总 token?每 GPU 秒、每美元还是每 MW?[S10][S16]
2. **哪个横轴、什么阈值?** P90 常规交互速度、整响应 ITL 口径,还是 E2E normalized?阈值是多少,是不是恰好取在对手曲线的端点?[S9][S15]
3. **是实测点还是插值?** [S15]
4. **TTFT 是多少?** 官方自己说交互速度和 TTFT 必须一起看 [S2]。
5. **哪个数据集变体?** full(1M)还是 256k?两者的输入中位数差近一倍(142,016 对 88,768)[S8]。
6. **哪个模型、精度、思考模式?** 该模型现在还在跑吗(DeepSeek V4 Pro、MiniMax M3、Kimi K3 都已退役)?[S9]
7. **哪个引擎?** 上游 vLLM/SGLang,还是 TRT-LLM、ATOM 这类厂商运行时?[S31]
8. **开了什么?** 投机解码(哪种草稿、多长、强制接受长度取多少)、分离式、KV 卸载(哪种、多少 DRAM)、哪种并行 [S2][S11]。
9. **用了多少芯片、多少并发?** 一条曲线上的不同点可能是不同的芯片数和不同的优化组合 [S2]。
10. **哪一天的运行、哪个 harness 小版本、哪个镜像?** 数字隔天就可能变两位数百分比 [S17];stream interval 是否不超过 20 [S14]。
11. **是官方结果吗?** 有没有 GitHub Actions 运行链接?厂商博客里用 AgentX harness 自测的结果按规则须标"unofficial" [S9]。
12. **成本数字用的谁的价格?** SemiAnalysis TCO 模型的自有成本、三年租价,还是厂商自己的假设?[S15]
13. **和 API 价格比的时候**,是否混淆了不同模型的能力?是否假设了理论满命中率?利用率假设是多少?[S10]
14. **这个负载像你的负载吗?** Claude Code、编程、单一公司员工、中位 14 万 token 输入、444 token 输出、5 分钟 TTL。聊天、RAG、语音、长输出推理类负载的排名可能完全不同 [S2]。
15. **质量有没有守住?** 通过的是哪个 eval?GSM8K 通过不等于长上下文编程任务不掉点 [S17][S9]。

---

## 来源

- [S1] SemiAnalysisAI/InferenceX 仓库 README(News、支持硬件表、致谢、Unofficial 声明)— https://github.com/SemiAnalysisAI/InferenceX
- [S2] SemiAnalysis,"AgentX - InferenceXv3: Does the CUDA Moat Hold Up in Agentic Inferencing?"(本次读了后 6 万字符及全文关键词检索,含"Future Next Steps"一节)— https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat ;小节锚点 https://newsletter.semianalysis.com/i/211976669/inferencexagentx-future-next-steps
- [S3] SemiAnalysis,"InferenceMAX: Open Source Inference Benchmarking"(2025-10)— https://newsletter.semianalysis.com/p/inferencemax-open-source-inference
- [S4] InferenceX "About" 页(定位、AgentX FAQ、可复现性四步、产物保留 90 天)— https://inferencex.semianalysis.com/about
- [S5] Tom's Hardware,InferenceMAX 报道(2025-10,二手)— https://www.tomshardware.com/tech-industry/inferencemax-ai-benchmark-tests-software-stacks-efficiency-and-tco-vendor-neutral-suite-runs-nightly-and-tracks-performance-changes-over-time
- [S6] SemiAnalysis,"InferenceX v2: NVIDIA Blackwell Vs AMD vs Hopper - Formerly InferenceMAX"(2026-02,只读了前 3 万字符)— https://newsletter.semianalysis.com/p/inferencex-v2-nvidia-blackwell-vs
- [S7] InferenceX 博客,"A Brief Overview of Agentic Workloads" — https://inferencex.semianalysis.com/blog/brief-overview-of-agentic-workloads
- [S8] InferenceX,"AgentX Methodology"及 /agentx 概览页 — https://inferencex.semianalysis.com/infx/agentx/methodology ; https://inferencex.semianalysis.com/agentx
- [S9] InferenceX 仓库 `inferencex-e2e/docs/MODELS.md`(场景状态、弃用日期、AgentX Guidelines、North Star 政策、引擎提交政策、KV 卸载政策、模型矩阵;只读了前 2 万字符)— https://github.com/SemiAnalysisAI/InferenceX/blob/main/inferencex-e2e/docs/MODELS.md
- [S10] vLLM 博客,"vLLM x AgentX: Optimizing for Real-World Agentic Serving"(2026-09-08)— https://vllm.ai/blog/2026-09-08-vllm-agentx
- [S11] InferenceX 仓库 `CONTRIBUTING.md`(评审流程、草稿精度规则、`/use` 复用、MLPerf 附录 C 对照)— https://github.com/SemiAnalysisAI/InferenceX/blob/main/CONTRIBUTING.md
- [S12] SemiAnalysisAI/agentx-harness 发布页(v1.0.0 至 v1.0.13,经 GitHub API)— https://github.com/SemiAnalysisAI/agentx-harness/releases
- [S13] InferenceX 仓库 issue(经 GitHub API):#99、#299、#1580、#1632、#1651、#2125、#3229、#3841、#3861、#4339 — https://github.com/SemiAnalysisAI/InferenceX/issues
- [S14] InferenceX 仓库 `inferencex-e2e/docs/PR_REVIEW_CHECKLIST.md`(上游配方要求、AgentX stream interval、签核模板)及 PR #2685 — https://github.com/SemiAnalysisAI/InferenceX/blob/main/inferencex-e2e/docs/PR_REVIEW_CHECKLIST.md
- [S15] SemiAnalysis,"Rubin NVL72 Agentic Inference: 67x better Performance per Dollar"(2026-09-14,公开部分)— https://inferencex.semianalysis.com/blog/vera-rubin-nvl72-agentic-inference
- [S16] InferenceX 博客,"Agentic Benchmark for LLM Inference: Metrics and Methodology" — https://inferencex.semianalysis.com/blog/agentic-benchmark-agent-benchmark-guide
- [S17] Ben Pouladian,"Free Isn't Cheap Enough"/"NVIDIA Blackwell vs Google TPU"(2026-09,X 长文,经 youmind 转载,二手;只读了前 2 万字符)— https://youmind.com/landing/x-viral-articles/nvidia-blackwell-vs-google-tpu-benchmarks
- [S18] NVIDIA AIPerf 文档,"InferenceX AgentX MVP Benchmark"(v0.13.0,只读了前 1.5 万字符)— https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/inference-x-agent-x-mvp-benchmark
- [S19] Artificial Analysis,"Language Model API Performance Benchmarking"方法页(含 3.0.0 版本说明与诚信条款)— https://artificialanalysis.ai/methodology/performance-benchmarking
- [S20] Hacker News(经 Algolia API):latchkey 的评论 49455668(2026-08-26)、47018915(2026-02-14);InferenceX 相关转帖列表 — https://news.ycombinator.com/item?id=49455668 ; https://news.ycombinator.com/item?id=47018915
- [S21] Qwen-Bailian 匿名追踪数据集 README;Wang 等,"KVCache Cache in the Wild"(USENIX ATC'25)— https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon ; https://arxiv.org/abs/2506.02634
- [S22] Zhu 等,"TraceLab: Characterizing Coding Agent Workloads for LLM Serving"(2026-06,读了摘要;95.7% 等数字来自搜索摘要)— https://arxiv.org/abs/2606.30560
- [S23] Tiwari 等,"CacheWise"(2026-06,读了摘要)— https://arxiv.org/abs/2606.16824
- [S24] Techstrong.ai,"MLPerf Inference v6.1 Updated for RAG, Agents and API-Based Serving"(2026-09-16,二手)— https://techstrong.ai/articles/mlperf-inference-v6-1-updated-for-rag-agents-and-api-based-serving/
- [S25] Lambda 博客,"MLPerf Inference v6.1: pioneering agent, VLM benchmarks"(提交方自述)— https://lambda.ai/blog/mlperf-inference-v6.1
- [S26] Wang 等,"BurstGPT"(读了摘要)— https://arxiv.org/abs/2401.17644
- [S27] Xiang 等,"ServeGen"(读了摘要)— https://arxiv.org/abs/2505.09999
- [S28] Abramovich 等,"SPEED-Bench"(读了摘要;arXiv API 返回的 2604.09557 条目)— https://arxiv.org/abs/2604.09557
- [S29] "Open vs Closed Loop: A Benchmarking Crime"(AgentX 文章自己链接的博客)— https://notpeerreviewed.com/blog/tail-latency/
- [S30] InferenceX 仓库 `.github/CODEOWNERS` 及仓库元数据(GitHub API)— https://github.com/SemiAnalysisAI/InferenceX/blob/main/.github/CODEOWNERS
- [S31] InferenceX 博客,"Kimi K3 on AgentX: MI355X ATOM Beats GB300 NVL72 on Part of the Curve" — https://inferencex.semianalysis.com/blog/kimi-k3-agentx-mi355x-atom-vs-gb300-nvl72
- [S32] Reddit r/LocalLLaMA 帖子快照(2026-05,经第三方快照站,二手)— https://reddit.com/r/LocalLLaMA/comments/1t18dfz/nvidia_semianalysis_misleading_marketing/
- [S33] Remio,"Vera Rubin NVL72 Beats GB200, but the TCO Case Is Narrower"(二手分析)— https://www.remio.ai/post/openai-semianalysis-lens-vera-rubin-nvl72-beats-gb200-but-the-tco-case-is-narrow

---

## 没查到 / 不确定

- **The Register 对 AgentX 或 InferenceX 的报道**:第一次搜索没有结果,第二次搜索因额度用尽未执行。不能断言不存在。
- **X(Twitter)上从业者的讨论串**:没有直接抓取;只通过 youmind 转载读到 Pouladian 一篇。
- **独立的学术或第三方方法论评审、对 AgentX 数字的系统性复现**:没找到。
- **E2E normalized interactivity 何时成为默认横轴**:MODELS.md 写了规则,没写生效日期。
- **AgentX v1.1 的发布日期和具体内容**:只有"计划做更大改动"的说法;截至 2026-11-10 发布页只有 1.0.x。
- **TPUv7x 的 AgentX 结果是否已上线**:README 称 2026-09 "正式支持" TPUv7x,但 9 月的 TPU 文章只有 8k1k;之后的状态没核实。
- **MI455X UALoE72、Rubin NVL8 的具体上线时间**:只有"Coming Soon"。
- **多模态、非编程智能体、更长会话**是否在路线图里:没找到明确承诺。
- **issue #3841(实测接受长度能否作为证据)的最终裁定**:issue 已关闭,结论没读到。
- **issue #3229(看板 13.53 与 AIPerf 217 tok/s 的差异)的解释**:所见评论里没有答复;推测与 P90 口径和整响应 ITL 有关,未证实。
- **AIPerf 文档与 SemiAnalysis 关于语料采集者的不同说法**(Callan Fox 还是 SemiAnalysis 自采):没有进一步证据裁定。
- **TraceLab 的 95.7% 命中率、84.4% 用户发起步骤命中率**:来自搜索摘要和二手博客,没在论文正文核对。
- **MLPerf 数据中心 LLM 场景的最新规则细节、数据中心版 Agentic 基准的发布时间**:只有二手报道说"即将推出";Server 场景为泊松到达是凭既有了解写的。
- **GuideLLM、LLMPerf、vLLM bench 的当前功能**:本次没有抓取文档,对比表中相关格子凭既有了解填写,需核对。
- **Azure 官方 LLM 推理追踪(AzurePublicDataset)**:只核对了 BurstGPT,没有单独查 Azure 官方数据集。
- **latchkey 的身份与利益关系、其提到的"长文档"**:未核实,未读到。
- **SemiAnalysis 与被测厂商之间是否有付费咨询或订阅关系的公开披露**:没找到专门声明;已知的只有硬件捐赠/访问、自述的对 AMD 的长期建议关系,以及其付费模型产品。
- **393 会话子集的抽样方法、与 8,271 会话母语料的分布对照**:没有公开材料。
- **AgentX 文章前 10 万字符中的结果章节**:按说明未重读,只做了关键词检索;其中若有对批评的直接回应,可能遗漏。
