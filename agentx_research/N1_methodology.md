# N1 · AgentX 的工作机制:数据集与重放 harness

调研日期:2026-11-10。口径说明:凡标 [9] 的数字是我下载公开数据集 `traces.jsonl` 后自己算的(不是官方公布值),其余来自官方仓库、文档与数据集卡片。InferenceX 仓库读的是 commit `0b99145`(2026-11-10),harness 读的是 tag `agentx-v1.0.13`(commit `60a094f`,2026-11-03)。

## 0. 组件关系(先看这个)

- AgentX 由三块公开件组成:(a) Hugging Face 上的 trace 数据集 `semianalysisai/cc-traces-weka-062126` 及其 `-256k` 变体 [7][8];(b) 负载生成器 `agentx-harness`(PyPI 包,命令 `agentx`),是 SemiAnalysis 对 NVIDIA AIPerf 的 fork,仓库 `SemiAnalysisAI/agentx-harness`,Apache-2.0,要求 Python 3.11–3.13 [10];(c) InferenceX 仓库里的编排与打分代码 `inferencex-e2e/infx/bench/agentic/`、`infx/results/agentic/`、`infx/datasets/` [3][4][5][6]。
- InferenceX 当前钉住的 harness 版本是 `agentx-harness==1.0.13`(`requirements.txt` 与 standalone 文档一致)[2][3]。PyPI 发布史:1.0.7(2026-10-12)→ 1.0.13(2026-11-03),另有一个 1.1.0(2026-10-14)[10]。PyPI 上看得到的最早版本是 10 月的,8 月发布时 harness 以什么形式分发我没有核实。
- harness 里有两个场景预设:`inferencex-agentx-mvp`(早期 MVP 规则)和 `agentx`(InferenceX 正式用的预设),二者共享校验规则但默认值不同,官方文档明说两者默认命令的结果"不可直接等同"[12][14]。
- trace 的文件格式叫 "Weka" 格式,源自 Callan Fox 的 kv-cache-tester 项目;AIPerf 文档称该格式逐请求记录 Claude Code 会话 [12][13]。
- 官网术语表对 AgentX 的定义很短:InferenceX 的智能体推理场景,负载形状来自自愿提供的 coding-agent trace,原始内容已移除 [16]。

## 1. 数据集构建

### 1.1 来源与采集方式
- 数据来自 SemiAnalysis 自建的 "claude-code-proxy / agentic-proxy":一个架在 Claude Code 与 Anthropic API 之间的代理,请求逐条落到 Postgres(Neon)库的 `public.requests` / `public.sessions` 表 [4][5]。数据集卡片自述为 "derived from SemiAnalysis Claude Code proxy traces" [7]。
- 只取 Anthropic 模型:SQL 过滤 `model LIKE 'claude-%'`;走 `/proxy/responses` 的 GPT-5.x(Codex)行被有意排除 [4]。
- 只取成功请求:`response_status_code = 200 AND error IS NULL` [4]。
- 隐私档位:库里每行有 `privacy_mode`,取值 `anon` 或 `full`;采样脚本默认且只取 `anon`(请求体已脱敏、指标列完整),脚本注明"设计上不允许两档混用",逐行再做一次安全校验 [4]。
- 时间下界:子智能体标签列的迁移 `013_add_subagent_label` 于 2026-04-16T16:00Z 上线,此前的行不可用 [4]。上界:数据集构建时间 2026-06-21 17:48 UTC [7]。所以采集期落在 2026-04-16 至 2026-06-21 之间;因为还要求 trace 版本恰为 v7、CLI ≥ 2.1.139,实际窗口更窄,但公开材料没给出确切起止日。
- 客户端:Claude Code CLI(从 user-agent `claude-cli/X.Y.Z` 解析版本)[4]。

### 1.2 筛选条件(062126 版,即 AgentX 1.0 用的那版)
数据集卡片给出的实际采样命令是:`--sampling top --min-trace-version 7 --max-trace-version 7 --min-requests 20 --require-cli-min 2.1.139 --max-parallel-subagents 10 --exclude-dynamic-workflow-bug` [7]。逐条含义:
- trace 版本恰为 v7(会话里任何一条成功请求不是 v7 就整会话剔除)[4][7]。
- 每会话至少 20 条 Anthropic 请求 [7]。没有设请求数上限,也没有设时长上限(脚本支持 `--max-requests`、`--max-span-hours`,但这版没用)[4][7]。
- 每一行都必须来自 Claude Code CLI ≥ 2.1.139——这是 `x-claude-code-agent-id` 请求头上线的版本,有了它才能可靠地把子智能体请求归组 [4][7]。
- 同时在飞的子智能体"组"峰值 ≤ 10。一组 = 一个 agent-id(一次 Task 工具调用);在飞区间从组内第一条请求开始到最后一条请求结束;用扫描线算峰值,同一时刻先结束后开始不算重叠 [4][7]。
- 剔除含图片的行(v1/v2 trace 中带 image 内容的会话整体排除)[4][7]。
- 剔除"分类器式"小调用:`max_tokens ≤ 64` 且无 tools 的请求(建议模式、标题生成、Security Monitor 等)[4][7]。
- 完全重复的代理行按 `(timestamp, model, in, out, duration_ms, agent_id)` 去重 [5][7]。
- 剔除"dynamic-workflow bug"会话:Claude Code CLI < 2.1.174 发出的 dynamic-workflow 子智能体不带标签头,在一个会话里表现为多条交错的无标签轨迹;若会话最高 CLI 版本 < 2.1.174 且并发的无标签多轮轨迹峰值 ≥ 3 就整会话丢弃 [4][7]。
- 单请求输入上限 990,016 token(最接近 99 万的 64 的倍数)。原因:`in` 字段 = 64-token 块数 × 64,不是真实分词计数;在重度 prompt caching 的长上下文尾部会比计费口径(input + cache_read + cache_write)高估最多约 26 万 token,使少数请求读数超过 1M,而实际没有任何请求的真实 prompt 超过 1M。这些行被当作伪影删掉,其余时间线保留 [7]。
- 采样方式是 `top`(按请求数从大到小)且不设 `--limit`,即"通过筛选的全部会话",不是随机抽样 [6][7]。

### 1.3 规模(官方 stats.txt)
| 指标 | 完整版 062126 [7] | 256k 版 [8] |
|---|---|---|
| traces(会话) | 393 | 393 |
| 顶层(主智能体)请求 | 56,798 | 28,444 |
| 子智能体组 | 1,697 | 1,697 |
| 子智能体内部请求 | 42,029 | 39,822 |
| 模型请求合计 | 98,827 | 68,266 |
| 输入 token 合计 | 21,635,381,376 | 6,891,228,864 |
| 输出 token 合计 | 106,474,498 | 58,728,807 |

- 完整版输入/输出 token 比约 203:1,256k 版约 117:1 [9]。
- 文件大小:完整版 `traces.jsonl` 1.85 GB,256k 版 0.57 GB;许可 Apache-2.0;公开、无需登录 [7][8]。截至 2026-11-10 HF 显示下载量约 9.8 万 / 5.2 万次 [17]。

### 1.4 产生这些 trace 的模型 [9]
按请求数(完整版):`claude-opus-4-8` 62,108(63%)、`claude-fable-5` 16,192(16%)、`claude-haiku-4-5-20251001` 7,973(8%)、`claude-opus-4-7` 6,419、`claude-opus-4-6` 5,452、`claude-sonnet-4-6` 576、`claude-sonnet-4-5` 107。
按会话的模型组合:仅 opus-4-8 的 156 条;opus-4-8 + haiku-4-5 的 87 条;仅 fable-5 的 39 条;其余为混合。即约一半会话在一条 trace 里混用了两个以上模型(典型是大模型主智能体 + Haiku 子智能体/辅助调用)。

### 1.5 分布统计(完整版,自算 [9];"主"= 顶层条目,"子"= subagent 组内请求)
- 每请求输入 token(全部 98,827 条):p5 3,008 / p10 32,704 / p25 64,768 / **p50 142,016** / p75 310,464 / p90 549,504 / p95 682,880 / p99 863,471 / 最大 989,824;均值 218,922。与官方公布的中位数 142,016 吻合。
- 每请求输出 token:p5 76 / p25 192 / **p50 444** / p75 1,105 / p90 2,659 / p95 4,298 / p99 9,311 / 最大 59,903;均值 1,077。与官方中位数 444 吻合。
- 主智能体请求输入:p25 134,848 / p50 254,080 / p75 452,928 / p90 651,008 / p99 897,672;输出 p50 616、p90 3,502。
- 子智能体请求输入:p25 36,352 / p50 64,000 / p75 119,680 / p90 196,326 / p99 455,163;输出 p50 328、p90 1,261。
- 输入超过阈值的请求占比:>64k 75.3%;>128k 53.9%;>200k 38.5%;>256k 30.8%;>500k 12.3%;>900k 0.55%。
- 会话峰值输入超过阈值的会话占比:>128k 79.9%;>256k 44.8%;>500k 19.3%;>900k 2.8%。会话峰值输入中位数 224,640。
- 每会话请求数(主+子):最小 20 / p25 40 / p50 86 / p75 233 / p90 618 / p99 2,474 / 最大 4,114;均值 251.5。
- 每会话顶层请求数:p25 33 / p50 65 / p75 140 / p90 330 / 最大 2,977。
- 含子智能体的会话:175/393 = 44.5%(与官方"44%"吻合)。每会话子智能体组数:p75 3 / p90 8 / p95 19 / 最大 153;每组内部请求数 p50 16、p90 47、最大 789;组持续时间 p50 136 s、p90 723 s。会话内并发组峰值:p75 3 / p90 4 / 最大 10(对应筛选上限)。
- 会话时长(首请求到末请求结束的墙钟跨度):p25 0.7 h / p50 2.8 h / p75 15.5 h / p90 47 h / 最大 255 h——很多会话是跨天挂着的,所以重放时必须处理长空闲(见 4.4)。
- 每会话累计输入 token:p50 1,120 万 / p75 4,070 万 / p90 1.42 亿 / 最大 15.1 亿。
- 请求间"思考时间"(上一请求结束到下一请求开始):主智能体 p25 2.3 s / p50 4.8 s / p75 11.2 s / p90 121 s / p95 212 s / p99 3,082 s;子智能体 p50 0 s、p90 3.6 s。
- 原始(Anthropic 侧)耗时,仅作参考:`api_time` p50 6.7 s、p90 26.3 s;流式请求记录的 TTFT p50 2.6 s、p90 7.0 s。
- 每轮上下文增长:相邻顶层请求的输入差,p25 +576 / p50 +1,664 / p75 +2,624 / p90 +5,120 token;在"纯追加"的相邻对里,新增块数中位数 27 块(≈1,728 token)。
- 前缀复用(理想前缀缓存命中率):按 token 加权,全部请求 98.3%,顶层 98.8%,子智能体 96.1%。算法:每条 trace 内按时间顺序维护"已见块"集合,一条请求的命中 = 开头连续已见块数。与官方宣传的"95%+ KV cache 命中率"一致 [9][16]。256k 版同口径为 96.2%。
- 前缀断裂:相邻顶层请求中,后一条不是前一条的纯扩展的占 10.4%(5,891/56,405);输入骤降到前一条一半以下的有 914 次,涉及 227 条会话。这里既包含真正的上下文压缩/编辑,也包含"被摊平记录在顶层的并行智能体"交错造成的假断裂(见 3.3),不能直接当作压缩次数。

256k 版对应值 [9]:输入 p50 88,768 / p90 204,288 / 最大 255,808;输出 p50 376 / p90 1,845;每会话请求数 p50 70;顶层请求输入 p50 135,104。

## 2. 脱敏

### 2.1 删了什么、留了什么
- 公开 trace 里**没有任何文本**:没有 prompt、没有回复、没有工具名或工具参数。每条请求只剩数值与结构字段 [5][7][11]。
- 保留的是:时间戳、模型名、输入/输出 token 数、KV 缓存块标识列表(`hash_ids`)、原始耗时、是否流式、子智能体归属 [5]。
- 官方 FAQ 的说法:原始采集含 token 计数与缓存块标识,不含原始 prompt 文本,后者为去除 PII 等敏感内容被匿名化掉 [11]。

### 2.2 64-token 块哈希如何保住前缀复用
- 代理把每条请求的 prompt 按 64 token 一块切开,对每块(连同其前缀)算哈希,库里存的是 24 位十六进制字符串 [5]。
- 发布前再做一次映射:每条 trace 内,第一次出现的哈希记为 0,第二个记为 1,依此类推,变成小整数;并声明 `hash_id_scope: "local"`——同一条 trace 内(主智能体与它自己的子智能体之间)相同整数代表相同内容,不同 trace 之间的整数互不相干 [5][13]。这一步顺带切断了跨会话关联的可能。
- 于是"第 k+1 轮的 `hash_ids` 以第 k 轮的列表开头再追加若干新 id"这一结构原样保留,这正是前缀缓存能命中的部分 [11]。
- `in` 字段优先取 `hash_token_count`(含不满一块的尾部),否则取块数 × 64;所以公开数据里 `in` 基本都是 64 的倍数 [5][7]。

### 2.3 合成 token 怎么生成
- 重放前,harness 用被测模型自己的分词器(`--tokenizer`,默认取 `--model`)把整个数据集"重建"一遍:每个 `hash_id` 确定性地映射到一段恰好 64 个 token 的填充内容;同一 id 在任何请求、任何角色片段里解码出的 token 序列逐字节相同 [11][13][15]。
- 填充 token 取自 harness 内置的已分词语料;不满一块的尾巴按 sha256(位置种子)在语料里取偏移切片,保证跨进程可复现 [15]。
- 块内容依赖随机种子(种子是磁盘缓存键的一部分),所以换种子会换一套填充文本,但结构不变 [11]。
- 发送的是"解码后的文本",服务器会再分词一次;当合成用的分词器与服务器一致时,每条消息只会漂移几个 token [11]。
- 助手消息也是合成的:历史里的 assistant 内容按**录制的**输出长度生成(按块对齐,可能比 `out` 多几个 token),**不是**被测服务器实际生成的文本。这样每一轮的块结构才能在不同服务器之间逐字节一致 [11]。

### 2.4 一条 trace 记录长什么样
实际公开文件(062126)的字段,经我核对 [9],与转换脚本 [5] 一致:
- trace 级:`id`、`models`、`block_size`(全部为 64)、`hash_id_scope`(全部为 `"local"`)、`requests`。
- 请求级:`t`(距会话开始的秒数)、`type`(`"s"` 流式 / `"n"` 非流式)、`model`、`in`、`out`、`hash_ids`、`api_time`(原始请求耗时,秒)、`think_time`(距上一条请求结束的间隔,秒;首条没有)、`ttft`(仅流式请求有)。
- 子智能体条目:`t`、`type: "subagent"`、`agent_id`(如 `subagent_001_<原始 id 末 8 位>`)、`subagent_type`(这版全部是 `"Subagent"`)、`duration_ms`、`total_tokens`、`tool_use_count`(恒为 null)、`status`(恒为 `"completed"`)、`models`、`requests`(内部请求列表,格式同上,`type` 一律 `"n"`)。
- 格式规范里还定义了 `input_types`、`output_types`、`stop`、`tool_tokens`、`system_tokens`,但这版公开数据**没有**这些字段 [9][15];因此工具调用往返、system prompt 边界都得靠 harness 从块结构推断(见 3.4)。
- 完整版流式请求 56,432 条、非流式 42,395 条;全部请求都带 `hash_ids` [9]。

## 3. 会话结构

### 3.1 主智能体轮次
- 每条 trace 变成一个根会话;顶层请求按时间顺序成为各"轮"。每轮在线上发送的是**完整的、不断增长的消息数组**(第 k 轮带第 0..k 轮),和普通聊天客户端一样;服务器无状态 [11]。
- 第 0 轮 = system 消息(工具定义 + 系统提示,按块对齐)+ 首条 user 消息;之后每轮追加"上一轮的 assistant 输出 + 新的 user/工具结果" [11]。

### 3.2 子智能体:spawn / join 依赖图
- 转换脚本先把代理行归组:有 `x-claude-code-agent-id` 的按 agent-id 归为一个子智能体条目;没有的旧式行按"连续的带标签行"归组;CLI ≥ 2.1.139 上只有标签没有 agent-id 的行(标题生成、状态栏等工具性调用)降级为普通主轮次 [5]。
- harness 把每个子智能体条目映射成子会话,挂在 AIPerf 的 DAG 结构上:在父会话"前一轮"上挂一个 `SPAWN` 分支,在父会话"后一轮"上挂一个 `SPAWN_JOIN` 前置条件——父会话要等这些子会话全部结束才发下一轮 [12][13]。
- 三种特例:没有前置父轮次的子智能体被丢弃;没有后续父轮次的成为后台分支(父不等);相邻且锚点相同的子智能体合并成一个多子分支 [13]。
- 本数据集里所有子智能体都是 "spawn"(全新上下文的子会话);"fork"(继承父上下文)是 AIPerf 的通用能力,这些 trace 没用到 [11]。
- 时间上重叠的子智能体并发发出;每个子流按**录制的相对偏移**启动,而不是父轮次一结束就齐发 [12][13]。
- 子智能体出错默认视同正常结束以释放父会话的 join;可选 `AIPERF_DAG_FAIL_FAST=true` 让首个子错误终止整轮运行 [11]。

### 3.3 被"摊平"的并行智能体
- 不少采集把并行扇出(例如不带 agent-id 头的 Workflow 工具智能体)记录成与主智能体交错的顶层请求。harness 用 `hash_ids` 的最长公共前缀(LCP)把它们识别出来并拆成独立子会话 [13]:
  - 完全延续某条链尾、时间上在其后、且模型相同 → 该链的下一轮;
  - 只保留链尾的一段前缀(压缩/上下文编辑),且更长的那个状态之后再没被碰过 → 同一智能体的延续("join seam");
  - 否则 → 从共享前缀分叉出的另一智能体,作为子会话并用 SPAWN/SPAWN_JOIN 连接。
- 拆出的链再分类:辅助 sidecar(短的一次性调用,或与主智能体不同模型,如 Opus 下的 Haiku 摘要)、reduction(大输入短输出的一次性调用:上下文压缩、子智能体结果摘要、工具输出摘要)、并行 worker 组(同一分叉点且时间确有重叠,默认至少 3 个成员)、单独智能体;优先级为 辅助 > reduction > worker 组 > 单独 [13]。
- 同样的检测在每个子智能体条目内部再跑一遍;文档举例:060826 版语料的 615 个子智能体条目里藏着约 3,100 条不同的上下文链 [13]。
- 可用 `AIPERF_DATASET_WEKA_SPLIT_FLATTENED_AGENTS=false` 关闭,回退到"顶层全部串成一个会话" [13]。

### 3.4 工具调用与工具时间
- trace 里没有工具模式。默认把工具结果轮次当作普通 user 消息发送;可选 `AIPERF_DATASET_WEKA_TOOL_SHAPED_MESSAGES=true` 发成合成的 OpenAI tool-call / tool 消息对(函数名和参数是占位符),代价是 token 数不再精确;默认关闭 [11][13]。
- 工具执行时间没有单独字段,它被包含在轮间间隔里:`think_time` = 上一条请求**结束**到本条请求**开始**的墙钟间隔(负值截为 0),既含工具运行时间也含人的停顿 [5]。重放时按"结束到开始"计时,因此不会把原服务器的响应时间重复计入 [11][12]。

### 3.5 上下文压缩 / 截断事件
- 没有显式的"压缩事件"标记。压缩在数据里表现为:某一轮的 `hash_ids` 只保留上一轮的一段前缀、`in` 骤降 [13][9]。
- harness 的处理:会话重建器逐轮比较块前缀;遇到前缀回退时把该轮标记为 `reset_context`,重新发出完整上下文而不是增量 [13][15]。执行压缩的那次"大输入、短输出"调用本身则被归为 reduction sidecar 单独重放 [13]。
- 256k 变体里的"截断"是另一回事:它在数据源头删掉超限请求,见第 5 节。

## 4. 重放

### 4.1 闭环语义与"并发"的含义
- 时序模式锁定为 `agentic_replay`,没有请求速率旋钮;`--request-rate`、`--arrival-pattern`、`--fixed-schedule`、`--adaptive-scale` 等一律禁止 [11][12]。
- `--concurrency N` = 同时存活 N 棵"会话树"(一个根会话 + 它的全部子智能体),不是 N 个在飞请求,也不是 N 条连接。一个并发位叫一条 lane [11]。
- 子智能体不占并发名额,它在父会话的 lane 里运行;所以扇出时瞬时在飞请求数可以高于 N [11][12]。
- 一条 lane 要等整棵树排空(根发完最后一轮**且**所有子智能体结束)才回收,然后立即从采样器取下一条 trace 作为新会话;因此存活的树恒为 N,不多不少 [11][12]。
- 会话内每一轮只在上一轮完成后、再等够录制的间隔才发出——闭环:服务器慢,整条会话就整体后移,负载不会堆积 [11]。

### 4.2 会话采样、起始偏移与种子
- 启动时每条 lane 取一条 trace,并在该 trace 的录制时长内均匀采样一个起点 t\*:`inferencex-agentx-mvp` 预设是 0%–100%;**正式的 `agentx` 预设是 25%–75%** [12][14]。由 t\* 推出起始轮 k_i,且保证预热之后至少还剩一轮可测 [12]。
- 太短、拆不出"预热轮 + 测量轮"的 trace 会被跳过 [12]。
- 回收后的新会话从第 0 轮完整重放,不再采样 t\* [11][12]。
- 并发数超过 trace 数(393)时同一条 trace 会同时落在多条 lane 上,各自有独立的 t\* [11]。只要语料大于 lane 数,顺序/洗牌采样器会先把每条 trace 至少放一遍再重复 [12]。
- 种子:`agentx` 预设默认 `--random-seed 42`,InferenceX 文档要求复现时保持 42 [2][14];MVP 预设则是每次自动生成并记录一个新种子 [12]。同数据集 + 同种子 = 同轨迹、同起点、同回收顺序 [12]。
- cache-bust 标记:每次"播放"一条 trace(首次或回收)都在首条 user 消息前插入一个 `[rid:<12 位十六进制>]` 加空行;同一次播放的所有轮次与其子智能体共用同一个标记,预热与测量阶段也共用;不同播放、不同运行的标记不同。作用是防止并发副本或回收副本之间虚假共享缓存,同时保留会话内共享。场景锁定为 `first_turn_prefix` [11][12][13]。

### 4.3 预热
- 基础预热:对每条在 t\* 处"进行中"的会话,发送 t\* 之前的最后一轮(它携带累积的完整前缀),把服务器缓存带到 t\* 时刻的状态;子智能体分支若在 k_i 时已活跃,每条活跃流各发一条预热请求 [11][12]。各 lane 的预热发送时刻被安排成让所有会话的 t\* 落在同一瞬间,即预热与测量的边界 [11]。
- `agentx` 预设在此之上再加"每 lane 10 条预热请求"(`--warmup-requests-per-lane 10`):基础预热完成后,每条 lane 沿同一棵活着的会话树继续发恰好 10 条线上请求,去掉录制的空闲间隔,且每条请求只生成 1 个输出 token;例如并发 16 就多发 160 条 [2][12][14]。另一种互斥的方式是按时长预热 `--agentic-cache-warmup-duration` [12]。
- 预热宽限期:`agentx` 预设 1,800 秒 [14];InferenceX 可通过 `AGENTIC_WARMUP_GRACE_PERIOD` 覆盖 [3]。
- 任何根会话的预热请求终态失败(每条只试一次、不重试)→ 立即中止整次运行,标为 `run_cancelled`;子智能体流的预热失败不触发中止 [11][12]。
- 预热流量不计入任何指标 [11][12]。

### 4.4 轮间延迟与空闲上限
- 录制的间隔照原样遵守,禁止 `--ignore-trace-delays` 与逐轮上限 `--inter-turn-delay-cap-seconds` [11][14]。
- 全局空闲上限 10 秒(`--system-idle-gap-cap-seconds 10`,场景锁定):当整个重放没有任何在飞或就绪请求时,所有待发定时器统一前移,使下一条请求在 10 秒内到达;顺序与相对间隔不变 [11][12][14]。
- 单树空闲上限:`agentx` 预设设为 300 秒(`--trace-idle-gap-cap-seconds 300`):某棵树(根 + 全部后代)连续 300 秒没有在飞请求时,只把这棵树的待发定时器统一前移;不改数据集时间戳,不绕过 spawn/join 依赖 [12][14]。MVP 预设默认不设 [12]。
- 这两条上限就是"长达数天的会话如何塞进一小时窗口"的答案:活跃段照原节奏放,长时间停顿被压到最多 300 秒 [9][12][14]。

### 4.5 测量窗口与运行如何结束
- `agentx` 预设测量时长 3,600 秒(MVP 预设 1,800 秒;两者下限都是 900 秒,低于下限拒绝启动)[14]。InferenceX 侧由 `DURATION` 环境变量传入 `--benchmark-duration`,小于 900 秒时自动加 `--unsafe-override`(结果被标为无效)[3]。
- 所有指标只算测量阶段 [11]。时长到点后不再发新请求;已在飞的请求在宽限期内排空并**计入**指标 [12]。
- 另有覆盖率检查:TTFT 或 token 间延迟的观测必须延伸到测量阶段的至少 95%(源码值 0.95;教程某处写 98%,以源码为准),否则非零退出并标 `insufficient_profile_metric_coverage` [11][12][14]。
- 每个并发点必须用**全新启动的服务器**:InferenceX 强制一次部署只测一个并发(`CONC_LIST` 必须等于 `CONC`),文档要求扫并发时每个点重启服务器、换输出目录,不要清缓存后复用活服务器 [2][3]。harness 层面也禁止在场景下用逗号列表扫并发 [11]。
- InferenceX 有一个实验性快速模式(`AIPERF_EXPERIMENTAL_FAST=1`):时长 1,200 秒、每 lane 1 条预热 [3]。

### 4.6 错误与超时
- 普通错误(HTTP 500、超时)对会话不是终态:不重试,会话照常进入下一轮;错误请求计入 `error_request_count`,不贡献 TTFT/ITL 样本 [11]。
- 上下文溢出错误(错误体匹配 "context length"、"maximum context"、"context_length_exceeded"、"prompt is too long" 等子串)对该会话是终态;溢出率严格大于 1% 则整次运行 `submission_valid: false` [11][14]。分母 = 成功数 + 错误数 + 因溢出跳过数 [12]。
- 实时失败阈值:`agentx` 预设 `--failed-request-threshold 0.10`,测量阶段错误比例越线即取消运行 [14];InferenceX 事后校验同样用 0.10 [3]。
- 客户端 TCP 用户超时 900,000 ms;`agentx` 预设关闭了进度停滞看门狗(理由是低并发下一条长请求可能合理地超过 600 秒)[14]。
- InferenceX 打分顺序:重放退出码 → 聚合 → 结果校验 → 功耗校验,取第一个失败 [3]。

### 4.7 max_tokens、停止条件与输出长度
- 每条请求的 `max_tokens` = 录制的输出 token 数 `out` [11][13]。
- 场景强制往请求里注入 `ignore_eos=true`(显式传 false 会被拒绝),让服务器无视结束符、生成满 `max_tokens`。所以被测模型"本来会生成多长"不起作用:输出长度被钉在录制值上 [11][12][14]。
- 被测模型实际生成的文本不会回灌进后续轮次(历史里用的是合成的 assistant 内容),因此输出内容不影响后续 prompt [11]。
- 流式强制开启;端点为 OpenAI 兼容的 `/v1/chat/completions` [2][11]。
- token 计数采用服务器 `usage` 字段(`--use-server-token-count`,`agentx` 预设默认开)[14]。
- 录制的模型名被改写成 `--model`;只给一个模型时,主智能体与子智能体请求全部打到同一个模型。原本"Opus 主 + Haiku 子"的模型混用只保留在树形结构里 [11][13]。

## 5. 变体

- **完整版** `semianalysis_cc_traces_weka_062126` → HF `semianalysisai/cc-traces-weka-062126`:单请求输入最高约 99 万 token [3][7]。
- **256k 版** `..._062126_256k`:同样 393 条会话,在数据源头删掉每一条 `in + out > 256,000` 的请求(按请求粒度,不是按会话);子智能体组若内部请求全被删则整组删、部分被删则保留剩余;幸存请求保持原相对时间戳(若首条被删则整体平移到 t=0)[6][8]。
- 256k 版删得很深:因为上下文逐轮累积,一旦越过 256k 之后的轮次通常全都超限,所以长会话的整段深上下文尾部都没了——顶层请求从 56,798 降到 28,444,输入 token 总量降到完整版的 32% [8][11]。
- 哪些模型用哪版(InferenceX 代码硬编码):模型前缀以 `dsr1`、`qwen3.5`、`qwen3.8next` 开头的用 256k 版;**其余所有模型家族默认用完整版**;可用 `WEKA_LOADER_OVERRIDE` 强制指定 [2][3]。按此规则,DeepSeek V4.1 Flash(前缀 `dsv41flash`)用的是完整版。
- `--max-context-length` 是另一种更粗的做法:加载时丢弃"峰值输入+输出超限"的**整条** trace;InferenceX 通过 `AIPERF_MAX_CONTEXT_LENGTH` 可选传入 [3][11]。
- 无子智能体版本存在于历史语料(如 `cc-traces-weka-no-subagents-051826`,98 条),但 AgentX 场景明确拒绝无子智能体别名 [13][14]。
- 历史语料:HF 组织下从 042026 到 062126 共十余个带日期的版本(例如 061526 版 233 条);场景允许列表里保留 060226、060526、060826、061326、061526、062126 及各自的 256k 版用于复现 [13][14][17]。早期版本的筛选更严(例如 060226 的示例命令用 `--min-main-turns 20 --max-parallel-subagents 5`、trace v6)[6]。

## 6. 推测解码、精度校验与配置规则

### 6.1 推测解码:合成接受长度(golden AL)
- 问题来源:AgentX 发的是合成填充文本且 `ignore_eos`,草稿头在这种文本上的真实接受率没有意义;所以吞吐测试时把推测解码的接受长度**强制**成一个事先定好的值 [18]。
- 每个(模型、思考模式、草稿长度)只有一个提交进仓库的 golden AL;提交方可以任选受支持的草稿长度,但不能换别的接受目标;同一模型同一模式的所有提交(不分硬件厂商)用同一条曲线 [18][19]。
- golden AL 的测法:在 B300 上用真实推测解码启动模型,跑 SPEED-Bench Qualitative 的 `coding` 类别(80 条 prompt,流式 `/v1/completions`,`max_tokens` 4096,不设 `ignore_eos`),读引擎计数器,`AL = 1 + 被接受的草稿 token 数 / 验证步数`(1 是目标模型保证产出的验证 token),保留两位小数;由 `speedbench-al.yml` 工作流一键执行,日志与输出可审计 [18]。选 coding 类别的理由:AL 随领域而变,而智能体编程占 token 量最大 [18]。
- 各引擎的注入方式:vLLM 用 `rejection_sample_method: synthetic` + `synthetic_acceptance_length`;SGLang 用环境变量 `SGLANG_SIMULATE_ACC_LEN`(配 `match-expected`、`real-draft-token`);TensorRT-LLM 用 `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`,要填 golden AL 减 1(它不含验证 token);ATOM 用 `--spec-decode-acceptance-length` [18]。
- DeepSeek V4.1 Flash 的曲线(方法 DSpark,草稿采样 probabilistic,拒绝采样 block,自适应验证关闭;镜像 `vllm/vllm-openai:deepseekv41-flash-0909`,TP4,temperature 1.0)[20]:
  - 思考关:草稿长度 1→1.88,2→2.62,3→3.27,4→3.73,5→4.07;
  - 思考开(reasoning_effort high):1→1.81,2→2.43,3→2.90,4→3.26,5→3.51。
  - 草稿长度 6–8 被该镜像拒绝,没有赋值 [18][20]。仓库里 H100 SGLang 配置的注释写明用"五 token DSpark、思考开、golden AL 3.51" [21]。
- 评测(eval)作业不使用合成接受:它去掉合成设置,走真实验证 [18]。
- "接受长度闸门"(AL gate):每个带推测解码的智能体 eval 作业,先在同一台服务器上用 golden 曲线记录的采集设置、并发 64 重放那 80 条 prompt,读真实 AL,与 golden 值比较,落在 5% 以内记 `pass`,否则 `below`/`above`。它只做报告,不会让 eval 或提交失败 [22]。
- 草稿精度规则:草稿头/草稿模型必须"按发布原样"服务——用随 checkpoint 发布的草稿、按其存储精度、走钉住的上游镜像的默认处理;不允许提交方自行量化、改 dtype、换 checkpoint 或打补丁把草稿精度降到默认以下;合成接受下的结果不能当作合规证据 [19][23]。

### 6.2 "每个数据点都经过精度校验"具体是什么
- 仓库里的机制是**与吞吐分开跑的 eval 作业**,用途是抓住并行、并发、内核等吞吐优化造成的精度回退 [22]。
- 智能体场景默认跑 GSM8K(通过 `lm-eval`):对每个"模型 × runner × 框架 × 精度 × 推测解码 × dp-attn × 镜像 × 引擎环境变量"分组,在该组**最高并发**上起一个 eval-only 作业;多节点按拓扑取最高合格并发。TP/EP、KV offload、CLI 参数不拆分组 [22]。
- 也就是说:不是字面上"每个并发点各跑一次精度测试",而是每个部署分组在其最高并发点验证一次;同组其余并发点共享该结论 [22]。(这是仓库文档的口径;对外宣传语"每个数据点都经过精度校验"的出处不在本笔记范围内。)
- 阈值:GSM8K 默认下限 0.90;个别模型更高(dsr1、dsv4、gptoss 0.91,minimaxm2.5 0.92,glm5、qwen3.5 0.94);GPQA diamond 下限 0.30 [24]。`dsv41flash` 没有单独条目,适用默认 0.90 [24]。
- 额外的厂商套件:Kimi K3 的每个点自动跑工具调用 schema 套件(204 例 × 2 种流模式 = 408 项检查),MiniMax M3 跑 102 例套件;但它们的阈值是 0.0,仅作诊断 [22][24]。BFCL 只在显式指定时跑 [22]。
- 检查清单还提到"private evals":目标/验证模型用更低精度可以接受,前提是配置通过 private evals [19]。private evals 的内容未公开。
- harness 自己的 `submission_valid` 是另一层:它只校验重放规则是否被遵守和运行是否健康(见 4.6),与模型输出质量无关 [11][12]。

### 6.3 允许什么配置
- **上游镜像**:声称把 vLLM/SGLang 当一等引擎支持的厂商,必须用上游 Docker 仓库(hub.docker.com 的 `vllm`、`lmsysorg`)的镜像提交;成熟模型要用从上游主干分支构建的镜像,新模型可暂用模型适配分支但须在合理时间内迁回 [19][25]。例外:全新硬件(MI455X UALoE72、Vera Rubin NVL72、Rubin NVL8 等)和上游确实尚不支持的新模型架构 [19]。
- **上游 recipe**:每个 vLLM/SGLang 生产配置(含 Dynamo 部署、单/多节点、聚合/解耦)的生产有用参数、环境变量和启动命令,必须写进官方 vLLM recipes 或 SGLang cookbook,且对应的上游 PR 要在 InferenceX PR 合并**之前**已合并。仅 InferenceX 用的 harness 管线和合成基准控制项不必上游化 [19]。
- **禁止给引擎打补丁**:钉住的镜像必须原样运行——不许 `.patch`、改 site-packages、monkey-patch、覆盖容器文件、装 fork 的 wheel。唯一例外是随 PR 提交的 waiver 文件,写明改了什么、为何原镜像跑不了、上游 PR 链接、移除计划 [19]。实例:PR #2508 的 waiver 覆盖 Kimi K3 在 MI355X 上的 vLLM 配置,列了约 10 个尚未进 nightly 的 vLLM/AITER 修复 [26]。
- **自定义内核 / 加速手段**:原则是"让同一计算跑得更快的都可以";不接受减少模型架构 FLOPs 的做法(例如用 `--hf-overrides` 每隔几层跳过 indexer)。北极星原则:只用在意精度的客户在生产中实际使用的优化 [19]。新模型须关闭整数 all-reduce [19]。
- **引擎顺序**:每个模型有指定的原生/上游主引擎(vLLM 或 SGLang);要先交主引擎,才能再交 TensorRT-LLM、ATOM、TokenSpeed 等非 vLLM/SGLang 引擎 [25]。
- **KV cache offload**:初版只允许卸载到 CPU DRAM,且可选(可用 vLLM Connector、LMCache、SGLang HiCache、Mooncake、Dynamo KVBM 等);NVMe 等其他层级不在初版范围,暂定 v3.5 或 v4 再议 [25]。配置里 `kv-offloading` 只接受 `none` 或 `dram` [3][27]。
- **主机 DRAM 上限**:`允许的 CPU DRAM = 每服务器基准 DRAM ×(配置所用 GPU 数 / 服务器 GPU 总数)`,逐服务器适用。无标准内存配置的机型(HGX B200、HGX B300、MI355X 机箱)基准上限为每服务器 3 TB(8 卡用 4 卡则最多 1.5 TB);有标准配置的(TPUv7、GB200/GB300 NVL72)以标准装机容量为基准。3 TB 上限的理由是防止厂商堆到 6 TB 的不现实内存竞赛;TCO 会按服务器 DDR5 总容量的成本归一 [25]。master 配置里另有 `dram-utilization`(如 0.80)表示可用于 offload 的节点 DRAM 比例 [21][27]。
- **P/D 解耦**:允许。仓库里有多份 AgentX 解耦 recipe(如 `dsv41flash/sglang/gb300-fp4/agentx/disagg-dep16-hicache-variants.yaml`、TileRT 的 `disagg-1p1d-tp4-tp8-mtp.yaml`),并要求解耦部署的 prefill、decode、router 命令同样有上游文档覆盖 [19][28]。
- **流式间隔**:每个新增或重跑的 AgentX 点,有效 stream interval 不得超过 20 token。原因:指标是客户端测的,短于流间隔的回复会一次性到达,TTFT 会吞掉解码时间;间隔为 100 时 MiniMax M3 工作负载约 30% 的请求受影响。20 这个值由 NVIDIA TRT-LLM 工程师建议,PR #2685 采纳 [19]。
- **精度细则**(以 Kimi K3.1 Pro 首周政策为例):线性层、共享 expert、MoE combine、AllReduce 必须保持原生 BF16;MoE 权重可 MXFP4/NVFP4;KV cache 可 FP8,FP4 只有原生 checkpoint 自带时才允许;只允许动态 EPLB [25]。
- **Pareto 点数**:每条受影响曲线强烈建议至少 5 个实测 Pareto 点,不足需核心维护者批准 [19]。
- **审核流程**:硬件厂商的 CODEOWNER 填写检查清单签字,CI 机器人独立复核,再由核心维护者终审 [19]。
- **非官方结果**:只有 SemiAnalysisAI/InferenceX 仓库的结果算官方;fork 与厂商博客的结果必须标注 Unofficial [1][25]。

### 6.4 InferenceX 采用的主指标(补充,影响怎么读曲线)
- AgentX 默认横轴是 "E2E normalized interactivity":对每条有效请求算 `r_i = E2EL_i / OSL_i`(秒/输出 token),取分位数后取倒数;看板默认 P90。近似于 `1 / (TPOT + TTFT/OSL)`,即把排队和 prefill 时间摊进去的每用户输出速率 [25]。
- 其他横轴视图(E2E 延迟、常规 interactivity、TTFT)上的前沿点,必须同时是该主指标下的前沿点 [25]。

## 7. 自己怎么跑

### 7.1 命令
- 安装:Python 3.11–3.13 虚拟环境里 `pip install 'agentx-harness==1.0.13'`(专用环境,不要与上游 `aiperf` 同装;Linux ARM64 需要 C 编译器)[2][10]。
- 对一台已启动的 OpenAI 兼容服务器 [2]:
  ```bash
  agentx profile \
    --url http://127.0.0.1:8000 --endpoint /v1/chat/completions \
    --model deepseek-ai/DeepSeek-V4.1-Flash --tokenizer deepseek-ai/DeepSeek-V4.1-Flash \
    --tokenizer-trust-remote-code \
    --concurrency 8 \
    --public-dataset semianalysis_cc_traces_weka_062126 \
    --output-artifact-dir ./results/agentx-c8/aiperf_artifacts
  ```
  `agentx` 预设自动补齐:1 小时测量、每 lane 10 条预热、种子 42、起点 25%–75%、失败阈值 0.10、300 秒单树空闲上限、全部 393 条 trace [2][12][14]。
- 可加 `--server-metrics <url>/metrics` 抓服务器 Prometheus 指标;多副本路由需要会话亲和时设 `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true` 等环境变量 [2][11]。
- 扫并发:每个并发值重启服务器、换输出目录,保持同一种子 [2][11]。
- 结果在 `profile_export_aiperf.json`(看 `metadata.submission_valid` 与 `error_request_count`)和 `profile_export.jsonl`;`aiperf analyze swim-lane <目录>` 可画每条 lane 的时间线 [11]。
- 查某条已发布曲线每个点用的 recipe:`uv run --locked python -m infx.results.published_curve --model ... --hardware ... --framework ... --agentic --output-dir ...`,数据来自看板的公开 API(`/api/v1/benchmarks`,契约见 `inferencex.semianalysis.com/api/openapi.json`)[2]。

### 7.2 运行时间
- 单个并发点 = 首次数据集重建(CPU 密集;大语料冷启动会超过默认 300 秒,`agentx` 预设把超时放到 1,800 秒;重建结果按"语料 + 分词器 + 种子"缓存在 `~/.cache/aiperf/dataset_mmap`,之后几秒恢复)+ 预热(宽限 1,800 秒)+ 测量 3,600 秒 + 在飞请求排空 [11][12][14]。即每点 1 小时出头到约 2 小时,再加服务器加载模型的时间。
- InferenceX 矩阵对智能体作业的默认时长同为 3,600 秒 [27];一条曲线通常 6–8 个并发点(例:DSV4.1 Flash 在 H100 vLLM 上的并发列表为 1、2、4、8、16、20、24、28)[21],每点都要重启服务器。
- 最短的"有效"运行是 900 秒;更短的冒烟测试要加 `--unsafe-override`,结果标记为无效 [11][14]。

### 7.3 硬件需求
- 被测端:任何能提供 OpenAI 兼容 `/v1/chat/completions` 且支持 `ignore_eos` 的服务器;跑完整版语料需要服务器上下文窗口接近 1M token(否则溢出率超过 1% 即无效),约 256k 窗口的模型改用 `_256k` 语料 [11]。
- 仓库里 DSV4.1 Flash 的智能体配置覆盖了从 H100(8 卡 TP8)到 GB200/GB300 的多种机型 [21];H100 配置的注释说明为了让权重、indexer 缓冲和 KV 池装进 80 GB 做了专门调整 [21]。
- 客户端:多进程负载生成,worker 数约为 CPU 核数的 75% 减 1、上限 32;任一 worker CPU 超过 85% 会告警"结果可能不准" [11]。客户端机器的最低配置官方未给出。
- 功耗:多节点运行默认采集 GPU/CPU 功率 [1][27]。

### 7.4 成本
- 公开文档没有给出"跑一次 AgentX 要多少钱"的数字。可以确定的只有结构:每个数据点至少占用被测硬件 1 小时测量 + 预热/加载,曲线上每个点独立重启;数据集与 harness 本身免费(Apache-2.0)[2][7][10]。

## 来源

1. SemiAnalysisAI/InferenceX 仓库 README(commit 0b99145,2026-11-10)— https://github.com/SemiAnalysisAI/InferenceX
2. InferenceX `inferencex-e2e/docs/agentx-standalone.md` — https://github.com/SemiAnalysisAI/InferenceX/blob/main/inferencex-e2e/docs/agentx-standalone.md
3. InferenceX `inferencex-e2e/infx/bench/agentic/`(`traces.py`、`replay.py`、`run.py`、`requirements.txt`)— https://github.com/SemiAnalysisAI/InferenceX/tree/main/inferencex-e2e/infx/bench/agentic
4. InferenceX `infx/datasets/sample_proxy_traces.py` — https://github.com/SemiAnalysisAI/InferenceX/blob/main/inferencex-e2e/infx/datasets/sample_proxy_traces.py
5. InferenceX `infx/datasets/proxy_to_weka.py` — https://github.com/SemiAnalysisAI/InferenceX/blob/main/inferencex-e2e/infx/datasets/proxy_to_weka.py
6. InferenceX `infx/datasets/build_weka_hf_dataset.py` — https://github.com/SemiAnalysisAI/InferenceX/blob/main/inferencex-e2e/infx/datasets/build_weka_hf_dataset.py
7. Hugging Face 数据集卡片 `semianalysisai/cc-traces-weka-062126`(含 stats.txt)— https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126
8. Hugging Face 数据集卡片 `semianalysisai/cc-traces-weka-062126-256k` — https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k
9. 自行计算:下载上述两个数据集的 `traces.jsonl`(2026-11-10)后统计。非官方数字。
10. SemiAnalysisAI/agentx-harness 仓库 README(tag agentx-v1.0.13)与 PyPI 元数据 — https://github.com/SemiAnalysisAI/agentx-harness ,https://pypi.org/project/agentx-harness/
11. agentx-harness `docs/benchmark-modes/semianalysis-agentx-faq.md` — https://github.com/SemiAnalysisAI/agentx-harness/blob/main/docs/benchmark-modes/semianalysis-agentx-faq.md
12. agentx-harness `docs/tutorials/agentx-mvp.md` — https://github.com/SemiAnalysisAI/agentx-harness/blob/main/docs/tutorials/agentx-mvp.md
13. agentx-harness `docs/tutorials/weka-trace.md` — https://github.com/SemiAnalysisAI/agentx-harness/blob/main/docs/tutorials/weka-trace.md
14. agentx-harness `src/aiperf/common/scenario/inferencex_agentx_mvp.py`(场景定义源码)— https://github.com/SemiAnalysisAI/agentx-harness/blob/main/src/aiperf/common/scenario/inferencex_agentx_mvp.py
15. agentx-harness `src/aiperf/dataset/loader/`(`weka_trace_models.py`、`hash_ids_synthesis.py`、`weka_synth_buf.py`)— https://github.com/SemiAnalysisAI/agentx-harness/tree/main/src/aiperf/dataset/loader
16. InferenceX 术语表 AgentX 条目 — https://inferencex.semianalysis.com/glossary/agentx
17. Hugging Face API:`semianalysisai` 组织数据集列表 — https://huggingface.co/api/datasets?author=semianalysisai

18. InferenceX `infx/golden_al_distribution/README.md` — https://github.com/SemiAnalysisAI/InferenceX/blob/main/inferencex-e2e/infx/golden_al_distribution/README.md
19. InferenceX `docs/PR_REVIEW_CHECKLIST.md` — https://github.com/SemiAnalysisAI/InferenceX/blob/main/inferencex-e2e/docs/PR_REVIEW_CHECKLIST.md
20. InferenceX `infx/golden_al_distribution/deepseek-v4.1-flash__dspark__native__probabilistic-block.yaml` — https://github.com/SemiAnalysisAI/InferenceX/blob/main/inferencex-e2e/infx/golden_al_distribution/deepseek-v4.1-flash__dspark__native__probabilistic-block.yaml
21. InferenceX `configs/nvidia-master.yaml`(`dsv41flash-*-agentic-*` 条目)— https://github.com/SemiAnalysisAI/InferenceX/blob/main/inferencex-e2e/configs/nvidia-master.yaml
22. InferenceX `infx/evals/EVALS.md` — https://github.com/SemiAnalysisAI/InferenceX/blob/main/inferencex-e2e/infx/evals/EVALS.md
23. InferenceX `CONTRIBUTING.md`("Draft-model precision" 一节)— https://github.com/SemiAnalysisAI/InferenceX/blob/main/CONTRIBUTING.md
24. InferenceX `infx/evals/thresholds.yaml` — https://github.com/SemiAnalysisAI/InferenceX/blob/main/inferencex-e2e/infx/evals/thresholds.yaml
25. InferenceX `docs/MODELS.md`("AgentX Guidelines"、KV offload 政策、模型支持表)— https://github.com/SemiAnalysisAI/InferenceX/blob/main/inferencex-e2e/docs/MODELS.md
26. InferenceX `docs/waiver/2508.md` — https://github.com/SemiAnalysisAI/InferenceX/blob/main/inferencex-e2e/docs/waiver/2508.md
27. InferenceX `configs/CONFIGS.md` — https://github.com/SemiAnalysisAI/InferenceX/blob/main/inferencex-e2e/configs/CONFIGS.md
28. InferenceX `benchmarks/multi_node/srt-slurm-recipes/`(目录与 RECIPES.md)— https://github.com/SemiAnalysisAI/InferenceX/tree/main/inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes

## 没查到 / 不确定

- **征集与同意流程**:公开代码只显示代理库里有 `privacy_mode`(`anon`/`full`)字段、采样只取 `anon`;贡献者是谁、如何招募、签了什么同意条款、是否可撤回,仓库与数据集卡片都没写。"自愿提供(opt-in)"的说法来自官网术语表和已读过的文章,没有找到流程文档。
- **采集时间段的确切起止**:只能推出在 2026-04-16(子智能体标签迁移)到 2026-06-21(构建日)之间,且受 trace v7 与 CLI ≥ 2.1.139 的限制会更窄;公开 trace 里时间戳是相对值,无法还原日期。
- **贡献者人数、项目类型、编程语言分布**:未公开,脱敏后的数据也无从推断。
- **"300 万美元数据集"的算法**:本笔记未核对。按完整版 216 亿输入 token、1.06 亿输出 token 用 API 标价能否算出这个数,取决于缓存读写计价的假设,我没有做这个推算。
- **代理端的哈希算法细节**:块哈希是滚动(含前缀)还是独立内容哈希、用什么哈希函数,公开仓库里没有代理源码;只知道库里存 24 位十六进制串、块大小 64。"相同 id = 相同前缀内容"是从 harness 文档对其用法的描述推出的。
- **v7 trace 格式与更早版本的差别**:只知道 v1/v2 含图片内容需排除、数据集限定 v7;版本变更记录未找到。
- **第 1.5 节的"前缀断裂 10.4%"不等于压缩次数**:它混入了摊平并行智能体的交错。harness 的链检测能区分二者,但我没有运行 harness 的检测来得到真正的压缩事件数。
- **覆盖率阈值 95% 还是 98%**:源码是 0.95,教程一处写 98%;以源码为准,文档疑似未同步。
- **MVP 预设与 `agentx` 预设在历史上何时切换**:8 月发布时用的是哪套默认值(1,800 秒 / 0–100% 起点,还是 3,600 秒 / 25–75%)没有查提交历史。已知材料说"只统计预热后的一小时窗口",与现在的 `agentx` 预设一致。
- **private evals 的内容、KVV 与 "TB 4.0" 指什么**:文档提到但未定义。
- **跑一次的成本、客户端最低配置**:官方未给。
- **AIPerf 上游(ai-dynamo/aiperf)是否已合入这些 AgentX 功能**:本次只读了 SemiAnalysis 的 fork;fork 的文档版权署 NVIDIA,说明大部分代码来自上游,但没有逐项核对上游仓库。
- **inferencex.semianalysis.com 的方法论页**:只抓到术语表条目(内容很短);`/datasets` 页与博客里的方法论细节未在本笔记中单独核对。
