# N2 · AgentX 测什么、数字怎么算(指标 / token 口径 / 成本与功耗模型 / 看板机制)

调研日期:2026-11-10。代码类事实取自当天 `main`/`master` 的浅克隆:InferenceX `0b991451`(2026-11-11 00:39 +08:00)、InferenceX-app `9e9e2390`(2026-11-10 12:54 -06:00)。代码会变,引用时请带上日期。

---

## 1. 指标定义与公式

### 1.1 数据从哪来
- AgentX 的所有延迟/吞吐指标都是**客户端侧**指标,由 AIPerf 的逐请求记录 `profile_export.jsonl` 聚合而来;服务端 `/metrics` 抓取(`server_metrics_export.json`)只用于缓存、KV 占用等辅助量。(S1, S3)
- 聚合脚本先丢掉预热请求和出错请求,只对成功且处于 profiling 阶段的记录做统计;结果里保留 `records_total / records_warmup_dropped / records_error_dropped`。(S1)
- 每个统计量都输出 mean、p50、p75、p90、p95、std 六个值;百分位用线性插值(`k=(n-1)·p/100`)。总体是"本次运行一小时窗口内的全部成功请求",**每个请求贡献一个样本**。(S1, S2)

### 1.2 TTFT
- 取 AIPerf 每条记录的 `time_to_first_token`(ms→s),跨请求取百分位。看板 x 轴为 TTFT 时默认用 `p90_ttft`,可选 median/p75/p90/p95/p99。(S1, S6)
- 术语表定义:从提交请求到返回第一个生成 token 的延迟。(S5)

### 1.3 ITL 与 TPOT
- 代码里 `tpot_stats = stats_for("tpot", itls)`,即 **TPOT 与 ITL 用的是同一组数**(AIPerf 的逐请求 `inter_token_latency`),只是换了个键名。(S1)
- 看板侧同样把二者当别名:`tpot ≡ itl`(注释写明"单输出场景下 time-per-output-token 等于 inter-token-latency")。(S7)
- 另有第二种 ITL:`full_response_inter_token_latency`("full-response ITL")。看板对 AgentX 行的规则是:**只要该行有 full-response ITL,就优先用它**;没有才退回普通 `itl`。(S1, S7)
- AIPerf 文档里的定义(上游 `ai-dynamo/aiperf` 主分支;SemiAnalysis 用的是自己的 fork,可能略有出入)(S28):
  - `inter_token_latency = (request_latency − TTFT) / (OSL − 1)`,即**一个请求内的平均 token 间隔**,不是逐个间隔的分布。
  - `full_response_inter_token_latency = (HTTP 响应结束时刻 − 首个有效响应时刻) / (OSL − 1)`;与普通 ITL 的差别是区间一直算到 HTTP 响应完成,而不是最后一个解析出的内容 chunk。
  - `OSL = 输出 token + reasoning token`,所以推理(thinking)token 计入速度。
  - `TTFT` = 第一个内容响应(含 reasoning token)到达时刻 − 请求开始时刻。
  - `request_latency` = 最后一个有效内容响应时刻 − 请求开始时刻。
  - `ISL` = 用分词器对 prompt 编码后的 token 数。
  - `theoretical_prefix_cache_hit = 100 × Σ命中 block / Σ总 block`,由 trace 的 hash block 重叠算出,是"理想无限缓存"的命中率。
- 术语表对 TPOT 的定义:第一个 token 到达之后,相邻生成 token 之间的平均延迟;关系式 `interactivity ≈ 1000 / TPOT(ms)`。(S5)

### 1.4 交互速度(interactivity,tok/s/user)与 P90 的取法
- 公式:`pXX_intvty = 1 / pXX_itl`。即**先对"逐请求的 ITL"取 P90,再取倒数**,而不是对"逐请求速度"取 P90。(S1, S7)
- 所以"P90 交互速度"是**慢尾**:它等于 90% 的请求都不比它更慢的那个流式速度。术语表明确写:这不是 token 速率的第 90 百分位,取倒数会把排序反过来。(S5)
- 粒度是**每请求一个样本**(该请求内部 token 间隔的平均),不是逐 token 间隔;总体是整个一小时窗口内所有会话、所有子智能体的成功请求。(S1)
- 看板不信任产物里自带的 `*_intvty` 字段:注释说 harness 的定义"漂移过"(有的版本输出的是 `p(1/ITL)`),因此入库和叠加两条路径都强制重算为 `1/p(ITL)`,ITL 缺失时直接丢掉该字段。(S7, S8)
- 例外:`std_intvty` 是对逐请求 `1/ITL` 求的标准差。(S1)
- 固定序列(8k/1k 等)场景的"mean streaming speed"是 `1 / mean TPOT`,不是各请求速度的算术平均。(S6)
- AgentX 的百分位固定 P90;P75 可选但在 feature gate 之后,URL 参数 `i_pctl`。利润估算页只用 P90。(S8, S9)

### 1.5 端到端延迟(E2EL)
- 取 AIPerf 的 `request_latency`(请求发出到最后一个 token)。看板对 AgentX 行做别名 `e2el ≡ ttlt`(time-to-last-token)。(S1, S7)
- SemiAnalysis 自己说 E2EL 在 AgentX 里"意义变小了":它与输出长度成正比,P90 E2EL 主要被最长的 10% 输出拖着走,建议改看 TPS + TTFT 的组合。(S4)

### 1.6 实验性指标:E2E normalized interactivity
- 逐请求计算 `r = E2EL(s) / OSL`,对 `r` 取百分位,再取倒数:`pXX_e2e_norm_intvty = 1 / pXX(E2EL/OSL)`。方向同样是慢尾。(S1)
- 等价于"输出 token 数 ÷ 端到端延迟",近似 `1 / (ITL + TTFT/OSL)`,也就是普通交互速度加上一项与首 token 等待成正比的惩罚;输出越短惩罚越重。"normalized"指按输出长度归一,不是 0–1 打分。(S5)
- 看板里是 AgentX 专用的实验性 x 轴模式(`xmode=e2e-normalized-interactivity`),需要已持久化的逐请求派生指标。术语表说它"有意不完美",对高 TTFT 惩罚很重,AgentX 提交仍分别优化 interactivity 与 TTFT。(S5, S6)
- 新闻稿后半段说 SemiAnalysis 正在定义一个把 TPS 与 TTFT 合起来的新指标,并希望反映"任务完成的端到端速度"。(S4)

### 1.7 吞吐(每芯片)
- 时间窗 `duration = max(request_end) − min(request_start)`(成功请求的墙钟跨度,约一小时)。(S1)
- `input_tput = Σ input_sequence_length / duration`
- `output_tput = Σ output_sequence_length / duration`
- `total_tput = (Σ输入 + Σ输出) / duration`
- 每芯片值 = 上述各项 ÷ `num_gpus`。聚合(agg)部署的 `num_gpus` 是单 worker 的芯片数;分离式(disagg)部署是 `prefill_workers×每worker芯片 + decode_workers×每worker芯片`,即**整个部署的全部芯片**。(S1, S2)
- 入库后看板会把分离式行的输入/输出吞吐改成"每 prefill 芯片 / 每 decode 芯片"口径(原值另存为 `*_per_total_gpu`);**总吞吐 `tput_per_gpu` 始终是"每整体芯片"**,聚合与分离式可直接比。(S10, S11)
- 因此在看板的 Input / Output 两种 token 类型下,分离式配置看起来"每芯片更快、每 token 更便宜":历史数据中位数约 2×,输入最高 18×、输出最高 7×;Total 不受影响。文档称这是有意不改已发布数字。(S9)
- 还输出 QPS:按 1 秒窗口统计完成请求数,再求 mean/p75/p95。(S1)

---

## 2. Token 口径:tok/s per GPU 里算哪些 token

- **默认口径是"total":输入 + 输出**,而输入用的是每个请求的**完整输入序列长度**(`input_sequence_length`),不区分是否命中缓存。所以被缓存复用的前缀 token 每一轮都会再计一次。(S1)
- 这与 vLLM 博客 AgentX 图注"吞吐计入缓存输入、未缓存输入、输出三者"一致(该图注为简报里已读来源,本次未重抓)。看板指标名:`tpPerGpu` = "Token Throughput per Chip (tok/s/chip)",`inputTputPerGpu`、`outputTputPerGpu` 为另两条独立 y 轴。(S12)
- 术语表原话的意思:total token throughput 计入输入与输出;Rubin AgentX 文章报告的是"包含被复用输入"的总吞吐,和 output-only 结果比较前要先核对口径。(S5)
- **开关一(token 类型)**:y 轴指标成套存在 total / input / output 三个版本——吞吐、每百万 token 成本、每美元 token、每 MW 吞吐、每 token 焦耳都是如此。(S12)
- **开关二(token basis)**:URL 参数 `i_tokens=total|uncached`。`uncached` 在界面上叫 **"Horace He metric"**,含义是"未缓存 suffix + 输出":只计每轮新增输入和输出。目前藏在 feature gate 后面,未解锁的访客永远看到 total。(S8, S13)
  - 算法:`input_uncached = input_tput × (1 − theoretical_cache_hit_rate)`,总吞吐相应扣掉前缀部分。(S7)
  - 扣除比例用的是 trace 的**理论(无限缓存)命中率**,不是服务器实测命中,所以不随系统缓存好坏变化;它是近似值(理论命中率按 KV block 统计,再乘到实测输入吞吐上)。(S13)
- **为什么差出数量级**:
  - 数据集单请求中位数:输入 142,016、输出 444,约 320:1。(简报已知;S4)
  - 看板文档对生产数据的实测(DeepSeek-V4-Pro 与 GLM-5.2,2026-08):agentic 输入:输出 token 比中位数 **133:1**,固定 8k/1k 约 8:1;agentic GPU 前缀缓存命中率中位数 **0.92**(最小 0.005,p90 0.96)。(S9)
  - 推论(本笔记自行推算,非来源原话):按 133:1,total 口径约是 output-only 的 134 倍;按 92% 命中再算 uncached 口径,约为 `133×0.08+1 ≈ 11.6` 倍 output。三种口径之间各差约一个数量级。
- 计费含义:文档指出,若把输入价直接乘到原始输入速率上,会把约 92% 的缓存读取按新输入计价,**把 agentic 毛利高估近一个数量级**。(S9)

---

## 3. 缓存指标

### 3.1 理论命中率 vs 实测命中率
- `theoretical_cache_hit_rate`:来自 AIPerf 聚合里的 `theoretical_prefix_cache_hit`,优先用 `sum/count`(命中 block 数 ÷ 总 block 数)。它是"无限缓存"下的上限,由 trace 本身决定。(S1)
- 实测命中率来自各引擎 `/metrics`,每个引擎一个适配器(vLLM、SGLang、TRT-LLM、Dynamo-vLLM、ATOM、TokenSpeed)。(S3)
- 新闻稿标题称 AgentX 的 KV cache 命中率在 95% 以上。(S14)

### 3.2 缓存来源分布(HBM / 主机内存 / 重算)
- vLLM 适配器读 `vllm:prompt_tokens_by_source`,三个标签:`local_cache_hit`(芯片 HBM 内 KV 命中)、`external_kv_transfer`(从 CPU/外部层取回)、`local_compute`(重新 prefill)。各自除以三者之和,得到 GPU 命中率、CPU/外部命中率和总命中率。没有该指标时退回 `prefix_cache_hits / prefix_cache_queries`。(S3)
- 结果 JSON 里对应 `prompt_by_source = {gpu_cache_hit, cpu_or_external_cache_hit, computed}`。(S3)
- 看板"Prefix Cache Reuse"页(`/infx/cache-reuse`)按并发堆叠三段:芯片 HBM KV cache、其后的主机层、重算。(S10)
  - 主机段取 CPU-offload 命中率,只有当行里没有 CPU 数字时才退回 `server_external_cache_hit_rate`,**两者从不相加**。(S10)
  - TensorRT-LLM 开 offload 时只报一个合并值,画成一段"Combined chip + CPU"。(S10)
  - 有的行 GPU 命中率 >1(GB300 Dynamo 1.012,另有 1.185),显示时裁剪到 100%,原值留在 tooltip。(S9, S10)
  - 只在 Dynamo 前端抓取的 GB300 扫描没有服务端计数器,对应并发画成缺口而非 0。(S10)
  - 固定序列场景完全不记录缓存层。(S10)

### 3.3 KV offload 如何"记功"
- 吞吐层面**没有单独加分**:offload 让更多前缀被复用,从而提高整体吞吐/并发,这个收益直接体现在 total tok/s 上。(S1,推断)
- 每个结果记录 `kv_offloading`、`kv_offload_backend`(名称+版本)、`allocated_cpu_dram_gb`;还有 GPU↔CPU 的搬运字节数、耗时、带宽(bytes/s)。(S2, S3)
- 主图上,开了 KV offload 的点外面多画一圈虚线圆;点开后显示 offload 类型、引擎、chip 命中率与 CPU 命中率。同一条曲线上可以混有开/不开 offload 的点。(S4)
- 规则:CPU KV offload 是可选优化(vLLM connector、LMCache、SGLang HiCache、Mooncake、Dynamo KVBM 等均可,也可关闭);NVMe offload 暂缓;CPU DRAM 须按所用 GPU 比例分配,非标准 DRAM 系统上限 3 TB。(S4)
- 计价层面(收入类指标):缓存命中率 = GPU 命中 + external 命中(若有 external 指标),否则 GPU + CPU,裁剪到 [0,1]。理由是 326 条生产行的核对显示 external 已包含 offload 层,再加 CPU 会重复计(106 行里 56 行超过理论上限)。(S9)

### 3.4 点详情页
- 包含:ISL/OSL 分布、interactivity 与 TTFT 随时间、KV cache 占用、请求队列深度、前缀缓存命中率、输入与 decode 吞吐、prompt token 来源拆分、unique 输入 token 随时间;可在 warmup 与 profiling 两阶段间切换。(S4)
- 另有请求时间线(按会话或按 worker 排列),每个请求可点回数据集页的对应会话与轮次。(S4)

---

## 4. 成本与功耗

### 4.1 每 GPU 小时成本(TCO 输入)
- 每种硬件在常量表里有四个数:`tdp`(W)、`power`(每 GPU 的 all-in 公用电力,kW)、`costh`(自有,$/GPU-hr)、`costr`(租赁,$/GPU-hr)。(S15)
- 三个成本档:**Owning at large hyperscaler volume**(`costh`)、**Rent – 3-year commit**(`costr`)、**Custom user values**。(S12)
- 来源标注:"SemiAnalysis Market July 2026 Pricing Surveys & AI Cloud TCO Model"。(S16)
- 术语表:自有口径含摊销的硬件、网络、设施、电力和资本成本;租赁口径是三年期云预留的客户价;二者是**二选一**的成本基准,不能相加。(S5)

| 硬件 | TDP (W) | all-in 功率 (kW/GPU) | 自有 $/GPU-hr | 租赁 $/GPU-hr |
|---|---|---|---|---|
| Vera Rubin NVL72 | 1800 | 3.30 | 3.61 | 8.50 |
| GB300 NVL72 | 1400 | 2.12 | 2.31 | 5.00 |
| GB200 NVL72 | 1200 | 1.87 | 1.86 | 4.00 |
| B300 | 1100 | 1.90 | 2.26 | 4.25 |
| B200 | 1000 | 1.71 | 1.73 | 3.70 |
| H200 | 700 | 1.37 | 1.22 | 2.90 |
| H100 | 700 | 1.37 | 1.17 | 2.00 |
| MI355X | 1400 | 2.09 | 1.50 | 2.90 |
| MI325X | 1000 | 1.69 | 1.10 | 1.60 |
| MI300X | 750 | 1.39 | 0.95 | 1.30 |
| TPU7x | 980 | 1.207 | 1.21(内部口径 1.03) | 2.00 |
| RTX PRO 6000 | 600 | 0.975 | 0.68 | 0.52 |

(S15;数值为 2026-11-10 代码中的值)

- TPU 另有 internal / external TCO 之分(`g_tco`),默认 internal;内部口径约 $1.03/芯片小时。(S5, S15)

### 4.2 每百万 token 成本、每美元 token
- `$/M tok = TCO($/chip-hr) × 1,000,000 / (3600 × tok/s/chip)`。(S5, S17)
- `tok/$ = tok/s/chip × 3600 / TCO($/chip-hr)`。(S17)
- 两者都有 total / input / output 三套,分母分别用对应的每芯片吞吐。(S12, S17)
- **看板默认 y 轴是"Total Tokens per $1 TCO"(自有口径,`y_tokensPerDollarH`)**。(S5, S6)
- 这类倒数指标在图上只在实测点上画;计算器做插值时是先对吞吐做样条,再推出成本,而不是直接对成本做样条(否则在节点之间会系统性偏高,中位数 +5.7%)。(S9)

### 4.3 收入类指标(Token Revenue per GPU Hour)
- 独立于 TCO 的另一族 y 轴:`收入 = 每小时 token 量 × 售价`。(S9, S12)
- 默认"normalized"定价:未缓存输入与输出都是 $1/M,缓存输入 $0.10/M。(S9)
- 另一选项 OpenRouter(`i_revenue=openrouter`):实时取所选模型在 OpenRouter 上的公开 prompt / completion / cache-read 价格;目录里没有 cache-read 价时按输入价的 10% 兜底。(S8, S9)
- 输入流先打折再计价:`billableInput = inputTput × (1 − hit × (1 − cacheReadRatio))`,`cacheReadRatio` 默认 10%(文档称 DeepSeek 与 Anthropic 公布的都是这个比例)。(S9)
- 没有缓存遥测的行按未缓存价计输入;GB300 行例外,退回理论命中率。(S9)
- Fleet Lifecycle / Profit Estimator 页再叠加利用率(只缩收入,不缩 TCO)、模型授权费(按收入比例)、可用性折减等业务假设。(S9)
- Profit Estimator 的默认 P90 交互速度目标(2026-10-25 的 API 扫描定的):DeepSeek V4.1 Flash **150**、DeepSeek V4 Pro 170、GLM-5.2/5.3 135、MiniMax M3 115、Kimi K3 60 tok/s/user。选法是"让有报价的官方 SKU 行数最多的最低目标"。(S9)

### 4.4 每兆瓦性能与功耗口径
- `tok/s/MW = tok/s/chip × 1000 / power(kW)`,`power` 是上表的 **all-in 公用电力**(含主机、网络、制冷,PUE 已含)——是容量模型,不是实测。(S10, S17)
- 术语表:TDP 只覆盖加速器本身,不适合做设施级容量规划;provisioned 口径与实测功率/能耗遥测是两回事。(S5)
- 能耗有四个"边界"(S10):
  - B1 GPU 实测:`avg_power_w`、`joules_per_output_token`、`joules_per_total_token`
  - B2 GPU 额定:TDP
  - B3 公用电力额定(all-in):`power × 1000`
  - B4 实测→IT→PUE 的模型推算(另有不含 PUE 的 B4-IT)
  - 同一点上恒有 B3 ≥ B4 ≥ B1、B2 ≥ B1。
- 实测部分("PowerX"):2026-11 起多节点运行默认采集实测功率,包括 NVL72 的 Grace CPU、逐主机 GPU、AMD EPYC CPU package 功率。(S18)
- 开源 power model(beta,2026-10):多数公开集群只有 GPU 级遥测,模型用它加上其余部件与损耗的估计推出系统功率;可输入每 GPU 实测瓦数、每 socket 的 CPU(+DRAM)实测瓦数,workload 分 `fixed-seq-len / agentic / agentic-cpu-offloading`。(S19)
- 新闻稿:正在采集细粒度与粗粒度功率遥测,目标是更准确的"每单位智能的焦耳数"。(S4)

---

## 5. 看板机制

### 5.1 URL 参数
- 前缀约定:`g_` 全局、`i_` 推理图、`e_` 评测、`c_` 计算器、`r_` 视频。写入有 150 ms 防抖。(S20)

| 参数 | 含义 | 来源 |
|---|---|---|
| `g_model` | 模型显示名 | S8 |
| `g_rundate` / `g_runid` | 截止日期 / 指定某次 GitHub Actions run 的快照 | S6, S8 |
| `g_tco` | TCO 口径 internal / external | S8, S15 |
| `i_seq` | 场景:`1k/1k`、`1k/8k`、`8k/1k`、`agentic-traces` | S6 |
| `i_prec` | 精度(fp4、fp8、bf16、int4 等) | S6 |
| `i_metric` | y 轴指标键,如 `y_tpPerGpu`、`y_tokensPerDollarH`、`y_costh` | S6, S8 |
| `i_xmode` | x 轴模式:`interactivity`、`ttft`、`e2e`、`e2e-normalized-interactivity` | S6 |
| `i_xmetric` | x 轴为 TTFT 时的百分位,默认 `p90_ttft` | S6, S8 |
| `i_pctl` | AgentX 延迟百分位 p90(默认)/ p75 | S6, S9 |
| `i_tokens` | token 口径 `total` / `uncached` | S8 |
| `i_revenue` | 收入定价来源 `normalized` / `openrouter` | S8 |
| `i_active` | 图例中激活的硬件_框架键,逗号分隔 | S8 |
| `i_gpus` | "GPU 对比"选中的硬件键(如 `gb300_vllm`),配合日期做同一配置的跨日期对比 | S8, S21 |
| `i_dstart` / `i_dend` | 对比的日期范围起止 | S8, S21 |
| `i_dates` | 指定若干日期;`YYYY-MM-DD~rRUN_ID` 可精确到某次 run | S20b |
| `i_best` | 每 SKU 只显示最佳框架(1/0) | S6, S8 |
| `i_optimal` | 只显示帕累托边界点(`0` 为显示全部) | S6, S8 |
| `i_frontier` | 高亮全局帕累托前沿 | S8 |
| `i_rulers` | 在固定 x 处的"Perf Ruler"对比线 | S10 |
| `c_ivtarget` | 计算器的交互速度目标 | S20 |

- 实例:vLLM 10-23 博客图 1 的出处链接是 `/inference/deepseek-v41-flash?i_xmode=interactivity&i_seq=agentic-traces&g_model=DeepSeek-V4.1-Flash&i_best=0&i_active=gb300_dynamo-vllm` ——即 AgentX 场景、x 轴为交互速度、关掉"每 SKU 只留最佳框架"、只激活 GB300 上的 Dynamo+vLLM 曲线。(S31)
- 提交页生成的对比链接正是 `g_model + g_rundate + i_gpus + i_dstart(上一次日期) + i_dend(本次日期) + i_prec`,用来看同一配置前后两次提交的差。(S21)

### 5.2 帕累托前沿与曲线
- 前沿 = 没有被其他实测点在两个维度上同时超过的点;代码按 x 排序后做一次扫描。(S5, S22)
- `optimal=true`(默认)时,按"硬件 × 精度 × 快照日期"各自只返回边界点。(S6)
- `best=true`(默认):同一 SKU 上多个框架按**其实测前沿从原点起覆盖的面积(hypervolume)**打分,只留最高的;越低越好的轴先取倒数。文档强调这不是"引擎的普遍优劣"。看板要同时显示 vLLM 与 SGLang 时会先要求用户确认"不挑事"。(S6)
- InferenceX v3 起,**同一(模型、SKU、引擎)只画一条"最佳可用"曲线**:曲线上不同的点可以用不同优化(投机解码、分离式、KV offload),不再为每种组合单画一条;每个点仍可点开看配置与 CI 出处。(S4)
- 生成前沿的方式:对同一部署扫"并发的 Claude Code 会话数"。(S4)
- 计算器的插值用 Steffen 单调三次样条(与 `d3.curveMonotoneX` 一致),并裁剪到实测范围内;结果是"由真实点推出的估计值"。(S9)
- PR 审查清单建议每条受影响曲线至少 5 个前沿点,不足需核心维护者介入。(S23)

### 5.3 日期、版本与运行频率
- 日期取自 GitHub Actions workflow run 的 `created_at`;默认取"每个配置在该日期当天或之前的最新一次 run"。(S6, S24)
- 一次正常的 workflow run 是一条曲线的完整快照,**最新的 run 替换旧曲线**;`append-only: true` 的 PR 只补点,并入最近一条同镜像曲线。(S23b, S24)
- **不是每晚全量重跑**。基准只在 PR 改动 `perf-changelog.yaml` 时触发(矩阵由新增条目决定);合并到 `main` 后不重跑,而是通过 `/use <run_id>` 复用 PR 上那次绿色的完整扫描并入库。(S23b)
- 仓库里的定时任务只有:镜像更新机器人"Klaud Cold"(每 4 小时一次自动扫描、每小时一次唤醒兜底)、每周的 AL gate、runner 离线日报等。README 用的词是"continuous"/"near real-time"。(S18, S25)
- 历史条目不可改;每个生成的 recipe 带确定性指纹。(S23b)
- harness 版本:目前是 v1.0.x 小修,v1.1 会有较大改动;**同一模型的所有提交始终跑同一个 minor 版本**以保证可比。(S4)
- 投机解码在 AgentX 里用"合成接受":接受长度取仓库里提交的 golden AL 曲线(按模型、thinking 模式、draft 长度,在 SPEED-Bench 上测得),不可自行替换。(S4, S23)
- AgentX 的 stream interval 必须 ≤ 20 token(PR #2685):因为指标在客户端测,回复比 stream interval 短时会整块到达,TTFT 会吞掉 decode 时间;interval 为 100 时 MiniMax M3 约 30% 的请求受影响。(S23)

### 5.4 提交如何进入看板
- 任何人可提 PR;流程:PR 通过验证并跑出绿色完整扫描(含 evals)→ 对应芯片厂商的 **CODEOWNER** 按清单签字 → 核心维护者(SemiAnalysis)最终批准 → 授权维护者发 `/use <run_id>` → 走复用路径合并并入库。(S23b)
- 外部 fork 的 PR 不能自己打标签,须由维护者打标签(只对当前 head SHA 生效)后才会跑。(S23b)
- CI 机器人会独立复核签字清单(目前为 advisory):是否 CODEOWNER、绿色扫描与 evals、上游 recipe 已合并、用的是上游 vLLM/SGLang 官方镜像、没有改模型结构的 hack、没有给引擎打补丁、draft 精度未被降低等。(S23, S23b)
- 硬性规则举例:生产有用的启动参数必须已写进官方 vLLM recipes 或 SGLang cookbook 且**先于** InferenceX PR 合并;不接受减少模型 FLOPs 的优化;镜像须原样运行。(S23)
- 硬件:跑在接入仓库的自托管 runner 上,算力由厂商与云商赞助——AMD 提供 MI355X 等,NVIDIA 提供 GB200 NVL72(经 OCI)与 B200,另有 Crusoe、CoreWeave、Nebius、TensorWave、Oracle、TogetherAI。(S18)
- 只有官方仓库的结果算"Official";fork 必须标"Unofficial"。未入库的 run 可用 `?unofficialrun=` 叠加到图上,但不进表格、CSV 与成本面板。(S9, S18)
- 写入公开数据库需要对该 run 及其产物的明确发布批准。(S24b)

---

## 6. /evaluation 与 /reliability

- **Reliability**:数据是每个硬件每天的 `n_success / total`(基准 job 成功数 ÷ 总数),按"近 3 天 / 7 天 / 1 个月 / 3 个月(默认)/ 全部"滚动聚合成成功率(保留两位小数)。(S26)
- **"Chip reliability"页面已被下线**(页面与入口删除),只保留三个 API:`/api/v1/reliability`、`/api/v1/reliability/daily`、`/api/v1/views/reliability`;数据与算法未变。(S20)
- **Evaluation**(`/infx/evaluation`):展示各配置的准确性评测结果,目的是确认量化/优化后的配置没有掉精度。扫描默认给选中的 8k1k 子集和 **AgentX 的 GSM8K 子集**跑 eval-only 任务;提交必须"通过 evals"。(S23, S25b)
- 新闻稿里有一例:握手 bug 导致 KV 没传过去、decode 从头重算,gsm8k 得 0 分——说明 eval 也用来抓正确性故障。(S4)
- evaluation 页的具体图表与阈值本次未读源码细节,见"没查到"。

---

## 7. 被当作 SLA 的交互速度阈值

- InferenceX 自己**不规定统一 SLO**:术语表说可接受目标因产品而异,应在自己的阈值处读前沿。(S5)
- 实际出现的数字:
  - **150 tok/s/user**:DeepSeek V4.1 Flash 在 Profit Estimator 的默认 P90 目标(由"覆盖最多 SKU"的扫描选出,不是体验研究);另有一篇官方博客标题就是"GLM-5.3 AgentX … SGLang 150 toks"。(S9, S27)
  - 170(DeepSeek V4 Pro)、135(GLM-5.2/5.3)、115(MiniMax M3)、60(Kimi K3);Kimi K3 的 MI355X 白皮书仍用 45。(S9)
  - 计算器文档举例用的是 200 tok/s/user。(S9)
  - SemiAnalysis 自己的文章反复用 **150 tok/s/user(P90)** 作比较点:AgentX v3 文章与 GLM-5.3 博客("NVIDIA 成本效率最高 5 倍")、InferenceX v2 文章(DeepSeek R1 8k/1k,开 MTP 后每百万 token 约 $2.35→$0.11)、GLM-5.3 稀疏注意力文章(GB200 约 $0.044/M total、$5.92/M output)。这些文章都没有给出"为什么是 150"的依据。(S27, S29)
  - vLLM 2026-09-08 博客用的是 **P90 50 tok/s/user**,称其为常见且要求较高的延迟 SLO;其吞吐列脚注写明 total 口径"计入输入、输出与缓存 token"。(S30)
  - vLLM 2026-10-23 博客的数字是结果而非阈值:954K tok/s/GPU 对应 P90 **196** tok/s/user;CPU 开销表里有 P90 约 40→**159**、109→153。搜索摘要称 10-07 那篇是在"150 TPS 约束"下报 5.3×(未重抓原文)。(S31)
- **160 tok/s/user** 作为阈值的出处没有找到;最接近的是上面 vLLM 表格里的 159。

---

## 8. 新闻稿后半段(第 100,000 字符之后)覆盖了什么

1. AMD/ROCm 相关上游 PR 收尾(Mooncake、MiniMax-M3 day-0 的三个 vLLM 正确性修复)。
2. AgentX 激活了哪些上游工作,以及"允许的优化"政策(CPU KV offload 可选、NVMe 暂缓、DRAM 3 TB 上限)。
3. 数据集的采集方式:自建代理拦截 Claude/Codex 的 HTTP 请求;累计 8,000+ 会话、340 万请求、6,100 亿 token、超过 300 万美元花费;开源的是其中 393 个会话的子集。
4. 脱敏与重建的局限(思考内容被加密、服务端模板与分词器不可见、图片/文档 token 数靠校准填充)。
5. 后处理规则(去掉 Claude Code 的安全监控与标题生成请求、去掉重建输入 >990k 的请求、去重),采用 WEKA trace 格式。
6. 分布:ISL 中位 142k、OSL 中位 444、轮间延迟中位 3.84 s(约 10% 超过 1 分钟);175 个会话含子智能体,共 1,697 次子智能体运行,每会话中位 4 次,单次中位墙钟 2.27 分钟。
7. 重放方法:会话建成 DAG,子智能体的 spawn/join、auxiliary 请求;预热两阶段(随机取 25%–75% 处重建状态,再推进 10 个请求,预热请求输出上限 1 token);profiling 一小时;每条流 5 分钟空闲上限(对应 Anthropic 默认缓存 TTL);每次重放加唯一的 cache-bust 标记。
8. 指标立场:TPS 与 TTFT 必须一起看;E2EL 意义下降;正在定义合并指标。
9. 投机解码的 golden AL 做法。
10. 看板改版:一条最佳曲线、点详情页、KV offload 虚线圈、请求时间线、会话火焰图。
11. 路线图:v1.0.x 小修、v1.1 大改、NVMe/SSD offload、更大更新的数据集(保留消息边界)、功率遥测。
12. 之后转入历史单轮(8k1k)数据。

(S4)

---

## 来源

- S1 InferenceX 仓库 `inferencex-e2e/infx/results/agentic/request_metrics.py`、`common.py`、`process_agentic_result.py` — https://github.com/SemiAnalysisAI/InferenceX/tree/main/inferencex-e2e/infx/results/agentic
- S2 同目录 `__init__.py`(`build_result`、`_gpu_shape`)
- S3 同目录 `server_metrics.py`、`backends/base.py`、`backends/vllm.py`
- S4 SemiAnalysis 新闻稿《AgentX - InferenceXv3: Does the CUDA Moat Hold Up…》后半段(offset 100000 起)— https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat
- S5 InferenceX 术语表(源文件 `packages/app/src/lib/content/glossary.ts`;线上 https://inferencex.semianalysis.com/glossary ,条目如 /glossary/interactivity、/glossary/p90-interactivity、/glossary/e2e-normalized-interactivity、/glossary/throughput、/glossary/tokens-per-megawatt)
- S6 InferenceX-app `packages/app/src/lib/views-api/docs/inference.ts`(`/api/v1/views/inference` 参数文档)— https://github.com/SemiAnalysisAI/InferenceX-app
- S7 InferenceX-app `packages/app/src/lib/benchmarks/benchmark-transform.ts`(agentic 指标别名、`TokenBasis`)
- S8 InferenceX-app `packages/app/src/components/inference/InferenceContext.tsx`(URL 参数读写)
- S9 InferenceX-app `docs/tco-calculator.md`
- S10 InferenceX-app `docs/data-transforms.md`
- S11 同 S9 "The disaggregated token-split trap" 一节
- S12 InferenceX-app `packages/app/src/components/inference/metric-registry.ts`
- S13 InferenceX-app `packages/app/src/components/inference/ui/TokenBasisToggle.tsx`
- S14 https://inferencex.semianalysis.com/glossary/agentx (页面上列出的文章副标题)
- S15 InferenceX-app `packages/constants/src/gpu-keys.ts`(`HW_REGISTRY`)与 `packages/app/src/lib/catalog/constants.ts`(`getGpuSpecs`)
- S16 InferenceX-app `packages/constants/src/tco.ts`;TCO 模型 https://semianalysis.com/ai-cloud-tco-model/
- S17 InferenceX-app `packages/app/src/lib/charts/chart-utils.ts`(`writeThroughputChartFields`)
- S18 InferenceX `README.md` — https://github.com/SemiAnalysisAI/InferenceX
- S19 InferenceX `power_model/README.md`
- S20 InferenceX-app `docs/architecture.md`;S20b `docs/state-ownership.md`
- S21 InferenceX-app `packages/app/src/components/submissions/submissions-utils.ts`
- S22 InferenceX-app `packages/app/src/lib/charts/pareto-frontier.ts`
- S23 InferenceX `inferencex-e2e/docs/PR_REVIEW_CHECKLIST.md`;S23b `CONTRIBUTING.md`
- S24 InferenceX-app `docs/data-pipeline.md`;S24b `docs/ingesting-runs.md`
- S25 InferenceX `.github/workflows/`(cron 定义,`klaud-plan.yml` 等);S25b `.github/workflows/README.md`
- S26 InferenceX-app `packages/app/src/components/reliability/aggregate.ts`、`src/lib/views-api/docs/reliability.ts`
- S27 InferenceX-app `packages/app/content/blog/glm-5-3-agentx-nvidia-vs-amd-sglang-150-toks.mdx`(仅据文件名/标题,正文未读)
- S28 AIPerf metrics reference — https://github.com/ai-dynamo/aiperf/blob/main/docs/metrics-reference.md
- S29 InferenceX-app `packages/app/content/blog/` 下的 `agentx-inferencexv3-does-cuda-moat.mdx`、`inferencex-v2-nvidia-blackwell-vs-amd-vs-hopper.mdx`、`sparse-savings-persistent-demand-inside-glm53.mdx`(官方博客源文件,只检索了含"150"的句子)
- S30 vLLM 博客 2026-09-08 — https://vllm.ai/blog/2026-09-08-vllm-agentx (经摘要工具读取)
- S31 vLLM 博客 2026-10-23 — https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep (经摘要工具读取);10-07 一篇仅见搜索摘要 https://vllm.ai/blog/2026-10-07-deepseek-v41-flash

## 没查到 / 不确定

- **"$38.96 per GPU-hour at DeepSeek off-peak prices"的算法没有公开**:该数字出自 vLLM 2026-10-23 博客的 TL;DR,原文只给了结果,没有给缓存输入/未缓存输入/输出的单价、假设的命中率、公式或脚注(S31)。它是"把吞吐折算成假想的 API 毛收入",不是成本也不是利润。看板的收入指标只有 normalized 与 OpenRouter 两种价格来源,代码里没有"DeepSeek off-peak"选项,所以这个数应是 vLLM 团队自行套 DeepSeek 官方错峰价算的——后半句是推测。对照:同一博客系列 09-08 那篇给 Opus 5 对比时写明了算法(缓存输入 $0.50/M、未缓存输入 $5/M、输出 $25/M,假设理论命中率全部命中,不含 cache-write 与长上下文溢价)(S30),DeepSeek 那个数很可能是同一套路。
- **160 tok/s/user** 阈值的出处:没找到。
- AIPerf 定义取自上游主分支文档的前 10 万字符;SemiAnalysis fork 在 AgentX 所用版本(v1.0.x)上是否完全一致未核对。
- "pending SemiAnalysis review"这一状态字样在两个仓库里都没搜到;看板提交页(`/infx/submissions`)的状态展示未读。
- `/infx/evaluation` 页面具体展示哪些基准、是否有及格线,未读源码细节;只确认了 GSM8K 用于 AgentX eval。
- 术语表把"P90 interactivity = 1/P90 full-response ITL"限定为"Rubin AgentX 分析中"的定义;代码显示看板对所有带该字段的 AgentX 行都这么做,但早期没有该字段的行用的是普通 ITL——**不同时期的点口径可能不完全相同**。
- 成本表数值的更新频率、以及 `costh`/`costr` 背后的具体折旧年限、资金成本、电价假设:在付费的 AI Cloud TCO Model 里,公开代码只有结果值。
- InferenceX 聚合脚本的每芯片输入/输出吞吐除以"全部芯片",看板入库后改成按 prefill/decode 池分别除——线上某张图具体用哪种,取决于 chip-basis 开关(`g_chips`),未逐图核对。
