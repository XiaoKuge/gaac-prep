# N3 · AgentX 的结果与生态影响(调研笔记)

调研日期:2026-11-10。每条事实后的方括号是来源编号,清单在文末。标"二手"的表示没有读到原文,只读到转述或搜索摘要。

读数提示:InferenceX 的"每芯片吞吐"在 AgentX 里通常是 **total tokens**(输入 + 缓存命中的输入 + 输出),所以数值很大(几万到近百万 tok/s/GPU),不能和 8k1k 时代的输出吞吐直接比 [2][15]。

---

## 1. 覆盖矩阵:模型 × 硬件 × 引擎

### 1.1 发布时(2026-08-24,AgentX 1.0 / InferenceX v3)

- 模型五个:DeepSeek V4 Pro 0813(约 1.6T 参数、49B 激活)、Kimi K3(2.8T)、MiniMax M3(文中写 432B,vLLM 与看板写 428B)、Qwen3.5 397B、GLM 5.3(在 GLM5.2 744B 基础上做了后训练)[1][2][18]。
- Qwen3.5 原生上下文 262k,因此跑的是 256K 截断数据集,其余模型跑 1M 数据集 [1]。
- 硬件:MI355X、GB300 NVL72、GB200 NVL72、B300、B200、MI325、MI300X、H200、RTX Pro Servers;原文预告"Rubin 本月稍后到,TPU 和 MI455X UALoE72 今年稍后到" [1]。
- 引擎与编排:vLLM、SGLang、TensorRT-LLM、AMD ATOM(AMD 版的 TensorRT-LLM 对应物)、NVIDIA Dynamo(作路由/编排,与 vLLM/SGLang/TRT-LLM 组合成 "Dynamo vLLM" 等);KV 层有 Mooncake、LMCache、vLLM SimpleCPU、SGLang HiCache、Dynamo KVBM [1][8]。
- 规模:完整矩阵约 2 MW、1000+ 芯片持续运行 [1]。
- 基准配置主要跟随 recipes.vllm.ai 和 SGLang cookbook 的上游镜像,目的是量"客户实际能拿到的性能"而不是专门调过的镜像 [1]。

### 1.2 现在(2026-11-10 看板 /overview 页,开源引擎视图,@100 tok/s/user 档)

- 行(模型)已换了一轮:Muse Spark 1.3/Open 816B、DeepSeek V4.1 Flash 552B、Kimi K3.1 Pro 2.8T、MiniMax M3.1 Flash 428B、GLM5.2/GLM5.3 744B、Qwen3.5 397B、Qwen3.8 Flash Next 176B;另保留一行 Qwen3.5 的 8K/1K 作对照 [7]。
- 列(硬件)五个:B200、MI355X、B300、GB200 NVL72、GB300 NVL72。H200、MI300X/MI325X、Rubin、TPU 不在这张总览表里 [7]。
- 页面有"Show inactive(已弃用/维护模式的模型)"开关,说明 DeepSeek V4 Pro、Kimi K3、MiniMax M3 这些首发模型已被后继版本替换或转入非活跃 [7]。Rubin 文章也写明"DeepSeek V4 Pro 的运行在成文时已被 DeepSeek V4.1 Flash 取代" [11]。
- 固定序列长度场景(8k1k 等)已被降为"maintenance mode" [13]。
- 精度:总览表的 AgentX 行全部是 FP4 [7]。
- SLO 滑块档位:30 / 50 / 75 / 100 / 150 / 200 tok/s/user [7]。

### 1.3 八月 → 十一月的变化(归纳)

- 模型:5 个 → 7 个活跃 AgentX 模型;新增 Muse Spark Open、Qwen3.8 Flash Next,三个首发模型升级为 V4.1 Flash / K3.1 Pro / M3.1 Flash [1][7]。
- 硬件:Vera Rubin NVL72 在 9 月中出现(TRT-LLM 预发布版,9-14 文章)[11],10-09 有 vLLM 的 Rubin 结果(MiniMax M3)[16]。TPU(v7 Ironwood)9-07 进入 InferenceX,但只有 8k1k、Qwen3.5 FP8,没有 AgentX 结果 [10]。
- 引擎:看板 11 月的最佳点里 SGLang 占比明显上升(DeepSeek V4.1 Flash 在 B300、GB300 上的最佳开源点都是 SGLang;MiniMax M3.1 Flash 五列里四列是 SGLang)[7]。vLLM 新增 TorchTPU 后端(10-20)[14]。
- 新出现的第三方/厂商引擎名:TileRT(AMD 文章称在 GLM-5.3 上拿到 AgentX 单用户最高速度,二手)[20];InferenceX 9-28 的 GLM5.3 文章标题里也有 "AgentX TileRT" [3]。

---

## 2. 头条结果

### 2.1 发布时(口径:2026-08-21 前后的看板快照,按 TCO 归一)

**DeepSeek V4 Pro**
- 所有 DSV4 运行的请求分布:ISL p50 88k / p90 272k / p95 404k / p99 675k;OSL p50 413 / p90 2.2k / p95 3.7k / p99 8.6k [1][5]。
- 8-21 之前:MI355X + SGLang 在端到端口径的 perf/$ 上与 B200 + vLLM 打平;但同一时间 B300 vLLM 和 B200 SGLang 已经领先 MI355X [1][5]。
- 8-21 之后:Inferact 与 NVIDIA 在 vLLM 里的优化让 B200 的 perf/$ 超过 MI355X;SemiAnalysis 的说法是"接近的比赛,没有定论" [1][5]。
- MI355X 上开源 vLLM 落后于厂商引擎 ATOM;ATOM 在 e2e 延迟口径上胜过 B200 vLLM,但不胜 B300 和 B200 SGLang [1][5]。
- NVIDIA 最强配置是 GB300 Dynamo TRT-LLM 与 GB200 Dynamo vLLM,都靠 P/D 分离;GB300 还用 Wide-EP(DEP32)解码实例 [1]。
- B300 vLLM 与 B200 vLLM 按 TCO 归一后很接近,B300 靠多 50% 的 HBM 在高吞吐端多挤出一截:B300 DEP8 + 3TB DRAM、384 并发时 HBM 缓存命中 91% + DRAM 命中 1.36%,HBM KV 工作集约 4300 万 token;B200 196 并发时 HBM 命中 73%、DRAM 命中近 20%,工作集约 2200 万 token [1]。
- H200 SGLang FP8 低并发下能服务 DSV4,perf/$ 甚至可与 B200/MI355X SGLang 相比,但 HBM 不够,高吞吐端没法比 [1]。

**Kimi K3(2.8T)**
- 模型大到单台 B200 服务器装不下,要用 wide EP / wide TP / pipeline 并行。早期 vLLM 里投机解码(DSpark)不能和 pipeline 并行同时用,B200 表现很差,被 MI355X 压过 [1]。
- MI355X vLLM 短上下文单轮 day-0 可用,但长上下文多轮第一周 AITER/Triton 内核出问题,上游 vLLM 在真实负载上"完全不可用" [1]。
- e2e 延迟 40–60 秒这段曲线上,MI355X + ATOM 的 perf/$ 胜过 GB300 NVL72 + vLLM [1][4]。
- Hopper 基本玩不动:SM90 需要专门调的内核和 TP32/EP32 形状,没人做 [1][4]。

**MiniMax M3**
- SemiAnalysis 的原话是 NVIDIA 在 M3 上"摧毁所有对手";B300 TRT-LLM TP2 是最佳 [1]。
- 单节点 B200/B300 按 TCO 归一后**胜过**机柜级 GB200/GB300:原因是 Dynamo 路由器会成瓶颈(工作量随活跃前缀的数量和长度增长),且 wide EP / wide DCP / wide TP 没有调好的内核,机柜 TCO 又更高 [1]。
- GB200、并发 40:TP4/EP4/DP-attention 只有纯 TP4 的 0.60 倍吞吐,P90 TTFT 高 3 倍以上;并发 32 时缓存命中 28.8%(理论 96.0%),因为每个 DP rank 独占四分之一缓存池 [1]。
- NVIDIA 的帕累托点在并发 20 以上全部用了 KV offload;AMD 的帕累托点一个都没用 DRAM offload,原因是 ROCm 7.14 之前缺 hipMemcpyBatchAsync,GPU↔CPU 拷贝只能串行 [1]。
- vLLM 与 TRT-LLM 在吞吐-P90 交互速度上很接近,vLLM 在吞吐-P90 TTFT 上更好 [1]。

**Qwen3.5 397B**(256K 数据集)
- SGLang 对 SGLang:90 tok/s/user 时 NVIDIA 领先 20 倍以上,AMD "零竞争" [1]。
- B300 FP4 对 H100:perf/$ 12 倍 [1]。
- NVIDIA(尤其 TRT-LLM)被指为了交互速度牺牲 TTFT;SGLang 提交的 P90 TTFT 明显更低 [1]。

**GLM 5.3**
- 开源 SGLang 口径、150 tok/s/user P90:NVIDIA 成本效率最高到 5 倍;SemiAnalysis 称此点上"即使对手芯片白送(仍付机房、电力和运维),每 token 成本还是 NVIDIA 更低" [1]。
- 但换成 ATOM,AMD 在"E2E Normalized Interactivity"的部分区间 perf/$ 高于 GB300 NVL72 的 SGLang,甚至高于 TRT-LLM [1][6]。
- Forbes 转述的补充数字(二手的二手):B200 对 MI355X,108 tok/s/user 时 NVIDIA 领先约 57%,141 tok/s/user 时 247% [19]。
- 看板 compare 页(GLM 5.3,FP4,AgentX;页面注明为插值):136 tok/s/user 时 GB300 NVL72 26,746 tok/s/chip、$0.024/M、1262 万 tok/s/MW,MI355X 9,835、$0.042/M、471 万 tok/s/MW;183 档 17,506 vs 5,566($0.037 vs $0.075);230 档 9,679 vs 4,709($0.066 vs $0.088)。引擎和运行日期页面未写 [9]。

### 2.2 vLLM 自报的首发成绩(2026-09-08,数据取自 AgentX 看板)

- 最高 130K total tok/GPU·s(DeepSeek V4 Pro);最高交互速度 376 tok/s(MiniMax M3)[2]。
- 在 P90 交互速度 > 50 tok/s/user 约束下的最高吞吐配置 [2]:
  - DeepSeek V4 Pro 1.6T:12 块 GB300(P/D)、并发 256,83K TPGS,P90 58.3 tok/s。
  - MiniMax M3 428B:2 块 B300、并发 24,70K TPGS,P90 74.2 tok/s。
  - Kimi K3 2.8T:16 块 GB300、并发 48,11.8K TPGS,P90 62.7 tok/s。
- 成本对比(vLLM 自己算,对手是 Opus 5 API 定价,假设理论满缓存命中):DSV4 Pro GPU TCO $27.72/小时 对应等量 token 的 Opus 5 费用 $2,926/小时(106×);M3 $4.52 对 $384(85×);K3 $36.96 对 $538(14.6×)。原文声明比的是服务成本不是模型质量 [2]。

### 2.3 现在(2026-11 看板,"Hyperscaler cost",开源引擎,@100 tok/s/user;单位页面未标,按上下文应为 $/百万 token)[7]

| 模型 | B200 | MI355X | B300 | GB200 NVL72 | GB300 NVL72 |
|---|---|---|---|---|---|
| DeepSeek V4.1 Flash 552B | 无 @100 点 | $0.003 vLLM(11-09,比 10-07 便宜 42%) | $0.002 SGLang(10-29,比 10-02 便宜 30%) | 无 @100 点 | $0.002 SGLang(11-03,比 10-02 便宜 30%) |
| Kimi K3.1 Pro 2.8T | 到不了 @100 | $0.076 vLLM(11-08) | $0.100 SGLang(11-05) | $0.154 Dynamo SGLang(11-09) | $0.056 Dynamo vLLM(11-06,8P+16D) |
| MiniMax M3.1 Flash 428B | $0.005 SGLang | $0.005 SGLang | $0.005 SGLang | $0.009 Dynamo vLLM | $0.005 SGLang |
| GLM5.2/5.3 744B | $0.029 Dynamo SGLang(比 10-02 便宜 53%) | $0.028 SGLang(比 10-06 贵 7%) | $0.030 Dynamo SGLang | 无 @100 点 | $0.026 Dynamo SGLang |
| Qwen3.5 397B | $0.007 SGLang(比 9-16 便宜 44%) | $0.007 SGLang(9-29) | $0.007 Dynamo SGLang(比 9-21 便宜 72%) | $0.007 Dynamo SGLang | $0.012 Dynamo SGLang(9-09) |
| Qwen3.8 Flash Next 176B | 无 @100 点 | $0.008 SGLang | $0.006 SGLang | 无数据 | $0.004 SGLang(比 10-09 便宜 57%) |
| Muse Spark 1.3/Open 816B | $0.012 SGLang | $0.013 SGLang | $0.013 SGLang | 无 @100 点 | $0.010 Dynamo SGLang |

读法:到 11 月,在 @100 档开源引擎口径下,MI355X 在多数模型上已与 B200/B300 同一量级(M3.1 Flash、Qwen3.5、GLM5.x、Muse Spark 基本打平);Kimi K3.1 Pro 上 MI355X($0.076)比 B300($0.100)便宜,GB300 Dynamo vLLM($0.056)最低。这与 8 月"M3 上 AMD 很差、Qwen3.5 上差 20 倍"的画面差别很大。注意这是单一 SLO 档、单一成本模型的切片 [7]。

### 2.4 十月的 AMD 对 NVIDIA 单点(vLLM 博客,AMD 与合作方撰写)

- Kimi K3、vLLM、8 块 MI355X 对 HGX B300,2026-10-09 看板快照,P90 交互速度 50 tok/s/user:MI355X 约 7,567 total tok/s/GPU,B300 6,083,多 24.4% [15]。
- 按 SemiAnalysis 建模 TCO($1.50 对 $2.26 每 GPU·小时):1816 万 对 969 万 total token/美元,多 87.4% [15]。
- 两者硬件规格:MI355X 稠密矩阵峰值 10.1 PFLOP/s(MXFP4),B300 13.5 PFLOP/s(FP4);都是 288 GB HBM3E、8 TB/s 峰值带宽 [15]。

### 2.5 机柜级对单节点

- 首发时 M3 上单节点 B200/B300 的 TCO 归一吞吐胜过 GB200/GB300(见 2.1)[1]。
- DSV4 Pro 上机柜级最强:GB300 的 3×DEP8 + 1×DEP16 与 GB200 的 2×DEP8 + 1×DEP12 等 P/D 配置;GB300 并发更高、子智能体流量更多、冷 prefill 更多,所以 TTFT 上优势比 TPS 上小 [1]。
- Kimi K3 上 NVL72 级系统 wide EP + DP(DEP16)在每 rank batch > 3 后比 DCP8 伸缩更好 [2]。
- 10-23 vLLM:DeepSeek-V4.1-Flash 在 GB300 NVL72 上 P/D + Wide-EP,954K tok/s/GPU @ P90 196 tok/s/user,是其聚合式单节点结果的 7.9 倍 [14]。
- AMD 的分布式推理(DI):1×DEP8 + 1×DEP8 分离配置只在高吞吐端略有收益,低延迟端反而更差,且 P90 TTFT 暴涨(并发 64 以上启用了 SGLang 的 `--enable-prefill-delayer`,chunked prefill 从 8,192 调到 65,536)[1]。

### 2.6 Vera Rubin 的数字与审核状态

- **NVIDIA 自测(待 SemiAnalysis 审核)**:DeepSeek V4 Pro、AgentX,Vera Rubin NVL72 对 GB300 NVL72 每 MW 吞吐在 110 tok/s/user 时 2×、130 时 10×、160 时 30×;开发者博客明确写这些结果"由 NVIDIA 测得,待 SemiAnalysis 审核" [13]。
- NVIDIA 企业博客(8-24 首发,9-15 更新)标题段写"30× 每 MW 吞吐、45× 更低 token 成本";搜索到的另一版本/转载写 35×。两个数字并存,应是更新前后的差异(pulse2 转载用 35×)[12][20]。该博客注明结果尚未反映 Vera CPU 在工具调用上的性能 [12]。
- **SemiAnalysis 自己的 Rubin 文章(9-14,标题含"67x better Performance per Dollar")**:Rubin 用 TRT-LLM(预发布版)、NVFP4,量产 SKU 2300W TDP,每 compute tray 1.5TB LPDDR5X [11]。
  - 170 TPS 时对 GB300 Dynamo TRT-LLM 约 67× token/TCO 美元;60–100 TPS 时 1.4–3×;80 TPS P90 时对单节点 B300 10×、对 H200 18×(120 TPS 时 39×)[11]。
  - 最高 P90 交互速度:Rubin 276.24 TPS,GB300 Dynamo TRT-LLM 171.53 TPS;但 GB300 用 SGLang 可达到与 Rubin 相近的交互速度 [11]。
  - 每 MW(DSV4 Pro,total tok/s per utility MW,100 TPS):Rubin 约 5940 万,GB300 Dynamo SGLang 2850 万,GB300 Dynamo TRT-LLM 2110 万(对较强者 2.09×);B200 SGLang 695 万,B300 vLLM 556 万,H200 Dynamo SGLang 226 万,MI355X SGLang 201 万 [11]。
  - **关键注意**:170 TPS 时对 GB300 TRT-LLM 是 62.9×,对 GB300 SGLang 只有 5.56×——头条倍数很大程度上取决于拿哪个引擎当基线 [11]。
  - 租赁口径:Rubin > $8.50/小时/芯片,GB300 约 $5;80 TPS P90 时每租赁美元多 62% token [11]。
  - 作者感谢 NVIDIA 人员"帮助验证"agentic 结果,属协作验证而非完全独立 [11]。
- **vLLM 的 Rubin 结果(10-09)**:MiniMax M3、AgentX,相同交互速度下每 GPU 吞吐最高为 GB200 NVL72 的 7.84×;150 TPS 约束下 5.18×。标注为"非常早期" [16]。

### 2.7 TPU

- 9-07 文章"TPU Inference Externalization Full Steam Ahead",标"InferenceX Official Preview" [10]。
- 只有 TPUv7 Ironwood;模型 Qwen3.5 397B FP8;负载 **8k1k,不是 AgentX**;软件是新的 TorchTPU vLLM 后端(当时私测,vLLM 于 10-20 公开发布 vllm-torchtpu)[10][14]。
- 100 tok/s/user:Ironwood $0.181/M total token,B200 $0.222,B300 $0.276(外部 TCO)[10]。
- 20 tok/s/user:每美元 token 比 B200 多 50.4%、比 B300 多 96.0%——"up to 50%"标题指的是对 B200 这一点 [10]。
- 内部 TCO($1.03/芯片·小时)、并发 256:perf/$ 领先 B200 76.7%、B300 130.2%,但平均 TTFT 5.41 s(B200 3.75 s,B300 2.40 s)[10]。
- GB300 NVL72 分离式对 TPUv7 聚合式(内部 TCO):GB300 在中段延迟有约 30% 的 perf/$ 优势 [10]。
- 局限:Ironwood 无原生 FP4,故只比 FP8;TPU 上分离式未优化;TPUv8i 未测;AgentX 的 TPU 结果"今年稍后" [10]。

### 2.8 "CUDA 护城河是否守住"的结论与限定

- SemiAnalysis 的总结口径很温和:"NVIDIA 在很多前沿模型上表现很好,AMD 在一些前沿模型的特定比较里也不错" [1]。
- 文章把护城河归到分布式系统软件而非硅片:DCP/PCP(上下文并行)被点名为"CUDA 护城河的一部分",因为 vLLM 支持矩阵里 AMD 后端全部未支持 [1][18]。
- 限定条件 [1][4][5][6]:
  1. AMD 的若干胜点只在厂商引擎 ATOM 下成立;SemiAnalysis 称中西方多数实验室不愿在生产中用 ATOM(缺很多功能),只有阿里一个小的广告业务单元在用,Qwen 主团队不用。
  2. 结论对软件版本极敏感:DSV4 Pro 的排名在一周内翻转(8-21)。
  3. 必须同时看 TPS 和 TTFT;"E2E Normalized Interactivity"是实验性指标,重罚高 TTFT,不能完整反映 P/D 分离的取舍。
  4. 结果按模型、延迟区间、引擎而异,没有统一倍数(Forbes 也这样提醒,二手 [19])。
- 到 11 月,开源引擎 @100 档的成本表上 AMD 与 NVIDIA 单节点已大体接近(见 2.3),"护城河"更多体现在机柜级 P/D + Wide-EP 和最高交互速度区间 [7][14](这一句是我的归纳)。

---

## 3. 优化时间线与上游 PR

### 3.1 时间线

| 日期 | 事件 | 来源 |
|---|---|---|
| 2026-08-19 | InferenceX 博客《A Brief Overview of Agentic Workloads》 | [3] |
| 2026-08-21 | 看板快照分界:此前 MI355X SGLang ≈ B200 vLLM(DSV4 Pro perf/$);此后 B200 vLLM 反超 | [1][5] |
| 2026-08-24 | AgentX 1.0 / InferenceX v3 发布;NVIDIA 同日发两篇博客;Forbes 报道 | [1][12][13][19] |
| 2026-09-07 | TPU Ironwood 进入 InferenceX(8k1k) | [10] |
| 2026-09-08 | vLLM × AgentX 总结文;GLM 5.3 Hybrid HiSparse offload | [2][14] |
| 2026-09-10 | vLLM:MiniMax M3 on MI355X;Tiered KV offloading | [14] |
| 2026-09-13 | vLLM:Kimi K3 "2.8× 吞吐之路" | [14] |
| 2026-09-14 | SemiAnalysis:Rubin NVL72 agentic(TRT-LLM) | [11] |
| 2026-09-15 | NVIDIA 企业博客更新数字;vLLM Kimi K3 DSpark | [12][14] |
| 2026-09-21 | vLLM:Qwen3.8-2.4T 在 GB300 NVL72 上 P/D,吞吐 5K、交互 180 | [14] |
| 2026-09-28 | InferenceX:GLM5.3 稀疏注意力与 HBM 占用 | [3] |
| 2026-10-07 | vLLM:DeepSeek-V4.1-Flash 发布三周内 AgentX 吞吐 5×、低并发 1.9× | [14] |
| 2026-10-09 | vLLM on Vera Rubin(M3,7.8× 对 GB200) | [16] |
| 2026-10-12 / 10-14 | vLLM:双向 KV 传输(TTFT 最多降 3.3×);NIXL push connector | [14] |
| 2026-10-16 | vLLM:Kimi K3 megakernel(GB300);Kimi K3 on MI355X(对 B300 +24.4%) | [15][17] |
| 2026-10-20 | vLLM TorchTPU 后端发布 | [14] |
| 2026-10-23 | vLLM:V4.1-Flash P/D + Wide-EP 954K tok/s/GPU(7.9×);GLM-5.3 在 16×H200 上;M3.1-Flash day-0 | [14] |
| 2026-11-04 | vLLM:Kimi K3.1 Pro day-0 | [14] |

SemiAnalysis 在 8-24 文中说"三到四周后发布 AgentX 更新文章";是否已发、叫什么名字,我没有确认(9-14 的 Rubin 文可能部分承担了这个角色)[1][11]。

### 3.2 软件带来的幅度(每条都带口径)

- DeepSeek-V4.1-Flash、vLLM:发布三周内 AgentX 吞吐 5×、低并发速度 1.9×,手段是 SWA bounded replay、CUDA graphs、DeepSeek 新内核、vLLM 内核融合(10-07 文摘要)[14]。
- DeepSeek-V4.1-Flash、vLLM、GB300 NVL72:P/D + Wide-EP 达到 954K tok/s/GPU @ P90 196 tok/s/user,为聚合式单节点结果的 7.9×;标题称"追平 DeepInfer"(10-23 文摘要)[14]。
- 看板自带的 30–60 天对比(@100 档成本)[7]:V4.1 Flash 在 MI355X 上比 10-07 便宜 42%,在 B300/GB300 上比 10-02 便宜 30%;GLM5.x 在 B200 上比 10-02 便宜 53%;Qwen3.5 在 B300 上比 9-21 便宜 72%、在 B200 上比 9-16 便宜 44%;Qwen3.8 Flash Next 在 GB300 上比 10-09 便宜 57%。也有倒退的:GLM5.x 在 MI355X 上比 10-06 贵 7%。
- Kimi K3、vLLM、GB300(megakernel,10-16):每层从 130 µs / 23 个 kernel 降到 69 µs / 13 个;DSpark batch 1、TP16 时 842 对 488 tok/s(1.73×);AgentX 并发 1 的 P90 交互速度 403.2 对 223.7 tok/s/user(1.8×,但前者 16 GPU、后者是较早的 8 GPU 结果,GPU 数不同)[17]。
- vLLM `--long-prefill-token-threshold` 512:DSV4 Pro、B300,TPGS 最多 +93%,P90 交互速度约 2.3× [2]。
- SGLang 解码间隔(prefill 后强制插入 decode 轮):DSV4 Pro AgentX 输出吞吐 +141%,p99 ITL −97.3%,代价是中位 TTFT 从 36.5 s 升到 59 s [1]。
- SGLang 把上下文长度改成运行时标量(免去逐请求编译):并发 384 输出吞吐 +26.75%,平均 TTFT −36.25% [1]。
- SGLang 丢弃 decode 侧无用的 prompt 传输:GB300 AgentX 每用户输出吞吐 +18.0%,每 GPU decode 吞吐 +12.7% [1]。
- ATOM 稀疏检查点保留(DSV4 滑窗):并发 48 实际前缀命中 5.6% → 96.45% [1]。
- Dynamo:批量 KV 匹配,并发 512 中位输出吞吐 +22.2%;request lease 使 AgentX 回放时间 −23.7%(vLLM 后端)/ −22.0%(SGLang)[1]。
- TRT-LLM 边界感知增量分词:Qwen3.5 AgentX trace 上 1,087 次转换全部与全量分词一致,平均处理时间 185.1 ms → 11.3 ms [1]。
- MiniMax M3、vLLM、MI355X(9-10 文,InferenceX 公开结果;**注意前四条是 8K/1K 固定负载的输出 token 口径,不是 AgentX**)[21]:
  - MXFP8、TP4/EP1、并发 32:109.1 → 342.4 输出 tok/s/GPU(3.14×),中位 TTFT 1.46 → 0.67 s,平均 TPOT 69.1 → 22.1 ms。
  - 同配置并发 128:297.8 → 623.7(2.09×)。
  - MXFP4 并发 128:212.1 → 716.8;改 TP2/EP1 后 943.5(初始值的 4.45×,但拓扑变了,属部署密度提升)。
  - 加 EAGLE3:TP4/EP1 并发 128 达 682.4 输出 tok/s/GPU。
  - 加 P/D 分离并重调拓扑:并发 512 达 6,370.5 **total** tok/s/GPU,中位 TTFT 1.32 s。
  - 文中一个可复用的结论:没有"永远最快"的后端,原生 MXFP8 linear 在低中并发更快,模拟路径只在长输入高并发时赢,最终配方只在 8K 输入且并发 ≥ 64 时启用后者。
- push 式 NIXL 连接器(10-14,AWS 贡献,最初在 Trainium 集群上开发):1P1D、EC2 P5en + EFA、Qwen3-32B,TTFT 最多好 3×、TPOT 最多好 30%。**这组数字不是在 AgentX 上测的** [22]。
- 更早的非 AgentX 参照:AMD ROCm 博客(7-21,数据截至 6-26,8K/1K)称 MiniMax M3 发布后几天内 MI355X + ATOM FP4 领先,随后 NVIDIA 很快追上;17–135 tok/s/user 区间 MI355X ATOM EAGLE3 FP4 为 3.6k–8.5k tok/s/GPU,B200 vLLM 为 2.5k–7.7k [23]。对照 8 月 AgentX 上"NVIDIA 在 M3 上全面领先"的结论,可见同一模型在短上下文单轮与长上下文多轮下排名相反。
- 更早的非 AgentX 参照:MI355X DeepSeek-V4-Pro SGLang 26 天内 8k1k 每 GPU 吞吐 110.5×(20 → 2,256 tok/s/GPU,31 个性能 PR)[3]。

### 3.3 "70+" 与 "50+" 的出处

- **两个数字出自同一篇文章** [1]:
  - 开头摘要段:"Over 70+ upstream PRs",覆盖 vLLM、SGLang、TensorRT-LLM、ATOM、AITER、Dynamo、LMCache、Mooncake。
  - 正文"优化"一节的起始句:"industry impact of 50+ upstream PRs"。
- 合理解释是正文那一节写得早、摘要后来更新而没同步;文章本身没解释差异(这是我的推断)。
- 后续 InferenceX 站内的切片文章(Kimi K3、DSV4 Pro、GLM 5.3 三篇)结尾统一写"70+ upstream PRs" [4][5][6];aiweekly 转述也用 70+ [18]。
- InferenceX 有专页 /agentx/optimizations 按项目列出:vLLM、SGLang、TensorRT-LLM、AMD ATOM、ROCm AITER、NVIDIA Dynamo、LMCache、Mooncake、MoRI UMBP 九个子页;页面说明"open"的改动是已提未合并 [8]。该页没给总数。
- PR 的性质:相当一部分是**正确性**而非性能(M3 的 NIXL 握手断言导致 KV 没传、gsm8k 得 0 分;FP8 e4m3fn 与 e4m3fnuz 编码混用;AITER 32 位偏移在大缓存池回绕;SGLang 分离式 127,500 token 共享前缀测试从 2/128 正确修到 128/128)[1][8]。另有一批是 AMD 侧的可安装性(Mooncake ROCm wheel、LMCache gfx942/gfx950 预编译 wheel、HIP dmabuf 注册)[1]。文中也注明对 NIXL 和 Mooncake 的直接搜索没找到更多带 AgentX 标记的运行时 PR [1][8]。

---

## 4. 在 AgentX 上起作用的技术

### 4.1 各技术与证据

- **前缀缓存 + 混合注意力下的保留策略**:理论前缀命中率 > 96%,能否兑现取决于滑窗/循环状态的检查点是否还在。vLLM 用"间隔保留 + Marconi 式选择性保留";SGLang 主动释放离窗页面、限制 compute lock;ATOM 做稀疏检查点保留 [1][2]。
- **KV offload 到主机 DRAM**:规则允许但不强制;DRAM 须按所用 GPU 比例缩放,非标准 DRAM 系统上限 3 TB;NVMe/SSD offload 推迟到后续版本("fast follow")[1][8]。写穿式缓存,DRAM 容量是 HBM KV 容量 1.5–3 倍时最有效 [1]。vLLM SimpleCPU 扩展到 DSV4 混合注意力后:输出吞吐 +81.7%、平均 e2e 延迟 −46.6%(相对前缀放不下时重算)[1]。ATOM + LMCache:32,000 token 前缀从 CPU 重载约 0.32 s,重算约 2.5 s [1]。
- **会话亲和(sticky)路由**:vLLM 的实验里,按队列深度、运行中 token、KV 利用率做均衡都不如简单的会话粘性路由——因为多数会话轮间间隔短(中位 3.84 s),下一轮到达时前缀还在原 GPU 上 [1][2]。SGLang 加了 DP cache affinity 和"缓存均衡"信号;Dynamo 的 agentic 路由预设在 8×H200 上输出吞吐 +8.26%、p95 TTFT −43.1% [1]。
- **P/D 分离 + rate matching**:NVIDIA 机柜级最强点都靠它;vLLM 给出两阶段方法(先分别压测 prefill-only / decode-only 的饱和点,再据此定 P/D 比并扫并发)[1][2]。push 式 NIXL 连接器(prefill 直接经 RDMA 写入 decode 内存)和双向 KV 传输(TTFT 最多降 3.3×)是 10 月的后续 [14]。
- **Wide-EP / DEP**:DSV4 的默认并行方式;GB300 用 DEP32 解码实例;DEP 下各 rank 因 MoE all-to-all 锁步,vLLM 用 `--prefill-schedule-interval` 把 prefill 对齐到同一步 [1][2]。
- **上下文并行 DCP / PCP**:Kimi K3 上 DCP8 优于 TP8(142K 会话的 MLA 负载从约 1.96 GB/GPU 降到约 245 MB)[2][15];DSV4 上 PCP8 对 32K prompt 的 prefill 比 TP8 快 2.65×,DCP 只能追平 DEP [2]。ATOM 上 PCP 平均 TTFT −35~43% [1]。
- **投机解码(MTP / EAGLE3 / DSpark)**:因回放内容是合成 token,接受率会失真;AgentX 的做法是让引擎强制接受长度,取值来自 SPEED-Bench 智能体编码集上实测的平均 AL(仓库 golden_al_distribution 目录)[1]。
- **调度**:chunked prefill 封顶解决队头阻塞(vLLM);prefill 后强制 decode 轮(SGLang)[1][2]。
- **内核融合 / megakernel**:Kimi K3 的 DCP 路径绕开 NCCL、用对称内存直写,每层延迟 −13% [2];GB300 上 CuTe DSL 统一解码路径 [17];NVIDIA 称 MegaMoE 把计算与 GPU 间通信融合为一次执行 [12]。
- **CUDA graphs(用于 prefill)**:列在 V4.1-Flash 5× 的手段里 [14];TRT-LLM 为 M3 做了 context graph producers,每用户输出吞吐 +12.58% [1]。
- **前端/请求面**:增量分词(TRT-LLM)、MessagePack 负载(Dynamo,吞吐 +8.1%)、去掉逐 token 的指标与日志锁 [1]。
- **不适用的**:pipeline 并行不适合"热"的智能体轮次(新增 token 太少,填不满流水线)[2]。

### 4.2 引擎 × 技术(根据读到的材料,空白表示没查到,不代表不支持)

| 技术 | vLLM | SGLang | TensorRT-LLM | ATOM(AMD) | Dynamo(编排层) |
|---|---|---|---|---|---|
| 混合注意力前缀保留 | 有(间隔 + Marconi)[2] | 有(滑窗页管理)[1] | | 有(稀疏检查点)[1] | — |
| DRAM KV offload | SimpleCPU、Mooncake Store、LMCache [1][2] | HiCache;MoRI UMBP [1][8] | | LMCache、multi-connector [1] | KVBM [1] |
| 会话/缓存感知路由 | vLLM router、llm-d [1] | model gateway,DP cache affinity [1] | 经 Dynamo [1] | ATOMesh(SGLang router 的精简分叉)[1] | KV-aware router [1][13] |
| P/D 分离 | NIXL(pull/push)[1][14] | 有,staging buffer 支持 radix cache [1] | 有,NIXL bounce path、流水化 KV 传输(open)[1] | 有,多节点 P/D 路由 [1] | 负责拆分与扩缩 [13] |
| Wide-EP | 有 [2] | 有 [1] | 有(WideEP)[13] | DEP 内核待优化 [1] | — |
| DCP / PCP | 有(NVIDIA 后端;AMD 后端支持矩阵里未支持,10 月起 ROCm AITER DCP 路径在补)[1][15] | | | 有(PCP、DCP + 前缀缓存 + MTP)[1] | — |
| 投机解码 | EAGLE3、DSpark、MTP [1][14] | MTP/EAGLE,分离式下传 draft 窗口 [1] | 有 [1] | EAGLE 分离式传 draft KV [1] | — |
| 增量分词 | | | 有 [1] | | 压缩 prompt 路径 [1] |

---

## 5. 厂商反应

### 5.1 NVIDIA

- 发布当天(8-24)同步两篇:企业博客《Up to 30x More Work Per Watt…》和开发者博客《NVIDIA Vera Rubin and Blackwell Set a New Standard for Agentic AI Performance per Watt》[12][13]。
- GB300 NVL72 对 H200 NVL8(AgentX):DeepSeek V4 Pro 每 MW 吞吐最高 15×、每百万 token 成本最低到 1/10;Kimi K3 2.8T 每 MW 约 80×,并把交互速度前沿推到约 215 tok/s/user [13]。企业博客里把 15× 的对手表述为"Hopper 架构" [12]。
- 对照:旧的 8k1k DeepSeek-R1 上 GB300 对 H200 是每 MW 最高 40× [13]。
- Rubin 预览:见 2.6;NVIDIA 另称 DSX MaxLPS 能在同一 MW 预算内多配最多 40% GPU [12]。
- NVIDIA 把 AgentX 的首要指标定为"tokens per megawatt",并列了四个体验指标(E2E 归一化交互速度、标准交互速度、E2E 延迟、TTFT)[13]。
- SemiAnalysis 反过来批评 NVIDIA:TRT-LLM 提交为交互速度牺牲 TTFT;Rubin 文里称 NVIDIA 之前的官方倍数"sandbagged"(GTC 说 3×,实测到 7×)——后者是 SemiAnalysis 的评语 [1][11]。

### 5.2 AMD

- ATOM 定位:AMD 的厂商引擎(相当于 TRT-LLM);路由器 ATOMesh;打包发行版叫 AMD Infera(与 Dynamo、llm-d 并列)[1][8]。SemiAnalysis 说 ATOM 原本只为单轮负载设计,为支持智能体负载在以 AgentX 为目标做重构,距 vLLM/SGLang 还有不小差距 [1]。
- SemiAnalysis 对 AMD 的反复要求:把 ATOM 的优化上游到 vLLM / SGLang [1][4][6]。
- AMD 在 vLLM 仓库公开了 DSV4 优化清单(issue #52911)[5]。
- AMD 借 vLLM 博客发声:9-10《MiniMax M3 on MI355X》、10-16《Kimi K3 on MI355X:每 GPU 吞吐追平并超过 HGX B300》[14][15]。
- AMD 开发者文章(二手,仅见搜索摘要,原文两次抓取失败):《469 tok/s/user on Million-Token Agent Sessions: TileRT Takes the Top Spot on AgentX with MI355X》称 8×MI355X 上 TileRT 跑 GLM-5.3(原始 FP8 权重、BF16 KV cache)单用户 469 tok/s,比第二名 GB300 NVL72(FP4)快 100 tok/s 以上;搜索摘要给的文章日期是 2026-09-24。Digg 上的转述写成"470 tok/s,比 GB300 TRT-LLM FP4 快 40% 以上"。这是单用户速度指标,不是成本效率指标;同一模型在 SGLang、150 tok/s/user 的成本口径上是 NVIDIA 领先最多 5×。另一篇 MoRI UMBP 文章称在 DSV4 Pro 公开榜上 token/美元领先 [20]。
- InferenceX 站内另有一篇切片文《GLM 5.3 on AgentX: NVIDIA Is Up to 5x Cheaper per Token at 150 tok/s/user》(只见标题)[20]。
- SemiAnalysis 对 AMD 管理层的批评很直接:M3 上 AMD 软件差是因为"工程领导层只激励调短上下文单轮负载" [1]。没查到 AMD 对这句话的公开回应。

### 5.3 争议与更正

- NVIDIA 企业博客的"token 成本降低"倍数存在 35× 与 45× 两个版本(8-24 原版与 9-15 更新版之间),博客顶部有更新说明 [12][20]。
- NVIDIA 的 Rubin 数字在其开发者博客里自注"待 SemiAnalysis 审核" [13];SemiAnalysis 9-14 自己发的 Rubin 数字与 NVIDIA 的不同口径(67× perf/$ @170 TPS;每 MW 对最强 GB300 引擎 2.09× @100 TPS)[11]。
- 第三方博客 Spheron 的 AgentX 文章,搜索摘要提示其具体数字与 SemiAnalysis 页面冲突,引用需谨慎(二手)[20]。
- 没查到 AMD 或 NVIDIA 对 AgentX 方法论提出正式异议的公开文字。

### 5.4 模型厂商

- SemiAnalysis 致谢名单把 MiniMax、Moonshot Kimi、Alibaba Qwen、Zhipu GLM 列为 InferenceX 开源计划的支持者(与 Meta、Microsoft、Oracle、OpenAI 并列);DeepSeek 不在这句名单里 [1]。
- vLLM 的 V4.1-Flash 文章提到采用了"DeepSeek 的新内核" [14],说明 DeepSeek 在向开源引擎贡献内核,但我没找到 DeepSeek 官方直接提 AgentX 的文字。
- Mooncake(Moonshot 生产流量在用的传输引擎)维护者在 AgentX 致谢名单里 [1]。
- 没查到 DeepSeek、Moonshot、MiniMax、智谱、阿里各自官方渠道如何称呼或引用 AgentX。

---

## 6. 采用情况

- SemiAnalysis 称其公开 REST API 已被"多家一线 AI 实验室的容量规划团队"使用(未点名)[1]。
- 引擎社区:vLLM 把 AgentX 当作"独立验证",9 月以后几乎每篇模型优化文都用 AgentX 帕累托图报成绩,并公开 harness 仓库 SemiAnalysisAI/agentx-harness [2][14]。致谢里出现 Inferact、Red Hat/llm-d、RadixArk/SGLang、LMCache/TensorMesh、Weka、Mooncake、AMD、NVIDIA、Anthropic、GitHub [1]。
- AIPerf(NVIDIA 的负载生成器)被 tenstorrent、AWS、AMD 等采用;SemiAnalysis 维护自己的分叉以保持厂商中立 [1]。
- Anthropic 为数据集采集修过 Claude Code 的两个 issue(#49207、#66761)[1]。
- 媒体:Forbes(Janakiram MSV,8-24)、AI Weekly 等报道 [18][19]。
- **MLPerf 有可比的智能体负载**(二手,来自新闻转述与搜索摘要)[24]:
  - MLCommons 于 2026-07-08 发布 MLPerf Inference 的 Agentic Inference 基准,放在 MLPerf Endpoints 框架下,比 AgentX 1.0 早约七周。
  - 数据中心负载:490 条智能体编码轨迹 + 500 条企业工作流轨迹(Workato 贡献,与 AMD、Intel、NVIDIA 合作),合计 3 万多个客户端轮次;模型用 Kimi K2.6 和 Qwen3.6-35B-A3B。
  - 度量:系统总吞吐与"每用户进度"之间的帕累托权衡,并有分层的准确性检查(防止靠缩短回答取胜)。
  - 边缘版:Qwen3.6-27B + BFCL v4 + 录制的编码回放,单流;MLPerf Inference v6.1(2026-09-16 公布)里加入了边缘智能体测试,并称数据中心版"即将发布"。
  - 与 AgentX 的差别(我的对比):MLPerf 有准确性门槛和企业工作流轨迹,模型较小;AgentX 是 1M 上下文、前沿大模型、持续更新的看板、按 TCO 和每 MW 归一,但回放内容是合成 token、不检查逐请求输出质量(靠另外的精度验证)。
- vLLM 的 Rubin 文章还引用了 MLPerf Inference v6.1 的 VLM 基准(Qwen3-VL-235B-A22B,Rubin 对 GB300 最高 3.7×)[16]。
- Artificial Analysis 是否有可比的智能体推理(系统侧)负载:两次搜索都没找到。
- 点名引用 AgentX 的云厂商 / neocloud(CoreWeave、Nebius、Together、Fireworks、Crusoe 等):搜索没找到。

---

## 来源

1. SemiAnalysis,《AgentX - InferenceXv3: Does CUDA Moat Hold up in Agentic Inferencing?》,2026-08-24。https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat (本次读了 0–16 万字符全文)
2. vLLM 博客,《vLLM x AgentX: Optimizing for Real-World Agentic Serving》,2026-09-08。https://vllm.ai/blog/2026-09-08-vllm-agentx
3. InferenceX 博客索引。https://inferencex.semianalysis.com/blog
4. InferenceX,《Kimi K3 on AgentX: MI355X ATOM Beats GB300 NVL72 on Part of the Curve》。https://inferencex.semianalysis.com/blog/kimi-k3-agentx-mi355x-atom-vs-gb300-nvl72
5. InferenceX,《DeepSeek V4 Pro on AgentX: MI355X vs B200, and the August 21 Flip》。https://inferencex.semianalysis.com/blog/deepseek-v4-pro-agentx-mi355x-vs-b200-august
6. InferenceX,《GLM 5.3 on AgentX: MI355X ATOM Beats GB300 NVL72 on Part of the Curve》。https://inferencex.semianalysis.com/blog/glm-5-3-agentx-mi355x-atom-vs-gb300-nvl72
7. InferenceX 看板总览《Agentic Inference Costs》,2026-11-10 读取。https://inferencex.semianalysis.com/overview
8. InferenceX,《AgentX Industry Impact: Optimizations for Agentic Workloads》。https://inferencex.semianalysis.com/agentx/optimizations
9. InferenceX 对比页 GLM 5.3 GB300 vs MI355X。https://inferencex.semianalysis.com/compare/glm-5-3-gb300-vs-mi355x
10. InferenceX / SemiAnalysis,《TPU Inference Externalization Full Steam Ahead》,2026-09-07。https://inferencex.semianalysis.com/blog/tpu-inferencex-full-steam (经摘要工具读取)
11. InferenceX / SemiAnalysis,《Rubin NVL72 Agentic Inference: 67x better Performance per Dollar》,2026-09-14。https://inferencex.semianalysis.com/blog/vera-rubin-nvl72-agentic-inference (经摘要工具读取)
12. NVIDIA 企业博客,《Up to 30x More Work Per Watt: NVIDIA Vera Rubin NVL72 Sets a New Efficiency Standard for AI Agents》,2026-08-24,2026-09-15 更新。https://blogs.nvidia.com/blog/vera-rubin-nvl72-efficiency-ai-agents/
13. NVIDIA 开发者博客,《NVIDIA Vera Rubin and Blackwell Set a New Standard for Agentic AI Performance per Watt》。https://developer.nvidia.com/blog/nvidia-vera-rubin-and-blackwell-set-a-new-standard-for-agentic-ai-performance-per-watt/
14. vLLM 博客索引(各文标题、日期与摘要)。https://vllm.ai/blog
15. vLLM 博客,《Scaling Kimi K3 on AMD Instinct MI355X: Matching NVIDIA HGX B300 Throughput per GPU on AgentX》,2026-10-16。https://vllm.ai/blog/2026-10-16-kimi-k3-mi355x
16. vLLM 博客,《vLLM Support for NVIDIA Vera Rubin NVL72: 7.8x Throughput over GB200 NVL72》,2026-10-09。https://vllm.ai/blog/2026-10-09-vera-rubin-preview
17. vLLM 博客,《Hyper Optimizing Kimi K3 in vLLM: Moving Fast and Responsibly》,2026-10-16。https://vllm.ai/blog/2026-10-16-kimi-k3-megakernel (经摘要工具读取)
18. AI Weekly,《SemiAnalysis AgentX benchmark says CUDA moat holds in agents》(二手)。https://aiweekly.co/alerts/semianalysis-agentx-benchmark-says-cuda-moat-holds-in-agents
19. aiunderstanding.org 对 Forbes 报道的转述(二手的二手);Forbes 原文:https://www.forbes.com/sites/janakirammsv/2026/08/24/nvidia-amd-ai-agent-benchmark/ (原文未读)。https://aiunderstanding.org/news/forbes-reports-nvidia-led-amd-by-up-to-5x-on-new-ai-agent-benchmark
20. 仅见搜索摘要、原文未读到的条目:AMD《TileRT Takes the Top Spot on AgentX…》https://www.amd.com/en/developer/resources/technical-articles/2026/tilert-agentx-on-amd-instinct-gpus.html ;AMD《MoRI UMBP…》https://www.amd.com/en/developer/resources/technical-articles/2026/mori-umbp-empowers-amd-instinct-gpus.html ;pulse2 转载 NVIDIA 博客 https://pulse2.com/nvidia-vera-rubin-nvl72-delivers-up-to-30x-more-agentic-ai-throughput-per-megawatt-and-35x-lower-token-cost/ ;Spheron 博客 https://www.spheron.network/blog/agentx-benchmark-amd-mi355x-vs-nvidia-for-agentic-coding-2026/
21. vLLM 博客,《Following the Bottleneck: Optimizing MiniMax M3 on AMD Instinct MI355X》,2026-09-10(读了前 7000 字符)。https://vllm.ai/blog/2026-09-10-minimax-m3-mi355x
22. vLLM 博客,《Push-Based KV Transfer to Improve TTFT and TPOT for Disaggregated Inference in vLLM》,2026-10-14(读了前 6000 字符)。https://vllm.ai/blog/2026-10-14-nixl-push-connector
23. AMD ROCm 博客,《Efficient MiniMax-M3 Inference on AMD Instinct GPUs with ATOM and ATOMesh》,2026-07-21。https://rocm.blogs.amd.com/artificial-intelligence/minmax-m3-atomesh/README.html
24. Let's Data Science,《MLCommons Adds Agentic Inference Benchmark To MLPerf》(二手);HPCwire 关于 MLPerf Inference v6.1 的通稿(仅见搜索摘要)https://www.hpcwire.com/off-the-wire/mlcommons-sets-participation-record-with-new-mlperf-inference-v6-1-benchmark-results/ 。https://letsdatascience.com/news/mlcommons-adds-agentic-inference-benchmark-to-mlperf-7d651bea

---

## 没查到 / 不确定

- **站点真实性提示**:用摘要工具读 inferencex.semianalysis.com 的若干页面时,工具报告页面上有一条横幅"Unofficial deployment: This copy of InferenceX is not affiliated with or endorsed by SemiAnalysis. Its data may be modified or out of date."。域名是 semianalysis.com 的子域,横幅可能是开源前端的默认提示,也可能另有原因;我没能判断。引用看板数字时建议注明读取日期并留意这一点 [7][9][10][11]。
- SemiAnalysis 预告的"AgentX 更新文章"是否已单独发布、标题与日期。
- "70+"与"50+"的差异没有官方解释;PR 的精确总数没有页面给出。
- /overview 成本表的单位页面未标(推定为 $/百万 token),也没给各 SKU 的 $/GPU·小时(仅从 vLLM 文得到 MI355X $1.50、B300 $2.26 两个值)。
- TPU 的 AgentX 结果:截至读到的材料没有;TPUv8i 未测。
- MI455X UALoE72 是否已有结果:没查到。
- MI300X / MI325X / RTX Pro Server 在 AgentX 上的具体数字:没查到(只知道在首发矩阵里)。
- AMD 两篇开发者文章(TileRT、MoRI UMBP)原文抓取失败,只有搜索摘要;TileRT 是谁开发的、是否在官方看板上、469 tok/s/user 的完整口径都未核实。
- AMD 对 SemiAnalysis 批评(管理层激励、ATOM 无人生产使用)的公开回应:没查到。
- DeepSeek、Moonshot、MiniMax、智谱、阿里官方对 AgentX 的提法:没查到一手材料。
- 明确把 AgentX 写进采购或对外宣传的云厂商 / neocloud 名单:没查到(只有 SemiAnalysis 自述"多家一线实验室容量规划团队在用 API")。
- MLPerf Agentic Inference 的信息来自二手报道,没读 MLCommons 原始公告;数据中心版是否已有正式提交结果不确定。Artificial Analysis 没查到可比负载。
- vLLM 10-07 与 10-23 两篇 V4.1-Flash 文章:本次只读到索引页摘要(工具限额),5× 与 7.9× 的精确基线、逐项优化的贡献拆分、"DeepInfer"是什么都未在此核实;共用说明里写这两篇已有人读过。M3 on MI355X 与 NIXL push 两篇只读了开头。
- TileRT 文章日期:搜索摘要给 9-24,另有第三方表格写榜单日期 8-24,未能核对。
- Forbes 原文未直接读到,相关数字经两层转述。
