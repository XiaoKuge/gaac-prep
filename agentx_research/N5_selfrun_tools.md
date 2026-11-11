# N5 · 可以自己跑的推理压测工具

调研日期:2026-11-10。方法:直接读各项目的 README 或文档页,并用 GitHub API 查仓库状态。每条事实后的方括号是来源编号。只记录页面上写明的内容;页面没写的标"未提及"。

## 1. GuideLLM

- 定位:面向 SLO 的压测与评估平台,模拟对 OpenAI 兼容服务器和 vLLM 原生服务器的调用 [1]。
- 归属:`vllm-project` 组织下;引用信息里的作者是 Neural Magic, Inc. [1]。Apache-2.0,未归档,2026-11-10 仍有提交 [6]。
- 安装:`pip install guidellm[recommended]`;Python 3.11–3.15;另有容器镜像 `ghcr.io/vllm-project/guidellm` [1]。
- 示例:`guidellm run --backend kind=openai_http,target=http://localhost:8000 --profile kind=sweep --constraint kind=max_duration,seconds=30 --data kind=synthetic_text,prompt_tokens=256,output_tokens=128` [1]。
- 负载形态(`--profile kind=`):`synchronous`、`concurrent`、`throughput`、`constant`、`poisson`、`sweep` [1]。
- 停止条件:`max_duration`、`max_requests`、`max_errors`、`over_saturation` [1]。
- 数据(`--data kind=`):`synthetic_text`、`huggingface`、`json_file`、`csv_file`、`text_file`、`trace_synthetic`,以及多模态的 `synthetic_image`、`synthetic_video` [1]。
- 指标:TTFT、ITL 的完整分布和端到端表现 [1]。
- 输出:JSON、YAML、CSV、HTML、图表 [1]。
- 多回合:README 未提及 [1]。

## 2. LLMPerf

- 定位:评估 LLM API 性能的工具,含负载测试和正确性测试 [2]。
- 归属:`ray-project` 组织下。**仓库已归档,最后一次提交是 2024-12-09** [6]。Apache-2.0 [6]。
- 负载测试:并发发请求;提示词是从莎士比亚十四行诗里随机抽行;所有 API 统一用 LlamaTokenizer 数 token [2]。
- 参数:`--mean-input-tokens`、`--stddev-input-tokens`、`--mean-output-tokens`、`--stddev-output-tokens`、`--num-concurrent-requests`、`--max-num-completed-requests`、`--llm-api` [2]。
- 指标:token 间延迟、生成吞吐 [2]。
- 支持的 API:OpenAI 兼容、Anthropic、Vertex AI、SageMaker,以及经 LiteLLM 接入的其他服务商 [2]。
- 自述的局限:结果取决于服务商后端,不一定反映特定硬件上的软件;随时段和负载变化;不一定与用户自己的负载相关 [2]。

## 3. vllm bench

- vLLM 自带的命令行:`vllm bench serve`、`vllm bench throughput`、`vllm bench forward`、`vllm bench mm-processor`、`vllm bench sweep serve` [3]。
- 数据集:公开的有 ShareGPT、BurstGPT 等;合成的有 Random、Prefix Repetition 等;还可用 SPEED-Bench、BFCL、自定义 JSONL 和带时间戳的追踪重放(`timed_trace`)[3]。文档页没有列出 Sonnet 数据集 [3]。
- 关键参数:`--request-rate`(默认无限大;设为有限值时默认泊松到达)、`--burstiness`、`--max-concurrency`、`--num-prompts`、`--seed` [3]。
- 指标:TTFT、ITL、TPOT 的均值、中位数、P99;端到端延迟;请求吞吐、输出 token 吞吐、总 token 吞吐 [3]。
- 前缀缓存:`prefix_repetition` 数据集可以设定前缀长度、后缀长度和前缀个数 [3]。
- 文档的提醒:对同一台服务器重复跑,会复用缓存的前缀,吞吐虚高;建议用 `vllm bench sweep serve`,它在每轮之间重置服务器缓存 [3]。
- 多回合:文档页没有描述多回合对话基准 [3]。

## 4. AIPerf

- 定位:NVIDIA 的生成式 AI 压测工具 [4]。README 有"从 GenAI-Perf 迁移"的指南,没有用"继任者"这个词 [4]。
- Apache-2.0,未归档,2026-11-11 仍有提交 [6]。安装:`pip install aiperf` [4]。
- 负载形态:并发、请求速率、带并发上限的请求速率、追踪重放、固定时间表;到达分布有恒定、泊松、伽马 [4]。
- 数据:合成;ShareGPT、SPEED-Bench 等公开数据集;追踪格式有 Bailian、BurstGPT、Weka、TraceLab 等 [4]。
- 多回合:支持 [4]。
- 智能体:自带编程智能体追踪生成器;能重放 Weka 和 TraceLab 的真实编程会话;**内置 AgentX 场景:`--scenario agentx`(InferenceX 的默认值)和 `--scenario inferencex-agentx-mvp`** [4]。
- 指标:TTFT、第二个 token 时间、请求延迟、ITL、输出 token 吞吐、每用户输出 token 吞吐、请求吞吐;另有 goodput(满足 SLO 的吞吐)、GPU 遥测 [4]。

## 5. inference-perf

- 定位:Kubernetes SIG 的生成式 AI 推理压测工具,不绑定具体的模型服务器;源自 wg-serving 工作组的指标标准化工作 [5]。
- Apache-2.0,未归档,2026-11-10 仍有提交 [6]。
- 负载形态:恒定速率、泊松到达、并发用户模拟、多阶段;支持追踪重放 [5]。
- 数据:ShareGPT、CNN DailyMail 等真实数据;合成数据;共享前缀和多回合对话;Weka 追踪、对话重放、合成的智能体负载 [5]。
- 指标:TTFT、TPOT、ITL、归一化 TPOT;输入、输出、总 token 吞吐;goodput [5]。
- 已验证的服务器:vLLM、SGLang、TGI [5]。

## 6. MLPerf Inference v6.1(补充)

- MLCommons 于 2026-09-16 公布;新增端到端 RAG 和 Edge Agentic 两项;Edge Agentic 面向单用户设备上的编程助手负载,框架改编自尚未发布的数据中心版智能体基准;30 家组织、486 项结果 [7]。
- 以上来自搜索结果摘要,转述的是 HPCwire 转载的 MLCommons 公告;没有逐字读原文 [7]。

## 对比

| | GuideLLM | LLMPerf | vllm bench | AIPerf | inference-perf |
|---|---|---|---|---|---|
| 维护状态 | 活跃 | 已归档 | 活跃 | 活跃 | 活跃 |
| 开环到达(泊松等) | 有 | 未提及 | 有 | 有 | 有 |
| 追踪重放 | 有合成追踪 | 无 | 有 | 有 | 有 |
| 多回合 | 未提及 | 无 | 未提及 | 有 | 有 |
| 前缀缓存负载 | 未提及 | 无 | 有 | 有 | 有 |
| 能直接跑 AgentX | 否 | 否 | 否 | 能 | 能重放 Weka 追踪 |

## 更正

之前 N4 笔记的脚注和口头回答里有三处与文档不符:
- 说 LLMPerf 可用。实际仓库已归档近两年。
- 说 `vllm bench` 支持 sonnet 数据集并"另有多回合基准脚本"。当前文档页没有列出 Sonnet,也没有描述多回合基准。
- 说 GuideLLM 的速率模式是 synchronous、concurrent、constant、poisson、sweep。实际还有 throughput。

## 来源

1. GuideLLM README — https://github.com/vllm-project/guidellm
2. LLMPerf README — https://github.com/ray-project/llmperf
3. vLLM 文档 Benchmark CLI — https://docs.vllm.ai/en/latest/benchmarking/cli/
4. AIPerf README — https://github.com/ai-dynamo/aiperf
5. inference-perf README — https://github.com/kubernetes-sigs/inference-perf
6. GitHub API 的仓库信息(归档状态、许可证、最后提交时间),2026-11-10 查询
7. 搜索结果摘要,指向 https://www.hpcwire.com/off-the-wire/mlcommons-sets-participation-record-with-new-mlperf-inference-v6-1-benchmark-results/

## 没查到 / 不确定

- 各页面由一个小模型摘要后返回,我没有逐字读原文;参数名可能有个别出入。
- GuideLLM 和 vllm bench 是否支持多回合:只能说 README 和 CLI 文档页没提,其他文档页没有查。
- GenAI-Perf 的现状没有查。
