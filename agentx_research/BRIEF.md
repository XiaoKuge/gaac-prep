# AgentX 深度调研:共用说明

目标:为一份中文深度调研报告收集关于 **AgentX** 的事实。AgentX 是 SemiAnalysis 的开源推理基准套件 InferenceX(原名 InferenceMAX)里的"智能体编程"场景。今天是 2026-11-10。

## 已经知道的(不用重复查,除非发现有误)
- AgentX 1.0 于 2026-08-24 随 InferenceX v3 公布,Apache 2.0;数据集来自 393 条自愿提供的 Claude Code 会话;每次请求输入 token 中位数 142,016、输出中位数 444;44% 的会话含子智能体;内容脱敏为每 64 token 一块的哈希;用 AIPerf 闭环重放;完整数据集上下文最长 1M,另有 256K 变体;只统计预热后的一小时窗口;指标含每芯片吞吐、P90 交互速度、TTFT;按帕累托前沿在相同交互速度下比较。
- SemiAnalysis 称完整矩阵跑在约 2 MW、1000 多块芯片上,数据集投入超过 300 万美元。
- 已读过的来源:newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat(只读了前 10 万字符,后 6 万字符没读)、inferencex.semianalysis.com/blog/agentic-benchmark-agent-benchmark-guide、NVIDIA 开发者博客关于 Vera Rubin/Blackwell 与 AgentX 的文章、vLLM 博客 2026-10-07 与 2026-10-23 两篇(DeepSeek-V4.1-Flash)。

## 可用的起点
- https://inferencex.semianalysis.com/ (看板、/glossary/agentx、/blog、/evaluation、/reliability 等子页)
- https://github.com/SemiAnalysisAI/InferenceX (仓库、文档、配置、PR、issue)
- https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat
- https://vllm.ai/blog/2026-09-08-vllm-agentx
- AIPerf(NVIDIA 的负载生成工具)仓库与文档
- 搜索引擎

## 输出
把结果写成一个 Markdown 笔记文件(路径见各自任务),简体中文,专有名词保留英文。结构:按小标题分节,每节是一组**具体事实**(数字、日期、名称、定义、公式、配置项),每条事实后面用括号标来源编号;文件末尾列"来源"清单(编号、标题、URL)和"没查到 / 不确定"清单。

## 纪律
- 用自己的话转述;需要引用原文时只引一两句短语,不要大段照抄来源文字。
- 优先一手来源(官方文档、仓库、官方博客),二手报道要注明。数字要带口径(哪个模型、哪种硬件、哪个日期、哪个指标)。
- 查不到就写进"没查到 / 不确定",不要编造。互相矛盾的说法两边都记下并注明出处。
- 搜索与抓取请求里不要带任何个人信息或本机路径。
- 只写指定的那个文件,不改其他文件。
- 最后的回复只需几行:写了多少条事实、最重要的 3–5 个发现、哪些没查到。
