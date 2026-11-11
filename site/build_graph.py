#!/usr/bin/env python3
"""生成知识图谱:site/assets/graph-data.js(数据)和 site/graph.html(可视化页面)。

用法:  python3 site/build_graph.py
节点七类:org(公司与机构)、model(模型)、hw(硬件)、engine(推理引擎与框架)、tech(优化技术)、bench(基准)、metric(指标)。
基准节点的简介取自 ../tech_report/bench/G*.json;其余节点和全部关系在本文件里手工整理,
依据是站内的两篇博客译文、技术报告、基准调研和 AgentX 调研。
"""
import json
import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent
BENCH = SITE.parent / "tech_report" / "bench"
sys.path.insert(0, str(SITE))
from build import page  # noqa: E402
from build_bench import norm, slug, GROUPS  # noqa: E402

# ----------------------------------------------------------------------------
# 公司与机构
# ----------------------------------------------------------------------------
ORGS = {
    "semianalysis": ("SemiAnalysis", "半导体与 AI 基础设施研究机构;InferenceX 和 AgentX 的维护方。", "research.html#semianalysis"),
    "inferact": ("Inferact", "vLLM 核心维护者创办的公司,两篇博客的第一署名方。", "research.html#inferact"),
    "deepseek": ("DeepSeek", "DeepSeek-V4.1-Flash 的开发者;开源了 DeepInfer 和一批推理内核。", "research.html#deepseek-v41-flash"),
    "nvidia": ("NVIDIA", "GB300 NVL72 等硬件的厂商;开发 TensorRT-LLM、Dynamo、AIPerf、NIXL。", "research.html#gb300-nvl72"),
    "amd": ("AMD", "MI355X 等硬件的厂商;开发 ATOM 引擎和 ATOMesh 路由器。", "agentx.html#results"),
    "anthropic": ("Anthropic", "Claude Code 的开发者。AgentX 的会话记录来自 Claude Code。", "agentx.html#dataset"),
    "openai": ("OpenAI", "HumanEval、GSM8K 的提出者;2026 年 2 月宣布不再报告 SWE-bench Verified。", "benchmarks.html#why"),
    "google": ("Google(DeepMind / Research)", "多个基准的提出者;TPU 的厂商。", ""),
    "meta": ("Meta", "MultiLoKo、ProgramBench 的参与方;Muse Spark Open 模型在 AgentX 上被测。", ""),
    "microsoft": ("Microsoft", "AGIEval 的提出方。", ""),
    "alibaba": ("阿里巴巴(Qwen)", "Qwen 系列模型在 AgentX 上被测;其 KV 缓存追踪研究被 AgentX 借鉴。", "agentx.html#compare"),
    "moonshot": ("Moonshot AI(Kimi)", "Kimi K3 系列在 AgentX 上被测;生产流量使用 Mooncake。", ""),
    "minimax": ("MiniMax", "MiniMax M3 系列在 AgentX 上被测。", ""),
    "zhipu": ("智谱 AI(GLM)", "GLM 5.x 在 AgentX 上被测;与清华合作 LongBench-V2。", ""),
    "bytedance": ("字节跳动 Seed", "NL2Repo-Bench 牵头方;参与 SuperGPQA。", ""),
    "huawei": ("华为", "DeepInfer 支持昇腾 950。", "research.html#deepinfer"),
    "aws": ("AWS", "向 vLLM 贡献了推送式 NIXL 连接器。", ""),
    "redhat": ("Red Hat", "llm-d 的主要推动者。", ""),
    "radixark": ("RadixArk(SGLang 团队)", "SGLang 的维护团队。", ""),
    "ucb": ("加州大学伯克利分校", "vLLM 的发源地;CyberGym、ExploitGym、Agents' Last Exam、LiveCodeBench、MATH 的出处之一。", ""),
    "tbteam": ("Terminal-Bench 团队", "由 Stanford 与 Laude Institute 发起,现由 Harbor 框架托管。", ""),
    "ethz": ("苏黎世联邦理工 MathArena 团队", "维护 MathArena 的三项数学基准。", ""),
    "nyu": ("纽约大学", "GPQA、CVBench 的出处。", ""),
    "tigerlab": ("TIGER-Lab(滑铁卢大学)", "MMLU-Pro 的提出方。", ""),
    "cais": ("Center for AI Safety", "与 Scale AI 牵头 Humanity's Last Exam。", ""),
    "scale": ("Scale AI", "SWE-Bench Pro 的发布方;参与 Humanity's Last Exam。", ""),
    "datacurve": ("Datacurve", "DeepSWE 的发布方。", ""),
    "proximal": ("Proximal", "FrontierSWE 的发布方。", ""),
    "uiuc": ("伊利诺伊大学厄巴纳-香槟分校", "SEC-Bench Pro 的出处。", ""),
    "zapier": ("Zapier", "AutomationBench 的发布方。", ""),
    "surge": ("Surge AI", "Chartography 的发布方。", ""),
    "unipat": ("UniPat AI", "BabyVision 牵头方。", ""),
    "epoch": ("Epoch AI", "独立评审机构;把 DeepSWE v1.1、Terminal-Bench 4.0、HLE 评为有缺陷。", "benchmarks.html#caveats"),
    "vals": ("Vals AI", "第三方榜单;在统一脚手架下复测 Terminal-Bench 4.0。", "benchmarks.html#caveats"),
    "mlcommons": ("MLCommons", "MLPerf Inference 的主办方。", "agentx.html#compare"),
    "aa": ("Artificial Analysis", "以黑盒方式测公开 API 的性能。", "agentx.html#compare"),
    "a16z": ("Andreessen Horowitz", "领投 Inferact 种子轮。", "research.html#inferact"),
    "lightspeed": ("Lightspeed", "领投 Inferact 种子轮。", "research.html#inferact"),
    "clouds": ("算力赞助云商", "Crusoe、CoreWeave、Nebius、TensorWave、Oracle、TogetherAI 为 InferenceX 提供算力。", "agentx.html#rules"),
    "tsinghua": ("清华大学", "C-Eval、LongBench-V2 的出处之一。", ""),
    "sjtu": ("上海交通大学", "C-Eval 的出处之一。", ""),
    "ai2": ("Allen Institute for AI", "DROP、HellaSwag 的出处之一。", ""),
    "map": ("M-A-P 开源社区", "SuperGPQA 的发起方;参与 NL2Repo-Bench。", ""),
    "mmmu": ("MMMU 团队", "MMMU-Pro 的提出方,作者来自卡内基梅隆大学等。", ""),
    "iiit": ("IIIT 海得拉巴", "DocVQA 的出处。", ""),
    "unc": ("北卡罗来纳大学教堂山分校", "RefCOCO、RefCOCO+ 的出处。", ""),
    "camhku": ("剑桥大学 / 香港大学", "ZeroBench 的主要作者单位。", ""),
    "bigcode": ("BigCode 社区", "BigCodeBench 的提出方。", ""),
}

# ----------------------------------------------------------------------------
# 推理引擎与框架  (id: 名称, 小类, 简介, 链接)
# ----------------------------------------------------------------------------
ENGINES = {
    "vllm": ("vLLM", "推理引擎", "开源 LLM 推理与服务引擎;两篇博客的主角,AgentX 规则里的两个主引擎之一。", "research.html#inferact"),
    "sglang": ("SGLang", "推理引擎", "开源推理引擎;AgentX 规则里的另一个主引擎。", "agentx.html#results"),
    "trtllm": ("TensorRT-LLM", "推理引擎", "NVIDIA 的厂商引擎。", "agentx.html#results"),
    "atom": ("ATOM", "推理引擎", "AMD 的厂商引擎。SemiAnalysis 称其在生产中很少被使用,AMD 对此有异议。", "agentx.html#results"),
    "deepinfer": ("DeepInfer", "推理引擎", "DeepSeek 自家的推理框架,主要基于 vLLM 构建;2026 年 10 月开源。", "research.html#deepinfer"),
    "tilert": ("TileRT", "推理引擎", "出现在 AgentX 上的另一个引擎;AMD 文章称其在 GLM-5.3 上取得最高单用户速度(二手信息)。", "agentx.html#results"),
    "dynamo": ("NVIDIA Dynamo", "编排与路由", "分布式推理的编排层,负责路由和预填充/解码拆分,与 vLLM、SGLang、TensorRT-LLM 组合使用。", "agentx.html#techniques"),
    "llmd": ("llm-d", "编排与路由", "面向 Kubernetes 的服务栈,编排多节点 vLLM。", "research.html#wide-ep"),
    "atomesh": ("ATOMesh", "编排与路由", "AMD 的路由器,SGLang 路由器的精简分叉。", ""),
    "nixl": ("NIXL", "KV 传输与缓存", "NVIDIA 的传输库;预填充与解码实例之间搬运 KV 缓存。", "research.html#pd-disaggregation"),
    "lmcache": ("LMCache", "KV 传输与缓存", "KV 缓存卸载后端。", "agentx.html#techniques"),
    "mooncake": ("Mooncake", "KV 传输与缓存", "KV 传输与存储引擎,Moonshot 的生产流量在用。", "agentx.html#techniques"),
    "aiperf": ("AIPerf", "负载生成器", "NVIDIA 的负载生成工具。", "agentx.html#what"),
    "harness": ("agentx-harness", "负载生成器", "SemiAnalysis 对 AIPerf 的分叉,AgentX 的重放工具。", "agentx.html#replay"),
}

# ----------------------------------------------------------------------------
# 技术报告之外的基准  (id: 名称, 简介, 链接)
# ----------------------------------------------------------------------------
EXTRA_BENCH = {
    "inferencex": ("InferenceX", "SemiAnalysis 的开源推理基准套件,原名 InferenceMAX。", "research.html#semianalysis"),
    "agentx": ("AgentX", "InferenceX 的智能体编程场景:重放 393 条 Claude Code 会话,测服务效率。", "agentx.html"),
    "fixed": ("InferenceX 固定场景(8k/1k 等)", "固定输入输出长度、随机 token 的早期场景。", "agentx.html#what"),
    "mlperf": ("MLPerf Inference", "MLCommons 的推理基准;2026 年新增智能体类负载。", "agentx.html#compare"),
    "aaperf": ("Artificial Analysis 性能基准", "测公开 API 的输出速度与首字时延;2026 年 11 月起含多回合智能体任务。", "agentx.html#compare"),
    "swev": ("SWE-bench Verified", "上一代代码智能体基准;因饱和、测试缺陷和污染被弃用。", "benchmarks.html#why"),
    "speedbench": ("SPEED-Bench", "推测解码基准;AgentX 的标准接受长度在它的编程类数据上测得。", "agentx.html#rules"),
}

# 技术报告里的基准 → 提出机构
BENCH_ORG = {
    "AGIEval": ["microsoft"], "MMLU-Pro": ["tigerlab"], "C-Eval": ["sjtu", "tsinghua"], "MultiLoKo": ["meta"],
    "SimpleQA-Verified": ["google"], "SuperGPQA": ["map", "bytedance"], "BBH": ["google"], "BBEH": ["google"],
    "DROP": ["ai2"], "HellaSwag": ["ai2"], "BigCodeBench": ["bigcode"], "HumanEval": ["openai"],
    "GSM8K": ["openai"], "MATH": ["ucb"], "MGSM": ["google"], "LongBench-V2": ["tsinghua", "zhipu"],
    "MMMU-Pro": ["mmmu"], "DocVQA": ["iiit"], "CVBench": ["nyu"], "RefCOCO": ["unc", "google"],
    "GPQA Diamond": ["nyu"], "HLE": ["cais", "scale"], "Codeforces": [], "MathArena Apex": ["ethz"],
    "Apex 2025 Shortlist": ["ethz"], "AIME 2026": ["ethz"], "IMO-AnswerBench": ["google"],
    "LiveCodeBench": ["ucb"], "Terminal-Bench 2.1": ["tbteam"], "Terminal-Bench 3.0": ["tbteam"],
    "Terminal-Bench 4.0": ["tbteam"], "DeepSWE v1.1": ["datacurve"], "ProgramBench": ["meta"],
    "NL2Repo-Bench": ["bytedance", "map"], "SWE-Bench Pro": ["scale"], "FrontierSWE v2": ["proximal"],
    "SEC-Bench Pro": ["uiuc"], "CyberGym": ["ucb"], "ExploitGym": ["ucb"], "AutomationBench": ["zapier"],
    "Agents' Last Exam": ["ucb"], "Chartography": ["surge"], "BabyVision": ["unipat"], "ZeroBench": ["camhku"],
}

# 其余关系  (源, 目标, 关系, 关系组)
#   关系组: make=开发/发布  use=采用/参评  base=基于/包含/继任  judge=评审/弃用  fund=投资/赞助
EDGES = [
    # 机构 → 引擎
    ("inferact", "vllm", "主导开发", "make"), ("ucb", "vllm", "发源地", "make"),
    ("radixark", "sglang", "维护", "make"), ("nvidia", "trtllm", "开发", "make"),
    ("nvidia", "dynamo", "开发", "make"), ("nvidia", "aiperf", "开发", "make"), ("nvidia", "nixl", "开发", "make"),
    ("amd", "atom", "开发", "make"), ("amd", "atomesh", "开发", "make"),
    ("deepseek", "deepinfer", "开发并开源", "make"), ("redhat", "llmd", "推动", "make"),
    ("semianalysis", "harness", "维护", "make"), ("aws", "nixl", "贡献推送式连接器", "make"),
    ("moonshot", "mooncake", "生产使用", "use"),
    ("deepseek", "vllm", "贡献内核", "make"), ("nvidia", "vllm", "协作优化", "make"), ("amd", "vllm", "协作优化", "make"),
    # 引擎 → 引擎
    ("deepinfer", "vllm", "主要基于", "base"), ("harness", "aiperf", "分叉自", "base"),
    ("atomesh", "sglang", "路由器分叉自", "base"),
    ("dynamo", "vllm", "编排", "base"), ("dynamo", "sglang", "编排", "base"), ("dynamo", "trtllm", "编排", "base"),
    ("llmd", "vllm", "编排", "base"), ("nixl", "vllm", "KV 传输", "base"), ("nixl", "dynamo", "KV 传输", "base"),
    ("lmcache", "vllm", "KV 卸载后端", "base"), ("lmcache", "atom", "KV 卸载后端", "base"),
    ("mooncake", "vllm", "KV 存储后端", "base"),
    ("vllm", "deepinfer", "对标并追平", "use"),
    # 引擎 → 基准
    ("vllm", "agentx", "参评(主引擎)", "use"), ("sglang", "agentx", "参评(主引擎)", "use"),
    ("trtllm", "agentx", "参评", "use"), ("atom", "agentx", "参评", "use"),
    ("dynamo", "agentx", "参评(编排层)", "use"), ("tilert", "agentx", "参评", "use"),
    ("harness", "agentx", "重放工具", "base"),
    ("vllm", "b:GSM8K", "精度验证", "use"), ("vllm", "b:GPQA Diamond", "精度验证", "use"),
    # 机构 → 基准(技术报告之外)
    ("semianalysis", "inferencex", "维护", "make"), ("semianalysis", "agentx", "发布", "make"),
    ("mlcommons", "mlperf", "主办", "make"), ("aa", "aaperf", "运营", "make"),
    ("nvidia", "speedbench", "提出", "make"),
    ("anthropic", "agentx", "Claude Code 会话来源", "base"),
    ("nvidia", "inferencex", "赞助硬件 / 签核配方", "fund"), ("amd", "inferencex", "赞助硬件 / 签核配方", "fund"),
    ("clouds", "inferencex", "赞助算力", "fund"),
    ("a16z", "inferact", "领投", "fund"), ("lightspeed", "inferact", "领投", "fund"),
    ("alibaba", "agentx", "脱敏方法被借鉴", "base"),
    # 评审 / 弃用
    ("epoch", "b:DeepSWE v1.1", "评为有缺陷", "judge"), ("epoch", "b:Terminal-Bench 4.0", "评为有缺陷", "judge"),
    ("epoch", "b:HLE", "评为有缺陷", "judge"), ("vals", "b:Terminal-Bench 4.0", "第三方复测", "judge"),
    ("openai", "swev", "宣布弃用", "judge"), ("openai", "b:SWE-Bench Pro", "审计出约三成问题题", "judge"),
    # 基准 → 基准
    ("inferencex", "agentx", "包含", "base"), ("inferencex", "fixed", "包含", "base"),
    ("fixed", "agentx", "被其取代为主场景", "base"),
    ("agentx", "b:GSM8K", "精度门槛", "base"), ("agentx", "speedbench", "接受长度来源", "base"),
    ("mlperf", "agentx", "同类(智能体推理)", "base"), ("aaperf", "agentx", "互补(黑盒对白盒)", "base"),
    ("swev", "b:SWE-Bench Pro", "继任者", "base"),
    ("b:Terminal-Bench 2.1", "b:Terminal-Bench 3.0", "后续版本", "base"),
    ("b:Terminal-Bench 3.0", "b:Terminal-Bench 4.0", "修订版", "base"),
    ("b:MathArena Apex", "b:Apex 2025 Shortlist", "同一次选题", "base"),
    ("b:SWE-Bench Pro", "b:DeepSWE v1.1", "被更新的基准接替", "base"),
]

# ----------------------------------------------------------------------------
# 模型  (id: 名称, 小类, 简介, 链接)
# ----------------------------------------------------------------------------
MODELS = {
    "dsv41flash": ("DeepSeek-V4.1-Flash", "AgentX 在测", "552B 参数的开源模型;技术报告和两篇 vLLM 博客的主角。", "research.html#deepseek-v41-flash"),
    "dsv4pro": ("DeepSeek V4 Pro", "AgentX 首发", "AgentX 发布时的五个模型之一,约 1.6T 参数;后被 V4.1-Flash 取代。", "agentx.html#results"),
    "dsr1": ("DeepSeek-R1", "早期场景", "InferenceX 固定场景(8k/1k)时期的代表模型。", "agentx.html#what"),
    "k3": ("Kimi K3", "AgentX 首发", "2.8T 参数;AgentX 发布时的五个模型之一。", "agentx.html#results"),
    "k31pro": ("Kimi K3.1 Pro", "AgentX 在测", "Kimi K3 的后继版本。", "agentx.html#results"),
    "m3": ("MiniMax M3", "AgentX 首发", "约 430B 参数;AgentX 发布时的五个模型之一。", "agentx.html#results"),
    "m31flash": ("MiniMax M3.1 Flash", "AgentX 在测", "MiniMax M3 的后继版本。", "agentx.html#results"),
    "glm53": ("GLM 5.3", "AgentX 在测", "在 GLM 5.2(744B)基础上做了后训练。", "agentx.html#results"),
    "qwen35": ("Qwen3.5", "AgentX 在测", "397B 参数;原生上下文 262k,所以跑 256k 版数据集。", "agentx.html#results"),
    "qwen38": ("Qwen3.8 Flash Next", "AgentX 在测", "176B 参数;发布之后新增的模型。", "agentx.html#results"),
    "muse": ("Muse Spark Open", "AgentX 在测", "Meta 的 816B 参数模型;发布之后新增。", "agentx.html#results"),
    "dspark": ("DSpark 草稿模型", "草稿模型", "V4.1-Flash 推测解码用的草稿头;草稿长度 5、思考开时标准接受长度 3.51。", "agentx.html#rules"),
    "claude": ("Claude 系列模型", "录制来源", "AgentX 的 393 条会话是 Claude Code 配 Claude 模型跑出来的;重放时模型名被改写成被测模型。", "flow.html"),
}

# ----------------------------------------------------------------------------
# 硬件  (id: 名称, 小类, 简介, 链接)
# ----------------------------------------------------------------------------
HARDWARE = {
    "gb300": ("GB300 NVL72", "NVIDIA 机柜级", "72 颗 GPU 的整机柜;vLLM 第二篇博客的测试平台。TCO 常数:自有 $2.31、租用 $5.00 每 GPU 小时。", "research.html#gb300-nvl72"),
    "gb200": ("GB200 NVL72", "NVIDIA 机柜级", "GB300 的上一代机柜。", "agentx.html#results"),
    "rubin": ("Vera Rubin NVL72", "NVIDIA 机柜级", "下一代机柜;2026 年 9 月出现首批 AgentX 结果,部分数字由 NVIDIA 自测、待审核。", "agentx.html#results"),
    "b300": ("B300", "NVIDIA 单节点", "8 卡服务器;标准接受长度在它上面测得。", "agentx.html#rules"),
    "b200": ("B200", "NVIDIA 单节点", "8 卡服务器。", "agentx.html#results"),
    "h200": ("H200", "NVIDIA 单节点", "Hopper 架构;NVIDIA 用它作代际对比的基线。", "agentx.html#results"),
    "h100": ("H100", "NVIDIA 单节点", "Hopper 架构;InferenceX 仓库里有 V4.1-Flash 的 H100 配置。", "agentx.html#results"),
    "rtxpro": ("RTX PRO 6000", "NVIDIA 单节点", "RTX PRO 服务器;在 InferenceX 的成本常数表里。", "agentx.html#cost"),
    "mi355x": ("MI355X", "AMD", "AMD 当前主力;AgentX 上与 NVIDIA 对比的主要对象。", "agentx.html#results"),
    "mi325x": ("MI325X", "AMD", "AMD 上一代。", "agentx.html#results"),
    "mi300x": ("MI300X", "AMD", "AMD 更早一代。", "agentx.html#results"),
    "mi455x": ("MI455X UALoE72", "AMD", "AMD 的机柜级产品;AgentX 发布时预告稍后加入。", "agentx.html#results"),
    "tpu7x": ("TPU7x Ironwood", "Google", "2026 年 9 月进入 InferenceX,只有固定场景结果,没有 AgentX 结果。", "agentx.html#results"),
    "ascend950": ("昇腾 950", "华为", "DeepInfer 支持的国产加速器。", "research.html#deepinfer"),
}

# ----------------------------------------------------------------------------
# 优化技术  (id: 名称, 小类, 简介, 链接)
# ----------------------------------------------------------------------------
TECHS = {
    "prefixcache": ("前缀缓存", "缓存", "复用之前轮次算过的开头部分。AgentX 的理论命中率在九成以上,能否兑现是成绩差距的主要来源。", "trace.html"),
    "kvoffload": ("KV 卸载到主机内存", "缓存", "显存放不下的 KV 缓存挪到主机内存。AgentX 规则允许但不强制,只限主机内存。", "agentx.html#techniques"),
    "sticky": ("会话粘性路由", "路由", "同一条会话的请求固定发到同一组 GPU,下一轮到达时前缀还在。", "agentx.html#techniques"),
    "pd": ("P/D 分离", "部署拓扑", "预填充和解码分别放在不同实例上,中间搬运 KV 缓存。", "research.html#pd-disaggregation"),
    "wideep": ("Wide-EP", "并行方式", "把专家并行铺到很多张卡上;DeepSeek V4 系列的默认并行方式。", "research.html#wide-ep"),
    "eplb": ("EPLB", "并行方式", "专家并行下的负载均衡:按专家的冷热重新分布。", "research.html#wide-ep"),
    "dcp": ("上下文并行 DCP / PCP", "并行方式", "把长上下文的注意力计算切到多张卡上。Kimi K3 上 DCP8 优于 TP8。", "agentx.html#techniques"),
    "specdec": ("推测解码", "解码", "草稿模型先猜几个 token,目标模型一次验证。AgentX 强制使用事先测好的接受长度。", "agentx.html#rules"),
    "swa": ("SWA 有界重放", "缓存", "滑窗注意力的状态丢失后,只重放有限的一段而不是整条前缀。vLLM 第一篇博客的手段之一。", "post-1.html"),
    "cudagraph": ("CUDA graphs", "执行", "把一串 GPU 操作录成图一次提交,减少调度开销。", "post-1.html"),
    "fusion": ("内核融合", "执行", "把多步计算合成一个内核,例如 MegaAttention、Mega-mHC、Mega-Gate。", "post-1.html"),
    "chunked": ("分块预填充调度", "调度", "给单次预填充封顶,避免长输入堵住后面的请求。", "agentx.html#techniques"),
    "fp4kv": ("FP4 KV 缓存", "精度", "V4.1-Flash 原生的 4 位 KV 缓存;AgentX 只在模型原生自带时允许。", "report.html"),
}

# ----------------------------------------------------------------------------
# 指标  (id: 名称, 小类, 简介, 链接)
# ----------------------------------------------------------------------------
METRICS = {
    "intvty": ("P90 交互速度", "延迟", "1 ÷ P90(每次请求的 token 间隔)。单位 tok/s/user,AgentX 曲线的横轴。", "flow.html"),
    "ttft": ("首 token 时间 TTFT", "延迟", "发出请求到第一个 token 返回,含排队和读输入。官方说必须和交互速度一起看。", "flow.html"),
    "e2enorm": ("端到端归一化交互速度", "延迟", "1 ÷ P90(总耗时 ÷ 输出 token 数)。把首 token 的等待摊进去。", "flow.html"),
    "tput": ("每芯片总吞吐", "吞吐", "输入加输出的 token 数 ÷ 时间 ÷ 芯片数。缓存命中的输入也计入。", "flow.html"),
    "uncached": ("未缓存口径吞吐", "吞吐", "只算每轮新增的输入加输出。按理论命中率扣除。", "flow.html"),
    "costm": ("每百万 token 成本", "成本", "每芯片小时成本 × 1,000,000 ÷ (3600 × 吞吐)。", "agentx.html#cost"),
    "tokdollar": ("每美元 token 数", "成本", "吞吐 × 3600 ÷ 每芯片小时成本。看板默认纵轴。", "agentx.html#cost"),
    "permw": ("每兆瓦吞吐", "能耗", "吞吐 × 1000 ÷ 每芯片全包功率(千瓦)。NVIDIA 的宣传多用这个口径。", "agentx.html#cost"),
    "cachehit": ("缓存命中率", "缓存", "分理论值(由数据集决定)和实测值(服务器上报)两种。", "flow.html"),
    "al": ("接受长度", "推测解码", "每个验证步平均产出多少 token。AgentX 为每个模型定了一个标准值。", "agentx.html#rules"),
}

# 新增四类节点的关系
EDGES2 = [
    # 机构 → 模型
    ("deepseek", "dsv41flash", "开发", "make"), ("deepseek", "dsv4pro", "开发", "make"), ("deepseek", "dsr1", "开发", "make"),
    ("deepseek", "dspark", "随模型发布", "make"),
    ("moonshot", "k3", "开发", "make"), ("moonshot", "k31pro", "开发", "make"),
    ("minimax", "m3", "开发", "make"), ("minimax", "m31flash", "开发", "make"),
    ("zhipu", "glm53", "开发", "make"), ("alibaba", "qwen35", "开发", "make"), ("alibaba", "qwen38", "开发", "make"),
    ("meta", "muse", "开发", "make"), ("anthropic", "claude", "开发", "make"),
    # 模型 → 模型
    ("dsv4pro", "dsv41flash", "被其取代", "base"), ("k3", "k31pro", "后继版本", "base"), ("m3", "m31flash", "后继版本", "base"),
    ("dspark", "dsv41flash", "它的草稿头", "base"),
    # 模型 → 基准
    ("dsv41flash", "agentx", "在测", "use"), ("k31pro", "agentx", "在测", "use"), ("m31flash", "agentx", "在测", "use"),
    ("glm53", "agentx", "在测", "use"), ("qwen35", "agentx", "在测(256k 版数据)", "use"), ("qwen38", "agentx", "在测", "use"),
    ("muse", "agentx", "在测", "use"),
    ("dsv4pro", "agentx", "首发模型", "use"), ("k3", "agentx", "首发模型", "use"), ("m3", "agentx", "首发模型", "use"),
    ("dsr1", "fixed", "代表模型", "use"), ("claude", "agentx", "会话的录制模型", "base"),
    # 机构 → 硬件
    ("nvidia", "gb300", "生产", "make"), ("nvidia", "gb200", "生产", "make"), ("nvidia", "rubin", "生产", "make"),
    ("nvidia", "b300", "生产", "make"), ("nvidia", "b200", "生产", "make"), ("nvidia", "h200", "生产", "make"),
    ("nvidia", "h100", "生产", "make"), ("nvidia", "rtxpro", "生产", "make"),
    ("amd", "mi355x", "生产", "make"), ("amd", "mi325x", "生产", "make"), ("amd", "mi300x", "生产", "make"), ("amd", "mi455x", "生产", "make"),
    ("google", "tpu7x", "生产", "make"), ("huawei", "ascend950", "生产", "make"),
    # 硬件 → 硬件
    ("gb200", "gb300", "下一代", "base"), ("gb300", "rubin", "下一代", "base"), ("b200", "b300", "下一代", "base"),
    ("h100", "h200", "下一代", "base"), ("mi300x", "mi325x", "下一代", "base"), ("mi325x", "mi355x", "下一代", "base"),
    ("mi355x", "mi455x", "机柜级后续", "base"),
    # 硬件 → 基准
    ("gb300", "agentx", "参评硬件", "use"), ("gb200", "agentx", "参评硬件", "use"), ("b300", "agentx", "参评硬件", "use"),
    ("b200", "agentx", "参评硬件", "use"), ("h200", "agentx", "参评硬件", "use"), ("h100", "agentx", "参评硬件", "use"),
    ("rtxpro", "agentx", "参评硬件", "use"), ("mi355x", "agentx", "参评硬件", "use"), ("mi325x", "agentx", "参评硬件", "use"),
    ("mi300x", "agentx", "参评硬件", "use"), ("rubin", "agentx", "参评硬件(9 月起)", "use"),
    ("tpu7x", "fixed", "只有固定场景结果", "use"), ("b300", "speedbench", "标准接受长度在其上测得", "base"),
    # 引擎 / 模型 → 硬件
    ("dsv41flash", "gb300", "vLLM 博客的测试组合", "use"), ("vllm", "gb300", "第二篇博客的平台", "use"),
    ("trtllm", "rubin", "首批结果(预发布版)", "use"), ("vllm", "tpu7x", "新增 TorchTPU 后端", "feat"),
    ("deepinfer", "ascend950", "支持", "feat"),
    # 引擎 → 技术
    ("vllm", "prefixcache", "间隔保留 + 选择性保留", "feat"), ("sglang", "prefixcache", "滑窗页管理", "feat"), ("atom", "prefixcache", "稀疏检查点保留", "feat"),
    ("vllm", "kvoffload", "支持", "feat"), ("sglang", "kvoffload", "HiCache", "feat"), ("atom", "kvoffload", "支持", "feat"),
    ("dynamo", "kvoffload", "KVBM", "feat"), ("lmcache", "kvoffload", "后端", "feat"), ("mooncake", "kvoffload", "后端", "feat"),
    ("vllm", "sticky", "支持", "feat"), ("llmd", "sticky", "支持", "feat"), ("sglang", "sticky", "缓存亲和", "feat"),
    ("atomesh", "sticky", "支持", "feat"), ("dynamo", "sticky", "KV 感知路由", "feat"),
    ("vllm", "pd", "支持", "feat"), ("sglang", "pd", "支持", "feat"), ("trtllm", "pd", "支持", "feat"), ("atom", "pd", "支持", "feat"),
    ("dynamo", "pd", "负责拆分与扩缩", "feat"), ("nixl", "pd", "搬运 KV 缓存", "feat"),
    ("vllm", "wideep", "支持", "feat"), ("sglang", "wideep", "支持", "feat"), ("trtllm", "wideep", "支持", "feat"), ("llmd", "wideep", "编排", "feat"),
    ("vllm", "eplb", "支持", "feat"), ("eplb", "wideep", "配套", "base"),
    ("vllm", "dcp", "支持", "feat"), ("atom", "dcp", "支持", "feat"),
    ("vllm", "specdec", "支持", "feat"), ("sglang", "specdec", "支持", "feat"), ("trtllm", "specdec", "支持", "feat"), ("atom", "specdec", "支持", "feat"),
    ("vllm", "swa", "实现", "feat"), ("vllm", "cudagraph", "用于预填充", "feat"), ("trtllm", "cudagraph", "支持", "feat"),
    ("vllm", "fusion", "集成", "feat"), ("deepseek", "fusion", "贡献内核", "make"), ("vllm", "chunked", "支持", "feat"),
    # 技术 ↔ 模型 / 基准 / 指标
    ("swa", "dsv41flash", "针对其混合注意力", "feat"), ("wideep", "dsv41flash", "默认并行方式", "feat"),
    ("fp4kv", "dsv41flash", "原生特性", "feat"), ("dcp", "k3", "优于 TP8", "feat"),
    ("dspark", "specdec", "草稿模型", "feat"), ("specdec", "al", "用它衡量", "base"),
    ("speedbench", "al", "标准值在其上测得", "base"), ("agentx", "al", "强制使用标准值", "base"),
    ("agentx", "prefixcache", "主要考察对象", "base"), ("agentx", "kvoffload", "规则允许(仅主机内存)", "base"),
    ("prefixcache", "cachehit", "用它衡量", "base"), ("swa", "prefixcache", "保住混合注意力下的缓存", "base"),
    ("pd", "ttft", "主要改善", "base"), ("specdec", "intvty", "主要改善", "base"), ("prefixcache", "tput", "主要改善", "base"),
    # 基准 → 指标
    ("agentx", "intvty", "横轴", "base"), ("agentx", "ttft", "采用", "base"), ("agentx", "e2enorm", "实验性横轴", "base"),
    ("agentx", "tput", "默认吞吐口径", "base"), ("agentx", "uncached", "可选口径", "base"), ("agentx", "cachehit", "采用", "base"),
    ("inferencex", "costm", "采用", "base"), ("inferencex", "tokdollar", "默认纵轴", "base"), ("inferencex", "permw", "采用", "base"),
    ("tput", "costm", "由它换算", "base"), ("tput", "tokdollar", "由它换算", "base"), ("tput", "permw", "由它换算", "base"),
    ("tput", "uncached", "扣掉缓存部分", "base"), ("cachehit", "uncached", "扣除比例", "base"),
    ("nvidia", "permw", "宣传口径", "use"),
]

REL_GROUPS = {"make": "开发 / 发布", "use": "采用 / 参评", "feat": "支持 / 实现", "base": "基于 / 包含 / 继任",
              "judge": "评审 / 弃用", "fund": "投资 / 赞助"}
TYPES = {"org": "公司与机构", "model": "模型", "hw": "硬件", "engine": "推理引擎与框架",
         "tech": "优化技术", "bench": "基准", "metric": "指标"}


def build_data():
    nodes, edges = [], []
    for oid, (label, desc, link) in ORGS.items():
        nodes.append(dict(id=oid, label=label, type="org", sub="", desc=desc, link=link))
    for eid, (label, sub, desc, link) in ENGINES.items():
        nodes.append(dict(id=eid, label=label, type="engine", sub=sub, desc=desc, link=link))
    for typ, table in (("model", MODELS), ("hw", HARDWARE), ("tech", TECHS), ("metric", METRICS)):
        for nid, (label, sub, desc, link) in table.items():
            nodes.append(dict(id=nid, label=label, type=typ, sub=sub, desc=desc, link=link))
    for bid, (label, desc, link) in EXTRA_BENCH.items():
        nodes.append(dict(id=bid, label=label, type="bench", sub="推理性能" if bid != "swev" else "模型能力", desc=desc, link=link))

    research = {}
    for f in sorted(BENCH.glob("G*.json")):
        for item in json.loads(f.read_text(encoding="utf-8")):
            research[norm(item["name"])] = item
    group_of = {n: title.split(" · ")[-1] for _, title, _, names in GROUPS for n in names}
    for name, orgs in BENCH_ORG.items():
        r = research.get(name, {})
        nodes.append(dict(id="b:" + name, label=name, type="bench", sub=group_of.get(name, ""),
                          desc=r.get("measures", ""), link="benchmarks.html#" + slug(name)))
        for o in orgs:
            edges.append(dict(s=o, t="b:" + name, rel="提出 / 维护", g="make"))
        edges.append(dict(s="dsv41flash", t="b:" + name, rel="技术报告里用它评测", g="use"))
    for s, t, rel, g in EDGES + EDGES2:
        edges.append(dict(s=s, t=t, rel=rel, g=g))

    ids = {n["id"] for n in nodes}
    bad = [(e["s"], e["t"]) for e in edges if e["s"] not in ids or e["t"] not in ids]
    assert not bad, f"边引用了不存在的节点: {bad}"
    return dict(nodes=nodes, edges=edges, types=TYPES, relGroups=REL_GROUPS)


def build():
    data = build_data()
    (SITE / "assets" / "graph-data.js").write_text(
        "window.GRAPH = " + json.dumps(data, ensure_ascii=False, indent=1) + ";\n", encoding="utf-8")
    counts = {t: sum(1 for n in data["nodes"] if n["type"] == t) for t in TYPES}
    body = (SITE / "graph_body.html").read_text(encoding="utf-8")
    body = (body.replace("{{N_NODES}}", str(len(data["nodes"]))).replace("{{N_EDGES}}", str(len(data["edges"])))
            .replace("{{N_ORG}}", str(counts["org"])).replace("{{N_BENCH}}", str(counts["bench"]))
            .replace("{{N_ENGINE}}", str(counts["engine"])).replace("{{N_MODEL}}", str(counts["model"]))
            .replace("{{N_HW}}", str(counts["hw"])).replace("{{N_TECH}}", str(counts["tech"]))
            .replace("{{N_METRIC}}", str(counts["metric"])))
    (SITE / "graph.html").write_text(page("知识图谱", "graph.html", body), encoding="utf-8")
    print(f"built: graph.html  节点 {len(data['nodes'])} {counts}  边 {len(data['edges'])}")


if __name__ == "__main__":
    build()
