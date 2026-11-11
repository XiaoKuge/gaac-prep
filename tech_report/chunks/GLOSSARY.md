# 翻译规范(四个分块共用)

把英文技术报告分块译成中文 Markdown。读者是懂 LLM 推理/训练的中文工程师。

## 必须原样保留
- 所有 `$...$`、`$$...$$` 数学公式,一个字符都不要改(包括里面的英文)。
- 公式外面那层由 `--` 和空格组成的表格框线(用来放公式编号 (1)(2)…),整块原样保留。
- 所有链接目标,例如 `(#bib.bib36)`、`(#S2.E2)`、`(https://…)`。引用的方括号文字(如 `[DeepSeek-AI, 2025]`)不译。
- 图片行 `![…](…)` 原样保留。
- Markdown 结构:标题层级(`#` 数量)和编号、列表、表格的行列结构、代码块。
- 模型名、基准名、产品名、人名、数字、单位:DeepSeek-V4.1-Flash、Kimi-K3、Terminal-Bench 2.1、Claude Code 等一律不译。
- 表格:只翻译表头和行标签里的普通词(如 Benchmark→基准,World Knowl.→世界知识),基准名/模型名/数字不动,表格的 ASCII 框线宽度不必对齐但行列数不能变。

## 以 `CAPTION:: ` 开头的行
是图表标题。保留 `CAPTION:: ` 前缀,把 `Figure N:` 译为 `图 N:`,`Table N:` 译为 `表 N:`,`Algorithm N` 译为 `算法 N`,其后文字照常翻译。

## 术语表(首次出现可在括号里附英文,之后直接用中文或保留英文)
- prefill → 预填充;decode → 解码;KV cache → KV 缓存;global KV → 全局 KV;main KV → 主 KV;indexer → 索引器;indexer K / indexer Q → 索引器 K / 索引器 Q;Top-K indices → Top-K 索引
- Sliding-Window Attention (SWA) → 滑动窗口注意力(SWA);SWA Bounded Replay → SWA 有界重放;replay → 重放
- Causal Encoder-Decoder (CED) → 因果编码器-解码器(CED);Compressed Sparse Attention 2 (CSA2) → 压缩稀疏注意力 2(CSA2)
- Full / Reindex / Reuse Mode → Full / Reindex / Reuse 模式(模式名保留英文)
- Hierarchical Sparse Indexer → 分层稀疏索引器;candidate pool → 候选池
- Mixture-of-Experts (MoE) → 混合专家(MoE);expert → 专家;routed expert → 路由专家;load balancing → 负载均衡
- persistent KV cache → 持久化 KV 缓存;host memory → 主机内存;HBM、SSD、RDMA、NVLink 保留
- Single-Pass mHC、Mega-mHC、Engram、DSpark、Muon、Sinkhorn、YOCO 保留英文;residual stream → 残差流;kernel → 内核;kernel fusion → 内核融合
- speculative decoding → 推测解码;draft / drafter → 草稿 / 草稿模型;verification → 验证
- pre-training → 预训练;post-training → 后训练;supervised fine-tuning (SFT) → 监督微调(SFT);reinforcement learning (RL) → 强化学习(RL);on-policy distillation (OPD) → 同策略蒸馏(OPD);rollout → rollout(保留)
- agent / agentic → 智能体 / 智能体式;scaffold → 脚手架(scaffold);reasoning effort → 推理强度;long-horizon → 长程;token、batch、checkpoint、logits 保留英文
- vision encoder → 视觉编码器;multimodal → 多模态;throughput → 吞吐;latency → 延迟;footprint → 占用
- benchmark → 基准;held-out → 留出(held-out)

## 文风
- 忠实、完整:逐段翻译,不删减、不合并段落、不加自己的评论或总结,不漏句子。
- 通顺的中文技术文体,不要逐词硬译;长句可以拆开。
- 用中文全角标点(,。:;()、),数字和英文两侧不必刻意加空格但要统一。
- "we" 译为"我们"。
- 输出文件只包含译文本身,不要加任何说明、前言或代码围栏。
