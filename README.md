# gaac-prep · DeepSeek-V4.1-Flash 推理资料

围绕"用 vLLM 服务 DeepSeek-V4.1-Flash"整理的中文阅读材料和调研笔记。

在线阅读:https://xiaokuge.github.io/gaac-prep/

## 目录

- `site/`:本地中文网站(GitHub Pages 发布的就是这个目录)
  - 两篇 vLLM 博客的中文译文
  - DeepSeek-V4.1-Flash 技术报告的中文译文
  - 调研报告:九个概念、技术报告里的评测基准、AgentX
  - 图解:AgentX 数据集中的一条会话
  - 知识图谱
  - `build*.py`:各页面的生成脚本
- `agentx_research/`:AgentX 调研笔记(带来源)
- `tech_report/`:技术报告的 PDF、分段译文和基准调研数据
- 两篇 vLLM 博客的原文和译文(Markdown)
- `inference_deployment_qa.md` 和几张图:DeepSeek 推理部署问答原文及配图

## 重新生成网站

```bash
python3 site/build.py && python3 site/build_report.py && python3 site/build_bench.py \
  && python3 site/build_agentx.py && python3 site/build_graph.py \
  && python3 site/build_trace.py && python3 site/build_flow.py
```

需要 Python 3、`markdown` 包和 pandoc。

## 来源与版权

- vLLM 博客两篇:原文版权归 vLLM 项目及原作者,原文见 https://vllm.ai/blog 。这里的中文译文是非官方翻译,仅供学习。
- DeepSeek-V4.1-Flash 技术报告与部署问答:版权归 DeepSeek。中文译文是非官方翻译。
- AgentX 示例会话的数字取自 SemiAnalysis 公开的数据集(Apache-2.0)。
- 调研报告、图解和知识图谱为本仓库原创整理,引用处均标注了来源。

版权方如有异议,请提 issue,会及时处理。
