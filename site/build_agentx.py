#!/usr/bin/env python3
"""生成 site/agentx.html:AgentX 深度调研报告。

用法:  python3 site/build_agentx.py
正文在 site/agentx_body.html(手写);原始调研笔记在 ../agentx_research/N1–N4。
"""
import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent
sys.path.insert(0, str(SITE))
from build import page  # noqa: E402


def build():
    body = (SITE / "agentx_body.html").read_text(encoding="utf-8")
    toc = []
    for m in re.finditer(r'<section[^>]*id="([^"]+)"[^>]*>(.*?)</section>', body, flags=re.S):
        sid, inner = m.group(1), m.group(2)
        kind = re.search(r'<span class="kind">(.*?)</span>', inner)
        h2 = re.search(r"<h2>(.*?)</h2>", inner, flags=re.S)
        label = kind.group(1) if kind else re.sub(r"<[^>]+>", "", h2.group(1))
        toc.append(f'<a href="#{sid}">{label}</a>')
    content = f"""<main class="wrap">
<div class="hero">
  <div class="eyebrow">深度调研</div>
  <h1>AgentX:vLLM 两篇博客唯一的考卷</h1>
  <p class="lead">SemiAnalysis 的智能体推理基准 AgentX 的完整拆解:数据集怎么来、怎么重放、每个指标怎么算、成本怎么折算、谁能上榜、目前的成绩,以及它没测到什么。</p>
  <div class="meta"><span>调研日期 2026-11-10</span><span>依据 InferenceX 仓库、agentx-harness、公开数据集与官方文章</span>
  <a href="research.html#agentx">概念调研里的简介 →</a><a href="post-2.html">用到它的 vLLM 博客 →</a></div>
</div>
<div class="report-grid">
<aside class="toc toc-long"><div class="t">目录</div>
{chr(10).join(toc)}
</aside>
<div class="report">
{body}
</div>
</div>
</main>"""
    (SITE / "agentx.html").write_text(page("AgentX 深度调研", "agentx.html", content), encoding="utf-8")
    print("built: agentx.html", f"({len(toc)} 节)")


if __name__ == "__main__":
    build()
