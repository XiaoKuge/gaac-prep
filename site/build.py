#!/usr/bin/env python3
"""把两篇中文译稿(上一级目录的 *_中文.md)生成本地静态网站。

用法:  python3 site/build.py
产物:  site/index.html, site/post-1.html, site/post-2.html
research.html 是手写的调研报告,不由本脚本生成。
"""
import html
import re
from pathlib import Path

import markdown

SITE = Path(__file__).resolve().parent
SRC = SITE.parent

POSTS = [
    ("post-1.html", "vLLM博客_2026-10-07_5x_Agentic_Throughput_Since_Day_0_中文.md",
     "https://vllm.ai/blog/2026-10-07-deepseek-v41-flash"),
    ("post-2.html", "vLLM博客_2026-10-23_Serving_DeepSeek-V4.1-Flash_at_Scale_中文.md",
     "https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep"),
]

# (显示名, research.html 里的锚点)
CONCEPTS = [
    ("SemiAnalysis", "semianalysis"),
    ("AgentX", "agentx"),
    ("P90", "p90"),
    ("DeepSeek-V4.1-Flash", "deepseek-v41-flash"),
    ("DeepInfer", "deepinfer"),
    ("Inferact", "inferact"),
    ("GB300 NVL72", "gb300-nvl72"),
    ("P/D 分离", "pd-disaggregation"),
    ("Wide-EP", "wide-ep"),
]


def nav(active):
    items = [("index.html", "首页"), ("post-1.html", "① 5 倍吞吐"),
             ("post-2.html", "② 大规模服务"), ("report.html", "技术报告"),
             ("benchmarks.html", "基准调研"), ("agentx.html", "AgentX 调研"),
             ("research.html", "概念调研"), ("graph.html", "知识图谱")]
    links = "".join(
        f'<a href="{h}"{" class=\"on\"" if h == active else ""}>{t}</a>' for h, t in items)
    return (f'<header class="nav"><div class="nav-inner">'
            f'<a class="brand" href="index.html">vLLM × DeepSeek-V4.1-Flash<small>中文读本</small></a>'
            f'<nav class="nav-links">{links}</nav></div></header>')


def page(title, active, body):
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
{nav(active)}
{body}
</body>
</html>
"""


def parse(path):
    text = path.read_text(encoding="utf-8")
    _, fm, body = text.split("---", 2)
    meta = {}
    for line in fm.splitlines():
        m = re.match(r'^(\w+):\s*"?(.*?)"?\s*$', line)
        if m:
            meta[m.group(1)] = m.group(2)
    return meta, body


def render(body):
    # 站内化:图片/交互页走本地 assets,第一篇的站外链接指向本地页
    body = body.replace('"/assets/', '"assets/').replace("](/assets/", "](assets/")
    body = body.replace(POSTS[0][2], POSTS[0][0])
    out = markdown.markdown(body, extensions=["extra", "sane_lists"])
    out = re.sub(r"<table>.*?</table>", lambda m: f'<div class="table-scroll">{m.group(0)}</div>',
                 out, flags=re.S)
    # 站外链接新标签页打开
    out = re.sub(r'<a href="(https?://[^"]+)"', r'<a href="\1" target="_blank" rel="noopener"', out)
    return out


def chips():
    c = "".join(f'<a class="chip" href="research.html#{a}">{n}</a>' for n, a in CONCEPTS)
    return ('<aside class="concepts"><div class="label">读前先认识这些名词 · '
            '<a href="research.html">打开完整调研报告 →</a></div>'
            f'<div class="chips">{c}</div></aside>')


def build():
    metas = []
    for i, (out_name, src_name, origin) in enumerate(POSTS):
        meta, body = parse(SRC / src_name)
        metas.append(meta)
        other = POSTS[1 - i]
        other_label = "下一篇:大规模服务 →" if i == 0 else "← 上一篇:5 倍吞吐"
        content = f"""<main class="wrap narrow">
<div class="hero">
  <div class="eyebrow">vLLM 博客 · 中文翻译</div>
  <h1>{html.escape(meta["title"])}</h1>
  <p class="lead">{html.escape(meta["summary"])}</p>
  <div class="meta"><span>{html.escape(meta["author"])}</span><span>{meta["date"][:10]}</span>
  <a href="{origin}" target="_blank" rel="noopener">英文原文 ↗</a></div>
</div>
{chips()}
<article class="article">
{render(body)}
</article>
<div class="post-foot"><a href="{other[0]}">{other_label}</a><a href="research.html">概念调研报告 →</a></div>
</main>"""
        (SITE / out_name).write_text(page(meta["title"], out_name, content), encoding="utf-8")

    cards = ""
    for (out_name, _, _), meta in zip(POSTS, metas):
        cards += f"""<a class="card" href="{out_name}">
  <div class="thumb" style="background-image:url('{meta["image"].lstrip("/")}')"></div>
  <div class="body"><div class="date">{meta["date"][:10]} · {html.escape(meta["author"])}</div>
  <h2>{html.escape(meta["title"])}</h2><p>{html.escape(meta["summary"])}</p>
  <div class="go">阅读全文 →</div></div></a>"""
    cards += """<a class="card" href="report.html">
  <div class="thumb" style="background-image:url('assets/report/fig3.png');background-size:contain;background-color:#fff"></div>
  <div class="body"><div class="date">2026-09 · DeepSeek-AI</div>
  <h2>DeepSeek-V4.1-Flash 技术报告:把 KV 缓存压缩推向极限</h2>
  <p>DeepSeek 官方技术报告的全文中文翻译:CED 架构、CSA2、Single-Pass mHC、Engram、DSpark、SWA 有界重放,以及预训练与后训练。</p>
  <div class="go">阅读全文 →</div></div></a>"""
    cards += """<a class="card report" href="benchmarks.html">
  <div class="thumb"><span>44 个基准</span></div>
  <div class="body"><div class="date">调研报告 · 2026-11-10</div>
  <h2>DeepSeek 为什么测这些:技术报告里的评测基准</h2>
  <p>技术报告用到的每一个评测基准:测什么、现在处境如何、DeepSeek 拿它证明什么。并回答总问题:这套评测为什么这样搭。</p>
  <div class="go">打开报告 →</div></div></a>"""
    cards += """<a class="card report" href="agentx.html">
  <div class="thumb"><span>AgentX</span></div>
  <div class="body"><div class="date">深度调研 · 2026-11-10</div>
  <h2>AgentX 深度调研:vLLM 两篇博客唯一的考卷</h2>
  <p>数据集怎么来的、怎么重放、每个指标怎么算、成本怎么折算、目前的成绩与各方反应,以及它的局限和读数陷阱。</p>
  <div class="go">打开报告 →</div></div></a>"""
    cards += """<a class="card report" href="trace.html">
  <div class="thumb"><span>一条会话</span></div>
  <div class="body"><div class="date">图解 · 2026-11-10</div>
  <h2>AgentX 数据集中的一条会话:输入是怎么攒出来的</h2>
  <p>拿 AgentX 数据集里一条真实会话(29 轮、3 个子智能体)画图:每一轮的输入由哪些部分组成,哪些能走缓存,哪些必须重算。</p>
  <div class="go">打开图解 →</div></div></a>"""
    cards += """<a class="card report" href="flow.html">
  <div class="thumb"><span>测一条会话</span></div>
  <div class="body"><div class="date">图解 · 2026-11-10</div>
  <h2>AgentX 数据集中的一条会话:它是怎么被测的</h2>
  <p>不考虑并发,把这条会话的测试流程画出来:固定的配置、每次请求的往返、整条时间线,以及每一轮过后什么变了。</p>
  <div class="go">打开图解 →</div></div></a>"""
    cards += """<a class="card report" href="graph.html">
  <div class="thumb"><span>知识图谱</span></div>
  <div class="body"><div class="date">可视化 · 2026-11-10</div>
  <h2>公司、模型、硬件、引擎、技术、基准与指标</h2>
  <p>把站内材料里的七类实体连成一张可以拖动、筛选、搜索的关系图。</p>
  <div class="go">打开图谱 →</div></div></a>"""
    cards += """<a class="card report" href="research.html">
  <div class="thumb"><span>九个概念</span></div>
  <div class="body"><div class="date">调研报告 · 2026-11-10</div>
  <h2>读懂这两篇博客的九个概念</h2>
  <p>SemiAnalysis、AgentX、P90、DeepSeek-V4.1-Flash、DeepInfer、Inferact、GB300 NVL72、P/D 分离、Wide-EP:各是什么、彼此什么关系、在博客里扮演什么角色。</p>
  <div class="go">打开报告 →</div></div></a>"""

    c = "".join(f'<a class="chip" href="research.html#{a}">{n}</a>' for n, a in CONCEPTS)
    index = f"""<main class="wrap">
<div class="hero">
  <div class="eyebrow">本地中文读本</div>
  <h1>vLLM 如何把 DeepSeek-V4.1-Flash 跑到整机柜规模</h1>
  <p class="lead">vLLM 官方博客 2026 年 10 月的两篇连载的中文翻译,加上 DeepSeek 官方技术报告的中文翻译和一份名词调研报告。第一篇讲单节点上三周内的 5 倍吞吐提升,第二篇讲用 P/D 分离和 Wide-EP 扩展到一整个 GB300 NVL72 机柜。</p>
</div>
<div class="section-title">文章与报告</div>
<div class="cards">{cards}</div>
<div class="section-title">名词速查</div>
<div class="chips">{c}</div>
</main>"""
    (SITE / "index.html").write_text(
        page("vLLM × DeepSeek-V4.1-Flash 中文读本", "index.html", index), encoding="utf-8")
    print("built:", ", ".join(p[0] for p in POSTS), "index.html")


if __name__ == "__main__":
    build()
