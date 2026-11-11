#!/usr/bin/env python3
"""把技术报告的中文译稿(../tech_report/chunks/*_zh.md)生成 site/report.html。

用法:  python3 site/build_report.py          # 用中文分块
        python3 site/build_report.py --en     # 用英文分块(调试排版用)
依赖:  pandoc(负责 Markdown→HTML,并把 LaTeX 公式转成浏览器原生 MathML,离线可看)
"""
import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent
REPORT = SITE.parent / "tech_report"
CHUNKS = REPORT / "chunks"
OUT_ASSETS = SITE / "assets" / "report"

sys.path.insert(0, str(SITE))
from build import page  # noqa: E402  复用站点的导航与页面骨架

TITLE = "DeepSeek-V4.1-Flash:把 KV 缓存压缩推向极限"
PDF_NAME = "DeepSeek_V41_Tech_Report.pdf"


def load(lang):
    parts = []
    for name in "ABC":
        parts.append((CHUNKS / f"{name}_{lang}.md").read_text(encoding="utf-8"))
    refs = (CHUNKS / "refs_en.md").read_text(encoding="utf-8")
    refs = re.sub(r"^## References", "## 参考文献", refs, flags=re.M)
    parts.append(refs)
    parts.append((CHUNKS / f"D_{lang}.md").read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def pre(md):
    # 去掉 arXiv 版里的占位图片行,图统一用从 PDF 裁出的版本
    md = re.sub(r"^!\[[^\]]*\]\([^)]*\)\s*$", "", md, flags=re.M)
    # 站内不存在的锚点(文献、公式、图表交叉引用)去掉链接,只留文字
    md = re.sub(r"\[([^\]]+)\]\(#[^)]*\)", r"\1", md)
    # 标题和作者行由页面头部展示
    md = re.sub(r"\A.*?(?=^###### )", "", md, count=1, flags=re.S | re.M)
    return md


def to_html(md):
    r = subprocess.run(
        ["pandoc", "-f", "markdown+east_asian_line_breaks", "-t", "html", "--mathml", "--wrap=none"],
        input=md, capture_output=True, text=True, check=True)
    if r.stderr.strip():
        print("pandoc warnings:\n" + r.stderr.strip()[:2000], file=sys.stderr)
    return r.stdout


def post(h):
    # 图表标题
    def cap(m):
        kind, num, rest = m.group(1), m.group(2), m.group(3)
        label = f"{kind} {num}"
        if kind == "图":
            return (f'<figure id="fig{num}"><img src="assets/report/fig{num}.png" alt="{label}" loading="lazy">'
                    f"<figcaption><b>{label}</b>{rest}</figcaption></figure>")
        return f'<p class="caption"><b>{label}</b>{rest}</p>'
    h = re.sub(r"<p>CAPTION:: ?(图|表|算法|Figure|Table|Algorithm) ?(\d+)(.*?)</p>", cap, h, flags=re.S)

    # 公式表(无表头且含公式)与数据表分开处理
    def table(m):
        t = m.group(0)
        if "<th" not in t and "<math" in t:
            return '<div class="eqn">' + t + "</div>"
        return '<div class="table-scroll">' + t + "</div>"
    h = re.sub(r"<table.*?</table>", table, h, flags=re.S)
    h = re.sub(r"<colgroup>.*?</colgroup>", "", h, flags=re.S)
    h = re.sub(r'<a href="(https?://[^"]+)"', r'<a href="\1" target="_blank" rel="noopener"', h)
    return h


def toc(h):
    items = []
    for m in re.finditer(r'<h([23]) id="([^"]+)">(.*?)</h\1>', h, flags=re.S):
        text = re.sub(r"<[^>]+>", "", m.group(3)).strip()
        cls = ' class="sub"' if m.group(1) == "3" else ""
        items.append(f'<a href="#{m.group(2)}"{cls}>{text}</a>')
    return "\n".join(items)


def build(lang="zh"):
    OUT_ASSETS.mkdir(parents=True, exist_ok=True)
    for f in (REPORT / "figs").glob("fig*.png"):
        shutil.copy2(f, OUT_ASSETS / f.name)
    shutil.copy2(REPORT / PDF_NAME, OUT_ASSETS / PDF_NAME)

    body = post(to_html(pre(load(lang))))
    content = f"""<main class="wrap">
<div class="hero">
  <div class="eyebrow">DeepSeek 技术报告 · 中文翻译</div>
  <h1>{html.escape(TITLE)}</h1>
  <p class="lead">DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression 的全文中文翻译。参考文献保留英文原文。</p>
  <div class="meta"><span>DeepSeek-AI</span><span>2026-09</span>
  <a href="assets/report/{PDF_NAME}" target="_blank">原文 PDF(本地,51 页)</a>
  <a href="https://arxiv.org/abs/2609.19969" target="_blank" rel="noopener">arXiv:2609.19969 ↗</a>
  <a href="research.html#deepseek-v41-flash">调研报告里的模型简介 →</a></div>
</div>
<div class="report-grid">
<aside class="toc toc-long"><div class="t">目录</div>
{toc(body)}
</aside>
<article class="article paper">
<blockquote><p>译注:本页由 DeepSeek 官方技术报告翻译而来,公式、图、表中的数字均取自原文;图为从原文 PDF 裁出的原图,图中文字未翻译。译文若与原文有出入,以原文为准。</p></blockquote>
{body}
</article>
</div>
</main>"""
    (SITE / "report.html").write_text(page(TITLE + " · 技术报告中文版", "report.html", content), encoding="utf-8")
    print("built: report.html", f"({lang})")


if __name__ == "__main__":
    build("en" if "--en" in sys.argv else "zh")
