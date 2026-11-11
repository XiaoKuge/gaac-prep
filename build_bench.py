#!/usr/bin/env python3
"""生成 site/benchmarks.html:技术报告里每个评测基准的调研卡片 + "为什么做这些评测"的分析。

用法:  python3 site/build_bench.py
输入:  ../tech_report/bench/G1.json … G4.json(逐基准调研结果)
        本文件里的 USAGE(DeepSeek 在报告里怎么用、得了多少分)和 INTRO(分析正文)
"""
import html
import json
import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent
BENCH = SITE.parent / "tech_report" / "bench"
sys.path.insert(0, str(SITE))
from build import page  # noqa: E402

E = html.escape

# ---------------------------------------------------------------------------
# DeepSeek 在技术报告里对每个基准的用法与分数(取自报告表 1、表 3、表 4、图 7–12)
# base3 = (V4-Flash-Base, V4-Pro-Base, V4.1-Flash-Base);final = 表 3 一行
# ---------------------------------------------------------------------------
T3_COLS = ["Opus-5", "GPT-5.6 Sol", "K3", "GLM-5.3", "DS-V4-Pro", "DS-V4-Flash", "DS-V4.1-Flash"]

USAGE = {
    # --- 预训练 / 基座模型(表 1) ---
    "AGIEval": dict(stage="base", shots="3–5-shot", metric="EM", base3=("83.9", "84.4", "83.4")),
    "MMLU-Pro": dict(stage="base", shots="5-shot", metric="EM", base3=("68.3", "73.5", "74.1")),
    "C-Eval": dict(stage="base", shots="5-shot", metric="EM", base3=("92.1", "93.1", "92.1")),
    "MultiLoKo": dict(stage="base", shots="5-shot", metric="LLM-Judge", base3=("42.6", "50.9", "45.5")),
    "SimpleQA-Verified": dict(stage="base", shots="25-shot", metric="EM", base3=("30.1", "55.2", "42.3"),
                              extra="同时是推理强度实验八基准之一。"),
    "SuperGPQA": dict(stage="base", shots="5-shot", metric="EM", base3=("46.5", "53.9", "53.1")),
    "BBH": dict(stage="base", shots="3-shot", metric="EM", base3=("86.9", "87.5", "86.1")),
    "BBEH": dict(stage="base", shots="1-shot", metric="EM", base3=("25.4", "29.8", "27.2")),
    "DROP": dict(stage="base", shots="1-shot", metric="F1", base3=("88.6", "88.7", "87.9")),
    "HellaSwag": dict(stage="base", shots="0-shot", metric="EM", base3=("85.7", "88.0", "87.2")),
    "BigCodeBench": dict(stage="base", shots="3-shot", metric="Pass@1", base3=("56.8", "59.2", "60.6")),
    "HumanEval": dict(stage="base", shots="0-shot", metric="Pass@1", base3=("69.5", "76.8", "79.4")),
    "GSM8K": dict(stage="base", shots="8-shot", metric="EM", base3=("90.8", "92.6", "93.0")),
    "MATH": dict(stage="base", shots="4-shot", metric="EM", base3=("57.4", "64.5", "61.1")),
    "MGSM": dict(stage="base", shots="8-shot", metric="EM", base3=("85.7", "84.4", "80.2")),
    "LongBench-V2": dict(stage="base", shots="1-shot", metric="EM", base3=("44.7", "51.5", "45.2")),
    "MMMU-Pro": dict(stage="base", shots="4-shot", metric="EM", base3=("-", "-", "56.5")),
    "CVBench": dict(stage="base", shots="4-shot", metric="EM", base3=("-", "-", "77.9")),
    "DocVQA": dict(stage="base", shots="4-shot", metric="LLM-Judge", base3=("-", "-", "95.6")),
    "RefCOCO": dict(stage="base", shots="0-shot", metric="Acc@0.5(三个子集平均)", base3=("-", "-", "86.0")),
    # --- 后训练 / 最终模型:推理(表 3) ---
    "GPQA Diamond": dict(stage="reason", metric="Pass@1", final=("93.4", "94.1", "92.9", "88.1", "92.4", "89.9", "90.9"),
                         extra="推理强度实验八基准之一。"),
    "HLE": dict(stage="reason", metric="Pass@1", final=("56.3", "44.5", "43.5", "42.0†", "42.7†", "37.8†", "36.8(39.1†)"),
                extra="† 为纯文本子集。另有“HLE w/ tools”(带工具)一行:63.6 / - / 59.8 / 62.5 / 60.0 / 51.5 / 63.9,DS-V4.1-Flash 最高。推理强度实验八基准之一。"),
    "Codeforces": dict(stage="reason", metric="Rating(内部基准)", final=("-", "-", "-", "-", "3348", "3289", "3471")),
    "MathArena Apex": dict(stage="reason", metric="Pass@1", final=("-", "-", "65.6", "-", "65.3", "58.6", "65.6"),
                           extra="推理强度实验八基准之一。"),
    "Apex 2025 Shortlist": dict(stage="effort", extra="只出现在推理强度实验(图 9、图 12)的八基准平均里,报告未单列分数。"),
    "AIME 2026": dict(stage="effort", extra="只出现在推理强度实验(图 9、图 12)的八基准平均里,报告未单列分数。"),
    "IMO-AnswerBench": dict(stage="effort", extra="只出现在推理强度实验(图 9、图 12)的八基准平均里,报告未单列分数。"),
    "LiveCodeBench": dict(stage="effort", extra="只出现在推理强度实验(图 9、图 12)的八基准平均里,报告未单列分数。"),
    # --- 代码智能体 ---
    "Terminal-Bench 2.1": dict(stage="code", metric="Pass@1", final=("89.1", "88.8", "88.3", "88.2", "87.9", "82.7", "90.6"),
                               extra="还用于:RL 训练进度曲线(图 7)、推理强度曲线(82.4→90.6)、八种脚手架对比(表 4:84.1–90.6)。无网络访问。"),
    "Terminal-Bench 3.0": dict(stage="code", metric="Pass@1", final=("43.3", "34.4", "17.7", "28.3", "11.8", "7.6", "30.0"),
                               extra="RL 训练进度曲线(图 7)用的是它的 no-gpu 子集。"),
    "Terminal-Bench 4.0": dict(stage="code", metric="Pass@1", final=("51.8", "39.9", "12.6", "37.9", "12.4", "7.0", "31.2"),
                               extra="报告在引言里点名它,承认在需要专家级领域知识的科学类智能体任务上与巨型模型仍有差距。"),
    "DeepSWE v1.1": dict(stage="code", metric="Resolved", final=("74.0", "73.0", "67.5", "66.9", "62.7", "54.4", "74.2"),
                         extra="按官方要求用 mini-SWE 脚手架。还用于:RL 训练进度曲线、推理强度曲线(66.0→74.2)、八种脚手架对比(表 4:65.5–74.2)。"),
    "ProgramBench": dict(stage="code", metric="Almost@1", final=("37.0", "23.0", "17.5", "19.0", "15.5", "-", "20.3"),
                         extra="多智能体实验(图 10)用其 172 题“黄金”子集:1–12 小时期限下,多智能体峰值 30.04%,单智能体 20.39%。"),
    "NL2Repo-Bench": dict(stage="code", metric="Score", final=("75.3", "56.8", "58.0", "58.0", "61.5", "54.2", "65.4")),
    "SWE-Bench Pro": dict(stage="monitor", extra="只出现在 RL 训练进度曲线(图 7),用来观察能力随 RL 步数的变化,未进入最终对比表。"),
    "FrontierSWE v2": dict(stage="multi", metric="Mean@5",
                           extra="只用于多智能体实验(图 10),取公开任务里不需要 GPU 的子集:1–20 小时期限下,多智能体从 13.50% 升到 32.90%,单智能体从 10.50% 升到 28.20%。"),
    # --- 安全 ---
    "CyberGym": dict(stage="sec", metric="Pass@1", final=("-", "84.5", "80.0", "84.5", "83.3", "76.7", "88.1"),
                     extra="报告提到评测中观察到模型“反编译 Ubuntu 核心包找漏洞”这类钻空子行为。"),
    "SEC-Bench Pro": dict(stage="sec", metric="Pass@1", final=("-", "74.3", "-", "-", "56.4", "30.9", "62.8"),
                          extra="使用 Claude Code 脚手架(看中其会话压缩设计),版本 260505。"),
    "ExploitGym": dict(stage="sec", metric="Pass@1", final=("22.1", "33.7", "-", "15.0", "5.4", "1.8", "15.3")),
    # --- 通用智能体 ---
    "AutomationBench": dict(stage="general", metric="Pass@1", final=("50.3", "45.8", "46.7", "48.8", "43.2", "37.7", "54.8"),
                            extra="v1.0.6 公开评测集,使用官方脚手架。"),
    "Agents' Last Exam": dict(stage="general", metric="Pass@1", final=("28.6", "26.7", "27.6", "28.5", "25.7", "25.2", "31.8"),
                              extra="ALE-CLI,使用官方脚手架。"),
    # --- 视觉智能体 ---
    "Chartography": dict(stage="visual", metric="Pass@1(带工具)", final=("84.0", "79.9", "68.1", "-", "-", "-", "78.9")),
    "BabyVision": dict(stage="visual", metric="Pass@1(带工具)", final=("94.1", "88.9", "85.7", "-", "-", "-", "89.6")),
    "ZeroBench": dict(stage="visual", metric="Pass@5(main 集,带工具)", final=("52.0", "53.0", "41.0", "-", "-", "-", "49.0")),
}

GROUPS = [
    ("base-know", "基座模型 · 世界知识", "预训练结束后、后训练之前,对基座模型做少样本测试。",
     ["AGIEval", "MMLU-Pro", "C-Eval", "MultiLoKo", "SimpleQA-Verified", "SuperGPQA"]),
    ("base-reason", "基座模型 · 语言理解与推理", "", ["BBH", "BBEH", "DROP", "HellaSwag"]),
    ("base-code", "基座模型 · 代码与数学", "", ["BigCodeBench", "HumanEval", "GSM8K", "MATH", "MGSM"]),
    ("base-long", "基座模型 · 长上下文与多模态", "多模态四项只有 V4.1-Flash 有分数,因为前两代基座不是多模态模型。",
     ["LongBench-V2", "MMMU-Pro", "CVBench", "DocVQA", "RefCOCO"]),
    ("reason", "最终模型 · 推理", "后训练之后,用最高推理强度(Max)与闭源、开源旗舰对比。",
     ["GPQA Diamond", "HLE", "Codeforces", "MathArena Apex"]),
    ("effort", "推理强度实验的补充基准", "只在“推理强度 25→100”的成本-精度曲线里出现。",
     ["Apex 2025 Shortlist", "AIME 2026", "IMO-AnswerBench", "LiveCodeBench"]),
    ("code", "最终模型 · 代码智能体", "",
     ["Terminal-Bench 2.1", "Terminal-Bench 3.0", "Terminal-Bench 4.0", "DeepSWE v1.1", "ProgramBench",
      "NL2Repo-Bench", "SWE-Bench Pro", "FrontierSWE v2"]),
    ("sec", "最终模型 · 网络安全", "", ["CyberGym", "SEC-Bench Pro", "ExploitGym"]),
    ("general", "最终模型 · 通用智能体", "", ["AutomationBench", "Agents' Last Exam"]),
    ("visual", "最终模型 · 视觉智能体", "", ["Chartography", "BabyVision", "ZeroBench"]),
]

# 调研结果里的名字 → 本页用的规范名
ALIASES = {
    "bbh (bigbench hard)": "BBH", "bigbench hard": "BBH", "bbeh (bigbench extra hard)": "BBEH",
    "bigbench extra hard": "BBEH", "cv-bench": "CVBench", "cvbench (cv-bench)": "CVBench",
    "refcoco / refcoco+ / refcocog": "RefCOCO", "refcoco/refcoco+/refcocog": "RefCOCO",
    "humanity's last exam": "HLE", "humanity's last exam (hle)": "HLE", "hle (humanity's last exam)": "HLE",
    "codeforces rating": "Codeforces", "matharena-apex": "MathArena Apex", "swe-bench pro": "SWE-Bench Pro",
    "automation-bench": "AutomationBench", "agents' last exam (ale)": "Agents' Last Exam",
    "zerobench (main)": "ZeroBench", "simpleqa verified": "SimpleQA-Verified", "longbench v2": "LongBench-V2",
    "refcoco": "RefCOCO", "sec-bench pro": "SEC-Bench Pro",
}


def norm(name):
    k = name.strip().lower().replace("’", "'")
    if k in ALIASES:
        return ALIASES[k]
    for canon in USAGE:
        if canon.lower() == k:
            return canon
    for canon in USAGE:  # 宽松匹配:去掉括号说明
        if re.sub(r"\s*\(.*?\)", "", k) == canon.lower():
            return canon
    return name.strip()


def load_research():
    out = {}
    for f in sorted(BENCH.glob("G*.json")):
        for item in json.loads(f.read_text(encoding="utf-8")):
            out[norm(item["name"])] = item
    return out


def slug(name):
    return "b-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def score_block(name):
    u = USAGE[name]
    parts = []
    if "base3" in u:
        a, b, c = u["base3"]
        parts.append(
            '<div class="table-scroll"><table class="scores"><tr><th>设置</th><th>V4-Flash-Base</th>'
            '<th>V4-Pro-Base</th><th class="me">V4.1-Flash-Base</th></tr>'
            f'<tr><td>{E(u["shots"])} · {E(u["metric"])}</td><td>{a}</td><td>{b}</td><td class="me">{c}</td></tr></table></div>')
    if "final" in u:
        vals = u["final"]
        nums = []
        for v in vals:
            m = re.match(r"[\d.]+", v)
            nums.append(float(m.group(0)) if m else None)
        best = max(n for n in nums if n is not None)
        head = "".join(f'<th{" class=\"me\"" if c == "DS-V4.1-Flash" else ""}>{c}</th>' for c in T3_COLS)
        cells = ""
        for c, v, n in zip(T3_COLS, vals, nums):
            cls = []
            if c == "DS-V4.1-Flash":
                cls.append("me")
            if n is not None and n == best:
                cls.append("best")
            cells += f'<td{" class=\"" + " ".join(cls) + "\"" if cls else ""}>{E(v)}</td>'
        parts.append(
            f'<div class="table-scroll"><table class="scores"><tr><th>{E(u["metric"])}</th>{head}</tr>'
            f"<tr><td>Max 推理强度</td>{cells}</tr></table></div>")
    if u.get("extra"):
        parts.append(f'<p class="ds-extra">{E(u["extra"])}</p>')
    return "".join(parts)


def card(name, r):
    u = USAGE[name]
    if r is None:
        body = '<p class="caveat">这一项的调研结果缺失。</p>'
        meta = ""
    else:
        conf = r.get("confidence", "")
        conf_label = {"high": "把握高", "medium": "把握中", "low": "把握低"}.get(conf, "")
        meta = " · ".join(x for x in [E(r.get("year", "")), E(r.get("org", ""))] if x)
        rows = [("测什么", r.get("measures")), ("题目形式", r.get("format")), ("指标", r.get("metric")),
                ("2026 年的处境", r.get("status")), ("业界为什么用", r.get("why_used")), ("补充", r.get("notes"))]
        dl = "".join(f"<dt>{k}</dt><dd>{E(v)}</dd>" for k, v in rows if v)
        srcs = " · ".join(
            f'<a href="{E(s["url"])}" target="_blank" rel="noopener">{E(s["title"])}</a>'
            for s in r.get("sources", []) if s.get("url"))
        cav = f'<p class="caveat">{conf_label}:{E(r["caveat"])}</p>' if r.get("caveat") else ""
        full = r.get("full_name", "")
        full_html = f'<div class="full">{E(full)}</div>' if full and full.lower() != name.lower() else ""
        body = f'{full_html}<dl class="facts">{dl}</dl>{cav}<p class="src">来源:{srcs}</p>'
    return (f'<section class="concept bench" id="{slug(name)}"><div class="bhead"><h3>{E(name)}</h3>'
            f'<span class="bmeta">{meta}</span></div>'
            f'<div class="inblog"><b>DeepSeek 怎么用:</b>{score_block(name)}</div>{body}</section>')


INTRO = (SITE / "bench_intro.html")


def build():
    research = load_research()
    known = {n for _, _, _, names in GROUPS for n in names}
    missing = sorted(known - set(research))
    extra = sorted(set(research) - known)
    if missing:
        print("缺调研结果:", missing, file=sys.stderr)
    if extra:
        print("未归组的调研条目:", extra, file=sys.stderr)

    toc = ['<a href="#why">为什么做这些评测</a>', '<a href="#map">全景表</a>']
    sections = []
    for gid, title, note, names in GROUPS:
        toc.append(f'<a href="#{gid}">{title}</a>')
        toc += [f'<a class="sub" href="#{slug(n)}">{E(n)}</a>' for n in names]
        note_html = f'<p class="gnote">{E(note)}</p>' if note else ""
        cards = "".join(card(n, research.get(n)) for n in names)
        sections.append(f'<h2 class="gtitle" id="{gid}">{title}</h2>{note_html}{cards}')
    toc.append('<a href="#caveats">读数提醒</a>')

    n_total = len(known)
    intro, _, tail = INTRO.read_text(encoding="utf-8").replace("{{N}}", str(n_total)).partition("<!--SPLIT-->")
    content = f"""<main class="wrap">
<div class="hero">
  <div class="eyebrow">调研报告</div>
  <h1>DeepSeek 为什么测这些:技术报告里的 {n_total} 个评测基准</h1>
  <p class="lead">DeepSeek-V4.1-Flash 技术报告用了 {n_total} 个评测基准。本报告逐个说明它测什么、现在处境如何、DeepSeek 拿它证明什么,并回答一个总问题:这套评测为什么这样搭。</p>
  <div class="meta"><span>调研日期 2026-11-10</span><a href="report.html#预训练">技术报告 4.3 节 →</a><a href="report.html#后训练">技术报告 5.3 节 →</a></div>
</div>
<div class="report-grid">
<aside class="toc toc-long"><div class="t">目录</div>
{chr(10).join(toc)}
</aside>
<div class="report">
{intro}
{"".join(sections)}
{tail}
</div>
</div>
</main>"""
    (SITE / "benchmarks.html").write_text(
        page("DeepSeek 为什么测这些 · 评测基准调研", "benchmarks.html", content), encoding="utf-8")
    print(f"built: benchmarks.html ({len(research)} 条调研 / {n_total} 个基准)")


if __name__ == "__main__":
    build()
