#!/usr/bin/env python3
"""生成 site/trace.html:用一条真实会话画出每一轮输入 token 的组成。

用法:  python3 site/build_trace.py
数据:  site/assets/trace-example.json(从公开数据集 semianalysisai/cc-traces-weka-062126
        的一条会话推导出的逐轮数字,不含任何文本内容)
"""
import html
import json
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent
sys.path.insert(0, str(SITE))
from build import page  # noqa: E402

E = html.escape
D = json.loads((SITE / "assets" / "trace-example.json").read_text(encoding="utf-8"))
ROWS = D["rows"]
B = D["block"]


def fmt(n):
    return f"{n:,}"


def chart_layers():
    """图一:每一行是一次请求的完整输入,按"这段内容是哪一轮第一次出现的"分层着色。"""
    L, W, RH, GAP = 96, 640, 17, 5          # 左边距、条形最大宽度、行高、行距
    maxin = max(r["inp"] for r in ROWS)
    sx = W / maxin
    order = {}                                # 来源行 → 出现顺序,用来定深浅
    for r in ROWS:
        order.setdefault(r["label"], len(order))
    n_order = len(order)
    y = 34
    out = []
    # 顶部刻度
    for tick in range(0, maxin + 1, 40000):
        x = L + tick * sx
        out.append(f'<line class="grid" x1="{x:.1f}" y1="22" x2="{x:.1f}" y2="{{H}}"/>'
                   f'<text class="tick" x="{x:.1f}" y="16" text-anchor="middle">{tick // 1000}K</text>')
    prev_kind = None
    for r in ROWS:
        main = r["kind"] == "main"
        gap = r.get("gap") or 0
        if main and gap >= 60:
            capped = "(重放时压到 300 秒)" if gap > 300 else ""
            out.append(f'<text class="idle" x="{L}" y="{y + 9}">⏸ 停顿 {gap:.0f} 秒{capped}</text>')
            y += 18
        if r["kind"] == "sub" and prev_kind != "sub":
            out.append(f'<text class="subhead" x="{L}" y="{y + 9}">↳ 派出 3 个子智能体,同时在跑;主智能体等它们全部结束</text>')
            y += 18
        if main and prev_kind == "sub":
            y += 6
        x = L + (16 if not main else 0)
        lab = f'第 {r["label"]} 轮' if main else f'子 {r["label"]}'
        out.append(f'<text class="rlab{"" if main else " sub"}" x="{L - 8}" y="{y + RH - 5}" text-anchor="end">{lab}</text>')
        for origin, nb, kind in r["segs"]:
            w = nb * B * sx
            if kind == "new":
                cls, style = "seg new", ""
                tip = f"本轮新增 {fmt(nb * B)} token:必须计算"
            elif kind == "pre":
                depth = 0.22 + 0.5 * (order[origin] / max(1, n_order - 1))
                cls, style = "seg pre", f' style="fill-opacity:{depth:.2f}"'
                who = f"第 {origin} 轮" if not origin.startswith("S") else f"子智能体请求 {origin}"
                tip = f"{who}第一次出现的内容,{fmt(nb * B)} token:与之前逐字相同,可从缓存复用"
            else:
                cls, style = "seg seen", ""
                tip = f"之前出现过、但不在开头连续段里的内容,{fmt(nb * B)} token:前缀缓存用不上"
            out.append(f'<rect class="{cls}"{style} x="{x:.1f}" y="{y}" width="{max(w, 0.6):.1f}" height="{RH}"><title>{tip}</title></rect>')
            x += w
        # 输出:同一比例尺,画在输入条的右边
        ow = r["out"] * sx
        out.append(f'<rect class="seg outp" x="{x + 3:.1f}" y="{y}" width="{max(ow, 1.2):.1f}" height="{RH}"><title>输出 {fmt(r["out"])} token</title></rect>')
        new_tok = sum(nb for _, nb, k in r["segs"] if k == "new") * B
        out.append(f'<text class="num" x="{L + W + 26}" y="{y + RH - 5}">{fmt(r["inp"])}</text>'
                   f'<text class="num new" x="{L + W + 96}" y="{y + RH - 5}">+{fmt(new_tok)}</text>'
                   f'<text class="num outp" x="{L + W + 162}" y="{y + RH - 5}">{fmt(r["out"])}</text>')
        y += RH + GAP
        prev_kind = r["kind"]
    H = y + 6
    head = (f'<text class="colh" x="{L + W + 26}" y="16">输入</text>'
            f'<text class="colh" x="{L + W + 96}" y="16">新增</text>'
            f'<text class="colh" x="{L + W + 162}" y="16">输出</text>')
    body = "".join(out).replace("{H}", str(H - 4))
    return f'<svg class="tr" viewBox="0 0 {L + W + 220} {H}" role="img" aria-label="每一轮输入 token 的分层结构">{head}{body}</svg>'


def chart_new():
    """图二:只看每轮新增的那一小段,拆成"上一轮模型输出回填 / 子智能体交回的结果 / 工具结果与用户消息"。"""
    mains = [r for r in ROWS if r["kind"] == "main" and int(r["label"]) >= 3]
    L, W, RH, GAP = 96, 640, 15, 5
    data = []
    for r in mains:
        new = r["new"] * B
        a = min(r["prev_out"] or 0, new)                 # 上一轮输出回填
        s = min(r.get("sub_out") or 0, new - a)          # 子智能体交回的结果
        rest = new - a - s                               # 其余:工具结果、用户消息
        data.append((r, new, a, s, rest))
    mx = max(d[1] for d in data)
    sx = W / mx
    y, out = 30, []
    for tick in range(0, mx + 1, 4000):
        x = L + tick * sx
        out.append(f'<line class="grid" x1="{x:.1f}" y1="20" x2="{x:.1f}" y2="{{H}}"/>'
                   f'<text class="tick" x="{x:.1f}" y="14" text-anchor="middle">{tick // 1000}K</text>')
    for r, new, a, s, rest in data:
        out.append(f'<text class="rlab" x="{L - 8}" y="{y + RH - 4}" text-anchor="end">第 {r["label"]} 轮</text>')
        x = L
        for val, cls, name in ((a, "back", "上一轮模型输出回填"), (s, "subr", "子智能体交回的结果"), (rest, "tool", "工具结果或用户消息")):
            if val <= 0:
                continue
            w = val * sx
            out.append(f'<rect class="seg {cls}" x="{x:.1f}" y="{y}" width="{max(w, 0.8):.1f}" height="{RH}"><title>{name}:约 {fmt(val)} token</title></rect>')
            x += w
        out.append(f'<text class="num" x="{x + 6:.1f}" y="{y + RH - 4}">{fmt(new)}</text>')
        y += RH + GAP
    H = y + 4
    return f'<svg class="tr" viewBox="0 0 {L + W + 90} {H}" role="img" aria-label="每轮新增输入的组成">{"".join(out).replace("{H}", str(H - 2))}</svg>'


def build():
    mains = [r for r in ROWS if r["kind"] == "main"]
    tot_in = sum(r["inp"] for r in ROWS)
    tot_out = sum(r["out"] for r in ROWS)
    tot_pre = sum(r["pre"] for r in ROWS) * B
    content = f"""<style>
  :root {{ --t-pre: #0f6e63; --t-new: #c98a16; --t-out: #c2410c; --t-seen: #8a867e; --t-back: #c2410c; --t-subr: #6b4fbb; --t-tool: #c98a16; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --t-pre: #5cc5b6; --t-new: #e8b04a; --t-out: #fb923c; --t-seen: #85827b; --t-back: #fb923c; --t-subr: #b09cf0; --t-tool: #e8b04a; }} }}
  .trwrap {{ max-width: 1000px; margin: 0 auto; padding: 0 20px 90px; }}
  .trcard {{ background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 20px 20px 14px; box-shadow: var(--shadow); overflow-x: auto; margin: 14px 0 10px; }}
  svg.tr {{ width: 100%; min-width: 760px; height: auto; display: block; font-family: var(--sans); }}
  svg.tr .grid {{ stroke: var(--line); stroke-width: 1; }}
  svg.tr .tick, svg.tr .colh {{ font-size: 10.5px; fill: var(--ink-3); }}
  svg.tr .colh {{ font-weight: 600; }}
  svg.tr .rlab {{ font-size: 11.5px; fill: var(--ink-2); }}
  svg.tr .rlab.sub {{ fill: var(--ink-3); font-size: 10.5px; }}
  svg.tr .num {{ font-size: 10.5px; fill: var(--ink-2); font-variant-numeric: tabular-nums; }}
  svg.tr .num.new {{ fill: var(--t-new); }} svg.tr .num.outp {{ fill: var(--t-out); }}
  svg.tr .idle {{ font-size: 10.5px; fill: var(--ink-3); font-style: italic; }}
  svg.tr .subhead {{ font-size: 10.5px; fill: var(--t-subr); font-weight: 600; }}
  svg.tr .seg {{ stroke: var(--surface); stroke-width: .6; }}
  svg.tr .pre {{ fill: var(--t-pre); }} svg.tr .new {{ fill: var(--t-new); }} svg.tr .outp {{ fill: var(--t-out); }}
  svg.tr .seen {{ fill: var(--t-seen); }} svg.tr .back {{ fill: var(--t-back); }} svg.tr .subr {{ fill: var(--t-subr); }} svg.tr .tool {{ fill: var(--t-tool); }}
  .lg {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13.5px; color: var(--ink-2); margin: 4px 0 0; }}
  .lg span {{ display: inline-flex; align-items: center; gap: 6px; }}
  .lg i {{ width: 22px; height: 11px; border-radius: 2px; display: inline-block; }}
  .trwrap h2 {{ font-size: 22px; margin: 44px 0 6px; }}
  .trwrap p, .trwrap li {{ color: var(--ink-2); max-width: 820px; }}
  .trwrap .note {{ font-size: 13.5px; color: var(--ink-3); }}
</style>
<main class="trwrap">
<div class="hero" style="padding-bottom:10px">
  <div class="eyebrow">图解</div>
  <h1>AgentX 数据集中的一条会话:{len(mains)} 轮的输入 token 是怎么攒出来的</h1>
  <p class="lead">取 AgentX 公开数据集里的一条真实会话(编号 {E(D["id"])}…,全程约 42 分钟)。数据里没有任何文字,只有每一轮的 token 数和"哪些内容块之前出现过"。把它画出来,就能看到输入是一层一层叠上去的。</p>
  <div class="meta"><span>合计输入 {fmt(tot_in)} · 输出 {fmt(tot_out)} · 比例约 {round(tot_in / tot_out)}:1</span><span>输入里 {tot_pre / tot_in * 100:.0f}% 是可复用的前缀</span><a href="flow.html">这条会话是怎么被测的 →</a><a href="agentx.html#replay">AgentX 的重放机制 →</a></div>
</div>

<h2>图一:每一轮的完整输入</h2>
<p>每一行是一次请求递给模型的全部输入,长度按 token 数画。颜色表示这段内容是哪一轮第一次出现的:越浅越早,越深越晚。最右边金色的一小截是这一轮新加的。</p>
<div class="lg">
  <span><i style="background:var(--t-pre);opacity:.3"></i>较早的轮次带进来的内容(可复用)</span>
  <span><i style="background:var(--t-pre);opacity:.75"></i>较晚的轮次带进来的内容(可复用)</span>
  <span><i style="background:var(--t-new)"></i>本轮新增(必须计算)</span>
  <span><i style="background:var(--t-out)"></i>本轮输出(同一比例尺)</span>
</div>
<div class="trcard">{chart_layers()}</div>
<p class="note">把鼠标停在任一色块上可以看它的来源和 token 数。条形右侧三列数字依次是:输入总量、其中新增的部分、输出。</p>

<h3 style="margin-top:26px">从图上能读出什么</h3>
<ul>
  <li><b>输入像年轮一样只增不减。</b>第 3 轮的输入 = 第 2 轮的全部输入 + 一小截新内容;第 4 轮 = 第 3 轮的全部 + 又一小截。到第 29 轮,最左边那一大段仍然是第 2 轮带进来的那 {fmt(ROWS[1]["inp"])} 个 token。</li>
  <li><b>第 2 轮的那一大块是"开场白"。</b>它一次带进来 6.4 万 token,之后每一轮都原样重复。按 Claude Code 的工作方式,这应该是系统提示、工具定义和最初的任务说明(这一点是推断,数据里看不到内容)。</li>
  <li><b>输出几乎看不见。</b>橙色的输出条和输入用的是同一把尺子。输入十几万,输出几百到三千,所以只是一条细线。</li>
  <li><b>子智能体是另起一摞。</b>三个子智能体不继承主智能体的上下文,各自从约 3.5 万 token 起步。第一个要从头算;第二、第三个的开头有 536 块与第一个逐字相同,可以复用。</li>
  <li><b>子智能体回来后,主智能体那一轮新增特别多。</b>第 6 轮新增 16,000 token,因为三个子智能体的结果要写回主对话。</li>
  <li><b>停顿之后缓存还在不在,决定这一轮贵不贵。</b>第 7 轮之前停了 433 秒。如果服务器把这条会话的缓存挤掉了,这一轮就不是算 2,400 个新 token,而是把 10 万 token 全部重算。</li>
</ul>

<h2>图二:放大看每轮新增的那一小截</h2>
<p>图一里金色的部分太窄,这里单独放大。每轮新增的输入可以拆成三种来源:</p>
<div class="lg">
  <span><i style="background:var(--t-back)"></i>上一轮模型自己的输出,被写回对话历史</span>
  <span><i style="background:var(--t-subr)"></i>子智能体交回的结果</span>
  <span><i style="background:var(--t-tool)"></i>工具返回的结果,或用户的新消息</span>
</div>
<div class="trcard">{chart_new()}</div>
<p class="note">这张图的拆分是估算:数据里只有新增块的总数,没有标注每块是什么。"上一轮输出回填"按上一轮的输出 token 数计,"子智能体结果"按各子智能体最后一次请求的输出计,剩下的归为工具结果或用户消息。第 6 轮的三段加起来与实际新增量对得上,第 17 轮的新增量(1,920)与第 16 轮的输出(1,914)几乎相等,说明这个拆法大体成立。</p>

<h3 style="margin-top:26px">这和评测有什么关系</h3>
<ul>
  <li><b>绿色的部分考的是记性。</b>服务器能把它留在缓存里,这一轮就只需要处理金色那一小截。留不住,整条都要重算。AgentX 的成绩差距主要来自这里。</li>
  <li><b>金色的部分考的是读得快不快。</b>也就是预填充速度。</li>
  <li><b>橙色的部分考的是写得快不快。</b>也就是解码速度,对应"每个用户每秒收到多少 token"。</li>
  <li><b>默认的吞吐数字是把整条都算进去的。</b>这条会话完整放一遍,计入吞吐的是 {fmt(tot_in + tot_out)} 个 token,其中真正需要计算的新增输入加输出只占一成左右。</li>
  <li><b>重放时,橙色和金色之间的联系被切断了。</b>真实对话里,这一轮的橙色输出会成为下一轮金色新增的一部分。AgentX 重放时,下一轮写回历史的是事先准备好的填充文字,不是被测模型刚写出来的内容,所以每个被测系统拿到的输入完全相同。</li>
</ul>
</main>"""
    (SITE / "trace.html").write_text(page("AgentX 数据集中的一条会话:输入是怎么攒出来的", "agentx.html", content), encoding="utf-8")
    print("built: trace.html", f"({len(ROWS)} 行)")


if __name__ == "__main__":
    build()
