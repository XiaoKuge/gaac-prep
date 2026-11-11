#!/usr/bin/env python3
"""生成 site/flow.html:一条会话在 AgentX 里是怎么被测的(不考虑并发)。

用法:  python3 site/build_flow.py
数据:  site/assets/trace-example.json(与 trace.html 同一条会话)
依据:  agentx_research/N1_methodology.md、N2_metrics.md(方法文档的整理,未在真机上跑过)
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
CAP = 300          # 单棵会话树的空闲上限(秒)


def fmt(n):
    return f"{n:,}"


def name(r):
    return f'第 {r["label"]} 轮' if r["kind"] == "main" else f'子 {r["label"]}'


# ---------- 图二:一次请求的往返(时序图) ----------
def chart_sequence(r, prev, nxt):
    pre, new = r["pre"] * B, r["inp"] - r["pre"] * B
    XH, XS, Wd = 300, 640, 940
    y = 64
    out = []

    def note(side, lines, yy, cls=""):
        x, anchor = (XH - 18, "end") if side == "h" else (XS + 18, "start")
        cx = XH if side == "h" else XS
        s = f'<circle class="dot {cls}" cx="{cx}" cy="{yy + 4}" r="5"/>'
        for i, t in enumerate(lines):
            c = "nt b" if i == 0 else "nt"
            s += f'<text class="{c}" x="{x}" y="{yy + 8 + i * 17}" text-anchor="{anchor}">{t}</text>'
        return s

    def msg(direction, title, sub, yy, cls=""):
        x1, x2 = (XH + 8, XS - 8) if direction == "r" else (XS - 8, XH + 8)
        s = (f'<line class="arr {cls}" x1="{x1}" y1="{yy}" x2="{x2}" y2="{yy}" marker-end="url(#ah)"/>'
             f'<text class="mt b" x="{(XH + XS) / 2}" y="{yy - 9}" text-anchor="middle">{title}</text>')
        if sub:
            s += f'<text class="mt" x="{(XH + XS) / 2}" y="{yy + 17}" text-anchor="middle">{sub}</text>'
        return s

    out.append(note("h", [f"① 读剧本的这一行", f"输入 {fmt(r['inp'])} · 输出 {fmt(r['out'])}",
                          f"{fmt(r['blocks'])} 个块编号 · 间隔 {r['gap']:.1f} 秒"], y)); y += 74
    out.append(note("h", [f"② 等 {r['gap']:.1f} 秒", f"从{name(prev)}收完那一刻算起"], y)); y += 58
    out.append(note("h", ["③ 按块编号拼出文字", f"前 {fmt(r['pre'])} 块与{name(prev)}逐字相同",
                          f"后 {fmt(r['blocks'] - r['pre'])} 块是这一轮新出现的"], y)); y += 80
    out.append(msg("r", "④ 发出请求,开始计时",
                   f"{fmt(r['inp'])} token 输入 · max_tokens={r['out']} · ignore_eos=true · stream=true", y)); y += 50
    out.append(note("s", ["⑤ 查前缀缓存", f"开头 {fmt(pre)} token 见过",
                          "直接复用,不用重算"], y, "pre")); y += 74
    out.append(note("s", ["⑥ 预填充", f"只读新增的 {fmt(new)} token"], y, "new")); y += 62
    out.append(msg("l", "⑦ 第一个 token 到", "记下「首 token 时间」", y)); y += 50
    out.append(note("s", ["⑧ 解码", f"一个一个写,写满 {fmt(r['out'])} 个",
                          "遇到结束符也不许停"], y, "outp"))
    for k in range(4):
        yy = y + 2 + k * 13
        out.append(f'<line class="arr thin" x1="{XS - 8}" y1="{yy}" x2="{XH + 8}" y2="{yy}" marker-end="url(#ah)"/>')
    out.append(f'<text class="mt" x="{(XH + XS) / 2}" y="{y + 62}" text-anchor="middle">流式返回,一小批一小批地到</text>')
    y += 96
    out.append(msg("l", "⑨ 写完,附带 token 计数", "记下「结束时间」和服务器报的输入、输出 token 数", y)); y += 52
    out.append(note("h", ["⑩ 扔掉文字,留下数字", "首 token 时间 · 总耗时", "输入 token 数 · 输出 token 数"], y))
    out.append(note("s", ["缓存里现在留着", f"{fmt(r['inp'])} token 的前缀", "下一轮可以接着用"], y, "pre")); y += 80
    out.append(note("h", [f"⑪ 读下一行({name(nxt)})", f"等 {nxt['gap']:.1f} 秒,再来一遍"], y)); y += 50
    H = y
    head = (f'<rect class="hd" x="{XH - 100}" y="8" width="200" height="34" rx="8"/>'
            f'<text class="hdt" x="{XH}" y="30" text-anchor="middle">压测程序</text>'
            f'<rect class="hd" x="{XS - 100}" y="8" width="200" height="34" rx="8"/>'
            f'<text class="hdt" x="{XS}" y="30" text-anchor="middle">被测系统</text>'
            f'<line class="life" x1="{XH}" y1="42" x2="{XH}" y2="{H}"/>'
            f'<line class="life" x1="{XS}" y1="42" x2="{XS}" y2="{H}"/>')
    defs = '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="ahp" d="M0,0 L10,5 L0,10 z"/></marker></defs>'
    return f'<svg class="fl" viewBox="0 0 {Wd} {H}" role="img" aria-label="一次请求的往返过程">{defs}{head}{"".join(out)}</svg>'


# ---------- 图三:整条会话的时间线 ----------
def chart_timeline():
    L, W = 78, 840
    tmax = max(r["t"] + r["api"] for r in ROWS)
    sx = W / tmax
    lanes = {"main": 0, "S1": 1, "S2": 2, "S3": 3}
    LH, top = 26, 58
    out = []
    for m in range(0, int(tmax // 60) + 1, 5):
        x = L + m * 60 * sx
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{top + 4 * LH}"/>'
                   f'<text class="tick" x="{x:.1f}" y="{top + 4 * LH + 14}" text-anchor="middle">{m} 分</text>')
    for key, i in lanes.items():
        lab = "主智能体" if key == "main" else f"子智能体 {key[1]}"
        out.append(f'<text class="rlab" x="{L - 8}" y="{top + i * LH + 15}" text-anchor="end">{lab}</text>')
    prev_end = None
    for r in ROWS:
        lane = lanes["main"] if r["kind"] == "main" else lanes[r["label"].split(".")[0]]
        x, w = L + r["t"] * sx, max(r["api"] * sx, 1.6)
        cls = "rq main" if r["kind"] == "main" else "rq sub"
        out.append(f'<rect class="{cls}" x="{x:.1f}" y="{top + lane * LH + 3}" width="{w:.1f}" height="{LH - 9}" rx="1.5">'
                   f'<title>{name(r)}:输入 {fmt(r["inp"])}、输出 {fmt(r["out"])};录制时耗时 {r["api"]:.1f} 秒</title></rect>')
        if r["kind"] == "main":
            gap = r.get("gap") or 0
            if gap >= 150 and prev_end is not None:
                xa, xb = L + prev_end * sx, x
                over = gap > CAP
                txt = f"停 {gap:.0f} 秒" + (f" → 压到 {CAP}" if over else "")
                out.append(f'<line class="gapl{" over" if over else ""}" x1="{xa:.1f}" y1="{top - 14}" x2="{xb:.1f}" y2="{top - 14}"/>'
                           f'<text class="gapt{" over" if over else ""}" x="{(xa + xb) / 2:.1f}" y="{top - 20}" text-anchor="middle">{txt}</text>')
            prev_end = r["t"] + r["api"]
    H = top + 4 * LH + 24
    return f'<svg class="fl" viewBox="0 0 {L + W + 20} {H}" role="img" aria-label="整条会话的时间线">{"".join(out)}</svg>'


# ---------- 表:逐轮账本 ----------
def ledger():
    mx = max(r["inp"] + r["out"] for r in ROWS)
    rows = []
    n = 0
    for r in ROWS:
        n += 1
        pre = r["pre"] * B
        new = r["inp"] - pre
        main = r["kind"] == "main"
        gap = r.get("gap")
        if not main:
            wait = "按录制时刻启动"
        elif gap is None:
            wait = "开场"
        elif gap > CAP:
            wait = f'<s>{gap:.0f} 秒</s> {CAP} 秒'
        else:
            wait = f"{gap:.1f} 秒"
        bar = (f'<span class="bar"><i class="pre" style="width:{pre / mx * 100:.2f}%"></i>'
               f'<i class="new" style="width:{max(new / mx * 100, 0.3 if new else 0):.2f}%"></i>'
               f'<i class="outp" style="width:{max(r["out"] / mx * 100, 0.3):.2f}%"></i></span>')
        rows.append(f'<tr class="{"" if main else "subrow"}"><td class="l">{name(r)}</td><td class="l">{wait}</td>'
                    f'<td>{fmt(r["inp"])}</td><td class="cpre">{fmt(pre)}</td><td class="cnew">{fmt(new)}</td>'
                    f'<td class="cout">{fmt(r["out"])}</td><td class="barc">{bar}</td><td>{n}</td></tr>')
    return ('<table class="ledger"><thead><tr><th class="l">请求</th><th class="l">发之前等多久</th><th>发出的输入</th>'
            '<th>缓存里已有</th><th>需要新算</th><th>要求写出</th><th class="l">同一比例尺</th><th>累计样本</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


METRICS = """
<h2>七、从四个数到全部指标</h2>
<p>成绩单上每次请求只有四个原始数。所有指标都由它们算出,成本和能耗类再配上几个查表得来的常数。</p>
<div class="flcard"><table class="cfg">
<thead><tr><th>符号</th><th>含义</th><th>来源</th></tr></thead><tbody>
<tr class="sec"><td colspan="3">每次请求量到的四个数(第 i 次成功请求)</td></tr>
<tr><td>TTFT<sub>i</sub></td><td>首 token 时间(秒):发出请求到第一个 token 返回</td><td>客户端掐表</td></tr>
<tr><td>L<sub>i</sub></td><td>总耗时(秒):发出请求到最后一个 token 返回</td><td>客户端掐表</td></tr>
<tr><td>ISL<sub>i</sub></td><td>输入 token 数,完整输入,不区分是否命中缓存</td><td>服务器返回的 <code>usage</code></td></tr>
<tr><td>OSL<sub>i</sub></td><td>输出 token 数,思考 token 也算在内</td><td>服务器返回的 <code>usage</code></td></tr>
<tr class="sec"><td colspan="3">不来自请求的量</td></tr>
<tr><td>T</td><td>时间窗(秒)= 最晚的请求结束时刻 − 最早的请求开始时刻,约 3,600</td><td>由请求的起止时刻算出</td></tr>
<tr><td>G</td><td>芯片数。预填充和解码分开部署时,算整个部署的全部芯片</td><td>部署配置</td></tr>
<tr><td>h</td><td>理论缓存命中率 = 命中的块数 ÷ 总块数,假定缓存无限大</td><td>由剧本算出,与被测系统无关</td></tr>
<tr><td>C</td><td>每芯片小时成本(美元)。例:GB300 NVL72 自有 2.31、租用 5.00</td><td>SemiAnalysis 的常数表</td></tr>
<tr><td>P</td><td>每芯片的全包供电功率(千瓦),含主机、网络、制冷。例:GB300 NVL72 为 2.12</td><td>SemiAnalysis 的常数表</td></tr>
</tbody></table></div>
<p class="note">统计范围是测量窗口内的全部成功请求。预热请求和出错的请求不算。</p>

<h3 style="margin-top:26px">延迟类:每次请求一个样本,再取分位数</h3>
<div class="flcard"><table class="cfg">
<thead><tr><th>指标</th><th>公式</th><th>说明</th></tr></thead><tbody>
<tr><td>首 token 时间</td><td class="fx">P90( TTFT<sub>i</sub> )</td><td>包含排队和读输入的时间</td></tr>
<tr><td>token 间隔 ITL</td><td class="fx">ITL<sub>i</sub> = ( L<sub>i</sub> − TTFT<sub>i</sub> ) ÷ ( OSL<sub>i</sub> − 1 )<br>再取 P90( ITL<sub>i</sub> )</td><td>也叫 TPOT。是一次请求内的平均间隔,不是逐个 token 的间隔</td></tr>
<tr><td>交互速度</td><td class="fx">1 ÷ P90( ITL<sub>i</sub> )</td><td>单位 tok/s/user,曲线的横轴。先取 P90 再取倒数,所以是慢的那一头</td></tr>
<tr><td>端到端延迟</td><td class="fx">P90( L<sub>i</sub> )</td><td>主要被最长的那批输出拖着走,在 AgentX 里参考价值小</td></tr>
<tr><td>端到端归一化交互速度</td><td class="fx">1 ÷ P90( L<sub>i</sub> ÷ OSL<sub>i</sub> )</td><td>实验性指标。把首 token 的等待也摊进去,输出越短惩罚越重</td></tr>
</tbody></table></div>

<h3 style="margin-top:26px">吞吐类:全部请求求和,再除以时间和芯片数</h3>
<div class="flcard"><table class="cfg">
<thead><tr><th>指标</th><th>公式</th><th>说明</th></tr></thead><tbody>
<tr><td>输入吞吐</td><td class="fx">Σ ISL<sub>i</sub> ÷ T ÷ G</td><td>单位 tok/s/chip。被复用的前缀每一轮都重新计一次</td></tr>
<tr><td>输出吞吐</td><td class="fx">Σ OSL<sub>i</sub> ÷ T ÷ G</td><td></td></tr>
<tr><td>总吞吐</td><td class="fx">( Σ ISL<sub>i</sub> + Σ OSL<sub>i</sub> ) ÷ T ÷ G</td><td>默认口径,曲线的纵轴由它派生</td></tr>
<tr><td>未缓存口径的输入吞吐</td><td class="fx">输入吞吐 × ( 1 − h )</td><td>只算每轮新增的输入</td></tr>
<tr><td>未缓存口径的总吞吐</td><td class="fx">输入吞吐 × ( 1 − h ) + 输出吞吐</td><td>看板上叫 Horace He metric,目前不是默认显示</td></tr>
<tr><td>每秒完成请求数</td><td class="fx">按 1 秒窗口数完成的请求,再取均值或分位数</td><td></td></tr>
</tbody></table></div>

<h3 style="margin-top:26px">成本和能耗类:吞吐乘以常数</h3>
<p>下表的"吞吐"可以代入总、输入、输出三种,各得一套数字。</p>
<div class="flcard"><table class="cfg">
<thead><tr><th>指标</th><th>公式</th><th>说明</th></tr></thead><tbody>
<tr><td>每百万 token 成本</td><td class="fx">C × 1,000,000 ÷ ( 3600 × 吞吐 )</td><td>单位美元</td></tr>
<tr><td>每美元 token 数</td><td class="fx">吞吐 × 3600 ÷ C</td><td>看板默认的纵轴,用自有成本</td></tr>
<tr><td>每兆瓦吞吐</td><td class="fx">吞吐 × 1000 ÷ P</td><td>单位 tok/s/MW。P 是容量模型里的数,不是实测功率</td></tr>
<tr><td>每 token 焦耳</td><td class="fx">实测平均功率(瓦) ÷ 每秒 token 数</td><td>分"每输出 token"和"每总 token"两种</td></tr>
</tbody></table></div>

<h3 style="margin-top:26px">收入类:再加上售价</h3>
<div class="flcard"><table class="cfg">
<thead><tr><th>指标</th><th>公式</th><th>说明</th></tr></thead><tbody>
<tr><td>计费输入吞吐</td><td class="fx">输入吞吐 × ( 1 − 命中率 × ( 1 − 缓存读取价比例 ) )</td><td>这里的命中率用服务器实测值。缓存读取价比例默认 10%</td></tr>
<tr><td>每芯片小时收入</td><td class="fx">3600 × ( 计费输入吞吐 × 输入价 + 输出吞吐 × 输出价 )</td><td>默认定价:未缓存输入和输出各 $1 / 百万 token</td></tr>
</tbody></table></div>

<h3 style="margin-top:26px">代入这条会话</h3>
<p>只有能从剧本直接得出的部分算得出来:</p>
<ul>
  <li>Σ ISL = @TOTIN@,Σ OSL = @TOTOUT@,合计 @TOT@ 个 token 计入总吞吐。</li>
  <li>理论缓存命中率 h = @H@%。未缓存口径下计入的是 @UNC@ 个 token,约为总口径的 @UNCPCT@%。</li>
  <li>延迟类的指标一个也算不出来,因为 TTFT 和 L 要实测才有。吞吐类也缺 T 和 G。</li>
</ul>
<p class="note">公式按 AgentX 聚合脚本和看板文档的口径整理。其中"每 token 焦耳"和"每芯片小时收入"两条是按定义写出的,资料里只有字段名和文字描述,没有逐字的公式。token 间隔另有一种算法,区间算到 HTTP 响应结束而不是最后一个内容块,看板对 AgentX 优先用那一种,数值略大。</p>
"""


ASSUME = """
<h2>八、AgentX 做了哪些简化和假设</h2>
<p>前面七节讲的流程,每一步都对真实情况做了取舍。这些取舍大部分是 SemiAnalysis 自己写在文档里的,少数是官方没有明说、但从规则能推出来的。两类分开列。</p>

<h3 style="margin-top:26px">官方文档里写明的</h3>
<div class="flcard"><table class="cfg wrap">
<thead><tr><th>简化或假设</th><th>官方怎么说</th><th>出处</th></tr></thead><tbody>
<tr><td>只有一种来源</td><td>393 条全部是 Claude Code 的会话。承认请求分布随智能体框架而变,也随模型的分词器而变。理由是全球很大一部分智能体编程流量走 Claude Code。说下一版会覆盖更多模型和框架</td><td>发布长文</td></tr>
<tr><td>内容是合成的填充文字</td><td>承认推测解码的接受、拒绝数会异常,因为草稿模型没在这种数据上训练过。补救办法是强制一个固定的接受长度</td><td>发布长文、仓库规则</td></tr>
<tr><td>合成内容下的接受率不算证据</td><td>仓库规则明说,合成 AgentX 下的接受率不能作为任何合规证据</td><td>仓库规则</td></tr>
<tr><td>闭环重放</td><td>承认不同配置在同一小时内完成的请求数不同,遇到的负载组合有自然差异,低并发时最明显。理由是智能体的后一步依赖前一步的结果,不可能预先发出。文章自己链接了一篇批评闭环基准的博客</td><td>发布长文、方法页、方法博客</td></tr>
<tr><td>低并发样本少</td><td>承认低并发时收敛差</td><td>发布长文</td></tr>
<tr><td>停顿压到 300 秒</td><td>说明是为了能在一小时内跑完。300 秒对应 Anthropic 默认的缓存有效期</td><td>发布长文</td></tr>
<tr><td>时间戳不等于因果</td><td>承认 HTTP 时间戳看不出主智能体是在等子智能体,还是在做别的事。称这是"小局限",希望后续版本改进</td><td>发布长文</td></tr>
<tr><td>看不到供应商一侧的内容</td><td>承认聊天模板、专有分词器、服务端工具、加密的思考内容、图片和文档展开成多少 token,都只能用占位和按模型校准的填充来近似</td><td>发布长文、方法页</td></tr>
<tr><td>公开的只是子集</td><td>说明 393 条是从 8,271 条里选出的"有代表性的子集"</td><td>发布长文、方法页</td></tr>
<tr><td>指标不完美</td><td>说端到端延迟在 AgentX 里意义变小;交互速度和首 token 时间必须一起看;端到端归一化交互速度"有意不完美";人们真正在乎的是任务端到端完成的速度,新指标还在定义</td><td>发布长文、术语表</td></tr>
<tr><td>内存卸载上限是人为规则</td><td>承认本地生成器对所有机型都套了 3 TB 上限,例外情形还没实现</td><td>发布长文</td></tr>
</tbody></table></div>

<h3 style="margin-top:26px">官方没有明说,从规则推出来的</h3>
<ul>
  <li><b>行为没有校准。</b>官方只校准了 token 数(比值中位数 1.004)。输出多长、用多少轮、什么时候派子智能体,都是 Claude 的行为。被测模型真去干同样的活,会走出不同的轨迹。</li>
  <li><b>依赖内容的技术都测不准。</b>官方只提了推测解码。按同样的道理,内容感知的缓存淘汰、提示压缩、基于注意力稀疏性的方法,在填充文字上的表现也不代表真实表现。</li>
  <li><b>模型自己何时停不在测量范围内。</b>输出长度被 <code>ignore_eos</code> 钉在录制值上。</li>
  <li><b>筛选有偏。</b>入选规则要求至少 20 个请求、并发子智能体不超过 10 个,短会话和极端扇出的会话被排除。抽样方法和与母语料的分布对照没有公布。</li>
  <li><b>测不出过载。</b>会话数固定,跑完一条换一条,没有"新用户不断到来"的过程,所以排队崩溃和准入控制的行为观察不到。</li>
  <li><b>一小时看不全长期状态。</b>跨小时的缓存淘汰、内存卸载层填满后的稳态,都在窗口之外。</li>
</ul>

<h3 style="margin-top:26px">对应到这条会话</h3>
<ul>
  <li>图上的 {n} 次请求,输出长度和间隔都是原模型留下的。DeepSeek 真做这个任务,轮数和每轮的长度都会不同。</li>
  <li>{ngap} 段超过 300 秒的停顿被压短。真实使用里,这几段停顿正是缓存最可能被挤掉的时候。</li>
  <li>主智能体和子智能体原本用不同的模型,重放时都打到同一个被测模型上。</li>
</ul>
<p class="note">出处:发布长文是 SemiAnalysis 的 <a href="https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat">AgentX - InferenceXv3</a>(含 Future Next Steps 一节);方法页是 <a href="https://inferencex.semianalysis.com/infx/agentx/methodology">AgentX Methodology</a>;方法博客是 <a href="https://inferencex.semianalysis.com/blog/agentic-benchmark-agent-benchmark-guide">Agentic Benchmark for LLM Inference: Metrics and Methodology</a>。本节按调研笔记整理,没有重新打开原文逐条核对;发布长文的前半部分没有通读,相关表述可能有遗漏。更完整的版本和各方批评见 <a href="agentx.html#limits">AgentX 调研的局限一节</a>。</p>
"""


def build():
    mains = [r for r in ROWS if r["kind"] == "main"]
    subs = [r for r in ROWS if r["kind"] == "sub"]
    tot_in = sum(r["inp"] for r in ROWS)
    tot_out = sum(r["out"] for r in ROWS)
    tot_pre = sum(r["pre"] for r in ROWS) * B
    tot_new = tot_in - tot_pre
    r3, r2, r4 = ROWS[2], ROWS[1], ROWS[3]
    long_gaps = [r["gap"] for r in mains if (r.get("gap") or 0) > CAP]
    saved = sum(g - CAP for g in long_gaps)
    metrics = (METRICS.replace("@TOTIN@", fmt(tot_in)).replace("@TOTOUT@", fmt(tot_out)).replace("@TOT@", fmt(tot_in + tot_out))
               .replace("@H@", f"{tot_pre / tot_in * 100:.0f}").replace("@UNC@", fmt(tot_new + tot_out))
               .replace("@UNCPCT@", f"{(tot_new + tot_out) / (tot_in + tot_out) * 100:.0f}"))
    assume = ASSUME.replace("{n}", str(len(ROWS))).replace("{ngap}", str(len(long_gaps)))
    content = f"""<style>
  :root {{ --t-pre: #0f6e63; --t-new: #c98a16; --t-out: #c2410c; --t-subr: #6b4fbb; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --t-pre: #5cc5b6; --t-new: #e8b04a; --t-out: #fb923c; --t-subr: #b09cf0; }} }}
  .flwrap {{ max-width: 1000px; margin: 0 auto; padding: 0 20px 90px; }}
  .flwrap h2 {{ font-size: 22px; margin: 46px 0 6px; }}
  .flwrap p, .flwrap li {{ color: var(--ink-2); max-width: 820px; }}
  .flwrap .note {{ font-size: 13.5px; color: var(--ink-3); }}
  .flcard {{ background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 18px 18px 12px; box-shadow: var(--shadow); overflow-x: auto; margin: 14px 0 10px; }}
  .cast {{ display: grid; grid-template-columns: 1fr 26px 1fr 26px 1fr; gap: 0; align-items: stretch; margin: 16px 0 6px; }}
  .cast .who {{ background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; box-shadow: var(--shadow); }}
  .cast .who h4 {{ margin: 0 0 2px; font-size: 16px; }}
  .cast .who .role {{ font-size: 12.5px; color: var(--ink-3); margin-bottom: 8px; }}
  .cast .who ul {{ margin: 0; padding-left: 18px; font-size: 14px; }}
  .cast .who li {{ margin: 3px 0; }}
  .cast .ar {{ display: flex; align-items: center; justify-content: center; color: var(--ink-3); font-size: 20px; }}
  .verdict {{ margin: 10px 0 0; background: var(--surface); border: 1px dashed var(--line); border-radius: 12px; padding: 12px 16px; font-size: 14px; color: var(--ink-2); }}
  @media (max-width: 760px) {{ .cast {{ grid-template-columns: 1fr; }} .cast .ar {{ transform: rotate(90deg); height: 26px; }} }}
  table.cfg, table.ledger {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  table.cfg th, table.cfg td {{ text-align: left; vertical-align: top; padding: 8px 10px; border-bottom: 1px solid var(--line); }}
  table.cfg th {{ font-size: 12.5px; color: var(--ink-3); font-weight: 600; }}
  table.cfg td:first-child {{ white-space: nowrap; font-weight: 600; }}
  table.cfg td code {{ font-size: 12.5px; }}
  table.cfg.wrap td:first-child {{ white-space: normal; min-width: 150px; }} table.cfg.wrap td:last-child {{ min-width: 120px; color: var(--ink-3); font-size: 13px; }}
  table.cfg td.fx {{ font-family: var(--mono, ui-monospace, Menlo, monospace); font-size: 13px; white-space: nowrap; }}
  table.cfg tr.sec td {{ background: color-mix(in srgb, var(--ink-3) 9%, transparent); font-size: 12.5px; color: var(--ink-2); font-weight: 600; }}
  table.ledger {{ min-width: 860px; font-variant-numeric: tabular-nums; }}
  table.ledger th, table.ledger td {{ padding: 5px 8px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }}
  table.ledger th {{ font-size: 12px; color: var(--ink-3); font-weight: 600; position: sticky; top: 0; background: var(--surface); }}
  table.ledger .l {{ text-align: left; }}
  table.ledger tr.subrow td {{ color: var(--ink-3); }} table.ledger tr.subrow td.l:first-child {{ padding-left: 22px; }}
  table.ledger s {{ color: var(--ink-3); }}
  .cpre {{ color: var(--t-pre); }} .cnew {{ color: var(--t-new); font-weight: 600; }} .cout {{ color: var(--t-out); }}
  .barc {{ width: 30%; min-width: 220px; }}
  .bar {{ display: flex; height: 11px; gap: 1px; }}
  .bar i {{ display: block; height: 100%; border-radius: 1px; }}
  .bar .pre {{ background: var(--t-pre); opacity: .55; }} .bar .new {{ background: var(--t-new); }} .bar .outp {{ background: var(--t-out); margin-left: 2px; }}
  svg.fl {{ width: 100%; min-width: 780px; height: auto; display: block; font-family: var(--sans); }}
  svg.fl .hd {{ fill: var(--surface); stroke: var(--ink-3); stroke-width: 1; }}
  svg.fl .hdt {{ font-size: 14px; font-weight: 700; fill: var(--ink); }}
  svg.fl .life {{ stroke: var(--line); stroke-width: 2; stroke-dasharray: 3 5; }}
  svg.fl .dot {{ fill: var(--ink-3); stroke: var(--surface); stroke-width: 2; }}
  svg.fl .dot.pre {{ fill: var(--t-pre); }} svg.fl .dot.new {{ fill: var(--t-new); }} svg.fl .dot.outp {{ fill: var(--t-out); }}
  svg.fl .nt, svg.fl .mt {{ font-size: 12.5px; fill: var(--ink-2); }}
  svg.fl .b {{ font-weight: 700; fill: var(--ink); font-size: 13.5px; }}
  svg.fl .arr {{ stroke: var(--ink-2); stroke-width: 1.6; }} svg.fl .arr.thin {{ stroke: var(--t-out); stroke-width: 1; }}
  svg.fl .ahp {{ fill: var(--ink-2); }}
  svg.fl .grid {{ stroke: var(--line); stroke-width: 1; }}
  svg.fl .tick {{ font-size: 10.5px; fill: var(--ink-3); }}
  svg.fl .rlab {{ font-size: 11.5px; fill: var(--ink-2); }}
  svg.fl .rq.main {{ fill: var(--t-pre); }} svg.fl .rq.sub {{ fill: var(--t-subr); }}
  svg.fl .gapl {{ stroke: var(--ink-3); stroke-width: 1; }} svg.fl .gapl.over {{ stroke: var(--t-out); stroke-width: 1.6; }}
  svg.fl .gapt {{ font-size: 10.5px; fill: var(--ink-3); }} svg.fl .gapt.over {{ fill: var(--t-out); font-weight: 600; }}
  .split {{ display: flex; height: 34px; border-radius: 8px; overflow: hidden; font-size: 12.5px; margin: 10px 0 4px; min-width: 620px; }}
  .split div {{ display: flex; align-items: center; justify-content: center; color: #fff; padding: 0 6px; white-space: nowrap; }}
  .split .w {{ background: var(--ink-3); flex: 2; }} .split .q {{ background: var(--t-new); flex: 3; }} .split .d {{ background: var(--t-out); flex: 6; }}
  .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px; }}
  .two > div {{ background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 14px 18px; }}
  .two h4 {{ margin: 0 0 6px; font-size: 15px; }} .two ul {{ margin: 0; padding-left: 18px; font-size: 14px; }}
  @media (max-width: 760px) {{ .two {{ grid-template-columns: 1fr; }} }}
  .lg {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13.5px; color: var(--ink-2); margin: 4px 0 0; }}
  .lg span {{ display: inline-flex; align-items: center; gap: 6px; }}
  .lg i {{ width: 22px; height: 11px; border-radius: 2px; display: inline-block; }}
</style>
<main class="flwrap">
<div class="hero" style="padding-bottom:10px">
  <div class="eyebrow">图解</div>
  <h1>AgentX 数据集中的一条会话:它是怎么被测的</h1>
  <p class="lead">先不管并发。只放 AgentX 数据集中的一条会话(编号 {E(D["id"])}…,{len(mains)} 轮主对话加 {len(subs)} 次子智能体请求),看压测程序拿到什么、设了什么、发出什么、收回什么,以及每一轮过后哪些东西变了。</p>
  <div class="meta"><span>{len(ROWS)} 次请求</span><span>合计输入 {fmt(tot_in)} · 输出 {fmt(tot_out)}</span><a href="trace.html">这条会话的输入是怎么攒出来的 →</a><a href="agentx.html#replay">AgentX 的重放机制 →</a></div>
</div>

<h2>一、三个角色</h2>
<div class="cast">
  <div class="who"><h4>剧本</h4><div class="role">数据集里的一条会话</div><ul>
    <li>{len(ROWS)} 行,一行是一次请求</li><li>输入多少 token、由哪些块组成</li><li>输出多少 token</li><li>离上一次请求结束隔了多久</li><li>属于主智能体还是哪个子智能体</li><li><b>没有任何文字</b></li></ul></div>
  <div class="ar">→</div>
  <div class="who"><h4>压测程序</h4><div class="role">agentx-harness,扮演 Claude Code</div><ul>
    <li>把块编号变回一段段文字</li><li>照剧本一行一行发请求</li><li>等够间隔再发下一行</li><li>掐表、记数</li><li>不看模型写了什么</li></ul></div>
  <div class="ar">⇄</div>
  <div class="who"><h4>被测系统</h4><div class="role">引擎 + DeepSeek-V4.1-Flash + 硬件</div><ul>
    <li>真的把输入读一遍</li><li>真的一个个 token 写出来</li><li>自己决定缓存留什么、留多久</li><li>不知道自己在被测</li></ul></div>
</div>
<div class="verdict"><b>成绩单</b>:每次请求留下一行,四个数:首 token 时间、总耗时、输入 token 数、输出 token 数。这条会话跑完一共 {len(ROWS)} 行。模型写出来的文字不进成绩单。</div>

<h2>二、开跑之前就定死的东西</h2>
<p>下面这些在整条会话里不变。上半部分由 AgentX 的规则锁定,参赛者改不了;下半部分由参赛者自己定,但跑起来之后也不能动。</p>
<div class="flcard"><table class="cfg">
<thead><tr><th>配置项</th><th>取值</th><th>为什么这样定</th></tr></thead><tbody>
<tr class="sec"><td colspan="3">压测程序这边(规则锁定)</td></tr>
<tr><td>接口</td><td>OpenAI 兼容的 <code>/v1/chat/completions</code>,强制流式</td><td>首 token 时间和 token 间隔要在客户端量,必须流式</td></tr>
<tr><td>模型名</td><td>剧本里记的原模型名,一律改成被测模型</td><td>主智能体和子智能体的请求都打到同一个模型上</td></tr>
<tr><td>分词器</td><td>被测模型自己的分词器</td><td>保证"一块 64 token"在被测模型眼里也是 64 个左右</td></tr>
<tr><td>随机种子</td><td>42</td><td>同一个块编号永远变成同一段填充文字,每次跑都一样</td></tr>
<tr><td><code>max_tokens</code></td><td>每轮不同,等于剧本里这一轮的输出 token 数</td><td>输出长度由剧本定,不由模型定</td></tr>
<tr><td><code>ignore_eos</code></td><td><code>true</code>,不许改</td><td>模型想提前收尾也不行,必须写满</td></tr>
<tr><td>历史里的模型回答</td><td>填充文字,长度等于剧本里记的输出</td><td>不用被测模型刚写的内容,所以下一轮的输入人人相同</td></tr>
<tr><td>工具结果</td><td>默认当作普通的 user 消息发</td><td>剧本里没有工具的格式信息</td></tr>
<tr><td>播放标记</td><td>在第一条 user 消息前插一个 <code>[rid:12 位十六进制]</code>,这次播放从头到尾用同一个</td><td>防止同一条会话的另一次播放蹭到这次的缓存</td></tr>
<tr><td>等待</td><td>照剧本的间隔等;整棵会话树闲置超过 {CAP} 秒,按 {CAP} 秒算</td><td>保留真实的节奏,又不让长停顿把时间耗光</td></tr>
<tr><td>token 计数</td><td>以服务器返回的 <code>usage</code> 为准</td><td>发出去的文字会被服务器重新分词,数量可能差几个</td></tr>
<tr class="sec"><td colspan="3">被测系统这边(参赛者自己定)</td></tr>
<tr><td>引擎和版本</td><td>vLLM、SGLang 等,用上游镜像原样运行</td><td>不许给引擎打补丁</td></tr>
<tr><td>并行方式</td><td>几张卡、怎么切分模型、预填充和解码是否分开</td><td>这是参赛者真正在调的东西</td></tr>
<tr><td>缓存</td><td>前缀缓存留多大,是否卸载到主机内存</td><td>决定图上绿色部分能不能真的省下来</td></tr>
<tr><td>推测解码</td><td>可以开;接受长度强制设为公布的固定值</td><td>填充文字上的真实接受率没有意义</td></tr>
<tr><td>流式间隔</td><td>不超过 20 token 回传一次</td><td>攒太多再发,客户端量不准首 token 时间</td></tr>
</tbody></table></div>

<h2>三、一次请求的往返</h2>
<p>拿{name(r3)}做例子。时间从上往下走,左边是压测程序做的事,右边是被测系统做的事,中间的箭头是网络上传的东西。</p>
<div class="flcard">{chart_sequence(r3, r2, r4)}</div>
<p>压测程序在一次请求里只关心时间怎么切:</p>
<div class="flcard"><div class="split"><div class="w">等待(不计入)</div><div class="q">排队 + 读输入 → 首 token 时间</div><div class="d">解码 {fmt(r3["out"])} 个 token → 总耗时 − 首 token 时间</div></div>
<p class="note" style="margin:8px 0 2px">这一次请求的"token 间隔" = (总耗时 − 首 token 时间) ÷ (输出 token 数 − 1)。每次请求算出一个值,就是一个样本。</p></div>
<p>第 ④ 步发出去的请求大致长这样(填充文字用省略号代替,<code>rid</code> 的值是示意):</p>
<div class="flcard"><pre style="margin:0;font-size:12.5px;line-height:1.55;white-space:pre-wrap">POST /v1/chat/completions
{{
  "model": "deepseek-ai/DeepSeek-V4.1-Flash",
  "stream": true,
  "max_tokens": {r3["out"]},          ← 剧本里这一轮的输出数
  "ignore_eos": true,
  "messages": [
    {{"role": "system",    "content": "……"}},                      ← 填充文字
    {{"role": "user",      "content": "[rid:3f9a1c0b7e52]\\n\\n……"}},  ← 开头带播放标记
    {{"role": "assistant", "content": "……"}},                      ← 填充文字,约 {r2["out"]} token(上一轮的录制输出长度)
    {{"role": "user",      "content": "……"}}                       ← 这一轮新来的内容
  ]
}}</pre></div>
<p class="note">前两条消息合起来是 {fmt(r2["inp"])} token,具体怎么分给 system 和 user,这条会话的数据里看不出来。</p>

<h2>四、整条会话的时间线</h2>
<p>把 {len(ROWS)} 次请求按时间排开。每个色块是一次请求从发出到收完,色块之间的空白是等待。子智能体在第 5 轮之后被派出去,三个同时跑,主智能体等它们都结束才发第 6 轮。</p>
<div class="lg"><span><i style="background:var(--t-pre)"></i>主智能体的请求</span><span><i style="background:var(--t-subr)"></i>子智能体的请求</span><span><i style="background:var(--t-out)"></i>超过 {CAP} 秒的停顿,重放时压到 {CAP} 秒</span></div>
<div class="flcard">{chart_timeline()}</div>
<p class="note">这张图画的是<b>录制时</b>的时间线。重放时有两处不同:色块的宽度换成被测系统自己的耗时,这正是要测的东西;{len(long_gaps)} 段超过 {CAP} 秒的停顿被压短,合计少等 {saved:.0f} 秒。其余空白照原样等。把鼠标停在色块上可以看是哪一轮。</p>

<h2>五、逐轮账本:每一轮过后什么变了</h2>
<p>一行是一次请求。绿色是服务器缓存里已经有的开头部分,金色是这一轮要新读的,橙色是要求写出的。</p>
<div class="flcard">{ledger()}</div>
<p class="note">"缓存里已有"是理想情况:假定被测系统把这条会话之前的前缀一直留着。留不住的话,那一轮的"需要新算"就等于整条输入。第 1 轮是一次很小的请求,和后面的轮次不共用内容。第 18 轮的输入比第 17 轮略短,所以没有新增。</p>

<div class="two">
  <div><h4>每一轮都在变的</h4><ul>
    <li>输入:比上一轮多一截,从 {fmt(r2["inp"])} 长到 {fmt(mains[-1]["inp"])}</li>
    <li><code>max_tokens</code>:换成这一轮的录制输出数</li>
    <li>发之前等多久:换成这一轮的录制间隔</li>
    <li>服务器的缓存:这条会话的前缀又长了一截</li>
    <li>成绩单:多一行</li></ul></div>
  <div><h4>从头到尾不变的</h4><ul>
    <li>第二节里的全部配置</li>
    <li>块编号到填充文字的对应关系</li>
    <li>播放标记 <code>rid</code></li>
    <li>剧本本身:无论模型写了什么,下一行照发</li>
    <li>模型权重和被测系统的部署</li></ul></div>
</div>

<h2>六、这条会话跑完,留下什么</h2>
<ul>
  <li><b>{len(ROWS)} 个延迟样本。</b>每次请求一个首 token 时间、一个 token 间隔。评测最后对所有样本取 P90,token 间隔的 P90 取倒数就是"交互速度"。</li>
  <li><b>{fmt(tot_in + tot_out)} 个计入吞吐的 token。</b>其中输入 {fmt(tot_in)}、输出 {fmt(tot_out)}。默认口径把缓存命中的输入也算进去。</li>
  <li><b>真正要算的只有一小部分。</b>缓存一直留得住的话,需要新读的输入是 {fmt(tot_new)} token,加上输出共 {fmt(tot_new + tot_out)},占全部的 {(tot_new + tot_out) / (tot_in + tot_out) * 100:.0f}%。</li>
  <li><b>没有留下的:</b>模型写的文字、答得对不对。答题质量由另一套单独的精度测试把关。</li>
</ul>
{metrics}
{assume}
<h2>九、还没讲的</h2>
<p>正式评测在这之上再叠三件事:每条会话从中间某处切入并先预热缓存;很多条会话同时跑;只统计一小时窗口里的请求。这些见 <a href="agentx.html#replay">AgentX 调研的重放一节</a>。</p>
<p class="note">本页的流程按 AgentX 压测程序的方法文档整理,没有在真机上跑过;图里没有任何实测耗时。请求示例里的字段是按文档拼的示意,不是抓包。</p>
</main>"""
    (SITE / "flow.html").write_text(page("AgentX 数据集中的一条会话:它是怎么被测的", "agentx.html", content), encoding="utf-8")
    print("built: flow.html", f"({len(ROWS)} 行)")


if __name__ == "__main__":
    build()
