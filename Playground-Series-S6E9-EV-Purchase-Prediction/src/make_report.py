"""Render reports/report.html from reports/report_data.json (charts as inline SVG)."""
import json, os, html
import common as C

D = json.load(open(os.path.join(C.ROOT, "reports", "report_data.json")))
esc = html.escape
def pct(v, d=1): return f"{v*100:.{d}f}%"
def fnum(v): return f"{v:,}".replace(",", " ")

# ---------------------------------------------------------------- svg helpers
def svg(w, h, body, label):
    return (f'<figure class="fig"><div class="fig-scroll">'
            f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}" '
            f'preserveAspectRatio="xMidYMid meet">{body}</svg></div>'
            f'<figcaption>{label}</figcaption></figure>')

def txt(x, y, s, cls="lab", anchor="start", size=None):
    st = f' font-size="{size}"' if size else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}"{st}>{esc(str(s))}</text>'

# ---- 1. the gate grid ------------------------------------------------------
def chart_gates():
    G = D["gate_grid"]; cw, ch = 132, 74; x0, y0 = 118, 52
    mx = max(max(r) for r in G["rate"])
    p = [txt(x0 + len(G["cols"])*cw/2, 24, "Range_Anxiety_Level", "axis-title", "middle")]
    for j, c in enumerate(G["cols"]):
        p.append(txt(x0 + j*cw + cw/2, 44, c, "lab", "middle"))
    p.append(f'<text x="16" y="{y0+len(G["rows"])*ch/2}" class="axis-title" '
             f'transform="rotate(-90 16 {y0+len(G["rows"])*ch/2})" text-anchor="middle">Subsidy_Available</text>')
    for i, r in enumerate(G["rows"]):
        p.append(txt(108, y0 + i*ch + ch/2 + 4, r, "lab", "end"))
        for j in range(len(G["cols"])):
            v, n = G["rate"][i][j], G["n"][i][j]
            o = 0.10 + 0.90 * (v / mx) ** 0.6
            x, yy = x0 + j*cw, y0 + i*ch
            p.append(f'<rect x="{x}" y="{yy}" width="{cw-6}" height="{ch-6}" rx="3" '
                     f'class="cell" style="opacity:{o:.3f}"/>')
            p.append(f'<text x="{x+(cw-6)/2}" y="{yy+30}" class="{"cell-v hi" if o>0.55 else "cell-v"}" '
                     f'text-anchor="middle">{pct(v,1)}</text>')
            p.append(f'<text x="{x+(cw-6)/2}" y="{yy+50}" class="{"cell-n hi" if o>0.55 else "cell-n"}" '
                     f'text-anchor="middle">{fnum(n)} rows</text>')
    return svg(560, y0 + 2*ch + 14, "".join(p),
               "Purchase rate by subsidy and range anxiety. The two gates multiply: "
               "either one closed drives the rate below 1%.")

# ---- 2. effect curves ------------------------------------------------------
def chart_curves():
    W, H, pad = 760, 250, {"l": 56, "r": 18, "t": 30, "b": 40}
    pw = (W - 40) / 2 - pad["l"] - pad["r"]
    ph = H - pad["t"] - pad["b"]
    out = []
    for k, (key, title, xlab, fmtx) in enumerate([
            ("env_curve", "Environmental_Concern_Level", "level", lambda v: str(int(v))),
            ("income_curve", "Annual_Income_USD", "decile median", lambda v: f"{v/1000:.0f}k")]):
        pts = D[key]; ox = k * ((W - 40) / 2 + 40)
        ymax = max(p["rate"] for p in pts) * 1.15
        X = lambda i: ox + pad["l"] + (pw * i / max(1, len(pts) - 1))
        Y = lambda v: pad["t"] + ph - ph * v / ymax
        out.append(txt(ox + pad["l"], 18, title, "axis-title"))
        for g in range(5):
            v = ymax * g / 4; yy = Y(v)
            out.append(f'<line x1="{ox+pad["l"]}" y1="{yy:.1f}" x2="{ox+pad["l"]+pw}" y2="{yy:.1f}" class="grid"/>')
            out.append(txt(ox + pad["l"] - 8, yy + 4, f"{v*100:.0f}%", "tick", "end"))
        d = " ".join(("M" if i == 0 else "L") + f"{X(i):.1f} {Y(p['rate']):.1f}" for i, p in enumerate(pts))
        out.append(f'<path d="{d}" class="line"/>')
        for i, p in enumerate(pts):
            out.append(f'<circle cx="{X(i):.1f}" cy="{Y(p["rate"]):.1f}" r="3.4" class="dot"/>')
        step = 1 if len(pts) <= 6 else 3
        for i, p in enumerate(pts):
            if i % step == 0 or i == len(pts) - 1:
                out.append(txt(X(i), H - pad["b"] + 20, fmtx(p["x"]), "tick", "middle"))
        out.append(txt(ox + pad["l"] + pw / 2, H - 6, xlab, "tick", "middle"))
    return svg(W, H, "".join(out),
               "The dial and the slope: purchase rate against environmental concern (left) "
               "and income decile (right), over all 668 665 training rows.")

# ---- 3. THE artifact chart -------------------------------------------------
def chart_digits():
    W, H = 760, 300
    pad = {"l": 54, "t": 54, "b": 52}; pw = (W - 46) / 2 - pad["l"] - 16; ph = H - pad["t"] - pad["b"]
    out = []
    panels = [("digits_synth", f"Competition train set · {fnum(D['n_train'])} rows", "bar-sig"),
              ("digits_orig",  f"Real survey it was generated from · {fnum(D['orig_rows'])} rows", "bar-neu")]
    ymax = 0.26
    for k, (key, title, cls) in enumerate(panels):
        h = D[key]["hundreds"]; ox = k * ((W - 46) / 2 + 46)
        base = sum(r * n for r, n in zip(h["rate"], h["n"])) / sum(h["n"])
        corr = h["corridor_pp"] / 100
        X = lambda i: ox + pad["l"] + pw * (i + 0.5) / 10
        Y = lambda v: pad["t"] + ph - ph * v / ymax
        out.append(txt(ox + pad["l"] - 6, 22, title, "axis-title"))
        out.append(txt(ox + pad["l"] - 6, 38,
                       f"spread {h['span_pp']:.2f}pp vs corridor {h['corridor_pp']:.2f}pp "
                       f"= {h['span_pp']/h['corridor_pp']:.1f}×",
                       "sig-note" if k == 0 else "neu-note"))
        for g in range(4):
            v = ymax * g / 3; yy = Y(v)
            out.append(f'<line x1="{ox+pad["l"]}" y1="{yy:.1f}" x2="{ox+pad["l"]+pw}" y2="{yy:.1f}" class="grid"/>')
            out.append(txt(ox + pad["l"] - 8, yy + 4, f"{v*100:.0f}%", "tick", "end"))
        # binomial null corridor around the base rate
        out.append(f'<rect x="{ox+pad["l"]}" y="{Y(base+corr/2):.1f}" width="{pw:.1f}" '
                   f'height="{max(1.5,(Y(base-corr/2)-Y(base+corr/2))):.1f}" class="corridor"/>')
        out.append(f'<line x1="{ox+pad["l"]}" y1="{Y(base):.1f}" x2="{ox+pad["l"]+pw}" y2="{Y(base):.1f}" class="base"/>')
        bw = pw / 10 * 0.62
        for i, r in enumerate(h["rate"]):
            out.append(f'<rect x="{X(i)-bw/2:.1f}" y="{Y(r):.1f}" width="{bw:.1f}" '
                       f'height="{max(0.6, Y(0)-Y(r)):.1f}" class="{cls}"/>')
            out.append(txt(X(i), H - pad["b"] + 18, i, "tick", "middle"))
        out.append(txt(ox + pad["l"] + pw / 2, H - 12, "hundreds digit of Annual_Income_USD", "tick", "middle"))
    return svg(W, H, "".join(out),
               "The generator artifact. Grey band = the binomial null corridor, the spread you would "
               "see if the digit meant nothing. Left: it is exceeded 25-fold. Right: the same test on "
               "the real survey stays inside it.")

# ---- 4. model progression --------------------------------------------------
def chart_progress():
    E = D["experiments"]
    LB = D.get("lb", {})
    W = 760; rowh = 40; H = 96 + rowh * len(E)
    lo, hi = 0.9410, 0.9472
    x0, x1 = 250, W - 46
    X = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)
    out = [txt(46, 26, "Cross-validated AUC, one shared 5-fold split", "axis-title")]
    for m, lab, cls in [(0.94186, "LB median", "ref"), (0.94610, "LB 90th pct", "ref"), (0.94644, "LB top", "ref")]:
        out.append(f'<line x1="{X(m):.1f}" y1="44" x2="{X(m):.1f}" y2="{H-32}" class="refline"/>')
        out.append(txt(X(m), 40, lab, "ref", "middle"))
    for i, e in enumerate(E):
        yy = 62 + rowh * i + rowh / 2
        out.append(txt(46, yy + 4, e["name"].split("_", 1)[1].replace("_", " "), "lab"))
        out.append(f'<line x1="{X(lo):.1f}" y1="{yy:.1f}" x2="{X(e["cv_auc"]):.1f}" y2="{yy:.1f}" class="track"/>')
        out.append(f'<circle cx="{X(e["cv_auc"]):.1f}" cy="{yy:.1f}" r="5.5" class="dot"/>')
        out.append(txt(X(e["cv_auc"]) + 12, yy + 4, f'{e["cv_auc"]:.6f}', "val"))
        if e["name"] in LB:
            out.append(f'<circle cx="{X(LB[e["name"]]):.1f}" cy="{yy:.1f}" r="4" class="dot-lb"/>')
    out.append(txt(46, H - 10, "filled = CV   ·   open = achieved public leaderboard", "tick"))
    return svg(W, H, "".join(out), "Every model on the same folds, against the public board.")

# ---------------------------------------------------------------- tables
def schema_rows():
    r = []
    for s in D["schema"]:
        c = "" if s["corr"] is None else f'{s["corr"]:+.3f}'
        r.append(f'<tr><td class="mono">{esc(s["name"])}</td><td class="dim">{s["kind"]}</td>'
                 f'<td class="mono dim">{esc(s["domain"])}</td><td class="mono num">{c}</td>'
                 f'<td>{esc(s["desc"])}</td></tr>')
    return "".join(r)

def digit_rows():
    r = []
    for k in ["units", "tens", "hundreds", "thousands"]:
        a, b = D["digits_synth"][k], D["digits_orig"][k]
        ra, rb = a["span_pp"]/a["corridor_pp"], b["span_pp"]/b["corridor_pp"]
        r.append(f'<tr><td class="mono">{k}</td>'
                 f'<td class="mono num">{a["span_pp"]:.2f}</td><td class="mono num dim">{a["corridor_pp"]:.2f}</td>'
                 f'<td class="mono num"><span class="chip sig">{ra:.1f}×</span></td>'
                 f'<td class="mono num">{b["span_pp"]:.2f}</td><td class="mono num dim">{b["corridor_pp"]:.2f}</td>'
                 f'<td class="mono num"><span class="chip neu">{rb:.1f}×</span></td></tr>')
    return "".join(r)

def exp_rows():
    LB = {k: f"{v:.5f}" for k, v in D.get("lb", {}).items()}
    r = []
    for e in D["experiments"]:
        r.append(f'<tr><td class="mono">{esc(e["name"])}</td>'
                 f'<td class="mono num">{e["cv_auc"]:.6f}</td>'
                 f'<td class="mono num dim">±{e["fold_std"]:.6f}</td>'
                 f'<td class="mono num">{LB.get(e["name"], "—")}</td>'
                 f'<td class="mono num dim">{e["n_features"]}</td>'
                 f'<td class="dim">{esc(e["notes"])}</td></tr>')
    return "".join(r)

def cat_bars():
    out = []
    for col in ["Subsidy_Available", "Range_Anxiety_Level", "Home_Charging_Possible",
                "City_Type", "Current_Car_Type", "Gender"]:
        lv = sorted(D["cat_effects"][col], key=lambda d: -d["rate"])
        mx = 0.30
        rows = "".join(
            f'<div class="cb-row"><span class="cb-lab">{esc(l["level"])}</span>'
            f'<span class="cb-track"><span class="cb-fill" style="width:{min(100,l["rate"]/mx*100):.1f}%"></span></span>'
            f'<span class="cb-val mono">{pct(l["rate"],1)}</span></div>' for l in lv)
        out.append(f'<div class="cb"><h4 class="mono">{esc(col)}</h4>{rows}</div>')
    return "".join(out)

# ---------------------------------------------------------------- page
CSS = """
:root{
  --paper:#f5f7f4; --surface:#ffffff; --ink:#16211d; --ink-2:#3a4742; --muted:#66736d;
  --line:#dde4df; --line-2:#eaefeb; --accent:#0d7561; --accent-soft:#d7ebe5;
  --signal:#b0521c; --signal-soft:#f4e2d5; --neutral-bar:#9aa8a2;
}
@media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){
  --paper:#0e1411; --surface:#151d19; --ink:#e7ece9; --ink-2:#bcc7c2; --muted:#8a9a93;
  --line:#26312c; --line-2:#1c2521; --accent:#46c3a6; --accent-soft:#14332c;
  --signal:#e08a4e; --signal-soft:#3a2618; --neutral-bar:#6a7a74;
}}
:root[data-theme="dark"]{
  --paper:#0e1411; --surface:#151d19; --ink:#e7ece9; --ink-2:#bcc7c2; --muted:#8a9a93;
  --line:#26312c; --line-2:#1c2521; --accent:#46c3a6; --accent-soft:#14332c;
  --signal:#e08a4e; --signal-soft:#3a2618; --neutral-bar:#6a7a74;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);
  font-family:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:16px;line-height:1.62;-webkit-font-smoothing:antialiased}
.mono{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  font-variant-numeric:tabular-nums}
.wrap{max-width:1080px;margin:0 auto;padding:0 28px 96px}
.col{max-width:68ch}
h1,h2,h3,h4{font-family:Spectral,Georgia,"Times New Roman",serif;text-wrap:balance;
  font-weight:600;line-height:1.22;margin:0}
/* masthead */
header.mast{border-bottom:1px solid var(--line);padding:56px 0 30px;margin-bottom:44px}
.kicker{font-family:"IBM Plex Mono",monospace;font-size:11.5px;letter-spacing:.15em;
  text-transform:uppercase;color:var(--accent);margin:0 0 14px}
h1{font-size:clamp(30px,4.6vw,46px);letter-spacing:-.012em}
.sub{color:var(--ink-2);font-size:18px;margin:14px 0 0;max-width:60ch}
.meta{display:flex;flex-wrap:wrap;gap:0 32px;margin-top:26px;padding-top:20px;
  border-top:1px solid var(--line-2)}
.meta div{display:flex;flex-direction:column;gap:2px}
.meta dt{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.12em;
  text-transform:uppercase;color:var(--muted)}
.meta dd{margin:0;font-family:"IBM Plex Mono",monospace;font-size:15px;font-variant-numeric:tabular-nums}
/* sections */
section{margin:0 0 60px;scroll-margin-top:20px}
.step{display:flex;align-items:baseline;gap:14px;margin:0 0 8px}
.step-n{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--accent);
  letter-spacing:.1em;padding-top:2px}
h2{font-size:26px;letter-spacing:-.008em}
h3{font-size:18.5px;margin:34px 0 8px}
p{margin:14px 0}
.lede{font-size:17.5px;color:var(--ink-2)}
strong{font-weight:600}
code{font-family:"IBM Plex Mono",monospace;font-size:.9em;background:var(--line-2);
  padding:1.5px 5px;border-radius:3px}
a{color:var(--accent)}
/* callout: reserved for the one finding */
.finding{border-left:3px solid var(--signal);background:var(--signal-soft);
  padding:20px 24px;margin:26px 0;border-radius:0 4px 4px 0}
.finding h3{margin:0 0 6px;font-size:19px;color:var(--signal)}
.finding p{margin:8px 0 0}
.note{border-left:2px solid var(--line);padding:2px 0 2px 18px;margin:22px 0;
  color:var(--ink-2);font-size:15px}
/* tables */
.tw{overflow-x:auto;margin:22px 0;border:1px solid var(--line);border-radius:5px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:14px;min-width:560px}
th{text-align:left;font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted);font-weight:500;padding:11px 14px;
  border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:9px 14px;border-bottom:1px solid var(--line-2);vertical-align:top}
tr:last-child td{border-bottom:none}
td.num,th.num{text-align:right}
td.dim{color:var(--muted)}
.chip{display:inline-block;padding:1px 7px;border-radius:9px;font-size:12px;font-weight:500}
.chip.sig{background:var(--signal-soft);color:var(--signal)}
.chip.neu{background:var(--line-2);color:var(--muted)}
/* figures */
.fig{margin:28px 0;background:var(--surface);border:1px solid var(--line);
  border-radius:5px;padding:18px 18px 4px}
.fig-scroll{overflow-x:auto}
.fig svg{display:block;width:100%;height:auto;min-width:460px}
figcaption{font-size:13.5px;color:var(--muted);padding:12px 2px 14px;
  border-top:1px solid var(--line-2);margin-top:10px;max-width:78ch}
.lab{fill:var(--ink);font-size:13px;font-family:"IBM Plex Sans",sans-serif}
.tick{fill:var(--muted);font-size:11.5px;font-family:"IBM Plex Mono",monospace}
.axis-title{fill:var(--ink);font-size:13px;font-weight:600;font-family:"IBM Plex Sans",sans-serif}
.val{fill:var(--ink);font-size:13px;font-family:"IBM Plex Mono",monospace}
.ref{fill:var(--muted);font-size:11px;font-family:"IBM Plex Mono",monospace}
.grid{stroke:var(--line);stroke-width:1}
.refline{stroke:var(--line);stroke-width:1;stroke-dasharray:3 4}
.line{fill:none;stroke:var(--accent);stroke-width:2}
.dot{fill:var(--accent)}
.dot-lb{fill:var(--surface);stroke:var(--accent);stroke-width:2}
.track{stroke:var(--line);stroke-width:2}
.cell{fill:var(--accent)}
.cell-v{fill:var(--ink);font-size:17px;font-weight:600;font-family:"IBM Plex Mono",monospace}
.cell-n{fill:var(--muted);font-size:11px;font-family:"IBM Plex Mono",monospace}
.cell-v.hi,.cell-n.hi{fill:#fff}
.bar-sig{fill:var(--signal)}
.bar-neu{fill:var(--neutral-bar)}
.corridor{fill:var(--muted);opacity:.22}
.base{stroke:var(--muted);stroke-width:1;stroke-dasharray:4 3}
.sig-note{fill:var(--signal);font-size:12px;font-family:"IBM Plex Mono",monospace}
.neu-note{fill:var(--muted);font-size:12px;font-family:"IBM Plex Mono",monospace}
/* categorical bars */
.cbs{display:grid;grid-template-columns:repeat(auto-fit,minmax(248px,1fr));gap:18px 30px;margin:24px 0}
.cb h4{font-size:12px;font-weight:500;letter-spacing:.04em;color:var(--muted);
  margin:0 0 8px;font-family:"IBM Plex Mono",monospace}
.cb-row{display:grid;grid-template-columns:82px 1fr 46px;align-items:center;gap:9px;margin:4px 0}
.cb-lab{font-size:13px;color:var(--ink-2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cb-track{height:7px;background:var(--line-2);border-radius:4px;overflow:hidden}
.cb-fill{display:block;height:100%;background:var(--accent);border-radius:4px}
.cb-val{font-size:12.5px;text-align:right;color:var(--ink)}
/* pipeline */
pre{background:var(--surface);border:1px solid var(--line);border-radius:5px;
  padding:16px 18px;overflow-x:auto;font-family:"IBM Plex Mono",monospace;
  font-size:12.5px;line-height:1.66;margin:20px 0}
ul{padding-left:20px;margin:14px 0}
li{margin:7px 0}
footer{border-top:1px solid var(--line);margin-top:56px;padding-top:22px;
  color:var(--muted);font-size:13.5px}
@media (prefers-reduced-motion:no-preference){a{transition:opacity .15s}}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
"""

def step(n, t):
    return f'<div class="step"><span class="step-n">{n}</span><h2>{t}</h2></div>'

top_lb, med_lb = 0.94644, 0.94186
best = max(D["experiments"], key=lambda e: e["cv_auc"])
base = D["experiments"][0]

HTML = f"""<title>EV Purchase Propensity S6E9</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Spectral:wght@500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">

<header class="mast">
  <p class="kicker">Kaggle Playground S6E9 · Analysis &amp; modelling report</p>
  <h1>Predicting Electric Vehicle Purchases</h1>
  <p class="sub">Thirteen survey columns, {fnum(D['n_train'])} rows, one binary outcome — and a
  wall at 0.941 that turned out to be a property of the <em>representation</em>, not of the data.</p>
  <div class="meta">
    <div><dt>Task</dt><dd>Binary probability</dd></div>
    <div><dt>Metric</dt><dd>ROC AUC</dd></div>
    <div><dt>Train / test</dt><dd>{fnum(D['n_train'])} / {fnum(D['n_test'])}</dd></div>
    <div><dt>Positive rate</dt><dd>{pct(D['pos_rate'],2)}</dd></div>
    <div><dt>Baseline CV</dt><dd>{base['cv_auc']:.6f}</dd></div>
    <div><dt>Best CV</dt><dd>{best['cv_auc']:.6f}</dd></div>
  </div>
</header>

<section id="task">
{step("01", "What goes in, what comes out")}
<div class="col">
<p class="lede">Each row is one survey respondent. The model returns the probability that
respondent buys an electric vehicle, and submissions are scored by ROC AUC — so only the
<em>ordering</em> of the {fnum(D['n_test'])} test probabilities matters, never their calibration.</p>
<p>The submission file is two columns, <code>id</code> and <code>Will_Buy_EV</code>, the second a
probability in [0,&nbsp;1]. The provided <code>sample_submission.csv</code> is the training base
rate, {pct(D['pos_rate'],4)}, repeated on every row — which scores exactly 0.5.</p>
<p>Neither file has a single missing value, and the train and test marginals are
indistinguishable: every numeric mean differs by less than 0.005 standard deviations. There is
no cleaning to do and no drift to correct.</p>
</div>
<div class="tw"><table>
<thead><tr><th>Column</th><th>Type</th><th>Domain</th><th class="num">r with target</th><th>Role</th></tr></thead>
<tbody>{schema_rows()}</tbody></table></div>
</section>

<section id="eda">
{step("02", "Two gates and one dial")}
<div class="col">
<p class="lede">The target is not driven by thirteen features acting together. It is driven by two
near-binary gates that must both be open, and then by how strongly the respondent cares.</p>
<p><code>Subsidy_Available</code> and <code>Range_Anxiety_Level</code> multiply rather than add.
Close either and the purchase rate collapses below one percent, regardless of income, age or
charging access.</p>
</div>
{chart_gates()}
<div class="col">
<p>Inside the open cell — subsidy available, low range anxiety, {fnum(D['gate_grid']['n'][1][0])} rows —
the rate is {pct(D['gate_grid']['rate'][1][0],1)}, and two features do nearly all the remaining work.</p>
</div>
{chart_curves()}
<div class="col">
<h3>Everything else, ranked honestly</h3>
<p>The remaining categoricals are second-order, and one of them does nothing at all.
Bars are drawn on a common 0–30% scale so the comparison is fair.</p>
</div>
<div class="cbs">{cat_bars()}</div>
<div class="col">
<p class="note"><strong>Structural quirks worth flagging.</strong>
{pct(D['income_floor_share'],1)} of rows sit exactly at the income floor of $30,000 and
{pct(D['commute_floor_share'],1)} exactly at the commute floor of 5.0&nbsp;km — both censored rather
than genuine values, so both get an explicit flag. There are {D['dup_rows']} duplicate feature rows,
and <code>id</code> is pure noise (AUC {D['id_auc']:.5f}), so there is no ordering leak.</p>
</div>
</section>

<section id="wall">
{step("03", "The wall at 0.941")}
<div class="col">
<p class="lede">A stock LightGBM scored CV {base['cv_auc']:.6f} and public LB 0.94163 — which was
almost exactly the leaderboard <em>median</em>. Three diagnostics agreed that tuning was not the
answer.</p>
<ul>
<li><strong>A capacity sweep saturated.</strong> Learning rates 0.01–0.05 and 16–256 leaves all
landed within 0.0007 of each other on fold 0, and the <em>shallowest</em> configuration won —
the underlying function is smooth and low-order, not deep and interaction-heavy.</li>
<li><strong>A neural net did worse.</strong> A GPU MLP with categorical embeddings reached 0.9372,
below the trees. So the gap was not a failure to model smoothness either.</li>
<li><strong>The model was already at its own ceiling.</strong> Sampling labels from the model's own
out-of-fold probabilities and re-scoring gives 0.9416 — the best AUC obtainable <em>if those
probabilities were the truth</em>. The model achieved 0.9418. It had extracted everything its
representation contained.</li>
</ul>
<p>That last number is the useful one. It does not say the task is capped at 0.9416; it says the
<em>features as given</em> are. Something had to change about the inputs.</p>
</div>
</section>

<section id="artifact">
{step("04", "What the income digits know")}
<div class="finding">
  <h3>The generator wrote the target into the low-order digits of income</h3>
  <p>Group the training rows by the hundreds digit of <code>Annual_Income_USD</code> and the
  purchase rate spans {D['digits_synth']['hundreds']['span_pp']:.2f} percentage points, against a
  binomial null corridor of {D['digits_synth']['hundreds']['corridor_pp']:.2f}pp — exceeded
  {D['digits_synth']['hundreds']['span_pp']/D['digits_synth']['hundreds']['corridor_pp']:.0f}-fold.
  The hundreds digit of a household income cannot causally matter. This is a fingerprint of the
  model that synthesised the data.</p>
</div>
<div class="col">
<p>The competition data is synthetic, generated from a real {fnum(D['orig_rows'])}-row survey
that is public. That survey is the control, and it settles the question: run the identical test on
it and the spread sits inside the corridor. The signal exists only in the synthetic copy.</p>
</div>
{chart_digits()}
<div class="tw"><table>
<thead><tr><th>Digit position</th>
<th class="num">Spread</th><th class="num">Corridor</th><th class="num">Ratio</th>
<th class="num">Spread</th><th class="num">Corridor</th><th class="num">Ratio</th></tr></thead>
<tbody>{digit_rows()}</tbody></table></div>
<div class="col">
<p class="note">Columns 2–4 are the competition training set, columns 5–7 the real survey.
Every digit position of income leaks in the synthetic data; none leaks in the original.</p>
<h3>What did not pan out</h3>
<p>A residual scan over roughly forty candidate keys — income and commute moduli, digit crosses,
age remainders — was then run against the improved model. <code>Age</code> came back overwhelmingly
<em>significant</em> and was still rejected: with 14,800 rows per age value, a ±0.8pp wobble is
five sigma and worth essentially nothing in AUC. Ranking candidates by significance rather than
effect size is the trap here, and it is worth stating that it caught one.</p>
<p>Adding the real survey as extra <em>training</em> rows is worth about +0.0002 at unit weight and
hurts above it. Its value in this project was as a control, not as data.</p>
</div>
</section>

<section id="pipeline">
{step("05", "The pipeline")}
<div class="col">
<p class="lede">Twenty-nine features, one shared cross-validation split, and target encodings
rebuilt inside every fold.</p>
<h3>Features</h3>
<p>The thirteen raw columns ordinal-encoded, plus the five base-10 digits of income,
<code>income mod 100</code>, <code>mod 1000</code>, <code>div 1000</code>, the digit sum, the commute
decimal digit and integer part, and the two floor flags. Engineered ratios — income per car,
km per station and similar — were measured and <strong>dropped</strong>; they cost 0.0001.</p>
<h3>Encodings, without the leak</h3>
<p>Smoothed mean-target encodings of the exact income value, the exact commute value and
<code>income // 100</code>. Inside each fold these are fitted on that fold's training rows only, and
the training-side copy uses an inner 5-fold so no row ever sees its own label through its own
encoding. Skipping either precaution inflates cross-validation by several thousandths and buys
nothing on the board.</p>
<h3>Validation and blending</h3>
<p>One <code>StratifiedKFold(5, shuffle, seed=42)</code> split is shared by every experiment, so the
out-of-fold matrix is coherent and blend weights are fitted on honest predictions. Only models
trained in this repository enter the blend. Predictions are converted to ranks, then combined by
plain rank mean, greedy hill climbing, and a cross-validated logistic stack; the best out-of-fold
blender is the one submitted.</p>
</div>
<pre>data/          train.csv  test.csv  sample_submission.csv  original/
src/common.py    loading · the shared 5-fold split · leak-free target encoding · runner
src/features.py  base_frame()      fold-independent columns
                 fold_transform()  fold-DEPENDENT encodings, rebuilt per fold
src/eda.py       figures + reports/eda_stats.json
src/exp*.py      one file per model  →  artifacts/oof/  artifacts/preds/
src/tune.py      Optuna over pre-transformed folds
src/blend.py     rank mean · hill climbing · logistic stack  →  submissions/</pre>
</section>

<section id="results">
{step("06", "Results")}
{chart_progress()}
<div class="tw"><table>
<thead><tr><th>Experiment</th><th class="num">CV AUC</th><th class="num">fold σ</th>
<th class="num">Public LB</th><th class="num">Feats</th><th>Notes</th></tr></thead>
<tbody>{exp_rows()}</tbody></table></div>
<div class="col">
<p>Cross-validation has tracked the public leaderboard to within 0.0002 on every submission, which
is why selection is done on CV. That matters more than it sounds: the DeLong standard error of a
public score on a 20% test slice is about 0.0006, so the entire visible top of the
board — median {med_lb:.5f} to top {top_lb:.5f} — spans roughly one standard error per rank cluster.
A cross-validated estimate built from all {fnum(D['n_train'])} training rows is several times more
precise than the number the board refreshes.</p>
</div>
</section>

<footer>
<p>Generated from <code>reports/report_data.json</code> by <code>src/make_report.py</code>.
Every figure is drawn from the competition data; the control column comes from the public
survey the competition data was generated from.</p>
</footer>
</div>
"""

out = os.path.join(C.ROOT, "reports", "report.html")
open(out, "w").write(HTML)
print("wrote", out, f"({len(HTML)//1024} KB)")
