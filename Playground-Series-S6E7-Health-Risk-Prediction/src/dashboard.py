"""Self-contained local dashboard for watching the model-zoo training run in
real time. No extra dependencies (stdlib http.server only) -- reads
artifacts/manifest.csv fresh on every poll, so it always reflects whatever
src.train / the watchdog have written, including across process restarts.

Usage:
    python -m src.dashboard             # serves http://localhost:8765
    python -m src.dashboard --port 9000
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.config import load_config
from src.model_zoo import build_registry
from src.train import load_manifest

_CFG = None
_FAMILY_TOTALS = None
_REGISTRY_TOTAL = None


def _init(cfg_path=None):
    global _CFG, _FAMILY_TOTALS, _REGISTRY_TOTAL
    _CFG = load_config(cfg_path) if cfg_path else load_config()
    registry = build_registry(_CFG)
    _FAMILY_TOTALS = Counter(s.family for s in registry)
    _REGISTRY_TOTAL = len(registry)


def _read_running():
    """Heartbeat files written by src.train workers (one per worker pid) --
    shows what's actually in flight right now, since the manifest only gets
    a row once a model finishes."""
    log_dir = _CFG["paths"]["log_dir"]
    if not os.path.isdir(log_dir):
        return []
    out = []
    for name in os.listdir(log_dir):
        if not (name.startswith("running_") and name.endswith(".json")):
            continue
        try:
            with open(os.path.join(log_dir, name), "r", encoding="utf-8") as f:
                entry = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue  # worker mid-write or just cleared it up; skip this poll
        entry["elapsed_seconds"] = round(time.time() - entry["started_at"], 1)
        out.append(entry)
    out.sort(key=lambda e: e["started_at"])
    return out


def _parse_ts(ts_str):
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").timestamp()
    except Exception:
        return None


def build_status():
    manifest = load_manifest(_CFG)
    rows = list(manifest.values())
    success = [r for r in rows if r["status"] == "success"]
    failed = [r for r in rows if r["status"] == "failed"]
    n_success, n_failed = len(success), len(failed)
    n_pending = max(0, _REGISTRY_TOTAL - n_success - n_failed)

    best = None
    if success:
        best_row = max(success, key=lambda r: float(r["balanced_accuracy_oof"]))
        best = {
            "id": best_row["id"], "family": best_row["family"],
            "feature_set": best_row["feature_set"],
            "score": float(best_row["balanced_accuracy_oof"]),
        }

    secs = [float(r["seconds"]) for r in success if r.get("seconds")]
    avg_seconds = statistics.mean(secs) if secs else None
    eta_minutes = (avg_seconds * n_pending / 60.0) if avg_seconds and n_pending else None

    all_ts = [t for t in (_parse_ts(r["timestamp"]) for r in rows) if t is not None]
    elapsed_minutes = (time.time() - min(all_ts)) / 60.0 if all_ts else None

    by_family = defaultdict(list)
    for r in success:
        by_family[r["family"]].append(float(r["balanced_accuracy_oof"]))
    fam_seconds = defaultdict(list)
    for r in success:
        if r.get("seconds"):
            fam_seconds[r["family"]].append(float(r["seconds"]))
    fam_failed = Counter(r["family"] for r in failed)

    family_stats = []
    for fam, total in _FAMILY_TOTALS.items():
        scores = by_family.get(fam, [])
        family_stats.append({
            "family": fam,
            "total": total,
            "done": len(scores),
            "failed": fam_failed.get(fam, 0),
            "mean_score": round(statistics.mean(scores), 5) if scores else None,
            "max_score": round(max(scores), 5) if scores else None,
            "avg_seconds": round(statistics.mean(fam_seconds[fam]), 1) if fam_seconds.get(fam) else None,
        })
    family_stats.sort(key=lambda f: (f["max_score"] is None, -(f["max_score"] or 0)))

    recent = sorted(rows, key=lambda r: r.get("timestamp", ""), reverse=True)[:20]
    recent_out = [{
        "id": r["id"], "family": r["family"], "feature_set": r["feature_set"],
        "status": r["status"],
        "score": float(r["balanced_accuracy_oof"]) if r.get("balanced_accuracy_oof") else None,
        "seconds": float(r["seconds"]) if r.get("seconds") else None,
        "timestamp": r["timestamp"], "error": r.get("error", ""),
    } for r in recent]

    trend_rows = sorted(success, key=lambda r: r.get("timestamp", ""))
    trend = [{
        "i": i + 1, "score": float(r["balanced_accuracy_oof"]),
        "id": r["id"], "family": r["family"],
    } for i, r in enumerate(trend_rows)]

    watchdog_tail = []
    wd_log = os.path.join(_CFG["paths"]["log_dir"], "watchdog.log")
    if os.path.exists(wd_log):
        with open(wd_log, "r", encoding="utf-8") as f:
            watchdog_tail = [line.rstrip() for line in f.readlines()[-8:]]

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device_mode": _CFG["device"]["mode"],
        "registry_total": _REGISTRY_TOTAL,
        "n_success": n_success,
        "n_failed": n_failed,
        "n_pending": n_pending,
        "running": _read_running(),
        "best": best,
        "avg_seconds": round(avg_seconds, 1) if avg_seconds else None,
        "eta_minutes": round(eta_minutes, 1) if eta_minutes else None,
        "elapsed_minutes": round(elapsed_minutes, 1) if elapsed_minutes else None,
        "family_stats": family_stats,
        "recent": recent_out,
        "trend": trend,
        "watchdog_tail": watchdog_tail,
    }


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Health Risk Model Zoo — Training Dashboard</title>
<style>
:root {
  --surface-1:      #fcfcfb;
  --page:           #f9f9f7;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #898781;
  --gridline:       #e1e0d9;
  --baseline:       #c3c2b7;
  --border:         rgba(11,11,11,0.10);
  --series-1:       #2a78d6;
  --good:           #0ca30c;
  --warning:        #fab219;
  --critical:       #d03b3b;
  --pending:        #c3c2b7;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface-1:      #1a1a19;
    --page:           #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --gridline:       #2c2c2a;
    --baseline:       #383835;
    --border:         rgba(255,255,255,0.10);
    --series-1:       #3987e5;
    --good:           #0ca30c;
    --warning:        #fab219;
    --critical:       #e66767;
    --pending:        #383835;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 24px; background: var(--page); color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}
h1 { font-size: 18px; margin: 0 0 4px; }
.sub { color: var(--text-secondary); font-size: 13px; margin-bottom: 20px; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--good); margin-right:6px; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
.running-row { display:flex; align-items:center; gap:10px; padding:8px 4px; border-bottom:1px solid var(--gridline); font-size:13px; }
.running-row:last-child { border-bottom:none; }
.running-row .dot2 { background: var(--warning); flex-shrink:0; }
.running-row .rid { font-family: ui-monospace, monospace; font-size:12px; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.running-row .relapsed { color: var(--text-secondary); font-variant-numeric: tabular-nums; flex-shrink:0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 20px; }
.tile {
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 16px;
}
.tile .label { font-size: 12px; color: var(--text-muted); margin-bottom: 6px; }
.tile .value { font-size: 26px; font-weight: 600; font-variant-numeric: tabular-nums; }
.tile .meta { font-size: 12px; color: var(--text-secondary); margin-top: 4px; }
.progress-bar { display:flex; height: 10px; border-radius: 5px; overflow:hidden; background: var(--pending); margin: 8px 0 20px; }
.progress-bar .seg { height: 100%; }
.progress-bar .success { background: var(--good); }
.progress-bar .failed { background: var(--critical); }
.legend { display:flex; gap:16px; font-size:12px; color: var(--text-secondary); margin-bottom: 20px; }
.legend span.sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }
.panel { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-bottom: 20px; }
.panel h2 { font-size: 14px; margin: 0 0 12px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; color: var(--text-muted); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: .03em; padding: 6px 8px; border-bottom: 1px solid var(--gridline); }
td { padding: 6px 8px; border-bottom: 1px solid var(--gridline); font-variant-numeric: tabular-nums; }
tr:last-child td { border-bottom: none; }
.bar-cell { display:flex; align-items:center; gap:8px; }
.bar-track { flex: 1; height: 6px; background: var(--gridline); border-radius: 3px; overflow:hidden; min-width: 60px; }
.bar-fill { height: 100%; background: var(--series-1); }
.status-badge { display:inline-flex; align-items:center; gap:5px; font-size:12px; }
.status-badge .sw { width:7px; height:7px; border-radius:50%; }
.grid2 { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
@media (max-width: 900px) { .grid2 { grid-template-columns: 1fr; } }
#tooltip { position:absolute; display:none; background: var(--surface-1); border:1px solid var(--border); border-radius:6px; padding:6px 10px; font-size:12px; pointer-events:none; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
.mono { font-family: ui-monospace, monospace; font-size: 12px; color: var(--text-secondary); }
pre.log { font-family: ui-monospace, monospace; font-size: 11px; color: var(--text-secondary); white-space: pre-wrap; margin: 0; max-height: 140px; overflow-y: auto; }
</style>
</head>
<body>
<h1><span class="dot"></span>Health Risk Model Zoo — Training Dashboard</h1>
<div class="sub" id="subheader">loading…</div>

<div class="tiles" id="tiles"></div>
<div class="progress-bar" id="progressBar"></div>
<div class="legend">
  <span><span class="sw" style="background:var(--good)"></span>success</span>
  <span><span class="sw" style="background:var(--critical)"></span>failed</span>
  <span><span class="sw" style="background:var(--pending)"></span>pending</span>
</div>

<div class="panel">
  <h2>Currently training</h2>
  <div id="runningList"></div>
</div>

<div class="grid2">
  <div class="panel">
    <h2>OOF balanced accuracy — as models complete</h2>
    <svg id="trendChart" width="100%" height="220" viewBox="0 0 800 220" preserveAspectRatio="none" style="overflow:visible"></svg>
  </div>
  <div class="panel">
    <h2>Watchdog activity</h2>
    <pre class="log" id="watchdogLog">—</pre>
  </div>
</div>

<div class="panel">
  <h2>Family breakdown</h2>
  <table id="familyTable"><thead><tr>
    <th>Family</th><th>Progress</th><th>Mean CV</th><th>Best CV</th><th>Avg sec/model</th>
  </tr></thead><tbody></tbody></table>
</div>

<div class="panel">
  <h2>Recent completions</h2>
  <table id="recentTable"><thead><tr>
    <th>Status</th><th>Model</th><th>Feature set</th><th>CV score</th><th>Seconds</th><th>Timestamp</th>
  </tr></thead><tbody></tbody></table>
</div>

<div id="tooltip"></div>

<script>
const fmt = (x, d=5) => (x === null || x === undefined) ? '—' : Number(x).toFixed(d);
const fmtInt = x => (x === null || x === undefined) ? '—' : Number(x).toLocaleString();

function statusColor(s) {
  return s === 'success' ? 'var(--good)' : (s === 'failed' ? 'var(--critical)' : 'var(--pending)');
}

function fmtElapsed(s) {
  if (s === null || s === undefined) return '—';
  const m = Math.floor(s / 60), r = Math.round(s % 60);
  return m > 0 ? `${m}m ${r}s` : `${r}s`;
}

function render(data) {
  const nRunning = data.running.length;
  document.getElementById('subheader').textContent =
    `device: ${data.device_mode.toUpperCase()}  ·  ${nRunning} worker${nRunning === 1 ? '' : 's'} active  ·  updated ${data.generated_at}  ·  refreshes every 5s`;

  const tiles = [
    { label: 'Trained', value: `${fmtInt(data.n_success + data.n_failed)} / ${fmtInt(data.registry_total)}`,
      meta: `${fmtInt(data.n_pending)} pending` },
    { label: 'Best CV (balanced acc.)', value: data.best ? fmt(data.best.score) : '—',
      meta: data.best ? `${data.best.family} · ${data.best.feature_set}` : 'no models yet' },
    { label: 'Success / Failed', value: `${fmtInt(data.n_success)} / ${fmtInt(data.n_failed)}`,
      meta: data.n_success + data.n_failed > 0 ? `${Math.round(100*data.n_success/(data.n_success+data.n_failed))}% success rate` : '' },
    { label: 'Avg sec / model', value: data.avg_seconds ? fmt(data.avg_seconds, 1) : '—', meta: 'successful models only' },
    { label: 'Elapsed', value: data.elapsed_minutes ? fmt(data.elapsed_minutes/60, 1) + 'h' : '—', meta: 'since first model' },
    { label: 'Est. remaining', value: data.eta_minutes ? fmt(data.eta_minutes/60, 1) + 'h' : '—', meta: 'rough, sequential CPU estimate' },
  ];
  document.getElementById('tiles').innerHTML = tiles.map(t =>
    `<div class="tile"><div class="label">${t.label}</div><div class="value">${t.value}</div><div class="meta">${t.meta}</div></div>`
  ).join('');

  const total = data.registry_total || 1;
  const pSuccess = 100 * data.n_success / total, pFailed = 100 * data.n_failed / total;
  document.getElementById('progressBar').innerHTML =
    `<div class="seg success" style="width:${pSuccess}%"></div><div class="seg failed" style="width:${pFailed}%"></div>`;

  const famBody = document.querySelector('#familyTable tbody');
  famBody.innerHTML = data.family_stats.map(f => {
    const pct = Math.round(100 * f.done / f.total);
    return `<tr>
      <td>${f.family}</td>
      <td><div class="bar-cell"><div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div><span class="mono">${f.done}/${f.total}</span></div></td>
      <td>${fmt(f.mean_score)}</td>
      <td>${fmt(f.max_score)}</td>
      <td>${f.avg_seconds ?? '—'}</td>
    </tr>`;
  }).join('');

  const recBody = document.querySelector('#recentTable tbody');
  recBody.innerHTML = data.recent.map(r => `<tr>
      <td><span class="status-badge"><span class="sw" style="background:${statusColor(r.status)}"></span>${r.status}</span></td>
      <td class="mono" title="${r.id}">${r.id.length > 42 ? r.id.slice(0,42)+'…' : r.id}</td>
      <td>${r.feature_set}</td>
      <td>${r.score !== null ? fmt(r.score) : (r.error ? r.error.slice(0,40) : '—')}</td>
      <td>${r.seconds ?? '—'}</td>
      <td class="mono">${r.timestamp}</td>
    </tr>`).join('');

  document.getElementById('watchdogLog').textContent = data.watchdog_tail.length ? data.watchdog_tail.join('\\n') : 'no watchdog log yet';

  const runningList = document.getElementById('runningList');
  runningList.innerHTML = data.running.length
    ? data.running.map(r => `<div class="running-row">
        <span class="dot dot2"></span>
        <span class="rid" title="${r.id}">${r.family} · ${r.feature_set}</span>
        <span class="relapsed">${fmtElapsed(r.elapsed_seconds)}</span>
      </div>`).join('')
    : `<div class="running-row" style="border:none; color:var(--text-muted)">no models in flight right now</div>`;

  drawTrend(data.trend);
}

function drawTrend(trend) {
  const svg = document.getElementById('trendChart');
  const W = 800, H = 220, PAD = 30;
  if (!trend.length) { svg.innerHTML = `<text x="${W/2}" y="${H/2}" fill="var(--text-muted)" font-size="13" text-anchor="middle">no completed models yet</text>`; return; }

  const scores = trend.map(t => t.score);
  let lo = Math.min(...scores), hi = Math.max(...scores);
  if (lo === hi) { lo -= 0.01; hi += 0.01; }
  const pad = (hi - lo) * 0.1;
  lo -= pad; hi += pad;

  const x = i => PAD + (i / Math.max(1, trend.length - 1)) * (W - 2*PAD);
  const y = s => H - PAD - ((s - lo) / (hi - lo)) * (H - 2*PAD);

  let gridlines = '';
  for (let g = 0; g <= 4; g++) {
    const gy = PAD + g * (H - 2*PAD) / 4;
    const val = hi - g * (hi - lo) / 4;
    gridlines += `<line x1="${PAD}" y1="${gy}" x2="${W-PAD}" y2="${gy}" stroke="var(--gridline)" stroke-width="1"/>`;
    gridlines += `<text x="4" y="${gy+3}" font-size="10" fill="var(--text-muted)">${val.toFixed(4)}</text>`;
  }

  const points = trend.map((t, i) => `${x(i)},${y(t.score)}`).join(' ');
  const circles = trend.map((t, i) =>
    `<circle cx="${x(i)}" cy="${y(t.score)}" r="3" fill="var(--series-1)" data-i="${i}" class="pt"/>`
  ).join('');

  svg.innerHTML = gridlines +
    `<polyline points="${points}" fill="none" stroke="var(--series-1)" stroke-width="2"/>` +
    circles;

  const tooltip = document.getElementById('tooltip');
  svg.querySelectorAll('.pt').forEach(c => {
    c.addEventListener('mouseenter', e => {
      const t = trend[+c.dataset.i];
      tooltip.style.display = 'block';
      tooltip.innerHTML = `<b>${fmt(t.score)}</b><br>${t.family}<br><span class="mono">#${t.i}</span>`;
      const r = svg.getBoundingClientRect();
      tooltip.style.left = (r.left + window.scrollX + x(+c.dataset.i) + 10) + 'px';
      tooltip.style.top = (r.top + window.scrollY + y(t.score) - 10) + 'px';
    });
    c.addEventListener('mouseleave', () => tooltip.style.display = 'none');
  });
}

async function tick() {
  try {
    const res = await fetch('/api/status');
    render(await res.json());
  } catch (e) { console.error(e); }
}
tick();
setInterval(tick, 5000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/status":
            try:
                data = build_status()
                body = json.dumps(data).encode("utf-8")
                self.send_response(200)
            except Exception as e:  # noqa: BLE001
                body = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    _init(args.config)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[dashboard] serving http://localhost:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
