"""
MULTI-SOURCE PER-TASK AUDIT (fast).

v1 was ~20h: it ran all ~266 examples for every one of 20 sources x 400 tasks.
Key speedup: a candidate is only interesting if it is CHEAPER than what we already
have. Cost needs a single inference (for the profiler trace); correctness needs 266.
So: measure cost first, discard anything not strictly cheaper, and only then pay for
full verification. That skips the large majority of candidates.

HARD RULES (task158/task233/task017 each scored ZERO on the real grader while passing
every local check): a local pass is NECESSARY BUT NOT SUFFICIENT. This script writes
CANDIDATES only -- it never touches repairs/ or tracker.db. Anything >0.5 pts must
additionally survive the generalization test or an isolated Kaggle probe.
"""
import copy, functools, json, math, os, sqlite3, sys
import numpy as np
import onnx, onnxruntime

print = functools.partial(print, flush=True)
sys.path.insert(0, os.path.join("data", "neurogolf_utils"))
import neurogolf_utils as ngu

SOURCES = json.load(open("scratch_onnx/resolved_sources.json"))
OUT = "scratch_onnx/multisource_wins.json"

_ex = {}
def examples(t):
    if t not in _ex:
        d = json.load(open(f"data/task{t:03d}.json"))
        _ex[t] = [b for b in (ngu.convert_to_numpy(e)
                  for e in d["train"] + d["test"] + d["arc-gen"]) if b]
    return _ex[t]


def prepare(path, tag):
    """Load + structural screen + session. Returns (session, sanitized) or None."""
    try:
        m = onnx.load(path)
    except Exception:
        return None
    if m.functions or m.graph.sparse_initializer:
        return None
    for op in m.opset_import:
        if op.domain not in {"", "ai.onnx"}:
            return None
    san = ngu.sanitize_model(copy.deepcopy(m))
    if san is None:
        return None
    try:
        onnx.checker.check_model(san, full_check=True)
    except Exception:
        return None
    o = onnxruntime.SessionOptions()
    o.enable_profiling = True
    o.log_severity_level = 3
    o.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    o.profile_file_prefix = tag
    try:
        return onnxruntime.InferenceSession(san.SerializeToString(), o), san
    except Exception:
        return None


def cost_of(sess, san, probe):
    """Cheap: one inference -> profiler trace -> cost."""
    try:
        ngu.run_network(sess, probe)
    except Exception:
        sess.end_profiling(); return None
    tp = sess.end_profiling()
    mem, par = ngu.score_network(san, tp)
    try: os.remove(tp)
    except Exception: pass
    if mem is None or par is None or mem < 0 or par < 0:
        return None
    return mem + par


con = sqlite3.connect("repairs/tracker.db")
tracker = {r[0]: (r[1], r[2]) for r in
           con.execute("SELECT task, our_cost, our_points FROM tasks")}

wins, total, checked, skipped = [], 0.0, 0, 0
for t in range(1, 401):
    our_cost, our_pts = tracker.get(t, (None, 1.0))
    if our_cost is None or our_cost == -1:      # negative-pads sentinel: never touch
        continue
    exs = examples(t)
    probe = exs[0]["input"]

    best = None
    for ref, d in SOURCES.items():
        p = os.path.join(d, f"task{t:03d}.onnx")
        if not os.path.exists(p):
            continue
        prep = prepare(p, f"ms{t}")
        if not prep:
            continue
        sess, san = prep
        c = cost_of(sess, san, probe)
        checked += 1
        # STAGE 1 GATE: only pay for full verification if it actually beats us
        if c is None or c >= our_cost or (best and c >= best[1]):
            skipped += 1
            continue
        # STAGE 2: full correctness on every train+test+arc-gen example
        ok = True
        for b in exs:
            try:
                if not np.array_equal(ngu.run_network(sess, b["input"]), b["output"]):
                    ok = False; break
            except Exception:
                ok = False; break
        sess.end_profiling()
        if ok:
            best = (ref, c, max(1.0, 25.0 - math.log(max(1.0, c))))

    if best and best[2] - our_pts > 0.0005:
        gain = best[2] - our_pts
        total += gain
        wins.append({"task": t, "src": best[0], "their_cost": best[1],
                     "their_pts": round(best[2], 4), "our_cost": our_cost,
                     "our_pts": round(our_pts, 4), "gain": round(gain, 4)})
        flag = "   <== BIG: needs isolated test" if gain > 0.5 else ""
        print(f"  WIN task{t:03d}: {our_pts:.4f} -> {best[2]:.4f} (+{gain:.4f})  "
              f"cost {our_cost}->{best[1]}  [{best[0].split('/')[-1][:30]}]{flag}")
    if t % 20 == 0:
        print(f"  ...{t}/400 | {len(wins)} wins | +{total:.2f} pts | {checked} evaluated, {skipped} skipped by cost-gate")

wins.sort(key=lambda w: -w["gain"])
print(f"\n=== DONE: {len(wins)} wins, total +{total:.2f} points ===")
for w in wins:
    print(f"  task{w['task']:03d}: +{w['gain']:.4f}  cost {w['our_cost']}->{w['their_cost']}  [{w['src']}]")
big = [w for w in wins if w["gain"] > 0.5]
print(f"\n{len(big)} wins >0.5pt REQUIRE isolated Kaggle / generalization test before shipping")
json.dump(wins, open(OUT, "w"), indent=1)
print(f"saved -> {OUT}")
