import os, sys, json, copy, math, sqlite3, shutil
sys.path.insert(0, os.path.join("data", "neurogolf_utils"))
import onnx, onnxruntime
import numpy as np
from onnx import helper
from collections import defaultdict
import neurogolf_utils as ngu

REP = "repairs"
BASE = "baseline_v22"
CAND_DIR = "scratch_onnx/demorgan_sweep"
os.makedirs(CAND_DIR, exist_ok=True)

def load_task(t):
    return json.load(open(os.path.join("data", f"task{t:03d}.json")))

def rebuild_consumers(nodes):
    prod = {}
    cons = defaultdict(list)
    for i, n in enumerate(nodes):
        for y in n.output:
            prod[y] = i
        for x in n.input:
            cons[x].append(i)
    return prod, cons

def apply_folds(m, allow_cmp_fold):
    nodes = list(m.graph.node)
    prod, cons = rebuild_consumers(nodes)
    remove = set(); replace = {}
    inv = {"Less": "GreaterOrEqual", "Greater": "LessOrEqual"}
    n1 = 0
    if allow_cmp_fold:
        for i, n in enumerate(nodes):
            if n.op_type not in inv or len(n.output) != 1:
                continue
            users = cons.get(n.output[0], [])
            if len(users) != 1:
                continue
            j = users[0]
            nn = nodes[j]
            if nn.op_type != "Not" or len(nn.output) != 1:
                continue
            replace[i] = helper.make_node(inv[n.op_type], list(n.input), [nn.output[0]], name=(n.name or f"n{i}") + "_fold")
            remove.add(j); n1 += 1
    new_nodes = []
    for i, n in enumerate(nodes):
        if i in remove: continue
        new_nodes.append(replace.get(i, n))
    del m.graph.node[:]; m.graph.node.extend(new_nodes)

    nodes = list(m.graph.node)
    prod, cons = rebuild_consumers(nodes)
    remove = set(); replace_list = {}
    n2 = 0
    for j, n in enumerate(nodes):
        if n.op_type != "Or" or len(n.input) != 2:
            continue
        p0, p1 = prod.get(n.input[0]), prod.get(n.input[1])
        if p0 is None or p1 is None:
            continue
        n0, n1_ = nodes[p0], nodes[p1]
        if n0.op_type == "Not" and n1_.op_type == "Not" and len(cons[n0.output[0]]) == 1 and len(cons[n1_.output[0]]) == 1:
            tmp = n.output[0] + "_andtmp"
            replace_list[j] = [
                helper.make_node("And", [n0.input[0], n1_.input[0]], [tmp], name=(n.name or f"n{j}") + "_demorgan_and"),
                helper.make_node("Not", [tmp], list(n.output), name=(n.name or f"n{j}") + "_demorgan_not"),
            ]
            remove.add(p0); remove.add(p1); n2 += 1
    new_nodes = []
    for i, n in enumerate(nodes):
        if i in remove: continue
        if i in replace_list:
            new_nodes.extend(replace_list[i])
        else:
            new_nodes.append(n)
    del m.graph.node[:]; m.graph.node.extend(new_nodes)
    return n1, n2

def audit_one(model, t, prefix):
    san = ngu.sanitize_model(copy.deepcopy(model))
    if san is None:
        return None, None, None, "sanitize failed"
    try:
        onnx.checker.check_model(san, full_check=True)
    except Exception as e:
        return None, None, None, f"checker: {str(e)[:200]}"
    o = onnxruntime.SessionOptions()
    o.enable_profiling = True; o.log_severity_level = 3
    o.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    o.profile_file_prefix = prefix
    try:
        s = onnxruntime.InferenceSession(san.SerializeToString(), o)
    except Exception as e:
        return None, None, None, f"session-load: {str(e)[:200]}"
    d = load_task(t); nfail = 0
    for ex in d["train"] + d["test"] + d["arc-gen"]:
        b = ngu.convert_to_numpy(ex)
        if not b: continue
        try:
            out = ngu.run_network(s, b["input"])
            if not np.array_equal(out, b["output"]):
                nfail += 1
        except Exception:
            nfail += 1
    tp = s.end_profiling()
    mem, par = ngu.score_network(san, tp)
    try: os.remove(tp)
    except Exception: pass
    if mem is None or par is None or mem < 0 or par < 0:
        return nfail, None, None, "cost could not be measured"
    cost = mem + par
    pts = max(1.0, 25.0 - math.log(max(1.0, cost)))
    return nfail, cost, pts, "ok"

con = sqlite3.connect(os.path.join(REP, "tracker.db"))
cur = con.cursor()
cur.execute("SELECT task, state, base_points, our_points, our_cost, n_fail FROM tasks")
tracker = {row[0]: row for row in cur.fetchall()}

candidates = []
for t in range(1, 401):
    row = tracker.get(t)
    if row is None: continue
    _, state, base_points, our_points, our_cost, n_fail = row
    if our_cost == -1:
        continue  # known negative-pads unmeasurable-locally task, don't touch
    ours_path = os.path.join(REP, f"task{t:03d}.onnx")
    base_path = os.path.join(BASE, f"task{t:03d}.onnx")
    use_ours = os.path.exists(ours_path) and (our_points or 0) >= (base_points or 0)
    src_path = ours_path if use_ours else base_path
    if not os.path.exists(src_path):
        src_path = ours_path if os.path.exists(ours_path) else base_path
        if not os.path.exists(src_path):
            continue
    cur_best = max(base_points or 0, our_points or 0)

    try:
        m = onnx.load(src_path)
    except Exception:
        continue
    opset_default = next((o.version for o in m.opset_import if o.domain in ("", "ai.onnx")), 0)
    allow_cmp = opset_default >= 12
    n1, n2 = apply_folds(m, allow_cmp)
    if n1 == 0 and n2 == 0:
        continue
    candidates.append((t, src_path, cur_best, n1, n2, m))

print(f"Found {len(candidates)} tasks with at least one fold opportunity, running full audits...")

wins = []
for idx, (t, src_path, cur_best, n1, n2, m) in enumerate(candidates):
    nfail, cost, pts, status = audit_one(m, t, f"sweep_{t}")
    tag = f"task{t:03d} (from {'repairs' if 'repairs' in src_path else 'baseline'}, folds={n1}+{n2})"
    if status != "ok":
        print(f"  SKIP {tag}: {status}")
        continue
    if nfail != 0:
        print(f"  SKIP {tag}: nfail={nfail} (correctness broke)")
        continue
    if pts <= cur_best + 1e-9:
        print(f"  NOWIN {tag}: {cur_best:.4f} -> {pts:.4f} (not better)")
        continue
    print(f"  WIN  {tag}: {cur_best:.4f} -> {pts:.4f} (cost={cost})")
    cand_file = os.path.join(CAND_DIR, f"task{t:03d}.onnx")
    onnx.save(m, cand_file)
    wins.append((t, cur_best, pts, cost, cand_file))
    if (idx+1) % 10 == 0:
        print(f"...{idx+1}/{len(candidates)} candidates processed")

print(f"\n{len(wins)} verified strict wins out of {len(candidates)} candidates with fold opportunities.")
import json as _json
_json.dump([[t, cur_best, pts, cost, cand_file] for t, cur_best, pts, cost, cand_file in wins],
           open("scratch_onnx/demorgan_sweep_wins.json", "w"))
print("wins saved to scratch_onnx/demorgan_sweep_wins.json")
