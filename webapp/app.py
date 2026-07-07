import os, sys, io, math, json, copy, csv, traceback, sqlite3, zipfile, subprocess, base64, contextlib
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, Response
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

PROJ  = os.environ.get("PROJECT_DIR", ".")
DATA  = os.path.join(PROJ, "data")
BASE  = os.path.join(PROJ, "baseline_v22")
REP   = os.path.join(PROJ, "repairs")
UTILS = os.path.join(DATA, "neurogolf_utils")
sys.path.insert(0, UTILS)

import numpy as np
import onnx, onnxruntime
from onnx import helper, numpy_helper, TensorProto
import neurogolf_utils as ngu   # official scorer (correct source of truth)
from graph_compiler import compile_graph, DEFAULT_GRAPH, GraphCompileError

USERCODE_DIR = os.path.join(REP, "user_code"); os.makedirs(USERCODE_DIR, exist_ok=True)
PREDICTED_DIR = os.path.join(PROJ, "predicted")
CATALOG      = os.path.join(REP, "catalog.csv")
DB_FILE      = os.path.join(REP, "tracker.db")
SUBZIP       = os.path.join(PROJ, "submission.zip")

# Load autosolved tasks (generic rule-based solves with no per-task script)
_AUTOSOLVED_FILE = os.path.join(REP, "autosolved.json")
AUTOSOLVED = {}
if os.path.exists(_AUTOSOLVED_FILE):
    try:
        AUTOSOLVED = {int(k): v for k, v in json.load(open(_AUTOSOLVED_FILE)).items()}
    except Exception:
        pass

app = Flask(__name__)

# ---------------------------------------------------------------- data helpers
def load_task(t):
    return json.load(open(os.path.join(DATA, f"task{t:03d}.json")))

def parse_onnx_to_graph(onnx_path):
    if not os.path.exists(onnx_path):
        return None
    try:
        model = onnx.load(onnx_path)
    except Exception:
        return None
    graph = model.graph
    nodes, edges = [], []
    y_pos = 100; producer_map = {}
    
    for i, inp in enumerate(graph.input):
        node_id = inp.name
        nodes.append({"id": node_id, "op": "Input", "attrs": {}, "position": {"x": 50, "y": y_pos + i*150}})
        producer_map[inp.name] = (node_id, 0)
        
    init_map = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    consumed_inits = set()
    x_pos = 250
    
    for idx, n in enumerate(graph.node):
        node_id = n.output[0] if n.output else f"n_{idx}"
        op = n.op_type
        attrs = {}
        for attr in n.attribute:
            if attr.type == onnx.AttributeProto.INTS: attrs[attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.INT: attrs[attr.name] = int(attr.i)
            elif attr.type == onnx.AttributeProto.FLOAT: attrs[attr.name] = float(attr.f)
            elif attr.type == onnx.AttributeProto.FLOATS: attrs[attr.name] = list(attr.floats)
            elif attr.type == onnx.AttributeProto.STRING: attrs[attr.name] = attr.s.decode('utf-8')

        inputs = list(n.input)
        def consume_init(inp_idx, attr_name):
            if len(inputs) > inp_idx and inputs[inp_idx] in init_map:
                name = inputs[inp_idx]
                val = init_map[name]
                attrs[attr_name] = val.item() if val.size == 1 else val.tolist()
                consumed_inits.add(name)

        if op == "Slice":
            consume_init(1, "starts"); consume_init(2, "ends"); consume_init(3, "axes"); consume_init(4, "steps")
            inputs = inputs[:1]
        elif op == "Pad":
            consume_init(1, "pads"); consume_init(2, "value")
            inputs = inputs[:1]
        elif op == "Tile":
            consume_init(1, "repeats")
            inputs = inputs[:1]
        elif op == "Resize":
            consume_init(3, "sizes")
            if len(inputs) > 1 and inputs[1] in init_map: consumed_inits.add(inputs[1])
            if len(inputs) > 2 and inputs[2] in init_map: consumed_inits.add(inputs[2])
            inputs = inputs[:1]
        elif op == "Conv":
            consume_init(1, "weight")
            inputs = inputs[:1]
                
        nodes.append({"id": node_id, "op": op, "attrs": attrs, "position": {"x": x_pos, "y": y_pos + (idx % 10) * 80}})
        if idx % 10 == 9: x_pos += 200
        
        for port_idx, inp_name in enumerate(inputs):
            edges.append({"from": inp_name, "fromPort": 0, "to": node_id, "toPort": port_idx, "toPortStr": f"in{port_idx}"})
                
        for port_idx, out_name in enumerate(n.output):
            producer_map[out_name] = (node_id, port_idx)
            
    for name, val in init_map.items():
        if name in consumed_inits: continue
        val_list = val.tolist() if val.size < 1000 else []
        nodes.append({"id": name, "op": "Constant", "attrs": {"shape": list(val.shape), "value": val_list}, "position": {"x": 50, "y": y_pos + 400}})
        producer_map[name] = (name, 0)
        y_pos += 80
        
    x_pos += 200
    for i, out in enumerate(graph.output):
        node_id = out.name
        nodes.append({"id": node_id, "op": "Output", "attrs": {}, "position": {"x": x_pos, "y": 200 + i*150}})
        if out.name in producer_map:
            edges.append({"from": out.name, "fromPort": 0, "to": node_id, "toPort": 0, "toPortStr": "input"})
            
    final_edges = []
    for e in edges:
        inp_name = e["from"]
        if inp_name in producer_map:
            src_id, src_port = producer_map[inp_name]
            e["from"] = src_id; e["fromPort"] = src_port
            final_edges.append(e)

    return {"nodes": nodes, "edges": final_edges}

def task_examples(t):
    d = load_task(t)
    return d["train"] + d["test"]  # arc-gen omitted from the UI (too many); used only in verify

def audit(model, t):
    """Official audit: returns (n_fail, cost, points) or (None,...) on error."""
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
    o.profile_file_prefix = "web"
    try:
        s = onnxruntime.InferenceSession(san.SerializeToString(), o)
    except Exception as e:
        return None, None, None, f"load: {str(e)[:200]}"
    d = load_task(t); nfail = 0
    for ex_id, ex in enumerate(d["train"] + d["test"] + d["arc-gen"]):
        b = ngu.convert_to_numpy(ex)
        if not b: continue
        try:
            out = ngu.run_network(s, b["input"])
            if not np.array_equal(out, b["output"]): 
                nfail += 1
                diff = np.abs(out - b["output"])
                print(f"Mismatch: sum(diff)={np.sum(diff)}")
                wh = np.where(diff > 0)
                print(f"Mismatch in ex {ex_id} at (c, r, c): list(zip(wh[1], wh[2], wh[3]))")
                for w_c, w_r, w_col in zip(wh[1], wh[2], wh[3]):
                    print(f"  c={w_c}, r={w_r}, col={w_col}: out={out[0, w_c, w_r, w_col]}, b={b['output'][0, w_c, w_r, w_col]}")
                    
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

def audit_verbose(model, t, max_fails=40):
    """Like audit(), but also reports WHICH examples fail (split + index) instead of
    just a pass/fail count. Used by the Quick Check page only — does not touch audit()
    so the main task-editor page's behavior is unaffected."""
    san = ngu.sanitize_model(copy.deepcopy(model))
    if san is None:
        return None, None, None, "sanitize failed", []
    try:
        onnx.checker.check_model(san, full_check=True)
    except Exception as e:
        return None, None, None, f"checker: {str(e)[:300]}", []
    o = onnxruntime.SessionOptions()
    o.enable_profiling = True; o.log_severity_level = 3
    o.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    o.profile_file_prefix = "qc"
    try:
        s = onnxruntime.InferenceSession(san.SerializeToString(), o)
    except Exception as e:
        return None, None, None, f"load: {str(e)[:300]}", []
    d = load_task(t)
    splits = [("train", d["train"]), ("test", d["test"]), ("arc-gen", d["arc-gen"])]
    nfail = 0; fails = []; combined = 0
    for split_name, exs in splits:
        for i, ex in enumerate(exs):
            b = ngu.convert_to_numpy(ex)
            if not b:
                combined += 1; continue
            try:
                out = ngu.run_network(s, b["input"])
                if not np.array_equal(out, b["output"]):
                    nfail += 1
                    if len(fails) < max_fails:
                        diff = np.abs(out - b["output"])
                        wh = np.where(diff > 0)
                        ncells = len(set(zip(wh[2].tolist(), wh[3].tolist()))) if len(wh) >= 4 and len(wh[2]) else 0
                        fails.append({"split": split_name, "idx": i, "combined": combined,
                                      "kind": "mismatch", "detail": f"{ncells} cell(s) differ"})
            except Exception as e:
                nfail += 1
                if len(fails) < max_fails:
                    fails.append({"split": split_name, "idx": i, "combined": combined,
                                  "kind": "exception", "detail": str(e)[:200]})
            combined += 1
    tp = s.end_profiling()
    mem, par = ngu.score_network(san, tp)
    try: os.remove(tp)
    except Exception: pass
    if mem is None or par is None or mem < 0 or par < 0:
        return nfail, None, None, "cost could not be measured", fails
    cost = mem + par
    pts = max(1.0, 25.0 - math.log(max(1.0, cost)))
    return nfail, cost, pts, "ok", fails

def run_output_grids(model, t, k=4):
    """Run the model on the first few examples; return predicted grids (for live preview)."""
    san = ngu.sanitize_model(copy.deepcopy(model))
    o = onnxruntime.SessionOptions(); o.log_severity_level = 3
    o.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = onnxruntime.InferenceSession(san.SerializeToString(), o)
    d = load_task(t); out = []
    for ex in (d["train"] + d["test"])[:k]:
        b = ngu.convert_to_numpy(ex)
        if not b: out.append(None); continue
        r = ngu.run_network(s, b["input"])
        out.append(decode(r))
    return out

def draw_onnx_png(model, title=None, max_nodes=60):
    """Render an onnx model's graph as a PNG (layered DAG, or op-histogram if huge). Returns bytes."""
    nodes = list(model.graph.node)
    fig, ax = plt.subplots(figsize=(7.5, 6), facecolor="#0b0f18")
    ax.set_facecolor("#0b0f18")
    if len(nodes) > max_nodes:
        c = Counter(n.op_type for n in nodes).most_common()
        ax.barh([k for k, _ in c][::-1], [v for _, v in c][::-1], color="#34d399")
        ax.set_title(f"{title or ''}\n{len(nodes)} nodes — op-type counts", color="#e8ecf4", fontsize=10)
        ax.tick_params(colors="#8b93a7", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#2a2f3a")
    else:
        prod = {o: i for i, n in enumerate(nodes) for o in n.output}
        preds = {i: [prod[x] for x in n.input if x in prod] for i, n in enumerate(nodes)}
        layer = {}
        def L(i):
            if i not in layer:
                layer[i] = 0 if not preds[i] else max(L(p) for p in preds[i]) + 1
            return layer[i]
        for i in range(len(nodes)): L(i)
        byl = defaultdict(list)
        for i, l in layer.items(): byl[l].append(i)
        pos = {}
        for l, ids in byl.items():
            for k, i in enumerate(ids): pos[i] = (l, -(k - (len(ids)-1)/2))
        for i in range(len(nodes)):
            x2, y2 = pos[i]
            for p in preds[i]:
                x1, y1 = pos[p]
                ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#3b4252", lw=0.8))
        for i, n in enumerate(nodes):
            x, y = pos[i]
            ax.text(x, y, n.op_type, ha="center", va="center", fontsize=7, color="#04121f",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#34d399", ec="#22d3ee", lw=0.8))
        ax.set_title(f"{title or 'graph'} — {len(nodes)} nodes", color="#e8ecf4", fontsize=10)
        ax.axis("off"); ax.set_xlim(-0.6, max(byl)+0.6)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=95, bbox_inches="tight", facecolor="#0b0f18")
    plt.close(fig); buf.seek(0); return buf.read()

def decode(t4):
    """[1,10,30,30] one-hot -> 2D grid (crop trailing empty rows/cols)."""
    a = (t4 > 0)
    H = W = 30; grid = []
    for r in range(H):
        row = []
        for c in range(W):
            cols = [ch for ch in range(10) if a[0, ch, r, c]]
            row.append(cols[0] if len(cols) == 1 else (11 if cols else 10))
        grid.append(row)
    # trim empty (all 10/'no color') trailing rows & cols
    def empty(v): return all(x in (10,) for x in v)
    while grid and empty(grid[-1]): grid.pop()
    if grid:
        maxc = max((max([i for i,x in enumerate(r) if x not in (10,)], default=-1) for r in grid), default=-1)
        grid = [r[:maxc+1] for r in grid]
    return grid

# ---------------------------------------------------------------- SQLite store
OURS_MARKER = {45,127,384,135,146,149,240,347,1,3,6,16,87,140,179,223,276,309,337,380,385}

def db():
    c = sqlite3.connect(DB_FILE); c.row_factory = sqlite3.Row; return c

def init_db():
    c = db(); c.execute("""CREATE TABLE IF NOT EXISTS tasks(
        task INTEGER PRIMARY KEY, state TEXT, base_points REAL,
        our_points REAL, our_cost INTEGER, n_fail INTEGER)""")
    existing_cols = {r[1] for r in c.execute("PRAGMA table_info(tasks)").fetchall()}
    if "notes" not in existing_cols:
        c.execute("ALTER TABLE tasks ADD COLUMN notes TEXT DEFAULT ''")
    c.execute("""CREATE TABLE IF NOT EXISTS versions(
        id INTEGER PRIMARY KEY AUTOINCREMENT, task INTEGER, kind TEXT,
        content TEXT, points REAL, cost INTEGER, n_nodes INTEGER, ts TEXT)""")
    c.commit()
    n = c.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    if n == 0:
        base = {}
        if os.path.exists(CATALOG):
            for r in csv.DictReader(open(CATALOG, encoding="utf-8")):
                try: base[int(r["task"])] = float(r["points"]) if r["points"] else 1.0
                except: pass
        for t in range(1, 401):
            op = oc = None; nf = None; state = "todo"
            p = os.path.join(REP, f"task{t:03d}.onnx")
            if os.path.exists(p) and t in OURS_MARKER:      # one-time audit of our existing solves
                try:
                    nfail, cost, pts, _ = audit(onnx.load(p), t)
                    if nfail == 0 and cost is not None:
                        op, oc, nf, state = round(pts,3), int(cost), 0, "ours"
                except Exception: pass
            c.execute("INSERT INTO tasks(task,state,base_points,our_points,our_cost,n_fail,notes) VALUES(?,?,?,?,?,?,?)",
                      (t, state, round(base.get(t,1.0),3), op, oc, nf, ""))
        c.commit()
    c.close()

def get_task(t):
    c = db(); r = c.execute("SELECT * FROM tasks WHERE task=?", (t,)).fetchone(); c.close(); return dict(r)

def set_task(t, **kw):
    c = db()
    cols = ",".join(f"{k}=?" for k in kw); vals = list(kw.values()) + [t]
    c.execute(f"UPDATE tasks SET {cols} WHERE task=?", vals); c.commit(); c.close()

def all_tasks():
    c = db(); rows = [dict(r) for r in c.execute("SELECT * FROM tasks ORDER BY task").fetchall()]; c.close(); return rows

def totals():
    total = 0.0; coded = 0
    for e in all_tasks():
        total += max(e["our_points"] or 0.0, e["base_points"] or 1.0)
        if e["state"] == "ours": coded += 1
    return round(total, 2), coded

def baseline_total():
    return round(sum((e["base_points"] or 1.0) for e in all_tasks()), 2)

# ---------------------------------------------------------------- version history
# every PASSING run/compile is appended here (never overwritten) so old solutions
# stay recoverable even after you paste in new, possibly-failing, code.
def add_version(t, kind, content, points, cost, n_nodes):
    c = db()
    c.execute("INSERT INTO versions(task,kind,content,points,cost,n_nodes,ts) VALUES(?,?,?,?,?,?,?)",
              (t, kind, content, points, cost, n_nodes, datetime.utcnow().isoformat(timespec="seconds")))
    c.commit(); c.close()

def list_versions(t):
    c = db()
    rows = [dict(r) for r in c.execute(
        "SELECT id,kind,points,cost,n_nodes,ts FROM versions WHERE task=? ORDER BY id DESC", (t,)).fetchall()]
    c.close(); return rows

def get_version(vid):
    c = db()
    r = c.execute("SELECT * FROM versions WHERE id=?", (vid,)).fetchone()
    c.close(); return dict(r) if r else None

def latest_version(t, kind):
    c = db()
    r = c.execute("SELECT * FROM versions WHERE task=? AND kind=? ORDER BY id DESC LIMIT 1", (t, kind)).fetchone()
    c.close(); return dict(r) if r else None

# ---------------------------------------------------------------- routes
@app.route("/")
def index():
    total, coded = totals(); base = baseline_total()
    tasks = [{"t": e["task"], "state": e["state"], "our_points": e["our_points"],
              "base_points": e["base_points"], "n_fail": e["n_fail"],
              "has_notes": bool((e.get("notes") or "").strip())} for e in all_tasks()]
    return render_template("index.html", tasks=tasks, total=total, coded=coded,
                           base=base, gain=round(total-base, 2))

@app.route("/task/<int:t>")
def task_page(t):
    e = get_task(t)
    ex = task_examples(t)
    grids = [{"input": x["input"], "output": x["output"]} for x in ex[:5]]
    n_versions = len(list_versions(t))
    # visual-graph state: draft (last attempt, pass or fail) else last-passing else default template
    gf = os.path.join(USERCODE_DIR, f"task{t:03d}.graph.json")
    gdraft_f = os.path.join(USERCODE_DIR, f"task{t:03d}.draft.graph.json")

    graph_json = None
    if os.path.exists(gdraft_f):
        graph_json = open(gdraft_f, encoding="utf-8").read()
    elif os.path.exists(gf):
        graph_json = open(gf, encoding="utf-8").read()
    else:
        onnx_path = os.path.join(REP, f"task{t:03d}.onnx")
        print(f"DEBUG: Checking onnx_path: {onnx_path}")
        if os.path.exists(onnx_path):
            print(f"DEBUG: {onnx_path} exists!")
            try:
                g = parse_onnx_to_graph(onnx_path)
                print(f"DEBUG: parse_onnx_to_graph returned a graph with {len(g['nodes'])} nodes" if g else "DEBUG: parse_onnx_to_graph returned None")
                graph_json = json.dumps(g) if g else json.dumps(DEFAULT_GRAPH)
            except Exception as e:
                import traceback
                print(f"DEBUG: Exception during parse_onnx_to_graph: {e}")
                traceback.print_exc()
                graph_json = json.dumps(DEFAULT_GRAPH)
        else:
            print(f"DEBUG: {onnx_path} DOES NOT EXIST!")
            graph_json = json.dumps(DEFAULT_GRAPH)

    code = _resolve_code(t, include_drafts=True)
    
    # baseline op summary
    try:
        bm = onnx.load(os.path.join(BASE, f"task{t:03d}.onnx"))
        from collections import Counter
        bops = "; ".join(f"{k}:{v}" for k,v in Counter(n.op_type for n in bm.graph.node).most_common())
    except Exception:
        bops = "(baseline not found)"
    return render_template("task.html", t=t, e=e, grids=json.dumps(grids), code=code, bops=bops,
                           n_versions=n_versions, graph_json=graph_json)

@app.route("/api/status/<int:t>", methods=["POST"])
def set_status(t):
    newstate = request.json.get("state", "todo")
    set_task(t, state=newstate)
    total, coded = totals()
    return jsonify(ok=True, state=newstate, total=total, coded=coded)

@app.route("/api/notes/<int:t>", methods=["POST"])
def save_notes(t):
    notes = request.json.get("notes", "")
    set_task(t, notes=notes)
    return jsonify(ok=True)

@app.route("/api/run/<int:t>", methods=["POST"])
def run_code(t):
    code = request.json.get("code", "")
    ns = {"onnx": onnx, "helper": helper, "numpy_helper": numpy_helper, "np": np,
          "TensorProto": TensorProto, "T": t}
    try:
        exec(code, ns)
    except Exception:
        return jsonify(ok=False, error="Python error:\n" + traceback.format_exc()[-1500:])
    draft_f = os.path.join(USERCODE_DIR, f"task{t:03d}.draft.py")
    good_f  = os.path.join(USERCODE_DIR, f"task{t:03d}.py")
    # the draft always reflects your most recent attempt (pass or fail) so a page
    # refresh never loses in-progress work; the good file / version history only
    # ever advance on a PASSING run, so a failed paste can never destroy history.
    open(draft_f, "w", encoding="utf-8").write(code)
    model = ns.get("model")
    if model is None:
        # not ONNX code — maybe a plain-Python reference algorithm (a solve_*() function).
        fn = next((v for k, v in ns.items() if callable(v) and k.startswith("solve")), None)
        if fn is not None:
            d = load_task(t); allex = d["train"] + d["test"] + d["arc-gen"]; nfail = 0
            for ex in allex:
                try:
                    pred = fn(ex["input"])
                    if [list(r) for r in pred] != [list(r) for r in ex["output"]]: nfail += 1
                except Exception: nfail += 1
            prev = []
            for ex in (d["train"] + d["test"])[:4]:
                try: prev.append([list(r) for r in fn(ex["input"])])
                except Exception: prev.append(None)
            if nfail == 0:
                open(good_f, "w", encoding="utf-8").write(code)
                add_version(t, "code", code, None, None, None)
                if get_task(t)["state"] == "todo": set_task(t, state="working")
            total, coded = totals()
            return jsonify(ok=True, algo=True, n_fail=nfail, n_total=len(allex),
                           preview=prev, total=total, coded=coded, state=get_task(t)["state"])
        return jsonify(ok=False, error="No ONNX model and no solve_*() function found. Either assign an "
                       "onnx model to `model`, or define a Python function named solve...(input_grid) to test the algorithm.")
    try:
        nfail, cost, pts, msg = audit(model, t)
    except Exception:
        return jsonify(ok=False, error="Audit error:\n" + traceback.format_exc()[-1200:])
    if nfail is None:
        return jsonify(ok=False, error=f"Invalid model: {msg}")
    e = get_task(t); preview = None
    solved = (nfail == 0 and cost is not None)
    if solved:
        onnx.save(model, os.path.join(REP, f"task{t:03d}.onnx"))
        open(good_f, "w", encoding="utf-8").write(code)
        add_version(t, "code", code, round(pts,3), int(cost), len(model.graph.node))
        set_task(t, state="ours", our_points=round(pts,3), our_cost=int(cost), n_fail=0)
        try: preview = run_output_grids(model, t)
        except Exception: preview = None
    else:
        newstate = e["state"] if e["state"] == "ours" else ("working" if e["state"]=="todo" else e["state"])
        set_task(t, n_fail=nfail, state=newstate)
    e = get_task(t); total, coded = totals()
    # graph image of the model they just ran (base64), so the UI can show its structure live
    graph_b64 = None
    try:
        graph_b64 = "data:image/png;base64," + base64.b64encode(
            draw_onnx_png(model, title=("your model · task%03d" % t))).decode()
    except Exception:
        pass
    return jsonify(ok=True, solved=solved, n_fail=nfail, cost=cost, points=(round(pts,3) if pts else None),
                   total=total, coded=coded, state=e["state"], preview=preview,
                   base_points=e["base_points"], graph=graph_b64, n_nodes=len(model.graph.node),
                   n_versions=len(list_versions(t)))

@app.route("/api/compile_graph/<int:t>", methods=["POST"])
def compile_graph_route(t):
    graph = request.json.get("graph", {})
    graph_txt = json.dumps(graph)
    draft_f = os.path.join(USERCODE_DIR, f"task{t:03d}_graph.draft.json")
    good_f  = os.path.join(USERCODE_DIR, f"task{t:03d}_graph.json")
    # same draft/good split as the code path: draft always reflects your latest attempt,
    # good + version history only ever advance on a PASSING compile.
    open(draft_f, "w", encoding="utf-8").write(graph_txt)
    try:
        model = compile_graph(graph)
    except GraphCompileError as e:
        return jsonify(ok=False, error=f"Graph error: {e}")
    except Exception:
        return jsonify(ok=False, error="Graph compile error:\n" + traceback.format_exc()[-1200:])
    try:
        nfail, cost, pts, msg = audit(model, t)
    except Exception:
        return jsonify(ok=False, error="Audit error:\n" + traceback.format_exc()[-1200:])
    if nfail is None:
        return jsonify(ok=False, error=f"Invalid model: {msg}")
    e = get_task(t); preview = None
    solved = (nfail == 0 and cost is not None)
    if solved:
        onnx.save(model, os.path.join(REP, f"task{t:03d}.onnx"))
        open(good_f, "w", encoding="utf-8").write(graph_txt)
        add_version(t, "graph", graph_txt, round(pts,3), int(cost), len(model.graph.node))
        set_task(t, state="ours", our_points=round(pts,3), our_cost=int(cost), n_fail=0)
        try: preview = run_output_grids(model, t)
        except Exception: preview = None
    else:
        newstate = e["state"] if e["state"] == "ours" else ("working" if e["state"]=="todo" else e["state"])
        set_task(t, n_fail=nfail, state=newstate)
    e = get_task(t); total, coded = totals()
    graph_b64 = None
    try:
        graph_b64 = "data:image/png;base64," + base64.b64encode(
            draw_onnx_png(model, title=("your graph · task%03d" % t))).decode()
    except Exception:
        pass
    return jsonify(ok=True, solved=solved, n_fail=nfail, cost=cost, points=(round(pts,3) if pts else None),
                   total=total, coded=coded, state=e["state"], preview=preview,
                   base_points=e["base_points"], graph=graph_b64, n_nodes=len(model.graph.node),
                   n_versions=len(list_versions(t)))

@app.route("/api/versions/<int:t>")
def api_versions(t):
    return jsonify(ok=True, versions=list_versions(t))

@app.route("/api/version/<int:vid>")
def api_version(vid):
    v = get_version(vid)
    if v is None or "id" not in v:
        return jsonify(ok=False, error="version not found")
    return jsonify(ok=True, version=v)

@app.route("/api/reset/<int:t>")
def api_reset(t):
    """Return the most recent PASSING code/graph for this task (or a starter template
    if it has never passed), so the Reset button can restore known-good state."""
    lv_code = latest_version(t, "code")
    lv_graph = latest_version(t, "graph")
    if lv_code:
        code = lv_code["content"]
    else:
        # Use the same fallback chain as task_page but skip drafts (reset = known-good)
        code = _resolve_code(t, include_drafts=False)
    graph = json.loads(lv_graph["content"]) if lv_graph else DEFAULT_GRAPH
    # restoring means the draft is that known-good state again too
    open(os.path.join(USERCODE_DIR, f"task{t:03d}.draft.py"), "w", encoding="utf-8").write(code)
    open(os.path.join(USERCODE_DIR, f"task{t:03d}_graph.draft.json"), "w", encoding="utf-8").write(json.dumps(graph))
    return jsonify(ok=True, code=code, graph=graph, had_code=bool(lv_code), had_graph=bool(lv_graph))

# ---------------------------------------------------------------- scoreboard (standalone page)
@app.route("/scoreboard")
def scoreboard_page():
    rows = all_tasks()
    data = []
    for e in rows:
        base = round(e["base_points"] if e["base_points"] is not None else 1.0, 3)
        ours = e["our_points"]
        delta = round(ours - base, 3) if ours is not None else None
        verdict = "none"
        if ours is not None:
            if ours > base: verdict = "surpass"
            elif ours == base: verdict = "tie"
            else: verdict = "below"
        data.append({"t": e["task"], "state": e["state"], "base": base, "ours": ours,
                      "cost": e["our_cost"], "n_fail": e["n_fail"], "delta": delta, "verdict": verdict})
    total, coded = totals()
    base_total = baseline_total()
    surpass_count = sum(1 for d in data if d["verdict"] == "surpass")
    tie_count = sum(1 for d in data if d["verdict"] == "tie")
    below_count = sum(1 for d in data if d["verdict"] == "below")
    return render_template("scoreboard.html", data=json.dumps(data), total=total, coded=coded,
                           base_total=base_total, gain=round(total - base_total, 2),
                           surpass_count=surpass_count, tie_count=tie_count, below_count=below_count,
                           n_tasks=len(data))

# ---------------------------------------------------------------- quick check (standalone page)
@app.route("/quickcheck")
def quickcheck_page():
    return render_template("quickcheck.html")

@app.route("/api/quickcheck", methods=["POST"])
def api_quickcheck():
    try:
        t = int(request.json.get("task"))
    except Exception:
        return jsonify(ok=False, error="Enter a valid task number.")
    if not (1 <= t <= 400):
        return jsonify(ok=False, error="Task number must be between 1 and 400.")
    code = request.json.get("code", "")
    e = get_task(t)

    log = []
    buf = io.StringIO()
    ns = {"onnx": onnx, "helper": helper, "numpy_helper": numpy_helper, "np": np, "TensorProto": TensorProto, "T": t}
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, ns)
    except Exception:
        log.append(buf.getvalue())
        log.append(traceback.format_exc())
        return jsonify(ok=True, solved=False, stage="exec_error", log="\n".join(p for p in log if p),
                       state=e["state"], base_points=e["base_points"])
    log.append(buf.getvalue())

    model = ns.get("model")
    if model is None:
        log.append("No `model` (onnx.ModelProto) variable was set at module scope.")
        return jsonify(ok=True, solved=False, stage="no_model", log="\n".join(p for p in log if p),
                       state=e["state"], base_points=e["base_points"])

    buf2 = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf2):
            nfail, cost, pts, msg, fails = audit_verbose(model, t)
    except Exception:
        log.append(buf2.getvalue())
        log.append(traceback.format_exc())
        return jsonify(ok=True, solved=False, stage="audit_error", log="\n".join(p for p in log if p),
                       state=e["state"], base_points=e["base_points"])
    log.append(buf2.getvalue())

    if nfail is None:
        log.append(f"Invalid model: {msg}")
        return jsonify(ok=True, solved=False, stage="invalid", log="\n".join(p for p in log if p),
                       state=e["state"], base_points=e["base_points"])

    solved = (nfail == 0 and cost is not None)
    good_f = os.path.join(USERCODE_DIR, f"task{t:03d}.py")
    if solved:
        onnx.save(model, os.path.join(REP, f"task{t:03d}.onnx"))
        open(good_f, "w", encoding="utf-8").write(code)
        add_version(t, "code", code, round(pts, 3), int(cost), len(model.graph.node))
        set_task(t, state="ours", our_points=round(pts, 3), our_cost=int(cost), n_fail=0)
    else:
        newstate = e["state"] if e["state"] == "ours" else ("working" if e["state"] == "todo" else e["state"])
        set_task(t, n_fail=nfail, state=newstate)
    e2 = get_task(t)

    graph_b64 = None
    try:
        graph_b64 = "data:image/png;base64," + base64.b64encode(
            draw_onnx_png(model, title=("quickcheck · task%03d" % t))).decode()
    except Exception:
        pass

    return jsonify(ok=True, solved=solved, stage="result", n_fail=nfail,
                   cost=cost, points=(round(pts, 3) if pts else None), base_points=e2["base_points"],
                   state=e2["state"], n_nodes=len(model.graph.node), graph=graph_b64, fails=fails,
                   log="\n".join(p for p in log if p))

# ---------------------------------------------------------------- upload & verify
@app.route("/uploadcheck")
def uploadcheck_page():
    return render_template("uploadcheck.html")

@app.route("/api/upload_verify", methods=["POST"])
def api_upload_verify():
    try:
        t = int(request.form.get("task"))
    except Exception:
        return jsonify(ok=False, error="Enter a valid task number.")
    if not (1 <= t <= 400):
        return jsonify(ok=False, error="Task number must be between 1 and 400.")
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(ok=False, error="Choose an ONNX file to upload.")

    raw = f.read()
    try:
        model = onnx.load_model_from_string(raw)
    except Exception as e:
        return jsonify(ok=False, error=f"Could not parse ONNX file: {str(e)[:300]}")

    e = get_task(t)
    old_points, old_cost, base_points = e["our_points"], e["our_cost"], e["base_points"]
    unmeasurable = (old_cost == -1)
    old_best = 25.0 if unmeasurable else max(old_points or 0.0, base_points or 0.0)

    try:
        nfail, cost, pts, msg = audit(model, t)
    except Exception:
        return jsonify(ok=False, error="Error while auditing the uploaded model:\n" + traceback.format_exc()[-1200:])

    if nfail is None:
        return jsonify(ok=True, valid=False, message=msg, old_points=old_points, old_cost=old_cost,
                       base_points=base_points, unmeasurable=unmeasurable, state=e["state"])

    is_correct = (nfail == 0 and cost is not None)
    is_better = is_correct and (not unmeasurable) and pts is not None and pts > old_best + 1e-9

    saved = False
    if is_better and request.form.get("save") == "1":
        onnx.save(model, os.path.join(REP, f"task{t:03d}.onnx"))
        set_task(t, our_points=round(pts, 4), our_cost=int(cost), n_fail=0, state="ours")
        add_version(t, "onnx-upload", base64.b64encode(raw).decode(), round(pts, 4), int(cost), len(model.graph.node))
        saved = True

    return jsonify(ok=True, valid=True, n_fail=nfail, cost=cost,
                   points=(round(pts, 4) if pts is not None else None),
                   is_correct=is_correct, is_better=is_better, saved=saved, unmeasurable=unmeasurable,
                   old_points=old_points, old_cost=old_cost, base_points=base_points,
                   old_best=round(old_best, 4), state=e["state"], n_nodes=len(model.graph.node))

# ---------------------------------------------------------------- submission
def build_submission_zip():
    """Pick the better ONNX per task (ours if our_points>=base_points, else baseline); write submission.zip.
    This is the ONE invariant Build/Submit must always uphold: every task in the resulting zip is the
    best verified option we have. Exception: our_cost==-1 always forces the repairs/ file regardless
    of the points comparison -- this flags negative-pads models our local checker can't score (see
    notes), confirmed higher on the real Kaggle grader via an isolated test submission, not by
    our_points (which is a fair-share estimate for display only, not per-task-accurate).
    Fails loudly (raises) rather than silently shipping a broken/incomplete zip -- a missing source
    file or a wrong final member count means something is wrong with the tracker/repo state, not
    something to paper over."""
    rows = {e["task"]: e for e in all_tasks()}
    used_ours = used_forced = 0
    if os.path.exists(SUBZIP): os.remove(SUBZIP)
    with zipfile.ZipFile(SUBZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for t in range(1, 401):
            fn = f"task{t:03d}.onnx"
            e = rows[t]; ours = os.path.join(REP, fn); basep = os.path.join(BASE, fn)
            pick = basep
            if e["our_cost"] == -1 and os.path.exists(ours):
                pick = ours; used_ours += 1; used_forced += 1
            elif os.path.exists(ours) and (e["our_points"] or 0) >= (e["base_points"] or 0):
                pick = ours; used_ours += 1
            elif os.path.exists(ours) and not os.path.exists(basep):
                pick = ours; used_ours += 1
            if not os.path.exists(pick):
                raise FileNotFoundError(f"task{t:03d}: no valid source file found (checked {ours} and {basep})")
            z.write(pick, fn)

    with zipfile.ZipFile(SUBZIP) as z:
        names = z.namelist()
        expected = {f"task{t:03d}.onnx" for t in range(1, 401)}
        if len(names) != 400 or len(set(names)) != 400 or set(names) != expected:
            raise ValueError(f"submission.zip integrity check failed: got {len(names)} members "
                              f"({len(set(names))} unique), expected exactly the 400 task files")
    return used_ours, used_forced

@app.route("/api/build_submission", methods=["POST"])
def api_build():
    try:
        used, forced = build_submission_zip()
        size = os.path.getsize(SUBZIP)
        total, coded = totals()
        return jsonify(ok=True, used_ours=used, used_forced=forced, size=size, path=SUBZIP, total=total)
    except Exception:
        return jsonify(ok=False, error=traceback.format_exc()[-1200:])

@app.route("/api/submit", methods=["POST"])
def api_submit():
    msg = request.json.get("message", "tracker submission")
    try:
        used, forced = build_submission_zip()
    except Exception:
        return jsonify(ok=False, error="zip build failed:\n"+traceback.format_exc()[-800:])
    try:
        r = subprocess.run([sys.executable, "-m", "kaggle", "competitions", "submit",
                            "neurogolf-2026", "-f", SUBZIP, "-m", msg],
                           capture_output=True, text=True, timeout=300)
        out = (r.stdout or "") + (r.stderr or "")
        ok = ("Successfully submitted" in out) or (r.returncode == 0 and "error" not in out.lower())
        return jsonify(ok=ok, used_ours=used, used_forced=forced, output=out[-1500:])
    except Exception:
        return jsonify(ok=False, used_ours=used, used_forced=forced, error="kaggle submit failed (creds mounted?):\n"+traceback.format_exc()[-800:])

@app.route("/download_submission")
def download_sub():
    if not os.path.exists(SUBZIP): build_submission_zip()
    return send_file(SUBZIP, as_attachment=True, download_name="submission.zip")

@app.route("/graph_img/<int:t>")
def graph_img(t):
    which = request.args.get("which", "auto")
    ours = os.path.join(REP, f"task{t:03d}.onnx"); basep = os.path.join(BASE, f"task{t:03d}.onnx")
    path = ours if (which == "ours" or (which == "auto" and os.path.exists(ours))) else basep
    title = ("OURS" if path == ours else "baseline") + f" · task{t:03d}"
    try:
        png = draw_onnx_png(onnx.load(path), title=title)
        return Response(png, mimetype="image/png")
    except Exception:
        return Response(b"", mimetype="image/png")

SOLVED_PREFILL = '''# task {t} is ALREADY SOLVED (our verified model). Click "Run & Verify" -> it turns green.
# This loads the saved model. The full FROM-SCRATCH construction code is in from_scratch.ipynb.
# To rewrite it yourself, replace the two lines below with your own graph that assigns `model`.
import onnx
model = onnx.load("{path}")
'''

AUTOSOLVED_PREFILL = '''# Task {t} — solved by generic rule: "{rule}"
# No per-task construction script exists. This task was solved by a rule-based
# category solver ({rule}) that handles multiple tasks at once.
# The verified model is at repairs/task{t:03d}.onnx
import onnx
model = onnx.load("{path}")
'''

STARTER = '''# Build an ONNX model for task {t}. Assign the final onnx.ModelProto to `model`.
# Available names: onnx, helper, numpy_helper, np, TensorProto, T (=task number)
# Contract: input tensor 'input' [1,10,30,30] one-hot; produce 'output' [1,10,30,30].
x = helper.make_tensor_value_info('input',  TensorProto.FLOAT, [1,10,30,30])
y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1,10,30,30])
node = helper.make_node('Identity', ['input'], ['output'])
model = helper.make_model(helper.make_graph([node],'g',[x],[y],[]),
                          ir_version=10, opset_imports=[helper.make_opsetid('',12)])
'''

def _resolve_code(t, include_drafts=True):
    """Resolve the best source code to show for a task.

    Priority:
      1. Draft file (most recent attempt, pass or fail) — only if include_drafts
      2. Good file in user_code/ (last passing code)
      3. Predicted source (predicted/test_onnx_taskNNN.py) — real construction code
      4. Autosolved note — for rule-based tasks with no per-task script
      5. ONNX-load stub — fallback for solved tasks with no source
      6. Starter template — for unsolved tasks
    """
    cf = os.path.join(USERCODE_DIR, f"task{t:03d}.py")
    draft_f = os.path.join(USERCODE_DIR, f"task{t:03d}.draft.py")
    onnx_path = os.path.join(REP, f"task{t:03d}.onnx")
    pred_f = os.path.join(PREDICTED_DIR, f"test_onnx_task{t:03d}.py")
    if not os.path.exists(pred_f):
        pred_f = os.path.join(PREDICTED_DIR, f"test_onnx_task{t}.py")

    # 1. Draft
    if include_drafts and os.path.exists(draft_f):
        return open(draft_f, encoding="utf-8").read()

    # 2. Good file in user_code/
    if os.path.exists(cf):
        return open(cf, encoding="utf-8").read()

    # 3. Predicted source from exploration
    if os.path.exists(pred_f):
        src = open(pred_f, encoding="utf-8").read()
        return _adapt_predicted(src, t, onnx_path)

    # 4. Autosolved
    if t in AUTOSOLVED and os.path.exists(onnx_path):
        return AUTOSOLVED_PREFILL.format(t=t, rule=AUTOSOLVED[t],
                                         path=onnx_path.replace("\\", "/"))

    # 5. ONNX-load stub
    if os.path.exists(onnx_path):
        return SOLVED_PREFILL.format(t=t, path=onnx_path.replace("\\", "/"))

    # 6. Starter template
    return STARTER.format(t=t)


def _adapt_predicted(src, t, onnx_path):
    """Read a predicted/test_onnx_taskNNN.py and return it with a header comment.

    The source is shown read-only in the editor. We prepend a header explaining
    the provenance and append `model = onnx.load(...)` so clicking Run still works.
    """
    import re as _re
    header = (
        f"# === Source: predicted/test_onnx_task{t:03d}.py ===\n"
        f"# This is the real ONNX graph construction code for task {t}.\n"
        f"# The verified model is loaded below so Run & Verify works immediately.\n"
        f"# To modify, edit the code above and re-run.\n\n"
    )
    # Strip the if __name__ block so it doesn't auto-execute side effects
    cleaned = _re.sub(r"\nif __name__\s*==\s*['\"]__main__['\"]:\s*\n.*", "", src, flags=_re.DOTALL)
    footer = (
        f"\n\n# --- Load the verified model so Run & Verify works ---\n"
        f"import onnx as _onnx\n"
        f"model = _onnx.load(\"{onnx_path.replace(chr(92), '/')}\")"
    )
    return header + cleaned.rstrip() + footer

if __name__ == "__main__":
    print("Initializing SQLite tracker DB (one-time audit of existing solves)...")
    init_db()
    print("Ready.")
    app.run(host="0.0.0.0", port=5000, debug=False)
