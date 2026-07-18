import os, sys, json, copy, math, sqlite3
sys.path.insert(0, os.path.join("data", "neurogolf_utils"))
import onnx, onnxruntime
import numpy as np
from onnx import helper, numpy_helper
from onnx import shape_inference
from collections import defaultdict
import neurogolf_utils as ngu

REP = "repairs"
CAND_DIR = "scratch_onnx/gathersplit_sweep"
os.makedirs(CAND_DIR, exist_ok=True)

def load_task(t):
    return json.load(open(os.path.join("data", f"task{t:03d}.json")))

def get_axis(n, default=0):
    for a in n.attribute:
        if a.name == "axis":
            return a.i
    return default

def resolve_scalar_const(name, init_map):
    init = init_map.get(name)
    if init is None:
        return None
    if init.data_type not in (onnx.TensorProto.INT64, onnx.TensorProto.INT32):
        return None
    arr = numpy_helper.to_array(init)
    if arr.size != 1:
        return None
    return int(arr.reshape(-1)[0])

def get_opset(model):
    for o in model.opset_import:
        if o.domain in ("", "ai.onnx"):
            return o.version
    return 0

def try_fold(model):
    opset = get_opset(model)
    try:
        inferred = shape_inference.infer_shapes(model, check_type=True, strict_mode=False)
    except Exception:
        inferred = model
    shape_map = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        dims = []
        ok = True
        for d in vi.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                dims.append(d.dim_value)
            else:
                ok = False
                break
        if ok:
            shape_map[vi.name] = dims

    init_map = {i.name: i for i in model.graph.initializer}
    groups = defaultdict(list)
    for idx, n in enumerate(model.graph.node):
        if n.op_type != "Gather" or len(n.input) != 2:
            continue
        axis = get_axis(n)
        groups[(n.input[0], axis)].append((idx, n))

    # BUG FIX (2026-07-08): the original version rebuilt model.graph.node inside this loop after
    # EVERY group, which invalidated the positional `idx` values stored for subsequent groups (once
    # nodes are removed/inserted, everyone after that point shifts). That caused later groups to
    # remove the WRONG nodes while still inserting a Split with the same output names -> a spurious
    # SSA "duplicate output" checker failure that looked like a genuine data conflict but wasn't
    # (this is exactly what caused task132/243/366 to be defensively skipped before). Fix: collect
    # every valid group's (node-object-set-to-remove, split-node-to-insert) FIRST, across the whole
    # model, without touching model.graph.node at all -- then do exactly ONE rebuild pass at the end
    # keyed by node OBJECT IDENTITY (id()), which is stable regardless of list position.
    all_seen_outputs = set()
    to_remove_objs = set()
    insertions = {}  # id(first_removed_node) -> split_node, so we know where to splice it in
    total_folds = 0
    for (src, axis), members in groups.items():
        if len(members) < 3:
            continue
        shape = shape_map.get(src)
        if not shape or axis >= len(shape) or axis < -len(shape):
            continue
        dim_size = shape[axis if axis >= 0 else axis + len(shape)]
        if dim_size <= 0 or dim_size != len(members):
            continue
        idx_to_output = {}
        valid = True
        for idx, n in members:
            v = resolve_scalar_const(n.input[1], init_map)
            if v is None or not (0 <= v < dim_size) or v in idx_to_output:
                valid = False
                break
            idx_to_output[v] = n.output[0]
        if not valid or len(idx_to_output) != dim_size:
            continue

        outputs = [idx_to_output[i] for i in range(dim_size)]
        if len(set(outputs)) != len(outputs) or any(o in all_seen_outputs for o in outputs):
            continue  # genuine cross-group name clash - skip defensively for real this time
        if opset >= 18:
            split_node = helper.make_node("Split", [src], outputs,
                                           name=f"split_fold_{src}_{axis}", axis=axis, num_outputs=dim_size)
        else:
            split_node = helper.make_node("Split", [src], outputs,
                                           name=f"split_fold_{src}_{axis}", axis=axis)
        all_seen_outputs.update(outputs)
        first_node = members[0][1]
        insertions[id(first_node)] = split_node
        for _, n in members:
            to_remove_objs.add(id(n))
        total_folds += 1

    if total_folds:
        new_nodes = []
        for n in model.graph.node:
            if id(n) in insertions:
                new_nodes.append(insertions[id(n)])
            if id(n) in to_remove_objs:
                continue
            new_nodes.append(n)
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
    return total_folds

def dce(model):
    needed = {o.name for o in model.graph.output}
    kept_nodes = []
    for n in reversed(model.graph.node):
        if any(y in needed for y in n.output):
            kept_nodes.append(n)
            for x in n.input:
                if x: needed.add(x)
    kept_nodes.reverse()
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    used = set()
    for n in model.graph.node:
        used.update(x for x in n.input if x)
    keep = [init for init in model.graph.initializer if init.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)

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
cur.execute("SELECT task, our_points, our_cost FROM tasks")
tracker = {row[0]: row for row in cur.fetchall()}

candidates = []
for t in range(1, 401):
    if t == 286:
        continue  # already hand-verified and merged
    row = tracker.get(t)
    if row is None: continue
    _, our_points, our_cost = row
    if our_cost == -1:
        continue
    path = os.path.join(REP, f"task{t:03d}.onnx")
    if not os.path.exists(path):
        continue
    try:
        m = onnx.load(path)
    except Exception:
        continue
    n_gather_before = sum(1 for n in m.graph.node if n.op_type == "Gather")
    if n_gather_before < 3:
        continue
    try:
        folds = try_fold(m)
    except Exception as e:
        print(f"task{t:03d}: fold attempt raised {str(e)[:150]}")
        continue
    if folds == 0:
        continue
    dce(m)
    try:
        onnx.checker.check_model(m)
    except Exception as e:
        print(f"task{t:03d}: {folds} fold(s) applied but checker FAILED: {str(e)[:150]}")
        continue
    candidates.append((t, our_points or 0.0, m, folds))

print(f"Found {len(candidates)} tasks with a valid Gather->Split fold opportunity")

wins = []
for t, cur_best, m, folds in candidates:
    nfail, cost, pts, status = audit_one(m, t, f"gathersplit_{t}")
    if status != "ok":
        print(f"  SKIP task{t:03d} ({folds} folds): {status}")
        continue
    if nfail != 0:
        print(f"  SKIP task{t:03d} ({folds} folds): nfail={nfail}")
        continue
    if pts <= cur_best + 1e-9:
        print(f"  NOWIN task{t:03d} ({folds} folds): {cur_best:.4f} -> {pts:.4f}")
        continue
    print(f"  WIN  task{t:03d} ({folds} folds): {cur_best:.4f} -> {pts:.4f} (cost={cost})")
    fn = os.path.join(CAND_DIR, f"task{t:03d}.onnx")
    onnx.save(m, fn)
    wins.append((t, cur_best, pts, cost, fn))

print(f"\n{len(wins)} verified wins out of {len(candidates)} candidates.")
import json as _json
_json.dump(wins, open("scratch_onnx/gathersplit_sweep_wins.json", "w"))
