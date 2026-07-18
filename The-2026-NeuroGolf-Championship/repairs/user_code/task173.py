import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime
F = TensorProto.FLOAT
I64 = TensorProto.INT64

# ===== Owned repair scaffolding (Claude, verified vs train+test+arc-gen) =====
import os as _os, copy as _copy
def _K(n,a,d): return numpy_helper.from_array(np.array(a,dtype=d),name=n)
def _rename_output(m,new):
    for nd in m.graph.node:
        for i,o in enumerate(nd.output):
            if o=="output": nd.output[i]=new; return
def _set_out_shape(m,dims):
    tt=m.graph.output[0].type.tensor_type; tt.elem_type=TensorProto.FLOAT; del tt.shape.dim[:]
    for d in dims: tt.shape.dim.add().dim_value=d
def _mask(m):
    """Same-shape task: zero the polluted 30x30 border via an input-presence mask."""
    _rename_output(m,"oh_raw")
    m.graph.node.append(helper.make_node("ReduceMax",["input"],["presence_m"],axes=[1],keepdims=1))
    m.graph.node.append(helper.make_node("Mul",["oh_raw","presence_m"],["output"]))
    _set_out_shape(m,[1,10,30,30]); return m
def _resolve_task_json(t):
    for base in [_os.environ.get("PROJECT_DIR","/project"), r"C:\\Users\\chand\\OneDrive\\Desktop\\get_a_job\\kaggle_competitions\\The 2026 NeuroGolf Championship", "."]:
        p=_os.path.join(base,"data","task%03d.json"%t)
        if _os.path.exists(p): return p
    raise FileNotFoundError("task%03d.json"%t)
def _reps(t,k=8):
    d=json.load(open(_resolve_task_json(t)))
    exs=sorted(d["train"]+d["test"]+d["arc-gen"], key=lambda e:(len(e["input"]),len(e["input"][0])))
    idx=set([0,len(exs)-1]) | set(int(j*(len(exs)-1)/(k-1)) for j in range(1,k-1))
    out=[]
    for i in sorted(idx):
        g=exs[i]["input"]; a=np.zeros((1,10,30,30),np.float32)
        for r,row in enumerate(g):
            for c,v in enumerate(row): a[0][v][r][c]=1.0
        out.append(a)
    return out
def _bake(m,t):
    import onnxruntime as _ort
    inf=onnx.shape_inference.infer_shapes(_copy.deepcopy(m),strict_mode=True)
    def sym(vi): return any(dd.HasField("dim_param") or not dd.HasField("dim_value") for dd in vi.type.tensor_type.shape.dim)
    good=set(vi.name for vi in inf.graph.value_info if vi.type.tensor_type.HasField("shape") and not sym(vi))
    good |= set(x.name for x in list(m.graph.input)+list(m.graph.output))
    missing=[]
    for nd in m.graph.node:
        for o in nd.output:
            if o and o!="output" and o not in good and o not in missing: missing.append(o)
    if not missing: return m
    tmp=_copy.deepcopy(m)
    for nm in missing:
        vi=onnx.ValueInfoProto(); vi.name=nm; tmp.graph.output.append(vi)
    so=_ort.SessionOptions(); so.log_severity_level=3
    so.graph_optimization_level=_ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s=_ort.InferenceSession(tmp.SerializeToString(),so)
    mx={}; dt={}
    for inp in _reps(t):
        for nm,arr in zip(missing,s.run(missing,{"input":inp})):
            sh=list(arr.shape); mx[nm]=[max(a,b) for a,b in zip(mx[nm],sh)] if nm in mx else sh; dt[nm]=arr.dtype
    keep=[vi for vi in m.graph.value_info if vi.name not in missing]
    del m.graph.value_info[:]; m.graph.value_info.extend(keep)
    conv={np.dtype("float32"):TensorProto.FLOAT,np.dtype("int64"):TensorProto.INT64,np.dtype("bool"):TensorProto.BOOL,np.dtype("int32"):TensorProto.INT32}
    for nm in missing:
        m.graph.value_info.append(helper.make_tensor_value_info(nm,conv.get(dt[nm],TensorProto.FLOAT),mx[nm]))
    return m

# ===== Task 173 (72322fa7) =====
# Ground-truth rule (arc_dsl_ref/solvers.py::solve_72322fa7), verified in pure numpy
# 266/266 on train+test+arc-gen:
#   Every input decomposes (8-connected components) into: (a) single-color "part" objects,
#   and (b) exactly-2-color "template" objects. Each template has a dominant color D (the
#   majority of its cells) and a single marker cell of a minority color M. Stats across all
#   266 examples show the marker is ALWAYS a single pixel, and the dominant cells always sit
#   at one of exactly 4 fixed symmetric offset-shapes around the marker:
#     'h'    = {(0,-1),(0,1)}                       (west+east)
#     'v'    = {(-1,0),(1,0)}                       (north+south)
#     'plus' = {(-1,0),(1,0),(0,-1),(0,1)}          (4 orthogonal neighbors)
#     'x'    = {(-1,-1),(-1,1),(1,-1),(1,1)}        (4 diagonal neighbors)
#   Rule: paint(I, marker-color occurrences stamped with the dominant shape) then paint(...,
#   dominant-shape occurrences [without a marker] stamped with the marker color) on top.
#
# Key simplification (verified against a from-scipy connected-components reference, exact
# match on all 266 examples): NO connected-components/label-propagation is needed. Because
# marker cells are always isolated singletons, and only 4 possible fixed offset-shapes exist,
# each template's (dominant-color D, marker-color M, shape) signature can be recovered purely
# from *local* per-pixel neighbor lookups on the color-index map (no Loop/Scan): for a cell of
# color M whose neighbors at a shape's offsets are all identical to some nonzero color D != M,
# that (D, shape) is M's template signature. Colors are always used consistently (marker colors
# never repeat across templates within one grid - verified), so each M has at most one signature,
# even when 2-3 templates share the same grid. This is computed for every color 1..9 in a single
# static/unrolled pass (no data-dependent control flow):
#   1. cm = ArgMax(input,axis=1) -> per-pixel color index (0 = background/out-of-grid).
#   2. Pad cm by 1 (zero) and static-Slice out the 8 neighbor maps (N,S,E,W,NW,NE,SW,SE).
#   3. Per shape, compute `uniform_s` (all its neighbor offsets equal) and `valid_s`
#      (uniform, neighbor color D>0, center>0, D!=center).
#   4. Per color c (1..9): has_{s,c} = any pixel of color c is a valid_s marker; D_{s,c} =
#      that D. Priority h<v<plus<x (later wins; matches the reference exactly, resolves the
#      benign case where a genuine 'plus' pixel trivially also satisfies its 'h'/'v' subsets)
#      picks a single (D_c, shape_c) per color.
#   5. Layer1 (paint dominant shape around every marker of color c): for each of the 8
#      neighbor directions, "is there a same-color-c marker in the opposite direction" is just
#      Equal(opposite-neighbor-map, c) (shifting commutes with the pointwise color-map, so no
#      separate mask-shift is needed) gated by whether that direction is compatible with c's
#      resolved shape.
#   6. Layer2 (paint marker M at every location whose neighbors already match the dominant
#      shape/color exactly, using the ORIGINAL colormap, on top of Layer1).
#   7. OneHot back to one-hot, then mask by input-presence (same-shape idiom) to zero the
#      30x30 padding exactly like the ground-truth 30x30 tensors (grid cells beyond the real
#      extent are all-zero across every channel, not "background color 0").
DIRS = {
    'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1),
    'NW': (-1, -1), 'NE': (-1, 1), 'SW': (1, -1), 'SE': (1, 1),
}
OPPOSITE = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W', 'NW': 'SE', 'SE': 'NW', 'NE': 'SW', 'SW': 'NE'}
SHAPES = {
    'h': ['W', 'E'],
    'v': ['N', 'S'],
    'plus': ['N', 'S', 'W', 'E'],
    'x': ['NW', 'NE', 'SW', 'SE'],
}
SHAPE_ORDER = ['h', 'v', 'plus', 'x']  # priority: later wins
DIR_CATEGORY = {'N': 'vert', 'S': 'vert', 'W': 'horiz', 'E': 'horiz',
                 'NW': 'diag', 'NE': 'diag', 'SW': 'diag', 'SE': 'diag'}


def create_model():
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])

    inits = []
    n = []
    made = set()

    def K(name, val, dtype=np.int64):
        if name not in made:
            inits.append(_K(name, val, dtype))
            made.add(name)
        return name

    def NODE(*args, **kwargs):
        nd = helper.make_node(*args, **kwargs)
        n.append(nd)
        return nd.output[0]

    K('ax23', [2, 3])
    K('zero_i64', [0])
    K('one_i64', [1])
    K('pads_hw', [0, 0, 1, 1, 0, 0, 1, 1])
    K('depth10', [10])
    K('oh_vals', [0.0, 1.0], np.float32)
    K('shape_1_30_30', [1, 30, 30])
    for c in range(1, 10):
        K(f'c{c}', [c])

    cm = NODE('ArgMax', ['input'], ['cm'], axis=1, keepdims=1)  # int64 [1,1,30,30]
    cm_pad = NODE('Pad', [cm, 'pads_hw', 'zero_i64'], ['cm_pad'], mode='constant')  # [1,1,32,32]

    neigh = {}
    for name, (dr, dc) in DIRS.items():
        sh, eh = 1 + dr, 1 + dr + 30
        sw, ew = 1 + dc, 1 + dc + 30
        K(f's_{name}', [sh, sw])
        K(f'e_{name}', [eh, ew])
        neigh[name] = NODE('Slice', [cm_pad, f's_{name}', f'e_{name}', 'ax23'], [f'nb_{name}'])

    cm_pos = NODE('Greater', [cm, 'zero_i64'], ['cm_pos'])

    Dfield = {}
    uniform = {}
    valid = {}
    for s in SHAPE_ORDER:
        dirs = SHAPES[s]
        nbs = [neigh[d] for d in dirs]
        if len(nbs) == 2:
            uni = NODE('Equal', [nbs[0], nbs[1]], [f'uniform_{s}'])
        else:
            eqA = NODE('Equal', [nbs[0], nbs[1]], [f'eqA_{s}'])
            eqB = NODE('Equal', [nbs[0], nbs[2]], [f'eqB_{s}'])
            eqC = NODE('Equal', [nbs[0], nbs[3]], [f'eqC_{s}'])
            eqAB = NODE('And', [eqA, eqB], [f'eqAB_{s}'])
            uni = NODE('And', [eqAB, eqC], [f'uniform_{s}'])
        Dfield[s] = nbs[0]
        uniform[s] = uni

        Dpos = NODE('Greater', [Dfield[s], 'zero_i64'], [f'Dpos_{s}'])
        Deqcm = NODE('Equal', [Dfield[s], cm], [f'Deqcm_{s}'])
        Dneqcm = NODE('Not', [Deqcm], [f'Dneqcm_{s}'])
        t1 = NODE('And', [uni, Dpos], [f't1_{s}'])
        t2 = NODE('And', [t1, cm_pos], [f't2_{s}'])
        valid[s] = NODE('And', [t2, Dneqcm], [f'valid_{s}'])

    Dc = {}
    Wc = {}
    for c in range(1, 10):
        colmask = NODE('Equal', [cm, f'c{c}'], [f'colmask_{c}'])
        has = {}
        Dcs = {}
        for s in SHAPE_ORDER:
            match = NODE('And', [valid[s], colmask], [f'match_{s}_{c}'])
            matchi = NODE('Cast', [match], [f'matchi_{s}_{c}'], to=I64)
            has[s] = NODE('ReduceMax', [matchi], [f'has_{s}_{c}'], axes=[2, 3], keepdims=1)
            Dm = NODE('Where', [match, Dfield[s], 'zero_i64'], [f'Dm_{s}_{c}'])
            Dcs[s] = NODE('ReduceMax', [Dm], [f'Dcs_{s}_{c}'], axes=[2, 3], keepdims=1)

        anyx = has['x']
        anyplusx = NODE('Max', [has['plus'], anyx], [f'anyplusx_{c}'])
        anyvplusx = NODE('Max', [has['v'], anyplusx], [f'anyvplusx_{c}'])

        Wx = has['x']
        notx = NODE('Sub', ['one_i64', anyx], [f'notx_{c}'])
        Wplus = NODE('Mul', [has['plus'], notx], [f'Wplus_{c}'])
        notplusx = NODE('Sub', ['one_i64', anyplusx], [f'notplusx_{c}'])
        Wv = NODE('Mul', [has['v'], notplusx], [f'Wv_{c}'])
        notvplusx = NODE('Sub', ['one_i64', anyvplusx], [f'notvplusx_{c}'])
        Wh = NODE('Mul', [has['h'], notvplusx], [f'Wh_{c}'])

        Wc[c] = {'h': Wh, 'v': Wv, 'plus': Wplus, 'x': Wx}

        term_h = NODE('Mul', [Dcs['h'], Wh], [f'term_h_{c}'])
        term_v = NODE('Mul', [Dcs['v'], Wv], [f'term_v_{c}'])
        term_plus = NODE('Mul', [Dcs['plus'], Wplus], [f'term_plus_{c}'])
        term_x = NODE('Mul', [Dcs['x'], Wx], [f'term_x_{c}'])
        sum1 = NODE('Add', [term_h, term_v], [f'sum1_{c}'])
        sum2 = NODE('Add', [sum1, term_plus], [f'sum2_{c}'])
        Dc[c] = NODE('Add', [sum2, term_x], [f'Dc_{c}'])

        compat_horiz = NODE('Max', [Wh, Wplus], [f'compat_horiz_{c}'])
        compat_vert = NODE('Max', [Wv, Wplus], [f'compat_vert_{c}'])
        compat_diag = Wx
        Wc[c]['compat'] = {'horiz': compat_horiz, 'vert': compat_vert, 'diag': compat_diag}

    # ---- layer1: stamp D at dominant offsets around every marker cell ----
    acc = cm
    for c in range(1, 10):
        for dname in DIRS:
            cat = DIR_CATEGORY[dname]
            compat = Wc[c]['compat'][cat]
            compat_bool = NODE('Greater', [compat, 'zero_i64'], [f'compatb_{c}_{dname}'])
            shifted_match = NODE('Equal', [neigh[OPPOSITE[dname]], f'c{c}'], [f'shiftm_{c}_{dname}'])
            cond = NODE('And', [shifted_match, compat_bool], [f'cond1_{c}_{dname}'])
            acc = NODE('Where', [cond, Dc[c], acc], [f'acc1_{c}_{dname}'])
    layer1 = acc

    # ---- layer2: paint marker color at every location whose dominant shape matches, using ORIGINAL cm ----
    acc2 = layer1
    for c in range(1, 10):
        terms = []
        for s in SHAPE_ORDER:
            Wbool = NODE('Greater', [Wc[c][s], 'zero_i64'], [f'Wbool_{s}_{c}'])
            Deq = NODE('Equal', [Dfield[s], Dc[c]], [f'Deq_{s}_{c}'])
            t = NODE('And', [uniform[s], Deq], [f'tA_{s}_{c}'])
            terms.append(NODE('And', [t, Wbool], [f'tB_{s}_{c}']))
        or1 = NODE('Or', [terms[0], terms[1]], [f'or1_{c}'])
        or2 = NODE('Or', [or1, terms[2]], [f'or2_{c}'])
        condc = NODE('Or', [or2, terms[3]], [f'condc_{c}'])
        acc2 = NODE('Where', [condc, f'c{c}', acc2], [f'acc2_{c}'])
    final_cm = acc2

    reshaped = NODE('Reshape', [final_cm, 'shape_1_30_30'], ['final_cm_r'])
    NODE('OneHot', [reshaped, 'depth10', 'oh_vals'], ['output'], axis=1)

    graph = helper.make_graph(n, 'task173', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])


def _make():
    return _mask(create_model())


model = _bake(_make(), 173)
