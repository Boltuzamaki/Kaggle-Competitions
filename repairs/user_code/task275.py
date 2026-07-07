import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

F = TensorProto.FLOAT
I64 = TensorProto.INT64
BOOL = TensorProto.BOOL


def _K(name, arr, dtype=np.int64):
    return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)


def create_model():
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 'H_out', 'W_out'])

    inits = [
        _K('c0', [0]), _K('c1', [1]), _K('c2', [2]), _K('c3', [3]),
        _K('starts00', [0, 0]), _K('ax23', [2, 3]),
        _K('axr1', [1]), _K('axr2', [2]), _K('axr3', [3]),
        _K('idx3', [3]),
        _K('shape1d', [-1]),
        _K('one_f', [1.0], dtype=np.float32),
        _K('zero_onehot', np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32).reshape(1, 10, 1, 1),
           dtype=np.float32),
    ]

    n = []

    # ---- recover real grid height H, width W (grid is anchored top-left, no gaps) ----
    n.append(helper.make_node('ReduceMax', ['input'], ['presence'], axes=[1], keepdims=1))
    n.append(helper.make_node('ReduceMax', ['presence'], ['row_any'], axes=[3], keepdims=1))
    n.append(helper.make_node('ReduceSum', ['row_any', 'axr2'], ['H_f'], keepdims=1))
    n.append(helper.make_node('ReduceMax', ['presence'], ['col_any'], axes=[2], keepdims=1))
    n.append(helper.make_node('ReduceSum', ['col_any', 'axr3'], ['W_f'], keepdims=1))
    n.append(helper.make_node('Cast', ['H_f'], ['H_i'], to=I64))
    n.append(helper.make_node('Cast', ['W_f'], ['W_i'], to=I64))
    n.append(helper.make_node('Reshape', ['H_i', 'shape1d'], ['H1']))
    n.append(helper.make_node('Reshape', ['W_i', 'shape1d'], ['W1']))

    # ---- portrait -> vsplit (axis=2), else hsplit (axis=3) ----
    n.append(helper.make_node('Greater', ['H1', 'W1'], ['portrait_b']))
    n.append(helper.make_node('Where', ['portrait_b', 'H1', 'W1'], ['splitDim']))
    n.append(helper.make_node('Where', ['portrait_b', 'c2', 'c3'], ['splitAxis']))
    n.append(helper.make_node('Div', ['splitDim', 'c2'], ['half']))
    n.append(helper.make_node('Mod', ['splitDim', 'c2'], ['offset']))
    n.append(helper.make_node('Add', ['half', 'offset'], ['start1']))

    # ---- split into 2 halves directly off `input` (also drops the 30x30 padding) ----
    # piece0 always starts at (0,0); piece1 starts at start1 along the split axis, 0 along the other.
    n.append(helper.make_node('Where', ['portrait_b', 'half', 'H1'], ['p0_end2']))
    n.append(helper.make_node('Where', ['portrait_b', 'W1', 'half'], ['p0_end3']))
    n.append(helper.make_node('Concat', ['p0_end2', 'p0_end3'], ['p0_ends'], axis=0))
    n.append(helper.make_node('Slice', ['input', 'starts00', 'p0_ends', 'ax23'], ['piece0']))

    n.append(helper.make_node('Where', ['portrait_b', 'start1', 'c0'], ['p1_start2']))
    n.append(helper.make_node('Where', ['portrait_b', 'c0', 'start1'], ['p1_start3']))
    n.append(helper.make_node('Concat', ['p1_start2', 'p1_start3'], ['p1_starts'], axis=0))
    n.append(helper.make_node('Concat', ['H1', 'W1'], ['p1_ends'], axis=0))
    n.append(helper.make_node('Slice', ['input', 'p1_starts', 'p1_ends', 'ax23'], ['piece1']))

    # ---- numcolors per half; x4 = fewer colors, x5 = more colors ----
    n.append(helper.make_node('ReduceMax', ['piece0'], ['pres0'], axes=[2, 3], keepdims=1))
    n.append(helper.make_node('ReduceSum', ['pres0', 'axr1'], ['nc0'], keepdims=1))
    n.append(helper.make_node('ReduceMax', ['piece1'], ['pres1'], axes=[2, 3], keepdims=1))
    n.append(helper.make_node('ReduceSum', ['pres1', 'axr1'], ['nc1'], keepdims=1))
    n.append(helper.make_node('Less', ['nc0', 'nc1'], ['cond']))
    n.append(helper.make_node('Where', ['cond', 'piece0', 'piece1'], ['x4']))
    n.append(helper.make_node('Where', ['cond', 'piece1', 'piece0'], ['x5']))

    # ---- x6 = width(x5) (== width of either piece, they share shape) ----
    n.append(helper.make_node('Shape', ['piece0'], ['piece_shape']))
    n.append(helper.make_node('Gather', ['piece_shape', 'idx3'], ['x6'], axis=0))
    n.append(helper.make_node('Cast', ['x6'], ['x6f'], to=F))

    # ---- x9 = upscale(x5, x6) via nearest-neighbor Resize (exact block replication) ----
    n.append(helper.make_node('Concat', ['one_f', 'one_f', 'x6f', 'x6f'], ['scales'], axis=0))
    n.append(helper.make_node('Resize', ['x5', '', 'scales'], ['x9'], mode='nearest',
                               coordinate_transformation_mode='asymmetric', nearest_mode='floor'))

    # ---- x8 = chain(dmirror, merge, rbind(repeat, x6)); applied twice: x10=x8(x4), x11=x8(x10) ----
    # Channel slicing commutes with Tile/Transpose (both act only on the spatial axes), so we
    # only need to carry x4's channel-0 (ZERO-color) plane through the tile/transpose chain,
    # instead of the full 10-channel one-hot tensor -- and we cast it to BOOL immediately so every
    # tensor in the chain is 1 byte/elem instead of 4.
    n.append(helper.make_node('Slice', ['x4', 'c0', 'c1', 'c1'], ['mask4']))
    n.append(helper.make_node('Cast', ['mask4'], ['mask4_b'], to=BOOL))
    n.append(helper.make_node('Concat', ['c1', 'c1', 'x6', 'c1'], ['repeats'], axis=0))
    n.append(helper.make_node('Tile', ['mask4_b', 'repeats'], ['tiledm4']))
    n.append(helper.make_node('Transpose', ['tiledm4'], ['m10'], perm=[0, 1, 3, 2]))
    n.append(helper.make_node('Tile', ['m10', 'repeats'], ['tiledm10']))
    n.append(helper.make_node('Transpose', ['tiledm10'], ['mask_b'], perm=[0, 1, 3, 2]))

    # ---- x12 = ofcolor(x11, ZERO); O = fill(x9, ZERO, x12) ----
    n.append(helper.make_node('Where', ['mask_b', 'zero_onehot', 'x9'], ['output']))

    graph = helper.make_graph(n, 'task275', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])


# ===== Owned repair scaffolding (Claude, verified vs train+test+arc-gen) =====
import os as _os, copy as _copy
def _K2(n,a,d): return numpy_helper.from_array(np.array(a,dtype=d),name=n)
def _rename_output(m,new):
    for nd in m.graph.node:
        for i,o in enumerate(nd.output):
            if o=="output": nd.output[i]=new; return
def _set_out_shape(m,dims):
    tt=m.graph.output[0].type.tensor_type; tt.elem_type=TensorProto.FLOAT; del tt.shape.dim[:]
    for d in dims: tt.shape.dim.add().dim_value=d
def _crop_pad(m):
    """OneHot 'output' is a dynamic [1,10,h,w] crop at top-left; Pad to static 30x30
    using h,w read from the tensor's own Shape (keeps padding all-zero)."""
    _rename_output(m,"oh_raw")
    m.graph.initializer.extend([_K2("__s2",[2],np.int64),_K2("__e4",[4],np.int64),_K2("__a0",[0],np.int64),
        _K2("__30x2",[30,30],np.int64),_K2("__pfx6",[0,0,0,0,0,0],np.int64),_K2("__pv",[0.0],np.float32)])
    m.graph.node.extend([
        helper.make_node("Shape",["oh_raw"],["__osh"]),
        helper.make_node("Slice",["__osh","__s2","__e4","__a0"],["__hw"]),
        helper.make_node("Sub",["__30x2","__hw"],["__padhw"]),
        helper.make_node("Concat",["__pfx6","__padhw"],["__pads"],axis=0),
        helper.make_node("Pad",["oh_raw","__pads","__pv"],["output"],mode="constant")])
    _set_out_shape(m,[1,10,30,30]); return m
def _resolve_task_json(t):
    for base in [_os.environ.get("PROJECT_DIR","/project"), r"C:\\Users\\chand\\OneDrive\\Desktop\\get_a_job\\kaggle_competitions\\The 2026 NeuroGolf Championship", "."]:
        p=_os.path.join(base,"data","task%03d.json"%t)
        if _os.path.exists(p): return p
    raise FileNotFoundError("task%03d.json"%t)
def _reps(t,k=8):
    import onnxruntime as _ort  # noqa
    d=json.load(open(_resolve_task_json(t)))
    exs=sorted(d["train"]+d["test"]+d["arc-gen"], key=lambda e:(len(e["input"]),len(e["input"][0])))
    idx=set([0,len(exs)-1])|{int(j*(len(exs)-1)/(k-1)) for j in range(1,k-1)}
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
    good={vi.name for vi in inf.graph.value_info if vi.type.tensor_type.HasField("shape") and not sym(vi)}
    good|={x.name for x in list(m.graph.input)+list(m.graph.output)}
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

def _make():
    return _crop_pad(create_model())

model = _bake(_make(), 275)
