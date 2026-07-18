import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime
F = TensorProto.FLOAT; I64 = TensorProto.INT64

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

# ===== Task030 rule =====
# For every color-object (each color forms one 8-connected object here, verified),
# find the row index 'ref_row' = lowermost row containing color 1.
# Shift every color's rows (columns unchanged) so that the color's own lowermost
# row lands on ref_row (drop/raise each object to align its bottom edge with color 1's
# bottom edge). Background (removed original objects) becomes 0; recompose one-hot.
def build_030():
    x = helper.make_tensor_value_info('input', F, [1,10,30,30])
    y = helper.make_tensor_value_info('output', F, [1,10,30,30])
    I = [
        _K('ax1', [1], np.int64),
        _K('row_idx', np.arange(30).reshape(1,1,30,1), np.int64),
        _K('col_idx', np.arange(30).reshape(1,1,1,30), np.int64),
        _K('m1', [-1], np.int64),
        _K('c0f', [0.0], np.float32),
        _K('c1f', [1.0], np.float32),
        _K('diff', (np.arange(30).reshape(30,1) - np.arange(30).reshape(1,30)).reshape(1,1,30,30), np.int64),
    ]
    n = []
    own_low = {}
    chc_name = {}
    for c in range(1, 10):
        I.append(_K(f'st{c}', [c], np.int64))
        I.append(_K(f'en{c}', [c+1], np.int64))
        n.append(helper.make_node('Slice', ['input', f'st{c}', f'en{c}', 'ax1'], [f'ch{c}']))
        n.append(helper.make_node('ReduceMax', [f'ch{c}'], [f'rowany{c}'], axes=[3], keepdims=1))
        n.append(helper.make_node('Greater', [f'rowany{c}', 'c0f'], [f'rowbool{c}']))
        n.append(helper.make_node('Where', [f'rowbool{c}', 'row_idx', 'm1'], [f'rowpres{c}']))
        n.append(helper.make_node('ReduceMax', [f'rowpres{c}'], [f'ownlow{c}'], axes=[2], keepdims=1))
        own_low[c] = f'ownlow{c}'
        chc_name[c] = f'ch{c}'

    ref_row = own_low[1]  # color 1's own lowermost row = the reference row

    outs = []
    for c in range(1, 10):
        n.append(helper.make_node('Sub', [ref_row, own_low[c]], [f'shift{c}']))
        n.append(helper.make_node('Equal', ['diff', f'shift{c}'], [f'maskb{c}']))
        n.append(helper.make_node('Cast', [f'maskb{c}'], [f'maskf{c}'], to=F))
        n.append(helper.make_node('MatMul', [f'maskf{c}', chc_name[c]], [f'out{c}']))
        outs.append(f'out{c}')

    n.append(helper.make_node('Concat', outs, ['stacked19'], axis=1))
    n.append(helper.make_node('ReduceSum', ['stacked19', 'ax1'], ['sum19'], keepdims=1))
    n.append(helper.make_node('Sub', ['c1f', 'sum19'], ['out0']))
    n.append(helper.make_node('Concat', ['out0', 'stacked19'], ['oh_raw'], axis=1))
    # zero out the zero-padded 30x30 border (real grid extent varies per example)
    n.append(helper.make_node('ReduceMax', ['input'], ['presence_m'], axes=[1], keepdims=1))
    n.append(helper.make_node('Mul', ['oh_raw', 'presence_m'], ['output']))

    return helper.make_model(helper.make_graph(n, 'task030', [x], [y], I),
                              ir_version=8, opset_imports=[helper.make_opsetid('', 13)])

def _make():
    return build_030()

model = _bake(_make(), 30)
