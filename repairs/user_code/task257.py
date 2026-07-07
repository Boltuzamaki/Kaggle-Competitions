import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

F = TensorProto.FLOAT
I64 = TensorProto.INT64


def build_257():
    # Rule (verified vs arc-dsl catalog + pure-numpy check on all 269 train+test+arc-gen
    # examples, all of which are a fixed 9x9 -> 4x4 shape):
    #   x1 = tophalf(I); x2 = bottomhalf(I)          -> rows[0:4]; rows[5:9]  (row4 is a separator)
    #   x3 = lefthalf(x1); x4 = righthalf(x1)         -> cols[0:4]; cols[5:9] of x1
    #   x5 = lefthalf(x2); x6 = righthalf(x2)         -> cols[0:4]; cols[5:9] of x2
    #   O = fill(fill(fill(x6, 8, ofcolor(x5,8)), 4, ofcolor(x4,4)), 7, ofcolor(x3,7))
    # i.e. start from bottom-right quadrant x6, then overwrite cells with 8 where x5==8,
    # then overwrite with 4 where x4==4, then overwrite with 7 where x3==7.
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])

    def K(name, arr, dtype=np.int64):
        return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)

    # one-hot column vectors [1,10,1,1] for colors 4,7,8
    def onehot_const(color):
        v = np.zeros((1, 10, 1, 1), dtype=np.float32)
        v[0, color, 0, 0] = 1.0
        return v

    inits = [
        # quadrant slices (rows, cols) on axes 2,3 : channels 0:10 kept (full)
        K('s_x3_h', [0]), K('e_x3_h', [4]),
        K('s_x4_h', [0]), K('e_x4_h', [4]),
        K('s_x56_h', [5]), K('e_x56_h', [9]),
        K('s_left_w', [0]), K('e_left_w', [4]),
        K('s_right_w', [5]), K('e_right_w', [9]),
        K('ax2', [2]), K('ax3', [3]),
        K('ax_row', [2]), K('ax_col', [3]),
        K('ax23', [2, 3]),
        # channel-slice bounds for masks
        K('ch7_s', [7]), K('ch7_e', [8]),
        K('ch4_s', [4]), K('ch4_e', [5]),
        K('ch8_s', [8]), K('ch8_e', [9]),
        K('ax1', [1]),
        K('oh7', onehot_const(7), np.float32),
        K('oh4', onehot_const(4), np.float32),
        K('oh8', onehot_const(8), np.float32),
        K('pads', [0, 0, 0, 0, 0, 0, 26, 26]),
        K('pad_val', [0.0], np.float32),
    ]

    n = []
    # x1 = tophalf(I) = rows[0:4]; x2 = bottomhalf(I) = rows[5:9]
    n.append(helper.make_node('Slice', ['input', 's_x3_h', 'e_x3_h', 'ax_row'], ['x1']))
    n.append(helper.make_node('Slice', ['input', 's_x56_h', 'e_x56_h', 'ax_row'], ['x2']))

    # x3 = lefthalf(x1); x4 = righthalf(x1)
    n.append(helper.make_node('Slice', ['x1', 's_left_w', 'e_left_w', 'ax_col'], ['x3']))
    n.append(helper.make_node('Slice', ['x1', 's_right_w', 'e_right_w', 'ax_col'], ['x4']))
    # x5 = lefthalf(x2); x6 = righthalf(x2)
    n.append(helper.make_node('Slice', ['x2', 's_left_w', 'e_left_w', 'ax_col'], ['x5']))
    n.append(helper.make_node('Slice', ['x2', 's_right_w', 'e_right_w', 'ax_col'], ['x6']))

    # masks: channel-slice then broadcast-compare via Cast to bool (values are exactly 0/1 floats)
    n.append(helper.make_node('Slice', ['x3', 'ch7_s', 'ch7_e', 'ax1'], ['m3_7f']))
    n.append(helper.make_node('Slice', ['x4', 'ch4_s', 'ch4_e', 'ax1'], ['m4_4f']))
    n.append(helper.make_node('Slice', ['x5', 'ch8_s', 'ch8_e', 'ax1'], ['m5_8f']))
    n.append(helper.make_node('Cast', ['m3_7f'], ['m3_7'], to=onnx.TensorProto.BOOL))
    n.append(helper.make_node('Cast', ['m4_4f'], ['m4_4'], to=onnx.TensorProto.BOOL))
    n.append(helper.make_node('Cast', ['m5_8f'], ['m5_8'], to=onnx.TensorProto.BOOL))

    # apply in priority order: base=x6, then 8 (from x5), then 4 (from x4), then 7 (from x3)
    n.append(helper.make_node('Where', ['m5_8', 'oh8', 'x6'], ['step1']))
    n.append(helper.make_node('Where', ['m4_4', 'oh4', 'step1'], ['step2']))
    n.append(helper.make_node('Where', ['m3_7', 'oh7', 'step2'], ['oh_raw']))

    # pad the 4x4 one-hot result up to static 30x30 with zeros (matches the harness's
    # zero-outside-grid convention exactly, since 4+26=30)
    n.append(helper.make_node('Pad', ['oh_raw', 'pads', 'pad_val'], ['output'], mode='constant'))

    graph = helper.make_graph(n, 'task257', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 12)])


# ===== Owned repair scaffolding (Claude, verified vs train+test+arc-gen) =====
import os as _os, copy as _copy
def _K(n,a,d): return numpy_helper.from_array(np.array(a,dtype=d),name=n)
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
    return build_257()


model = _bake(_make(), 257)
