import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

def create_model():
    F = TensorProto.FLOAT
    I64 = TensorProto.INT64
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])
    
    def K(name, arr, dtype=np.int64): 
        return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)
        
    mask_top_arr = np.zeros((1, 1, 30, 30), dtype=np.float32)
    mask_top_arr[0, 0, 0, 0:10] = 1
    mask_top_arr[0, 0, 2, 0:10] = 1
    mask_top_arr[0, 0, 1, [0, 9]] = 1
    mask_top_arr[0, 0, 3, [0, 9]] = 1
    mask_top_arr[0, 0, 4, [0, 9]] = 1
    
    mask_bot_arr = np.zeros((1, 1, 30, 30), dtype=np.float32)
    mask_bot_arr[0, 0, 7, 0:10] = 1
    mask_bot_arr[0, 0, 9, 0:10] = 1
    mask_bot_arr[0, 0, 5, [0, 9]] = 1
    mask_bot_arr[0, 0, 6, [0, 9]] = 1
    mask_bot_arr[0, 0, 8, [0, 9]] = 1
    
    inits = [
        K('ax_1', [1]), K('ax_2', [2]),
        K('c0_s', [0]), K('c5_e', [5]), K('c10_e', [10]),
        K('mask_top', mask_top_arr, dtype=np.float32),
        K('mask_bot', mask_bot_arr, dtype=np.float32),
        K('c0', 0.5 * np.ones((1, 1, 30, 30), dtype=np.float32), dtype=np.float32),
        K('depth10', 10), K('oh_vals', [0.0, 1.0], dtype=np.float32)
    ]
    
    nodes = []
    out_channels = ['c0']
    
    for k in range(1, 10):
        inits.extend([
            K(f'k_s_{k}', [k]), K(f'k_e_{k}', [k+1])
        ])
        
        nodes.append(helper.make_node('Slice', ['input', f'k_s_{k}', f'k_e_{k}', 'ax_1'], [f'is_k_{k}']))
        nodes.append(helper.make_node('Slice', [f'is_k_{k}', 'c0_s', 'c5_e', 'ax_2'], [f'top_half_{k}']))
        nodes.append(helper.make_node('Slice', [f'is_k_{k}', 'c5_e', 'c10_e', 'ax_2'], [f'bot_half_{k}']))
        
        nodes.append(helper.make_node('ReduceMax', [f'top_half_{k}'], [f'c_top_{k}'], axes=[2, 3], keepdims=1))
        nodes.append(helper.make_node('ReduceMax', [f'bot_half_{k}'], [f'c_bot_{k}'], axes=[2, 3], keepdims=1))
        
        nodes.append(helper.make_node('Mul', [f'c_top_{k}', 'mask_top'], [f'out_top_{k}']))
        nodes.append(helper.make_node('Mul', [f'c_bot_{k}', 'mask_bot'], [f'out_bot_{k}']))
        nodes.append(helper.make_node('Add', [f'out_top_{k}', f'out_bot_{k}'], [f'out_{k}']))
        
        out_channels.append(f'out_{k}')
        
    nodes.append(helper.make_node('Concat', out_channels, ['out_logits'], axis=1))
    nodes.append(helper.make_node('ArgMax', ['out_logits'], ['pred_idx'], axis=1, keepdims=0))
    nodes.append(helper.make_node('OneHot', ['pred_idx', 'depth10', 'oh_vals'], ['output'], axis=1))
    
    graph = helper.make_graph(nodes, 'task028', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])


# ===== Owned export-bug repair (Claude, verified vs train+test+arc-gen) =====
# The from-scratch construction above is correct INSIDE the grid, but emitted a
# full-30x30 OneHot whose empty border got channel-0=1 (mismatch vs the all-zero
# padding the official scorer expects), and dynamic-Slice intermediates left the
# cost unmeasurable. This repair: (mask) zero the border with an input-presence
# mask / (pad) crop+Pad the native grid up to a static 30x30, then bake concrete
# value_info onto the dynamic-Slice tensors so the network is scoreable.
import os as _os, copy as _copy
import onnxruntime as _ort
_T = 28
_FIX = ('mask',)

def _resolve_task_json(t):
    for base in [_os.environ.get("PROJECT_DIR", "/project"), r"C:\\Users\\chand\\OneDrive\\Desktop\\get_a_job\\kaggle_competitions\\The 2026 NeuroGolf Championship", "."]:
        p = _os.path.join(base, "data", "task%03d.json" % t)
        if _os.path.exists(p): return p
    raise FileNotFoundError("task%03d.json not found" % t)

def _rename_output(m, new):
    for n in m.graph.node:
        for i, o in enumerate(n.output):
            if o == "output": n.output[i] = new; return

def _set_out_shape(m, dims):
    tt = m.graph.output[0].type.tensor_type; tt.elem_type = TensorProto.FLOAT
    del tt.shape.dim[:]
    for d in dims: tt.shape.dim.add().dim_value = d

def _K(name, arr, dtype):
    return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)

def _apply_fix(m):
    _rename_output(m, "oh_raw")
    if _FIX[0] == "mask":
        m.graph.node.append(helper.make_node("ReduceMax", ["input"], ["presence_m"], axes=[1], keepdims=1))
        m.graph.node.append(helper.make_node("Mul", ["oh_raw", "presence_m"], ["output"]))
    else:
        _, crop_rows, pads = _FIX
        src = "oh_raw"
        if crop_rows is not None:
            m.graph.initializer.extend([_K("__cr_s",[0],np.int64), _K("__cr_e",[crop_rows],np.int64), _K("__cr_ax",[2],np.int64)])
            m.graph.node.append(helper.make_node("Slice", ["oh_raw","__cr_s","__cr_e","__cr_ax"], ["oh_crop"]))
            src = "oh_crop"
        m.graph.initializer.extend([_K("__pads",pads,np.int64), _K("__pv",[0.0],np.float32)])
        m.graph.node.append(helper.make_node("Pad", [src,"__pads","__pv"], ["output"], mode="constant"))
    _set_out_shape(m, [1,10,30,30])
    return m

def _reps(t, k=6):
    d = json.load(open(_resolve_task_json(t)))
    exs = sorted(d["train"]+d["test"]+d["arc-gen"], key=lambda e:(len(e["input"]),len(e["input"][0])))
    idx = set([0, len(exs)-1]) | {int(j*(len(exs)-1)/(k-1)) for j in range(1,k-1)}
    outs=[]
    for i in sorted(idx):
        # inline one-hot (self-contained; no scorer import needed here)
        ex = exs[i]; g = ex["input"]
        arr = np.zeros((1,10,30,30), np.float32)
        for r,row in enumerate(g):
            for c,col in enumerate(row): arr[0][col][r][c]=1.0
        outs.append(arr)
    return outs

def _bake(m, t):
    inf = onnx.shape_inference.infer_shapes(_copy.deepcopy(m), strict_mode=True)
    def sym(vi): return any(dd.HasField("dim_param") or not dd.HasField("dim_value") for dd in vi.type.tensor_type.shape.dim)
    good = {vi.name for vi in inf.graph.value_info if vi.type.tensor_type.HasField("shape") and not sym(vi)}
    good |= {x.name for x in list(m.graph.input)+list(m.graph.output)}
    missing=[]
    for n in m.graph.node:
        for o in n.output:
            if o and o!="output" and o not in good and o not in missing: missing.append(o)
    if not missing: return m
    tmp=_copy.deepcopy(m)
    for nm in missing:
        vi=onnx.ValueInfoProto(); vi.name=nm; tmp.graph.output.append(vi)
    so=_ort.SessionOptions(); so.log_severity_level=3
    so.graph_optimization_level=_ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s=_ort.InferenceSession(tmp.SerializeToString(), so)
    mx={}; dt={}
    for inp in _reps(t):
        for nm,arr in zip(missing, s.run(missing, {"input":inp})):
            sh=list(arr.shape); mx[nm]=[max(a,b) for a,b in zip(mx[nm],sh)] if nm in mx else sh; dt[nm]=arr.dtype
    keep=[vi for vi in m.graph.value_info if vi.name not in missing]
    del m.graph.value_info[:]; m.graph.value_info.extend(keep)
    conv={np.dtype("float32"):TensorProto.FLOAT, np.dtype("int64"):TensorProto.INT64, np.dtype("bool"):TensorProto.BOOL, np.dtype("int32"):TensorProto.INT32}
    for nm in missing:
        m.graph.value_info.append(helper.make_tensor_value_info(nm, conv.get(dt[nm], TensorProto.FLOAT), mx[nm]))
    return m

model = _bake(_apply_fix(create_model()), _T)
