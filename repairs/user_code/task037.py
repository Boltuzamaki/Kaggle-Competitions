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
        
    inits = [
        K('ax_1', [1]), K('ax_2', [2]), K('ax_3', [3]),
        K('c1', [1]), K('c0', [0]), K('c2', [2]),
        K('c0_5_f', [0.5], dtype=np.float32),
        K('c2_f', [2.0], dtype=np.float32),
        K('row_indices', np.arange(30).reshape(1,1,30,1), dtype=np.int64),
        K('col_indices', np.arange(30).reshape(1,1,1,30), dtype=np.int64),
        K('m1', [-1]), K('p999', [999]),
        K('shape_1d', [-1]),
        K('depth10', 10), K('oh_vals', [0.0, 1.0], dtype=np.float32)
    ]
    
    nodes = []
    
    out_channels = []
    
    # channel 0 bias
    nodes.append(helper.make_node('Slice', ['input', 'c1', 'c2', 'ax_1'], ['is_k_dummy'])) # just for shape
    nodes.append(helper.make_node('Cast', ['c0'], ['c0_f0'], to=F))
    nodes.append(helper.make_node('Mul', ['is_k_dummy', 'c0_f0'], ['zeros_H_W']))
    nodes.append(helper.make_node('Add', ['zeros_H_W', 'c0_5_f'], ['drawn_0']))
    out_channels.append('drawn_0')
    
    for k in range(1, 10):
        inits.extend([K(f'k_s_{k}', [k]), K(f'k_e_{k}', [k+1])])
        nodes.append(helper.make_node('Slice', ['input', f'k_s_{k}', f'k_e_{k}', 'ax_1'], [f'is_k_{k}']))
        
        # r_min, r_max
        nodes.append(helper.make_node('ReduceMax', [f'is_k_{k}'], [f'row_any_float_{k}'], axes=[3], keepdims=1))
        nodes.append(helper.make_node('Greater', [f'row_any_float_{k}', 'c0_5_f'], [f'row_any_bool_{k}']))
        nodes.append(helper.make_node('Where', [f'row_any_bool_{k}', 'row_indices', 'm1'], [f'row_present_{k}']))
        nodes.append(helper.make_node('ReduceMax', [f'row_present_{k}'], [f'r_max_{k}'], axes=[2], keepdims=1))
        nodes.append(helper.make_node('Where', [f'row_any_bool_{k}', 'row_indices', 'p999'], [f'row_present_min_{k}']))
        nodes.append(helper.make_node('ReduceMin', [f'row_present_min_{k}'], [f'r_min_{k}'], axes=[2], keepdims=1))
        
        # c_min, c_max
        nodes.append(helper.make_node('ReduceMax', [f'is_k_{k}'], [f'col_any_float_{k}'], axes=[2], keepdims=1))
        nodes.append(helper.make_node('Greater', [f'col_any_float_{k}', 'c0_5_f'], [f'col_any_bool_{k}']))
        nodes.append(helper.make_node('Where', [f'col_any_bool_{k}', 'col_indices', 'm1'], [f'col_present_{k}']))
        nodes.append(helper.make_node('ReduceMax', [f'col_present_{k}'], [f'c_max_{k}'], axes=[3], keepdims=1))
        nodes.append(helper.make_node('Where', [f'col_any_bool_{k}', 'col_indices', 'p999'], [f'col_present_min_{k}']))
        nodes.append(helper.make_node('ReduceMin', [f'col_present_min_{k}'], [f'c_min_{k}'], axes=[3], keepdims=1))
        
        nodes.append(helper.make_node('Equal', ['row_indices', f'r_min_{k}'], [f'is_rmin_{k}']))
        nodes.append(helper.make_node('Equal', ['col_indices', f'c_min_{k}'], [f'is_cmin_{k}']))
        nodes.append(helper.make_node('Equal', ['col_indices', f'c_max_{k}'], [f'is_cmax_{k}']))
        
        nodes.append(helper.make_node('Cast', [f'is_rmin_{k}'], [f'is_rmin_f_{k}'], to=F))
        nodes.append(helper.make_node('Cast', [f'is_cmin_{k}'], [f'is_cmin_f_{k}'], to=F))
        nodes.append(helper.make_node('Cast', [f'is_cmax_{k}'], [f'is_cmax_f_{k}'], to=F))
        
        nodes.append(helper.make_node('Mul', [f'is_rmin_f_{k}', f'is_cmin_f_{k}'], [f'tl_mask_{k}']))
        nodes.append(helper.make_node('Mul', [f'is_rmin_f_{k}', f'is_cmax_f_{k}'], [f'tr_mask_{k}']))
        
        nodes.append(helper.make_node('Mul', [f'is_k_{k}', f'tl_mask_{k}'], [f'tl_pixel_{k}']))
        nodes.append(helper.make_node('Mul', [f'is_k_{k}', f'tr_mask_{k}'], [f'tr_pixel_{k}']))
        
        nodes.append(helper.make_node('ReduceMax', [f'tl_pixel_{k}'], [f'tl_f_{k}'], axes=[2, 3], keepdims=1))
        nodes.append(helper.make_node('ReduceMax', [f'tr_pixel_{k}'], [f'tr_f_{k}'], axes=[2, 3], keepdims=1))
        
        nodes.append(helper.make_node('Sub', ['row_indices', f'r_min_{k}'], [f'r_dist_{k}']))
        nodes.append(helper.make_node('Sub', ['col_indices', f'c_min_{k}'], [f'c_dist_main_{k}']))
        nodes.append(helper.make_node('Sub', [f'c_max_{k}', 'col_indices'], [f'c_dist_anti_{k}']))
        
        nodes.append(helper.make_node('Equal', [f'r_dist_{k}', f'c_dist_main_{k}'], [f'on_main_b_{k}']))
        nodes.append(helper.make_node('Equal', [f'r_dist_{k}', f'c_dist_anti_{k}'], [f'on_anti_b_{k}']))
        nodes.append(helper.make_node('Cast', [f'on_main_b_{k}'], [f'on_main_f_{k}'], to=F))
        nodes.append(helper.make_node('Cast', [f'on_anti_b_{k}'], [f'on_anti_f_{k}'], to=F))
        
        nodes.append(helper.make_node('Mul', [f'on_main_f_{k}', f'tl_f_{k}'], [f'draw_main_{k}']))
        nodes.append(helper.make_node('Mul', [f'on_anti_f_{k}', f'tr_f_{k}'], [f'draw_anti_{k}']))
        nodes.append(helper.make_node('Max', [f'draw_main_{k}', f'draw_anti_{k}'], [f'draw_any_{k}']))
        
        # in_box
        nodes.append(helper.make_node('GreaterOrEqual', ['row_indices', f'r_min_{k}'], [f'in_r1_{k}']))
        nodes.append(helper.make_node('LessOrEqual', ['row_indices', f'r_max_{k}'], [f'in_r2_{k}']))
        nodes.append(helper.make_node('GreaterOrEqual', ['col_indices', f'c_min_{k}'], [f'in_c1_{k}']))
        nodes.append(helper.make_node('LessOrEqual', ['col_indices', f'c_max_{k}'], [f'in_c2_{k}']))
        
        nodes.append(helper.make_node('Cast', [f'in_r1_{k}'], [f'in_r1_f_{k}'], to=F))
        nodes.append(helper.make_node('Cast', [f'in_r2_{k}'], [f'in_r2_f_{k}'], to=F))
        nodes.append(helper.make_node('Cast', [f'in_c1_{k}'], [f'in_c1_f_{k}'], to=F))
        nodes.append(helper.make_node('Cast', [f'in_c2_{k}'], [f'in_c2_f_{k}'], to=F))
        
        nodes.append(helper.make_node('Mul', [f'in_r1_f_{k}', f'in_r2_f_{k}'], [f'in_r_{k}']))
        nodes.append(helper.make_node('Mul', [f'in_c1_f_{k}', f'in_c2_f_{k}'], [f'in_c_{k}']))
        nodes.append(helper.make_node('Mul', [f'in_r_{k}', f'in_c_{k}'], [f'in_box_{k}']))
        
        nodes.append(helper.make_node('Mul', [f'draw_any_{k}', f'in_box_{k}'], [f'drawn_raw_{k}']))
        nodes.append(helper.make_node('Mul', [f'drawn_raw_{k}', 'c2_f'], [f'drawn_{k}']))
        
        out_channels.append(f'drawn_{k}')
        
    nodes.append(helper.make_node('Concat', out_channels, ['drawn_channels'], axis=1)) # [1, 10, 30, 30]
    
    nodes.append(helper.make_node('Add', ['input', 'drawn_channels'], ['final_logits']))
    
    nodes.append(helper.make_node('ArgMax', ['final_logits'], ['pred_idx'], axis=1, keepdims=0))
    nodes.append(helper.make_node('OneHot', ['pred_idx', 'depth10', 'oh_vals'], ['output'], axis=1))
    
    graph = helper.make_graph(nodes, 'task037', [x], [y], inits)
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
_T = 37
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
