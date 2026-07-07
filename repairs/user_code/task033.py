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
        K('c1', [1]), K('c10', [10]),
        K('c5', [5]), K('c6', [6]),
        K('pad_12', [0, 0, 12, 12, 0, 0, 12, 12]),
        K('c0_f', [0.0], dtype=np.float32),
        K('c0_5_f', [0.5], dtype=np.float32),
        K('c2_f', [2.0], dtype=np.float32),
        K('depth10', 10), K('oh_vals', [0.0, 1.0], dtype=np.float32)
    ]
    
    nodes = []
    
    # Extract c_grid
    nodes.append(helper.make_node('Slice', ['input', 'c5', 'c6', 'ax_2'], ['c_grid_y'])) # [1, 10, 1, 30]
    nodes.append(helper.make_node('Slice', ['c_grid_y', 'c5', 'c6', 'ax_3'], ['c_grid'])) # [1, 10, 1, 1]
    nodes.append(helper.make_node('Greater', ['c_grid', 'c0_5_f'], ['c_grid_bool']))
    nodes.append(helper.make_node('Cast', ['c_grid_bool'], ['c_grid_mask'], to=F)) # [1, 10, 1, 1]
    
    # grid_mask_any
    nodes.append(helper.make_node('Mul', ['input', 'c_grid_mask'], ['grid_mask'])) # [1, 10, 30, 30]
    nodes.append(helper.make_node('ReduceMax', ['grid_mask'], ['grid_mask_any'], axes=[1], keepdims=1)) # [1, 1, 30, 30]
    
    # shape_mask
    nodes.append(helper.make_node('Slice', ['input', 'c1', 'c10', 'ax_1'], ['is_not_0'])) # [1, 9, 30, 30]
    nodes.append(helper.make_node('ReduceMax', ['is_not_0'], ['is_any_color'], axes=[1], keepdims=1)) # [1, 1, 30, 30]
    nodes.append(helper.make_node('Greater', ['is_any_color', 'c0_5_f'], ['is_any_bool']))
    nodes.append(helper.make_node('Cast', ['is_any_bool'], ['is_any_f'], to=F))
    nodes.append(helper.make_node('Sub', ['is_any_f', 'grid_mask_any'], ['shape_mask'])) # [1, 1, 30, 30]
    
    # Pad shape_mask
    nodes.append(helper.make_node('Pad', ['shape_mask', 'pad_12'], ['shape_mask_padded'])) # [1, 1, 54, 54]
    
    union_parts = []
    
    for dy in [-12, -6, 0, 6, 12]:
        for dx in [-12, -6, 0, 6, 12]:
            start_r = 12 - dy
            end_r = 42 - dy
            start_c = 12 - dx
            end_c = 42 - dx
            
            inits.extend([
                K(f'sr_{dy}_{dx}', [start_r]), K(f'er_{dy}_{dx}', [end_r]),
                K(f'sc_{dy}_{dx}', [start_c]), K(f'ec_{dy}_{dx}', [end_c])
            ])
            
            nodes.append(helper.make_node('Slice', ['shape_mask_padded', f'sr_{dy}_{dx}', f'er_{dy}_{dx}', 'ax_2'], [f's_y_{dy}_{dx}']))
            nodes.append(helper.make_node('Slice', [f's_y_{dy}_{dx}', f'sc_{dy}_{dx}', f'ec_{dy}_{dx}', 'ax_3'], [f'shifted_{dy}_{dx}']))
            
            union_parts.append(f'shifted_{dy}_{dx}')
            
    # Max over 25 parts
    nodes.append(helper.make_node('Max', union_parts, ['union_mask']))
    
    # new_grid_pixels
    nodes.append(helper.make_node('Sub', ['union_mask', 'shape_mask'], ['new_grid_pixels']))
    
    # add to c_grid channel
    nodes.append(helper.make_node('Mul', ['new_grid_pixels', 'c_grid_mask'], ['added_one_hot'])) # [1, 10, 30, 30]
    nodes.append(helper.make_node('Mul', ['added_one_hot', 'c2_f'], ['added_one_hot_2'])) # Add 2.0 to override background
    
    nodes.append(helper.make_node('Add', ['input', 'added_one_hot_2'], ['final_logits']))
    
    # output
    nodes.append(helper.make_node('ArgMax', ['final_logits'], ['pred_idx'], axis=1, keepdims=0))
    nodes.append(helper.make_node('OneHot', ['pred_idx', 'depth10', 'oh_vals'], ['output'], axis=1))
    
    graph = helper.make_graph(nodes, 'task033', [x], [y], inits)
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
_T = 33
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
