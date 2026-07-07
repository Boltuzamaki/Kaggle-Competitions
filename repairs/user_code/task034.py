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
        K('c1', [1]), K('c2', [2]), K('c3', [3]), K('c10', [10]),
        K('c0_f', [0.0], dtype=np.float32),
        K('c0_5_f', [0.5], dtype=np.float32),
        K('row_indices', np.arange(30).reshape(1,1,30,1), dtype=np.int64),
        K('col_indices', np.arange(30).reshape(1,1,1,30), dtype=np.int64),
        K('m1', [-1]), K('p999', [999]),
        K('pad_9', [0, 0, 9, 9, 0, 0, 9, 9]),
        K('shape_1d', [-1]),
        K('mask_channels', np.array([0, 1, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32).reshape(1, 10, 1, 1), dtype=np.float32),
        K('channel_0_bias', (np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * 0.5).reshape(1, 10, 1, 1), dtype=np.float32),
        K('depth10', 10), K('oh_vals', [0.0, 1.0], dtype=np.float32)
    ]
    
    nodes = []
    
    # is_2
    nodes.append(helper.make_node('Slice', ['input', 'c2', 'c3', 'ax_1'], ['is_2']))
    
    # base_mask (input > 0 for colors 1..9)
    nodes.append(helper.make_node('Slice', ['input', 'c1', 'c10', 'ax_1'], ['is_not_0']))
    nodes.append(helper.make_node('ReduceMax', ['is_not_0'], ['is_any_color'], axes=[1], keepdims=1))
    nodes.append(helper.make_node('Greater', ['is_any_color', 'c0_5_f'], ['base_mask_bool']))
    nodes.append(helper.make_node('Cast', ['base_mask_bool'], ['base_mask'], to=F))
    
    # r_min, r_max
    nodes.append(helper.make_node('ReduceMax', ['base_mask'], ['row_any_float'], axes=[3], keepdims=1))
    nodes.append(helper.make_node('Greater', ['row_any_float', 'c0_5_f'], ['row_any_bool']))
    nodes.append(helper.make_node('Where', ['row_any_bool', 'row_indices', 'm1'], ['row_present']))
    nodes.append(helper.make_node('ReduceMax', ['row_present'], ['r_max'], axes=[2], keepdims=1))
    nodes.append(helper.make_node('Where', ['row_any_bool', 'row_indices', 'p999'], ['row_present_min']))
    nodes.append(helper.make_node('ReduceMin', ['row_present_min'], ['r_min'], axes=[2], keepdims=1))
    
    # c_min, c_max
    nodes.append(helper.make_node('ReduceMax', ['base_mask'], ['col_any_float'], axes=[2], keepdims=1))
    nodes.append(helper.make_node('Greater', ['col_any_float', 'c0_5_f'], ['col_any_bool']))
    nodes.append(helper.make_node('Where', ['col_any_bool', 'col_indices', 'm1'], ['col_present']))
    nodes.append(helper.make_node('ReduceMax', ['col_present'], ['c_max'], axes=[3], keepdims=1))
    nodes.append(helper.make_node('Where', ['col_any_bool', 'col_indices', 'p999'], ['col_present_min']))
    nodes.append(helper.make_node('ReduceMin', ['col_present_min'], ['c_min'], axes=[3], keepdims=1))
    
    # r_min_1d, r_max_1d, c_min_1d, c_max_1d
    nodes.append(helper.make_node('Reshape', ['r_min', 'shape_1d'], ['r_min_1d']))
    nodes.append(helper.make_node('Reshape', ['r_max', 'shape_1d'], ['r_max_1d']))
    nodes.append(helper.make_node('Reshape', ['c_min', 'shape_1d'], ['c_min_1d']))
    nodes.append(helper.make_node('Reshape', ['c_max', 'shape_1d'], ['c_max_1d']))
    
    nodes.append(helper.make_node('Add', ['r_min', 'c1'], ['r_min_p1']))
    nodes.append(helper.make_node('Reshape', ['r_min_p1', 'shape_1d'], ['r_min_p1_1d']))
    nodes.append(helper.make_node('Add', ['r_max', 'c1'], ['r_max_p1']))
    nodes.append(helper.make_node('Reshape', ['r_max_p1', 'shape_1d'], ['r_max_p1_1d']))
    nodes.append(helper.make_node('Add', ['c_min', 'c1'], ['c_min_p1']))
    nodes.append(helper.make_node('Reshape', ['c_min_p1', 'shape_1d'], ['c_min_p1_1d']))
    nodes.append(helper.make_node('Add', ['c_max', 'c1'], ['c_max_p1']))
    nodes.append(helper.make_node('Reshape', ['c_max_p1', 'shape_1d'], ['c_max_p1_1d']))
    
    # tl, tr, bl, br
    nodes.append(helper.make_node('Slice', ['is_2', 'r_min_1d', 'r_min_p1_1d', 'ax_2'], ['tl_y']))
    nodes.append(helper.make_node('Slice', ['tl_y', 'c_min_1d', 'c_min_p1_1d', 'ax_3'], ['tl']))
    nodes.append(helper.make_node('Slice', ['tl_y', 'c_max_1d', 'c_max_p1_1d', 'ax_3'], ['tr']))
    
    nodes.append(helper.make_node('Slice', ['is_2', 'r_max_1d', 'r_max_p1_1d', 'ax_2'], ['bl_y']))
    nodes.append(helper.make_node('Slice', ['bl_y', 'c_min_1d', 'c_min_p1_1d', 'ax_3'], ['bl']))
    nodes.append(helper.make_node('Slice', ['bl_y', 'c_max_1d', 'c_max_p1_1d', 'ax_3'], ['br']))
    
    nodes.append(helper.make_node('Greater', ['tl', 'c0_5_f'], ['tl_b']))
    nodes.append(helper.make_node('Cast', ['tl_b'], ['tl_f'], to=F))
    nodes.append(helper.make_node('Greater', ['tr', 'c0_5_f'], ['tr_b']))
    nodes.append(helper.make_node('Cast', ['tr_b'], ['tr_f'], to=F))
    nodes.append(helper.make_node('Greater', ['bl', 'c0_5_f'], ['bl_b']))
    nodes.append(helper.make_node('Cast', ['bl_b'], ['bl_f'], to=F))
    nodes.append(helper.make_node('Greater', ['br', 'c0_5_f'], ['br_b']))
    nodes.append(helper.make_node('Cast', ['br_b'], ['br_f'], to=F))
    
    # padded base_mask
    nodes.append(helper.make_node('Pad', ['base_mask', 'pad_9'], ['base_padded']))
    
    union_parts = ['base_mask']
    
    for d in range(1, 10):
        for name, r_shift, c_shift in [('tl', d, d), ('tr', d, -d), ('bl', -d, d), ('br', -d, -d)]:
            start_r = 9 + r_shift
            end_r = start_r + 30
            start_c = 9 + c_shift
            end_c = start_c + 30
            
            inits.extend([
                K(f'sr_{name}_{d}', [start_r]), K(f'er_{name}_{d}', [end_r]),
                K(f'sc_{name}_{d}', [start_c]), K(f'ec_{name}_{d}', [end_c])
            ])
            
            nodes.append(helper.make_node('Slice', ['base_padded', f'sr_{name}_{d}', f'er_{name}_{d}', 'ax_2'], [f'sy_{name}_{d}']))
            nodes.append(helper.make_node('Slice', [f'sy_{name}_{d}', f'sc_{name}_{d}', f'ec_{name}_{d}', 'ax_3'], [f'shifted_{name}_{d}']))
            nodes.append(helper.make_node('Mul', [f'shifted_{name}_{d}', f'{name}_f'], [f'ray_{name}_{d}']))
            union_parts.append(f'ray_{name}_{d}')
            
    nodes.append(helper.make_node('Max', union_parts, ['union_mask']))
    
    # c_color
    nodes.append(helper.make_node('Mul', ['input', 'mask_channels'], ['filtered_input']))
    nodes.append(helper.make_node('ReduceMax', ['filtered_input'], ['color_presence'], axes=[2, 3], keepdims=1)) # [1, 10, 1, 1]
    
    nodes.append(helper.make_node('Mul', ['union_mask', 'color_presence'], ['out_c_color'])) # [1, 10, 30, 30]
    
    nodes.append(helper.make_node('Add', ['out_c_color', 'channel_0_bias'], ['final_logits']))
    
    nodes.append(helper.make_node('ArgMax', ['final_logits'], ['pred_idx'], axis=1, keepdims=0))
    nodes.append(helper.make_node('OneHot', ['pred_idx', 'depth10', 'oh_vals'], ['output'], axis=1))
    
    graph = helper.make_graph(nodes, 'task034', [x], [y], inits)
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
_T = 34
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
