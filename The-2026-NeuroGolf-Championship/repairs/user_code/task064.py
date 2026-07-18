import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

def create_model():
    F = TensorProto.FLOAT
    I64 = TensorProto.INT64
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, ['batch', 10, 'height', 'width'])
    
    def K(name, arr, dtype=np.int64): 
        return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)
        
    inits = [
        K('c0_i64', [0]),
        K('c1_i64', [1]),
        K('cm1_i64', [-1]),
        K('c999_i64', [999]),
        K('ax_2_3_1d', [2, 3]),
        K('ax_3_1d', [3]),
        K('ax_2_1d', [2]),
        K('ax_m1_1d', [-1]),
        K('c_0_to_9', np.arange(10)),
        K('shape_1_10_1_1', [1, 10, 1, 1]),
        K('shape_1_10', [1, 10]),
        K('shape_1_1_1_1', [1, 1, 1, 1]),
        K('shape_h1', [1, 1, 30, 30, 1]),
        K('shape_h2', [1, 1, 1, 30, 1]),
        K('shape_h3', [1, 1, 1, 1, 30]),
        K('shape_v1', [1, 1, 30, 1, 30]),
        K('shape_v2', [1, 1, 30, 1, 1]),
        K('shape_v3', [1, 1, 1, 30, 1]),
        K('r', np.arange(30).reshape(1, 1, 30, 1)),
        K('c', np.arange(30).reshape(1, 1, 1, 30)),
        K('c_oh_vals', [0.0, 1.0], dtype=np.float32),
        K('c_depth_10', [10]),
    ]
    
    nodes = []
    
    nodes.append(helper.make_node('ArgMax', ['input'], ['argmax'], axis=1, keepdims=1))
    
    nodes.append(helper.make_node('Reshape', ['c_0_to_9', 'shape_1_10_1_1'], ['color_idx']))
    nodes.append(helper.make_node('Equal', ['argmax', 'color_idx'], ['color_masks'])) 
    
    nodes.append(helper.make_node('Cast', ['color_masks'], ['color_masks_i64'], to=I64))
    nodes.append(helper.make_node('ReduceSum', ['color_masks_i64', 'ax_2_3_1d'], ['counts'], keepdims=0)) 
    
    nodes.append(helper.make_node('Reshape', ['c_0_to_9', 'shape_1_10'], ['c_0_to_9_flat']))
    nodes.append(helper.make_node('Equal', ['c_0_to_9_flat', 'c0_i64'], ['is_0_color']))
    nodes.append(helper.make_node('Not', ['is_0_color'], ['is_not_0_color']))
    
    nodes.append(helper.make_node('Cast', ['is_not_0_color'], ['is_not_0_color_i64'], to=I64))
    nodes.append(helper.make_node('Mul', ['counts', 'is_not_0_color_i64'], ['counts_no_0']))
    
    nodes.append(helper.make_node('ArgMax', ['counts_no_0'], ['bg_color_idx'], axis=1)) 
    
    nodes.append(helper.make_node('Where', ['color_masks', 'r', 'cm1_i64'], ['r_indices']))
    nodes.append(helper.make_node('ReduceMax', ['r_indices'], ['r_max'], axes=[2, 3], keepdims=0)) 
    nodes.append(helper.make_node('Where', ['color_masks', 'r', 'c999_i64'], ['r_min_cand']))
    nodes.append(helper.make_node('ReduceMin', ['r_min_cand'], ['r_min'], axes=[2, 3], keepdims=0)) 
    
    nodes.append(helper.make_node('Where', ['color_masks', 'c', 'cm1_i64'], ['c_indices']))
    nodes.append(helper.make_node('ReduceMax', ['c_indices'], ['c_max'], axes=[2, 3], keepdims=0))
    nodes.append(helper.make_node('Where', ['color_masks', 'c', 'c999_i64'], ['c_min_cand']))
    nodes.append(helper.make_node('ReduceMin', ['c_min_cand'], ['c_min'], axes=[2, 3], keepdims=0))
    
    nodes.append(helper.make_node('Sub', ['r_max', 'r_min'], ['h_m1']))
    nodes.append(helper.make_node('Add', ['h_m1', 'c1_i64'], ['h']))
    nodes.append(helper.make_node('Sub', ['c_max', 'c_min'], ['w_m1']))
    nodes.append(helper.make_node('Add', ['w_m1', 'c1_i64'], ['w']))
    nodes.append(helper.make_node('Mul', ['h', 'w'], ['area']))
    
    nodes.append(helper.make_node('Equal', ['area', 'counts'], ['area_eq_counts']))
    nodes.append(helper.make_node('Greater', ['counts', 'c1_i64'], ['counts_gt_1']))
    
    nodes.append(helper.make_node('And', ['area_eq_counts', 'counts_gt_1'], ['is_block_1']))
    nodes.append(helper.make_node('And', ['is_block_1', 'is_not_0_color'], ['is_block']))
    
    nodes.append(helper.make_node('Cast', ['is_block'], ['is_block_i64'], to=I64))
    nodes.append(helper.make_node('ArgMax', ['is_block_i64'], ['block_color_idx'], axis=1)) 
    
    nodes.append(helper.make_node('GatherElements', ['r_min', 'block_color_idx'], ['b_r_min'], axis=1))
    nodes.append(helper.make_node('GatherElements', ['r_max', 'block_color_idx'], ['b_r_max'], axis=1))
    nodes.append(helper.make_node('GatherElements', ['c_min', 'block_color_idx'], ['b_c_min'], axis=1))
    nodes.append(helper.make_node('GatherElements', ['c_max', 'block_color_idx'], ['b_c_max'], axis=1))
    
    nodes.append(helper.make_node('Equal', ['c_0_to_9_flat', 'bg_color_idx'], ['is_bg']))
    nodes.append(helper.make_node('Equal', ['c_0_to_9_flat', 'block_color_idx'], ['is_blk']))
    nodes.append(helper.make_node('Not', ['is_bg'], ['not_bg']))
    nodes.append(helper.make_node('Not', ['is_blk'], ['not_blk']))
    nodes.append(helper.make_node('Greater', ['counts', 'c0_i64'], ['count_gt_0']))
    
    nodes.append(helper.make_node('And', ['count_gt_0', 'not_bg'], ['is_line_1']))
    nodes.append(helper.make_node('And', ['is_line_1', 'not_blk'], ['is_line_2']))
    nodes.append(helper.make_node('And', ['is_line_2', 'is_not_0_color'], ['is_line']))
    
    nodes.append(helper.make_node('Cast', ['is_line'], ['is_line_i64'], to=I64))
    nodes.append(helper.make_node('ArgMax', ['is_line_i64'], ['line_color_idx'], axis=1))
    
    nodes.append(helper.make_node('Reshape', ['line_color_idx', 'shape_1_1_1_1'], ['line_color']))
    nodes.append(helper.make_node('Reshape', ['b_r_min', 'shape_1_1_1_1'], ['b_r_min_4d']))
    nodes.append(helper.make_node('Reshape', ['b_r_max', 'shape_1_1_1_1'], ['b_r_max_4d']))
    nodes.append(helper.make_node('Reshape', ['b_c_min', 'shape_1_1_1_1'], ['b_c_min_4d']))
    nodes.append(helper.make_node('Reshape', ['b_c_max', 'shape_1_1_1_1'], ['b_c_max_4d']))
    
    nodes.append(helper.make_node('Equal', ['argmax', 'line_color'], ['mask_line'])) 
    
    nodes.append(helper.make_node('Reshape', ['mask_line', 'shape_h1'], ['mask_line_h']))
    nodes.append(helper.make_node('Reshape', ['c', 'shape_h2'], ['c_key_h']))
    nodes.append(helper.make_node('Reshape', ['c', 'shape_h3'], ['c_query_h']))
    
    nodes.append(helper.make_node('LessOrEqual', ['c_key_h', 'c_query_h'], ['c_le']))
    nodes.append(helper.make_node('And', ['mask_line_h', 'c_le'], ['valid_left']))
    nodes.append(helper.make_node('Cast', ['valid_left'], ['valid_left_i64'], to=I64))
    nodes.append(helper.make_node('ReduceMax', ['valid_left_i64'], ['any_valid_left'], axes=[3], keepdims=0))
    nodes.append(helper.make_node('Cast', ['any_valid_left'], ['any_valid_left_b'], to=TensorProto.BOOL))
    
    nodes.append(helper.make_node('GreaterOrEqual', ['c_key_h', 'c_query_h'], ['c_ge']))
    nodes.append(helper.make_node('And', ['mask_line_h', 'c_ge'], ['valid_right']))
    nodes.append(helper.make_node('Cast', ['valid_right'], ['valid_right_i64'], to=I64))
    nodes.append(helper.make_node('ReduceMax', ['valid_right_i64'], ['any_valid_right'], axes=[3], keepdims=0))
    nodes.append(helper.make_node('Cast', ['any_valid_right'], ['any_valid_right_b'], to=TensorProto.BOOL))
    
    nodes.append(helper.make_node('Less', ['c', 'b_c_min_4d'], ['c_lt_b']))
    nodes.append(helper.make_node('Greater', ['c', 'b_c_max_4d'], ['c_gt_b']))
    nodes.append(helper.make_node('And', ['c_lt_b', 'any_valid_left_b'], ['paint_left']))
    nodes.append(helper.make_node('And', ['c_gt_b', 'any_valid_right_b'], ['paint_right']))
    
    nodes.append(helper.make_node('GreaterOrEqual', ['r', 'b_r_min_4d'], ['r_ge_bmin']))
    nodes.append(helper.make_node('LessOrEqual', ['r', 'b_r_max_4d'], ['r_le_bmax']))
    nodes.append(helper.make_node('And', ['r_ge_bmin', 'r_le_bmax'], ['in_block_row']))
    
    nodes.append(helper.make_node('Or', ['paint_left', 'paint_right'], ['paint_h_any']))
    nodes.append(helper.make_node('And', ['in_block_row', 'paint_h_any'], ['paint_h']))
    
    nodes.append(helper.make_node('Reshape', ['mask_line', 'shape_v1'], ['mask_line_v']))
    nodes.append(helper.make_node('Reshape', ['r', 'shape_v2'], ['r_key_v']))
    nodes.append(helper.make_node('Reshape', ['r', 'shape_v3'], ['r_query_v']))
    
    nodes.append(helper.make_node('LessOrEqual', ['r_key_v', 'r_query_v'], ['r_le']))
    nodes.append(helper.make_node('And', ['mask_line_v', 'r_le'], ['valid_up']))
    nodes.append(helper.make_node('Cast', ['valid_up'], ['valid_up_i64'], to=I64))
    nodes.append(helper.make_node('ReduceMax', ['valid_up_i64'], ['any_valid_up'], axes=[2], keepdims=0))
    nodes.append(helper.make_node('Cast', ['any_valid_up'], ['any_valid_up_b'], to=TensorProto.BOOL))
    
    nodes.append(helper.make_node('GreaterOrEqual', ['r_key_v', 'r_query_v'], ['r_ge']))
    nodes.append(helper.make_node('And', ['mask_line_v', 'r_ge'], ['valid_down']))
    nodes.append(helper.make_node('Cast', ['valid_down'], ['valid_down_i64'], to=I64))
    nodes.append(helper.make_node('ReduceMax', ['valid_down_i64'], ['any_valid_down'], axes=[2], keepdims=0))
    nodes.append(helper.make_node('Cast', ['any_valid_down'], ['any_valid_down_b'], to=TensorProto.BOOL))
    
    nodes.append(helper.make_node('Less', ['r', 'b_r_min_4d'], ['r_lt_b']))
    nodes.append(helper.make_node('Greater', ['r', 'b_r_max_4d'], ['r_gt_b']))
    nodes.append(helper.make_node('And', ['r_lt_b', 'any_valid_up_b'], ['paint_up']))
    nodes.append(helper.make_node('And', ['r_gt_b', 'any_valid_down_b'], ['paint_down']))
    
    nodes.append(helper.make_node('GreaterOrEqual', ['c', 'b_c_min_4d'], ['c_ge_bmin']))
    nodes.append(helper.make_node('LessOrEqual', ['c', 'b_c_max_4d'], ['c_le_bmax']))
    nodes.append(helper.make_node('And', ['c_ge_bmin', 'c_le_bmax'], ['in_block_col']))
    
    nodes.append(helper.make_node('Or', ['paint_up', 'paint_down'], ['paint_v_any']))
    nodes.append(helper.make_node('And', ['in_block_col', 'paint_v_any'], ['paint_v']))
    
    nodes.append(helper.make_node('Or', ['paint_h', 'paint_v'], ['paint_all']))
    
    nodes.append(helper.make_node('Where', ['paint_all', 'line_color', 'argmax'], ['out_val']))
    
    nodes.append(helper.make_node('OneHot', ['out_val', 'c_depth_10', 'c_oh_vals'], ['pred_oh'], axis=-1))
    nodes.append(helper.make_node('Transpose', ['pred_oh'], ['pred_trans'], perm=[0, 4, 1, 2, 3]))
    nodes.append(helper.make_node('Squeeze', ['pred_trans', 'ax_2_1d'], ['output']))
    
    graph = helper.make_graph(nodes, 'task064', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])

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
    _rename_output(m,"oh_raw")
    m.graph.node.append(helper.make_node("ReduceMax",["input"],["presence_m"],axes=[1],keepdims=1))
    m.graph.node.append(helper.make_node("Mul",["oh_raw","presence_m"],["output"]))
    _set_out_shape(m,[1,10,30,30]); return m
def _crop_pad(m):
    _rename_output(m,"oh_raw")
    m.graph.initializer.extend([_K("__s2",[2],np.int64),_K("__e4",[4],np.int64),_K("__a0",[0],np.int64),
        _K("__30x2",[30,30],np.int64),_K("__pfx6",[0,0,0,0,0,0],np.int64),_K("__pv",[0.0],np.float32)])
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

def _make():
    return _mask(create_model())

model = _bake(_make(), 64)
