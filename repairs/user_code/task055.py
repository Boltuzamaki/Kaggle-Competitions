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
        K('c0_i64', [0], dtype=np.int64),
        K('c1_i64', [1], dtype=np.int64),
        K('c2_i64', [2], dtype=np.int64),
        K('c3_i64', [3], dtype=np.int64),
        K('c4_i64', [4], dtype=np.int64),
        K('c6_i64', [6], dtype=np.int64),
        K('c8_i64', [8], dtype=np.int64),
        K('ax_2_1d', [2], dtype=np.int64),
        K('ax_3_1d', [3], dtype=np.int64),
        K('L_upper', np.triu(np.ones((30, 30), dtype=np.int64)), dtype=np.int64),
        K('c_oh_vals', [0.0, 1.0], dtype=np.float32),
        K('c_depth_10', [10], dtype=np.int64),
    ]
    
    nodes = []
    
    nodes.append(helper.make_node('ArgMax', ['input'], ['argmax'], axis=1, keepdims=1))
    
    nodes.append(helper.make_node('Equal', ['argmax', 'c8_i64'], ['is_8']))
    nodes.append(helper.make_node('Cast', ['is_8'], ['is_8_i64'], to=I64))
    
    nodes.append(helper.make_node('ReduceSum', ['is_8_i64', 'ax_3_1d'], ['row_sums'], keepdims=1))
    nodes.append(helper.make_node('ReduceSum', ['is_8_i64', 'ax_2_1d'], ['col_sums'], keepdims=1))
    
    nodes.append(helper.make_node('Greater', ['row_sums', 'c2_i64'], ['is_hline']))
    nodes.append(helper.make_node('Greater', ['col_sums', 'c2_i64'], ['is_vline']))
    
    nodes.append(helper.make_node('Cast', ['is_hline'], ['is_hline_i64'], to=I64))
    nodes.append(helper.make_node('Cast', ['is_vline'], ['is_vline_i64'], to=I64))
    
    # cum_r
    nodes.append(helper.make_node('Transpose', ['is_hline_i64'], ['is_hline_T'], perm=[0, 1, 3, 2]))
    nodes.append(helper.make_node('MatMul', ['is_hline_T', 'L_upper'], ['cum_r_T']))
    nodes.append(helper.make_node('Transpose', ['cum_r_T'], ['cum_r'], perm=[0, 1, 3, 2]))
    
    # cum_c
    nodes.append(helper.make_node('MatMul', ['is_vline_i64', 'L_upper'], ['cum_c']))
    
    # row regions
    nodes.append(helper.make_node('Not', ['is_hline'], ['not_hline']))
    nodes.append(helper.make_node('Equal', ['cum_r', 'c0_i64'], ['is_top']))
    nodes.append(helper.make_node('Equal', ['cum_r', 'c1_i64'], ['cum_r_1']))
    nodes.append(helper.make_node('And', ['cum_r_1', 'not_hline'], ['is_mid_r']))
    nodes.append(helper.make_node('Equal', ['cum_r', 'c2_i64'], ['cum_r_2']))
    nodes.append(helper.make_node('And', ['cum_r_2', 'not_hline'], ['is_bot']))
    
    # col regions
    nodes.append(helper.make_node('Not', ['is_vline'], ['not_vline']))
    nodes.append(helper.make_node('Equal', ['cum_c', 'c0_i64'], ['is_left']))
    nodes.append(helper.make_node('Equal', ['cum_c', 'c1_i64'], ['cum_c_1']))
    nodes.append(helper.make_node('And', ['cum_c_1', 'not_vline'], ['is_mid_c']))
    nodes.append(helper.make_node('Equal', ['cum_c', 'c2_i64'], ['cum_c_2']))
    nodes.append(helper.make_node('And', ['cum_c_2', 'not_vline'], ['is_right']))
    
    # combinations
    nodes.append(helper.make_node('And', ['is_mid_r', 'is_mid_c'], ['center_mask']))
    nodes.append(helper.make_node('And', ['is_top', 'is_mid_c'], ['top_mask']))
    nodes.append(helper.make_node('And', ['is_bot', 'is_mid_c'], ['bot_mask']))
    nodes.append(helper.make_node('And', ['is_mid_r', 'is_left'], ['left_mask']))
    nodes.append(helper.make_node('And', ['is_mid_r', 'is_right'], ['right_mask']))
    
    # where logic
    nodes.append(helper.make_node('Where', ['center_mask', 'c6_i64', 'c0_i64'], ['out_1']))
    nodes.append(helper.make_node('Where', ['top_mask', 'c2_i64', 'out_1'], ['out_2']))
    nodes.append(helper.make_node('Where', ['bot_mask', 'c1_i64', 'out_2'], ['out_3']))
    nodes.append(helper.make_node('Where', ['left_mask', 'c4_i64', 'out_3'], ['out_4']))
    nodes.append(helper.make_node('Where', ['right_mask', 'c3_i64', 'out_4'], ['out_5']))
    
    nodes.append(helper.make_node('Where', ['is_8', 'c8_i64', 'out_5'], ['out_final']))
    
    # output formatting
    nodes.append(helper.make_node('OneHot', ['out_final', 'c_depth_10', 'c_oh_vals'], ['pred_oh'], axis=-1))
    nodes.append(helper.make_node('Transpose', ['pred_oh'], ['pred_trans'], perm=[0, 4, 1, 2, 3]))
    nodes.append(helper.make_node('Squeeze', ['pred_trans', 'ax_2_1d'], ['output']))
    
    graph = helper.make_graph(nodes, 'task055', [x], [y], inits)
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

model = _bake(_make(), 55)
