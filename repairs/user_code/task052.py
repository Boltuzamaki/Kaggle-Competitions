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
        K('c5_i64', [5], dtype=np.int64),
        K('ax_0_1d', [0], dtype=np.int64),
        K('ax_2_1d', [2], dtype=np.int64),
        K('ax_3_1d', [3], dtype=np.int64),
        K('c_oh_vals', [0.0, 1.0], dtype=np.float32),
        K('c_depth_10', [10], dtype=np.int64),
    ]
    
    nodes = []
    
    nodes.append(helper.make_node('ArgMax', ['input'], ['argmax'], axis=1, keepdims=1))
    
    has_colors = []
    for c in range(1, 10):
        c_name = f'c_{c}_i64'
        inits.append(K(c_name, [c], dtype=np.int64))
        
        mask_name = f'mask_{c}'
        nodes.append(helper.make_node('Equal', ['argmax', c_name], [mask_name]))
        
        mask_i64 = f'mask_{c}_i64'
        nodes.append(helper.make_node('Cast', [mask_name], [mask_i64], to=I64))
        
        has_c = f'has_{c}'
        # ReduceMax over axis 3 (columns) to check if color is present in the row
        nodes.append(helper.make_node('ReduceMax', [mask_i64], [has_c], axes=[3], keepdims=1))
        has_colors.append(has_c)
        
    nodes.append(helper.make_node('Concat', has_colors, ['concat_colors'], axis=0)) # [9, 1, 1, 30, 1] - wait, axis=0 makes it [9, 1, 30, 1] because has_c is [1, 1, 30, 1]
    
    nodes.append(helper.make_node('ReduceSum', ['concat_colors', 'ax_0_1d'], ['num_colors'], keepdims=1))
    
    nodes.append(helper.make_node('Equal', ['num_colors', 'c1_i64'], ['is_uniform']))
    
    nodes.append(helper.make_node('Greater', ['argmax', 'c0_i64'], ['valid_mask']))
    nodes.append(helper.make_node('And', ['is_uniform', 'valid_mask'], ['fill_mask']))
    
    nodes.append(helper.make_node('Where', ['fill_mask', 'c5_i64', 'c0_i64'], ['out_val']))
    
    nodes.append(helper.make_node('OneHot', ['out_val', 'c_depth_10', 'c_oh_vals'], ['pred_oh'], axis=-1))
    nodes.append(helper.make_node('Transpose', ['pred_oh'], ['pred_trans'], perm=[0, 4, 1, 2, 3]))
    nodes.append(helper.make_node('Squeeze', ['pred_trans', 'ax_2_1d'], ['output']))
    
    graph = helper.make_graph(nodes, 'task052', [x], [y], inits)
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

model = _bake(_make(), 52)
