import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

def create_model():
    F = TensorProto.FLOAT
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])
    
    def K(name, arr, dtype=np.int64): 
        return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)
        
    inits = [
        K('ax_1', [1]), K('ax_2', [2]),
        K('c0', [0]), K('c1', [1]),
        K('c5_s', [5]), K('c6_e', [6]),
        K('c0_5_f', [0.5], dtype=np.float32),
        K('row_indices', np.arange(30).reshape(1,1,30,1), dtype=np.int64),
        K('c0_i64', [0], dtype=np.int64),
        K('c2_oh', [0,0,1,0,0,0,0,0,0,0], dtype=np.float32)
    ]
    
    nodes = []
    
    nodes.append(helper.make_node('Slice', ['input', 'c5_s', 'c6_e', 'ax_1'], ['is_5']))
    
    nodes.append(helper.make_node('Slice', ['is_5', 'c0', 'c1', 'ax_2'], ['pattern_row']))
    nodes.append(helper.make_node('Greater', ['pattern_row', 'c0_5_f'], ['pattern_row_b']))
    
    nodes.append(helper.make_node('ReduceMax', ['is_5'], ['row_has_5'], axes=[3], keepdims=1))
    nodes.append(helper.make_node('Greater', ['row_has_5', 'c0_5_f'], ['row_has_5_b']))
    
    nodes.append(helper.make_node('Equal', ['row_indices', 'c0_i64'], ['is_R0']))
    nodes.append(helper.make_node('Not', ['is_R0'], ['not_R0']))
    
    nodes.append(helper.make_node('And', ['row_has_5_b', 'not_R0'], ['row_mask']))
    
    nodes.append(helper.make_node('And', ['pattern_row_b', 'row_mask'], ['place_2_b']))
    
    nodes.append(helper.make_node('Reshape', ['c2_oh', 'shape_1_10_1_1'], ['c2_oh_reshaped']))
    inits.append(K('shape_1_10_1_1', [1, 10, 1, 1]))
    
    nodes.append(helper.make_node('Where', ['place_2_b', 'c2_oh_reshaped', 'input'], ['output']))
    
    graph = helper.make_graph(nodes, 'task043', [x], [y], inits)
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
    """Same-shape task: zero the polluted 30x30 border via an input-presence mask."""
    _rename_output(m,"oh_raw")
    m.graph.node.append(helper.make_node("ReduceMax",["input"],["presence_m"],axes=[1],keepdims=1))
    m.graph.node.append(helper.make_node("Mul",["oh_raw","presence_m"],["output"]))
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
    return create_model()

model = _bake(_make(), 43)
