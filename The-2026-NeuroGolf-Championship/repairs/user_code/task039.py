import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

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
def _crop_pad(m):
    """OneHot 'output' is a dynamic [1,10,h,w] crop at top-left; Pad to static 30x30
    using h,w read from the tensor's own Shape (keeps padding all-zero)."""
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
    F=TensorProto.FLOAT
    x=helper.make_tensor_value_info('input',F,[1,10,30,30]); y=helper.make_tensor_value_info('output',F,[1,10,30,30])
    I=[_K('ch1',[1],np.int64),_K('ch10',[10],np.int64),_K('ax1',[1],np.int64),_K('ax2',[2],np.int64),_K('ax3',[3],np.int64),
       _K('row_idx',np.arange(30).reshape(1,1,30,1),np.int64),_K('col_idx',np.arange(30).reshape(1,1,1,30),np.int64),
       _K('p999',[999],np.int64),_K('half',[0.5],np.float32),_K('c3',[3],np.int64),_K('shape1d',[-1],np.int64),
       _K('pads',[0,0,0,0,0,0,27,27],np.int64),_K('pv',[0.0],np.float32)]
    n=[helper.make_node('Slice',['input','ch1','ch10','ax1'],['fgc']),
       helper.make_node('ReduceMax',['fgc'],['fg'],axes=[1],keepdims=1),
       helper.make_node('ReduceMax',['fg'],['r_any'],axes=[3],keepdims=1),
       helper.make_node('Greater',['r_any','half'],['r_any_b']),
       helper.make_node('Where',['r_any_b','row_idx','p999'],['r_pres']),
       helper.make_node('ReduceMin',['r_pres'],['r_min'],axes=[2],keepdims=1),
       helper.make_node('ReduceMax',['fg'],['c_any'],axes=[2],keepdims=1),
       helper.make_node('Greater',['c_any','half'],['c_any_b']),
       helper.make_node('Where',['c_any_b','col_idx','p999'],['c_pres']),
       helper.make_node('ReduceMin',['c_pres'],['c_min'],axes=[3],keepdims=1),
       helper.make_node('Reshape',['r_min','shape1d'],['r0']),
       helper.make_node('Reshape',['c_min','shape1d'],['c0']),
       helper.make_node('Add',['r0','c3'],['r1']), helper.make_node('Add',['c0','c3'],['c1']),
       helper.make_node('Slice',['input','r0','r1','ax2'],['cy']),
       helper.make_node('Slice',['cy','c0','c1','ax3'],['crop']),
       helper.make_node('Pad',['crop','pads','pv'],['output'],mode='constant')]
    return helper.make_model(helper.make_graph(n,'task039',[x],[y],I),ir_version=8,opset_imports=[helper.make_opsetid('',13)])

model = _bake(_make(), 39)
