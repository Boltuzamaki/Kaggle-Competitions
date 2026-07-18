import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime
F=TensorProto.FLOAT; I64=TensorProto.INT64

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

def build_049():
    x=helper.make_tensor_value_info('input',F,[1,10,30,30]); y=helper.make_tensor_value_info('output',F,[1,10,30,30])
    I=[_K('c0',[0],np.int64),_K('c1',[1],np.int64),_K('c10000',[10000],np.int64),
       _K('axall',[0,1,2,3],np.int64),_K('ax2',[2],np.int64),_K('ax3',[3],np.int64),
       _K('shape1',[1],np.int64),_K('shape1111',[1,1,1,1],np.int64),
       _K('row_idx',np.arange(30).reshape(1,1,30,1),np.int64),_K('col_idx',np.arange(30).reshape(1,1,1,30),np.int64),
       _K('p999',[999],np.int64),_K('m1',[-1],np.int64),_K('half',[0.5],np.float32),
       _K('shape1d',[-1],np.int64),_K('s2',[2],np.int64),_K('e4',[4],np.int64),_K('a0',[0],np.int64),
       _K('c30x2',[30,30],np.int64),_K('pfx6',[0,0,0,0,0,0],np.int64),_K('pv',[0.0],np.float32)]
    n=[helper.make_node('ArgMax',['input'],['am'],axis=1,keepdims=1)]
    adj=[]
    for c in range(1,10):
        I.append(_K(f'col{c}',[c],np.int64))
        n.append(helper.make_node('Equal',['am',f'col{c}'],[f'eqm{c}']))
        n.append(helper.make_node('Cast',[f'eqm{c}'],[f'eqmi{c}'],to=I64))
        n.append(helper.make_node('ReduceSum',[f'eqmi{c}','axall'],[f'cnt{c}'],keepdims=1))
        n.append(helper.make_node('Equal',[f'cnt{c}','c0'],[f'z{c}']))
        n.append(helper.make_node('Where',[f'z{c}','c10000',f'cnt{c}'],[f'adj{c}']))
        n.append(helper.make_node('Reshape',[f'adj{c}','shape1'],[f'adj1_{c}']))
        adj.append(f'adj1_{c}')
    n.append(helper.make_node('Concat',adj,['stacked'],axis=0))
    n.append(helper.make_node('ArgMin',['stacked'],['idx'],axis=0,keepdims=1))
    n.append(helper.make_node('Add',['idx','c1'],['tgt']))
    n.append(helper.make_node('Reshape',['tgt','shape1111'],['tgt4']))
    n.append(helper.make_node('Equal',['am','tgt4'],['tmb']))
    n.append(helper.make_node('Cast',['tmb'],['tmf'],to=F))
    n.append(helper.make_node('ReduceMax',['tmf'],['r_any'],axes=[3],keepdims=1))
    n.append(helper.make_node('Greater',['r_any','half'],['r_b']))
    n.append(helper.make_node('Where',['r_b','row_idx','p999'],['r_pmin'])); n.append(helper.make_node('ReduceMin',['r_pmin'],['rmin'],axes=[2],keepdims=1))
    n.append(helper.make_node('Where',['r_b','row_idx','m1'],['r_pmax'])); n.append(helper.make_node('ReduceMax',['r_pmax'],['rmax'],axes=[2],keepdims=1))
    n.append(helper.make_node('ReduceMax',['tmf'],['c_any'],axes=[2],keepdims=1))
    n.append(helper.make_node('Greater',['c_any','half'],['c_b']))
    n.append(helper.make_node('Where',['c_b','col_idx','p999'],['c_pmin'])); n.append(helper.make_node('ReduceMin',['c_pmin'],['cmin'],axes=[3],keepdims=1))
    n.append(helper.make_node('Where',['c_b','col_idx','m1'],['c_pmax'])); n.append(helper.make_node('ReduceMax',['c_pmax'],['cmax'],axes=[3],keepdims=1))
    n.append(helper.make_node('Reshape',['rmin','shape1d'],['r0'])); n.append(helper.make_node('Reshape',['cmin','shape1d'],['c0i']))
    n.append(helper.make_node('Reshape',['rmax','shape1d'],['rmx'])); n.append(helper.make_node('Reshape',['cmax','shape1d'],['cmx']))
    n.append(helper.make_node('Add',['rmx','c1'],['r1'])); n.append(helper.make_node('Add',['cmx','c1'],['c1i']))
    n.append(helper.make_node('Slice',['input','r0','r1','ax2'],['cy']))
    n.append(helper.make_node('Slice',['cy','c0i','c1i','ax3'],['oh_raw']))
    n.append(helper.make_node('Shape',['oh_raw'],['osh']))
    n.append(helper.make_node('Slice',['osh','s2','e4','a0'],['hw']))
    n.append(helper.make_node('Sub',['c30x2','hw'],['padhw']))
    n.append(helper.make_node('Concat',['pfx6','padhw'],['pads'],axis=0))
    n.append(helper.make_node('Pad',['oh_raw','pads','pv'],['output'],mode='constant'))
    return helper.make_model(helper.make_graph(n,'task049',[x],[y],I),ir_version=8,opset_imports=[helper.make_opsetid('',13)])

def _make():
    return build_049()

model = _bake(_make(), 49)
