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
    for base in [_os.environ.get("PROJECT_DIR","/project"), r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship", "."]:
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

def build_134():
    x=helper.make_tensor_value_info('input',F,[1,10,30,30])
    y=helper.make_tensor_value_info('output',F,[1,10,'H_out','W_out'])

    I=[
        _K('conv_w',np.ones((10,1,3,3),dtype=np.float32),np.float32),
        _K('ax23',[2,3],np.int64),
        _K('ax1',[1],np.int64),
        _K('s1',[1],np.int64), _K('e10',[10],np.int64),
        _K('idx19',[[1,2,3,4,5,6,7,8,9]],np.int64),
        _K('c0f',[0.0],np.float32),
        _K('one_f',[1.0],np.float32),
        _K('c1',[1],np.int64),
        _K('c3',[3],np.int64),
        _K('isch0_10',[[1,0,0,0,0,0,0,0,0,0]],np.float32),
        _K('row_idx',np.arange(30).reshape(1,1,30,1),np.int64),
        _K('col_idx',np.arange(30).reshape(1,1,1,30),np.int64),
        _K('m1',[-1],np.int64), _K('p999',[999],np.int64),
        _K('shape1d',[-1],np.int64),
        _K('shape1_9',[1,9],np.int64),
        _K('shape1_10',[1,10],np.int64),
        _K('shape_bc',[1,10,1,1],np.int64),
        _K('ax2',[2],np.int64), _K('ax3',[3],np.int64),
        _K('starts2',[0,0],np.int64), _K('ends2',[9999,9999],np.int64), _K('axes23',[2,3],np.int64),
    ]
    n=[]

    # local 3x3 density per channel (depthwise conv, all 10 channels at once)
    n.append(helper.make_node('Conv',['input','conv_w'],['dens'],group=10,pads=[1,1,1,1],strides=[1,1],kernel_shape=[3,3]))
    n.append(helper.make_node('ReduceMax',['dens'],['dens10'],axes=[2,3],keepdims=0))  # [1,10]
    n.append(helper.make_node('Slice',['dens10','s1','e10','ax1'],['dens9']))  # [1,9]
    n.append(helper.make_node('ArgMax',['dens9'],['mainIdx0'],axis=1,keepdims=1))  # [1,1] int64 in 0..8

    n.append(helper.make_node('Add',['mainIdx0','c1'],['mainIdx']))  # channel number 1..9, shape [1,1]

    n.append(helper.make_node('ReduceSum',['input','ax23'],['count10'],keepdims=0))  # [1,10]
    n.append(helper.make_node('Slice',['count10','s1','e10','ax1'],['count9']))  # [1,9]
    n.append(helper.make_node('Greater',['count9','c0f'],['presence9_b']))
    n.append(helper.make_node('Cast',['presence9_b'],['presence9'],to=F))

    n.append(helper.make_node('Equal',['idx19','mainIdx'],['mainOH9_b']))  # [1,9] bool, broadcasting mainIdx[1,1]
    n.append(helper.make_node('Cast',['mainOH9_b'],['mainOH9'],to=F))

    n.append(helper.make_node('Sub',['one_f','mainOH9'],['notMain9']))
    n.append(helper.make_node('Mul',['presence9','notMain9'],['noiseOH9']))

    # pad to length 10 (prepend zero for channel0)
    n.append(helper.make_node('Concat',['zero1_1','mainOH9'],['mainOH10'],axis=1))
    n.append(helper.make_node('Concat',['zero1_1','noiseOH9'],['noiseOH10'],axis=1))
    I.append(_K('zero1_1',[[0.0]],np.float32))

    n.append(helper.make_node('Reshape',['mainOH10','shape_bc'],['mainOH10_4d']))
    n.append(helper.make_node('Reshape',['noiseOH10','shape_bc'],['noiseOH10_4d']))
    n.append(helper.make_node('Reshape',['isch0_10','shape_bc'],['isch0_10_4d']))

    n.append(helper.make_node('Mul',['input','mainOH10_4d'],['mainMasked']))
    n.append(helper.make_node('ReduceSum',['mainMasked','ax1'],['mainMaskFull'],keepdims=1))  # [1,1,30,30]

    n.append(helper.make_node('Mul',['input','noiseOH10_4d'],['noiseMasked']))
    n.append(helper.make_node('ReduceSum',['noiseMasked','ax1'],['noiseMaskFull'],keepdims=1))  # [1,1,30,30]

    n.append(helper.make_node('Sub',['one_f','mainOH10_4d'],['notMain10_4d']))
    n.append(helper.make_node('Sub',['one_f','noiseOH10_4d'],['notNoise10_4d']))
    n.append(helper.make_node('Mul',['input','notMain10_4d'],['base_a']))
    n.append(helper.make_node('Mul',['base_a','notNoise10_4d'],['base']))

    n.append(helper.make_node('Mul',['mainMaskFull','noiseOH10_4d'],['plus_m2n']))
    n.append(helper.make_node('Mul',['noiseMaskFull','isch0_10_4d'],['plus_n2bg']))

    n.append(helper.make_node('Add',['base','plus_m2n'],['final_a']))
    n.append(helper.make_node('Add',['final_a','plus_n2bg'],['final']))  # swapped full one-hot grid [1,10,30,30]

    # bbox from mainMaskFull (same pattern as task031's bbox, based on presence mask)
    n.append(helper.make_node('ReduceMax',['mainMaskFull'],['row_any_f'],axes=[3],keepdims=1))
    n.append(helper.make_node('Greater',['row_any_f','c0f'],['row_any_b']))
    n.append(helper.make_node('Where',['row_any_b','row_idx','p999'],['row_pmin']))
    n.append(helper.make_node('ReduceMin',['row_pmin'],['r_min'],axes=[2],keepdims=1))
    n.append(helper.make_node('Where',['row_any_b','row_idx','m1'],['row_pmax']))
    n.append(helper.make_node('ReduceMax',['row_pmax'],['r_max'],axes=[2],keepdims=1))

    n.append(helper.make_node('ReduceMax',['mainMaskFull'],['col_any_f'],axes=[2],keepdims=1))
    n.append(helper.make_node('Greater',['col_any_f','c0f'],['col_any_b']))
    n.append(helper.make_node('Where',['col_any_b','col_idx','p999'],['col_pmin']))
    n.append(helper.make_node('ReduceMin',['col_pmin'],['c_min'],axes=[3],keepdims=1))
    n.append(helper.make_node('Where',['col_any_b','col_idx','m1'],['col_pmax']))
    n.append(helper.make_node('ReduceMax',['col_pmax'],['c_max'],axes=[3],keepdims=1))

    n.append(helper.make_node('Add',['r_max','c1'],['end_r']))
    n.append(helper.make_node('Add',['c_max','c1'],['end_c']))

    n.append(helper.make_node('Reshape',['r_min','shape1d'],['start_r_1d']))
    n.append(helper.make_node('Reshape',['end_r','shape1d'],['end_r_1d']))
    n.append(helper.make_node('Reshape',['c_min','shape1d'],['start_c_1d']))
    n.append(helper.make_node('Reshape',['end_c','shape1d'],['end_c_1d']))

    n.append(helper.make_node('Slice',['final','start_r_1d','end_r_1d','ax2'],['cropped_y']))
    n.append(helper.make_node('Slice',['cropped_y','start_c_1d','end_c_1d','ax3'],['cropped']))

    # height & downscale factor
    n.append(helper.make_node('Sub',['r_max','r_min'],['hgt_m1']))
    n.append(helper.make_node('Add',['hgt_m1','c1'],['hgt']))
    n.append(helper.make_node('Div',['hgt','c3'],['factor']))  # int64 floor div, shape [1,1,1,1]
    n.append(helper.make_node('Reshape',['factor','shape1d'],['factor_1d']))  # [1]
    n.append(helper.make_node('Concat',['factor_1d','factor_1d'],['steps2'],axis=0))  # [2]

    n.append(helper.make_node('Slice',['cropped','starts2','ends2','axes23','steps2'],['oh_raw']))

    graph = helper.make_graph(n,'task134',[x],[y],I)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('',13)])

def _make():
    return _crop_pad(build_134())

model = _bake(_make(), 134)
