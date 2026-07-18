import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime
F=TensorProto.FLOAT; I64=TensorProto.INT64; B=TensorProto.BOOL

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

# ===== task268: box-outline interior fill + outward straight & diagonal rays (color 4) =====
def build_268():
    x=helper.make_tensor_value_info('input',F,[1,10,30,30])
    y=helper.make_tensor_value_info('output',F,[1,10,30,30])
    oh4=np.zeros((1,10,1,1),np.float32); oh4[0,4,0,0]=1.0
    I=[_K('R',np.arange(30).reshape(1,1,30,1),np.int64),
       _K('C',np.arange(30).reshape(1,1,1,30),np.int64),
       _K('c999',[999],np.int64),_K('cm1',[-1],np.int64),_K('cp1',[1],np.int64),_K('c0i',[0],np.int64),
       _K('half',[0.5],np.float32),
       _K('sl_s',[1],np.int64),_K('sl_e',[10],np.int64),_K('sl_a',[1],np.int64),
       _K('oh4',oh4,np.float32)]
    n=[]
    A=lambda *a,**k: n.append(helper.make_node(*a,**k))
    # non-background presence: any color in 1..9
    A('Slice',['input','sl_s','sl_e','sl_a'],['col19'])
    A('ReduceMax',['col19'],['nonbg'],axes=[1],keepdims=1)
    A('Greater',['nonbg','half'],['mask_b'])                 # [1,1,30,30] bool
    A('Cast',['mask_b'],['mask_f'],to=F)
    A('Not',['mask_b'],['notmask_b'])
    # bbox of mask
    A('ReduceMax',['mask_f'],['row_any_f'],axes=[3],keepdims=1)
    A('Greater',['row_any_f','half'],['row_any_b'])
    A('Where',['row_any_b','R','c999'],['r0w']); A('ReduceMin',['r0w'],['r0'],axes=[2],keepdims=1)
    A('Where',['row_any_b','R','cm1'],['r1w']);  A('ReduceMax',['r1w'],['r1'],axes=[2],keepdims=1)
    A('ReduceMax',['mask_f'],['col_any_f'],axes=[2],keepdims=1)
    A('Greater',['col_any_f','half'],['col_any_b'])
    A('Where',['col_any_b','C','c999'],['c0w']); A('ReduceMin',['c0w'],['c0'],axes=[3],keepdims=1)
    A('Where',['col_any_b','C','cm1'],['c1w']);  A('ReduceMax',['c1w'],['c1'],axes=[3],keepdims=1)
    # in_bbox / border / box / x4 / x5
    A('GreaterOrEqual',['R','r0'],['ge_r0']); A('LessOrEqual',['R','r1'],['le_r1'])
    A('GreaterOrEqual',['C','c0'],['ge_c0']); A('LessOrEqual',['C','c1'],['le_c1'])
    A('And',['ge_r0','le_r1'],['in_r']); A('And',['ge_c0','le_c1'],['in_c'])
    A('And',['in_r','in_c'],['in_bbox'])
    A('Equal',['R','r0'],['eq_r0']); A('Equal',['R','r1'],['eq_r1'])
    A('Equal',['C','c0'],['eq_c0']); A('Equal',['C','c1'],['eq_c1'])
    A('Or',['eq_r0','eq_r1'],['bor_r']); A('Or',['eq_c0','eq_c1'],['bor_c'])
    A('Or',['bor_r','bor_c'],['on_border'])
    A('And',['in_bbox','on_border'],['box_b'])
    A('And',['box_b','notmask_b'],['x4_b'])
    A('And',['in_bbox','notmask_b'],['x5_b'])
    A('Cast',['x4_b'],['x4_f'],to=F)
    # has opening
    A('ReduceMax',['x4_f'],['hasop_f'],axes=[0,1,2,3],keepdims=1)
    A('Greater',['hasop_f','half'],['hasop_b'])
    # x4 col/row spans
    A('ReduceMax',['x4_f'],['x4col_f'],axes=[2],keepdims=1); A('Greater',['x4col_f','half'],['x4col_b'])
    A('ReduceMax',['x4_f'],['x4row_f'],axes=[3],keepdims=1); A('Greater',['x4row_f','half'],['x4row_b'])
    A('Where',['x4col_b','C','c999'],['cminw']); A('ReduceMin',['cminw'],['cmin'],axes=[3],keepdims=1)
    A('Where',['x4col_b','C','cm1'],['cmaxw']);  A('ReduceMax',['cmaxw'],['cmax'],axes=[3],keepdims=1)
    A('Where',['x4row_b','R','c999'],['rminw']); A('ReduceMin',['rminw'],['rmin'],axes=[2],keepdims=1)
    A('Where',['x4row_b','R','cm1'],['rmaxw']);  A('ReduceMax',['rmaxw'],['rmax'],axes=[2],keepdims=1)
    # edge indicators (priority top>bottom>left>right)
    A('Equal',['rmax','r0'],['e_top']); A('And',['hasop_b','e_top'],['is_top'])
    A('Not',['is_top'],['n_top'])
    A('Equal',['rmin','r1'],['e_bot']); A('And',['hasop_b','n_top'],['hb1']); A('And',['hb1','e_bot'],['is_bot'])
    A('Not',['is_bot'],['n_bot'])
    A('Equal',['cmax','c0'],['e_lef']); A('And',['hb1','n_bot'],['hb2']); A('And',['hb2','e_lef'],['is_lef'])
    A('Not',['is_lef'],['n_lef'])
    A('Equal',['cmin','c1'],['e_rig']); A('And',['hb2','n_lef'],['hb3']); A('And',['hb3','e_rig'],['is_rig'])
    # straight fill
    A('LessOrEqual',['R','r0'],['R_le_r0']); A('GreaterOrEqual',['R','r1'],['R_ge_r1'])
    A('LessOrEqual',['C','c0'],['C_le_c0']); A('GreaterOrEqual',['C','c1'],['C_ge_c1'])
    A('And',['is_top','x4col_b'],['st1']); A('And',['st1','R_le_r0'],['s_top'])
    A('And',['is_bot','x4col_b'],['sb1']); A('And',['sb1','R_ge_r1'],['s_bot'])
    A('And',['is_lef','x4row_b'],['sl1']); A('And',['sl1','C_le_c0'],['s_lef'])
    A('And',['is_rig','x4row_b'],['sr1']); A('And',['sr1','C_ge_c1'],['s_rig'])
    A('Or',['s_top','s_bot'],['s_a']); A('Or',['s_lef','s_rig'],['s_b']); A('Or',['s_a','s_b'],['straight_b'])
    # blend helpers -> int64 scalars
    A('Cast',['is_top'],['itop'],to=I64); A('Cast',['is_bot'],['ibot'],to=I64)
    A('Cast',['is_lef'],['ilef'],to=I64); A('Cast',['is_rig'],['irig'],to=I64)
    def blend(name,vt,vb,vl,vr):
        A('Mul',['itop',vt],[name+'_t']); A('Mul',['ibot',vb],[name+'_b'])
        A('Mul',['ilef',vl],[name+'_l']); A('Mul',['irig',vr],[name+'_r'])
        A('Add',[name+'_t',name+'_b'],[name+'_tb']); A('Add',[name+'_l',name+'_r'],[name+'_lr'])
        A('Add',[name+'_tb',name+'_lr'],[name])
    blend('sIA','r0','r1','rmin','rmin')
    blend('sJA','cmin','cmin','c0','c1')
    blend('pIA','cm1','cp1','cm1','cm1')
    blend('pJA','cm1','cm1','cm1','cp1')
    blend('sIB','r0','r1','rmax','rmax')
    blend('sJB','cmax','cmax','c0','c1')
    blend('pIB','cm1','cp1','cp1','cp1')
    blend('pJB','cp1','cp1','cm1','cp1')
    # diag A
    A('Sub',['R','sIA'],['dRA']); A('Mul',['pIA','dRA'],['kAi'])
    A('Sub',['C','sJA'],['dCA']); A('Mul',['pJA','dCA'],['kAj'])
    A('Equal',['kAi','kAj'],['dA_eq']); A('GreaterOrEqual',['kAi','c0i'],['dA_ge'])
    A('And',['dA_eq','dA_ge'],['dA1']); A('And',['dA1','hasop_b'],['diagA_b'])
    # diag B
    A('Sub',['R','sIB'],['dRB']); A('Mul',['pIB','dRB'],['kBi'])
    A('Sub',['C','sJB'],['dCB']); A('Mul',['pJB','dCB'],['kBj'])
    A('Equal',['kBi','kBj'],['dB_eq']); A('GreaterOrEqual',['kBi','c0i'],['dB_ge'])
    A('And',['dB_eq','dB_ge'],['dB1']); A('And',['dB1','hasop_b'],['diagB_b'])
    # combine fill4
    A('Or',['x5_b','straight_b'],['f1']); A('Or',['diagA_b','diagB_b'],['f2']); A('Or',['f1','f2'],['fill4_b'])
    # build one-hot output
    A('Where',['fill4_b','oh4','input'],['output'])
    return helper.make_model(helper.make_graph(n,'task268',[x],[y],I),ir_version=8,opset_imports=[helper.make_opsetid('',13)])

def _make():
    return _mask(build_268())

model = _bake(_make(), 268)
