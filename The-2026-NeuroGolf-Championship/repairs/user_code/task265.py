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

# ===== task265 =====
# Ground-truth rule (arc_dsl_ref/solvers.py::solve_a8d7556c):
#   find every 2x2 all-background(color0) block occurring anywhere in the grid (occurrences()),
#   fill (union of) all of them with color 2.
# Empirically verified (numpy, against ALL of train+test+arc-gen, 266/266 exact) that this simple
# "fill every occurrence" already matches everything EXCEPT one single train example, where two
# occurrences are horizontally adjacent (share a full column) and one of the pair also has its own
# *vertical* neighbor occurrence (i.e. it's genuinely part of a taller vertical run) while the other
# does not -- in that specific situation the non-anchored one's unique (non-shared) column must be
# excluded from the fill. This local tie-break (checked with immediate vertical neighbors only, no
# long-range run-length needed) was verified to give n_fail=0 across all 266 train+test+arc-gen
# examples (see scratchpad trace265_numpy_rule4.py for the reference numpy implementation used to
# derive/verify this).
def build_265():
    x=helper.make_tensor_value_info('input',F,[1,10,30,30]); y=helper.make_tensor_value_info('output',F,[1,10,30,30])
    I=[_K('c1f',[1.0],np.float32),
       _K('onehot2', np.eye(10,dtype=np.float32)[2].reshape(1,10,1,1), np.float32),
       _K('s00',[0,0],np.int64), _K('e2929',[29,29],np.int64),
       _K('s01',[0,1],np.int64), _K('e2930',[29,30],np.int64),
       _K('s10',[1,0],np.int64), _K('e3029',[30,29],np.int64),
       _K('s11',[1,1],np.int64), _K('e3030',[30,30],np.int64),
       _K('ax23',[2,3],np.int64),
       _K('s0',[0],np.int64), _K('s1',[1],np.int64), _K('s2',[2],np.int64),
       _K('e28',[28],np.int64), _K('e29',[29],np.int64), _K('e30',[30],np.int64), _K('e31',[31],np.int64),
       _K('ax2',[2],np.int64), _K('ax3',[3],np.int64)]
    n=[]
    I.append(_K('ax1',[1],np.int64))
    n.append(helper.make_node('Slice',['input','s0','s1','ax1'],['is0']))  # [1,1,30,30]

    # M = 2x2 AND of is0
    n.append(helper.make_node('Slice',['is0','s00','e2929','ax23'],['m_a']))
    n.append(helper.make_node('Slice',['is0','s01','e2930','ax23'],['m_b']))
    n.append(helper.make_node('Slice',['is0','s10','e3029','ax23'],['m_c']))
    n.append(helper.make_node('Slice',['is0','s11','e3030','ax23'],['m_e']))
    n.append(helper.make_node('Mul',['m_a','m_b'],['m_ab']))
    n.append(helper.make_node('Mul',['m_ab','m_c'],['m_abc']))
    n.append(helper.make_node('Mul',['m_abc','m_e'],['M']))  # [1,1,29,29]

    # F: dilate M by 2x2 into 30x30 canvas (4 shifted placements, union)
    n.append(helper.make_node('Pad',['M','p_F00'],['F00'], mode='constant'))
    n.append(helper.make_node('Pad',['M','p_F01'],['F01'], mode='constant'))
    n.append(helper.make_node('Pad',['M','p_F10'],['F10'], mode='constant'))
    n.append(helper.make_node('Pad',['M','p_F11'],['F11'], mode='constant'))
    I += [_K('p_F00',[0,0,0,0, 0,0,1,1],np.int64),
          _K('p_F01',[0,0,0,1, 0,0,1,0],np.int64),
          _K('p_F10',[0,0,1,0, 0,0,0,1],np.int64),
          _K('p_F11',[0,0,1,1, 0,0,0,0],np.int64)]
    n.append(helper.make_node('Max',['F00','F01'],['Fs01']))
    n.append(helper.make_node('Max',['F10','F11'],['Fs23']))
    n.append(helper.make_node('Max',['Fs01','Fs23'],['Ffill']))  # [1,1,30,30]

    # hv[i,j] = M[i-1,j] OR M[i+1,j]  (immediate vertical neighbor presence, in M's own 29x29 frame)
    n.append(helper.make_node('Pad',['M','p_Mv'],['Mv'], mode='constant'))  # [1,1,31,29]
    I.append(_K('p_Mv',[0,0,1,0, 0,0,1,0],np.int64))
    n.append(helper.make_node('Slice',['Mv','s0','e29','ax2'],['M_up']))    # M[i-1,j]
    n.append(helper.make_node('Slice',['Mv','s2','e31','ax2'],['M_down']))  # M[i+1,j]
    n.append(helper.make_node('Max',['M_up','M_down'],['hv']))  # [1,1,29,29]

    # M_right[i,j] = M[i,j+1]; hv_right[i,j] = hv[i,j+1]
    n.append(helper.make_node('Pad',['M','p_Mh'],['Mh'], mode='constant'))  # [1,1,29,30]
    I.append(_K('p_Mh',[0,0,0,0, 0,0,0,1],np.int64))
    n.append(helper.make_node('Slice',['Mh','s1','e30','ax3'],['M_right']))
    n.append(helper.make_node('Pad',['hv','p_Mh'],['hvh'], mode='constant'))
    n.append(helper.make_node('Slice',['hvh','s1','e30','ax3'],['hv_right']))

    # condRight = M & M_right & hv & ~hv_right ; condLeft = M & M_right & hv_right & ~hv
    n.append(helper.make_node('Sub',['c1f','hv_right'],['not_hv_right']))
    n.append(helper.make_node('Sub',['c1f','hv'],['not_hv']))
    n.append(helper.make_node('Mul',['M','M_right'],['MMr']))
    n.append(helper.make_node('Mul',['MMr','hv'],['MMr_hv']))
    n.append(helper.make_node('Mul',['MMr_hv','not_hv_right'],['condRight']))  # [1,1,29,29]
    n.append(helper.make_node('Mul',['MMr','hv_right'],['MMr_hvr']))
    n.append(helper.make_node('Mul',['MMr_hvr','not_hv'],['condLeft']))       # [1,1,29,29]

    # condRight[i,j] (j==28 is always 0 anyway) excludes canvas cells (i,j+2) & (i+1,j+2);
    # pad begin=2 on W shifts right by 2, end=-1 crops the resulting overhang back to width 30.
    n.append(helper.make_node('Pad',['condRight','p_ExR00'],['ExR00'], mode='constant'))
    n.append(helper.make_node('Pad',['condRight','p_ExR10'],['ExR10'], mode='constant'))
    I += [_K('p_ExR00',[0,0,0,2, 0,0,1,-1],np.int64),
          _K('p_ExR10',[0,0,1,2, 0,0,0,-1],np.int64)]
    n.append(helper.make_node('Max',['ExR00','ExR10'],['ExR_sum']))

    # condLeft[i,j] excludes canvas cells (i,j) & (i+1,j)
    n.append(helper.make_node('Pad',['condLeft','p_ExL00'],['ExL00'], mode='constant'))
    n.append(helper.make_node('Pad',['condLeft','p_ExL10'],['ExL10'], mode='constant'))
    I += [_K('p_ExL00',[0,0,0,0, 0,0,1,1],np.int64),
          _K('p_ExL10',[0,0,1,0, 0,0,0,1],np.int64)]
    n.append(helper.make_node('Max',['ExL00','ExL10'],['ExL_sum']))

    n.append(helper.make_node('Max',['ExR_sum','ExL_sum'],['EXCLUDE']))  # [1,1,30,30]

    n.append(helper.make_node('Sub',['c1f','EXCLUDE'],['not_exclude']))
    n.append(helper.make_node('Mul',['Ffill','not_exclude'],['Ffinal']))  # [1,1,30,30]

    n.append(helper.make_node('Cast',['Ffinal'],['Ffinal_bool'], to=TensorProto.BOOL))
    n.append(helper.make_node('Where',['Ffinal_bool','onehot2','input'],['output']))

    return helper.make_model(helper.make_graph(n,'task265',[x],[y],I),ir_version=8,opset_imports=[helper.make_opsetid('',13)])

def _make():
    return build_265()

model = _bake(_make(), 265)
