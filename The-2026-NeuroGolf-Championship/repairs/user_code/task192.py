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

# ===== task192 =====
# DSL rule (verified against ALL train+test+arc-gen = 265 examples in pure numpy, 0 fails):
#   x1 = leastcolor(I)                    -- rarest color (excl. background 0, never a tie in the data)
#   x3 = replace(I, x1, 0)                -- remove x1's cells (paint background)
#   x4 = leastcolor(x3)                   -- rarest color of what's left (excl. 0, never a tie)
#   For each x1-colored cell p: look at its 4 dneighbors in x3.
#     Convert p to x4 iff it has >=1 vertical (up/down) neighbor == x4
#                      AND >=1 horizontal (left/right) neighbor == x4.
#     (NOTE: this is NOT the same as "count(x4-neighbors) >= 2" -- empirically,
#      2 matching neighbors that are collinear/opposite (both up+down, or both
#      left+right) do NOT convert; only when the two matches span one vertical +
#      one horizontal direction (i.e. touch a "corner") does it convert. This was
#      confirmed by tabulating every rarest-color cell across all 265 examples:
#      nb=0,1 always no-convert; nb=3,4 always convert; nb=2 splits 197 convert
#      (mixed v+h) / 44 no-convert (pure opposite pair) -- the v-AND-h condition
#      explains 100% of cases.)
#     Otherwise: p becomes background (0), i.e. it's simply erased.
#   Cells that were not color x1 are unchanged.

def build_192():
    x=helper.make_tensor_value_info('input',F,[1,10,30,30]); y=helper.make_tensor_value_info('output',F,[1,10,30,30])
    I=[_K('c0',[0],np.int64),_K('c1',[1],np.int64),
       _K('c10000f',[10000.0],np.float32),_K('c0f',[0.0],np.float32),
       _K('shape1',[1],np.int64),_K('ax23',[2,3],np.int64),
       _K('depth10',[10],np.int64),_K('oh_vals',[0.0,1.0],np.float32),
       _K('shape130',[1,30,30],np.int64),
       _K('idx09',np.arange(9).reshape(1,9,1,1),np.int64),
       _K('half',[0.5],np.float32),
       _K('kernel_vert',[[[[0.0,1.0,0.0],[0.0,0.0,0.0],[0.0,1.0,0.0]]]],np.float32),
       _K('kernel_horiz',[[[[0.0,0.0,0.0],[1.0,0.0,1.0],[0.0,0.0,0.0]]]],np.float32)]
    n=[helper.make_node('ArgMax',['input'],['am'],axis=1,keepdims=1)]  # per-cell color (0-9), int64

    # x1 = leastcolor(I), excluding color 0 -- straight per-channel spatial sums
    n.append(helper.make_node('ReduceSum',['input','ax23'],['counts_all'],keepdims=1))  # [1,10,1,1]
    n.append(helper.make_node('Slice',['counts_all','c1','depth10','shape1'],['counts19']))  # [1,9,1,1], channels 1..9
    n.append(helper.make_node('Equal',['counts19','c0f'],['zero_b']))
    n.append(helper.make_node('Where',['zero_b','c10000f','counts19'],['adj']))
    n.append(helper.make_node('ArgMin',['adj'],['idx1'],axis=1,keepdims=1))  # [1,1,1,1], 0-based over channels 1..9
    n.append(helper.make_node('Add',['idx1','c1'],['x1']))  # actual color 1..9, shape [1,1,1,1]

    # x1 mask
    n.append(helper.make_node('Equal',['am','x1'],['x1_mask_b']))

    # x4 = leastcolor(x3) = leastcolor(I with x1's count zeroed out)
    n.append(helper.make_node('Equal',['idx09','idx1'],['is_idx1']))  # [1,9,1,1] broadcast
    n.append(helper.make_node('Where',['is_idx1','c10000f','adj'],['adj2']))
    n.append(helper.make_node('ArgMin',['adj2'],['idx2'],axis=1,keepdims=1))
    n.append(helper.make_node('Add',['idx2','c1'],['x4']))

    # x4 mask (x4 != x1, so x4's cells in x3 == x4's cells in I -- unaffected by removing x1)
    n.append(helper.make_node('Equal',['am','x4'],['x4_mask_b']))
    n.append(helper.make_node('Cast',['x4_mask_b'],['x4_mask_f'],to=F))

    # vertical (up/down) and horizontal (left/right) x4-neighbor presence via fixed-kernel Conv
    n.append(helper.make_node('Conv',['x4_mask_f','kernel_vert'],['vert_count'],pads=[1,1,1,1],kernel_shape=[3,3]))
    n.append(helper.make_node('Conv',['x4_mask_f','kernel_horiz'],['horiz_count'],pads=[1,1,1,1],kernel_shape=[3,3]))
    n.append(helper.make_node('Greater',['vert_count','half'],['vert_b']))
    n.append(helper.make_node('Greater',['horiz_count','half'],['horiz_b']))
    n.append(helper.make_node('And',['vert_b','horiz_b'],['both_b']))
    n.append(helper.make_node('And',['x1_mask_b','both_b'],['to_convert_b']))

    # final: x1 cells -> 0 unless to_convert_b -> x4; everything else unchanged
    n.append(helper.make_node('Where',['x1_mask_b','c0','am'],['tmp_am']))
    n.append(helper.make_node('Where',['to_convert_b','x4','tmp_am'],['final_am']))
    n.append(helper.make_node('Reshape',['final_am','shape130'],['pred_idx']))
    n.append(helper.make_node('OneHot',['pred_idx','depth10','oh_vals'],['output'],axis=1))
    return helper.make_model(helper.make_graph(n,'task192',[x],[y],I),ir_version=8,opset_imports=[helper.make_opsetid('',13)])

def _make():
    return _mask(build_192())

model = _bake(_make(), 192)
