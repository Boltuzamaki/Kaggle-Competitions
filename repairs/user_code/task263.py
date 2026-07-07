import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime
F = TensorProto.FLOAT; I64 = TensorProto.INT64

# Task 263 DSL rule (verified byte-exact vs train+test+arc-gen in pure numpy first):
#   x1=numcolors(I) | x2=dmirror(I) | x3=portrait(I) | x4=branch(x3,dmirror,identity)
#   x5=x4(I) | x6=decrement(x1) | x7=hsplit(x5,x6) | x8=rbind(ofcolor,ZERO)
#   x9=apply(x8,x7) | x10=leastcommon(x9) | x11=matcher(x8,x10) | x12=extract(x7,x11) | O=x4(x12)
# i.e.: count distinct colors (incl. 0) -> n = numcolors-1 strips; orient grid to landscape
# (transpose iff portrait); split landscape grid into n equal-width strips; for each strip take
# its exact "which cells are color 0" pattern; pick the strip whose pattern is the LEAST COMMON
# (by exact-set-equality, not just count) among the n patterns; extract that strip; un-transpose
# if we transposed earlier. Verified 267/267 (train+test+arc-gen) always: landscape height==3,
# strip width==3 (so output is always 3x3), width always evenly divisible by n, and no ties ever
# occur among minimum-count patterns (so first-occurrence tie-break is never actually exercised,
# but the graph below reproduces it correctly regardless via ArgMin's first-index-on-ties default).


def create_model():
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])

    def K(name, arr, dtype=np.int64):
        return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)

    inits = [
        K('zerof', [0.0], np.float32),
        K('c0', [0]), K('c1', [1]), K('c2', [2]), K('c3', [3]), K('c4', [4]), K('c10', [10]),
        K('cneg1', [-1]), K('c00', [0, 0]), K('c23', [2, 3]), K('c3030', [30, 30]),
        K('pfx6', [0, 0, 0, 0, 0, 0]),
    ]
    n = []

    # ---- presence / bounding box (real grid content is always top-left anchored & contiguous) ----
    n.append(helper.make_node('ReduceMax', ['input'], ['presence'], axes=[1], keepdims=1))        # [1,1,30,30]
    n.append(helper.make_node('ReduceMax', ['presence'], ['row_any'], axes=[3], keepdims=1))       # [1,1,30,1]
    n.append(helper.make_node('ReduceMax', ['presence'], ['col_any'], axes=[2], keepdims=1))       # [1,1,1,30]
    n.append(helper.make_node('Greater', ['row_any', 'zerof'], ['row_b']))
    n.append(helper.make_node('Greater', ['col_any', 'zerof'], ['col_b']))
    n.append(helper.make_node('Cast', ['row_b'], ['row_i'], to=I64))
    n.append(helper.make_node('Cast', ['col_b'], ['col_i'], to=I64))
    n.append(helper.make_node('ReduceSum', ['row_i', 'c2'], ['H'], keepdims=1))                    # [1,1,1,1]
    n.append(helper.make_node('ReduceSum', ['col_i', 'c3'], ['W'], keepdims=1))                    # [1,1,1,1]

    # ---- numcolors = palette size, including color 0 ----
    n.append(helper.make_node('ReduceMax', ['input'], ['chan_present'], axes=[2, 3], keepdims=1))   # [1,10,1,1]
    n.append(helper.make_node('Greater', ['chan_present', 'zerof'], ['chan_b']))
    n.append(helper.make_node('Cast', ['chan_b'], ['chan_i'], to=I64))
    n.append(helper.make_node('ReduceSum', ['chan_i', 'c1'], ['numcolors'], keepdims=1))            # [1,1,1,1]
    n.append(helper.make_node('Sub', ['numcolors', 'c1'], ['n_strips']))                            # [1,1,1,1]

    # ---- portrait flag. Reorientation (transpose iff portrait) is done as an elementwise blend
    # so no If/subgraph is needed (subgraphs are banned). Rather than padding both orientation
    # candidates up to a common square shape (which wastes memory: e.g. a 3x15 grid would get
    # padded to 15x15), we exploit that a Reshape is valid as long as the total element count
    # matches: raw_crop [1,10,H,W] reshaped to [1,10,Hn,Wn] is a no-op when not portrait (H==Hn,
    # W==Wn) and correctly-transposed data's Reshape to [1,10,Hn,Wn] is a no-op when portrait
    # (Transpose(raw_crop) has actual shape [1,10,W,H]==[1,10,Hn,Wn] when portrait); the "other"
    # (wrong-branch) candidate's Reshape still succeeds (same H*W element count either way) but
    # produces throwaway/garbage data that Where simply discards. This keeps every intermediate
    # tensor sized to the REAL H*W (never padded to max(H,W)^2), and -- crucially -- also avoids
    # an ORT memory-planner bug where feeding a Transpose node directly from a Pad node's output
    # crashes with "Shape mismatch attempting to re-use buffer" on every dynamic-shape example
    # (verified empirically); Transpose here is fed straight from a Slice, which is safe.
    n.append(helper.make_node('Greater', ['H', 'W'], ['portrait']))                                 # [1,1,1,1] bool
    n.append(helper.make_node('Where', ['portrait', 'W', 'H'], ['Hn']))                             # [1,1,1,1]
    n.append(helper.make_node('Where', ['portrait', 'H', 'W'], ['Wn']))                             # [1,1,1,1]

    n.append(helper.make_node('Reshape', ['H', 'cneg1'], ['H_1d']))
    n.append(helper.make_node('Reshape', ['W', 'cneg1'], ['W_1d']))
    n.append(helper.make_node('Reshape', ['Hn', 'cneg1'], ['Hn_1d']))                               # [1]
    n.append(helper.make_node('Reshape', ['Wn', 'cneg1'], ['Wn_1d']))                               # [1]
    n.append(helper.make_node('Reshape', ['n_strips', 'cneg1'], ['n_1d']))                          # [1]

    n.append(helper.make_node('Concat', ['H_1d', 'W_1d'], ['raw_ends'], axis=0))
    n.append(helper.make_node('Slice', ['input', 'c00', 'raw_ends', 'c23'], ['raw_crop']))          # [1,10,H,W] SMALL
    n.append(helper.make_node('Transpose', ['raw_crop'], ['raw_crop_T'], perm=[0, 1, 3, 2]))        # [1,10,W,H]

    n.append(helper.make_node('Concat', ['c1', 'c10', 'Hn_1d', 'Wn_1d'], ['shapeHW'], axis=0))       # [1,10,Hn,Wn]
    n.append(helper.make_node('Reshape', ['raw_crop', 'shapeHW'], ['candA']))
    n.append(helper.make_node('Reshape', ['raw_crop_T', 'shapeHW'], ['candB']))
    n.append(helper.make_node('Where', ['portrait', 'candB', 'candA'], ['cropped']))                 # [1,10,Hn,Wn]

    # ---- split landscape grid into n strips: reshape+permute so axis0 = strip index ----
    n.append(helper.make_node('Concat', ['c1', 'c10', 'Hn_1d', 'n_1d', 'cneg1'], ['shape5'], axis=0))
    n.append(helper.make_node('Reshape', ['cropped', 'shape5'], ['cropped5']))                      # [1,10,Hn,n,stripW]
    n.append(helper.make_node('Transpose', ['cropped5'], ['permuted'], perm=[3, 0, 1, 2, 4]))        # [n,1,10,Hn,stripW]

    # ---- per-strip "which cells are color 0" pattern (channel 0 of the one-hot) ----
    n.append(helper.make_node('Slice', ['permuted', 'c0', 'c1', 'c2'], ['zero_channel']))            # [n,1,1,Hn,stripW]
    n.append(helper.make_node('Concat', ['n_1d', 'cneg1'], ['flat_shape'], axis=0))                  # [n,-1]
    n.append(helper.make_node('Reshape', ['zero_channel', 'flat_shape'], ['flat_masks']))            # [n,HW]

    # ---- find the strip whose pattern is the LEAST COMMON (exact match count, then first-index) ----
    n.append(helper.make_node('Concat', ['n_1d', 'c1', 'cneg1'], ['shapeA'], axis=0))
    n.append(helper.make_node('Reshape', ['flat_masks', 'shapeA'], ['A']))                           # [n,1,HW]
    n.append(helper.make_node('Concat', ['c1', 'n_1d', 'cneg1'], ['shapeB'], axis=0))
    n.append(helper.make_node('Reshape', ['flat_masks', 'shapeB'], ['B']))                           # [1,n,HW]
    n.append(helper.make_node('Equal', ['A', 'B'], ['eq']))                                          # [n,n,HW]
    n.append(helper.make_node('Cast', ['eq'], ['eq_i'], to=I64))
    n.append(helper.make_node('ReduceMin', ['eq_i'], ['match_all'], axes=[2], keepdims=0))            # [n,n] (AND across HW)
    n.append(helper.make_node('ReduceSum', ['match_all', 'c1'], ['counts'], keepdims=0))              # [n]
    n.append(helper.make_node('ArgMin', ['counts'], ['winner_idx'], axis=0, keepdims=0))               # first min-count idx
    n.append(helper.make_node('Reshape', ['winner_idx', 'cneg1'], ['winner_idx_1d']))                  # [1]

    # ---- extract winning strip (full one-hot content) ----
    n.append(helper.make_node('Gather', ['permuted', 'winner_idx_1d'], ['gathered'], axis=0))         # [1,1,10,Hn,stripW]
    n.append(helper.make_node('Concat', ['c1', 'c10', 'Hn_1d', 'cneg1'], ['shape4'], axis=0))
    n.append(helper.make_node('Reshape', ['gathered', 'shape4'], ['winner_full']))                     # [1,10,Hn,stripW]

    # ---- un-transpose the (tiny) extracted strip via the same Reshape-blend trick (no padding,
    # no Pad->Transpose adjacency) ----
    n.append(helper.make_node('Shape', ['winner_full'], ['osh']))
    n.append(helper.make_node('Slice', ['osh', 'c3', 'c4', 'c0'], ['stripW_1d']))                    # [1]
    n.append(helper.make_node('Transpose', ['winner_full'], ['winner_full_T'], perm=[0, 1, 3, 2]))    # [1,10,stripW,Hn]

    n.append(helper.make_node('Reshape', ['portrait', 'cneg1'], ['portrait_1d']))                   # [1] bool
    n.append(helper.make_node('Where', ['portrait_1d', 'stripW_1d', 'Hn_1d'], ['outH_1d']))
    n.append(helper.make_node('Where', ['portrait_1d', 'Hn_1d', 'stripW_1d'], ['outW_1d']))
    n.append(helper.make_node('Concat', ['c1', 'c10', 'outH_1d', 'outW_1d'], ['shapeOut'], axis=0))  # [1,10,outH,outW]
    n.append(helper.make_node('Reshape', ['winner_full', 'shapeOut'], ['candA2']))
    n.append(helper.make_node('Reshape', ['winner_full_T', 'shapeOut'], ['candB2']))
    n.append(helper.make_node('Where', ['portrait', 'candB2', 'candA2'], ['final_small']))            # [1,10,outH,outW] tiny

    # ---- pad the tiny final answer up to the fixed 30x30 'output' (excluded from memory cost) ----
    n.append(helper.make_node('Shape', ['final_small'], ['osh2']))
    n.append(helper.make_node('Slice', ['osh2', 'c2', 'c4', 'c0'], ['hw2']))
    n.append(helper.make_node('Sub', ['c3030', 'hw2'], ['padhw3']))
    n.append(helper.make_node('Concat', ['pfx6', 'padhw3'], ['pads3'], axis=0))
    n.append(helper.make_node('Pad', ['final_small', 'pads3', 'zerof'], ['output'], mode='constant'))

    graph = helper.make_graph(n, 'task263', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])


# ===== Owned repair scaffolding (Claude, verified vs train+test+arc-gen) =====
import os as _os, copy as _copy
def _K(n,a,d): return numpy_helper.from_array(np.array(a,dtype=d),name=n)
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

model = _bake(_make(), 263)
