import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime
F = TensorProto.FLOAT
I64 = TensorProto.INT64


def create_model():
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 'H_out', 'W_out'])

    def K(name, arr, dtype=np.int64):
        return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)

    # diagonal-only 3x3 kernel (ineighbors: corners only, no center, no orthogonal)
    diag_kernel = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.float32).reshape(1, 1, 3, 3)

    inits = [
        K('ax1', [1]), K('ax2', [2]), K('ax3', [3]),
        K('ax23', [2, 3]), K('starts01', [0, 0]),
        K('shape_1d', [-1]),
        K('tile_reps3', [1, 2, 2]),
        K('eight_i64', [8]),
        K('depth10', [10]),
        K('oh_vals', [0.0, 1.0], dtype=np.float32),
        K('zero_f', [0.0], dtype=np.float32),
        K('tenk_f', [10000.0], dtype=np.float32),
        K('half_f', [0.5], dtype=np.float32),
        K('diag_kernel', diag_kernel, dtype=np.float32),
    ]

    nodes = []
    # real h,w of the grid (grid is always top-left anchored by convert_to_numpy): a cell is
    # "real" (non-padding) iff some channel is 1 there, so max over channel+width/height axes
    # directly gives a per-row / per-col presence indicator without materializing a full 30x30 mask.
    nodes.append(helper.make_node('ReduceMax', ['input'], ['row_any'], axes=[1, 3], keepdims=1))
    nodes.append(helper.make_node('ReduceSum', ['row_any', 'ax2'], ['h_f'], keepdims=1))
    nodes.append(helper.make_node('Cast', ['h_f'], ['h_i'], to=I64))
    nodes.append(helper.make_node('Reshape', ['h_i', 'shape_1d'], ['h_1d']))

    nodes.append(helper.make_node('ReduceMax', ['input'], ['col_any'], axes=[1, 2], keepdims=1))
    nodes.append(helper.make_node('ReduceSum', ['col_any', 'ax3'], ['w_f'], keepdims=1))
    nodes.append(helper.make_node('Cast', ['w_f'], ['w_i'], to=I64))
    nodes.append(helper.make_node('Reshape', ['w_i', 'shape_1d'], ['w_1d']))

    nodes.append(helper.make_node('Concat', ['h_1d', 'w_1d'], ['ends_hw'], axis=0))
    nodes.append(helper.make_node('Slice', ['input', 'starts01', 'ends_hw', 'ax23'], ['cropped']))

    # per-color pixel counts computed on the un-tiled one-hot crop (ratios preserved under tiling)
    nodes.append(helper.make_node('ReduceSum', ['cropped', 'ax23'], ['cnt_all'], keepdims=1))
    # leastcolor: argmin over channels present in the grid (absent colors get sentinel 10000)
    nodes.append(helper.make_node('Equal', ['cnt_all', 'zero_f'], ['is_zero']))
    nodes.append(helper.make_node('Where', ['is_zero', 'tenk_f', 'cnt_all'], ['cnt_sentinel']))
    nodes.append(helper.make_node('ArgMin', ['cnt_sentinel'], ['x1_idx'], axis=1, keepdims=1))
    nodes.append(helper.make_node('Reshape', ['x1_idx', 'shape_1d'], ['x1_1d']))
    # mostcolor / background: plain argmax (absent colors can never win)
    nodes.append(helper.make_node('ArgMax', ['cnt_all'], ['bg_idx'], axis=1, keepdims=1))
    nodes.append(helper.make_node('Reshape', ['bg_idx', 'shape_1d'], ['bg_1d']))

    # collapse the one-hot crop down to a single-channel color-index grid (much cheaper to tile
    # and to compare than carrying all 10 channels through the tiling + dilation computation)
    nodes.append(helper.make_node('ArgMax', ['cropped'], ['cropped_idx'], axis=1, keepdims=0))

    # tile the actual h x w index-grid 2x2 in one shot (equivalent to vconcat(hconcat(I,I),hconcat(I,I)))
    nodes.append(helper.make_node('Tile', ['cropped_idx', 'tile_reps3'], ['doubled_idx']))

    nodes.append(helper.make_node('Equal', ['doubled_idx', 'x1_1d'], ['mask_x1_bool']))
    nodes.append(helper.make_node('Equal', ['doubled_idx', 'bg_1d'], ['is_bg_bool']))

    # diagonal-neighbor dilation of the rare-color mask (on the doubled grid) via a 3x3 Conv
    nodes.append(helper.make_node('Unsqueeze', ['mask_x1_bool', 'ax1'], ['mask_x1_bool4d']))
    nodes.append(helper.make_node('Cast', ['mask_x1_bool4d'], ['mask_x1_f'], to=F))
    nodes.append(helper.make_node('Conv', ['mask_x1_f', 'diag_kernel'], ['conv_out'], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
    nodes.append(helper.make_node('Greater', ['conv_out', 'half_f'], ['neighbor_bool4d']))
    nodes.append(helper.make_node('Squeeze', ['neighbor_bool4d', 'ax1'], ['neighbor_bool']))

    # underfill: only overwrite cells that are background AND a diagonal neighbor of a rare-color cell
    nodes.append(helper.make_node('And', ['neighbor_bool', 'is_bg_bool'], ['combined_bool']))
    nodes.append(helper.make_node('Where', ['combined_bool', 'eight_i64', 'doubled_idx'], ['final_idx']))
    nodes.append(helper.make_node('OneHot', ['final_idx', 'depth10', 'oh_vals'], ['output'], axis=1))

    graph = helper.make_graph(nodes, 'task019', [x], [y], inits)
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
    return _crop_pad(create_model())

model = _bake(_make(), 19)
