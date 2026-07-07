import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

F = TensorProto.FLOAT
I64 = TensorProto.INT64


def K(name, arr, dtype=np.int64):
    return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)


def create_model():
    """task148 (arc_dsl_ref solve_673ef223 intent, corrected):
    - Two vertical lines of color 2 always sit at real columns 0 and (real_width-1).
    - Color-8 marker cells each shoot a ray of color 8 along their row, in the
      direction of whichever of the two color-2 lines is topmost (LEFT if that
      line is the one at column 0, RIGHT otherwise) - filling only background
      cells - and the marker cell itself becomes color 4.
    - Separately, every marker's row is duplicated `gap` rows down (gap = the
      absolute difference between the two color-2 lines' top rows) and the
      WHOLE row there is filled with color 8 wherever it is background.
    (The literal DSL solver picks the direction from the topmost object over
    ALL objects/colors, which has an unresolved tie whenever an 8-marker is as
    topmost as the reference 2-line; using only the two color-2 lines for that
    decision is what actually matches every train+test+arc-gen example.)
    """
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])

    inits = [
        K('half', [0.5], np.float32),
        K('row_idx', np.arange(30).reshape(1, 1, 30, 1), np.int64),
        K('col_idx', np.arange(30).reshape(1, 1, 1, 30), np.int64),
        K('sent_hi', [999]),
        K('sent_lo', [-1]),
        K('one_f', [1.0], np.float32),
        K('s0', [0]), K('e0', [1]), K('ax3', [3]),
    ]
    n = []

    ch_names = [f'ch{k}' for k in range(10)]
    n.append(helper.make_node('Split', ['input'], ch_names, axis=1))

    # presence bools for color2 / color8 channels
    n.append(helper.make_node('Greater', ['ch2', 'half'], ['pres2']))   # [1,1,30,30]
    n.append(helper.make_node('Greater', ['ch8', 'half'], ['pres8']))   # [1,1,30,30]

    # ---- uppermost row of the color-2 line at column 0 (u0) ----
    n.append(helper.make_node('Slice', ['pres2', 's0', 'e0', 'ax3'], ['pres2_col0']))
    n.append(helper.make_node('Where', ['pres2_col0', 'row_idx', 'sent_hi'], ['col0_src']))
    n.append(helper.make_node('ReduceMin', ['col0_src'], ['u0'], axes=[2], keepdims=1))  # [1,1,1,1]

    # ---- uppermost row per column, over all columns ----
    n.append(helper.make_node('Where', ['pres2', 'row_idx', 'sent_hi'], ['col_src_hi']))
    n.append(helper.make_node('ReduceMin', ['col_src_hi'], ['upcol_hi'], axes=[2], keepdims=1))  # [1,1,1,30], 999 if col has no 2

    n.append(helper.make_node('ReduceMax', ['ch2'], ['active_col_f'], axes=[2], keepdims=1))  # [1,1,1,30] float
    n.append(helper.make_node('Greater', ['active_col_f', 'half'], ['active_col_b']))
    n.append(helper.make_node('Where', ['active_col_b', 'upcol_hi', 'sent_lo'], ['upcol_for_max']))  # inactive cols -> -1

    n.append(helper.make_node('ReduceMin', ['upcol_hi'], ['u_min'], axes=[3], keepdims=1)) # min uppermost among the 2 color-2 lines
    n.append(helper.make_node('ReduceMax', ['upcol_for_max'], ['u_max'], axes=[3], keepdims=1)) # max uppermost among the 2 color-2 lines

    n.append(helper.make_node('Sub', ['u_max', 'u_min'], ['gap']))          # vertical shift amount
    n.append(helper.make_node('Add', ['u_min', 'u_max'], ['sum_mm']))
    n.append(helper.make_node('Sub', ['sum_mm', 'u0'], ['u_other']))        # uppermost of the OTHER (non-col0) line

    n.append(helper.make_node('Less', ['u0', 'u_other'], ['is_left']))      # col0 line is topmost -> shoot LEFT

    # ---- per-row min/max marker column ----
    n.append(helper.make_node('Where', ['pres8', 'col_idx', 'sent_lo'], ['mrk_src_max']))
    n.append(helper.make_node('ReduceMax', ['mrk_src_max'], ['c_max_row'], axes=[3], keepdims=1))  # [1,1,30,1]
    n.append(helper.make_node('Where', ['pres8', 'col_idx', 'sent_hi'], ['mrk_src_min']))
    n.append(helper.make_node('ReduceMin', ['mrk_src_min'], ['c_min_row'], axes=[3], keepdims=1))  # [1,1,30,1]

    n.append(helper.make_node('Less', ['col_idx', 'c_max_row'], ['cond_left']))      # ray columns if shooting LEFT
    n.append(helper.make_node('Greater', ['col_idx', 'c_min_row'], ['cond_right']))  # ray columns if shooting RIGHT
    n.append(helper.make_node('Cast', ['cond_left'], ['cond_left_f'], to=F))
    n.append(helper.make_node('Cast', ['cond_right'], ['cond_right_f'], to=F))
    n.append(helper.make_node('Where', ['is_left', 'cond_left_f', 'cond_right_f'], ['cond_ray_f']))
    n.append(helper.make_node('Greater', ['cond_ray_f', 'half'], ['cond_ray']))

    n.append(helper.make_node('Greater', ['ch0', 'half'], ['is_bg']))       # background = original color-0 cells
    n.append(helper.make_node('And', ['cond_ray', 'is_bg'], ['ray_mask_b']))

    # ---- frontier: duplicate each marker's row, shifted down by `gap` ----
    n.append(helper.make_node('ReduceMax', ['ch8'], ['has_marker_row_f'], axes=[3], keepdims=1))  # [1,1,30,1]
    n.append(helper.make_node('Greater', ['has_marker_row_f', 'half'], ['has_marker_row_b']))
    n.append(helper.make_node('Transpose', ['has_marker_row_b'], ['has_marker_row3_b'], perm=[0, 1, 3, 2]))  # [1,1,1,30]

    n.append(helper.make_node('Sub', ['row_idx', 'col_idx'], ['diff']))    # diff[r',r] = r' - r  (row_idx along ax2, col_idx along ax3)
    n.append(helper.make_node('Equal', ['diff', 'gap'], ['match_b']))      # r' == r + gap

    n.append(helper.make_node('And', ['match_b', 'has_marker_row3_b'], ['contrib_b']))
    n.append(helper.make_node('Cast', ['contrib_b'], ['contrib_f'], to=F))
    n.append(helper.make_node('ReduceMax', ['contrib_f'], ['shifted_ind_f'], axes=[3], keepdims=1))  # [1,1,30,1]
    n.append(helper.make_node('Greater', ['shifted_ind_f', 'half'], ['shifted_ind_b']))

    n.append(helper.make_node('And', ['shifted_ind_b', 'is_bg'], ['frontier_mask_b']))

    n.append(helper.make_node('Or', ['ray_mask_b', 'frontier_mask_b'], ['become8_b']))
    n.append(helper.make_node('Cast', ['become8_b'], ['become8_f'], to=F))

    n.append(helper.make_node('Sub', ['one_f', 'become8_f'], ['not_become8']))
    n.append(helper.make_node('Mul', ['ch0', 'not_become8'], ['new_ch0']))     # bg cells that turned into 8 lose bg flag

    n.append(helper.make_node('Add', ['ch4', 'ch8'], ['new_ch4']))            # original 8-markers recolor to 4

    n.append(helper.make_node('Identity', ['become8_f'], ['new_ch8']))

    out_parts = ['new_ch0', 'ch1', 'ch2', 'ch3', 'new_ch4', 'ch5', 'ch6', 'ch7', 'new_ch8', 'ch9']
    n.append(helper.make_node('Concat', out_parts, ['output'], axis=1))

    graph = helper.make_graph(n, 'task148', [x], [y], inits)
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
    return _mask(create_model())

model = _bake(_make(), 148)
