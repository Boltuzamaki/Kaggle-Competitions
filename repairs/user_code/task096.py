import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime
F = TensorProto.FLOAT; I64 = TensorProto.INT64; BOOLT = TensorProto.BOOL

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
    """OneHot 'output' is a dynamic [1,10,h,w] canvas at top-left; Pad to static 30x30
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

# ===== task096: color-group mosaic (solve_4290ef0e intent) =====
#
# Rule (numpy-verified against ALL train+test+arc-gen, n_fail=0):
#  1. background = most common color in the grid.
#  2. Each other color present forms a "group" = ALL cells of that color (regardless
#     of connectivity). Order the groups descending by:
#         x16 = spacing_term + 2*max_component_width
#     where max_component_width = the largest (max_col-min_col+1) among that color's
#     4-connected components, and spacing_term = (min pairwise Manhattan distance
#     between DIFFERENT same-color components) - 1, or -2 if the color forms a single
#     component.
#  3. For each group (in that order, rank 0,1,2,...): crop to its own bounding box,
#     then among {vmirror, hmirror, dmirror(transpose), cmirror(anti-transpose)} pick
#     the variant whose occupied cells appear earliest in row-major reading order
#     (equivalently: lexicographically largest when read as a dense 0/1 grid).
#  4. Let x26 = number of groups, x28 = whether any group has exactly 1 cell.
#     x30 = x26 if x28 else x26+1.  Canvas size = 2*x30-1, filled with background.
#  5. Place each chosen (already-cropped) shape's top-left corner at (rank,rank) on
#     the canvas (paint, overwriting on overlap in rank order).
#  6. Rotate canvas 90 deg clockwise and re-paint the SAME rank-shapes on top, 3 times
#     total (4 paints, 3 rotations) to build a rotationally-symmetric mosaic.
#
# ONNX notes: connected components use Jacobi label-propagation (min-index over
# 4-neighbors of matching color), matching task042's established pattern. Mirror
# selection uses a "first differing position wins" pairwise tournament over the 4
# candidates (each padded to a fixed 30x30 top-left canvas so comparisons are static
# shape). All group/rank loops are statically unrolled (10 colors, 10 ranks) -- no
# Loop/Scan/NonZero used anywhere.

N = 30
ITERS = 12          # cc-label propagation rounds (max observed component span across
                     # the whole dataset is 4; this has generous margin)
BIG = 10_000_000
NCOL = 10
RMAX = 9            # rank slots 0..9 (max observed present-colors is 6)

def build_096():
    I = []
    def add(name, arr, dtype=np.int64):
        I.append(_K(name, np.array(arr, dtype=dtype), dtype))
    n = []

    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 'H', 'W'])

    add('zero_i64', 0); add('one_i64', 1); add('two_i64', 2); add('neg1_i64', -1); add('neg2_i64', -2)
    add('big_i64', BIG); add('negbig_i64', -BIG)
    add('zero_f', 0.0, np.float32); add('half_f', 0.5, np.float32)
    add('shape900', [900]); add('shape1', [1])
    add('shape900x1', [900, 1]); add('shape1x900', [1, 900])
    add('axes01', [0, 1]); add('ax0', [0]); add('ax1', [1])
    add('m1', -1); add('p999', 999)
    add('axes01_arr', [0, 1])
    add('negend', -9223372036854775807)
    add('m1_1d', [-1])
    add('negend_1d', [-9223372036854775807])
    add('zeros2', [0, 0])
    add('shape1x1', [1, 1])
    add('shape10x1', [10, 1]); add('shape1x10', [1, 10])
    add('thirty', 30)
    add('one_1', [1])
    add('depth10', [10]); add('oh_vals', [0.0, 1.0], np.float32)

    idx900 = np.arange(900, dtype=np.int64)
    add('idx900', idx900)
    idx900_2d = idx900.reshape(30, 30)
    add('idx900_2d', idx900_2d)
    R_flat = (idx900 // 30); C_flat = (idx900 % 30)
    add('R_flat_row', R_flat.reshape(1, 900)); add('C_flat_row', C_flat.reshape(1, 900))
    add('R_flat_col', R_flat.reshape(900, 1)); add('C_flat_col', C_flat.reshape(900, 1))
    add('row_idx_col', np.arange(30, dtype=np.int64).reshape(30, 1))
    add('col_idx_row', np.arange(30, dtype=np.int64).reshape(1, 30))

    # ---- per-channel extraction + counts ----
    count_names = []
    chan_names = []
    for c in range(NCOL):
        add(f'cidx_{c}', c)
        n.append(helper.make_node('Gather', ['input', f'cidx_{c}'], [f'chan3_{c}'], axis=1))
        n.append(helper.make_node('Squeeze', [f'chan3_{c}', 'ax0'], [f'chan_{c}']))
        chan_names.append(f'chan_{c}')
        n.append(helper.make_node('ReduceSum', [f'chan_{c}', 'axes01_arr'], [f'count_{c}'], keepdims=0))
        count_names.append(f'count_{c}')

    r1_names = []
    for c in range(NCOL):
        n.append(helper.make_node('Reshape', [f'count_{c}', 'shape1'], [f'count1_{c}']))
        r1_names.append(f'count1_{c}')
    n.append(helper.make_node('Concat', r1_names, ['counts_vec'], axis=0))
    n.append(helper.make_node('ArgMax', ['counts_vec'], ['bg_idx'], axis=0, keepdims=0))

    # ---- col2d / presence ----
    n.append(helper.make_node('ArgMax', ['input'], ['am_'], axis=1, keepdims=1))
    n.append(helper.make_node('Squeeze', ['am_', 'axes01'], ['col2d']))
    n.append(helper.make_node('ReduceMax', ['input'], ['pres_'], axes=[1], keepdims=1))
    n.append(helper.make_node('Squeeze', ['pres_', 'axes01'], ['presence2d']))
    n.append(helper.make_node('Greater', ['presence2d', 'half_f'], ['presence_bool']))

    n.append(helper.make_node('Equal', ['col2d', 'bg_idx'], ['is_bg2d']))
    n.append(helper.make_node('Not', ['is_bg2d'], ['not_bg2d']))
    n.append(helper.make_node('And', ['presence_bool', 'not_bg2d'], ['real_nonbg']))
    n.append(helper.make_node('Where', ['real_nonbg', 'col2d', 'neg1_i64'], ['eff_col']))

    # ---- connected components (4-conn) via Jacobi propagation ----
    n.append(helper.make_node('Where', ['real_nonbg', 'idx900_2d', 'big_i64'], ['label_0']))
    offs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for d, (di, dj) in enumerate(offs):
        t = max(-di, 0); b = max(di, 0); l = max(-dj, 0); r = max(dj, 0)
        add(f'pads_{d}', [t, l, b, r])
        add(f'starts_{d}', [max(di, 0), max(dj, 0)])
        add(f'ends_{d}', [max(di, 0) + 30, max(dj, 0) + 30])
        n.append(helper.make_node('Pad', ['eff_col', f'pads_{d}', 'neg1_i64'], [f'pad_eff_{d}'], mode='constant'))
        n.append(helper.make_node('Slice', [f'pad_eff_{d}', f'starts_{d}', f'ends_{d}', 'axes01'], [f'shift_eff_{d}']))
        n.append(helper.make_node('Equal', [f'shift_eff_{d}', 'eff_col'], [f'eq_eff_{d}']))
        n.append(helper.make_node('And', [f'eq_eff_{d}', 'real_nonbg'], [f'match_{d}']))

    cur = 'label_0'
    for it in range(ITERS):
        cands = [cur]
        for d, (di, dj) in enumerate(offs):
            n.append(helper.make_node('Pad', [cur, f'pads_{d}', 'big_i64'], [f'padl_{it}_{d}'], mode='constant'))
            n.append(helper.make_node('Slice', [f'padl_{it}_{d}', f'starts_{d}', f'ends_{d}', 'axes01'], [f'shl_{it}_{d}']))
            n.append(helper.make_node('Where', [f'match_{d}', f'shl_{it}_{d}', 'big_i64'], [f'cand_{it}_{d}']))
            cands.append(f'cand_{it}_{d}')
        nxt = f'label_{it+1}'
        n.append(helper.make_node('Min', cands, [nxt]))
        cur = nxt
    label_final = cur

    # ---- flatten + pairwise matrices ----
    n.append(helper.make_node('Reshape', [label_final, 'shape900'], ['label_flat']))
    n.append(helper.make_node('Reshape', ['real_nonbg', 'shape900'], ['real_flat']))
    n.append(helper.make_node('Reshape', ['col2d', 'shape900'], ['col_flat']))

    n.append(helper.make_node('Reshape', ['label_flat', 'shape900x1'], ['label_col']))
    n.append(helper.make_node('Reshape', ['label_flat', 'shape1x900'], ['label_row']))
    n.append(helper.make_node('Reshape', ['real_flat', 'shape900x1'], ['real_col']))
    n.append(helper.make_node('Reshape', ['real_flat', 'shape1x900'], ['real_row']))
    n.append(helper.make_node('Reshape', ['col_flat', 'shape900x1'], ['col_col']))
    n.append(helper.make_node('Reshape', ['col_flat', 'shape1x900'], ['col_row']))

    n.append(helper.make_node('Equal', ['label_col', 'label_row'], ['same_label']))
    n.append(helper.make_node('And', ['same_label', 'real_col'], ['same_a']))
    n.append(helper.make_node('And', ['same_a', 'real_row'], ['same_full']))

    n.append(helper.make_node('Not', ['same_label'], ['diff_label']))
    n.append(helper.make_node('Equal', ['col_col', 'col_row'], ['same_color']))
    n.append(helper.make_node('And', ['real_col', 'real_row'], ['both_real']))
    n.append(helper.make_node('And', ['diff_label', 'same_color'], ['dc_a']))
    n.append(helper.make_node('And', ['dc_a', 'both_real'], ['diffcomp_samecolor']))

    n.append(helper.make_node('Sub', ['R_flat_col', 'R_flat_row'], ['dR']))
    n.append(helper.make_node('Sub', ['C_flat_col', 'C_flat_row'], ['dC']))
    n.append(helper.make_node('Abs', ['dR'], ['adR']))
    n.append(helper.make_node('Abs', ['dC'], ['adC']))
    n.append(helper.make_node('Add', ['adR', 'adC'], ['dist_matrix']))

    # ---- per-cell own component bbox ----
    n.append(helper.make_node('Where', ['same_full', 'R_flat_row', 'big_i64'], ['Rq_min']))
    n.append(helper.make_node('ReduceMin', ['Rq_min'], ['r0_flat'], axes=[1], keepdims=0))
    n.append(helper.make_node('Where', ['same_full', 'R_flat_row', 'negbig_i64'], ['Rq_max']))
    n.append(helper.make_node('ReduceMax', ['Rq_max'], ['r1_flat'], axes=[1], keepdims=0))
    n.append(helper.make_node('Where', ['same_full', 'C_flat_row', 'big_i64'], ['Cq_min']))
    n.append(helper.make_node('ReduceMin', ['Cq_min'], ['c0_flat'], axes=[1], keepdims=0))
    n.append(helper.make_node('Where', ['same_full', 'C_flat_row', 'negbig_i64'], ['Cq_max']))
    n.append(helper.make_node('ReduceMax', ['Cq_max'], ['c1_flat'], axes=[1], keepdims=0))
    n.append(helper.make_node('Sub', ['c1_flat', 'c0_flat'], ['w_m1_flat']))
    n.append(helper.make_node('Add', ['w_m1_flat', 'one_i64'], ['width_percell']))

    # ---- per color v: present, size, width_max, spacing, x16 ----
    present_names = []
    x16eff_names = []
    for v in range(NCOL):
        add(f'vconst_{v}', v)
        n.append(helper.make_node('Equal', [f'vconst_{v}', 'bg_idx'], [f'isbg_{v}']))
        n.append(helper.make_node('Not', [f'isbg_{v}'], [f'notbg_{v}']))
        n.append(helper.make_node('Greater', [f'count_{v}', 'zero_f'], [f'cntpos_{v}']))
        n.append(helper.make_node('And', [f'cntpos_{v}', f'notbg_{v}'], [f'present_{v}']))
        present_names.append(f'present_{v}')

        n.append(helper.make_node('Cast', [f'count_{v}'], [f'size_i_{v}'], to=I64))
        n.append(helper.make_node('Equal', [f'size_i_{v}', 'one_i64'], [f'issingle_{v}']))

        n.append(helper.make_node('Equal', ['col_flat', f'vconst_{v}'], [f'iscolor_flat_{v}']))
        n.append(helper.make_node('And', [f'iscolor_flat_{v}', 'real_flat'], [f'colormask_flat_{v}']))
        n.append(helper.make_node('Where', [f'colormask_flat_{v}', 'width_percell', 'negbig_i64'], [f'maskedwidth_{v}']))
        n.append(helper.make_node('ReduceMax', [f'maskedwidth_{v}'], [f'widthmax_{v}'], axes=[0], keepdims=0))

        n.append(helper.make_node('Equal', ['col_col', f'vconst_{v}'], [f'colcol_eq_{v}']))
        n.append(helper.make_node('And', ['diffcomp_samecolor', f'colcol_eq_{v}'], [f'pairmask_{v}']))
        n.append(helper.make_node('Where', [f'pairmask_{v}', 'dist_matrix', 'big_i64'], [f'maskeddist_{v}']))
        n.append(helper.make_node('ReduceMin', [f'maskeddist_{v}'], [f'mindist_{v}'], keepdims=0))
        n.append(helper.make_node('Less', [f'mindist_{v}', 'big_i64'], [f'hasother_{v}']))
        n.append(helper.make_node('Sub', [f'mindist_{v}', 'one_i64'], [f'spacing_alt_{v}']))
        n.append(helper.make_node('Where', [f'hasother_{v}', f'spacing_alt_{v}', 'neg2_i64'], [f'spacing_{v}']))

        n.append(helper.make_node('Mul', [f'widthmax_{v}', 'two_i64'], [f'widthmax2_{v}']))
        n.append(helper.make_node('Add', [f'spacing_{v}', f'widthmax2_{v}'], [f'x16_{v}']))
        n.append(helper.make_node('Where', [f'present_{v}', f'x16_{v}', 'negbig_i64'], [f'x16eff_{v}']))
        x16eff_names.append(f'x16eff_{v}')

    # ---- x26, x28, x30, x32 ----
    for v in range(NCOL):
        n.append(helper.make_node('Cast', [f'present_{v}'], [f'present_i_{v}'], to=I64))
        n.append(helper.make_node('Reshape', [f'present_i_{v}', 'shape1'], [f'present_i1_{v}']))
        n.append(helper.make_node('And', [f'present_{v}', f'issingle_{v}'], [f'singlepresent_{v}']))
        n.append(helper.make_node('Cast', [f'singlepresent_{v}'], [f'singlepresent_i_{v}'], to=I64))
        n.append(helper.make_node('Reshape', [f'singlepresent_i_{v}', 'shape1'], [f'singlepresent_i1_{v}']))
    n.append(helper.make_node('Concat', [f'present_i1_{v}' for v in range(NCOL)], ['present_vec'], axis=0))
    n.append(helper.make_node('ReduceSum', ['present_vec', 'ax0'], ['x26'], keepdims=0))
    n.append(helper.make_node('Concat', [f'singlepresent_i1_{v}' for v in range(NCOL)], ['singlepresent_vec'], axis=0))
    n.append(helper.make_node('ReduceSum', ['singlepresent_vec', 'ax0'], ['x28count'], keepdims=0))
    n.append(helper.make_node('Greater', ['x28count', 'zero_i64'], ['x28']))
    n.append(helper.make_node('Add', ['x26', 'one_i64'], ['x29']))
    n.append(helper.make_node('Where', ['x28', 'x26', 'x29'], ['x30']))
    n.append(helper.make_node('Mul', ['x30', 'two_i64'], ['x31']))
    n.append(helper.make_node('Sub', ['x31', 'one_i64'], ['x32']))

    # ---- rank_v (descending by x16eff among present colors) ----
    for v in range(NCOL):
        n.append(helper.make_node('Reshape', [f'x16eff_{v}', 'shape1'], [f'x16eff1_{v}']))
    n.append(helper.make_node('Concat', [f'x16eff1_{v}' for v in range(NCOL)], ['x16vec'], axis=0))
    n.append(helper.make_node('Reshape', ['x16vec', 'shape10x1'], ['x16_col']))
    n.append(helper.make_node('Reshape', ['x16vec', 'shape1x10'], ['x16_row']))
    n.append(helper.make_node('Reshape', ['present_vec', 'shape1x10'], ['present_row_i']))
    n.append(helper.make_node('Greater', ['x16_row', 'x16_col'], ['gt_mat']))
    n.append(helper.make_node('Cast', ['gt_mat'], ['gt_mat_i'], to=I64))
    n.append(helper.make_node('Mul', ['gt_mat_i', 'present_row_i'], ['gated_mat']))
    n.append(helper.make_node('ReduceSum', ['gated_mat', 'ax1'], ['rank_vec'], keepdims=0))
    for v in range(NCOL):
        n.append(helper.make_node('Gather', ['rank_vec', f'vconst_{v}'], [f'rank_{v}'], axis=0))

    # ---- per color v: crop own bbox, pick best mirror variant, position on canvas ----
    def compare_first_diff(n, Aname, Bname, tag):
        """bool scalar: True if A has a 1 at the first row-major position where A,B differ
        (i.e. A's occupied cells start earlier in reading order -> A preferred)."""
        n.append(helper.make_node('Reshape', [Aname, 'shape900'], [f'{tag}_Af']))
        n.append(helper.make_node('Reshape', [Bname, 'shape900'], [f'{tag}_Bf']))
        n.append(helper.make_node('Equal', [f'{tag}_Af', f'{tag}_Bf'], [f'{tag}_eq']))
        n.append(helper.make_node('Not', [f'{tag}_eq'], [f'{tag}_diff']))
        n.append(helper.make_node('Where', [f'{tag}_diff', 'idx900', 'big_i64'], [f'{tag}_idxm']))
        n.append(helper.make_node('ReduceMin', [f'{tag}_idxm'], [f'{tag}_firstidx'], axes=[0], keepdims=0))
        n.append(helper.make_node('Equal', ['idx900', f'{tag}_firstidx'], [f'{tag}_ind']))
        n.append(helper.make_node('Where', [f'{tag}_ind', f'{tag}_Af', 'zero_f'], [f'{tag}_Asel']))
        n.append(helper.make_node('ReduceSum', [f'{tag}_Asel', 'ax0'], [f'{tag}_Aval'], keepdims=0))
        n.append(helper.make_node('Greater', [f'{tag}_Aval', 'half_f'], [f'{tag}_Awins']))
        return f'{tag}_Awins'

    n.append(helper.make_node('Reshape', ['x32', 'shape1'], ['x32_1']))

    maskbool_names = []
    valuecontrib_names = []
    for v in range(NCOL):
        chan_v = chan_names[v]
        n.append(helper.make_node('ReduceMax', [chan_v], [f'rowany_{v}'], axes=[1], keepdims=1))
        n.append(helper.make_node('Greater', [f'rowany_{v}', 'half_f'], [f'rowanyb_{v}']))
        n.append(helper.make_node('Where', [f'rowanyb_{v}', 'row_idx_col', 'm1'], [f'rowpmax_{v}']))
        n.append(helper.make_node('ReduceMax', [f'rowpmax_{v}'], [f'r1v_raw_{v}'], axes=[0], keepdims=0))
        n.append(helper.make_node('Where', [f'rowanyb_{v}', 'row_idx_col', 'p999'], [f'rowpmin_{v}']))
        n.append(helper.make_node('ReduceMin', [f'rowpmin_{v}'], [f'r0v_raw_{v}'], axes=[0], keepdims=0))
        n.append(helper.make_node('ReduceMax', [chan_v], [f'colany_{v}'], axes=[0], keepdims=1))
        n.append(helper.make_node('Greater', [f'colany_{v}', 'half_f'], [f'colanyb_{v}']))
        n.append(helper.make_node('Where', [f'colanyb_{v}', 'col_idx_row', 'm1'], [f'colpmax_{v}']))
        n.append(helper.make_node('ReduceMax', [f'colpmax_{v}'], [f'c1v_raw_{v}'], axes=[1], keepdims=0))
        n.append(helper.make_node('Where', [f'colanyb_{v}', 'col_idx_row', 'p999'], [f'colpmin_{v}']))
        n.append(helper.make_node('ReduceMin', [f'colpmin_{v}'], [f'c0v_raw_{v}'], axes=[1], keepdims=0))

        n.append(helper.make_node('Where', [f'present_{v}', f'r0v_raw_{v}', 'zero_i64'], [f'r0v_{v}']))
        n.append(helper.make_node('Where', [f'present_{v}', f'r1v_raw_{v}', 'zero_i64'], [f'r1v_{v}']))
        n.append(helper.make_node('Where', [f'present_{v}', f'c0v_raw_{v}', 'zero_i64'], [f'c0v_{v}']))
        n.append(helper.make_node('Where', [f'present_{v}', f'c1v_raw_{v}', 'zero_i64'], [f'c1v_{v}']))

        n.append(helper.make_node('Add', [f'r1v_{v}', 'one_i64'], [f'r1p1_{v}']))
        n.append(helper.make_node('Add', [f'c1v_{v}', 'one_i64'], [f'c1p1_{v}']))
        n.append(helper.make_node('Sub', [f'r1p1_{v}', f'r0v_{v}'], [f'hv_{v}']))
        n.append(helper.make_node('Sub', [f'c1p1_{v}', f'c0v_{v}'], [f'wv_{v}']))

        n.append(helper.make_node('Reshape', [f'r0v_{v}', 'shape1'], [f'r0v1_{v}']))
        n.append(helper.make_node('Reshape', [f'c0v_{v}', 'shape1'], [f'c0v1_{v}']))
        n.append(helper.make_node('Reshape', [f'r1p1_{v}', 'shape1'], [f'r1p1_1_{v}']))
        n.append(helper.make_node('Reshape', [f'c1p1_{v}', 'shape1'], [f'c1p1_1_{v}']))
        n.append(helper.make_node('Concat', [f'r0v1_{v}', f'c0v1_{v}'], [f'cropstart_{v}'], axis=0))
        n.append(helper.make_node('Concat', [f'r1p1_1_{v}', f'c1p1_1_{v}'], [f'cropend_{v}'], axis=0))
        n.append(helper.make_node('Slice', [chan_v, f'cropstart_{v}', f'cropend_{v}', 'axes01'], [f'crop_{v}']))

        n.append(helper.make_node('Slice', [f'crop_{v}', 'm1_1d', 'negend_1d', 'ax1', 'm1_1d'], [f'vX_{v}']))
        n.append(helper.make_node('Slice', [f'crop_{v}', 'm1_1d', 'negend_1d', 'ax0', 'm1_1d'], [f'hX_{v}']))
        n.append(helper.make_node('Transpose', [f'crop_{v}'], [f'dX_{v}'], perm=[1, 0]))
        n.append(helper.make_node('Slice', [f'crop_{v}', 'm1_1d', 'negend_1d', 'ax0', 'm1_1d'], [f'bothflip1_{v}']))
        n.append(helper.make_node('Slice', [f'bothflip1_{v}', 'm1_1d', 'negend_1d', 'ax1', 'm1_1d'], [f'bothflip_{v}']))
        n.append(helper.make_node('Transpose', [f'bothflip_{v}'], [f'cX_{v}'], perm=[1, 0]))

        n.append(helper.make_node('Sub', ['thirty', f'hv_{v}'], [f'padb_vh_{v}']))
        n.append(helper.make_node('Sub', ['thirty', f'wv_{v}'], [f'padr_vh_{v}']))
        n.append(helper.make_node('Sub', ['thirty', f'wv_{v}'], [f'padb_dc_{v}']))
        n.append(helper.make_node('Sub', ['thirty', f'hv_{v}'], [f'padr_dc_{v}']))
        n.append(helper.make_node('Reshape', [f'padb_vh_{v}', 'shape1'], [f'padb_vh1_{v}']))
        n.append(helper.make_node('Reshape', [f'padr_vh_{v}', 'shape1'], [f'padr_vh1_{v}']))
        n.append(helper.make_node('Reshape', [f'padb_dc_{v}', 'shape1'], [f'padb_dc1_{v}']))
        n.append(helper.make_node('Reshape', [f'padr_dc_{v}', 'shape1'], [f'padr_dc1_{v}']))
        n.append(helper.make_node('Concat', ['zeros2', f'padb_vh1_{v}', f'padr_vh1_{v}'], [f'pads_vh_{v}'], axis=0))
        n.append(helper.make_node('Concat', ['zeros2', f'padb_dc1_{v}', f'padr_dc1_{v}'], [f'pads_dc_{v}'], axis=0))

        n.append(helper.make_node('Pad', [f'vX_{v}', f'pads_vh_{v}', 'zero_f'], [f'pv_{v}'], mode='constant'))
        n.append(helper.make_node('Pad', [f'hX_{v}', f'pads_vh_{v}', 'zero_f'], [f'ph_{v}'], mode='constant'))
        n.append(helper.make_node('Pad', [f'dX_{v}', f'pads_dc_{v}', 'zero_f'], [f'pd_{v}'], mode='constant'))
        n.append(helper.make_node('Pad', [f'cX_{v}', f'pads_dc_{v}', 'zero_f'], [f'pc_{v}'], mode='constant'))

        w1 = compare_first_diff(n, f'pv_{v}', f'ph_{v}', f'cmp1_{v}')
        w2 = compare_first_diff(n, f'pd_{v}', f'pc_{v}', f'cmp2_{v}')
        n.append(helper.make_node('Where', [w1, f'pv_{v}', f'ph_{v}'], [f'data1_{v}']))
        n.append(helper.make_node('Where', [w2, f'pd_{v}', f'pc_{v}'], [f'data2_{v}']))
        w3 = compare_first_diff(n, f'data1_{v}', f'data2_{v}', f'cmp3_{v}')
        n.append(helper.make_node('Where', [w3, f'data1_{v}', f'data2_{v}'], [f'finaldata_{v}']))
        n.append(helper.make_node('Where', [w3, f'hv_{v}', f'wv_{v}'], [f'finalh_{v}']))
        n.append(helper.make_node('Where', [w3, f'wv_{v}', f'hv_{v}'], [f'finalw_{v}']))

        n.append(helper.make_node('Reshape', [f'finalh_{v}', 'shape1'], [f'finalh1_{v}']))
        n.append(helper.make_node('Reshape', [f'finalw_{v}', 'shape1'], [f'finalw1_{v}']))
        n.append(helper.make_node('Concat', [f'finalh1_{v}', f'finalw1_{v}'], [f'finalend_{v}'], axis=0))
        n.append(helper.make_node('Slice', [f'finaldata_{v}', 'zeros2', f'finalend_{v}', 'axes01'], [f'finalcrop_{v}']))

        n.append(helper.make_node('Reshape', [f'rank_{v}', 'shape1'], [f'rank1_{v}']))
        n.append(helper.make_node('Sub', ['x32_1', f'rank1_{v}'], [f'xmr_{v}']))
        n.append(helper.make_node('Sub', [f'xmr_{v}', f'finalh1_{v}'], [f'padbot_{v}']))
        n.append(helper.make_node('Sub', [f'xmr_{v}', f'finalw1_{v}'], [f'padrig_{v}']))
        n.append(helper.make_node('Concat', [f'rank1_{v}', f'rank1_{v}', f'padbot_{v}', f'padrig_{v}'], [f'pospads_{v}'], axis=0))
        n.append(helper.make_node('Pad', [f'finalcrop_{v}', f'pospads_{v}', 'zero_f'], [f'positioned_{v}'], mode='constant'))

        n.append(helper.make_node('Cast', [f'present_{v}'], [f'presentf_{v}'], to=F))
        n.append(helper.make_node('Mul', [f'positioned_{v}', f'presentf_{v}'], [f'maskcontrib_{v}']))
        n.append(helper.make_node('Greater', [f'maskcontrib_{v}', 'half_f'], [f'maskbool_{v}']))
        add(f'vfloat_{v}', float(v), np.float32)
        n.append(helper.make_node('Mul', [f'maskcontrib_{v}', f'vfloat_{v}'], [f'valuecontrib_{v}']))
        maskbool_names.append(f'maskbool_{v}')
        valuecontrib_names.append(f'valuecontrib_{v}')

    # ---- rank layering: combine the (up to one) color occupying each rank slot ----
    maskbool_r_names = []
    value_r_names = []
    for r in range(RMAX + 1):
        add(f'rconst_{r}', r)
        contrib_mask_names = []
        contrib_val_names = []
        for v in range(NCOL):
            n.append(helper.make_node('Equal', [f'rank_{v}', f'rconst_{r}'], [f'rankeq_{r}_{v}']))
            n.append(helper.make_node('And', [f'present_{v}', f'rankeq_{r}_{v}'], [f'gate_{r}_{v}']))
            n.append(helper.make_node('Cast', [f'gate_{r}_{v}'], [f'gatef_{r}_{v}'], to=F))
            n.append(helper.make_node('Cast', [f'maskbool_{v}'], [f'maskf_{v}_{r}'], to=F))
            n.append(helper.make_node('Mul', [f'maskf_{v}_{r}', f'gatef_{r}_{v}'], [f'cmask_{r}_{v}']))
            n.append(helper.make_node('Mul', [f'valuecontrib_{v}', f'gatef_{r}_{v}'], [f'cval_{r}_{v}']))
            contrib_mask_names.append(f'cmask_{r}_{v}')
            contrib_val_names.append(f'cval_{r}_{v}')
        n.append(helper.make_node('Max', contrib_mask_names, [f'maskr_{r}']))
        n.append(helper.make_node('Sum', contrib_val_names, [f'valuer_{r}']))
        n.append(helper.make_node('Greater', [f'maskr_{r}', 'half_f'], [f'maskrbool_{r}']))
        maskbool_r_names.append(f'maskrbool_{r}')
        value_r_names.append(f'valuer_{r}')

    # ---- initial canvas (bg-filled, x32 x x32) ----
    n.append(helper.make_node('Cast', ['bg_idx'], ['bg_f'], to=F))
    n.append(helper.make_node('Reshape', ['bg_f', 'shape1x1'], ['bg_f_11']))
    n.append(helper.make_node('Concat', ['x32_1', 'x32_1'], ['canvas_shape'], axis=0))
    n.append(helper.make_node('Expand', ['bg_f_11', 'canvas_shape'], ['canvas_init']))

    # ---- paint + rotate x4 (builds the rotationally-symmetric mosaic) ----
    cur = 'canvas_init'
    for app in range(4):
        for r in range(RMAX + 1):
            nxt = f'canvas_p_{app}_{r}'
            n.append(helper.make_node('Where', [f'maskrbool_{r}', f'valuer_{r}', cur], [nxt]))
            cur = nxt
        if app < 3:
            n.append(helper.make_node('Transpose', [cur], [f'canvas_t_{app}'], perm=[1, 0]))
            n.append(helper.make_node('Slice', [f'canvas_t_{app}', 'm1_1d', 'negend_1d', 'ax1', 'm1_1d'], [f'canvas_r_{app}']))
            cur = f'canvas_r_{app}'
    final_canvas = cur

    # ---- to one-hot output ----
    n.append(helper.make_node('Cast', [final_canvas], ['canvas_i'], to=I64))
    n.append(helper.make_node('Concat', ['one_1', 'x32_1', 'x32_1'], ['canvas3dshape'], axis=0))
    n.append(helper.make_node('Reshape', ['canvas_i', 'canvas3dshape'], ['canvas_3d']))
    n.append(helper.make_node('OneHot', ['canvas_3d', 'depth10', 'oh_vals'], ['output'], axis=1))

    graph = helper.make_graph(n, 'task096', [x], [y], I)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])

def _make():
    return _crop_pad(build_096())

model = _bake(_make(), 96)
