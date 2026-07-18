import json
import math
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data", "neurogolf_utils"))
import neurogolf_utils as ngu


F = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
I64 = TensorProto.INT64
I32 = TensorProto.INT32
I8 = TensorProto.INT8
U8 = TensorProto.UINT8
B = TensorProto.BOOL

nodes = []
inits = []
counter = 0


def name(prefix):
    global counter
    counter += 1
    return f"{prefix}_{counter}"


def K(value, dtype=None, nm=None):
    arr = np.asarray(value, dtype=dtype)
    nm = nm or name("k")
    inits.append(numpy_helper.from_array(arr, nm))
    return nm


def N(op, inputs, nm=None, outputs=1, **attrs):
    if outputs == 1:
        outs = [nm or name(op.lower())]
    else:
        outs = [name(f"{op.lower()}_{i}") for i in range(outputs)]
    nodes.append(helper.make_node(op, inputs, outs, name=name("node"), **attrs))
    return outs[0] if outputs == 1 else outs


axes_cache = {}


def AX(axes):
    axes = tuple(axes)
    if axes not in axes_cache:
        axes_cache[axes] = K(axes, np.int64, f"axes_{'_'.join(map(str, axes))}")
    return axes_cache[axes]


def Reduce(op, x, axes=None, keepdims=0, nm=None):
    ins = [x]
    if axes is not None:
        ins.append(AX(axes))
    return N(op, ins, nm=nm, keepdims=keepdims)


def Cast(x, to, nm=None):
    return N("Cast", [x], nm=nm, to=to)


def Reshape(x, shape, nm=None):
    return N("Reshape", [x, K(shape, np.int64)], nm=nm)


def Unsqueeze(x, axes, nm=None):
    return N("Unsqueeze", [x, AX(axes)], nm=nm)


def Squeeze(x, axes, nm=None):
    return N("Squeeze", [x, AX(axes)], nm=nm)


zero_i32 = K(0, np.int32, "zero_i32")
one_i32 = K(1, np.int32, "one_i32")
two_i32 = K(2, np.int32, "two_i32")
big_i32 = K(999, np.int32, "big_i32")
zero_i8 = K(0, np.int8, "zero_i8")
maxr_i8 = K(14, np.int8, "maxr_i8")
maxc_i8 = K(16, np.int8, "maxc_i8")
seventeen_i8 = K(17, np.uint8, "seventeen_u8")
seventeen_i32 = K(17, np.int32, "seventeen_i32")
zero_u8 = K(0, np.uint8, "zero_u8")
one_u8 = K(1, np.uint8, "one_u8")
ten_u8 = K(10, np.uint8, "ten_u8")
false_b = K(False, np.bool_, "false_bool")
true_b = K(True, np.bool_, "true_bool")

r15 = K(np.arange(15, dtype=np.int32), nm="r15")
c17 = K(np.arange(17, dtype=np.int32), nm="c17")
r16 = K(np.arange(16, dtype=np.int32).reshape(1, 16), nm="r16")
c18 = K(np.arange(18, dtype=np.int32).reshape(1, 18), nm="c18")

inp = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
out = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])

# Fused one-hot -> colour-index conversion.  Valid colours are encoded 1..10;
# zero is reserved for the absent/padded region.  This removes the full panel
# validity mask and lets the final comparison ignore padding for free.
colour_w = K(np.arange(1, 11, dtype=np.float32).reshape(1, 10, 1, 1), nm="colour_w")
grid_f = N("Conv", ["input", colour_w], nm="grid_f")
grid_u8 = Cast(grid_f, U8, "grid_u8")

# Actual input size and split orientation.
row_occ = Reduce("ReduceMax", grid_u8, [0, 1, 3], 0, "row_occ")
col_occ = Reduce("ReduceMax", grid_u8, [0, 1, 2], 0, "col_occ")
h_first_zero = Cast(N("ArgMin", [row_occ], nm="h_first_zero_i64", axis=0, keepdims=0), I32, "h_first_zero")
w_first_zero = Cast(N("ArgMin", [col_occ], nm="w_first_zero_i64", axis=0, keepdims=0), I32, "w_first_zero")
h_full = N("Greater", [Reduce("ReduceMin", row_occ, None, 0), zero_u8], nm="h_full")
w_full = N("Greater", [Reduce("ReduceMin", col_occ, None, 0), zero_u8], nm="w_full")
h = N("Where", [h_full, K(np.int32(30), nm="thirty_i32"), h_first_zero], nm="h")
w = N("Where", [w_full, K(np.int32(30), nm="thirty_i32_b"), w_first_zero], nm="w")
horizontal = N("Greater", [w, h], nm="horizontal")
h2 = N("Div", [h, two_i32], nm="h2")
w2 = N("Div", [w, two_i32], nm="w2")
ph = N("Where", [horizontal, h, h2], nm="panel_h")
pw = N("Where", [horizontal, w2, w], nm="panel_w")

rv = N("Less", [r15, ph], nm="row_valid")
cv = N("Less", [c17, pw], nm="col_valid")
rv_u8 = Cast(Unsqueeze(rv, [1]), U8, "row_valid_u8")
cv_u8 = Cast(Unsqueeze(cv, [0]), U8, "col_valid_u8")
gm_u8 = N(
    "QLinearMatMul",
    [
        rv_u8, K(np.float32(1), nm="qgm_a_scale"), zero_u8,
        cv_u8, K(np.float32(1), nm="qgm_b_scale"), zero_u8,
        K(np.float32(1), nm="qgm_y_scale"), zero_u8,
    ],
    nm="grid_mask_u8",
)

# First panel is always top-left. Second panel starts at (ph,0) or (0,pw).
starts = K([0, 0], np.int64, "slice_starts")
ends = K([15, 17], np.int64, "slice_ends")
slice_axes = K([2, 3], np.int64, "slice_axes")
a4 = N("Slice", [grid_u8, starts, ends, slice_axes], nm="panel_a4")
a = a4
roff = N("Where", [horizontal, zero_i32, ph], nm="row_offset")
coff = N("Where", [horizontal, pw, zero_i32], nm="col_offset")
bri = N("Add", [r15, roff], nm="b_row_indices")
bci0 = N("Add", [c17, coff], nm="b_col_indices0")
bci = N("Clip", [bci0, zero_i32, K(29, np.int32, "twentynine_i32")], nm="b_col_indices")
brow = N("Gather", [grid_u8, bri], nm="panel_b_rows", axis=2)
b4 = N("Gather", [brow, bci], nm="panel_b4", axis=3)
b = b4


def corner_background(panel, prefix):
    hm1 = N("Sub", [ph, one_i32], nm=f"{prefix}_hm1")
    wm1 = N("Sub", [pw, one_i32], nm=f"{prefix}_wm1")
    row0 = N("Gather", [panel, zero_i32], nm=f"{prefix}_row0", axis=2)
    rowl = N("Gather", [panel, hm1], nm=f"{prefix}_rowl", axis=2)
    tl = N("Gather", [row0, zero_i32], nm=f"{prefix}_tl", axis=2)
    bl = N("Gather", [rowl, zero_i32], nm=f"{prefix}_bl", axis=2)
    br = N("Gather", [rowl, wm1], nm=f"{prefix}_br", axis=2)
    return N("Where", [N("Equal", [tl, br]), tl, bl], nm=f"{prefix}_bg")


bg_a = corner_background(a, "a")
bg_b = corner_background(b, "b")

# The dense side has at least eight non-background cells; the sparse side has
# at most six. This is the same structural discriminator as the verified model.
a_eq_bg = N("Equal", [a, bg_a], nm="a_eq_bg")
a_fg_u8 = N("Where", [a_eq_bg, zero_u8, gm_u8], nm="a_fg_u8")
a_fg_f16 = Cast(a_fg_u8, F16, "a_fg_f16")
a_nfg = Reduce("ReduceSum", a_fg_f16, None, 0, "a_nfg")
a_dense = N("Greater", [a_nfg, K(np.float16(7.5), np.float16, "seven_half")], nm="a_dense")
dense = N("Where", [a_dense, a, b], nm="dense")
sparse = N("Where", [a_dense, b, a], nm="sparse")
dbg = N("Where", [a_dense, bg_a, bg_b], nm="dense_bg")
sbg = N("Where", [a_dense, bg_b, bg_a], nm="sparse_bg")
dense_eq_bg = N("Equal", [dense, dbg], nm="dense_eq_bg")
dmask = N("Where", [dense_eq_bg, zero_u8, gm_u8], nm="dense_mask")

# Exact foreground histogram; ScatterElements with reduction=add avoids a
# ten-channel one-hot materialization.
dense_flat = Reshape(dense, [255], "dense_flat")
dense_i32 = Cast(dense_flat, I32, "dense_i32")
dmask_u8 = Reshape(dmask, [255], "dense_mask_u8")
hist = N(
    "ScatterElements",
    [K(np.zeros(11, np.uint8), nm="hist_zero"), dense_i32, dmask_u8],
    nm="hist",
    axis=0,
    reduction="add",
)
filler_i64 = N("ArgMax", [hist], nm="filler_i64", axis=0, keepdims=0)
filler = Cast(filler_i64, U8, "filler")

# Detect the top-left of every solid rectangle with one quantized convolution:
# score = 3*center - up - left.  TopK can consume the score directly; a
# separate 255-cell Equal mask would only materialize redundant data.
dmask4 = dmask
qscore = N(
    "QLinearConv",
    [
        dmask4,
        K(np.float32(1), nm="qx_scale"),
        zero_u8,
        K(np.array([[[[0, -1], [-1, 3]]]], np.int8), nm="corner_w"),
        K(np.float32(1), nm="qw_scale"),
        K(np.int8(0), nm="qw_zero"),
        K(np.float32(1), nm="qy_scale"),
        zero_u8,
    ],
    nm="corner_score",
    pads=[1, 1, 0, 0],
)
qscore_flat = Reshape(qscore, [255], "corner_score_flat")
corner_scores_f16 = Cast(qscore_flat, F16, "corner_scores_f16")
corner_vals, corner_idx = N("TopK", [corner_scores_f16, K([3], np.int64, "k3")], outputs=2, axis=0, largest=1, sorted=1)
corner_present = N("Equal", [corner_vals, K(np.float16(3), nm="three_f16")], nm="corner_present")
corner_idx32 = Cast(corner_idx, I32, "corner_idx32")
corner_r = N("Div", [corner_idx32, seventeen_i32], nm="corner_r")
corner_c = N("Mod", [corner_idx32, seventeen_i32], nm="corner_c")

# Rectangle extents from the first background cell below/right of each corner.
col_slices4 = N("Gather", [dmask, corner_c], nm="corner_cols4", axis=3)
col_slices = Squeeze(col_slices4, [0, 1], nm="corner_cols")
col_slices_t = N("Transpose", [col_slices], nm="corner_cols_t", perm=[1, 0])
col_ext = N("Concat", [col_slices_t, K(np.zeros((3, 1), np.uint8), nm="zero_3x1")], nm="corner_cols_ext", axis=1)
valid_h = N(
    "And",
    [N("Equal", [col_ext, zero_u8]), N("GreaterOrEqual", [r16, Unsqueeze(corner_r, [1])])],
    nm="height_candidates_valid",
)
h_candidates = N("Where", [valid_h, r16, big_i32], nm="height_candidates")
rect_bottom = Reduce("ReduceMin", h_candidates, [1], 0, "rect_bottom")
rect_h = N("Sub", [rect_bottom, corner_r], nm="rect_h")

row_slices4 = N("Gather", [dmask, corner_r], nm="corner_rows4", axis=2)
row_slices = Squeeze(row_slices4, [0, 1], nm="corner_rows")
row_ext = N("Concat", [row_slices, K(np.zeros((3, 1), np.uint8), nm="zero_3x1_b")], nm="corner_rows_ext", axis=1)
valid_w = N(
    "And",
    [N("Equal", [row_ext, zero_u8]), N("GreaterOrEqual", [c18, Unsqueeze(corner_c, [1])])],
    nm="width_candidates_valid",
)
w_candidates = N("Where", [valid_w, c18, big_i32], nm="width_candidates")
rect_right = Reduce("ReduceMin", w_candidates, [1], 0, "rect_right")
rect_w = N("Sub", [rect_right, corner_c], nm="rect_w")

# Up to six non-filler cells describe all dense anchor constellations.
dense_eq_fill = N("Equal", [dense_flat, filler], nm="dense_eq_fill")
dot_mask = N("Where", [dense_eq_fill, zero_u8, dmask_u8], nm="dense_dot_mask")
dot_f16 = Cast(dot_mask, F16, "dense_dot_f16")
dot_vals, dot_idx = N("TopK", [dot_f16, K([6], np.int64, "k6")], outputs=2, axis=0, largest=1, sorted=1)
dot_present = N("Greater", [dot_vals, K(np.float16(0), nm="zero_f16_b")], nm="dense_dot_present")
dot_idx32 = Cast(dot_idx, I32, "dense_dot_idx32")
dot_r = N("Div", [dot_idx32, seventeen_i32], nm="dense_dot_r")
dot_c = N("Mod", [dot_idx32, seventeen_i32], nm="dense_dot_c")
dot_colour = N("Gather", [dense_flat, dot_idx], nm="dense_dot_colour", axis=0)

dr6 = Unsqueeze(dot_r, [1], "dense_r_6x1")
dc6 = Unsqueeze(dot_c, [1], "dense_c_6x1")
cr13 = Unsqueeze(corner_r, [0], "corner_r_1x3")
cc13 = Unsqueeze(corner_c, [0], "corner_c_1x3")
rb13 = Unsqueeze(N("Add", [corner_r, rect_h]), [0], "rect_bottom_1x3")
rr13 = Unsqueeze(N("Add", [corner_c, rect_w]), [0], "rect_right_1x3")
assign = N(
    "And",
    [
        N("And", [N("GreaterOrEqual", [dr6, cr13]), N("Less", [dr6, rb13])]),
        N("And", [N("GreaterOrEqual", [dc6, cc13]), N("Less", [dc6, rr13])]),
    ],
    nm="dense_anchor_assignment0",
)
assign = N("And", [assign, Unsqueeze(dot_present, [1])], nm="dense_anchor_assignment")
assign_i32 = Cast(assign, I32, "assign_i32")
anchor_counts = Reduce("ReduceSum", assign_i32, [0], 0, "anchor_counts")
object_score = N("Where", [corner_present, anchor_counts, K(np.int32(-1), nm="minus_one_i32")], nm="object_score")
_, object_order = N("TopK", [object_score, K([3], np.int64, "k3_b")], outputs=2, axis=0, largest=1, sorted=1)
orow = N("Gather", [corner_r, object_order], nm="object_row", axis=0)
ocol = N("Gather", [corner_c, object_order], nm="object_col", axis=0)
oh = N("Gather", [rect_h, object_order], nm="object_h", axis=0)
ow = N("Gather", [rect_w, object_order], nm="object_w", axis=0)
opresent = N("Gather", [corner_present, object_order], nm="object_present", axis=0)
oassign = N("Gather", [assign, object_order], nm="object_assign", axis=1)

# Sparse markers.
sparse_eq_bg = N("Equal", [sparse, sbg], nm="sparse_eq_bg")
sparse_flat = Reshape(sparse, [255], "sparse_flat")
sparse_marks = N("Where", [sparse_eq_bg, zero_u8, gm_u8], nm="sparse_marks")
sparse_marks_flat = Reshape(sparse_marks, [255], "sparse_marks_flat")
sparse_f16 = Cast(sparse_marks_flat, F16, "sparse_mask_f16")
svals, sidx = N("TopK", [sparse_f16, K([6], np.int64, "k6_b")], outputs=2, axis=0, largest=1, sorted=1)
spresent = N("Greater", [svals, K(np.float16(0), nm="zero_f16_c")], nm="sparse_present")
sidx32 = Cast(sidx, I32, "sparse_idx32")
sr = N("Div", [sidx32, seventeen_i32], nm="sparse_r")
sc = N("Mod", [sidx32, seventeen_i32], nm="sparse_c")
scolour = N("Gather", [sparse_flat, sidx], nm="sparse_colour", axis=0)

# Vectorised pattern matcher: evaluate the 6 candidate sparse anchors against
# all 6 dense anchors and all 3 rectangles in one compact [6,6,3] tensor.
dr8 = Cast(dot_r, I8, "dense_r_i8")
dc8 = Cast(dot_c, I8, "dense_c_i8")
sr8 = Cast(sr, I8, "sparse_r_i8")
sc8 = Cast(sc, I8, "sparse_c_i8")
or8 = Cast(orow, I8, "object_r_i8")
oc8 = Cast(ocol, I8, "object_c_i8")
rel_r = N("Sub", [Unsqueeze(dr8, [1]), Unsqueeze(or8, [0])], nm="relative_r")
rel_c = N("Sub", [Unsqueeze(dc8, [1]), Unsqueeze(oc8, [0])], nm="relative_c")
oassign_u8 = Cast(oassign, U8, "object_assign_u8")
anchor_index = N("ArgMax", [oassign_u8], nm="anchor_index", axis=0, keepdims=0)
anchor_r = N("Gather", [dr8, anchor_index], nm="anchor_r", axis=0)
anchor_c = N("Gather", [dc8, anchor_index], nm="anchor_c", axis=0)
anchor_rel_r = N("Sub", [anchor_r, or8], nm="anchor_relative_r")
anchor_rel_c = N("Sub", [anchor_c, oc8], nm="anchor_relative_c")
cand_r = N("Sub", [Unsqueeze(sr8, [1]), Unsqueeze(anchor_rel_r, [0])], nm="candidate_r")
cand_c = N("Sub", [Unsqueeze(sc8, [1]), Unsqueeze(anchor_rel_c, [0])], nm="candidate_c")

target_r = N("Add", [Unsqueeze(cand_r, [1]), Unsqueeze(rel_r, [0])], nm="target_r")
target_c = N("Add", [Unsqueeze(cand_c, [1]), Unsqueeze(rel_c, [0])], nm="target_c")
target_r_clip = N("Clip", [target_r, zero_i8, maxr_i8], nm="target_r_clip")
target_c_clip = N("Clip", [target_c, zero_i8, maxc_i8], nm="target_c_clip")
target_ru8 = Cast(target_r_clip, U8, "target_r_u8")
target_cu8 = Cast(target_c_clip, U8, "target_c_u8")
target_linear_u8 = N("Add", [N("Mul", [target_ru8, seventeen_i8]), target_cu8], nm="target_linear_u8")
target_linear = Cast(target_linear_u8, I32, "target_linear")
target_colours = N("Gather", [sparse_flat, target_linear], nm="target_colours", axis=0)

dc61 = Unsqueeze(dot_colour, [1], "dense_colour_6x1")
obj_colour_cells = N("Where", [oassign, dc61, zero_u8], nm="object_colour_cells")
obj_colour = Reduce("ReduceMax", obj_colour_cells, [0], 0, "object_colour")
colour_ok = N("Equal", [target_colours, Reshape(obj_colour, [1, 1, 3])], nm="target_colour_ok")
ignore = Reshape(N("Not", [oassign]), [1, 6, 3], "ignore_nonanchors")
match_or_ignore = N("Or", [colour_ok, ignore], nm="match_or_ignore")
match_u8 = Cast(match_or_ignore, U8, "match_u8")
pattern_ok_u8 = Reduce("ReduceMin", match_u8, [1], 0, "pattern_ok_u8")
pattern_ok = Cast(pattern_ok_u8, B, "pattern_ok")

ph8 = Cast(ph, I8, "panel_h_i8")
pw8 = Cast(pw, I8, "panel_w_i8")
oh8 = Cast(oh, I8, "object_h_i8")
ow8 = Cast(ow, I8, "object_w_i8")
max_top_r = N("Sub", [ph8, oh8], nm="max_top_r")
max_top_c = N("Sub", [pw8, ow8], nm="max_top_c")
in_r = N("And", [N("GreaterOrEqual", [cand_r, zero_i8]), N("LessOrEqual", [cand_r, Unsqueeze(max_top_r, [0])])], nm="candidate_in_r")
in_c = N("And", [N("GreaterOrEqual", [cand_c, zero_i8]), N("LessOrEqual", [cand_c, Unsqueeze(max_top_c, [0])])], nm="candidate_in_c")
candidate_ok = N("And", [pattern_ok, N("And", [in_r, in_c])], nm="candidate_ok0")
candidate_ok = N("And", [candidate_ok, Unsqueeze(spresent, [1])], nm="candidate_ok1")
candidate_ok = N("And", [candidate_ok, Unsqueeze(opresent, [0])], nm="candidate_ok")

cand_r32 = Cast(cand_r, I32, "candidate_r_i32")
cand_c32 = Cast(cand_c, I32, "candidate_c_i32")
candidate_key = N("Add", [N("Mul", [cand_r32, seventeen_i32]), cand_c32], nm="candidate_key")

def choose_object(index, consumed):
    valid_col = N("Gather", [candidate_ok, K(index, np.int64)], nm=f"valid_{index}", axis=1)
    if consumed is not None:
        valid_col = N("And", [valid_col, N("Not", [consumed])], nm=f"valid_unconsumed_{index}")
    key_col = N("Gather", [candidate_key, K(index, np.int64)], nm=f"key_{index}", axis=1)
    masked_key = N("Where", [valid_col, key_col, big_i32], nm=f"masked_key_{index}")
    best = Reduce("ReduceMin", masked_key, None, 0, f"best_{index}")
    placed = N("Less", [best, big_i32], nm=f"placed_{index}")
    yy = N("Div", [best, seventeen_i32], nm=f"place_r_{index}")
    xx = N("Mod", [best, seventeen_i32], nm=f"place_c_{index}")
    hh = N("Gather", [oh, K(index, np.int64)], nm=f"place_h_{index}", axis=0)
    ww = N("Gather", [ow, K(index, np.int64)], nm=f"place_w_{index}", axis=0)
    inside = N(
        "And",
        [
            N("And", [N("GreaterOrEqual", [sr, yy]), N("Less", [sr, N("Add", [yy, hh])])]),
            N("And", [N("GreaterOrEqual", [sc, xx]), N("Less", [sc, N("Add", [xx, ww])])]),
        ],
        nm=f"inside_{index}",
    )
    inside = N("And", [inside, placed], nm=f"consumed_{index}")
    return yy, xx, placed, inside


y0, x0, p0, cons0 = choose_object(0, None)
y1, x1, p1, cons1a = choose_object(1, cons0)
cons01 = N("Or", [cons0, cons1a], nm="consumed_01")
y2, x2, p2, cons2 = choose_object(2, cons01)
yy = N("Concat", [Unsqueeze(y0, [0]), Unsqueeze(y1, [0]), Unsqueeze(y2, [0])], nm="place_r", axis=0)
xx = N("Concat", [Unsqueeze(x0, [0]), Unsqueeze(x1, [0]), Unsqueeze(x2, [0])], nm="place_c", axis=0)
placed = N("Concat", [Unsqueeze(p0, [0]), Unsqueeze(p1, [0]), Unsqueeze(p2, [0])], nm="placed", axis=0)

# Rectangle union via one quantized matrix multiplication.  Row membership is
# [15,3], column membership is [3,17], and their product is the number of
# selected rectangles covering each output cell.
y3 = Unsqueeze(yy, [1], "place_r_3x1")
x3 = Unsqueeze(xx, [1], "place_c_3x1")
row_membership = N(
    "And",
    [N("GreaterOrEqual", [Unsqueeze(r15, [0]), y3]), N("Less", [Unsqueeze(r15, [0]), N("Add", [y3, Unsqueeze(oh, [1])])])],
    nm="row_membership0",
)
row_membership = N("And", [row_membership, Unsqueeze(placed, [1])], nm="row_membership")
col_membership = N(
    "And",
    [N("GreaterOrEqual", [Unsqueeze(c17, [0]), x3]), N("Less", [Unsqueeze(c17, [0]), N("Add", [x3, Unsqueeze(ow, [1])])])],
    nm="col_membership",
)
row_u8 = Cast(N("Transpose", [row_membership], nm="row_membership_t", perm=[1, 0]), U8, "row_membership_u8")
col_u8 = Cast(col_membership, U8, "col_membership_u8")
cover_count = N(
    "QLinearMatMul",
    [
        row_u8, K(np.float32(1), nm="qmm_a_scale"), zero_u8,
        col_u8, K(np.float32(1), nm="qmm_b_scale"), zero_u8,
        K(np.float32(1), nm="qmm_y_scale"), zero_u8,
    ],
    nm="cover_count",
)
rect_union = N("Greater", [cover_count, zero_u8], nm="rectangle_union")

paint_bg = N("And", [rect_union, sparse_eq_bg], nm="paint_background")
base = N("Mul", [sparse, gm_u8], nm="base")
painted = N("Where", [paint_bg, filler, base], nm="painted")
pads = K([0, 0, 0, 0, 0, 0, 15, 13], np.int64, "pads")
padded = N("Pad", [painted, pads, zero_u8], nm="padded", mode="constant")
channels = K(np.arange(1, 11, dtype=np.uint8).reshape(1, 10, 1, 1), nm="channels")
N("Equal", [channels, padded], nm="output")

model = helper.make_model(
    helper.make_graph(nodes, "task366_rewrite", [inp], [out], inits),
    ir_version=10,
    opset_imports=[helper.make_opsetid("", 18)],
)


def audit(path):
    model = onnx.load(path)
    sanitized = ngu.sanitize_model(model)
    onnx.checker.check_model(sanitized, full_check=True)
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    options.profile_file_prefix = os.path.join(ROOT, "scratch_onnx", "task366_rewrite_profile")
    session = ort.InferenceSession(sanitized.SerializeToString(), options)
    data = json.load(open(os.path.join(ROOT, "data", "task366.json")))
    failures = []
    for i, example in enumerate(data["train"] + data["test"] + data["arc-gen"]):
        benchmark = ngu.convert_to_numpy(example)
        if benchmark is None:
            continue
        prediction = ngu.run_network(session, benchmark["input"])
        if not np.array_equal(prediction, benchmark["output"]):
            failures.append(i)
    trace = session.end_profiling()
    memory, params = ngu.score_network(sanitized, trace)
    try:
        os.remove(trace)
    except OSError:
        pass
    cost = memory + params
    points = 25 - math.log(cost)
    print(f"nodes={len(model.graph.node)} nfail={len(failures)} failures={failures[:20]}")
    print(f"memory={memory} params={params} cost={cost} points={points:.6f}")
    return len(failures), cost, points


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "scratch_onnx"), exist_ok=True)
    candidate = os.path.join(ROOT, "scratch_onnx", "task366_rewrite.onnx")
    onnx.save(model, candidate)
    print(candidate)
    audit(candidate)
