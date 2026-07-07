import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

F = TensorProto.FLOAT


def _K(name, value, dtype=None):
    arr = np.asarray(value, dtype=dtype)
    return numpy_helper.from_array(arr, name=name)


def _rect_constants():
    """All 10x10 axis-aligned rectangles: mask, area, and center-id."""
    rect_masks = []
    areas = []
    center_ids = []

    centers = [(r2 / 2.0, c2 / 2.0) for r2 in range(19) for c2 in range(19)]
    center_to_id = {center: i for i, center in enumerate(centers)}

    for r0 in range(10):
        for r1 in range(r0, 10):
            for c0 in range(10):
                for c1 in range(c0, 10):
                    m = np.zeros((10, 10), dtype=np.float32)
                    m[r0:r1 + 1, c0:c1 + 1] = 1.0
                    rect_masks.append(m.reshape(100))

                    areas.append(float((r1 - r0 + 1) * (c1 - c0 + 1)))
                    center = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
                    center_ids.append(center_to_id[center])

    return (
        np.stack(rect_masks).astype(np.float32),   # [3025,100]
        np.asarray(areas, dtype=np.float32),       # [3025]
        np.asarray(center_ids, dtype=np.int64),    # [3025]
    )


def _rot(r, c, cr, cc, k):
    rr = float(r)
    cc2 = float(c)
    for _ in range(k):
        dr = rr - cr
        dc = cc2 - cc
        rr = cr - dc
        cc2 = cc + dr
    return rr, cc2


def _src_indices():
    """
    For each possible 10x10 rectangle center, and each output target cell,
    store the 4 inverse-rotation source indices.
    Index 100 is a zero sentinel appended to the flattened source grid.
    """
    centers = [(r2 / 2.0, c2 / 2.0) for r2 in range(19) for c2 in range(19)]
    src = np.full((len(centers), 100, 4), 100, dtype=np.int64)

    for ci, (cr, cc) in enumerate(centers):
        for tr in range(10):
            for tc in range(10):
                t = tr * 10 + tc
                for k in range(4):
                    # inverse of k quarter-turns
                    sr, sc = _rot(tr, tc, cr, cc, (4 - k) % 4)
                    if abs(sr - round(sr)) < 1e-6 and abs(sc - round(sc)) < 1e-6:
                        sr = int(round(sr))
                        sc = int(round(sc))
                        if 0 <= sr < 10 and 0 <= sc < 10:
                            src[ci, t, k] = sr * 10 + sc

    return src


def build_onnx_model():
    rect_masks, areas, center_ids = _rect_constants()
    src_idx = _src_indices()

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])

    init = [
        _K("sl_st", [0, 1, 0, 0], np.int64),
        _K("sl_en", [1, 10, 10, 10], np.int64),
        _K("sl_ax", [0, 1, 2, 3], np.int64),

        _K("shape_mflat", [1, 100], np.int64),
        _K("shape_nzflat", [1, 9, 100], np.int64),
        _K("shape_nz10", [1, 9, 10, 10], np.int64),

        _K("RECT", rect_masks, np.float32),
        _K("AREAS", areas, np.float32),
        _K("CENTER_IDS", center_ids, np.int64),
        _K("SRC_IDX", src_idx, np.int64),

        _K("zero_src", np.zeros((1, 9, 1), dtype=np.float32), np.float32),
        _K("pad_nz", [0, 0, 0, 0, 0, 0, 20, 20], np.int64),
        _K("zero_f", [0.0], np.float32),
        _K("ones30", np.ones((1, 1, 30, 30), dtype=np.float32), np.float32),
    ]

    nodes = [
        # Use only colours 1..9 and the real ARC area 10x10.
        helper.make_node("Slice", ["input", "sl_st", "sl_en", "sl_ax"], ["nz10"]),

        # Non-zero mask of the 10x10 task grid.
        helper.make_node("ReduceSum", ["nz10"], ["mask10"], axes=[1], keepdims=1),
        helper.make_node("Reshape", ["mask10", "shape_mflat"], ["mflat"]),
        helper.make_node("Transpose", ["mflat"], ["mcol"], perm=[1, 0]),

        # Score every possible filled rectangle by area; invalid rectangles score 0.
        helper.make_node("MatMul", ["RECT", "mcol"], ["rect_cnt_col"]),
        helper.make_node("Squeeze", ["rect_cnt_col"], ["rect_cnt"], axes=[1]),
        helper.make_node("Equal", ["rect_cnt", "AREAS"], ["rect_valid"]),
        helper.make_node("Cast", ["rect_valid"], ["rect_valid_f"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["rect_valid_f", "AREAS"], ["rect_score"]),
        helper.make_node("ArgMax", ["rect_score"], ["best_rect"], axis=0, keepdims=0),

        # Convert best rectangle to one of the 361 possible half-integer centers.
        helper.make_node("Gather", ["CENTER_IDS", "best_rect"], ["center_id"], axis=0),
        helper.make_node("Gather", ["SRC_IDX", "center_id"], ["sel_src"], axis=0),

        # Rotate all non-zero colour channels around that center and union them.
        helper.make_node("Reshape", ["nz10", "shape_nzflat"], ["nzflat"]),
        helper.make_node("Concat", ["nzflat", "zero_src"], ["nzext"], axis=2),
        helper.make_node("Gather", ["nzext", "sel_src"], ["gathered"], axis=2),
        helper.make_node("ReduceMax", ["gathered"], ["nzoutflat"], axes=[3], keepdims=0),
        helper.make_node("Reshape", ["nzoutflat", "shape_nz10"], ["nzout10"]),

        # Pad colours 1..9 back to 30x30, then rebuild background channel 0.
        helper.make_node("Pad", ["nzout10", "pad_nz", "zero_f"], ["nzout30"], mode="constant"),
        helper.make_node("ReduceMax", ["nzout30"], ["occ"], axes=[1], keepdims=1),
        helper.make_node("Sub", ["ones30", "occ"], ["bg"]),
        helper.make_node("Concat", ["bg", "nzout30"], ["output"], axis=1),
    ]

    model = helper.make_model(
        helper.make_graph(nodes, "task361", [x], [y], init),
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 12)],
    )
    onnx.checker.check_model(model)
    return model


model = build_onnx_model()
