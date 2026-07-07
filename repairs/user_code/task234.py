
# ARC task234: move the non-rectangular object's solid block to the opposite end
# of its own bounding box, removing the one-cell connector tail.
# Input:  one-hot [1,10,30,30]
# Output: one-hot [1,10,30,30]

import numpy as np


def solve_234_numpy(grid):
    """Reference raw-grid solver; not used by the checker."""
    a = np.asarray(grid, dtype=np.int64)
    H, W = a.shape
    out = np.zeros_like(a)

    for colour in range(1, 10):
        m = (a == colour).astype(np.int64)
        if m.sum() == 0:
            continue

        row_has = (m.sum(axis=1) > 0).astype(np.int64)
        col_has = (m.sum(axis=0) > 0).astype(np.int64)
        h = int(row_has.sum())
        w = int(col_has.sum())
        count = int(m.sum())

        # A normal solid rectangle stays as-is.
        if count == h * w:
            out[m.astype(bool)] = colour
            continue

        # For the moving object, its bbox contains a solid block plus a 1-cell tail.
        row_count = m.sum(axis=1)
        col_count = m.sum(axis=0)

        full_row = ((row_count == w) & (row_has == 1)).astype(np.int64)
        full_col = ((col_count == h) & (col_has == 1)).astype(np.int64)

        krow = int(full_row.sum())
        kcol = int(full_col.sum())

        rank_top = np.cumsum(row_has)
        rank_bottom = np.cumsum(row_has[::-1])[::-1]
        top_k = ((row_has == 1) & (rank_top <= krow)).astype(np.int64)
        bottom_k = ((row_has == 1) & (rank_bottom <= krow)).astype(np.int64)

        rank_left = np.cumsum(col_has)
        rank_right = np.cumsum(col_has[::-1])[::-1]
        left_k = ((col_has == 1) & (rank_left <= kcol)).astype(np.int64)
        right_k = ((col_has == 1) & (rank_right <= kcol)).astype(np.int64)

        source_top = (full_row * top_k).sum() == krow
        source_left = (full_col * left_k).sum() == kcol

        v_area = krow * w
        h_area = kcol * h

        # Use the solid rows if they form a real 2D block and the solid columns
        # are only a 1-cell tail, or if the row-block is the larger candidate.
        if krow >= 2 and (kcol < 2 or v_area >= h_area):
            target_rows = bottom_k if source_top else top_k
            mask = target_rows[:, None] * col_has[None, :]
        else:
            target_cols = right_k if source_left else left_k
            mask = row_has[:, None] * target_cols[None, :]

        out[mask.astype(bool)] = colour

    return out.tolist()


def build_onnx_model():
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    F = TensorProto.FLOAT
    I64 = TensorProto.INT64

    init = []
    nodes = []
    used = set()

    def K(name, value, dtype=np.float32):
        if dtype == F:
            np_dtype = np.float32
        elif dtype == I64:
            np_dtype = np.int64
        else:
            np_dtype = dtype

        base = name
        i = 0
        while name in used:
            i += 1
            name = f"{base}_{i}"
        used.add(name)
        init.append(numpy_helper.from_array(np.asarray(value, dtype=np_dtype), name=name))
        return name

    def N(op, inputs, outputs, **attrs):
        nodes.append(helper.make_node(op, inputs, outputs, **attrs))

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])

    zero = K("zero", [0.0], F)
    one = K("one", [1.0], F)
    two = K("two", [2.0], F)
    axis2 = K("axis2", np.array(2, dtype=np.int64), I64)
    axis3 = K("axis3", np.array(3, dtype=np.int64), I64)
    rev30 = K("rev30", np.arange(29, -1, -1, dtype=np.int64), I64)

    # Valid ARC cells have exactly one active input channel; padded cells are all-zero.
    N("ReduceSum", ["input"], ["valid_sum"], axes=[1], keepdims=1)
    N("Greater", ["valid_sum", zero], ["valid_bool"])
    N("Cast", ["valid_bool"], ["valid"], to=F)

    out_channels = []

    for colour in range(1, 10):
        p = f"c{colour}"
        idx = K(f"{p}_idx", [colour], I64)

        # M: mask for this colour, [1,1,30,30]
        N("Gather", ["input", idx], [f"{p}_m"], axis=1)

        N("ReduceSum", [f"{p}_m"], [f"{p}_cnt"], axes=[2, 3], keepdims=1)

        N("ReduceSum", [f"{p}_m"], [f"{p}_row_count"], axes=[3], keepdims=1)
        N("Greater", [f"{p}_row_count", zero], [f"{p}_row_has_b"])
        N("Cast", [f"{p}_row_has_b"], [f"{p}_row_has"], to=F)

        N("ReduceSum", [f"{p}_m"], [f"{p}_col_count"], axes=[2], keepdims=1)
        N("Greater", [f"{p}_col_count", zero], [f"{p}_col_has_b"])
        N("Cast", [f"{p}_col_has_b"], [f"{p}_col_has"], to=F)

        N("ReduceSum", [f"{p}_row_has"], [f"{p}_h"], axes=[2, 3], keepdims=1)
        N("ReduceSum", [f"{p}_col_has"], [f"{p}_w"], axes=[2, 3], keepdims=1)
        N("Mul", [f"{p}_h", f"{p}_w"], [f"{p}_bbox_area"])

        N("Greater", [f"{p}_cnt", zero], [f"{p}_present"])
        N("Equal", [f"{p}_cnt", f"{p}_bbox_area"], [f"{p}_full_tmp"])
        N("And", [f"{p}_present", f"{p}_full_tmp"], [f"{p}_full_b"])
        N("Cast", [f"{p}_full_b"], [f"{p}_full_f"], to=F)

        N("Not", [f"{p}_full_b"], [f"{p}_not_full_tmp"])
        N("And", [f"{p}_present", f"{p}_not_full_tmp"], [f"{p}_nonfull_b"])
        N("Cast", [f"{p}_nonfull_b"], [f"{p}_nonfull_f"], to=F)

        # Full rows/cols inside the colour bounding box.
        N("Equal", [f"{p}_row_count", f"{p}_w"], [f"{p}_full_row_tmp"])
        N("And", [f"{p}_full_row_tmp", f"{p}_row_has_b"], [f"{p}_full_row_b"])
        N("Cast", [f"{p}_full_row_b"], [f"{p}_full_row"], to=F)

        N("Equal", [f"{p}_col_count", f"{p}_h"], [f"{p}_full_col_tmp"])
        N("And", [f"{p}_full_col_tmp", f"{p}_col_has_b"], [f"{p}_full_col_b"])
        N("Cast", [f"{p}_full_col_b"], [f"{p}_full_col"], to=F)

        N("ReduceSum", [f"{p}_full_row"], [f"{p}_krow"], axes=[2, 3], keepdims=1)
        N("ReduceSum", [f"{p}_full_col"], [f"{p}_kcol"], axes=[2, 3], keepdims=1)

        # Row ranks from top/bottom of the bbox.
        N("CumSum", [f"{p}_row_has", axis2], [f"{p}_rank_top"])
        N("Gather", [f"{p}_row_has", rev30], [f"{p}_row_rev"], axis=2)
        N("CumSum", [f"{p}_row_rev", axis2], [f"{p}_rank_bottom_rev"])
        N("Gather", [f"{p}_rank_bottom_rev", rev30], [f"{p}_rank_bottom"], axis=2)

        N("LessOrEqual", [f"{p}_rank_top", f"{p}_krow"], [f"{p}_top_k_tmp"])
        N("And", [f"{p}_top_k_tmp", f"{p}_row_has_b"], [f"{p}_top_k_b"])
        N("Cast", [f"{p}_top_k_b"], [f"{p}_top_k"], to=F)

        N("LessOrEqual", [f"{p}_rank_bottom", f"{p}_krow"], [f"{p}_bottom_k_tmp"])
        N("And", [f"{p}_bottom_k_tmp", f"{p}_row_has_b"], [f"{p}_bottom_k_b"])
        N("Cast", [f"{p}_bottom_k_b"], [f"{p}_bottom_k"], to=F)

        # Column ranks from left/right of the bbox.
        N("CumSum", [f"{p}_col_has", axis3], [f"{p}_rank_left"])
        N("Gather", [f"{p}_col_has", rev30], [f"{p}_col_rev"], axis=3)
        N("CumSum", [f"{p}_col_rev", axis3], [f"{p}_rank_right_rev"])
        N("Gather", [f"{p}_rank_right_rev", rev30], [f"{p}_rank_right"], axis=3)

        N("LessOrEqual", [f"{p}_rank_left", f"{p}_kcol"], [f"{p}_left_k_tmp"])
        N("And", [f"{p}_left_k_tmp", f"{p}_col_has_b"], [f"{p}_left_k_b"])
        N("Cast", [f"{p}_left_k_b"], [f"{p}_left_k"], to=F)

        N("LessOrEqual", [f"{p}_rank_right", f"{p}_kcol"], [f"{p}_right_k_tmp"])
        N("And", [f"{p}_right_k_tmp", f"{p}_col_has_b"], [f"{p}_right_k_b"])
        N("Cast", [f"{p}_right_k_b"], [f"{p}_right_k"], to=F)

        # Is the solid block at the top/left edge of the bbox?
        N("Mul", [f"{p}_full_row", f"{p}_top_k"], [f"{p}_top_full"])
        N("ReduceSum", [f"{p}_top_full"], [f"{p}_top_full_cnt"], axes=[2, 3], keepdims=1)
        N("Equal", [f"{p}_top_full_cnt", f"{p}_krow"], [f"{p}_source_top_b"])

        N("Mul", [f"{p}_full_col", f"{p}_left_k"], [f"{p}_left_full"])
        N("ReduceSum", [f"{p}_left_full"], [f"{p}_left_full_cnt"], axes=[2, 3], keepdims=1)
        N("Equal", [f"{p}_left_full_cnt", f"{p}_kcol"], [f"{p}_source_left_b"])

        # Choose vertical vs horizontal motion.
        N("Mul", [f"{p}_krow", f"{p}_w"], [f"{p}_v_area"])
        N("Mul", [f"{p}_kcol", f"{p}_h"], [f"{p}_h_area"])
        N("GreaterOrEqual", [f"{p}_krow", two], [f"{p}_krow_ge2"])
        N("Less", [f"{p}_kcol", two], [f"{p}_kcol_lt2"])
        N("GreaterOrEqual", [f"{p}_v_area", f"{p}_h_area"], [f"{p}_v_ge_h"])
        N("Or", [f"{p}_kcol_lt2", f"{p}_v_ge_h"], [f"{p}_v_reason"])
        N("And", [f"{p}_krow_ge2", f"{p}_v_reason"], [f"{p}_orient_v_tmp"])
        N("And", [f"{p}_orient_v_tmp", f"{p}_nonfull_b"], [f"{p}_orient_v_b"])

        # If source block is top/left, target goes bottom/right; otherwise top/left.
        N("Where", [f"{p}_source_top_b", f"{p}_bottom_k", f"{p}_top_k"], [f"{p}_target_rows"])
        N("Mul", [f"{p}_target_rows", f"{p}_col_has"], [f"{p}_target_v"])

        N("Where", [f"{p}_source_left_b", f"{p}_right_k", f"{p}_left_k"], [f"{p}_target_cols"])
        N("Mul", [f"{p}_row_has", f"{p}_target_cols"], [f"{p}_target_h"])

        N("Where", [f"{p}_orient_v_b", f"{p}_target_v", f"{p}_target_h"], [f"{p}_target"])
        N("Mul", [f"{p}_target", f"{p}_nonfull_f"], [f"{p}_moving_out"])

        N("Mul", [f"{p}_m", f"{p}_full_f"], [f"{p}_full_out"])
        N("Add", [f"{p}_full_out", f"{p}_moving_out"], [f"{p}_out"])
        out_channels.append(f"{p}_out")

    # Background is valid area minus all nonzero colour channels.
    acc = out_channels[0]
    for i, name in enumerate(out_channels[1:], start=2):
        N("Add", [acc, name], [f"sum_nonzero_{i}"])
        acc = f"sum_nonzero_{i}"

    N("Sub", ["valid", acc], ["out0"])
    N("Concat", ["out0"] + out_channels, ["output"], axis=1)

    graph = helper.make_graph(nodes, "task234", [x], [y], init)
    model = helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 12)],
    )
    onnx.checker.check_model(model)
    return model


model = build_onnx_model()
