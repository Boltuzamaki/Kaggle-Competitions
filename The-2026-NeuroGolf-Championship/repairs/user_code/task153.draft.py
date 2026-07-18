
# ARC task153: assemble two separated cropped coloured shapes into their unique 3x3 full overlay.
# Input:  one-hot [1,10,30,30]
# Output: one-hot [1,10,30,30], with the 3x3 answer in the top-left and all padded cells zero.

import numpy as np


def solve_153_numpy(grid):
    """Reference solver for raw ARC grids. Returns a 3x3 grid."""
    a = np.asarray(grid, dtype=np.int64)
    colors = [int(c) for c in np.unique(a) if c != 0]
    objs = []
    for col in colors:
        rr, cc = np.where(a == col)
        r0, r1 = int(rr.min()), int(rr.max())
        c0, c1 = int(cc.min()), int(cc.max())
        mask = (a[r0:r1 + 1, c0:c1 + 1] == col).astype(np.int64)
        objs.append((col, mask))

    def placements(mask):
        h, w = mask.shape
        for ro in range(4 - h):
            for co in range(4 - w):
                canvas = np.zeros((3, 3), dtype=np.int64)
                canvas[ro:ro + h, co:co + w] = mask
                yield ro, co, canvas

    (c1, m1), (c2, m2) = objs
    for _, _, p1 in placements(m1):
        for _, _, p2 in placements(m2):
            if np.all(p1 + p2 == 1):
                out = np.zeros((3, 3), dtype=np.int64)
                out[p1 == 1] = c1
                out[p2 == 1] = c2
                return out.tolist()
    return np.zeros((3, 3), dtype=np.int64).tolist()


def build_onnx_model():
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    F = TensorProto.FLOAT
    I64 = TensorProto.INT64

    init = []
    nodes = []

    def K(name, value, dtype):
        arr = np.asarray(value, dtype=dtype)
        t = numpy_helper.from_array(arr, name=name)
        init.append(t)
        return name

    # Common constants.
    K("zero_f", [0.0], np.float32)
    K("one_f", [1.0], np.float32)
    K("nine_f", [9.0], np.float32)
    K("zero_i", np.array([0], dtype=np.int64).reshape(1, 1, 1, 1), np.int64)
    K("one_i", np.array([1], dtype=np.int64).reshape(1, 1, 1, 1), np.int64)
    K("nine_i", np.array([9], dtype=np.int64).reshape(1, 1, 1, 1), np.int64)
    K("ten_i", np.array([10], dtype=np.int64).reshape(1, 1, 1, 1), np.int64)
    K("zero_idx_3", np.zeros((1, 1, 3, 3), dtype=np.int64), np.int64)
    K("shape100", np.array([100], dtype=np.int64), np.int64)
    K("axes_hw10", np.array([2, 3], dtype=np.int64), np.int64)
    K("axes_h10", np.array([3], dtype=np.int64), np.int64)
    K("axes_w10", np.array([2], dtype=np.int64), np.int64)
    K("axes_hw3", np.array([2, 3], dtype=np.int64), np.int64)
    K("rev10", np.arange(9, -1, -1, dtype=np.int64), np.int64)
    K("pad_to_30", np.array([0, 0, 0, 0, 0, 0, 27, 27], dtype=np.int64), np.int64)
    K("pad_value", [0.0], np.float32)
    K("zero3", np.zeros((1, 1, 3, 3), dtype=np.float32), np.float32)

    # Slice constants for the 10x10 real input area.
    K("s_x10", np.array([0, 0, 0, 0], dtype=np.int64), np.int64)
    K("e_x10", np.array([1, 10, 10, 10], dtype=np.int64), np.int64)
    K("axes4", np.array([0, 1, 2, 3], dtype=np.int64), np.int64)

    # Delta grids for the four allowed placements of a 2/3-sized crop in a 3x3 canvas.
    for ro in [0, 1]:
        for co in [0, 1]:
            dr = np.array([[0 - ro, 0 - ro, 0 - ro],
                           [1 - ro, 1 - ro, 1 - ro],
                           [2 - ro, 2 - ro, 2 - ro]], dtype=np.int64).reshape(1, 1, 3, 3)
            dc = np.array([[0 - co, 1 - co, 2 - co],
                           [0 - co, 1 - co, 2 - co],
                           [0 - co, 1 - co, 2 - co]], dtype=np.int64).reshape(1, 1, 3, 3)
            K(f"dr_{ro}{co}", dr, np.int64)
            K(f"dc_{ro}{co}", dc, np.int64)
            # A row offset of 1 is valid only for height-2 crops; row offset 0 allows height 2 or 3.
            K(f"thr_r_{ro}{co}", np.array([2 - ro], dtype=np.int64).reshape(1, 1, 1, 1), np.int64)
            K(f"thr_c_{ro}{co}", np.array([2 - co], dtype=np.int64).reshape(1, 1, 1, 1), np.int64)

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])

    nodes.append(helper.make_node("Slice", ["input", "s_x10", "e_x10", "axes4"], ["x10"]))

    masks = {}  # (colour, ro, co) -> [1,1,3,3] placed binary mask

    for col in range(1, 10):
        # Extract one colour channel in the 10x10 area.
        K(f"s_c{col}", np.array([0, col, 0, 0], dtype=np.int64), np.int64)
        K(f"e_c{col}", np.array([1, col + 1, 10, 10], dtype=np.int64), np.int64)
        nodes.append(helper.make_node("Slice", ["input", f"s_c{col}", f"e_c{col}", "axes4"], [f"ch{col}"]))

        # Active colour flag and bbox min/max.
        nodes += [
            helper.make_node("ReduceSum", [f"ch{col}", "axes_hw10"], [f"cnt{col}"], keepdims=1),
            helper.make_node("Greater", [f"cnt{col}", "zero_f"], [f"actb{col}"]),
            helper.make_node("Cast", [f"actb{col}"], [f"act{col}"], to=F),

            helper.make_node("ReduceSum", [f"ch{col}", "axes_h10"], [f"row_sum{col}"], keepdims=1),
            helper.make_node("Greater", [f"row_sum{col}", "zero_f"], [f"row_b{col}"]),
            helper.make_node("Cast", [f"row_b{col}"], [f"row_f{col}"], to=F),
            helper.make_node("ArgMax", [f"row_f{col}"], [f"rmin{col}"], axis=2, keepdims=1),
            helper.make_node("Gather", [f"row_f{col}", "rev10"], [f"row_rev{col}"], axis=2),
            helper.make_node("ArgMax", [f"row_rev{col}"], [f"rmax_from_end{col}"], axis=2, keepdims=1),
            helper.make_node("Sub", ["nine_i", f"rmax_from_end{col}"], [f"rmax{col}"]),
            helper.make_node("Sub", [f"rmax{col}", f"rmin{col}"], [f"rspan{col}"]),

            helper.make_node("ReduceSum", [f"ch{col}", "axes_w10"], [f"col_sum{col}"], keepdims=1),
            helper.make_node("Greater", [f"col_sum{col}", "zero_f"], [f"col_b{col}"]),
            helper.make_node("Cast", [f"col_b{col}"], [f"col_f{col}"], to=F),
            helper.make_node("ArgMax", [f"col_f{col}"], [f"cmin{col}"], axis=3, keepdims=1),
            helper.make_node("Gather", [f"col_f{col}", "rev10"], [f"col_rev{col}"], axis=3),
            helper.make_node("ArgMax", [f"col_rev{col}"], [f"cmax_from_end{col}"], axis=3, keepdims=1),
            helper.make_node("Sub", ["nine_i", f"cmax_from_end{col}"], [f"cmax{col}"]),
            helper.make_node("Sub", [f"cmax{col}", f"cmin{col}"], [f"cspan{col}"]),

            helper.make_node("Reshape", [f"ch{col}", "shape100"], [f"flat{col}"]),
        ]

        for ro in [0, 1]:
            for co in [0, 1]:
                p = f"{col}_{ro}{co}"
                # Dynamic 3x3 coordinate grid in input-space for this crop placement.
                nodes += [
                    helper.make_node("LessOrEqual", [f"rspan{col}", f"thr_r_{ro}{co}"], [f"rvb{p}"]),
                    helper.make_node("LessOrEqual", [f"cspan{col}", f"thr_c_{ro}{co}"], [f"cvb{p}"]),
                    helper.make_node("And", [f"rvb{p}", f"cvb{p}"], [f"vb0{p}"]),
                    helper.make_node("Cast", [f"vb0{p}"], [f"valid_size{p}"], to=F),

                    helper.make_node("Add", [f"rmin{col}", f"dr_{ro}{co}"], [f"tr{p}"]),
                    helper.make_node("Add", [f"cmin{col}", f"dc_{ro}{co}"], [f"tc{p}"]),

                    helper.make_node("LessOrEqual", ["zero_i", f"tr{p}"], [f"r_ge0{p}"]),
                    helper.make_node("LessOrEqual", [f"tr{p}", "nine_i"], [f"r_le9{p}"]),
                    helper.make_node("And", [f"r_ge0{p}", f"r_le9{p}"], [f"r_ok{p}"]),
                    helper.make_node("LessOrEqual", ["zero_i", f"tc{p}"], [f"c_ge0{p}"]),
                    helper.make_node("LessOrEqual", [f"tc{p}", "nine_i"], [f"c_le9{p}"]),
                    helper.make_node("And", [f"c_ge0{p}", f"c_le9{p}"], [f"c_ok{p}"]),
                    helper.make_node("And", [f"r_ok{p}", f"c_ok{p}"], [f"coord_ok_b{p}"]),
                    helper.make_node("Cast", [f"coord_ok_b{p}"], [f"coord_ok{p}"], to=F),

                    helper.make_node("Mul", [f"tr{p}", "ten_i"], [f"tr10{p}"]),
                    helper.make_node("Add", [f"tr10{p}", f"tc{p}"], [f"lin{p}"]),
                    helper.make_node("Where", [f"coord_ok_b{p}", f"lin{p}", "zero_idx_3"], [f"lin_safe{p}"]),
                    helper.make_node("Gather", [f"flat{col}", f"lin_safe{p}"], [f"sample_raw{p}"], axis=0),
                    helper.make_node("Mul", [f"sample_raw{p}", f"coord_ok{p}"], [f"sample_coord{p}"]),
                    helper.make_node("Mul", [f"sample_coord{p}", f"valid_size{p}"], [f"sample_size{p}"]),
                    helper.make_node("Mul", [f"sample_size{p}", f"act{col}"], [f"mask{p}"]),
                ]
                masks[(col, ro, co)] = f"mask{p}"

    # Try every unordered colour pair and every allowed placement pair.
    channel_terms = {c: [] for c in range(10)}
    candidate_id = 0
    for c1 in range(1, 10):
        for c2 in range(c1 + 1, 10):
            for ro1 in [0, 1]:
                for co1 in [0, 1]:
                    for ro2 in [0, 1]:
                        for co2 in [0, 1]:
                            m1 = masks[(c1, ro1, co1)]
                            m2 = masks[(c2, ro2, co2)]
                            cid = f"cand{candidate_id}"
                            candidate_id += 1
                            nodes += [
                                helper.make_node("Add", [m1, m2], [f"sum_{cid}"]),
                                helper.make_node("Equal", [f"sum_{cid}", "one_f"], [f"eq_{cid}"]),
                                helper.make_node("Cast", [f"eq_{cid}"], [f"eqf_{cid}"], to=F),
                                helper.make_node("ReduceSum", [f"eqf_{cid}", "axes_hw3"], [f"cover_{cid}"], keepdims=1),
                                helper.make_node("Equal", [f"cover_{cid}", "nine_f"], [f"validb_{cid}"]),
                                helper.make_node("Cast", [f"validb_{cid}"], [f"valid_{cid}"], to=F),
                                helper.make_node("Mul", [m1, f"valid_{cid}"], [f"term_{cid}_a"]),
                                helper.make_node("Mul", [m2, f"valid_{cid}"], [f"term_{cid}_b"]),
                            ]
                            channel_terms[c1].append(f"term_{cid}_a")
                            channel_terms[c2].append(f"term_{cid}_b")

    out_channels = []
    for col in range(10):
        if col == 0 or not channel_terms[col]:
            out_channels.append("zero3")
        else:
            name = f"out_ch{col}"
            nodes.append(helper.make_node("Sum", channel_terms[col], [name]))
            out_channels.append(name)

    nodes += [
        helper.make_node("Concat", out_channels, ["out3"], axis=1),
        helper.make_node("Pad", ["out3", "pad_to_30", "pad_value"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task153", [x], [y], init)
    model = helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


model = build_onnx_model()
