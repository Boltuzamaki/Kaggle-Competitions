
import numpy as np

def build_onnx_model():
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    F = TensorProto.FLOAT
    I64 = TensorProto.INT64
    B = TensorProto.BOOL

    def K(name, value, dtype=np.float32):
        if dtype == I64:
            arr = np.asarray(value, dtype=np.int64)
        elif dtype == F:
            arr = np.asarray(value, dtype=np.float32)
        elif dtype == B:
            arr = np.asarray(value, dtype=np.bool_)
        else:
            arr = np.asarray(value, dtype=dtype)
        return numpy_helper.from_array(arr, name=name)

    H = 30
    W = 30

    # Candidate scaled placements for the miniature recipe.
    # Each candidate is generic: scale in {2,3,4}, offset y/x in 0..29.
    # Values 0..14 index a 3x5 recipe cell; 15 is the zero sentinel.
    cand_maps = []
    cand_scales = []
    for s in (2, 3, 4):
        for oy0 in range(H):
            for ox0 in range(W):
                m = np.full((H, W), 15, dtype=np.int64)
                for y in range(H):
                    dy = y - oy0
                    if dy < 0:
                        continue
                    ry = dy // s
                    if ry < 0 or ry >= 3:
                        continue
                    for x in range(W):
                        dx = x - ox0
                        if dx < 0:
                            continue
                        rx = dx // s
                        if 0 <= rx < 5:
                            m[y, x] = ry * 5 + rx
                cand_maps.append(m)
                cand_scales.append(s)

    cand_maps = np.stack(cand_maps, axis=0).astype(np.int64)
    cand_scales = np.asarray(cand_scales, dtype=np.float32)
    C = int(cand_scales.shape[0])

    cand_s2 = (cand_scales == 2).astype(np.float32)
    cand_s3 = (cand_scales == 3).astype(np.float32)
    cand_s4 = (cand_scales == 4).astype(np.float32)

    row_grid = np.arange(H, dtype=np.int64)[:, None].repeat(W, axis=1)
    col_grid = np.arange(W, dtype=np.int64)[None, :].repeat(H, axis=0)
    row_idx = np.arange(H, dtype=np.float32)
    col_idx = np.arange(W, dtype=np.float32)

    x = helper.make_tensor_value_info("input", F, [1, 10, H, W])
    y = helper.make_tensor_value_info("output", F, [1, 10, H, W])

    init = [
        K("ch4", np.array([4], np.int64), I64),
        K("ch_non04", np.array([1,2,3,5,6,7,8,9], np.int64), I64),
        K("zero_i", np.array([0], np.int64), I64),
        K("zero_f", np.array([0.0], np.float32), F),
        K("one_f", np.array([1.0], np.float32), F),
        K("big_neg", np.array([-100000.0], np.float32), F),
        K("twenty9_i", np.array([29], np.int64), I64),
        K("row_grid", row_grid, I64),
        K("col_grid", col_grid, I64),
        K("row_idx", row_idx, F),
        K("col_idx", col_idx, F),
        K("rec_rows", np.arange(3, dtype=np.int64)[:, None].repeat(5, axis=1), I64),
        K("rec_cols", np.arange(5, dtype=np.int64)[None, :].repeat(3, axis=0), I64),
        K("shape_15", np.array([15], np.int64), I64),
        K("cand_maps", cand_maps, I64),
        K("cand_ids", np.arange(C, dtype=np.int64), I64),
        K("cand_s2", cand_s2, F),
        K("cand_s3", cand_s3, F),
        K("cand_s4", cand_s4, F),
        K("cand_mask_shape", np.array([C,1,1], np.int64), I64),
        K("one_const", np.array([1.0], np.float32), F),
        K("k3w", np.ones((8,1,3,3), dtype=np.float32), F),
        K("k4w", np.ones((8,1,4,4), dtype=np.float32), F),
        K("nine_f", np.array([9.0], np.float32), F),
        K("sixteen_f", np.array([16.0], np.float32), F),
    ]
    # color constants for final manual one-hot
    for c in range(10):
        init.append(K("color_%d" % c, np.array([c], np.int64), I64))

    nodes = []

    # Raw color index map.
    nodes += [
        helper.make_node("ArgMax", ["input"], ["idx_b"], axis=1, keepdims=0),
        helper.make_node("Squeeze", ["idx_b"], ["idx"], axes=[0]),

        helper.make_node("Gather", ["input", "ch4"], ["ch4_b"], axis=1),
        helper.make_node("Squeeze", ["ch4_b"], ["mask4"], axes=[0,1]),

        helper.make_node("ReduceMax", ["mask4"], ["row_has4"], axes=[1], keepdims=0),
        helper.make_node("ReduceMax", ["mask4"], ["col_has4"], axes=[0], keepdims=0),
        helper.make_node("ArgMax", ["row_has4"], ["top"], axis=0, keepdims=1),
        helper.make_node("Mul", ["row_has4", "row_idx"], ["row_pos"]),
        helper.make_node("ArgMax", ["row_pos"], ["bottom"], axis=0, keepdims=1),
        helper.make_node("ArgMax", ["col_has4"], ["left"], axis=0, keepdims=1),
        helper.make_node("Mul", ["col_has4", "col_idx"], ["col_pos"]),
        helper.make_node("ArgMax", ["col_pos"], ["right"], axis=0, keepdims=1),

        helper.make_node("Sub", ["bottom", "top"], ["frame_hm1"]),
        helper.make_node("Sub", ["right", "left"], ["frame_wm1"]),
    ]

    # Crop frame to the top-left of the output by dynamic GatherND.
    nodes += [
        helper.make_node("Add", ["row_grid", "top"], ["src_rows_raw"]),
        helper.make_node("Add", ["col_grid", "left"], ["src_cols_raw"]),
        helper.make_node("Min", ["src_rows_raw", "twenty9_i"], ["src_rows"]),
        helper.make_node("Min", ["src_cols_raw", "twenty9_i"], ["src_cols"]),
        helper.make_node("Unsqueeze", ["src_rows"], ["src_rows_u"], axes=[2]),
        helper.make_node("Unsqueeze", ["src_cols"], ["src_cols_u"], axes=[2]),
        helper.make_node("Concat", ["src_rows_u", "src_cols_u"], ["crop_gather_idx"], axis=2),
        helper.make_node("GatherND", ["idx", "crop_gather_idx"], ["crop_idx_gathered"]),

        helper.make_node("LessOrEqual", ["row_grid", "frame_hm1"], ["valid_y"]),
        helper.make_node("LessOrEqual", ["col_grid", "frame_wm1"], ["valid_x"]),
        helper.make_node("And", ["valid_y", "valid_x"], ["valid_crop"]),
        helper.make_node("Where", ["valid_crop", "crop_idx_gathered", "zero_i"], ["crop_idx"]),
    ]

    # Infer scale: 4 if any same-color 4x4 solid block exists, else 3 if any 3x3, else 2.
    nodes += [
        helper.make_node("Gather", ["input", "ch_non04"], ["non04_b"], axis=1),
        helper.make_node("Conv", ["non04_b", "k4w"], ["conv4"], group=8),
        helper.make_node("Equal", ["conv4", "sixteen_f"], ["eq4"]),
        helper.make_node("Cast", ["eq4"], ["eq4f"], to=F),
        helper.make_node("ReduceMax", ["eq4f"], ["has4"], axes=[1,2,3], keepdims=0),
        helper.make_node("Conv", ["non04_b", "k3w"], ["conv3"], group=8),
        helper.make_node("Equal", ["conv3", "nine_f"], ["eq3"]),
        helper.make_node("Cast", ["eq3"], ["eq3f"], to=F),
        helper.make_node("ReduceMax", ["eq3f"], ["has3_raw"], axes=[1,2,3], keepdims=0),
        helper.make_node("Sub", ["one_f", "has4"], ["not_has4"]),
        helper.make_node("Mul", ["has3_raw", "not_has4"], ["has3"]),
        helper.make_node("Sub", ["one_f", "has3"], ["not_has3"]),
        helper.make_node("Mul", ["not_has4", "not_has3"], ["has2"]),
    ]

    # Miniature recipe: all nonzero non-4 pixels below the bottom frame marker.
    nodes += [
        helper.make_node("ReduceMax", ["non04_b"], ["non04_mask_b"], axes=[1], keepdims=0),
        helper.make_node("Squeeze", ["non04_mask_b"], ["non04_mask"], axes=[0]),
        helper.make_node("Greater", ["non04_mask", "zero_f"], ["non04_bool"]),
        helper.make_node("Greater", ["row_grid", "bottom"], ["below_bottom"]),
        helper.make_node("And", ["non04_bool", "below_bottom"], ["recipe_bool"]),
        helper.make_node("Cast", ["recipe_bool"], ["recipe_f"], to=F),
        helper.make_node("ReduceMax", ["recipe_f"], ["recipe_row_has"], axes=[1], keepdims=0),
        helper.make_node("ReduceMax", ["recipe_f"], ["recipe_col_has"], axes=[0], keepdims=0),
        helper.make_node("ArgMax", ["recipe_row_has"], ["recipe_top"], axis=0, keepdims=1),
        helper.make_node("ArgMax", ["recipe_col_has"], ["recipe_left"], axis=0, keepdims=1),

        helper.make_node("Add", ["rec_rows", "recipe_top"], ["rec_src_r_raw"]),
        helper.make_node("Add", ["rec_cols", "recipe_left"], ["rec_src_c_raw"]),
        helper.make_node("Min", ["rec_src_r_raw", "twenty9_i"], ["rec_src_r"]),
        helper.make_node("Min", ["rec_src_c_raw", "twenty9_i"], ["rec_src_c"]),
        helper.make_node("Unsqueeze", ["rec_src_r"], ["rec_src_r_u"], axes=[2]),
        helper.make_node("Unsqueeze", ["rec_src_c"], ["rec_src_c_u"], axes=[2]),
        helper.make_node("Concat", ["rec_src_r_u", "rec_src_c_u"], ["recipe_gather_idx"], axis=2),
        helper.make_node("GatherND", ["idx", "recipe_gather_idx"], ["recipe_vals"]),
        helper.make_node("Reshape", ["recipe_vals", "shape_15"], ["recipe_flat"]),
        helper.make_node("Concat", ["recipe_flat", "zero_i"], ["recipe_ext"], axis=0),
    ]

    # Candidate scoring and selection.
    nodes += [
        helper.make_node("Gather", ["recipe_ext", "cand_maps"], ["cand_color"]),
        helper.make_node("Unsqueeze", ["crop_idx"], ["crop_idx_b"], axes=[0]),
        helper.make_node("Equal", ["cand_color", "crop_idx_b"], ["cand_same0"]),
        helper.make_node("Greater", ["cand_color", "zero_i"], ["cand_paint_bool"]),
        helper.make_node("Greater", ["crop_idx_b", "zero_i"], ["crop_nonzero_bool"]),
        helper.make_node("And", ["cand_same0", "cand_paint_bool"], ["cand_same1"]),
        helper.make_node("And", ["cand_same1", "crop_nonzero_bool"], ["cand_same"]),
        helper.make_node("Cast", ["cand_same"], ["cand_same_f"], to=F),
        helper.make_node("ReduceSum", ["cand_same_f"], ["score_raw"], axes=[1,2], keepdims=0),

        helper.make_node("Mul", ["cand_s4", "has4"], ["scale4_part"]),
        helper.make_node("Mul", ["cand_s3", "has3"], ["scale3_part"]),
        helper.make_node("Mul", ["cand_s2", "has2"], ["scale2_part"]),
        helper.make_node("Add", ["scale4_part", "scale3_part"], ["scale43"]),
        helper.make_node("Add", ["scale43", "scale2_part"], ["scale_match"]),
        helper.make_node("Sub", ["scale_match", "one_const"], ["scale_miss_neg"]),
        helper.make_node("Mul", ["scale_miss_neg", "big_neg"], ["scale_penalty_pos"]),  # (0-1)*(-100000)=+100000 for miss
        helper.make_node("Sub", ["score_raw", "scale_penalty_pos"], ["score"]),

        helper.make_node("ArgMax", ["score"], ["best_idx"], axis=0, keepdims=1),
        helper.make_node("Equal", ["cand_ids", "best_idx"], ["best_bool"]),
        helper.make_node("Cast", ["best_bool"], ["best_f"], to=F),
        helper.make_node("Reshape", ["best_f", "cand_mask_shape"], ["best_mask"]),
        helper.make_node("Cast", ["cand_color"], ["cand_color_f"], to=F),
        helper.make_node("Mul", ["cand_color_f", "best_mask"], ["chosen_color_f_all"]),
        helper.make_node("ReduceSum", ["chosen_color_f_all"], ["paint_f"], axes=[0], keepdims=0),
        helper.make_node("Cast", ["paint_f"], ["paint_idx"], to=I64),

        helper.make_node("Greater", ["paint_idx", "zero_i"], ["paint_nonzero"]),
        helper.make_node("Where", ["paint_nonzero", "paint_idx", "crop_idx"], ["merged_idx"]),
        helper.make_node("Where", ["valid_crop", "merged_idx", "zero_i"], ["merged_valid_idx"]),
    ]

    # Manual one-hot so cells outside the cropped output remain all-zero.
    ch_names = []
    for c in range(10):
        eqn = "eq_color_%d" % c
        andn = "valid_color_%d" % c
        fln = "color_f_%d" % c
        unsq = "color_u_%d" % c
        nodes += [
            helper.make_node("Equal", ["merged_valid_idx", "color_%d" % c], [eqn]),
            helper.make_node("And", [eqn, "valid_crop"], [andn]),
            helper.make_node("Cast", [andn], [fln], to=F),
            helper.make_node("Unsqueeze", [fln], [unsq], axes=[0]),
        ]
        ch_names.append(unsq)
    nodes += [
        helper.make_node("Concat", ch_names, ["out_chw"], axis=0),
        helper.make_node("Unsqueeze", ["out_chw"], ["output"], axes=[0]),
    ]

    graph = helper.make_graph(nodes, "task209_rule_graph", [x], [y], initializer=init)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 12)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model

model = build_onnx_model()
