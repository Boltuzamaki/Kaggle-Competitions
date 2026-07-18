
import numpy as np

def build_onnx_model():
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    F = TensorProto.FLOAT
    I64 = TensorProto.INT64

    def K(name, value, dtype=np.float32):
        if dtype == I64:
            dtype = np.int64
        elif dtype == F:
            dtype = np.float32
        return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)

    H = 30
    W = 30
    REAL = 10
    offsets = list(range(-9, 10))

    def diag_mask(d):
        m = np.zeros((1, 1, H, W), dtype=np.float32)
        for r in range(REAL):
            c = r + d
            if 0 <= c < REAL:
                m[0, 0, r, c] = 1.0
        return m

    valid = np.zeros((1, 1, H, W), dtype=np.float32)
    valid[:, :, :REAL, :REAL] = 1.0
    zero_chan = np.zeros((1, 1, H, W), dtype=np.float32)

    x = helper.make_tensor_value_info("input", F, [1, 10, H, W])
    y = helper.make_tensor_value_info("output", F, [1, 10, H, W])

    init = [
        K("valid10", valid, F),
        K("zero_chan", zero_chan, F),
        K("zero_scalar", np.array(0.0, dtype=np.float32), F),
        K("half_scalar", np.array(0.5, dtype=np.float32), F),
        K("one_scalar", np.array(1.0, dtype=np.float32), F),
        K("idx5", np.array([5], dtype=np.int64), I64),
    ]
    nodes = []

    # Static diagonal masks for the real 10x10 task area, embedded in the 30x30 tensor.
    for d in offsets:
        init.append(K(f"dm_{d+9}", diag_mask(d), F))

    # Five-marker diagonal support.  The 5 wedge determines the new parallel diagonal(s).
    nodes.append(helper.make_node("Gather", ["input", "idx5"], ["ch5"], axis=1))
    five_present = {}
    for d in offsets:
        tag = d + 9
        nodes.append(helper.make_node("Mul", ["ch5", f"dm_{tag}"], [f"five_on_{tag}"]))
        nodes.append(helper.make_node("ReduceSum", [f"five_on_{tag}"], [f"five_sum_{tag}"], axes=[2, 3], keepdims=0))
        nodes.append(helper.make_node("Greater", [f"five_sum_{tag}", "zero_scalar"], [f"five_gt_{tag}"]))
        nodes.append(helper.make_node("Cast", [f"five_gt_{tag}"], [f"five_p_{tag}"], to=F))
        five_present[d] = f"five_p_{tag}"

    color_outputs = {}
    colored_channels = []

    for c in range(1, 10):
        if c == 5:
            color_outputs[c] = "zero_chan"
            continue

        init.append(K(f"idx{c}", np.array([c], dtype=np.int64), I64))
        nodes.append(helper.make_node("Gather", ["input", f"idx{c}"], [f"ch{c}"], axis=1))

        # Presence of this color on each possible slope-1 diagonal.
        c_present = {}
        for d in offsets:
            tag = d + 9
            nodes.append(helper.make_node("Mul", [f"ch{c}", f"dm_{tag}"], [f"c{c}_on_{tag}"]))
            nodes.append(helper.make_node("ReduceSum", [f"c{c}_on_{tag}"], [f"c{c}_sum_{tag}"], axes=[2, 3], keepdims=0))
            nodes.append(helper.make_node("Greater", [f"c{c}_sum_{tag}", "zero_scalar"], [f"c{c}_gt_{tag}"]))
            nodes.append(helper.make_node("Cast", [f"c{c}_gt_{tag}"], [f"c{c}_p_{tag}"], to=F))
            c_present[d] = f"c{c}_p_{tag}"

        add_masks = [f"ch{c}"]  # Preserve the original colored diagonal and erase all 5 cells.

        for out_d in offsets:
            out_tag = out_d + 9
            cond_terms = []

            # Low-side boundary: if the 5 wedge starts at lo=out_d+2 and the color diagonal is to its right,
            # draw the diagonal at lo-2.
            lo = out_d + 2
            if lo in offsets:
                below = [five_present[d] for d in offsets if d < lo]
                if below:
                    nodes.append(helper.make_node("Sum", below, [f"c{c}_low_below_{out_tag}"]))
                else:
                    nodes.append(helper.make_node("Identity", ["zero_scalar"], [f"c{c}_low_below_{out_tag}"]))
                nodes.append(helper.make_node("Less", [f"c{c}_low_below_{out_tag}", "half_scalar"], [f"c{c}_low_no_below_bool_{out_tag}"]))
                nodes.append(helper.make_node("Cast", [f"c{c}_low_no_below_bool_{out_tag}"], [f"c{c}_low_no_below_{out_tag}"], to=F))

                greater_c = [c_present[d] for d in offsets if d > lo]
                if greater_c:
                    nodes.append(helper.make_node("Sum", greater_c, [f"c{c}_low_cgt_sum_{out_tag}"]))
                else:
                    nodes.append(helper.make_node("Identity", ["zero_scalar"], [f"c{c}_low_cgt_sum_{out_tag}"]))
                nodes.append(helper.make_node("Greater", [f"c{c}_low_cgt_sum_{out_tag}", "zero_scalar"], [f"c{c}_low_cgt_bool_{out_tag}"]))
                nodes.append(helper.make_node("Cast", [f"c{c}_low_cgt_bool_{out_tag}"], [f"c{c}_low_cgt_{out_tag}"], to=F))

                nodes.append(helper.make_node("Mul", [five_present[lo], f"c{c}_low_no_below_{out_tag}"], [f"c{c}_low_a_{out_tag}"]))
                nodes.append(helper.make_node("Mul", [f"c{c}_low_a_{out_tag}", f"c{c}_low_cgt_{out_tag}"], [f"c{c}_low_cond_{out_tag}"]))
                nodes.append(helper.make_node("Mul", [f"c{c}_low_cond_{out_tag}", f"dm_{out_tag}"], [f"c{c}_low_mask_{out_tag}"]))
                cond_terms.append(f"c{c}_low_mask_{out_tag}")

            # High-side boundary: if the 5 wedge ends at hi=out_d-2 and the color diagonal is to its left,
            # draw the diagonal at hi+2.
            hi = out_d - 2
            if hi in offsets:
                above = [five_present[d] for d in offsets if d > hi]
                if above:
                    nodes.append(helper.make_node("Sum", above, [f"c{c}_high_above_{out_tag}"]))
                else:
                    nodes.append(helper.make_node("Identity", ["zero_scalar"], [f"c{c}_high_above_{out_tag}"]))
                nodes.append(helper.make_node("Less", [f"c{c}_high_above_{out_tag}", "half_scalar"], [f"c{c}_high_no_above_bool_{out_tag}"]))
                nodes.append(helper.make_node("Cast", [f"c{c}_high_no_above_bool_{out_tag}"], [f"c{c}_high_no_above_{out_tag}"], to=F))

                lesser_c = [c_present[d] for d in offsets if d < hi]
                if lesser_c:
                    nodes.append(helper.make_node("Sum", lesser_c, [f"c{c}_high_clt_sum_{out_tag}"]))
                else:
                    nodes.append(helper.make_node("Identity", ["zero_scalar"], [f"c{c}_high_clt_sum_{out_tag}"]))
                nodes.append(helper.make_node("Greater", [f"c{c}_high_clt_sum_{out_tag}", "zero_scalar"], [f"c{c}_high_clt_bool_{out_tag}"]))
                nodes.append(helper.make_node("Cast", [f"c{c}_high_clt_bool_{out_tag}"], [f"c{c}_high_clt_{out_tag}"], to=F))

                nodes.append(helper.make_node("Mul", [five_present[hi], f"c{c}_high_no_above_{out_tag}"], [f"c{c}_high_a_{out_tag}"]))
                nodes.append(helper.make_node("Mul", [f"c{c}_high_a_{out_tag}", f"c{c}_high_clt_{out_tag}"], [f"c{c}_high_cond_{out_tag}"]))
                nodes.append(helper.make_node("Mul", [f"c{c}_high_cond_{out_tag}", f"dm_{out_tag}"], [f"c{c}_high_mask_{out_tag}"]))
                cond_terms.append(f"c{c}_high_mask_{out_tag}")

            if cond_terms:
                if len(cond_terms) == 1:
                    add_masks.append(cond_terms[0])
                else:
                    nodes.append(helper.make_node("Sum", cond_terms, [f"c{c}_both_mask_{out_tag}"]))
                    add_masks.append(f"c{c}_both_mask_{out_tag}")

        nodes.append(helper.make_node("Sum", add_masks, [f"out_c{c}"]))
        color_outputs[c] = f"out_c{c}"
        colored_channels.append(f"out_c{c}")

    nodes.append(helper.make_node("Sum", colored_channels, ["colored_any_raw"]))
    # This task has disjoint generated diagonals, so colored_any_raw is already 0/1 in the real 10x10 area.
    nodes.append(helper.make_node("Sub", ["valid10", "colored_any_raw"], ["out_c0"]))

    concat_inputs = ["out_c0"] + [color_outputs[c] for c in range(1, 10)]
    nodes.append(helper.make_node("Concat", concat_inputs, ["output"], axis=1))

    graph = helper.make_graph(nodes, "task260_diagonal_wedge", [x], [y], initializer=init)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model

model = build_onnx_model()
