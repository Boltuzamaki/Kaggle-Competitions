# ARC task112: complete the colour-2 shape by reflecting it around the 2x2 colour-3 anchor.
# Input:  one-hot [1,10,30,30]
# Output: one-hot [1,10,30,30]
#
# Clean checker format: no JSON loading, no base64, no __main__; global `model` is built directly.

import numpy as np


def build_onnx_model():
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    F = TensorProto.FLOAT
    I64 = TensorProto.INT64

    def K(name, value, dtype):
        return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])

    init = [
        K("axesC", [1], np.int64),
        K("axesH", [2], np.int64),
        K("axesW", [3], np.int64),
        K("s2", [2], np.int64),
        K("e3", [3], np.int64),
        K("s3", [3], np.int64),
        K("e4", [4], np.int64),
        K("zeroF", [0.0], np.float32),
        K("oneF", [1.0], np.float32),
        K("zeroI", [0], np.int64),
        K("oneI", [1], np.int64),
        K("maxI", [29], np.int64),
        K("rowGrid", np.arange(30, dtype=np.int64).reshape(1, 1, 30, 1), np.int64),
        K("colGrid", np.arange(30, dtype=np.int64).reshape(1, 1, 1, 30), np.int64),
        K("zeroRows", np.zeros((1, 1, 30, 1), dtype=np.int64), np.int64),
        K("zeroCols", np.zeros((1, 1, 1, 30), dtype=np.int64), np.int64),
    ]

    nodes = []

    # Extract colour-2 shape and colour-3 anchor masks.
    nodes += [
        helper.make_node("Slice", ["input", "s2", "e3", "axesC"], ["x2"]),
        helper.make_node("Slice", ["input", "s3", "e4", "axesC"], ["x3"]),

        # valid = real ARC cells. Padding outside the true grid has all channels zero.
        helper.make_node("ReduceSum", ["input", "axesC"], ["validSum"], keepdims=1),
        helper.make_node("Greater", ["validSum", "zeroF"], ["validBool"]),
        helper.make_node("Cast", ["validBool"], ["valid"], to=F),

        # Find top-left row/col of the 2x2 colour-3 anchor.
        helper.make_node("ReduceSum", ["x3", "axesW"], ["row3sum"], keepdims=1),
        helper.make_node("Greater", ["row3sum", "zeroF"], ["row3bool"]),
        helper.make_node("Cast", ["row3bool"], ["row3"], to=F),
        helper.make_node("ArgMax", ["row3"], ["r0"], axis=2, keepdims=1),

        helper.make_node("ReduceSum", ["x3", "axesH"], ["col3sum"], keepdims=1),
        helper.make_node("Greater", ["col3sum", "zeroF"], ["col3bool"]),
        helper.make_node("Cast", ["col3bool"], ["col3"], to=F),
        helper.make_node("ArgMax", ["col3"], ["c0"], axis=3, keepdims=1),

        # Reflection axes are between the two marker rows/cols:
        # reflected_r = 2*r0 + 1 - r, reflected_c = 2*c0 + 1 - c
        helper.make_node("Add", ["r0", "r0"], ["r0x2"]),
        helper.make_node("Add", ["r0x2", "oneI"], ["axisR"]),
        helper.make_node("Add", ["c0", "c0"], ["c0x2"]),
        helper.make_node("Add", ["c0x2", "oneI"], ["axisC"]),

        helper.make_node("Sub", ["axisR", "rowGrid"], ["mirRBase"]),
        helper.make_node("Add", ["mirRBase", "zeroCols"], ["mirR"]),
        helper.make_node("Sub", ["axisC", "colGrid"], ["mirCBase"]),
        helper.make_node("Add", ["mirCBase", "zeroRows"], ["mirC"]),

        # Build in-bounds masks and clipped indices for GatherElements.
        helper.make_node("Less", ["mirR", "zeroI"], ["mirR_lt0"]),
        helper.make_node("Not", ["mirR_lt0"], ["mirR_ge0"]),
        helper.make_node("Greater", ["mirR", "maxI"], ["mirR_gt29"]),
        helper.make_node("Not", ["mirR_gt29"], ["mirR_le29"]),
        helper.make_node("And", ["mirR_ge0", "mirR_le29"], ["mirR_ok"]),
        helper.make_node("Cast", ["mirR_ok"], ["mirR_okF"], to=F),
        helper.make_node("Where", ["mirR_lt0", "zeroI", "mirR"], ["mirR_nonneg"]),
        helper.make_node("Greater", ["mirR_nonneg", "maxI"], ["mirR_clip_hi"]),
        helper.make_node("Where", ["mirR_clip_hi", "maxI", "mirR_nonneg"], ["mirRclip"]),

        helper.make_node("Less", ["mirC", "zeroI"], ["mirC_lt0"]),
        helper.make_node("Not", ["mirC_lt0"], ["mirC_ge0"]),
        helper.make_node("Greater", ["mirC", "maxI"], ["mirC_gt29"]),
        helper.make_node("Not", ["mirC_gt29"], ["mirC_le29"]),
        helper.make_node("And", ["mirC_ge0", "mirC_le29"], ["mirC_ok"]),
        helper.make_node("Cast", ["mirC_ok"], ["mirC_okF"], to=F),
        helper.make_node("Where", ["mirC_lt0", "zeroI", "mirC"], ["mirC_nonneg"]),
        helper.make_node("Greater", ["mirC_nonneg", "maxI"], ["mirC_clip_hi"]),
        helper.make_node("Where", ["mirC_clip_hi", "maxI", "mirC_nonneg"], ["mirCclip"]),

        # Reflect colour-2 pixels vertically, horizontally, and both.
        helper.make_node("GatherElements", ["x2", "mirRclip"], ["x2Vraw"], axis=2),
        helper.make_node("Mul", ["x2Vraw", "mirR_okF"], ["x2V"]),

        helper.make_node("GatherElements", ["x2", "mirCclip"], ["x2Hraw"], axis=3),
        helper.make_node("Mul", ["x2Hraw", "mirC_okF"], ["x2H"]),

        helper.make_node("GatherElements", ["x2Hraw", "mirRclip"], ["x2Braw0"], axis=2),
        helper.make_node("Mul", ["mirR_okF", "mirC_okF"], ["mirBoth_okF"]),
        helper.make_node("Mul", ["x2Braw0", "mirBoth_okF"], ["x2B"]),

        # Union of original + three reflections.
        helper.make_node("Add", ["x2", "x2V"], ["u0"]),
        helper.make_node("Add", ["x2H", "x2B"], ["u1"]),
        helper.make_node("Add", ["u0", "u1"], ["u"]),
        helper.make_node("Greater", ["u", "zeroF"], ["uBool"]),
        helper.make_node("Cast", ["uBool"], ["uOH"], to=F),

        # Preserve the colour-3 anchor; never overwrite it with colour 2.
        helper.make_node("Sub", ["oneF", "x3"], ["not3"]),
        helper.make_node("Mul", ["uOH", "not3"], ["out2a"]),
        helper.make_node("Mul", ["out2a", "valid"], ["out2"]),

        helper.make_node("Mul", ["x3", "valid"], ["out3"]),

        # Background channel for real cells only; padding stays all-zero.
        helper.make_node("Add", ["out2", "out3"], ["occupied"]),
        helper.make_node("Sub", ["valid", "occupied"], ["out0"]),

        helper.make_node("Mul", ["out0", "zeroF"], ["zeroCh"]),
        helper.make_node(
            "Concat",
            ["out0", "zeroCh", "out2", "out3", "zeroCh", "zeroCh", "zeroCh", "zeroCh", "zeroCh", "zeroCh"],
            ["output"],
            axis=1,
        ),
    ]

    graph = helper.make_graph(nodes, "task112", [x], [y], init)
    model = helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


model = build_onnx_model()
