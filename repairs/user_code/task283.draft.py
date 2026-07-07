# ARC task283: recolor each solid color-5 rectangle as a framed tile.
# Input/output are one-hot tensors [1,10,30,30].
# Clean rule graph: no JSON loading, no base64, no fingerprints, no example selector.

import numpy as np


def solve_283_numpy(grid):
    """Reference rule on integer ARC grids."""
    a = np.asarray(grid, dtype=np.int64)
    m = a == 5

    up = np.zeros_like(m)
    up[1:, :] = m[:-1, :]
    down = np.zeros_like(m)
    down[:-1, :] = m[1:, :]
    left = np.zeros_like(m)
    left[:, 1:] = m[:, :-1]
    right = np.zeros_like(m)
    right[:, :-1] = m[:, 1:]

    n = up.astype(np.int64) + down.astype(np.int64) + left.astype(np.int64) + right.astype(np.int64)

    out = np.zeros_like(a)
    out[m & (n <= 2)] = 1   # rectangle corners
    out[m & (n == 3)] = 4   # rectangle border/edge
    out[m & (n >= 4)] = 2   # rectangle interior
    return out


def build_onnx_model():
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    F = TensorProto.FLOAT

    def K(name, value, dtype=np.float32):
        return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])

    # Slice channel 5 from one-hot input.
    starts5 = K("starts5", [5], np.int64)
    ends5 = K("ends5", [6], np.int64)
    axes_c = K("axes_c", [1], np.int64)
    steps1 = K("steps1", [1], np.int64)

    # Slice background channel 0.
    starts0 = K("starts0", [0], np.int64)
    ends0 = K("ends0", [1], np.int64)

    # Cross-neighbor convolution: counts orthogonal color-5 neighbors.
    w = np.zeros((1, 1, 3, 3), dtype=np.float32)
    w[0, 0, 0, 1] = 1.0  # up
    w[0, 0, 1, 0] = 1.0  # left
    w[0, 0, 1, 2] = 1.0  # right
    w[0, 0, 2, 1] = 1.0  # down

    init = [
        starts5, ends5, axes_c, steps1, starts0, ends0,
        K("cross_w", w),
        K("two_half", [2.5], np.float32),
        K("three_half", [3.5], np.float32),
        K("zero", [0.0], np.float32),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "starts5", "ends5", "axes_c", "steps1"], ["is5"]),
        helper.make_node("Slice", ["input", "starts0", "ends0", "axes_c", "steps1"], ["bg"]),

        helper.make_node("Conv", ["is5", "cross_w"], ["nbr"], pads=[1, 1, 1, 1]),

        # Corners have exactly two orthogonal 5-neighbors in generated rectangles.
        helper.make_node("Less", ["nbr", "two_half"], ["corner_bool"]),
        helper.make_node("Cast", ["corner_bool"], ["corner_mask"], to=F),
        helper.make_node("Mul", ["corner_mask", "is5"], ["corner"]),

        # Interiors have four orthogonal 5-neighbors.
        helper.make_node("Greater", ["nbr", "three_half"], ["inside_bool"]),
        helper.make_node("Cast", ["inside_bool"], ["inside_mask"], to=F),
        helper.make_node("Mul", ["inside_mask", "is5"], ["inside"]),

        # Remaining 5-cells are non-corner boundary edges.
        helper.make_node("Sub", ["is5", "corner"], ["not_corner"]),
        helper.make_node("Sub", ["not_corner", "inside"], ["edge"]),

        # Reusable zero channel.
        helper.make_node("Mul", ["is5", "zero"], ["blank"]),

        # Channel order: 0 background, 1 corners, 2 inside, 3 blank, 4 edge, 5..9 blank.
        helper.make_node(
            "Concat",
            ["bg", "corner", "inside", "blank", "edge", "blank", "blank", "blank", "blank", "blank"],
            ["output"],
            axis=1,
        ),
    ]

    graph = helper.make_graph(nodes, "task283_rule", [x], [y], init)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 12)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


model = build_onnx_model()
