# ARC task202: propagate each zero defect through its own solid colour band.
# Clean model file: no JSON loading, no base64, no input fingerprints.
# Input convention: padded one-hot [1,10,42,38], where invalid padding has all channels 0.
# Output: padded one-hot [1,10,42,38]; invalid padding remains all-zero.

import numpy as np


def solve_202_numpy(grid):
    """Reference solver for raw ARC grids."""
    a = np.asarray(grid, dtype=np.int64)
    out = a.copy()
    H, W = a.shape
    colors = [int(c) for c in np.unique(a) if int(c) != 0]

    # Vertical bands span the full height; horizontal bands span the full width.
    vertical_bands = any(np.any(a == c) and np.where(a == c)[0].min() == 0 and np.where(a == c)[0].max() == H - 1 for c in colors)

    for c in colors:
        ys, xs = np.where(a == c)
        if len(ys) == 0:
            continue
        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())
        holes = np.argwhere(a[r0:r1 + 1, c0:c1 + 1] == 0)
        if vertical_bands:
            # Vertical colour stripes: a defect clears the whole row only within that stripe.
            for rr, _ in holes:
                out[r0 + int(rr), c0:c1 + 1] = 0
        else:
            # Horizontal colour stripes: a defect clears the whole column only within that band.
            for _, cc in holes:
                out[r0:r1 + 1, c0 + int(cc)] = 0
    return out.tolist()


def build_onnx_model():
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    F = TensorProto.FLOAT
    I64 = TensorProto.INT64

    def K(name, value, dtype=np.float32):
        # numpy_helper.from_array needs a NumPy dtype, not a TensorProto enum.
        if dtype == I64:
            dtype = np.int64
        elif dtype == F:
            dtype = np.float32
        return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)

    x = helper.make_tensor_value_info("input", F, [1, 10, 42, 38])
    y = helper.make_tensor_value_info("output", F, [1, 10, 42, 38])

    init = [
        K("zero", [0.0]),
        K("half", [0.5]),
        K("one", [1.0]),
        K("s0", [0], I64),
        K("s1", [1], I64),
        K("axisC", [1], I64),
        # Channel masks.
        K("nonzero_color", np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32).reshape(1, 10, 1, 1)),
        K("zero_color", np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32).reshape(1, 10, 1, 1)),
    ]

    nodes = []

    # Valid cells: in padded inputs, real cells have exactly one active channel; invalid padding has all channels zero.
    nodes += [
        helper.make_node("ReduceSum", ["input"], ["sumC"], axes=[1], keepdims=1),       # [1,1,H,W]
        helper.make_node("Greater", ["sumC", "half"], ["validBool"]),
        helper.make_node("Cast", ["validBool"], ["valid"], to=F),
        helper.make_node("ReduceMax", ["valid"], ["validRows"], axes=[3], keepdims=1), # [1,1,H,1]
        helper.make_node("ReduceMax", ["valid"], ["validCols"], axes=[2], keepdims=1), # [1,1,1,W]
    ]

    # Nonzero colour channels only.  Zero channel contains the defect cells in real area.
    nodes += [
        helper.make_node("Mul", ["input", "nonzero_color"], ["nz"]),
        helper.make_node("Slice", ["input", "s0", "s1", "axisC"], ["zeroMaskRaw"]),
        helper.make_node("Mul", ["zeroMaskRaw", "valid"], ["zeroMask"]),
    ]

    # Orientation detection must ignore invalid padded rows/cols.
    # Vertical-band case: some nonzero colour appears in every valid row.
    nodes += [
        helper.make_node("ReduceMax", ["nz"], ["rowAny"], axes=[3], keepdims=1),
        helper.make_node("Greater", ["rowAny", "half"], ["rowPresentBool"]),
        helper.make_node("Cast", ["rowPresentBool"], ["rowPresent"], to=F),
        helper.make_node("Sub", ["one", "rowPresent"], ["rowMissing"]),
        helper.make_node("Mul", ["rowMissing", "validRows"], ["missingValidRows"]),
        helper.make_node("ReduceMax", ["missingValidRows"], ["anyMissingRow"], axes=[2], keepdims=1),
        helper.make_node("Sub", ["one", "anyMissingRow"], ["fullValidRowsPerColor"]),
        helper.make_node("Mul", ["fullValidRowsPerColor", "nonzero_color"], ["fullValidRowsNZ"]),
        helper.make_node("ReduceMax", ["fullValidRowsNZ"], ["fullHAny"], axes=[1], keepdims=1),
        helper.make_node("Greater", ["fullHAny", "half"], ["verticalBool"]),
    ]

    # Vertical colour stripes: columns owned by each colour; each zero in those columns clears its row within that stripe.
    nodes += [
        helper.make_node("ReduceMax", ["nz"], ["colMask"], axes=[2], keepdims=1),       # [1,10,1,W]
        helper.make_node("Mul", ["zeroMask", "colMask"], ["zeroInCols"]),
        helper.make_node("ReduceMax", ["zeroInCols"], ["rowHole"], axes=[3], keepdims=1), # [1,10,H,1]
        helper.make_node("Mul", ["nz", "rowHole"], ["clearVert"]),
    ]

    # Horizontal colour bands: rows owned by each colour; each zero in those rows clears its column within that band.
    nodes += [
        helper.make_node("ReduceMax", ["nz"], ["rowMask"], axes=[3], keepdims=1),       # [1,10,H,1]
        helper.make_node("Mul", ["zeroMask", "rowMask"], ["zeroInRows"]),
        helper.make_node("ReduceMax", ["zeroInRows"], ["colHole"], axes=[2], keepdims=1), # [1,10,1,W]
        helper.make_node("Mul", ["nz", "colHole"], ["clearHoriz"]),
    ]

    # Select orientation, clear nonzero channels, and write zero channel at cleared real cells.
    nodes += [
        helper.make_node("Where", ["verticalBool", "clearVert", "clearHoriz"], ["clearSel"]),
        helper.make_node("Sub", ["one", "clearSel"], ["keepMask"]),
        helper.make_node("Mul", ["input", "keepMask"], ["kept"]),
        helper.make_node("ReduceMax", ["clearSel"], ["clearAny"], axes=[1], keepdims=1),
        helper.make_node("Mul", ["clearAny", "zero_color"], ["newZeros"]),
        helper.make_node("Add", ["kept", "newZeros"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task202_scoreable", [x], [y], init)
    model = helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 12)],
    )
    onnx.checker.check_model(model)
    return model


# Required by the checker: it execs this file and reads global `model`.
model = build_onnx_model()
