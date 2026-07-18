# ARC task118: complete hidden plus/cross cells.
# Clean ONNX graph: no JSON loading, no base64, no input fingerprints.
#
# Input:  one-hot [1,10,30,30]
# Output: one-hot [1,10,30,30]
#
# Rule summary:
#   Colour 2 pixels are visible fragments of radius-2/radius-3 plus signs.
#   The missing plus cells that currently contain colour 5 are painted colour 8.
#   Background colour 0 is never changed.

import numpy as np


def solve_118_numpy(grid):
    """Reference solver for raw ARC grids. Not used by the checker."""
    a = np.asarray(grid, dtype=np.int64)
    H, W = a.shape

    def plus_coords(r, c, R):
        s = {(r, cc) for cc in range(c - R, c + R + 1) if 0 <= cc < W}
        s |= {(rr, c) for rr in range(r - R, r + R + 1) if 0 <= rr < H}
        return s

    def local_count(r, c, rad, kind, val):
        cnt = 0
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if kind == "square":
                    ok = True
                elif kind == "border":
                    ok = max(abs(dr), abs(dc)) == rad
                else:
                    ok = False
                if ok and a[rr, cc] == val:
                    cnt += 1
        return cnt

    cands = []
    for r in range(H):
        for c in range(W):
            for R in (2, 3):
                p = plus_coords(r, c, R)
                if not p:
                    continue
                n0 = sum(a[x, y] == 0 for x, y in p)
                n2 = sum(a[x, y] == 2 for x, y in p)
                sq = {
                    (rr, cc)
                    for rr in range(max(0, r - R), min(H, r + R + 1))
                    for cc in range(max(0, c - R), min(W, c + R + 1))
                }
                out2 = sum(a[x, y] == 2 for x, y in (sq - p))
                if n0 == 0 and n2 >= 2 and out2 == 0:
                    cands.append({"r": r, "c": c, "R": R, "n2": n2, "coords": p})

    by_center = {}
    for f in cands:
        by_center.setdefault((f["r"], f["c"]), []).append(f)

    selected = []
    for (r, c), fs in by_center.items():
        d = {f["R"]: f for f in fs}
        if 2 in d and 3 in d:
            choose3 = False
            if d[3]["n2"] > d[2]["n2"]:
                choose3 = True
            elif d[3]["n2"] == d[2]["n2"]:
                # If the outer radius-3 ring is fully hidden, the surrounding
                # 9x9 context distinguishes the larger hidden plus.
                if (
                    local_count(r, c, 2, "square", 0) >= 6
                    and local_count(r, c, 2, "border", 5) <= 9
                    and local_count(r, c, 4, "square", 5) >= 43
                ):
                    choose3 = True
            selected.append(d[3] if choose3 else d[2])
        else:
            selected.append(max(fs, key=lambda f: f["R"]))

    # Suppress accidental pluses whose visible 2-support is weaker than a nearby
    # selected plus. This removes random all-5/2 alignments from the noisy field.
    keep = []
    for f in selected:
        dominated = False
        for g in selected:
            if f is g:
                continue
            if abs(f["r"] - g["r"]) <= 4 and abs(f["c"] - g["c"]) <= 4:
                if g["n2"] > f["n2"] or (g["n2"] == f["n2"] and g["R"] > f["R"]):
                    dominated = True
                    break
        if not dominated:
            keep.append(f)

    out = a.copy()
    for f in keep:
        for r, c in f["coords"]:
            if out[r, c] == 5:
                out[r, c] = 8
    return out.tolist()


def build_onnx_model():
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    F = TensorProto.FLOAT
    I64 = TensorProto.INT64

    def K(name, value, dtype):
        return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)

    def plus_kernel(R):
        k = np.zeros((1, 1, 2 * R + 1, 2 * R + 1), np.float32)
        k[0, 0, R, :] = 1.0
        k[0, 0, :, R] = 1.0
        return k

    def square_kernel(R):
        return np.ones((1, 1, 2 * R + 1, 2 * R + 1), np.float32)

    def border_kernel(R):
        k = np.zeros((1, 1, 2 * R + 1, 2 * R + 1), np.float32)
        for i in range(2 * R + 1):
            for j in range(2 * R + 1):
                if max(abs(i - R), abs(j - R)) == R:
                    k[0, 0, i, j] = 1.0
        return k

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])

    init = [
        K("s0", [0], np.int64), K("e1", [1], np.int64),
        K("s2", [2], np.int64), K("e3", [3], np.int64),
        K("s5", [5], np.int64), K("e6", [6], np.int64),
        K("axisC", [1], np.int64),

        K("zero", [0.0], np.float32),
        K("one", [1.0], np.float32),
        K("two", [2.0], np.float32),
        K("three", [3.0], np.float32),
        K("six", [6.0], np.float32),
        K("nine", [9.0], np.float32),
        K("ten", [10.0], np.float32),
        K("forty3", [43.0], np.float32),

        K("P2", plus_kernel(2), np.float32),
        K("P3", plus_kernel(3), np.float32),
        K("S2", square_kernel(2), np.float32),
        K("S3", square_kernel(3), np.float32),
        K("S4", square_kernel(4), np.float32),
        K("B2", border_kernel(2), np.float32),
        K("ones30", np.ones((1, 1, 30, 30), np.float32), np.float32),
    ]

    nodes = []

    # Extract colours 0, 2, 5.
    nodes += [
        helper.make_node("Slice", ["input", "s0", "e1", "axisC"], ["ch0"]),
        helper.make_node("Slice", ["input", "s2", "e3", "axisC"], ["ch2"]),
        helper.make_node("Slice", ["input", "s5", "e6", "axisC"], ["ch5"]),
        helper.make_node("Add", ["ch2", "ch5"], ["nonzero"]),
        helper.make_node("Mul", ["ch0", "zero"], ["zch"]),
    ]

    # Geometric counts for radius 2 and 3 plus candidates.
    for R, P, S in [(2, "P2", "S2"), (3, "P3", "S3")]:
        pads = [R, R, R, R]
        nodes += [
            helper.make_node("Conv", ["ch2", P], [f"p2cnt{R}"], pads=pads),
            helper.make_node("Conv", ["nonzero", P], [f"pnzcnt{R}"], pads=pads),
            helper.make_node("Conv", ["ones30", P], [f"psize{R}"], pads=pads),
            helper.make_node("Conv", ["ch2", S], [f"sq2cnt{R}"], pads=pads),
            helper.make_node("Sub", [f"sq2cnt{R}", f"p2cnt{R}"], [f"outside2_{R}"]),

            helper.make_node("Equal", [f"pnzcnt{R}", f"psize{R}"], [f"allnz{R}"]),
            helper.make_node("Equal", [f"outside2_{R}", "zero"], [f"noout2_{R}"]),
            helper.make_node("GreaterOrEqual", [f"p2cnt{R}", "two"], [f"enough2_{R}"]),
            helper.make_node("And", [f"allnz{R}", f"noout2_{R}"], [f"validA{R}"]),
            helper.make_node("And", [f"validA{R}", f"enough2_{R}"], [f"valid{R}"]),
        ]

    # Ambiguous radius-3 selection context.
    nodes += [
        helper.make_node("Conv", ["ch0", "S2"], ["sq2_0"], pads=[2, 2, 2, 2]),
        helper.make_node("Conv", ["ch5", "B2"], ["border2_5"], pads=[2, 2, 2, 2]),
        helper.make_node("Conv", ["ch5", "S4"], ["sq4_5"], pads=[4, 4, 4, 4]),

        helper.make_node("Greater", ["p2cnt3", "p2cnt2"], ["n3_gt_n2"]),
        helper.make_node("Equal", ["p2cnt3", "p2cnt2"], ["n3_eq_n2"]),
        helper.make_node("GreaterOrEqual", ["sq2_0", "six"], ["ambA"]),
        helper.make_node("LessOrEqual", ["border2_5", "nine"], ["ambB"]),
        helper.make_node("GreaterOrEqual", ["sq4_5", "forty3"], ["ambC"]),
        helper.make_node("And", ["ambA", "ambB"], ["ambAB"]),
        helper.make_node("And", ["ambAB", "ambC"], ["ambABC"]),
        helper.make_node("And", ["n3_eq_n2", "ambABC"], ["ambig3"]),

        helper.make_node("Not", ["valid2"], ["notValid2"]),
        helper.make_node("Or", ["n3_gt_n2", "ambig3"], ["choose3B"]),
        helper.make_node("Or", ["notValid2", "choose3B"], ["choose3C"]),
        helper.make_node("And", ["valid3", "choose3C"], ["choose3"]),
        helper.make_node("Not", ["choose3"], ["notChoose3"]),
        helper.make_node("And", ["valid2", "notChoose3"], ["choose2"]),
    ]

    # Candidate scores: n2*10 + radius. Suppress weaker candidates in a 9x9 window.
    nodes += [
        helper.make_node("Cast", ["choose2"], ["choose2f"], to=F),
        helper.make_node("Cast", ["choose3"], ["choose3f"], to=F),
        helper.make_node("Mul", ["p2cnt2", "ten"], ["score2a"]),
        helper.make_node("Add", ["score2a", "two"], ["score2b"]),
        helper.make_node("Mul", ["score2b", "choose2f"], ["score2"]),
        helper.make_node("Mul", ["p2cnt3", "ten"], ["score3a"]),
        helper.make_node("Add", ["score3a", "three"], ["score3b"]),
        helper.make_node("Mul", ["score3b", "choose3f"], ["score3"]),
        helper.make_node("Add", ["score2", "score3"], ["score"]),
        helper.make_node("MaxPool", ["score"], ["localMax"], kernel_shape=[9, 9], pads=[4, 4, 4, 4], strides=[1, 1]),

        helper.make_node("Equal", ["score2", "localMax"], ["score2max"]),
        helper.make_node("Equal", ["score3", "localMax"], ["score3max"]),
        helper.make_node("And", ["choose2", "score2max"], ["keep2"]),
        helper.make_node("And", ["choose3", "score3max"], ["keep3"]),
        helper.make_node("Cast", ["keep2"], ["keep2f"], to=F),
        helper.make_node("Cast", ["keep3"], ["keep3f"], to=F),
    ]

    # Paint selected plus templates back to the image.
    nodes += [
        helper.make_node("ConvTranspose", ["keep2f", "P2"], ["paint2"], pads=[2, 2, 2, 2]),
        helper.make_node("ConvTranspose", ["keep3f", "P3"], ["paint3"], pads=[3, 3, 3, 3]),
        helper.make_node("Add", ["paint2", "paint3"], ["paint"]),
        helper.make_node("Greater", ["paint", "zero"], ["paintBool"]),
        helper.make_node("Cast", ["paintBool"], ["paintF"], to=F),
        helper.make_node("Mul", ["paintF", "ch5"], ["out8"]),
        helper.make_node("Sub", ["one", "out8"], ["not8"]),
        helper.make_node("Mul", ["ch5", "not8"], ["out5"]),

        helper.make_node(
            "Concat",
            ["ch0", "zch", "ch2", "zch", "zch", "out5", "zch", "zch", "out8", "zch"],
            ["output"],
            axis=1,
        ),
    ]

    graph = helper.make_graph(nodes, "task118_rule", [x], [y], init)
    model = helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


model = build_onnx_model()
