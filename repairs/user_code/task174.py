"""Exact two-probe rewrite for task174.

There are exactly three non-background single-color objects and exactly one is
horizontally symmetric.  Test the first two candidates; if neither matches, the
third candidate is necessarily the winner.  Only then compute a bounding box.
"""

import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(os.environ.get("PROJECT_DIR", Path.cwd()))
OUT = ROOT / "scratch_onnx" / "task174_two_probe_int.onnx"

F = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
I64 = TensorProto.INT64
I32 = TensorProto.INT32
I8 = TensorProto.INT8
U8 = TensorProto.UINT8
B = TensorProto.BOOL


def K(name, value, dtype):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)


def N(op, inputs, outputs, **attrs):
    return helper.make_node(op, inputs, outputs, **attrs)


def build():
    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])

    crop = np.zeros((10, 30), np.float32)
    crop[np.arange(10), np.arange(10)] = 1
    symtable = np.full((10, 10, 1), 16, np.uint8)
    for center in range(10):
        pair = 0
        for left in range(10):
            right = center - left
            if left >= right:
                continue
            if 0 <= right < 10:
                weight = 1 << pair
                symtable[center, left, 0] = 16 + weight
                symtable[center, right, 0] = 16 - weight
                pair += 1

    init = [
        K("Wcv", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32),
        K("cropS", crop, np.float32),
        K("k2", [2], np.int64),
        K("symtable", symtable, np.uint8),
        K("sym_zp", 16, np.uint8),
        K("sym_out_zp", 32, np.uint8),
        K("qscale", 1, np.float32),
        K("zero_u8", 0, np.uint8),
        K("sentinel_u8", 255, np.uint8),
        K("nine_i8", 9, np.int8),
        K("eighteen_i8", 18, np.int8),
        K("one_i8", 1, np.int8),
        K("idx5", np.arange(5), np.int8),
        K("igrid", np.arange(5).reshape(1, 1, 5, 1), np.int8),
        K("jgrid", np.arange(5).reshape(1, 1, 1, 5), np.int8),
        K("chid", np.arange(10).reshape(1, 10, 1, 1), np.uint8),
        K("ax1", [1], np.int64),
        K("ax01", [0, 1], np.int64),
        K("ax23", [2, 3], np.int64),
        K("rev10", np.arange(9, -1, -1), np.int32),
        K("pads", [0, 0, 0, 0, 0, 0, 25, 25], np.int64),
    ]

    n = []
    n += [
        N("Einsum", ["input", "Wcv", "cropS", "cropS"], ["cg"], equation="nchw,ocuv,rh,sw->nors"),
        N("ReduceMax", ["input"], ["pres"], axes=[2, 3], keepdims=1),
        N("Mul", ["pres", "Wcv"], ["weighted"]),
        N("TopK", ["weighted", "k2"], ["tkv", "tki"], axis=1, largest=1, sorted=1),
        N("Cast", ["tki"], ["col2"], to=F),
        N("ReduceSum", ["weighted", "ax1"], ["sum_all"], keepdims=1),
        N("ReduceSum", ["tkv", "ax1"], ["sum_top2"], keepdims=1),
        N("Sub", ["sum_all", "sum_top2"], ["third_col"]),
        N("Equal", ["cg", "col2"], ["P2b"]),
        N("Cast", ["P2b"], ["M8"], to=U8),
        N("ReduceMax", ["M8"], ["colpres"], axes=[2], keepdims=0),
        N("ArgMax", ["colpres"], ["right64"], axis=2, keepdims=0, select_last_index=1),
        N("Cast", ["right64"], ["right"], to=I8),
        N("ArgMax", ["colpres"], ["left64"], axis=2, keepdims=0, select_last_index=0),
        N("Cast", ["left64"], ["left"], to=I8),
        N("Add", ["left", "right"], ["s0"]),
        N("Sub", ["eighteen_i8", "s0"], ["sflip"]),
        N("Min", ["s0", "sflip"], ["sbase"]),
        N("Cast", ["sbase"], ["sbase32"], to=I32),
        N("Gather", ["symtable", "sbase32"], ["symbase"], axis=0),
        N("Gather", ["symbase", "rev10"], ["symrev"], axis=2),
        N("Greater", ["s0", "nine_i8"], ["use_rev"]),
        N("Unsqueeze", ["use_rev", "ax23"], ["use_rev4"]),
        N("Where", ["use_rev4", "symrev", "symbase"], ["symweights"]),
        N(
            "QLinearMatMul",
            ["M8", "qscale", "zero_u8", "symweights", "qscale", "sym_zp", "qscale", "sym_out_zp"],
            ["symdot"],
        ),
        N("ReduceMin", ["symdot"], ["symmin"], axes=[2, 3], keepdims=0),
        N("ReduceMax", ["symdot"], ["symmax"], axes=[2, 3], keepdims=0),
        N("Equal", ["symmin", "sym_out_zp"], ["symmin_ok"]),
        N("Equal", ["symmax", "sym_out_zp"], ["symmax_ok"]),
        N("And", ["symmin_ok", "symmax_ok"], ["pickb"]),
        N("Cast", ["pickb"], ["pick2"], to=U8),
        N("Sub", ["col2", "third_col"], ["coldelta"]),
        N("Cast", ["pick2"], ["pickf"], to=F),
        N("Einsum", ["coldelta", "pickf"], ["kdelta"], equation="ncuv,nc->nuv"),
        N("Add", ["third_col", "kdelta"], ["kcol"]),
        N("Cast", ["kcol"], ["kcol8"], to=U8),
        N("Equal", ["cg", "kcol"], ["winmask"]),
        N("Cast", ["winmask"], ["win8"], to=U8),
        N("ReduceMax", ["win8"], ["win_colpres"], axes=[2], keepdims=0),
        N("ReduceMax", ["win8"], ["win_rowpres"], axes=[3], keepdims=0),
    ]

    for src, out, axis, last in [
        ("win_colpres", "right2", 2, 1),
        ("win_colpres", "left2", 2, 0),
        ("win_rowpres", "bot2", 2, 1),
        ("win_rowpres", "top2", 2, 0),
    ]:
        n.append(N("ArgMax", [src], [out + "_64"], axis=axis, keepdims=0, select_last_index=last))
        if out in {"right2", "bot2"}:
            n.append(N("Cast", [out + "_64"], [out], to=I8))
        else:
            n.append(N("Cast", [out + "_64"], [out + "_v"], to=I8))
            n.append(N("Squeeze", [out + "_v", "ax01"], [out]))

    n += [
        N("Sub", ["right2", "left2"], ["Wm1"]),
        N("Add", ["Wm1", "one_i8"], ["Wb"]),
        N("Sub", ["bot2", "top2"], ["Hm1"]),
        N("Add", ["Hm1", "one_i8"], ["Hb"]),
        N("Add", ["top2", "idx5"], ["ri0"]),
        N("Clip", ["ri0", "", "nine_i8"], ["ridx"]),
        N("Cast", ["ridx"], ["ridx32"], to=I32),
        N("Gather", ["winmask", "ridx32"], ["c0"], axis=2),
        N("Add", ["left2", "idx5"], ["ci0"]),
        N("Clip", ["ci0", "", "nine_i8"], ["cidx"]),
        N("Cast", ["cidx"], ["cidx32"], to=I32),
        N("Gather", ["c0", "cidx32"], ["patt"], axis=3),
        N("Less", ["igrid", "Hb"], ["bi"]),
        N("Less", ["jgrid", "Wb"], ["bj"]),
        N("And", ["bi", "bj"], ["Bb"]),
        N("Where", ["patt", "kcol8", "zero_u8"], ["colored"]),
        N("Where", ["Bb", "colored", "sentinel_u8"], ["vg"]),
        N("Equal", ["chid", "vg"], ["out5r"]),
        N("Pad", ["out5r", "pads"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(n, "task174_two_probe", [x], [y], init)
    model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 17)])
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUT)
    print(OUT)
    return model


model = build()
