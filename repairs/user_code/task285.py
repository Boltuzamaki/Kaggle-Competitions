"""Build narrowly scoped task285 graph experiments from the repair model."""

from pathlib import Path
import copy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path.cwd()
SOURCE = ROOT / "repairs" / "task285.onnx"
OUT = ROOT / "other_model_onnx"


def prune(model):
    required = {item.name for item in model.graph.output}
    kept = []
    for node in reversed(model.graph.node):
        if any(name in required for name in node.output if name):
            kept.append(node)
            required.update(name for name in node.input if name)
    kept.reverse()
    del model.graph.node[:]
    model.graph.node.extend(kept)
    referenced = {name for node in kept for name in node.input}
    initializers = [item for item in model.graph.initializer if item.name in referenced]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    produced = {name for node in kept for name in node.output if name}
    value_info = [item for item in model.graph.value_info if item.name in produced]
    del model.graph.value_info[:]
    model.graph.value_info.extend(value_info)
    return model


def uint8_topk():
    model = copy.deepcopy(onnx.load(SOURCE))
    for node in model.graph.node:
        if node.op_type == "TopK" and "gf" in node.input:
            node.input[0] = "g"
    model = prune(model)
    onnx.checker.check_model(model, full_check=True)
    path = OUT / "task285_uint8_topk.onnx"
    onnx.save(model, path)
    print(path)


def replace_initializer(model, name, array):
    for index, item in enumerate(model.graph.initializer):
        if item.name == name:
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(np.asarray(array), name=name)
            )
            return
    model.graph.initializer.append(numpy_helper.from_array(np.asarray(array), name=name))


def int8_padding():
    model = copy.deepcopy(onnx.load(SOURCE))

    # Conv(input, [1..10], bias=-1) produces the original color id on every
    # real cell and -1 on zero-padded cells.  Signed padding is therefore
    # intrinsically excluded from the final one-hot comparison.
    replace_initializer(
        model,
        "Wc",
        np.arange(1, 11, dtype=np.float32).reshape(1, 10, 1, 1),
    )
    replace_initializer(model, "pad_bias", np.array([-1.0], dtype=np.float32))
    replace_initializer(model, "ar10", np.arange(10, dtype=np.int8).reshape(1, 10, 1, 1))
    replace_initializer(model, "u80", np.array(0, dtype=np.uint8))
    replace_initializer(model, "i80", np.array(0, dtype=np.int8))
    replace_initializer(model, "u81", np.array(1, dtype=np.int8))

    for node in model.graph.node:
        if node.op_type == "Conv" and node.output == ["cf"]:
            node.input.append("pad_bias")
        if node.op_type == "Cast" and node.output[0] in {"c2d", "equ8"}:
            node.attribute[0].i = TensorProto.INT8
        if node.op_type == "TopK" and node.input[0] in {"gf", "scoref", "mf16"}:
            node.input[0] = {"gf": "g", "scoref": "score", "mf16": "mflat"}[
                node.input[0]
            ]
        if node.output[0] in {"avalid", "values"}:
            node.input[-1] = "i80"
        elif node.output[0] == "memv":
            node.input[-1] = "u80"
        for index, name in enumerate(node.input):
            if name == "c":
                node.input[index] = "tv"
            elif name == "fin":
                node.input[index] = "out2d"

    rewritten = []
    for node in model.graph.node:
        if node.output == ["values"]:
            rewritten.append(
                helper.make_node("Cast", ["cond"], ["condi"], to=TensorProto.INT8)
            )
            rewritten.append(helper.make_node("Add", ["vals", "u81"], ["vals1"]))
            rewritten.append(helper.make_node("Mul", ["vals1", "condi"], ["values1"]))
            node.op_type = "Sub"
            del node.input[:]
            node.input.extend(["values1", "u81"])
        rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)

    model = prune(model)
    onnx.checker.check_model(model, full_check=True)
    path = OUT / "task285_int8_padding.onnx"
    onnx.save(model, path)
    print(path)


def direct_membership():
    model = copy.deepcopy(onnx.load(OUT / "task285_int8_padding.onnx"))
    for node in model.graph.node:
        if node.output == ["mflat"]:
            node.input[0] = "mem2d"
    model = prune(model)
    onnx.checker.check_model(model, full_check=True)
    path = OUT / "task285_direct_membership.onnx"
    onnx.save(model, path)
    print(path)


def one_membership_pool():
    model = copy.deepcopy(onnx.load(OUT / "task285_int8_padding.onnx"))
    for node in model.graph.node:
        if node.output == ["mflat"]:
            node.input[0] = "m1"
    model = prune(model)
    onnx.checker.check_model(model, full_check=True)
    path = OUT / "task285_one_membership_pool.onnx"
    onnx.save(model, path)
    print(path)


def boolean_anchor():
    model = copy.deepcopy(onnx.load(OUT / "task285_int8_padding.onnx"))
    rewritten = []
    for node in model.graph.node:
        output = node.output[0]
        if output == "equ8":
            continue
        if output == "nzu":
            rewritten.append(helper.make_node("Greater", ["NN", "i80"], ["nz"] ))
            continue
        if output == "diffu":
            rewritten.append(helper.make_node("Xor", ["nz", "eq"], ["diffu"] ))
            continue
        if output == "noniso":
            node.input[0] = "eq"
        elif output == "sc1":
            node.op_type = "And"
        elif output == "score":
            node.op_type = "Where"
            del node.input[:]
            node.input.extend(["sc1", "tv", "i80"])
        rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    model = prune(model)
    onnx.checker.check_model(model, full_check=True)
    path = OUT / "task285_boolean_anchor.onnx"
    onnx.save(model, path)
    print(path)


def compact_indices():
    model = copy.deepcopy(onnx.load(OUT / "task285_int8_padding.onnx"))
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    replace_initializer(model, "OFF8", initializers["OFF8"].astype(np.uint16))
    replace_initializer(model, "WLUT", initializers["WLUT"].astype(np.uint16))
    replace_initializer(model, "i2s", np.array(2, dtype=np.uint8))

    to_u16 = {"t", "tidx", "ei", "f30i"}
    to_u8 = {"enegi", "fnegi"}
    for node in model.graph.node:
        if node.op_type == "Cast" and node.output[0] in to_u16:
            node.attribute[0].i = TensorProto.UINT16
        elif node.op_type == "Cast" and node.output[0] in to_u8:
            node.attribute[0].i = TensorProto.UINT8

    onnx.checker.check_model(model, full_check=True)
    path = OUT / "task285_compact_indices.onnx"
    onnx.save(model, path)
    print(path)


def candidate_limit(k):
    model = copy.deepcopy(onnx.load(OUT / "task285_int8_padding.onnx"))
    replace_initializer(model, "k32", np.array([k], dtype=np.int64))
    onnx.checker.check_model(model, full_check=True)
    path = OUT / f"task285_k{k}.onnx"
    onnx.save(model, path)
    print(path)


def static_first_pool():
    model = copy.deepcopy(onnx.load(OUT / "task285_int8_padding.onnx"))
    mask = np.zeros((1, 1, 5, 9), dtype=np.uint8)
    mask[:, :, :3, 2:7] = 1
    replace_initializer(model, "p1mask", mask)
    for node in model.graph.node:
        if node.output == ["m1"]:
            node.input[0] = "p1mask"
    model = prune(model)
    onnx.checker.check_model(model, full_check=True)
    path = OUT / "task285_static_first_pool.onnx"
    onnx.save(model, path)
    print(path)


def cropped_first_stage():
    model = copy.deepcopy(onnx.load(OUT / "task285_int8_padding.onnx"))
    replace_initializer(model, "crop_starts", np.array([0, 2], dtype=np.int64))
    replace_initializer(model, "crop_ends", np.array([3, 7], dtype=np.int64))
    replace_initializer(model, "crop_axes", np.array([2, 3], dtype=np.int64))
    rewritten = []
    for node in model.graph.node:
        if node.output == ["p2"]:
            rewritten.append(
                helper.make_node(
                    "Slice",
                    ["mem2d", "crop_starts", "crop_ends", "crop_axes"],
                    ["mseed"],
                )
            )
            node.input[0] = "mseed"
            for attribute in node.attribute:
                if attribute.name == "pads":
                    del attribute.ints[:]
                    attribute.ints.extend([2, 4, 4, 4])
        rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    model = prune(model)
    onnx.checker.check_model(model, full_check=True)
    path = OUT / "task285_cropped_first_stage.onnx"
    onnx.save(model, path)
    print(path)


def shaped_window():
    model = copy.deepcopy(onnx.load(OUT / "task285_cropped_first_stage.onnx"))
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    replace_initializer(model, "WLUT", arrays["WLUT"].reshape(4, 1, 5, 9))
    replace_initializer(model, "a_axes", np.array([1, 2], dtype=np.int64))
    rewritten = []
    for node in model.graph.node:
        if node.output == ["widx"]:
            rewritten.append(helper.make_node("Unsqueeze", ["a", "a_axes"], ["a4"]))
            for index, name in enumerate(node.input):
                if name == "a":
                    node.input[index] = "a4"
        if node.output == ["memb"]:
            rewritten.append(
                helper.make_node("Unsqueeze", ["acol", "a_axes"], ["acol4"])
            )
            for index, name in enumerate(node.input):
                if name == "acol":
                    node.input[index] = "acol4"
        if node.output == ["membu"]:
            node.output[0] = "mem2d"
        elif node.output == ["mem2d"]:
            continue
        rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    model = prune(model)
    onnx.checker.check_model(model, full_check=True)
    path = OUT / "task285_shaped_window.onnx"
    onnx.save(model, path)
    print(path)


def shaped_coordinates():
    model = copy.deepcopy(onnx.load(OUT / "task285_shaped_window.onnx"))
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    replace_initializer(model, "J1d", arrays["J1d"].reshape(45, 1))
    replace_initializer(model, "I1d", arrays["I1d"].reshape(45, 1))
    replace_initializer(model, "WLUT", arrays["WLUT"].reshape(4, 5, 9))
    rewritten = []
    for node in model.graph.node:
        if node.op_type == "Unsqueeze" and node.output[0] in {"K3", "I3"}:
            continue
        if node.output == ["si2"]:
            continue
        if node.output == ["K"]:
            node.output[0] = "K3"
        elif node.output == ["II"]:
            node.output[0] = "I3"
        elif node.output == ["a"]:
            node.input[1] = "si"
            node.output[0] = "a0"
            rewritten.append(node)
            rewritten.append(helper.make_node("Unsqueeze", ["a0", "axs1"], ["a"]))
            continue
        elif node.output == ["acol"]:
            node.input[1] = "si"
            node.output[0] = "acol0"
            rewritten.append(node)
            rewritten.append(helper.make_node("Unsqueeze", ["acol0", "axs1"], ["acol"]))
            continue
        elif node.output == ["woff"]:
            node.input[1] = "var2"
        rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    model = prune(model)
    onnx.checker.check_model(model, full_check=True)
    path = OUT / "task285.onnx"
    onnx.save(model, path)
    print(path)


def uint8_sentinel():
    model = copy.deepcopy(onnx.load(OUT / "task285.onnx"))
    replace_initializer(model, "u8255", np.array(255, dtype=np.uint8))
    rewritten = []
    skip = {"condi", "vals1", "values1", "values"}
    inserted = False
    for node in model.graph.node:
        if node.output[0] in skip:
            if not inserted:
                rewritten.extend(
                    [
                        helper.make_node(
                            "Cast", ["vals"], ["valsu"], to=TensorProto.UINT8
                        ),
                        helper.make_node(
                            "Where", ["cond", "valsu", "u8255"], ["valuesu"]
                        ),
                        helper.make_node(
                            "Cast", ["valuesu"], ["values"], to=TensorProto.INT8
                        ),
                    ]
                )
                inserted = True
            continue
        rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    model = prune(model)
    onnx.checker.check_model(model, full_check=True)
    path = OUT / "task285.onnx"
    onnx.save(model, path)
    print(path)


if __name__ == "__main__":
    uint8_topk()
    int8_padding()
    direct_membership()
    one_membership_pool()
    static_first_pool()
    cropped_first_stage()
    shaped_window()
    shaped_coordinates()
    uint8_sentinel()


# Promotion/checker convention: expose the strongest fully audited artifact.
model = onnx.load(OUT / "task285.onnx")
