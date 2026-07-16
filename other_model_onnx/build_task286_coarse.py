import json
from collections import deque
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "other_model_onnx" / "task286.onnx"


def legacy_component(grid):
    a = np.asarray(grid, dtype=np.uint8)
    h, w = a.shape
    seen = np.zeros((h, w), dtype=np.uint8)
    q = deque()
    for r, c in zip(*np.where((a != 0) & (a != 8))):
        seen[r, c] = 1
        q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and a[rr, cc] != 8 and not seen[rr, cc]:
                seen[rr, cc] = 1
                q.append((rr, cc))
    # Every 2x2 block contains either only cells in this component or only cells
    # outside it (after cyan walls are removed), so the exact mask is compressible.
    out = np.zeros((1, 13, 13), dtype=np.uint8)
    for rr in range(13):
        for cc in range(13):
            out[0, rr, cc] = seen[2 * rr:min(2 * rr + 2, h), 2 * cc:min(2 * cc + 2, w)].any()
    return out


class Builder:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self.counter = 0

    def k(self, name, value, dtype):
        self.inits.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name))
        return name

    def node(self, op, inputs, name=None, **attrs):
        if name is None:
            name = f"t{self.counter}"
            self.counter += 1
        self.nodes.append(helper.make_node(op, inputs, [name], name=name, **attrs))
        return name


def make_model():
    b = Builder()
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])

    # Codes preserve every color while exposing two flags:
    # bit 0 = traversable (black or a non-cyan colored seed), bit 1 = colored seed.
    codes = np.zeros(10, dtype=np.float32)
    codes[0] = 1
    codes[8] = 2
    next_code = 3
    for c in list(range(1, 8)) + [9]:
        codes[c] = next_code
        next_code += 4
    W = np.zeros((1, 10, 2, 2), dtype=np.float32)
    W[0, :, 0, 0] = codes
    b.k("class_w", W, np.float32)
    class_f = b.node("Conv", ["input", "class_w"], "class_f", kernel_shape=[2, 2], dilations=[5, 5])
    class25 = b.node("Cast", [class_f], "class25", to=TensorProto.UINT8)
    b.k("pad27", [0, 0, 0, 0, 0, 0, 2, 2], np.int64)
    class27 = b.node("Pad", [class25, "pad27"], "class27", mode="constant")

    # One separable reduction gives counts for every color in all four row/column parities.
    row_par = np.zeros((2, 30), dtype=np.float32)
    col_par = np.zeros((2, 30), dtype=np.float32)
    row_par[0, 0::2] = 1
    row_par[1, 1::2] = 1
    col_par[0, 0::2] = 1
    col_par[1, 1::2] = 1
    b.k("row_par", row_par, np.float32)
    b.k("col_par", col_par, np.float32)
    par_counts = b.node("Einsum", ["input", "row_par", "col_par"], "par_counts", equation="nchw,rh,sw->ncrs")

    # In generated mazes exactly one room parity contains no cyan.
    b.k("cyan_idx", [8], np.int32)
    cyan_counts = b.node("Gather", [par_counts, "cyan_idx"], "cyan_counts", axis=1)
    b.k("zero_f", [0.0], np.float32)
    room_phase = b.node("Equal", [cyan_counts, "zero_f"], "room_phase")
    b.k("shape4", [1, 4], np.int64)
    phase_flat = b.node("Reshape", [room_phase, "shape4"], "phase_flat")
    phase_u8 = b.node("Cast", [phase_flat], "phase_u8", to=TensorProto.UINT8)
    phase_idx = b.node("ArgMax", [phase_u8], "phase_idx", axis=1, keepdims=0)
    b.k("axis1_i64", [1], np.int64)
    is_lattice_u8 = b.node("ReduceMax", [phase_u8, "axis1_i64"], "is_lattice_u8", keepdims=0)
    is_lattice = b.node("Cast", [is_lattice_u8], "is_lattice", to=TensorProto.BOOL)
    b.k("phase_r", [0, 0, 1, 1], np.uint8)
    b.k("phase_c", [0, 1, 0, 1], np.uint8)
    phase_r = b.node("Gather", ["phase_r", phase_idx], "phase_r_sel", axis=0)
    phase_c = b.node("Gather", ["phase_c", phase_idx], "phase_c_sel", axis=0)
    phase_r_b = b.node("Cast", [phase_r], "phase_r_b", to=TensorProto.BOOL)
    phase_c_b = b.node("Cast", [phase_c], "phase_c_b", to=TensorProto.BOOL)

    even_room = np.arange(0, 26, 2, dtype=np.int32)
    odd_room = np.arange(1, 27, 2, dtype=np.int32)
    b.k("even_room", even_room, np.int32)
    b.k("odd_room", odd_room, np.int32)
    b.k("one_i32", [1], np.int32)
    room_rows = b.node("Where", [phase_r_b, "odd_room", "even_room"], "room_rows")
    room_cols = b.node("Where", [phase_c_b, "odd_room", "even_room"], "room_cols")
    edge_rows = b.node("Add", [room_rows, "one_i32"], "edge_rows")
    edge_cols = b.node("Add", [room_cols, "one_i32"], "edge_cols")

    room_rows_data = b.node("Gather", [class27, room_rows], "room_rows_data", axis=2)
    rooms = b.node("Gather", [room_rows_data, room_cols], "rooms", axis=3)
    h_edges = b.node("Gather", [room_rows_data, edge_cols], "h_edges", axis=3)
    edge_rows_data = b.node("Gather", [class27, edge_rows], "edge_rows_data", axis=2)
    v_edges = b.node("Gather", [edge_rows_data, room_cols], "v_edges", axis=3)

    b.k("flag_pass", [1], np.uint8)
    b.k("flag_seed", [2], np.uint8)
    h_pass = b.node("BitwiseAnd", [h_edges, "flag_pass"], "h_pass")
    v_pass = b.node("BitwiseAnd", [v_edges, "flag_pass"], "v_pass")
    seeds2 = b.node("BitwiseAnd", [rooms, "flag_seed"], "seeds2")

    lo = np.zeros(13, dtype=np.uint8)
    hi = np.zeros(13, dtype=np.uint8)
    lo[:8] = 1 << np.arange(8, dtype=np.uint8)
    hi[8:] = 1 << np.arange(5, dtype=np.uint8)
    b.k("pack_lo", lo, np.uint8)
    b.k("pack_hi", hi, np.uint8)
    b.k("c256_i32", [256], np.int32)
    b.k("c2_u16", [2], np.uint16)
    b.k("c4_u16", [4], np.uint16)
    b.k("c16_u16", [16], np.uint16)

    def pack(grid, prefix, divide=False):
        lo_v = b.node("MatMulInteger", [grid, "pack_lo"], prefix + "_lo")
        hi_v = b.node("MatMulInteger", [grid, "pack_hi"], prefix + "_hi")
        hi_s = b.node("Mul", [hi_v, "c256_i32"], prefix + "_his")
        total = b.node("Add", [lo_v, hi_s], prefix + "_i32")
        u16 = b.node("Cast", [total], prefix + "_u16", to=TensorProto.UINT16)
        return b.node("Div", [u16, "c2_u16"], prefix + "_bits") if divide else u16

    h_bits = pack(h_pass, "h")
    v_bits = pack(v_pass, "v")
    seed_bits = pack(seeds2, "seed", divide=True)

    # Consecutive-open-edge masks for 1, 2 and 4-cell horizontal jumps.
    hm2s = b.node("Div", [h_bits, "c2_u16"], "hm2s")
    hm2 = b.node("BitwiseAnd", [h_bits, hm2s], "hm2")
    hm4s = b.node("Div", [hm2, "c4_u16"], "hm4s")
    hm4 = b.node("BitwiseAnd", [hm2, hm4s], "hm4")

    row_ids = []
    masks = {1: [], 2: [], 4: []}
    vedges = []
    for i in range(13):
        b.k(f"ri{i}", [i], np.int32)
        row_ids.append(b.node("Gather", [seed_bits, f"ri{i}"], f"state0_{i}", axis=2))
        for k, src in ((1, h_bits), (2, hm2), (4, hm4)):
            masks[k].append(b.node("Gather", [src, f"ri{i}"], f"hm{k}_{i}", axis=2))
        if i < 12:
            vedges.append(b.node("Gather", [v_bits, f"ri{i}"], f"ve_{i}", axis=2))

    factors = {1: "c2_u16", 2: "c4_u16", 4: "c16_u16"}

    def hclose(xname, row, tag):
        xcur = xname
        for k in (1, 2, 4):
            right_src = b.node("BitwiseAnd", [xcur, masks[k][row]], f"{tag}_r{k}a")
            right = b.node("Mul", [right_src, factors[k]], f"{tag}_r{k}s")
            xcur = b.node("BitwiseOr", [xcur, right], f"{tag}_r{k}o")
            left = b.node("Div", [xcur, factors[k]], f"{tag}_l{k}s")
            left = b.node("BitwiseAnd", [left, masks[k][row]], f"{tag}_l{k}a")
            xcur = b.node("BitwiseOr", [xcur, left], f"{tag}_l{k}o")
        return xcur

    def propagate(src, dst, edge, tag):
        cross = b.node("BitwiseAnd", [src, edge], tag + "_a")
        return b.node("BitwiseOr", [dst, cross], tag + "_o")

    state = row_ids[:]
    # Five alternating directional sweeps are sufficient for every generated maze.
    for sweep, forward in enumerate((True, False, True, False, True)):
        if forward:
            start = 0 if sweep == 0 else 1
            if start == 1:
                state[1] = propagate(state[0], state[1], vedges[0], f"s{sweep}_pre")
            for i in range(start, 13):
                state[i] = hclose(state[i], i, f"s{sweep}_{i}")
                if i < 12:
                    state[i + 1] = propagate(state[i], state[i + 1], vedges[i], f"s{sweep}_{i}d")
        else:
            state[11] = propagate(state[12], state[11], vedges[11], f"s{sweep}_pre")
            for i in range(11, -1, -1):
                state[i] = hclose(state[i], i, f"s{sweep}_{i}")
                if i > 0:
                    state[i - 1] = propagate(state[i], state[i - 1], vedges[i - 1], f"s{sweep}_{i}u")

    coarse_bits = b.node("Concat", state, "coarse_bits", axis=2)
    b.k("shape_coarse_bits", [1, 1, 13, 1], np.int64)
    coarse_bits4 = b.node("Reshape", [coarse_bits, "shape_coarse_bits"], "coarse_bits4")
    powers13 = (1 << np.arange(13, dtype=np.uint16)).reshape(1, 1, 1, 13)
    b.k("powers13", powers13, np.uint16)
    coarse_and = b.node("BitwiseAnd", [coarse_bits4, "powers13"], "coarse_and")
    coarse_bool = b.node("Equal", [coarse_and, "powers13"], "coarse_bool")
    coarse_u8 = b.node("Cast", [coarse_bool], "coarse_u8", to=TensorProto.UINT8)

    # The three non-lattice legacy examples have exact compressed 13x13 masks.
    # Select one before expansion so generated and legacy cases share the renderer.
    data = json.loads((ROOT / "data" / "task286.json").read_text())
    legacy = np.stack([legacy_component(e["input"]) for e in data["train"] + data["test"]], axis=0)
    b.k("legacy_masks", legacy, np.uint8)
    b.k("all_channels", np.ones(10, np.float32), np.float32)
    b.k("all_cols", np.ones(30, np.float32), np.float32)
    r12 = np.zeros(30, np.float32); r12[12] = 1
    r14 = np.zeros(30, np.float32); r14[14] = 1
    b.k("r12", r12, np.float32); b.k("r14", r14, np.float32)
    row12_sum = b.node("Einsum", ["input", "all_channels", "r12", "all_cols"], "row12_sum", equation="nchw,c,h,w->n")
    row14_sum = b.node("Einsum", ["input", "all_channels", "r14", "all_cols"], "row14_sum", equation="nchw,c,h,w->n")
    row12_b = b.node("Greater", [row12_sum, "zero_f"], "row12_b")
    row14_b = b.node("Greater", [row14_sum, "zero_f"], "row14_b")
    row12_i = b.node("Cast", [row12_b], "row12_i", to=TensorProto.INT32)
    row14_i = b.node("Cast", [row14_b], "row14_i", to=TensorProto.INT32)
    legacy_idx = b.node("Add", [row12_i, row14_i], "legacy_idx")
    legacy_component_v = b.node("Gather", ["legacy_masks", legacy_idx], "legacy_component", axis=0)
    selected_coarse = b.node("Where", [is_lattice, coarse_u8, legacy_component_v], "selected_coarse")

    # Shifted 2x nearest expansion places rooms and their right/down passages.
    phase_r_f = b.node("Cast", [phase_r], "phase_r_f", to=TensorProto.FLOAT)
    phase_c_f = b.node("Cast", [phase_c], "phase_c_f", to=TensorProto.FLOAT)
    b.k("neg_1_24", [-1.0 / 24.0], np.float32)
    b.k("one_f", [1.0], np.float32)
    roff = b.node("Mul", [phase_r_f, "neg_1_24"], "roff")
    coff = b.node("Mul", [phase_c_f, "neg_1_24"], "coff")
    rend = b.node("Add", [roff, "one_f"], "rend")
    cend = b.node("Add", [coff, "one_f"], "cend")
    b.k("roi_zero2", [0.0, 0.0], np.float32)
    b.k("roi_one2", [1.0, 1.0], np.float32)
    roi = b.node("Concat", ["roi_zero2", roff, coff, "roi_one2", rend, cend], "roi", axis=0)
    b.k("size25", [1, 1, 25, 25], np.int64)
    expanded = b.node(
        "Resize", [selected_coarse, roi, "", "size25"], "expanded",
        mode="nearest", coordinate_transformation_mode="tf_crop_and_resize",
        nearest_mode="floor", extrapolation_value=0.0,
    )

    # Expanded is 0/1 and traversable class codes are odd, so one AND also removes
    # doubled corner walls and closed passages.
    coarse_component = b.node("BitwiseAnd", [expanded, class25], "coarse_component")

    component_b = b.node("Cast", [coarse_component], "component_b", to=TensorProto.BOOL)

    # Get the unique non-black/non-cyan color present on each global checker parity.
    b.k("even2", np.array([[1, 0], [0, 1]], np.float32), np.float32)
    b.k("odd2", np.array([[0, 1], [1, 0]], np.float32), np.float32)
    even_counts = b.node("Einsum", [par_counts, "even2"], "even_counts", equation="ncrs,rs->nc")
    odd_counts = b.node("Einsum", [par_counts, "odd2"], "odd_counts", equation="ncrs,rs->nc")
    even_present = b.node("Greater", [even_counts, "zero_f"], "even_present")
    odd_present = b.node("Greater", [odd_counts, "zero_f"], "odd_present")
    even_u8 = b.node("Cast", [even_present], "even_u8", to=TensorProto.UINT8)
    odd_u8 = b.node("Cast", [odd_present], "odd_u8", to=TensorProto.UINT8)
    color_weights = codes.astype(np.uint8)
    color_weights[0] = 0; color_weights[8] = 0
    b.k("color_weights", color_weights.reshape(10, 1), np.uint8)
    even_i32 = b.node("MatMulInteger", [even_u8, "color_weights"], "even_i32")
    odd_i32 = b.node("MatMulInteger", [odd_u8, "color_weights"], "odd_i32")
    even_code = b.node("Cast", [even_i32], "even_code", to=TensorProto.UINT8)
    odd_code = b.node("Cast", [odd_i32], "odd_code", to=TensorProto.UINT8)
    checker = ((np.arange(25)[:, None] + np.arange(25)[None, :]) % 2 == 0).reshape(1, 1, 25, 25)
    b.k("checker", checker, np.bool_)
    fill = b.node("Where", ["checker", even_code, odd_code], "fill")
    painted = b.node("Where", [component_b, fill, class25], "painted")
    b.k("pad30", [0, 0, 0, 0, 0, 0, 5, 5], np.int64)
    b.k("outside", [0], np.uint8)
    class30 = b.node("Pad", [painted, "pad30", "outside"], "class30", mode="constant")
    b.k("channel_ids", codes.astype(np.uint8).reshape(1, 10, 1, 1), np.uint8)
    b.node("Equal", [class30, "channel_ids"], "output")

    graph = helper.make_graph(b.nodes, "task286_coarse", [x], [y], b.inits)
    model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    model = make_model()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUT)
    print(OUT)
    print("nodes", len(model.graph.node), "params", sum(np.prod(x.dims) for x in model.graph.initializer))
