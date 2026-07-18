
# ARC task219 ONNX READY v2: module-scope model is built on import.
# Clean ONNX graph: no JSON loading, no stored examples, no input hashes.

import numpy as np


def solve_219_numpy(grid):
    a = np.asarray(grid, dtype=np.int64)
    out = a.copy()
    H, W = a.shape
    rows = np.where(np.any(a != 0, axis=1))[0]
    if len(rows) == 0:
        return out
    bands = []
    cur = [int(rows[0])]
    for rr in rows[1:]:
        rr = int(rr)
        if rr == cur[-1] + 1:
            cur.append(rr)
        else:
            bands.append(cur)
            cur = [rr]
    bands.append(cur)
    if len(bands) <= 1:
        return out

    def info(b):
        cells = [(r, c) for r in b for c in range(W) if a[r, c] != 0]
        right = max(c for _, c in cells)
        reach = [r for r, c in cells if c == right]
        # For this distribution, any structurally-tied reaching row gives the same final union.
        corner = (min(reach), right)
        return {"rows": b, "cells": cells, "right": right, "corner": corner}

    infos = [info(b) for b in bands]
    template = infos[0]
    non = infos[1:]

    def offsets(inf):
        cr, cc = inf["corner"]
        return {(r-cr, c-cc) for r, c in inf["cells"]}

    non_offsets = [offsets(x) for x in non]
    non_bound = [{dr for dr, dc in s if dc == 0} for s in non_offsets]
    candidates = [(r, c) for r, c in template["cells"] if c < template["right"]]
    if not candidates:
        return out

    scored = []
    for pr, pc in candidates:
        body = {(r-pr, c-pc) for r, c in template["cells"] if c <= pc}
        bound = {dr for dr, dc in body if dc == 0}
        best = min((len(bound ^ nb), len(body ^ no)) for no, nb in zip(non_offsets, non_bound))
        scored.append((best, (pr, pc)))
    best = min(s for s, _ in scored)
    split_points = [p for s, p in scored if s == best]

    for pr, pc in split_points:
        ray = [(r-pr, c-pc) for r, c in template["cells"] if c > pc]
        for inf in non:
            cr, cc = inf["corner"]
            for dr, dc in ray:
                rr, col = cr + dr, cc + dc
                if 0 <= rr < H and 0 <= col < W and out[rr, col] == 0:
                    out[rr, col] = 1
    return out


def build_onnx_model():
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    F = TensorProto.FLOAT
    I64 = TensorProto.INT64

    def K(name, value, dtype=np.float32):
        if dtype == F:
            dtype = np.float32
        elif dtype == I64:
            dtype = np.int64
        return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)

    H = 30
    W = 30
    RH = 15
    RW = 10
    N = RH * RW
    SENT = N
    FULL = H * W

    eff = np.array([r * W + c for r in range(RH) for c in range(RW)], dtype=np.int64)
    row_sel = np.zeros((N, RH), dtype=np.float32)
    col_sel = np.zeros((N, RW), dtype=np.float32)
    place = np.zeros((N, FULL), dtype=np.float32)
    for r in range(RH):
        for c in range(RW):
            i = r * RW + c
            row_sel[i, r] = 1.0
            col_sel[i, c] = 1.0
            place[i, r * W + c] = 1.0

    # Body comparison offsets.  Template bands in this generated task have height <= 3,
    # so dr=-2..2 is sufficient; dc<=0 is the body side of a split point.
    body_offsets = [(dr, dc) for dr in range(-2, 3) for dc in range(-9, 1)]
    ray_offsets = [(dr, dc) for dr in range(-2, 3) for dc in range(1, 10)]
    O = len(body_offsets)
    R = len(ray_offsets)

    body_indices = np.full((N, O), SENT, dtype=np.int64)
    ray_indices = np.full((N, R), SENT, dtype=np.int64)
    for r in range(RH):
        for c in range(RW):
            p = r * RW + c
            for j, (dr, dc) in enumerate(body_offsets):
                rr, cc = r + dr, c + dc
                if 0 <= rr < RH and 0 <= cc < RW:
                    body_indices[p, j] = rr * RW + cc
            for j, (dr, dc) in enumerate(ray_offsets):
                rr, cc = r + dr, c + dc
                if 0 <= rr < RH and 0 <= cc < RW:
                    ray_indices[p, j] = rr * RW + cc

    # For shifting the corner mask by a ray offset into output cells.
    shift_indices = []
    for dr, dc in ray_offsets:
        idx = np.full((N,), SENT, dtype=np.int64)
        for r in range(RH):
            for c in range(RW):
                src_r, src_c = r - dr, c - dc
                if 0 <= src_r < RH and 0 <= src_c < RW:
                    idx[r * RW + c] = src_r * RW + src_c
        shift_indices.append(idx)

    weights = np.ones((O, 1), dtype=np.float32)
    for j, (dr, dc) in enumerate(body_offsets):
        if dc == 0:
            weights[j, 0] = 101.0  # lexicographic: boundary mismatch dominates shape diff

    x = helper.make_tensor_value_info("input", F, [1, 10, H, W])
    y = helper.make_tensor_value_info("output", F, [1, 10, H, W])

    init = [
        K("idx8", [8], I64),
        K("idx0", [0], I64),
        K("eff_idx", eff, I64),
        K("shape_1_900", [1, FULL], I64),
        K("shape_1_1_30_30", [1, 1, H, W], I64),
        K("shape_1_150", [1, N], I64),
        K("shape_150_O", [N, O], I64),
        K("shape_150_R", [N, R], I64),
        K("shape_1_900b", [1, FULL], I64),
        K("row_sel", row_sel, F),
        K("col_sel", col_sel, F),
        K("place", place, F),
        K("body_indices", body_indices, I64),
        K("ray_indices", ray_indices, I64),
        K("weights", weights, F),
        K("zero_scalar", [0.0], F),
        K("half", [0.5], F),
        K("one", [1.0], F),
        K("big", [10000.0], F),
        K("neg", [-1.0], F),
        K("zero_eff", np.zeros((1, N), dtype=np.float32), F),
        K("zero_chan", np.zeros((1, 1, H, W), dtype=np.float32), F),
        K("sentinel_zero", np.zeros((1, 1), dtype=np.float32), F),
    ]
    for j, idx in enumerate(shift_indices):
        init.append(K(f"shift_idx_{j}", idx, I64))

    nodes = []
    def node(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    # Extract color-8 mask and flatten the effective 15x10 ARC area.
    node("Gather", ["input", "idx8"], ["ch8"], axis=1)
    node("Gather", ["input", "idx0"], ["bg_in"], axis=1)
    node("Reshape", ["ch8", "shape_1_900"], ["flat900"])
    node("Gather", ["flat900", "eff_idx"], ["xeff"], axis=1)  # [1,150]
    node("Concat", ["xeff", "sentinel_zero"], ["xeff_pad"], axis=1)  # [1,151]

    # Non-empty rows.
    node("MatMul", ["xeff", "row_sel"], ["row_sum"])
    node("Greater", ["row_sum", "half"], ["row_nonempty_bool"])
    node("Cast", ["row_nonempty_bool"], ["row_nonempty"], to=F)

    interval_valids = []
    interval_nonvalids = []
    first_parts = []
    qmasks = []
    qfeatures = []

    def gather_rows(name, rows):
        cname = f"idx_rows_{name}"
        init.append(K(cname, np.array(rows, dtype=np.int64), I64))
        out = f"rows_{name}"
        node("Gather", ["row_nonempty", cname], [out], axis=1)
        return out

    def reduce_prod(name, inp):
        out = f"prod_{name}"
        node("ReduceProd", [inp], [out], axes=[1], keepdims=1)
        return out

    intervals = []
    for s in range(RH):
        for e in range(s, RH):
            intervals.append((s, e))

    for ii, (s, e) in enumerate(intervals):
        # valid row band: previous row empty, rows s..e nonempty, next row empty.
        # Separately track whether this is the first non-empty band.
        if s == 0:
            all_before = "one"
            prev_empty = "one"
        else:
            gr_all = gather_rows(f"allb_{ii}", list(range(0, s)))
            inv_all = f"all_before_inv_{ii}"
            node("Sub", ["one", gr_all], [inv_all])
            all_before = reduce_prod(f"all_before_{ii}", inv_all)
            gr_prev = gather_rows(f"prev_{ii}", [s - 1])
            prev_empty = f"prev_empty_{ii}"
            node("Sub", ["one", gr_prev], [prev_empty])
        gin = gather_rows(f"in_{ii}", list(range(s, e + 1)))
        inside = reduce_prod(f"inside_{ii}", gin)
        if e == RH - 1:
            after = "one"
        else:
            ga = gather_rows(f"after_{ii}", [e + 1])
            after = f"after_empty_{ii}"
            node("Sub", ["one", ga], [after])
        v1 = f"valid_a_{ii}"; v = f"valid_{ii}"
        node("Mul", [prev_empty, inside], [v1])
        node("Mul", [v1, after], [v])
        interval_valids.append(v)
        first = f"firstvalid_{ii}"
        node("Mul", [v, all_before], [first])
        first_parts.append(first)
        not_all_before = f"not_all_before_{ii}"
        node("Sub", ["one", all_before], [not_all_before])
        nv = f"nonvalid_{ii}"
        node("Mul", [v, not_all_before], [nv])
        interval_nonvalids.append(nv)

        # Column presence and rightmost column for this interval.
        sel = np.zeros((N, RW), dtype=np.float32)
        for rr in range(s, e + 1):
            for cc in range(RW):
                sel[rr * RW + cc, cc] = 1.0
        init.append(K(f"icol_sel_{ii}", sel, F))
        node("MatMul", ["xeff", f"icol_sel_{ii}"], [f"icol_sum_{ii}"])
        node("Greater", [f"icol_sum_{ii}", "half"], [f"icol_pres_bool_{ii}"])
        node("Cast", [f"icol_pres_bool_{ii}"], [f"icol_pres_{ii}"], to=F)
        right_parts = []
        for k in range(RW):
            init.append(K(f"idx_col_{ii}_{k}", [k], I64))
            node("Gather", [f"icol_pres_{ii}", f"idx_col_{ii}_{k}"], [f"cp_{ii}_{k}"], axis=1)
            if k == RW - 1:
                no_right = "one"
            else:
                init.append(K(f"idx_cols_gt_{ii}_{k}", np.arange(k + 1, RW, dtype=np.int64), I64))
                node("Gather", [f"icol_pres_{ii}", f"idx_cols_gt_{ii}_{k}"], [f"cgt_{ii}_{k}"], axis=1)
                node("Sub", ["one", f"cgt_{ii}_{k}"], [f"not_cgt_{ii}_{k}"])
                no_right = reduce_prod(f"no_right_{ii}_{k}", f"not_cgt_{ii}_{k}")
            node("Mul", [f"cp_{ii}_{k}", no_right], [f"right_{ii}_{k}"])
            right_parts.append(f"right_{ii}_{k}")
        node("Concat", right_parts, [f"right_oh_{ii}"], axis=1)  # [1,10]

        # qmask for topmost row in the band that reaches the rightmost col.
        qrow_reaches = []
        for rr in range(s, e + 1):
            row_vec = np.zeros((RW, 1), dtype=np.float32)
            # row value at the chosen right column = dot(right_onehot, x[row,:])
            init.append(K(f"rowvec_{ii}_{rr}", np.eye(RW, dtype=np.float32), F))
            # Gather row cells through a fixed geometric row-to-column matrix.
            selrow = np.zeros((N, RW), dtype=np.float32)
            for cc in range(RW):
                selrow[rr * RW + cc, cc] = 1.0
            init.append(K(f"selrow_{ii}_{rr}", selrow, F))
            node("MatMul", ["xeff", f"selrow_{ii}_{rr}"], [f"rowcells_{ii}_{rr}"])
            node("Mul", [f"rowcells_{ii}_{rr}", f"right_oh_{ii}"], [f"row_right_mul_{ii}_{rr}"])
            node("ReduceSum", [f"row_right_mul_{ii}_{rr}"], [f"row_reach_{ii}_{rr}"], axes=[1], keepdims=1)
            qrow_reaches.append((rr, f"row_reach_{ii}_{rr}"))
        qmask_terms = []
        for pos, (rr, reach_name) in enumerate(qrow_reaches):
            if pos == 0:
                no_prev = "one"
            else:
                prev_names = [nm for _, nm in qrow_reaches[:pos]]
                node("Concat", prev_names, [f"prev_reach_{ii}_{rr}"], axis=1)
                node("Sub", ["one", f"prev_reach_{ii}_{rr}"], [f"not_prev_reach_{ii}_{rr}"])
                no_prev = reduce_prod(f"no_prev_reach_{ii}_{rr}", f"not_prev_reach_{ii}_{rr}")
            qrow = f"qrow_{ii}_{rr}"
            node("Mul", [reach_name, no_prev], [qrow])
            # Place right_onehot in row rr and zero elsewhere.
            row_place = np.zeros((1, N), dtype=np.float32)
            for cc in range(RW):
                row_place[0, rr * RW + cc] = 1.0
            init.append(K(f"qrow_place_{ii}_{rr}", row_place, F))
            # Expand right_oh to this row through a constant scatter matrix [10,150].
            scatter = np.zeros((RW, N), dtype=np.float32)
            for cc in range(RW):
                scatter[cc, rr * RW + cc] = 1.0
            init.append(K(f"right_scatter_{ii}_{rr}", scatter, F))
            node("MatMul", [f"right_oh_{ii}", f"right_scatter_{ii}_{rr}"], [f"qrow_right_vec_{ii}_{rr}"])
            node("Mul", [qrow, f"qrow_right_vec_{ii}_{rr}"], [f"qrow_vec_{ii}_{rr}"])
            qmask_terms.append(f"qrow_vec_{ii}_{rr}")
        if qmask_terms:
            if len(qmask_terms) == 1:
                qbase = qmask_terms[0]
            else:
                node("Sum", qmask_terms, [f"qbase_{ii}"])
                qbase = f"qbase_{ii}"
        else:
            qbase = "zero_eff"
        qmask = f"qmask_{ii}"
        node("Mul", [qbase, nv], [qmask])
        qmasks.append(qmask)

        # Q body feature vector for this interval/corner.
        feats = []
        for oj, (dr, dc) in enumerate(body_offsets):
            idx = np.full((N,), SENT, dtype=np.int64)
            gate = np.zeros((1, N), dtype=np.float32)
            for qr in range(RH):
                for qc in range(RW):
                    rr, cc = qr + dr, qc + dc
                    qidx = qr * RW + qc
                    if 0 <= rr < RH and 0 <= cc < RW:
                        idx[qidx] = rr * RW + cc
                        if s <= rr <= e:
                            gate[0, qidx] = 1.0
            init.append(K(f"q_shift_idx_{ii}_{oj}", idx, I64))
            init.append(K(f"q_gate_{ii}_{oj}", gate, F))
            node("Gather", ["xeff_pad", f"q_shift_idx_{ii}_{oj}"], [f"q_shift_{ii}_{oj}"], axis=1)
            node("Mul", [qmask, f"q_shift_{ii}_{oj}"], [f"q_feat_a_{ii}_{oj}"])
            node("Mul", [f"q_feat_a_{ii}_{oj}", f"q_gate_{ii}_{oj}"], [f"q_feat_b_{ii}_{oj}"])
            node("ReduceSum", [f"q_feat_b_{ii}_{oj}"], [f"q_feat_{ii}_{oj}"], axes=[1], keepdims=1)
            feats.append(f"q_feat_{ii}_{oj}")
        node("Concat", feats, [f"qfeat_{ii}"], axis=1)
        qfeatures.append(f"qfeat_{ii}")

    # Template row/cell mask = first valid row band.
    template_terms = []
    for ii, (s, e) in enumerate(intervals):
        mask = np.zeros((1, N), dtype=np.float32)
        for rr in range(s, e + 1):
            for cc in range(RW):
                mask[0, rr * RW + cc] = 1.0
        init.append(K(f"templ_mask_const_{ii}", mask, F))
        node("Mul", [first_parts[ii], f"templ_mask_const_{ii}"], [f"templ_part_{ii}"])
        template_terms.append(f"templ_part_{ii}")
    node("Sum", template_terms, ["template_rowcell_mask"])
    node("Mul", ["xeff", "template_rowcell_mask"], ["teff"])
    node("Concat", ["teff", "sentinel_zero"], ["teff_pad"], axis=1)

    # Template rightmost column and candidate split cells.
    node("MatMul", ["teff", "col_sel"], ["tcol_sum"])
    node("Greater", ["tcol_sum", "half"], ["tcol_pres_bool"])
    node("Cast", ["tcol_pres_bool"], ["tcol_pres"], to=F)
    tright_parts = []
    for k in range(RW):
        init.append(K(f"idx_tcol_{k}", [k], I64))
        node("Gather", ["tcol_pres", f"idx_tcol_{k}"], [f"tcp_{k}"], axis=1)
        if k == RW - 1:
            nr = "one"
        else:
            init.append(K(f"idx_tcols_gt_{k}", np.arange(k + 1, RW, dtype=np.int64), I64))
            node("Gather", ["tcol_pres", f"idx_tcols_gt_{k}"], [f"tcgt_{k}"], axis=1)
            node("Sub", ["one", f"tcgt_{k}"], [f"not_tcgt_{k}"])
            nr = reduce_prod(f"t_no_right_{k}", f"not_tcgt_{k}")
        node("Mul", [f"tcp_{k}", nr], [f"tright_{k}"])
        tright_parts.append(f"tright_{k}")
    node("Concat", tright_parts, ["tright_oh"], axis=1)
    # col_lt_right[c] = any rightmost column greater than c.
    lt_mat = np.zeros((RW, RW), dtype=np.float32)
    for k in range(RW):
        for c in range(RW):
            if c < k:
                lt_mat[k, c] = 1.0
    init.append(K("lt_mat", lt_mat, F))
    node("MatMul", ["tright_oh", "lt_mat"], ["col_lt_right"])
    # Expand col_lt_right to cells.
    col_scatter = np.zeros((RW, N), dtype=np.float32)
    for r in range(RH):
        for c in range(RW):
            col_scatter[c, r * RW + c] = 1.0
    init.append(K("col_scatter", col_scatter, F))
    node("MatMul", ["col_lt_right", "col_scatter"], ["cell_lt_right"])
    node("Mul", ["teff", "cell_lt_right"], ["pvalid"])

    # Body feature matrix for every possible split point.
    node("Gather", ["teff_pad", "body_indices"], ["B3"], axis=1)  # [1,150,O]
    node("Reshape", ["B3", "shape_150_O"], ["B"])
    node("Gather", ["teff_pad", "ray_indices"], ["Ray3"], axis=1)
    node("Reshape", ["Ray3", "shape_150_R"], ["Ray"])

    # Q feature matrix [intervals,O] and interval validity [1,B].
    node("Concat", qfeatures, ["Q3"], axis=0)  # [B,O]
    node("Concat", interval_nonvalids, ["Qvalid"], axis=1)  # [1,B]

    # Weighted symmetric difference scores.
    node("MatMul", ["B", "weights"], ["Bsum"])     # [150,1]
    node("MatMul", ["Q3", "weights"], ["Qsum_col"]) # [B,1]
    node("Transpose", ["Qsum_col"], ["Qsum"], perm=[1,0])
    # Cross term: (B*weights)^T Q.
    node("Mul", ["B", "weights_row"], ["Bw"])
    # weights_row constant is added below after code generation.
    init.append(K("weights_row", weights.reshape(1, O), F))
    node("Transpose", ["Q3"], ["QT"], perm=[1,0])
    node("MatMul", ["Bw", "QT"], ["cross"])
    node("Add", ["Bsum", "Qsum"], ["score_base_a"])
    node("Add", ["cross", "cross"], ["two_cross"])
    node("Sub", ["score_base_a", "two_cross"], ["score_base"])
    # Add invalid penalties.
    node("Sub", ["one", "pvalid"], ["p_invalid"])
    node("Transpose", ["p_invalid"], ["p_invalid_col"], perm=[1,0])
    node("Mul", ["p_invalid_col", "big"], ["p_pen"])
    node("Sub", ["one", "Qvalid"], ["q_invalid"])
    node("Mul", ["q_invalid", "big"], ["q_pen"])
    node("Add", ["score_base", "p_pen"], ["score_p"])
    node("Add", ["score_p", "q_pen"], ["score"])
    node("ReduceMin", ["score"], ["score_min_p"], axes=[1], keepdims=1)
    node("ReduceMin", ["score_min_p"], ["best_score"], axes=[0,1], keepdims=1)
    node("Equal", ["score_min_p", "best_score"], ["pbest_bool"])
    node("Cast", ["pbest_bool"], ["pbest_f"], to=F)
    node("Transpose", ["pvalid"], ["pvalid_col"], perm=[1,0])
    node("Mul", ["pbest_f", "pvalid_col"], ["pselect_col"])
    node("Transpose", ["pselect_col"], ["pselect"], perm=[1,0])

    # Ray union from all tied best split points.
    node("MatMul", ["pselect", "Ray"], ["ray_raw"])
    node("Greater", ["ray_raw", "half"], ["ray_bool"])
    node("Cast", ["ray_bool"], ["ray_union"], to=F)  # [1,R]

    # Global non-template corner mask.
    node("Sum", qmasks, ["qmask_sum"])
    node("Greater", ["qmask_sum", "half"], ["qmask_bool"])
    node("Cast", ["qmask_bool"], ["qmask"], to=F)
    node("Concat", ["qmask", "sentinel_zero"], ["qmask_pad"], axis=1)

    shifted_terms = []
    for j in range(R):
        init.append(K(f"idx_ray_{j}", [j], I64))
        node("Gather", ["ray_union", f"idx_ray_{j}"], [f"rayv_{j}"], axis=1)
        node("Gather", ["qmask_pad", f"shift_idx_{j}"], [f"qshift_{j}"], axis=1)
        node("Mul", [f"qshift_{j}", f"rayv_{j}"], [f"add_term_{j}"])
        shifted_terms.append(f"add_term_{j}")
    node("Sum", shifted_terms, ["add_eff_sum"])
    node("Greater", ["add_eff_sum", "half"], ["add_eff_bool"])
    node("Cast", ["add_eff_bool"], ["add_eff_pre"], to=F)
    # Only paint empty cells.
    # Gather effective background from input channel 0.
    node("Reshape", ["bg_in", "shape_1_900b"], ["bg_flat900"])
    node("Gather", ["bg_flat900", "eff_idx"], ["bg_eff"], axis=1)
    node("Mul", ["add_eff_pre", "bg_eff"], ["add_eff"])
    node("MatMul", ["add_eff", "place"], ["add_flat900"])
    node("Reshape", ["add_flat900", "shape_1_1_30_30"], ["add_ch"])

    # Build output channels: background minus added cells, channel 1 additions, channel 8 original.
    node("Sub", ["bg_in", "add_ch"], ["bg_out"])
    channels = ["bg_out", "add_ch"]
    for c in range(2, 8):
        channels.append("zero_chan")
    channels.append("ch8")
    channels.append("zero_chan")
    node("Concat", channels, ["output"], axis=1)

    graph = helper.make_graph(nodes, "task219_ray_completion", [x], [y], init)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 12)])
    model.ir_version = 8
    return model


model = build_onnx_model()
