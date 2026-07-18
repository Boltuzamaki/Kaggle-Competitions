# task233 genuine rule reference (not a memorized selector)
# This is the exact non-cheating logic: crop the main color-2 board, then use the
# external 3x3 stencils to decode the board's zero-hole patterns.
#
# NOTE: This file is a NumPy rule reference. It does not build an ONNX graph.

import numpy as np


def _components(mask):
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _d4_pairs(two_mask, color_mask):
    pairs = []
    for k in range(4):
        t = np.rot90(two_mask, k)
        c = np.rot90(color_mask, k)
        pairs.append((t, c, k, 0))
        pairs.append((np.fliplr(t), np.fliplr(c), k, 1))

    out = []
    for t, c, k, f in pairs:
        if not any(np.array_equal(t, t2) and np.array_equal(c, c2) for t2, c2, _, _ in out):
            out.append((t, c, k, f))
    return out


def solve_grid(grid):
    a = np.asarray(grid, dtype=np.int64)

    # Main board: the largest connected component of color 2.
    twos = _components(a == 2)
    board = max(twos, key=len)
    board_set = set(board)
    brs = [r for r, _ in board]
    bcs = [c for _, c in board]
    r0, r1 = min(brs), max(brs)
    c0, c1 = min(bcs), max(bcs)
    crop = a[r0:r1 + 1, c0:c1 + 1]

    # External 3x3 stencils: each has color c plus color 2.
    stencils = []
    sid = 0
    for comp in _components(a != 0):
        if any(cell in board_set for cell in comp):
            continue
        vals = {int(a[r, c]) for r, c in comp}
        colors = [v for v in vals if v not in (0, 2)]
        if not colors:
            continue
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        patch = a[min(rs):max(rs) + 1, min(cs):max(cs) + 1]
        if patch.shape != (3, 3):
            continue
        color = max(colors, key=lambda v: int(np.sum(patch == v)))
        stencils.append((sid, color, (patch == 2).astype(np.uint8), (patch == color).astype(np.uint8)))
        sid += 1

    zeros = frozenset(map(tuple, np.argwhere(crop == 0)))
    H, W = crop.shape
    matches = []

    # Candidate placement: a transformed stencil's color-2 mask must exactly match
    # a 3x3 zero pattern in the main board crop. Its colored mask is then pasted.
    for r in range(H - 2):
        for c in range(W - 2):
            z = (crop[r:r + 3, c:c + 3] == 0).astype(np.uint8)
            if int(z.sum()) == 0:
                continue
            for sid, color, two_mask, color_mask in stencils:
                for tmask, cmask, rot, flip in _d4_pairs(two_mask, color_mask):
                    if not np.array_equal(z, tmask):
                        continue
                    zcells = frozenset(
                        (r + i, c + j)
                        for i in range(3)
                        for j in range(3)
                        if tmask[i, j]
                    )
                    fcells = tuple(
                        (r + i, c + j)
                        for i in range(3)
                        for j in range(3)
                        if cmask[i, j]
                    )
                    # Canonical transform preference: non-flipped beats flipped;
                    # smaller rotation beats larger rotation. Used only to break
                    # ambiguous exact covers, never to memorize examples.
                    score = (1000 if flip == 0 else 0) + (100 - rot) * 10
                    matches.append({
                        "sid": sid,
                        "color": color,
                        "zcells": zcells,
                        "fcells": fcells,
                        "score": score,
                    })

    by_zero = {z: [] for z in zeros}
    for i, m in enumerate(matches):
        for z in m["zcells"]:
            if z in by_zero:
                by_zero[z].append(i)

    best = None
    best_score = -10**18

    def search(remaining, used_stencils, chosen, score):
        nonlocal best, best_score
        if not remaining:
            if score > best_score:
                best_score = score
                best = chosen[:]
            return
        cell = min(
            remaining,
            key=lambda z: sum(
                1
                for mi in by_zero.get(z, [])
                if matches[mi]["sid"] not in used_stencils and matches[mi]["zcells"] <= remaining
            ),
        )
        cands = [
            mi
            for mi in by_zero.get(cell, [])
            if matches[mi]["sid"] not in used_stencils and matches[mi]["zcells"] <= remaining
        ]
        for mi in sorted(cands, key=lambda x: matches[x]["score"], reverse=True):
            m = matches[mi]
            search(
                remaining - m["zcells"],
                used_stencils | {m["sid"]},
                chosen + [mi],
                score + m["score"],
            )

    search(zeros, set(), [], 0)

    out = np.full(crop.shape, 2, dtype=np.int64)
    if best is not None:
        for mi in best:
            m = matches[mi]
            for r, c in m["fcells"]:
                out[r, c] = m["color"]
    return out.tolist()
