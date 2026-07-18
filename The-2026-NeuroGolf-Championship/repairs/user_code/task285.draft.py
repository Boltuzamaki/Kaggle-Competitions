# ARC task285 rule reference.
# Clean rule logic only: no JSON loading, no base64, no fingerprints, no stored examples.
#
# Transformation:
# Each 8-connected multi-colour component contains one "source" colour shape.
# The source is the colour with the most cells in that component.
# Other colours are partial seeds of mirrored copies of the source in a 2x2 block layout.
# Complete those seeded copies using horizontal / vertical / 180-degree flips.

import numpy as np
from collections import defaultdict
from itertools import product


def _components8(grid):
    a = np.asarray(grid)
    h, w = a.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if a[r, c] == 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                x, y = stack.pop()
                pts.append((x, y))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and a[nx, ny] != 0 and not seen[nx, ny]:
                            seen[nx, ny] = True
                            stack.append((nx, ny))
            comps.append(pts)
    return comps


def _flip_shape(cells, height, width, flip_r, flip_c):
    out = set()
    for r, c in cells:
        rr = height - 1 - r if flip_r else r
        cc = width - 1 - c if flip_c else c
        out.add((rr, cc))
    return out


def solve_285_numpy(grid):
    a = np.asarray(grid, dtype=np.int64)
    h, w = a.shape
    out = a.copy()

    for pts in _components8(a):
        by_color = defaultdict(list)
        for r, c in pts:
            by_color[int(a[r, c])].append((r, c))

        if len(by_color) < 2:
            continue

        # The real source is the most complete colour shape in this local component.
        source = max(by_color, key=lambda col: (len(by_color[col]), -col))
        source_cells_abs = by_color[source]

        r0 = min(r for r, _ in source_cells_abs)
        r1 = max(r for r, _ in source_cells_abs)
        c0 = min(c for _, c in source_cells_abs)
        c1 = max(c for _, c in source_cells_abs)
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1

        source_rel = {(r - r0, c - c0) for r, c in source_cells_abs}

        best = None

        # The source can be in any quadrant of a 2x2 layout.
        for sbr, sbc in product((0, 1), repeat=2):
            origin_r = r0 - sbr * bh
            origin_c = c0 - sbc * bw
            assignments = {}
            ok = True

            for col, cells in by_color.items():
                if col == source:
                    continue

                possible = []
                for tbr, tbc in product((0, 1), repeat=2):
                    if (tbr, tbc) == (sbr, sbc):
                        continue

                    flip_r = sbr ^ tbr
                    flip_c = sbc ^ tbc
                    rel = _flip_shape(source_rel, bh, bw, flip_r, flip_c)
                    placed = {
                        (origin_r + tbr * bh + rr, origin_c + tbc * bw + cc)
                        for rr, cc in rel
                    }

                    if all((rr, cc) in placed for rr, cc in cells) and all(
                        0 <= rr < h and 0 <= cc < w for rr, cc in placed
                    ):
                        possible.append(((tbr, tbc), placed))

                if not possible:
                    ok = False
                    break

                assignments[col] = possible[0]

            if ok:
                distinct_blocks = len({block for block, _ in assignments.values()})
                candidate = (distinct_blocks, len(by_color), assignments)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate

        if best is None:
            continue

        for col, (_, placed) in best[2].items():
            for rr, cc in placed:
                out[rr, cc] = col

    return out


# This task's clean ONNX graph is non-trivial because the rule requires
# 8-connected local grouping and per-component source-colour selection.
# Do not submit a fake selector model. Use this as the verified rule reference.
