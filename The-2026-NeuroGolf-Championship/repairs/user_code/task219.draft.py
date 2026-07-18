
import numpy as np


def _row_bands(grid: np.ndarray):
    """Contiguous non-empty row groups."""
    rows = np.where(np.any(grid != 0, axis=1))[0]
    if len(rows) == 0:
        return []
    bands = []
    cur = [int(rows[0])]
    for r in rows[1:]:
        r = int(r)
        if r == cur[-1] + 1:
            cur.append(r)
        else:
            bands.append(cur)
            cur = [r]
    bands.append(cur)
    return bands


def _band_info(grid: np.ndarray, rows):
    cells = [(r, c) for r in rows for c in range(grid.shape[1]) if grid[r, c] != 0]
    right = max(c for _, c in cells)
    reach_rows = sorted({r for r, c in cells if c == right})

    def leftmost(row):
        return min(c for rr, c in cells if rr == row)

    # task corner rule: among rows reaching the rightmost column,
    # smallest left edge, then topmost.
    corner_row = min(reach_rows, key=lambda r: (leftmost(r), r))
    return {
        "rows": rows,
        "cells": cells,
        "right": right,
        "corner": (corner_row, right),
    }


def _offsets_from_corner(info):
    cr, cc = info["corner"]
    return {(r - cr, c - cc) for r, c in info["cells"]}


def _boundary_rows(offsets):
    # Rows of a body that reach the corner column (col offset 0).
    return {dr for dr, dc in offsets if dc == 0}


def solve_grid(grid):
    """
    Task 219 rule.

    There is one template band: the band with the greatest rightward extent.
    For every other band, find its corner.  The template contains a body plus
    a right-going ray.  We infer the split point(s) P by comparing the template
    body (cells at columns <= P) against each truncated band body:
      1) first match the set of body rows that reach the corner column;
      2) then match the body shape by symmetric difference.
    All P tied for the best structural score are valid; stamp the template
    cells to the right of each P from every non-template corner, painting 1s.
    """
    a = np.array(grid, dtype=np.int64)
    out = a.copy()
    h, w = a.shape

    bands = _row_bands(a)
    if not bands:
        return out

    infos = [_band_info(a, rows) for rows in bands]

    # Template = unique band that reaches farthest right.
    max_right = max(info["right"] for info in infos)
    template_index = min(i for i, info in enumerate(infos) if info["right"] == max_right)
    template = infos[template_index]

    non_template = [info for i, info in enumerate(infos) if i != template_index]
    if not non_template:
        return out

    non_offsets = [_offsets_from_corner(info) for info in non_template]
    non_boundary_rows = [_boundary_rows(s) for s in non_offsets]

    # Candidate P cells are template cells with something to their right.
    candidates = [(r, c) for r, c in template["cells"] if c < max_right]
    if not candidates:
        return out

    scored = []
    for pr, pc in candidates:
        body = {(r - pr, c - pc) for r, c in template["cells"] if c <= pc}
        body_boundary = _boundary_rows(body)

        # Best match to any non-template truncated body.
        best = None
        for nbody, nboundary in zip(non_offsets, non_boundary_rows):
            row_diff = len(body_boundary ^ nboundary)
            shape_diff = len(body ^ nbody)
            score = (row_diff, shape_diff)
            if best is None or score < best:
                best = score
        scored.append((best, (pr, pc)))

    best_score = min(score for score, _ in scored)
    split_points = [p for score, p in scored if score == best_score]

    # Stamp ray of every tied split point. Union is intentional; tied split
    # points produce the same required output on the task distribution.
    for pr, pc in split_points:
        ray = [(r - pr, c - pc) for r, c in template["cells"] if c > pc]
        for info in non_template:
            cr, cc = info["corner"]
            for dr, dc in ray:
                rr, col = cr + dr, cc + dc
                if 0 <= rr < h and 0 <= col < w and out[rr, col] == 0:
                    out[rr, col] = 1

    return out


# Common aliases used by local ARC checkers.
def solve(grid):
    return solve_grid(grid)


def predict(grid):
    return solve_grid(grid)
