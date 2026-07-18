# ============================================================================
# task004 — REFERENCE ALGORITHM (verified correct: 265/265 examples).
# NOTE: this is a plain-Python grid transform, NOT an ONNX graph. It will NOT
# "Run to green" here (the app needs code that assigns an onnx model to `model`).
# We keep task004 on the BASELINE because hoangvux implements the SAME result in
# just 19 nodes via a local Conv trick (cost 5394, 16.41 pts) — with NO connected
# components — whereas this CC-based approach would compile to a much pricier
# unrolled-labeling ONNX graph (~12-13 pts). Kept here as documentation.
# Rule: for each 8-connected object, shift its boundaries right and redraw it as
# an outline (top/bottom rows solid, middle rows hollow walls).
# ============================================================================

def solve_task4(input_grid):
    grid = [list(row) for row in input_grid]
    rows = len(grid); cols = len(grid[0])

    # 1. connected components (8-connected), grouped by colour
    visited = set(); components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and (r, c) not in visited:
                color = grid[r][c]; comp = []; q = [(r, c)]; visited.add((r, c))
                while q:
                    cr, cc = q.pop(0); comp.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == color and (nr, nc) not in visited:
                            visited.add((nr, nc)); q.append((nr, nc))
                components.append((color, comp))

    # 2. redraw each object shifted + outlined
    new_grid = [[0]*cols for _ in range(rows)]
    for color, comp in components:
        r_min = min(r for r, c in comp); r_max = max(r for r, c in comp); H = r_max - r_min + 1
        for r in range(r_min, r_max + 1):
            row_cells = [c for cr, c in comp if cr == r]
            if not row_cells: continue
            c_min = min(row_cells); c_max = max(row_cells); row_idx = r - r_min
            new_c_min = c_min + 1 if row_idx < H - 1 else c_min
            new_c_max = c_max + 1 if row_idx < H - 2 else c_max
            new_c_min = min(new_c_min, cols - 1); new_c_max = min(new_c_max, cols - 1)
            if row_idx == 0 or row_idx == H - 1:          # solid top / bottom
                for c in range(new_c_min, new_c_max + 1): new_grid[r][c] = color
            else:                                          # hollow walls
                new_grid[r][new_c_min] = color; new_grid[r][new_c_max] = color
    return new_grid

# --- To make this a submittable ONNX we'd need connected-component labeling in a
# --- static graph (unrolled) + per-object row logic — correct but pricier than
# --- baseline, so it's intentionally NOT compiled. task004 stays on baseline.
