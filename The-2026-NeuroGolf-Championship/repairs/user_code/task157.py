import json
import os
from collections import deque

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    groups = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rs = [rr for rr, _ in cells]
            cs = [cc for _, cc in cells]
            groups.append((min(rs), min(cs), mask[min(rs) : max(rs) + 1, min(cs) : max(cs) + 1]))
    return groups


def solve(grid):
    grid = np.asarray(grid, dtype=np.int64)
    out = np.where(grid == 5, 0, grid).copy()
    top_holes = grid[:3] == 0
    glyphs = components(grid[6:] == 5)

    candidates = []
    for _, _, glyph in glyphs:
        gh, gw = glyph.shape
        opts = []
        for rr in range(1, 3):
            for cc in range(-gw + 1, 15):
                cover = np.zeros((3, 15), dtype=bool)
                ok = True
                any_cover = False
                for gr in range(gh):
                    for gc in range(gw):
                        if not glyph[gr, gc]:
                            continue
                        ar, ac = rr + gr, cc + gc
                        if 0 <= ar < 3:
                            if not (0 <= ac < 15) or not top_holes[ar, ac]:
                                ok = False
                            else:
                                cover[ar, ac] = True
                                any_cover = True
                if ok and any_cover:
                    opts.append((rr, cc, cover, glyph))
        candidates.append(opts)

    order = sorted(range(len(glyphs)), key=lambda i: len(candidates[i]))
    placement = None

    def search(k, used, placed):
        nonlocal placement
        if placement is not None:
            return
        if k == len(order):
            if np.array_equal(used, top_holes):
                placement = placed[:]
            return
        gi = order[k]
        for rr, cc, cover, glyph in candidates[gi]:
            if np.any(used & cover):
                continue
            placed.append((rr, cc, glyph))
            search(k + 1, used | cover, placed)
            placed.pop()

    search(0, np.zeros((3, 15), dtype=bool), [])
    if placement is None:
        return out

    for rr, cc, glyph in placement:
        gh, gw = glyph.shape
        for gr in range(gh):
            for gc in range(gw):
                ar, ac = rr + gr, cc + gc
                if glyph[gr, gc] and 0 <= ar < 10 and 0 <= ac < 15:
                    out[ar, ac] = 1
    return out


def check():
    with open(os.path.join(ROOT, "data", "task157.json"), encoding="utf-8") as f:
        data = json.load(f)
    wrong = 0
    for split in ("train", "test", "arc-gen"):
        for i, ex in enumerate(data[split]):
            pred = solve(ex["input"])
            if not np.array_equal(pred, np.asarray(ex["output"], dtype=np.int64)):
                print(f"{split} {i} failed")
                wrong += 1
    print(f"python check: {wrong} fail")
    return wrong == 0


if __name__ == "__main__":
    raise SystemExit(0 if check() else 1)
