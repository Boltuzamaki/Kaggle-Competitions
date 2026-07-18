import json, csv, os, math
import numpy as np

TASK_DIR = r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\data"
REPAIRS_DIR = os.path.dirname(os.path.abspath(__file__))

def A(grid):
    return np.array(grid, dtype=np.int64)

def all_examples(t):
    d = json.load(open(os.path.join(TASK_DIR, f"task{t:03d}.json")))
    return [(A(e["input"]), A(e["output"])) for e in d["train"] + d["test"] + d["arc-gen"]]

def try_identity(exs):
    return all(np.array_equal(i, o) for i, o in exs)

def try_geometric(exs):
    """reflections, rotations, transpose"""
    transforms = {
        "flip_h": lambda g: g[:, ::-1],
        "flip_v": lambda g: g[::-1, :],
        "rot90": lambda g: np.rot90(g, 1),
        "rot180": lambda g: np.rot90(g, 2),
        "rot270": lambda g: np.rot90(g, 3),
        "transpose": lambda g: g.T,
        "anti_transpose": lambda g: np.rot90(g, 2).T,
    }
    hits = []
    for name, fn in transforms.items():
        if all(i.shape[::-1] == o.shape or i.shape == o.shape for i, o in exs):
            if all(fn(i).shape == o.shape and np.array_equal(fn(i), o) for i, o in exs):
                hits.append(name)
    return hits

def try_crop_offset(exs):
    """output is a fixed sub-rectangle of input at a consistent (r0,c0) offset"""
    ihapes = set(i.shape for i, o in exs)
    oshapes = set(o.shape for i, o in exs)
    if len(ihapes) != 1 or len(oshapes) != 1:
        return None
    ih, iw = ihapes.pop()
    oh, ow = oshapes.pop()
    if oh > ih or ow > iw:
        return None
    for r0 in range(ih - oh + 1):
        for c0 in range(iw - ow + 1):
            if all(np.array_equal(i[r0:r0+oh, c0:c0+ow], o) for i, o in exs):
                return (r0, c0, oh, ow, ih, iw)
    return None

def try_color_remap(exs):
    """same shape, pure per-cell color mapping"""
    if not all(i.shape == o.shape for i, o in exs):
        return None
    mapping = {}
    for i, o in exs:
        for a, b in zip(i.flat, o.flat):
            a, b = int(a), int(b)
            if a in mapping and mapping[a] != b:
                return None
            mapping[a] = b
    return mapping

def try_tile(exs):
    """output is input tiled ky x kx times"""
    for i, o in exs[:1]:
        if o.shape[0] % i.shape[0] or o.shape[1] % i.shape[1]:
            return None
        ky, kx = o.shape[0] // i.shape[0], o.shape[1] // i.shape[1]
        if ky < 1 or kx < 1 or (ky == 1 and kx == 1):
            return None
    def tile_ok(ky, kx):
        for i, o in exs:
            if o.shape != (i.shape[0]*ky, i.shape[1]*kx):
                return False
            if not np.array_equal(np.tile(i, (ky, kx)), o):
                return False
        return True
    i0, o0 = exs[0]
    ky, kx = o0.shape[0]//i0.shape[0], o0.shape[1]//i0.shape[1]
    return (ky, kx) if tile_ok(ky, kx) else None

def main():
    rows = list(csv.DictReader(open(os.path.join(REPAIRS_DIR, "cost_profile.csv"))))
    cost_map = {int(r["task"]): (float(r["cost"]), float(r["points"])) for r in rows}
    cost_map[240] = (2651, 17.12)  # post-fix

    found = []
    for t in range(1, 401):
        try:
            exs = all_examples(t)
        except Exception:
            continue
        cost, pts = cost_map.get(t, (0, 25))
        rule = None
        if try_identity(exs):
            rule = ("identity", None)
        elif (g := try_geometric(exs)):
            rule = ("geometric", g[0])
        elif (c := try_crop_offset(exs)):
            rule = ("crop", c)
        elif (tl := try_tile(exs)):
            rule = ("tile", tl)
        elif (m := try_color_remap(exs)) is not None and len(set(m.values())) > 0:
            # only flag remap if non-trivial (not identity map) and current cost is high
            if any(k != v for k, v in m.items()):
                rule = ("color_remap", m)
        if rule and cost > 500:  # only worth golfing if current cost is non-trivial
            found.append((t, cost, pts, rule))

    found.sort(key=lambda x: -x[1])
    print(f"Tasks with a simple rule + current cost > 500: {len(found)}\n")
    est_gain = 0.0
    for t, cost, pts, (kind, detail) in found:
        # crude estimate: a minimal graph costs ~a few hundred bytes -> ~19 pts
        est_pts = 19.0 if kind in ("crop", "geometric", "tile") else pts
        gain = max(0, est_pts - pts)
        est_gain += gain
        d = detail if kind != "color_remap" else f"{len(detail)} colors"
        print(f"  task{t:03d}: cost={cost:>8.0f} pts={pts:5.2f}  rule={kind}({d})  est_gain~+{gain:.1f}")
    print(f"\nRough total estimated gain from these: +{est_gain:.1f} points")

if __name__ == "__main__":
    main()
