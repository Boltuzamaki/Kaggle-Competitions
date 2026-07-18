import json, csv, os
import numpy as np

TASK_DIR = r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\data"
REPAIRS_DIR = os.path.dirname(os.path.abspath(__file__))

def A(g): return np.array(g, dtype=np.int64)

def all_examples(t):
    d = json.load(open(os.path.join(TASK_DIR, f"task{t:03d}.json")))
    return [(A(e["input"]), A(e["output"])) for e in d["train"] + d["test"] + d["arc-gen"]]

def bbox(g, bg=0):
    ys, xs = np.where(g != bg)
    if len(ys) == 0: return None
    return ys.min(), ys.max()+1, xs.min(), xs.max()+1

def rule_bbox_crop(exs):
    for i, o in exs:
        b = bbox(i)
        if b is None: return False
        r0, r1, c0, c1 = b
        if not (i[r0:r1, c0:c1].shape == o.shape and np.array_equal(i[r0:r1, c0:c1], o)):
            return False
    return True

def rule_constant(exs):
    o0 = exs[0][1]
    return all(o.shape == o0.shape and np.array_equal(o, o0) for _, o in exs)

def rule_subsample(exs):
    """O[i,j] = I[i*ky, j*kx]"""
    i0, o0 = exs[0]
    if o0.shape[0] == 0 or o0.shape[1] == 0: return None
    if i0.shape[0] % o0.shape[0] or i0.shape[1] % o0.shape[1]: return None
    ky, kx = i0.shape[0]//o0.shape[0], i0.shape[1]//o0.shape[1]
    if ky < 1 or kx < 1 or (ky == 1 and kx == 1): return None
    for i, o in exs:
        if i.shape != (o.shape[0]*ky, o.shape[1]*kx): return None
        if not np.array_equal(i[::ky, ::kx], o): return None
    return (ky, kx)

def rule_upscale(exs):
    """O = kron(I, ones(ky,kx))"""
    i0, o0 = exs[0]
    if i0.shape[0] == 0 or i0.shape[1] == 0: return None
    if o0.shape[0] % i0.shape[0] or o0.shape[1] % i0.shape[1]: return None
    ky, kx = o0.shape[0]//i0.shape[0], o0.shape[1]//i0.shape[1]
    if ky < 1 or kx < 1 or (ky == 1 and kx == 1): return None
    for i, o in exs:
        up = np.kron(i, np.ones((ky, kx), dtype=np.int64))
        if up.shape != o.shape or not np.array_equal(up, o): return None
    return (ky, kx)

def rule_block_reduce(exs):
    """O[i,j] = mode/first of I block (ky,kx) -- try taking top-left of each block already covered by subsample; here test majority"""
    i0, o0 = exs[0]
    if o0.shape[0]==0 or o0.shape[1]==0: return None
    if i0.shape[0] % o0.shape[0] or i0.shape[1] % o0.shape[1]: return None
    ky, kx = i0.shape[0]//o0.shape[0], i0.shape[1]//o0.shape[1]
    if ky<1 or kx<1 or (ky==1 and kx==1): return None
    for i, o in exs:
        if i.shape != (o.shape[0]*ky, o.shape[1]*kx): return None
        for r in range(o.shape[0]):
            for c in range(o.shape[1]):
                block = i[r*ky:(r+1)*ky, c*kx:(c+1)*kx]
                vals, counts = np.unique(block, return_counts=True)
                if vals[np.argmax(counts)] != o[r, c]:
                    return None
    return (ky, kx)

def main():
    rows = list(csv.DictReader(open(os.path.join(REPAIRS_DIR, "cost_profile.csv"))))
    cost_map = {int(r["task"]): (float(r["cost"]), float(r["points"])) for r in rows}
    cost_map[240] = (2651, 17.12); cost_map[135] = (374, 19.08)

    found = []
    for t in range(1, 401):
        try:
            exs = all_examples(t)
        except Exception:
            continue
        cost, pts = cost_map.get(t, (0, 25))
        if cost <= 500:
            continue
        rule = None
        if rule_constant(exs): rule = ("constant", None)
        elif rule_bbox_crop(exs): rule = ("bbox_crop", None)
        elif (s := rule_subsample(exs)): rule = ("subsample", s)
        elif (u := rule_upscale(exs)): rule = ("upscale", u)
        elif (b := rule_block_reduce(exs)): rule = ("block_majority", b)
        if rule:
            found.append((t, cost, pts, rule))

    found.sort(key=lambda x: -x[1])
    print(f"Tasks with an (expanded) simple rule + cost > 500: {len(found)}\n")
    for t, cost, pts, (kind, detail) in found:
        print(f"  task{t:03d}: cost={cost:>8.0f} pts={pts:5.2f}  rule={kind}({detail})")

if __name__ == "__main__":
    main()
