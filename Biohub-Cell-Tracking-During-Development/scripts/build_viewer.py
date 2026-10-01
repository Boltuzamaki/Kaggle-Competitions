import sys, json, base64, io, warnings
sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
import biohub_ct as B

NAME = "44b6_0113de3b"
vol = B.open_volume(f"data/comp/test/{NAME}.zarr")
gt  = B.read_geff(f"data/comp/train/{NAME}.geff")
T, Z, Y, X = vol.shape
print("volume", vol.shape, "| GT nodes", gt.n_nodes, "| est cells", gt.meta.get("estimated_number_of_nodes"))

cfg = B.Config()
frames, projs_xy, projs_zy = [], [], []
lo = hi = None
for t in range(T):
    raw = vol.frame(t)
    if lo is None:
        lo, hi = np.percentile(raw, [1.0, 99.5])
    # z-max projection (top view) and y-max projection (side view)
    for arr, store, w, h in ((raw.max(0), projs_xy, 128, 128), (raw.max(1), projs_zy, 128, 64)):
        a = np.clip((arr.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
        im = Image.fromarray((a * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
        store.append(base64.b64encode(buf.getvalue()).decode())
    c, _ = B.detect_dog(raw, vol.quantiles, cfg.xy_downsample, cfg.dog_scales,
                        cfg.min_distance_um, cfg.rel_threshold, cfg.max_peaks)
    if len(c): c = B.refine_centroids(raw, c)
    frames.append(c)
    if (t+1) % 25 == 0: print(f"  {t+1}/{T}")

g = B.build_graph(frames, cfg)
print("graph:", g.n_nodes, "nodes", g.n_edges, "edges")

# assign a track id per connected component
comp = B._components(g)
tid = {}
for k, (root, members) in enumerate(comp.items()):
    for m in members: tid[m] = k

by_t = {}
for nid, t, z, y, x in zip(g.ids, g.t, g.z, g.y, g.x):
    by_t.setdefault(int(t), []).append([round(float(x),1), round(float(y),1), round(float(z),1), int(tid[int(nid)])])

gt_by_t = {}
order = np.argsort(gt.t)
for t, z, y, x in zip(gt.t[order], gt.z[order], gt.y[order], gt.x[order]):
    gt_by_t[int(t)] = [round(float(x),1), round(float(y),1), round(float(z),1)]

out = {
  "name": NAME, "T": T, "Z": Z, "Y": Y, "X": X,
  "scale": [1.625, 0.40625, 0.40625],
  "est_cells": gt.meta.get("estimated_number_of_nodes"),
  "projXY": projs_xy, "projZY": projs_zy,
  "nodes": {str(k): v for k, v in by_t.items()},
  "gt": {str(k): v for k, v in gt_by_t.items()},
  "n_tracks": len(comp),
}
p = "/tmp/claude-1000/-home-boltuzamaki-Work-get-a-job-kaggle-competitions-biohub/f5efc746-0c0c-4c7f-a177-0dc842daed1b/scratchpad/viz_data.json"
json.dump(out, open(p, "w"), separators=(",", ":"))
import os; print("wrote %.2f MB" % (os.path.getsize(p)/1e6))
