import json
from pathlib import Path

DATA_DIR = Path("data")
task_files = sorted(DATA_DIR.glob("task*.json"))

identity_count = 0
color_remap_count = 0
spatial_count = 0

for task_path in task_files[:20]:  # peek at first 20
    with open(task_path) as f:
        d = json.load(f)
    train = d["train"]
    ex0 = train[0]

    # Identity?
    is_identity = all(e["input"] == e["output"] for e in train)

    # Pure color remap? (output color depends only on input color, not position)
    color_map = {}
    is_color_remap = True
    for ex in train:
        inp, out = ex["input"], ex["output"]
        h = min(len(inp), len(out))
        for r in range(h):
            w = min(len(inp[r]) if r < len(inp) else 0,
                    len(out[r]) if r < len(out) else 0)
            for c in range(w):
                ci, co = inp[r][c], out[r][c]
                if ci in color_map and color_map[ci] != co:
                    is_color_remap = False
                    break
                color_map[ci] = co
            if not is_color_remap:
                break
        if not is_color_remap:
            break

    in_colors  = sorted({c for row in ex0["input"]  for c in row})
    out_colors = sorted({c for row in ex0["output"] for c in row})
    n_train = len(train)
    n_agen  = len(d.get("arc-gen", []))

    tag = "IDENTITY" if is_identity else ("COLOR_MAP" if is_color_remap else "SPATIAL")
    print(f"{task_path.stem}: {tag:10s}  train={n_train}  arc-gen={n_agen:4d}  "
          f"in_colors={in_colors}  out_colors={out_colors}")
