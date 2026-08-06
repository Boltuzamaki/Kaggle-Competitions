"""Fork a public ROGII kernel into our account with the minimum fixes needed to run.

Fixes applied (no modelling change):
  * competition/artifact roots resolved for both Kaggle mount layouts, because a
    freshly created kernel gets `/kaggle/input/<slug>` while these notebooks
    hardcode `/kaggle/input/{competitions,datasets}/...`
  * optionally neutralise LightGBM `device='gpu'` so the kernel can run on CPU
    when GPU batch slots are contended (the tree models are loaded from mounted
    artifacts, so this touches only the dead training fallback)

Usage:
  python fork_and_patch.py <src_dir> <new_slug> <title> [--cpu]
"""
import json
import sys
from pathlib import Path

PATH_PAIRS = [
    ("COMPETITION_DATA_ROOT = '/kaggle/input/competitions/rogii-wellbore-geology-prediction'",
     """COMPETITION_DATA_ROOT = _fk_pick([
    '/kaggle/input/competitions/rogii-wellbore-geology-prediction',
    '/kaggle/input/rogii-wellbore-geology-prediction',
], must='train')"""),
    ("RIDGE_ARTIFACT_ROOT = '/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts'",
     """RIDGE_ARTIFACT_ROOT = _fk_pick([
    '/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts',
    '/kaggle/input/wellbore-geology-prediction-artifacts',
], must='models')"""),
]

HELPER = '''import os as _fk_os

def _fk_pick(cands, must=None):
    for c in cands:
        if c and _fk_os.path.isdir(c):
            if must is None or _fk_os.path.exists(_fk_os.path.join(c, must)):
                return c
    return cands[0]

'''


def main():
    src_dir = Path(sys.argv[1])
    slug, title = sys.argv[2], sys.argv[3]
    cpu = "--cpu" in sys.argv

    nb_path = next(src_dir.glob("*.ipynb"))
    nb = json.loads(nb_path.read_text())
    meta = json.loads((src_dir / "kernel-metadata.json").read_text())

    patched = {"paths": 0, "gpu": 0, "helper": 0}
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s = "".join(cell["source"])
        for old, new in PATH_PAIRS:
            if old in s:
                if not patched["helper"]:
                    s = HELPER + s
                    patched["helper"] = 1
                s = s.replace(old, new, 1)
                patched["paths"] += 1
        if cpu and "device='gpu'" in s:
            patched["gpu"] += s.count("device='gpu'")
            s = s.replace("device='gpu'", "device='cpu'")
        cell["source"] = s.splitlines(keepends=True)

    out_dir = Path(__file__).resolve().parent / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    code_file = f"{slug}.ipynb"
    (out_dir / code_file).write_text(json.dumps(nb, indent=1))

    meta["id"] = f"boltuzamaki/{slug}"
    meta["title"] = title
    meta["code_file"] = code_file
    meta["is_private"] = True
    meta.pop("id_no", None)
    meta["dataset_sources"] = [d for d in (meta.get("dataset_sources") or []) if d]
    if cpu:
        meta["enable_gpu"] = False
        meta["machine_shape"] = "Cpu"
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"{slug}: patched={patched} gpu={meta.get('enable_gpu')} -> {out_dir}")


if __name__ == "__main__":
    main()
