"""Build five private Kaggle CPU lanes confirming Crustle v29 vs Grim v2."""
from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kaggle_remote" / "kernels"
DATASET = "boltuzamaki/ptcg-current-candidate-assets-v1"
OPPONENTS = (
    "public-alakazam",
    "public-archaludon",
    "public-garchomp-v28",
    "router-v12",
    "meta-crustle",
)

PATCH_SETUP = r'''\
import glob, shutil
patches = glob.glob("/kaggle/input/**/agent_patch.tar.gz", recursive=True)
expanded = glob.glob("/kaggle/input/**/agent_patch/agent", recursive=True)
if patches:
    shutil.unpack_archive(patches[0], work_project)
elif expanded:
    shutil.copytree(expanded[0], work_project / "agent", dirs_exist_ok=True)
else:
    raise RuntimeError("neither packed nor expanded agent patch is attached")
# Kaggle expands nested gzip members while ingesting dataset archives. Restore
# the exact filenames expected by the frozen Grim package when that occurs.
import gzip
for stem in ("feature_schema.pkl", "policy_ensemble.bin"):
    raw = work_project / "agent/grim_candy_v2/models" / stem
    packed = raw.with_name(raw.name + ".gz")
    if raw.is_file() and not packed.is_file():
        with raw.open("rb") as src, gzip.open(packed, "wb") as dst:
            shutil.copyfileobj(src, dst)
required = [
    work_project / "agent/grim_candy_v2/main.py",
    work_project / "agent/grim_candy_v2/models/feature_schema.pkl.gz",
    work_project / "agent/grim_candy_v2/models/policy_ensemble.bin.gz",
    work_project / "agent/router_v12/main.py",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise RuntimeError("agent patch incomplete: " + repr(missing))
print("agent patch verified", len(required))
'''


def main() -> None:
    for index, opponent in enumerate(OPPONENTS):
        short = opponent.replace("public-", "").replace("meta-", "meta-")
        slug = f"ptcg-crustle-v29-confirm-{short}"
        target = KERNELS / f"crustle_v29_confirm_{index}"
        target.mkdir(parents=True, exist_ok=True)
        metadata = {
            "id": f"boltuzamaki/{slug}",
            "title": f"PTCG Crustle v29 confirm {short}",
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": False,
            "keywords": ["research"],
            "dataset_sources": [DATASET, "kiyotah/cg-lib"],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        }
        run = f'''\
from pathlib import Path
import subprocess, sys

results = []
for candidate, label in (("current-crustle-counter-v29", "candidate"),
                         ("grim-candy-v2", "baseline")):
    prefix = f"/kaggle/working/{{label}}"
    command = [sys.executable, "tools/arena.py", "--games", "300",
               "--competitors", candidate + ",{opponent}", "--out", prefix]
    completed = subprocess.run(command, text=True, capture_output=True)
    Path(prefix + ".log").write_text(completed.stdout + "\\nSTDERR\\n" + completed.stderr,
                                     encoding="utf-8")
    print(completed.stdout[-4000:])
    if completed.returncode:
        raise RuntimeError(label + " arena failed: " + completed.stderr[-2000:])
print("submission_performed=False")
'''
        target.joinpath("kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        target.joinpath("notebook.ipynb").write_text(
            json.dumps(notebook([("markdown", "# Held-out Crustle v29 paired confirmation"),
                                 ("code", SETUP), ("code", PATCH_SETUP), ("code", run)]), indent=1),
            encoding="utf-8",
        )
        print(target)


if __name__ == "__main__":
    main()
