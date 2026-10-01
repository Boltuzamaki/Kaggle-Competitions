"""Generate and push the route-fit shards as Kaggle kernels.

Each shard forces every route tape in turn and plays the whole opponent pool
from both seats on its own slice of seeds, so the shards never duplicate work
and the results concatenate.  Output is one `route_fit.csv` per kernel.

  python sweep/launch_route_fit.py --dry-run
  python sweep/launch_route_fit.py --shards 5 --gpu-shards 2 --seeds 40
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
USER = "boltuzamaki"
DATASET = f"{USER}/kaggriculture-arena"
ENGINE = "1.32.7"

CODE = '''import os, sys, glob, shutil, subprocess, zipfile
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "kaggle-environments=={engine}"], check=True)
import importlib.metadata as _m
print("engine", _m.version("kaggle-environments"), "| cpus", os.cpu_count(), flush=True)

SRC = "/kaggle/input/kaggriculture-arena"
WORK = "/kaggle/working"
AG = os.path.join(WORK, "agents")
os.makedirs(AG, exist_ok=True)
zips = glob.glob(os.path.join(SRC, "*.zip"))
if zips:
    for z in zips:
        zipfile.ZipFile(z).extractall(AG if "agents" in os.path.basename(z) else WORK)
for p in glob.glob(os.path.join(SRC, "*.py")):
    shutil.copy(p, WORK)
# agents.zip may unpack either flat or into an agents/ subdir
nested = os.path.join(AG, "agents")
if os.path.isdir(nested):
    for p in glob.glob(os.path.join(nested, "*.py")):
        shutil.move(p, AG)
print("agents:", sorted(os.path.basename(p) for p in glob.glob(AG + "/*.py")), flush=True)

sys.argv = ["route_fit.py",
            "--base", os.path.join(AG, "aurax7.py"),
            "--opponents"] + [os.path.join(AG, f) for f in {opponents}] + [
            "--routes"] + [str(r) for r in {routes}] + [
            "--seeds", "{seeds}", "--seed-offset", "{offset}",
            "--procs", str(max(1, os.cpu_count() or 2)),
            "--work", os.path.join(WORK, "_routes"),
            "--out", os.path.join(WORK, "route_fit.csv")]
exec(open(os.path.join(WORK, "route_fit.py")).read(), {{"__name__": "__main__"}})
'''

OPPONENTS = ["ahmedberatozer.py", "flexonafft.py", "thomastschinkel.py",
             "tetsutani.py", "boatlee.py"]


def notebook(code):
    return {"cells": [{"cell_type": "code", "execution_count": None,
                       "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=5, help="CPU kernels")
    ap.add_argument("--gpu-shards", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=40, help="seeds per shard")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total = args.shards + args.gpu_shards
    pushed = []
    for i in range(total):
        gpu = i >= args.shards
        slug = f"kaggriculture-route-fit-{i:02d}"
        d = os.path.join(HERE, "kernels", slug)
        os.makedirs(d, exist_ok=True)
        code = CODE.format(engine=ENGINE, opponents=OPPONENTS, routes=list(range(9)),
                           seeds=args.seeds, offset=i * args.seeds)
        json.dump(notebook(code), open(os.path.join(d, f"{slug}.ipynb"), "w"))
        json.dump({"id": f"{USER}/{slug}",
                   "title": f"Kaggriculture Route Fit {i:02d}",
                   "code_file": f"{slug}.ipynb", "language": "python",
                   "kernel_type": "notebook", "is_private": "true",
                   "enable_gpu": "true" if gpu else "false",
                   "enable_internet": "true",
                   "dataset_sources": [DATASET],
                   "competition_sources": [], "kernel_sources": []},
                  open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
        print(f"  built {slug}  seeds {i*args.seeds}..{(i+1)*args.seeds-1}"
              f"  {'GPU' if gpu else 'CPU'}")
        if not args.dry_run:
            r = subprocess.run(["kaggle", "kernels", "push", "-p", d],
                               capture_output=True, text=True)
            line = (r.stdout + r.stderr).strip().splitlines()[-1][:140]
            print(f"    push: {line}")
            pushed.append(slug)
    if pushed:
        print("\nstatus:  " + "  ".join(f"kaggle kernels status {USER}/{s}" for s in pushed[:1]))


if __name__ == "__main__":
    main()
