"""Push a sharded sweep across the Kaggle notebook quota.

Every kernel runs the same harness script from the `kaggriculture-arena`
dataset with its own shard index, so the shards never overlap and their CSVs
concatenate.  GPU is requested only to spend the GPU-notebook quota -- this
workload is pure episode simulation and never touches a GPU -- so those kernels
are just extra CPU.

  python sweep/launch.py --job pair --shards 5 --gpu-shards 2 --dry-run
"""
import argparse, json, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
USER = "boltuzamaki"
DATASET = f"{USER}/kaggriculture-arena"
ENGINE = "1.32.7"

PREAMBLE = '''import os, sys, glob, shutil, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "kaggle-environments=={engine}"], check=True)
import importlib.metadata as _m
WORK = "/kaggle/working"
print("engine", _m.version("kaggle-environments"), "| cpus", os.cpu_count(), flush=True)
# Find the attached dataset by content rather than by mount path: the mount name
# has not been reliable, and a wrong guess fails 60s into a multi-hour kernel.
roots = sorted(glob.glob("/kaggle/input/*"))
print("input roots:", roots, flush=True)
SRC = next((r for r in roots if os.path.exists(os.path.join(r, "aurax7.py"))), None)
if SRC is None:
    hits = glob.glob("/kaggle/input/**/aurax7.py", recursive=True)
    SRC = os.path.dirname(hits[0]) if hits else None
if SRC is None:
    raise SystemExit("kaggriculture-arena not attached: " + repr(roots))
print("using", SRC, "->", sorted(os.listdir(SRC))[:40], flush=True)
for p in glob.glob(os.path.join(SRC, "*.py")) + glob.glob(os.path.join(SRC, "*.json")):
    shutil.copy(p, WORK)
os.chdir(WORK)
sys.path.insert(0, WORK)
'''

JOBS = {
    "pair": {
        "title": "Kaggriculture Pair Sweep",
        "slug": "kaggriculture-pair-sweep",
        "script": "pair_sweep.py",
        "args": '["--base", os.path.join(WORK, "aurax7.py"),'
                ' "--index", os.path.join(WORK, "seed_pairs.json"),'
                ' "--shard", "{shard}", "--shards", "{shards}",'
                ' "--seeds-per-pair", "{seeds}",'
                ' "--procs", str(max(1, os.cpu_count() or 2)),'
                ' "--work", os.path.join(WORK, "_pairs"),'
                ' "--out", os.path.join(WORK, "pair_sweep.csv")]',
    },
    "mirror": {
        "title": "Kaggriculture Mirror Sweep",
        "slug": "kaggriculture-mirror-sweep",
        "script": "mirror_eval.py",
        "args": '["--base", os.path.join(WORK, "aurax7.py"),'
                ' "--specs", os.path.join(WORK, "specs.json"),'
                ' "--seeds", "{seeds}", "--seed-offset", "{offset}",'
                ' "--procs", str(max(1, os.cpu_count() or 2)),'
                ' "--work", os.path.join(WORK, "_variants"),'
                ' "--out", os.path.join(WORK, "mirror_eval.csv")]',
    },
}


def notebook(code):
    return {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                       "outputs": [], "source": code.splitlines(True)}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", choices=sorted(JOBS), default="pair")
    ap.add_argument("--shards", type=int, default=5, help="CPU kernels")
    ap.add_argument("--gpu-shards", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="push only these shard indices; the rest keep running. "
                         "Re-pushing a shard RESTARTS it and loses its progress.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    job = JOBS[args.job]
    total = args.shards + args.gpu_shards
    for i in range(total):
        if args.only is not None and i not in args.only:
            continue
        gpu = i >= args.shards
        slug = f"{job['slug']}-{i:02d}"
        d = os.path.join(HERE, "kernels", slug)
        os.makedirs(d, exist_ok=True)
        argv = job["args"].format(shard=i, shards=total, seeds=args.seeds,
                                  offset=i * args.seeds)
        code = (PREAMBLE.format(engine=ENGINE)
                + f'sys.argv = ["{job["script"]}"] + {argv}\n'
                + f'_script = os.path.join(WORK, "{job["script"]}")\n'
                # Run as a real subprocess, not exec(): the harness fans out over a
                # ProcessPoolExecutor, and a top-level function defined inside an
                # exec()'d globals dict has no importable __main__ to be pickled by.
                + 'rc = subprocess.run([sys.executable, _script] + sys.argv[1:]).returncode\n'
                + 'print("exit", rc, flush=True)\n'
                + 'assert rc == 0, "harness exited %d" % rc\n')
        json.dump(notebook(code), open(os.path.join(d, f"{slug}.ipynb"), "w"))
        json.dump({"id": f"{USER}/{slug}",
                   "title": f"{job['title']} {i:02d}",
                   "code_file": f"{slug}.ipynb", "language": "python",
                   "kernel_type": "notebook", "is_private": "true",
                   "enable_gpu": "true" if gpu else "false",
                   "enable_internet": "true",
                   "dataset_sources": [DATASET],
                   "competition_sources": [], "kernel_sources": []},
                  open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
        print(f"  {slug}  shard {i}/{total}  {'GPU' if gpu else 'CPU'}")
        if not args.dry_run:
            r = subprocess.run(["kaggle", "kernels", "push", "-p", d],
                               capture_output=True, text=True)
            print("    " + (r.stdout + r.stderr).strip().splitlines()[-1][:120])


if __name__ == "__main__":
    main()
