#!/usr/bin/env python3
"""Build a Kaggle notebook that times the real portfolio script in the real container.

Local timings cannot answer the only runtime question that matters: does the portfolio
finish inside a 60-minute session on Kaggle's CPU image? This notebook writes the
shipped scripts verbatim, runs them on the largest and most categorical tasks, and
prints wall-clock per task so the time budget can be set from measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DRIVER = r'''
import os, sys, json, time, shutil, subprocess, tempfile
from pathlib import Path

# The hardest tasks for the time budget: the two biggest tables and the two with the
# most categorical columns. If these fit, everything else does.
WANTED = ["train_12", "train_11", "train_14", "train_15", "train_13"]

def find_tasks(root="/kaggle/input"):
    found = {}
    for dirpath, _d, files in os.walk(root):
        if "train.csv" in files and "test.csv" in files:
            found[Path(dirpath).name] = Path(dirpath)
    return found

tasks = find_tasks()
print("discovered task folders:", len(tasks))
print("sample:", sorted(tasks)[:20])

import multiprocessing
print("cpu_count:", multiprocessing.cpu_count())
for mod in ("pandas", "numpy", "sklearn", "catboost", "lightgbm", "xgboost", "pyarrow"):
    try:
        m = __import__(mod)
        print(f"  {mod:10s} {getattr(m, '__version__', '?')}")
    except Exception as exc:
        print(f"  {mod:10s} MISSING ({exc})")

results = []
for name in WANTED:
    if name not in tasks:
        print(f"skip {name}: not present")
        continue
    src = tasks[name]
    work = Path(tempfile.mkdtemp(prefix=f"parity_{name}_"))
    for fn in ("train.csv", "test.csv", "sample_submission.csv"):
        if (src / fn).is_file():
            shutil.copy(src / fn, work / fn)

    env = dict(os.environ)
    env["ROBUST_TABULAR_WORKDIR"] = str(work)
    env["PYTHONWARNINGS"] = "ignore"

    t0 = time.time()
    quick = subprocess.run([sys.executable, "/kaggle/working/scripts/quick_baseline.py"],
                           cwd=work, env=env, capture_output=True, text=True)
    t_quick = time.time() - t0

    t1 = time.time()
    proc = subprocess.run([sys.executable, "/kaggle/working/scripts/run_portfolio.py"],
                          cwd=work, env=env, capture_output=True, text=True)
    t_port = time.time() - t1

    manifest = None
    for line in proc.stdout.splitlines():
        if line.startswith("PORTFOLIO_MANIFEST="):
            manifest = json.loads(line.split("=", 1)[1])

    row = {
        "task": name,
        "quick_s": round(t_quick, 1),
        "portfolio_s": round(t_port, 1),
        "total_s": round(t_quick + t_port, 1),
        "quick_rc": quick.returncode,
        "portfolio_rc": proc.returncode,
        "n_candidates": len(manifest["candidates"]) if manifest else 0,
        "errors": manifest.get("errors") if manifest else proc.stderr[-300:],
    }
    print(json.dumps(row))
    results.append(row)
    shutil.rmtree(work, ignore_errors=True)

Path("/kaggle/working/parity_results.json").write_text(json.dumps(results, indent=2))
print("\n" + "=" * 60)
worst = max((r["total_s"] for r in results), default=0)
print(f"worst-case task total: {worst:.0f}s  ({worst/60:.1f} min of the 60-min session)")
print("VERDICT:", "FITS with room for the freeroll" if worst < 1500 else "TOO SLOW - cut the portfolio")
'''


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--out", default="kaggle_notebook_parity")
    ap.add_argument("--slug", default="agent-portfolio-runtime-parity")
    ap.add_argument("--user", default="boltuzamaki")
    args = ap.parse_args()

    scripts = Path(args.package).resolve() / "skills" / "robust-tabular" / "scripts"
    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = [{
        "cell_type": "markdown", "metadata": {},
        "source": ["# Portfolio runtime parity — real Kaggle CPU image\n\n",
                   "Writes the shipped skill scripts verbatim and times them on the\n",
                   "hardest organizer tasks, so the session time budget is set from\n",
                   "measurement in the deployment environment rather than from local timings.\n"],
    }, code("import os\nos.makedirs('/kaggle/working/scripts', exist_ok=True)\nprint('ok')\n")]

    for fn in ("common.py", "quick_baseline.py", "run_portfolio.py"):
        body = (scripts / fn).read_text()
        cells.append(code(f"%%writefile /kaggle/working/scripts/{fn}\n{body}"))
    cells.append(code(DRIVER))

    nb_name = f"{args.slug}.ipynb"
    (out_dir / nb_name).write_text(json.dumps({
        "cells": cells,
        "metadata": {"kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5,
    }, indent=1))
    (out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.user}/{args.slug}",
        "title": "Agent Portfolio Runtime Parity",
        "code_file": nb_name,
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "keywords": [], "dataset_sources": [], "kernel_sources": [],
        "competition_sources": ["autonomous-agent-prediction-beta"], "model_sources": [],
    }, indent=2))
    print(f"wrote {out_dir / nb_name}")
    print(f"push with: kaggle kernels push -p {out_dir}")


if __name__ == "__main__":
    main()
