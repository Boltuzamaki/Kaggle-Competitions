"""Offline replica of one Kaggle session, run against tasks with known labels.

For each task we copy only the three files the container exposes, run the package's
portfolio script exactly as the skill would, then score every candidate it emitted.
Selection is simulated the way the harness behaves when the agent leaves the slots
empty: the top two public scores are auto-selected and the better private one wins.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, engine="pyarrow")


def score_candidate(pred_path: Path, solution: pd.DataFrame, id_col: str, target_col: str):
    """Return (public_auc, private_auc, full_auc) for one candidate CSV."""
    pred = _read(pred_path)
    pred_col = [c for c in pred.columns if c != id_col]
    if not pred_col:
        return None
    merged = solution.merge(
        pred[[id_col, pred_col[-1]]].rename(columns={pred_col[-1]: "_p"}), on=id_col, how="left"
    )
    if merged["_p"].isna().any():
        return None
    out = {}
    for label, mask in (
        ("public", merged["Usage"].str.lower() == "public"),
        ("private", merged["Usage"].str.lower() == "private"),
        ("full", pd.Series(True, index=merged.index)),
    ):
        sub = merged[mask]
        if sub[target_col].nunique() < 2:
            out[label] = float("nan")
        else:
            out[label] = float(roc_auc_score(sub[target_col], sub["_p"]))
    return out


def run_one_task(package: str, task_dir: str, timeout: int, threads: int):
    """Run the package's scripts on one task inside an isolated workdir."""
    package_p, task_p = Path(package).resolve(), Path(task_dir).resolve()
    name = task_p.name
    started = time.time()
    work = Path(tempfile.mkdtemp(prefix=f"bench_{name}_"))
    result = {"task": name, "package": package_p.name}
    try:
        for fn in ("train.csv", "test.csv", "sample_submission.csv"):
            if (task_p / fn).is_file():
                shutil.copy(task_p / fn, work / fn)

        scripts = package_p / "skills" / "robust-tabular" / "scripts"
        env = dict(os.environ)
        env.update(
            ROBUST_TABULAR_WORKDIR=str(work),
            OMP_NUM_THREADS=str(threads),
            MKL_NUM_THREADS=str(threads),
            PYTHONWARNINGS="ignore",
        )

        candidates: list[str] = []
        # Stage 1: the quick baseline the first agent always submits.
        quick = subprocess.run(
            [sys.executable, str(scripts / "quick_baseline.py")],
            cwd=work, env=env, capture_output=True, text=True, timeout=timeout,
        )
        if (work / "quick_baseline.csv").is_file():
            candidates.append(str(work / "quick_baseline.csv"))
        result["quick_rc"] = quick.returncode

        # Stage 2: the portfolio, whose manifest lists what the agent may submit.
        proc = subprocess.run(
            [sys.executable, str(scripts / "run_portfolio.py")],
            cwd=work, env=env, capture_output=True, text=True, timeout=timeout,
        )
        result["portfolio_rc"] = proc.returncode
        manifest = None
        for line in proc.stdout.splitlines():
            if line.startswith("PORTFOLIO_MANIFEST="):
                manifest = json.loads(line.split("=", 1)[1])
        if manifest is None:
            result["error"] = f"no manifest (rc={proc.returncode}) {proc.stderr[-400:]}"
        else:
            candidates.extend(manifest.get("candidates", [])[:8])
            result["cv"] = {
                c["kind"]: round(c["cv_auc"], 6)
                for c in manifest.get("candidate_metrics", [])
            }
            result["errors"] = manifest.get("errors", {})

        solution = _read(task_p / "solution.csv")
        id_col = solution.columns[0]
        target_col = [c for c in solution.columns if c.lower() not in (id_col.lower(), "usage")][0]

        scored = []
        for path in candidates:
            p = Path(path)
            if not p.is_file():
                continue
            s = score_candidate(p, solution, id_col, target_col)
            if s:
                scored.append({"name": p.stem, **s})
        result["candidates"] = scored

        if scored:
            # Harness default with no select_submission call: top two public are kept.
            ranked = sorted(scored, key=lambda r: r["public"], reverse=True)
            picked = ranked[:2]
            result["selected"] = [p["name"] for p in picked]
            result["selected_private"] = max(p["private"] for p in picked)
            result["selected_full"] = max(p["full"] for p in picked)
            result["oracle_private"] = max(p["private"] for p in scored)
            result["best_public"] = ranked[0]["public"]
    except subprocess.TimeoutExpired:
        result["error"] = f"timeout after {timeout}s"
    except Exception as exc:  # keep the sweep alive; a dead task is a data point
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
    result["seconds"] = round(time.time() - started, 1)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--tasks", default=str(ROOT / "competition_data" / "data"))
    ap.add_argument("--glob", default="train_*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    tasks = sorted(Path(args.tasks).glob(args.glob))
    if args.only:
        keep = {t.strip() for t in args.only.split(",")}
        tasks = [t for t in tasks if t.name in keep]
    print(f"package={args.package}  tasks={len(tasks)}  workers={args.workers}", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one_task, args.package, str(t), args.timeout, args.threads): t
            for t in tasks
        }
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            if "error" in r:
                print(f"  {r['task']:10s} ERROR {r['error'][:90]}  ({r['seconds']}s)", flush=True)
            else:
                print(
                    f"  {r['task']:10s} sel={r.get('selected_private', float('nan')):.5f} "
                    f"oracle={r.get('oracle_private', float('nan')):.5f} "
                    f"picks={r.get('selected')}  ({r['seconds']}s)",
                    flush=True,
                )

    results.sort(key=lambda r: r["task"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    ok = [r for r in results if "selected_private" in r]
    print("\n" + "=" * 68)
    print(f"tasks scored : {len(ok)}/{len(results)}")
    if ok:
        print(f"MEAN selected private AUC : {np.mean([r['selected_private'] for r in ok]):.6f}")
        print(f"MEAN oracle   private AUC : {np.mean([r['oracle_private'] for r in ok]):.6f}")
        print(f"total wall seconds        : {sum(r['seconds'] for r in results):.0f}")
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
