"""Generate and push N independent Kaggle sweep kernels, one per experiment.

Each kernel is self-contained: it writes the current main.py, builds mid-tier
opponents from the agent's own config space, runs its grid across both seats and
paired seeds, and ranks by MATCH SCORE (what the ladder pays) with mean bank as
a diagnostic.

GPU is deliberately off: there is no model here, only episode simulation, which
is CPU-bound.

  python launch_sweeps.py            # generate + push all
  python launch_sweeps.py --dry-run  # just write the files
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
USER = json.load(open(os.path.expanduser("~/.kaggle/credentials.json")))["username"]

# Each entry: slug-suffix -> (Title, grid, seeds)
EXPERIMENTS = {
    "opening": ("Kaggriculture Sweep Opening", {
        "melon_tiles":        [6, 8, 10, 12],
        "seed_cash_floor":    [700, 1100, 1600],
        "hands_d0":           [4, 6, 8],
        "filler_start_day":   [6, 9, 12],
    }, 4),
    "herd": ("Kaggriculture Sweep Herd", {
        "target_cows":        [6, 8, 10],
        "target_sheep":       [3, 4, 6],
        "target_geese":       [0, 3, 6],
        "wheat_buffer_days":  [2, 4, 6],
    }, 4),
    "market": ("Kaggriculture Sweep Market", {
        "shed_pressure":      [30, 45, 60],
        "wheat_cap":          [16, 26, 40],
        "flush_hour":         [16, 20, 23],
        "liquidate_day":      [27, 28, 29],
    }, 4),
    # Ranges set from mined ladder-winner milestones: winners hold 36-38
    # strawberry and ramp wheat to ~44 late, while keeping only 2-7 tiles free.
    # The original ranges (straw<=22, wheat<=10) did not even contain the answer.
    "crops": ("Kaggriculture Sweep Crops", {
        "straw_tiles":        [14, 24, 36],
        "wheat_tiles":        [4, 20, 40],
        "carrot_tiles":       [8, 16],
        "target_cows":        [7, 10],
    }, 4),
    "land": ("Kaggriculture Sweep Land", {
        "land_empty_gate":    [10, 16, 22, 30],
        "land_last_day":      [[24,22,18], [26,24,22], [20,18,14]],
        "demand_ratio_max":   [1.0, 1.5, 2.0],
    }, 5),
}

NB_HEAD = """import sys, subprocess, os, json, itertools, statistics, time, csv
from concurrent.futures import ProcessPoolExecutor
try:
    import kaggle_environments
except ImportError:
    subprocess.run([sys.executable,"-m","pip","install","-q",
                    "kaggle-environments==1.32.6"], check=True)
import importlib.metadata as m
print("engine", m.version("kaggle-environments"), "| cpus", os.cpu_count())"""

NB_OPP = '''import os
os.makedirs("opponents", exist_ok=True)
# The ladder lives at 65-90k bank. A pool of only weak + elite agents is bimodal
# and cannot resolve anything in between, so spar against our own config space.
VARIANTS = {
  "mid_v6":      {"melon_tiles":8,"straw_tiles":8,"land_empty_gate":8,"target_sheep":6},
  "mid_melon14": {"melon_tiles":14,"straw_tiles":8,"land_empty_gate":8},
  "mid_geese":   {"target_geese":6,"straw_tiles":8},
}
base = open("main.py").read()
for name, ov in VARIANTS.items():
    hook = "\\n# fixed sparring config\\nP.update(%r)\\n" % ov
    open("opponents/%s.py" % name, "w").write(
        base.replace("def _dist(a, b):", hook + "\\n\\ndef _dist(a, b):", 1))
OPPONENTS = ["opponents/%s.py" % n for n in VARIANTS]
print(OPPONENTS)'''

NB_RUN = '''def _run(task):
    name, override, seed, seat, opp = task
    os.environ["KAGGRICULTURE_P"] = json.dumps(override)
    from kaggle_environments import make
    agents = ["main.py", opp] if seat == 0 else [opp, "main.py"]
    try:
        env = make("kaggriculture",
                   configuration={"episodeSteps": 720, "seed": seed}, debug=False)
        env.run(agents)
        last = env.steps[-1]
        me, them = last[seat].reward or 0.0, last[1-seat].reward or 0.0
        ok = str(last[seat].status)=="DONE" and str(last[1-seat].status)=="DONE"
        return name, me, them, ok
    except Exception:
        return name, 0.0, 0.0, False

keys = list(GRID)
tasks = []
for combo in itertools.product(*(GRID[k] for k in keys)):
    ov = dict(zip(keys, combo))
    name = ",".join("%s=%s" % (k, v) for k, v in ov.items())
    for opp in OPPONENTS:
        for s in range(SEEDS):
            for seat in (0, 1):
                tasks.append((name, ov, s, seat, opp))
print("configs", len(list(itertools.product(*GRID.values()))), "| games", len(tasks))
t0 = time.time()
with ProcessPoolExecutor(max_workers=max(1, os.cpu_count())) as ex:
    results = list(ex.map(_run, tasks))
print("done in %.0fs" % (time.time()-t0))

agg = {}
for name, me, them, ok in results:
    d = agg.setdefault(name, {"bank": [], "score": 0.0, "n": 0, "err": 0})
    d["bank"].append(me); d["n"] += 1
    d["score"] += 1.0 if me > them else (0.5 if me == them else 0.0)
    d["err"] += (0 if ok else 1)
rows = sorted(((d["score"]/max(1,d["n"]), statistics.mean(d["bank"]), n, d["n"], d["err"])
               for n, d in agg.items()), reverse=True)
print(f"{'score':>7}{'mean bank':>13}{'games':>7}{'err':>5}  config")
print("-"*100)
for sc, bank, name, n, err in rows[:40]:
    print(f"{sc:>7.3f}{bank:>13,.0f}{n:>7}{err:>5}  {name}")
with open("sweep_results.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["score","mean_bank","games","errors","config"])
    for sc, bank, name, n, err in rows:
        w.writerow(["%.4f" % sc, "%.0f" % bank, n, err, name])
print("wrote sweep_results.csv")'''


def build(slug, title, grid, seeds, agent_src):
    md = lambda s: {"cell_type":"markdown","metadata":{},"source":s}
    code = lambda s: {"cell_type":"code","metadata":{},"execution_count":None,
                      "outputs":[],"source":s}
    cells = [
        md("# %s\n\nParallel parameter sweep. Scored **win/tie/loss** (what the "
           "ladder pays), both seats, paired seeds, against mid-tier opponents.\n\n"
           "CPU-only by design: there is no model here, only episode simulation."
           % title),
        code(NB_HEAD),
        code("%%writefile main.py\n" + agent_src),
        code(NB_OPP),
        md("## Grid"),
        code("GRID = %s\nSEEDS = %d" % (json.dumps(grid, indent=4), seeds)),
        code(NB_RUN),
        md("## Next\n\nTake the top config back to the local repo and run it "
           "through `promote.py`. A sweep winner at %d seeds is a hypothesis; the "
           "gate needs a paired 95%% bootstrap CI above zero against the current "
           "champion." % seeds),
    ]
    return {"cells": cells,
            "metadata": {"kernelspec":{"display_name":"Python 3","language":"python",
                                       "name":"python3"},
                         "language_info":{"name":"python","version":"3.11"}},
            "nbformat":4, "nbformat_minor":5}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    agent_src = open(os.path.join(HERE, "main.py")).read()
    for suffix, (title, grid, seeds) in EXPERIMENTS.items():
        if a.only and a.only != suffix:
            continue
        d = os.path.join(HERE, "kaggle_sweep", suffix)
        os.makedirs(d, exist_ok=True)
        fn = "sweep-%s.ipynb" % suffix
        json.dump(build(suffix, title, grid, seeds, agent_src),
                  open(os.path.join(d, fn), "w"), indent=1)
        slug = "kaggriculture-sweep-%s" % suffix
        json.dump({"id": "%s/%s" % (USER, slug), "title": title,
                   "code_file": fn, "language": "python", "kernel_type": "notebook",
                   "is_private": "true", "enable_gpu": "false",
                   "enable_internet": "true", "dataset_sources": [],
                   "competition_sources": ["kaggriculture"], "kernel_sources": []},
                  open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
        n = len(list(__import__("itertools").product(*grid.values())))
        print("built %-8s %2d configs -> %s" % (suffix, n, slug))
        if not a.dry_run:
            r = subprocess.run(["kaggle","kernels","push","-p",d],
                               capture_output=True, text=True)
            out = (r.stdout + r.stderr).strip().splitlines()
            print("   ", out[-1][:110] if out else "no output")
