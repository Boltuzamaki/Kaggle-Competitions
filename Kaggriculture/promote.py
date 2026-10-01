"""Promotion gate - the only thing allowed to submit.

A candidate replaces the champion and gets submitted ONLY if it clears every
gate below. The gates encode what the ladder actually punishes:

  1. CLEAN      no ERROR / non-DONE game anywhere. An erroring agent is an
                automatic ladder loss, so this is a hard fail, not a low score.
  2. FAST       worst-case per-step act time inside budget.
  3. STRONGER   beats the champion head-to-head, and the paired bootstrap CI
                on those games must exclude a coin flip. Since 1.32.7 the
                champion beats every public agent we can extract 100% of the
                time, so the gauntlet has no resolving power above it and the
                mirror is the only opponent left that can separate candidates.
  4. NO-REGRESS the gauntlet score must not fall more than GAUNTLET_TOLERANCE
                below the champion's. It can
                no longer be the deciding statistic -- both sides sit at 1.000
                -- but a candidate that starts losing to the public field has
                broken something, so it still has to clear the floor.

Submission is NEVER automatic: --submit must be passed explicitly, and the
daily budget (5/day, latest 2 active) is checked first.

Usage:
  python promote.py                      # evaluate main.py vs champion, report
  python promote.py --submit             # ...and submit if all gates pass
  python promote.py --init               # adopt current main.py as champion
  python promote.py --candidate x.py
"""
import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import arena

HERE = os.path.dirname(os.path.abspath(__file__))
CHAMP_DIR = os.path.join(HERE, "champion")
CHAMP = os.path.join(CHAMP_DIR, "main.py")
CHAMP_META = os.path.join(CHAMP_DIR, "meta.json")
HISTORY = os.path.join(HERE, "promotion_log.jsonl")

MAX_STEP_MS = 200.0        # generous vs the framework budget
MIN_H2H = 0.55             # mirror is the deciding signal now, so ask for
                           # a real edge, not a coin flip
GAUNTLET_TOLERANCE = 0.02  # Non-inferiority margin on the gauntlet floor.
                           # 11 of the 12 opponents sit pinned at 1.000, so that
                           # score can only ever move DOWN and cannot supply
                           # positive evidence of non-regression. Demanding a
                           # non-negative CI lower bound there vetoes real
                           # improvements over a single unlucky game (0.004 of
                           # the score). 0.02 is ~5 games in 240 -- still tight
                           # enough to catch an agent that genuinely starts
                           # losing to the public field.


# --------------------------------------------------------------------------
def act_time_ms(agent, seed=11):
    """Worst-case seconds/step for `agent`, measured over a full episode."""
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed},
               debug=False)
    t0 = time.time()
    env.run([agent, "starter"])
    total = time.time() - t0
    ok = all(str(s.status) == "DONE" for s in env.steps[-1])
    return (total / 720.0) * 1000.0, ok


def paired_bootstrap(diffs, iters=4000, seed=0):
    """95% CI on the mean of paired per-game score differences."""
    if not diffs:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(iters):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (statistics.mean(diffs), means[int(0.025 * iters)], means[int(0.975 * iters)])


def gauntlet_scores(agent, opps, seeds, steps, procs, cache):
    recs = arena.run_games([(agent, o) for o in opps], seeds, steps, procs, cache)
    # key each game so candidate and champion can be paired exactly
    by_key, clean = {}, True
    for r in recs:
        opp = r["b"] if r["a"] == agent else r["a"]
        by_key[(opp, r["seed"], r["seat"])] = arena.score_for(r, agent)
        if not arena.is_clean(r):
            clean = False
    return by_key, clean, recs


def submissions_today():
    """How many submissions already used today (limit 5)."""
    try:
        out = subprocess.run(["kaggle", "competitions", "submissions", "kaggriculture"],
                             capture_output=True, text=True, timeout=90).stdout
    except Exception:
        return None
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return sum(1 for line in out.splitlines() if today in line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="main.py")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--submit", action="store_true",
                    help="actually submit to Kaggle if every gate passes")
    ap.add_argument("--init", action="store_true",
                    help="adopt the candidate as champion without gating")
    ap.add_argument("--message", default=None)
    args = ap.parse_args()

    os.makedirs(CHAMP_DIR, exist_ok=True)
    cand = args.candidate

    if args.init or not os.path.exists(CHAMP):
        shutil.copy(cand, CHAMP)
        json.dump({"adopted": datetime.now(timezone.utc).isoformat(),
                   "source": cand}, open(CHAMP_META, "w"), indent=2)
        print(f"champion initialised from {cand}")
        if args.init:
            return

    if arena.agent_key(cand) == arena.agent_key(CHAMP):
        # Already promoted. Allow submitting it without re-running the gauntlet,
        # so "promote, inspect, then submit" is a normal two-step workflow.
        print("candidate is byte-identical to the champion.")
        if not args.submit:
            print("  (pass --submit to send the current champion to Kaggle)")
            return
        used = submissions_today()
        if used is not None and used >= 5:
            print(f"  !! daily submission limit reached ({used}/5) - not submitting.")
            return 1
        meta = json.load(open(CHAMP_META)) if os.path.exists(CHAMP_META) else {}
        msg = args.message or (
            f"champion gauntlet {meta.get('gauntlet', float('nan')):.3f} "
            f"vs {len(opps) if (opps := arena.gauntlet_agents()) else 0} opponents")
        print(f"  submitting champion: {msg}")
        r = subprocess.run(["kaggle", "competitions", "submit", "kaggriculture",
                            "-f", CHAMP, "-m", msg], capture_output=True, text=True)
        ok = "Successfully submitted" in (r.stdout + r.stderr)
        print("  " + (r.stdout + r.stderr).strip().splitlines()[-1][:160])
        with open(HISTORY, "a") as f:
            f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                                "champion_key": arena.agent_key(CHAMP),
                                "gauntlet": meta.get("gauntlet"),
                                "promoted": True, "submitted": ok}) + "\n")
        return 0 if ok else 1

    opps = arena.gauntlet_agents()
    cache = arena.load_cache()

    print(f"\n{'='*78}\n  PROMOTION TEST\n{'='*78}")
    print(f"  candidate : {cand}  [{arena.agent_key(cand)}]")
    print(f"  champion  : {CHAMP}  [{arena.agent_key(CHAMP)}]")
    print(f"  opponents : {len(opps)}   seeds: {args.seeds}   "
          f"({len(opps)*args.seeds*2} games each)\n")

    print("  -- champion --")
    champ_scores, champ_clean, champ_recs = gauntlet_scores(
        CHAMP, opps, args.seeds, args.steps, args.procs, cache)
    print("  -- candidate --")
    cand_scores, cand_clean, cand_recs = gauntlet_scores(
        cand, opps, args.seeds, args.steps, args.procs, cache)
    print("  -- head to head --")
    h2h_recs = arena.run_games([(cand, CHAMP)], args.seeds, args.steps,
                               args.procs, cache)

    arena.report_gauntlet(cand, opps, cand_recs)

    champ_overall = statistics.mean(list(champ_scores.values()))
    cand_overall = statistics.mean(list(cand_scores.values()))
    shared = sorted(set(champ_scores) & set(cand_scores))
    diffs = [cand_scores[k] - champ_scores[k] for k in shared]
    mean_d, lo_d, hi_d = paired_bootstrap(diffs)

    h2h = statistics.mean([arena.score_for(r, cand) for r in h2h_recs])
    h2h_clean = all(arena.is_clean(r) for r in h2h_recs)

    ms, timing_ok = act_time_ms(cand)

    print(f"\n{'='*78}\n  GATES\n{'='*78}")
    gates = []
    gates.append(("CLEAN    no errored games",
                  cand_clean and h2h_clean,
                  "all DONE" if (cand_clean and h2h_clean) else "ERRORS PRESENT"))
    gates.append((f"FAST     act time < {MAX_STEP_MS:.0f} ms/step",
                  ms < MAX_STEP_MS and timing_ok, f"{ms:.1f} ms/step"))
    h2h_scores = [arena.score_for(r, cand) for r in h2h_recs]
    h2h_mean, h2h_lo, h2h_hi = paired_bootstrap([x - 0.5 for x in h2h_scores])
    gates.append((f"STRONGER h2h vs champion >= {MIN_H2H:.2f}",
                  h2h >= MIN_H2H, f"{h2h:.2f} over {len(h2h_recs)} games"))
    gates.append(("SIGNIF   h2h 95% CI excludes a coin flip",
                  h2h_lo > 0.0,
                  f"edge {h2h_mean:+.3f}  CI [{h2h_lo:+.3f}, {h2h_hi:+.3f}]"))
    gates.append((f"NO-REGRESS gauntlet within {GAUNTLET_TOLERANCE:.3f} of champion",
                  cand_overall >= champ_overall - GAUNTLET_TOLERANCE,
                  f"{cand_overall:.3f} vs champion {champ_overall:.3f}"))
    gates.append((f"PAIRED   gauntlet CI above -{GAUNTLET_TOLERANCE:.3f}",
                  lo_d > -GAUNTLET_TOLERANCE,
                  f"delta {mean_d:+.3f}  CI [{lo_d:+.3f}, {hi_d:+.3f}]"))

    for name, ok, detail in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<40} {detail}")

    passed = all(ok for _, ok, _ in gates)
    print(f"\n  VERDICT: {'PROMOTE' if passed else 'REJECT'}")

    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "candidate": cand, "candidate_key": arena.agent_key(cand),
        "champion_key": arena.agent_key(CHAMP),
        "cand_overall": cand_overall, "champ_overall": champ_overall,
        "delta": mean_d, "ci": [lo_d, hi_d], "h2h": h2h,
        "ms_per_step": ms, "clean": cand_clean and h2h_clean,
        "promoted": bool(passed), "submitted": False,
    }

    if not passed:
        with open(HISTORY, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print("  champion unchanged; nothing submitted.")
        return 1

    shutil.copy(cand, CHAMP)
    json.dump({"adopted": entry["time"], "source": cand,
               "gauntlet": cand_overall}, open(CHAMP_META, "w"), indent=2)
    print(f"  champion updated -> {CHAMP}")

    if args.submit:
        used = submissions_today()
        if used is not None and used >= 5:
            print(f"  !! daily submission limit reached ({used}/5) - not submitting.")
        else:
            msg = args.message or (
                f"gauntlet {cand_overall:.2f} (champ {champ_overall:.2f}), "
                f"h2h {h2h:.2f}, {len(shared)} paired games")
            print(f"  submitting: {msg}")
            r = subprocess.run(["kaggle", "competitions", "submit", "kaggriculture",
                                "-f", cand, "-m", msg], capture_output=True, text=True)
            ok = "Successfully submitted" in (r.stdout + r.stderr)
            print("  " + (r.stdout + r.stderr).strip().splitlines()[-1][:160])
            entry["submitted"] = ok
    else:
        print("  (--submit not passed: promoted locally, NOT submitted)")

    with open(HISTORY, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
