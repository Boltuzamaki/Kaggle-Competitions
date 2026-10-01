"""Turn pair-sweep shards into a proposed router table.

Each (shop pair, route) cell is a small paired sample, so the argmax alone
overfits: with 9 routes and a handful of games per cell the best-looking route
is often just the luckiest.  An entry is only proposed when the winner clears
the incumbent by a margin on BOTH signals the cell carries -- match score, which
is what the ladder pays, and mean bank margin, which is the finer-grained one --
and when the cell has enough games to mean anything.

  python sweep/analyze_pairs.py research/kaggle_out/pair_*/pair_sweep.csv
"""
import argparse, collections, csv, glob, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import variants  # noqa: E402


def load(paths):
    rows = []
    for p in paths:
        for f in glob.glob(p):
            rows.extend(csv.DictReader(open(f)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--base", default=os.path.join(ROOT, "gauntlet2", "aurax7.py"))
    ap.add_argument("--min-games", type=int, default=8)
    ap.add_argument("--min-score", type=float, default=0.60,
                    help="cell must beat the base by this match score")
    ap.add_argument("--min-margin", type=float, default=2000.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "sweep", "specs", "router.json"))
    args = ap.parse_args()

    rows = load(args.csvs)
    if not rows:
        print("no rows"); return 1
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["pair"], int(r["route"]))].append(
            (float(r["score"]), float(r["mine"]) - float(r["theirs"]), r["err"]))

    incumbent, default, _ = variants.parse_router(open(args.base).read())
    proposal = {}
    print(f"{'shop pair':<34}{'route':>6}{'score':>7}{'margin':>10}{'n':>5}   incumbent")
    for pair in sorted({p for p, _ in cells}):
        best = None
        for (p, route), vals in cells.items():
            if p != pair or len(vals) < args.min_games:
                continue
            if any(v[2] for v in vals):          # a crash is a ladder loss, drop the cell
                continue
            sc = statistics.mean(v[0] for v in vals)
            mg = statistics.mean(v[1] for v in vals)
            if best is None or (sc, mg) > (best[1], best[2]):
                best = (route, sc, mg, len(vals))
        if best is None:
            continue
        route, sc, mg, n = best
        cur = incumbent.get(tuple(pair.split("|")), default)
        keep = sc >= args.min_score and mg >= args.min_margin and route != cur
        flag = "  <- PROPOSE" if keep else ""
        print(f"{pair:<34}{route:>6}{sc:>7.2f}{mg:>+10,.0f}{n:>5}   {cur}{flag}")
        if keep:
            proposal[pair] = route

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    specs = {"router_refit": {"router_fix": proposal}} if proposal else {}
    json.dump(specs, open(args.out, "w"), indent=1)
    print(f"\n{len(proposal)} entries proposed -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
