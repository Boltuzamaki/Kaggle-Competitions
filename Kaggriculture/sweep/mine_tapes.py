"""Harvest opponent action tapes from our own ladder episodes.

Every episode we play is a two-sided recording: the replay carries the full
action stream for *both* seats, so each game against a stronger agent hands us
their tape for free.  Our own submission is the sensor -- the stronger our
rating, the stronger the opponents we get matched against, and the better the
tapes we can mine.

Raw replays are ~22 MB each, so nothing is kept on disk: each one is parsed
down to a compact zlib record (seed, team names, final banks, the day-6 shop
pair the router keys on, and both tapes) and then deleted.

  python sweep/mine_tapes.py --submission 56328723 --limit 40
  python sweep/mine_tapes.py --report
"""
import argparse, base64, glob, json, os, subprocess, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "research", "tapes")
TMP = os.path.join(ROOT, "research", "replays")
DAY6_STEP = 144


def episodes(submission):
    r = subprocess.run(["kaggle", "competitions", "episodes", str(submission), "-v"],
                       capture_output=True, text=True)
    ids = []
    for line in r.stdout.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 4 and "COMPLETED" in line:
            ids.append(parts[0].strip())
    return ids


def compact(replay):
    st = replay["steps"]
    fin = st[-1][0]["observation"]["farms"]
    shops = st[DAY6_STEP][0]["observation"]["town"]["unlocked_shops"][:2] \
        if len(st) > DAY6_STEP else []
    tapes = [[s[p].get("action") for s in st] for p in (0, 1)]
    return {
        "episode": replay["info"]["EpisodeId"],
        "seed": replay["info"].get("seed"),
        "teams": replay["info"].get("TeamNames"),
        "banks": [f["money"] for f in fin],
        "statuses": replay.get("statuses", [])[-1] if replay.get("statuses") else None,
        "shop_pair": shops,
        "tapes_b64": [base64.b64encode(zlib.compress(
            json.dumps(t, separators=(",", ":")).encode())).decode() for t in tapes],
    }


def load(path):
    rec = json.load(open(path))
    rec["tapes"] = [json.loads(zlib.decompress(base64.b64decode(t)))
                    for t in rec.pop("tapes_b64")]
    return rec


def harvest(submission, limit):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    ids = episodes(submission)
    print(f"{len(ids)} completed episodes for submission {submission}")
    done = {os.path.basename(p).split(".")[0] for p in glob.glob(os.path.join(OUT, "*.json"))}
    kept = 0
    for eid in ids:
        if kept >= limit:
            break
        if eid in done:
            continue
        r = subprocess.run(["kaggle", "competitions", "replay", eid, "-p", TMP],
                           capture_output=True, text=True)
        path = os.path.join(TMP, f"episode-{eid}-replay.json")
        if not os.path.exists(path):
            print(f"  {eid}: no replay ({(r.stdout + r.stderr).strip()[:80]})")
            continue
        try:
            rec = compact(json.load(open(path)))
            json.dump(rec, open(os.path.join(OUT, f"{eid}.json"), "w"))
            kept += 1
            print(f"  {eid}  {rec['teams']}  banks {rec['banks']}  "
                  f"shops {rec['shop_pair']}")
        except Exception as exc:
            print(f"  {eid}: parse failed {exc!r}")
        finally:
            os.remove(path)
    print(f"kept {kept} -> {OUT}")


def report():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        rec = json.load(open(p))
        us = 0 if (rec["teams"] or ["", ""])[0] == "Boltuzamaki" else 1
        them = 1 - us
        rows.append((rec["banks"][them], rec["teams"][them], rec["banks"][us],
                     "|".join(rec["shop_pair"]), rec["episode"]))
    rows.sort(reverse=True)
    print(f"{'opp bank':>10} {'our bank':>10}  {'team':<28} {'day-6 shops':<30} episode")
    for b, t, ub, sp, e in rows:
        print(f"{b:>10,.0f} {ub:>10,.0f}  {t:<28} {sp:<30} {e}")
    if rows:
        print(f"\n{len(rows)} episodes | we won "
              f"{sum(1 for b, _, ub, _, _ in rows if ub > b)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report or not a.submission:
        report()
    else:
        harvest(a.submission, a.limit)
