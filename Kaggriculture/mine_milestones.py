"""Mine day-by-day STATE milestones from public ladder replays.

Why milestones and not actions: the demonstrators are deterministic tapes, so N
replays of a player carry the information of ~1 trajectory -- behavioural cloning
would memorise one path and then drift (error compounds as O(eps*T^2), T=720).
Milestones transfer instead: our planner keeps doing the executing, we only
borrow the TARGETS. That is the one thing that has measurably worked here
(copying tetsutani's land timing was our single biggest gain).

Reads replays already on disk. Every replay contains both players' full state,
so each episode yields two trajectories, labelled by final bank.

  python mine_milestones.py --glob '/tmp/ep*/*.json' --min-bank 90000
"""
import argparse, collections, glob, json, statistics

FIELDS = ("money", "COW", "SHEEP", "GOOSE", "MELON", "STRAWBERRY", "TOMATO",
          "WHEAT", "CARROT", "PASTURE", "COOP", "free", "quads", "hands")


def trajectory(replay, seat):
    """Per-day snapshot of one player's farm."""
    days = {}
    for st in replay["steps"]:
        obs = st[0]["observation"]
        day, hour = obs.get("day", 0), obs.get("hour", 0)
        if hour != 0 or day in days:
            continue
        farm = obs["farms"][seat]
        c = collections.Counter()
        free = 0
        for row in farm["tiles"]:
            for t in row:
                if t is None:
                    free += 1
                elif isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        c[t["crop"]] += 1
                    elif "animal" in t:
                        c[t["animal"]] += 1
                    elif k in ("COOP", "PASTURE"):
                        c[k] += 1
        days[day] = {"money": farm["money"], "free": free,
                     "quads": len(farm.get("unlocked_quadrants", [])),
                     "hands": len(farm.get("hands", []) or []),
                     **{k: c[k] for k in FIELDS if k in
                        ("COW","SHEEP","GOOSE","MELON","STRAWBERRY","TOMATO",
                         "WHEAT","CARROT","PASTURE","COOP")}}
    return days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/tmp/ep*/*.json")
    ap.add_argument("--min-bank", type=float, default=90000)
    ap.add_argument("--exclude", default="Boltuzamaki",
                    help="skip our own trajectories when setting targets")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    winners, ours = [], []
    for p in sorted(glob.glob(a.glob)):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        names = d.get("info", {}).get("TeamNames", [])
        last = d["steps"][-1]
        for seat in (0, 1):
            bank = last[seat].get("reward") or 0
            traj = trajectory(d, seat)
            name = names[seat] if seat < len(names) else "?"
            if name == a.exclude:
                ours.append((bank, traj))
            elif bank >= a.min_bank:
                winners.append((bank, name, traj))

    if not winners:
        print(f"no trajectories at >= {a.min_bank:,.0f}. "
              f"Lower --min-bank or download more replays.")
        return
    print(f"mined {len(winners)} winning trajectories "
          f"(>= {a.min_bank:,.0f}) from {len(set(w[1] for w in winners))} players; "
          f"{len(ours)} of ours for comparison\n")

    print(f"{'day':>4}" + "".join(f"{k[:6]:>8}" for k in
          ("money","quads","COW","SHEEP","MELON","STRAW","WHEAT","free"))
          + "     | ours (median)")
    rows = {}
    for day in range(30):
        wv = [t[day] for _, _, t in winners if day in t]
        ov = [t[day] for _, t in ours if day in t]
        if not wv:
            continue
        med = lambda vs, k: statistics.median([v.get(k, 0) for v in vs]) if vs else 0
        rows[day] = {k: med(wv, k) for k in
                     ("money","quads","COW","SHEEP","MELON","STRAWBERRY","WHEAT","free")}
        o = {k: med(ov, k) for k in
             ("money","quads","COW","SHEEP","MELON","STRAWBERRY","WHEAT","free")} if ov else None
        line = (f"{day:>4}{rows[day]['money']:>8,.0f}{rows[day]['quads']:>8.0f}"
                f"{rows[day]['COW']:>8.0f}{rows[day]['SHEEP']:>8.0f}"
                f"{rows[day]['MELON']:>8.0f}{rows[day]['STRAWBERRY']:>8.0f}"
                f"{rows[day]['WHEAT']:>8.0f}{rows[day]['free']:>8.0f}")
        if o:
            line += (f"     | ${o['money']:>7,.0f} q{o['quads']:.0f} "
                     f"c{o['COW']:.0f} s{o['SHEEP']:.0f} m{o['MELON']:.0f} "
                     f"st{o['STRAWBERRY']:.0f} f{o['free']:.0f}")
        print(line)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=2)
        print("\nwrote", a.json)


if __name__ == "__main__":
    main()
