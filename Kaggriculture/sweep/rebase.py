"""Keep pace with the public meta: find the strongest new public agent, and if it
beats our champion, rebase onto it and re-apply our front-run delta.

The delta is small and portable. Every top-tier public agent so far ships the
chassis `front_run` layer with `front_run: False` and nothing supplying
`opponent_plan`, so the layer is dead code as published; switching it on and
feeding it a proxy for the opponent's tape is worth a mirror win against every
base we have tried it on. The product list is NOT portable -- on one base the
full nine-product race beat the shipped four, on another it lost -- so the width
is re-measured against each new base rather than carried over.

  python sweep/rebase.py --pull 20          # pull + extract + rank, no submit
  python sweep/rebase.py --pull 20 --apply  # ...and build the rebased candidate
"""
import argparse, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import variants  # noqa: E402

NB_DIR = os.path.join(ROOT, "research", "nb_auto")
OUT_DIR = os.path.join(ROOT, "gauntlet_auto")


def newest_kernels(limit):
    r = subprocess.run(["kaggle", "kernels", "list", "--competition", "kaggriculture",
                        "--sort-by", "dateRun", "-v", "--page-size", str(limit)],
                       capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 2 and "/" in parts[0]:
            try:
                votes = int(parts[-1])
            except ValueError:
                votes = 0
            out.append((parts[0].strip(), votes))
    return out


def pull(refs):
    os.makedirs(NB_DIR, exist_ok=True)
    got = []
    for ref, votes in refs:
        d = os.path.join(NB_DIR, ref.replace("/", "_"))
        if os.path.isdir(d) and any(f.endswith(".ipynb") for f in os.listdir(d)):
            got.append(d)
            continue
        os.makedirs(d, exist_ok=True)
        r = subprocess.run(["kaggle", "kernels", "pull", ref, "-p", d],
                           capture_output=True, text=True)
        if any(f.endswith(".ipynb") for f in os.listdir(d)):
            got.append(d)
            print(f"  pulled {ref} ({votes} votes)")
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", type=int, default=25, help="how many recent kernels to scan")
    ap.add_argument("--apply", action="store_true", help="build the rebased candidate")
    ap.add_argument("--champion", default=os.path.join(ROOT, "champion", "main.py"))
    args = ap.parse_args()

    refs = newest_kernels(args.pull)
    print(f"{len(refs)} recent kernels")
    pull(refs)

    env = dict(os.environ, NB_DIR=NB_DIR, GAUNTLET_OUT=OUT_DIR)
    subprocess.run([sys.executable, os.path.join(ROOT, "extract_gauntlet.py")], env=env)

    cands = [os.path.join(OUT_DIR, f) for f in sorted(os.listdir(OUT_DIR))
             if f.endswith(".py")] if os.path.isdir(OUT_DIR) else []
    print(f"\n{len(cands)} agents extracted -> rank them with:")
    print(f"  .venv/bin/python arena.py roundrobin {args.champion} " + " ".join(cands[:6]))

    if args.apply:
        print("\n--apply builds the delta onto a base you name:")
        print("  python -c \"import sys; sys.path.insert(0,'sweep'); import variants;"
              " src=open(BASE).read();"
              " open(OUT,'w').write(variants.build_spec(src,"
              " {'opponent_plan':True,'settings':dict(variants.parse_settings(src),"
              " front_run=True)}))\"")


if __name__ == "__main__":
    main()
