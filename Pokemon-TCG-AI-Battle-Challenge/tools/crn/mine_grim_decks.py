"""Mine exact Grimmsnarl lists from locally cached high-rated replays."""
from __future__ import annotations
import csv, glob, json, multiprocessing as mp, os
from collections import Counter, defaultdict

ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIN=float(os.environ.get("MGD_MIN","900"))
TARGET=int(os.environ.get("MGD_CARD","648"))

def inspect(args):
    path,score=args
    try:d=json.load(open(path))
    except Exception:return []
    steps=d.get("steps") or []; rewards=d.get("rewards") or [0,0]; out=[]
    for seat in (0,1):
        deck=None
        for si in (1,0):
            if si<len(steps):
                action=steps[si][seat].get("action")
                if isinstance(action,list) and len(action)==60:deck=action;break
        if deck and TARGET in deck:
            out.append((tuple(sorted(deck)),int(rewards[seat]>rewards[1-seat]),score))
    return out

def main():
    scores={}
    for path in glob.glob(os.path.join(ROOT,"data","ep_*","manifest.csv")):
        for row in csv.DictReader(open(path)):
            try:scores[row["episode_id"]]=(float(row["min_score"]),float(row["avg_score"]))
            except Exception:pass
    jobs=[]
    for path in glob.glob(os.path.join(ROOT,"data","ep_*","*.json")):
        score=scores.get(os.path.basename(path)[:-5])
        if score and score[0]>=MIN:jobs.append((path,score[1]))
    count=Counter();wins=Counter();ratings=defaultdict(list)
    with mp.Pool(int(os.environ.get("MGD_WORKERS","16"))) as pool:
        for rows in pool.imap_unordered(inspect,jobs,chunksize=8):
            for sig,won,rating in rows:count[sig]+=1;wins[sig]+=won;ratings[sig].append(rating)
    print(f"minimum={MIN:g} card={TARGET} lists={len(count)} games={sum(count.values())}")
    for rank,sig in enumerate(sorted(count,key=lambda x:(-count[x],-wins[x]))[:30]):
        print(rank,count[sig],wins[sig],round(sum(ratings[sig])/len(ratings[sig]),1),dict(sorted(Counter(sig).items())))

if __name__=="__main__":main()
