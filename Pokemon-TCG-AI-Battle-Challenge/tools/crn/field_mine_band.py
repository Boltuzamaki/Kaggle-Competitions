"""Mine the deck metagame AT OUR RATING BAND, not at >=1100.

Why this exists: agent/field_decks.json was mined from games where BOTH players
were rated >=1100. We are playing at ~790. Every local optimisation this project
has run was therefore scored against a metagame we do not actually face, which is
a concrete mechanical reason local gains have failed to transfer to the ladder
(the Ogerpon candidate is the clearest case: +8.4 field-weighted on the >=1100
panel, ~130 ladder points BELOW family_v1 in reality).

Episodes carry info.TeamNames; the public leaderboard maps team -> score. Joining
them lets us bucket every observed deck by the rating of the player piloting it.
Deck is read from steps[1][seat].action, the 60-card list an agent returns on the
deck-request observation.
"""
from __future__ import annotations
import collections, glob, json, multiprocessing as mp, os, sys

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
LB=json.load(open(os.path.join(ROOT,"scratchpad","lb","lb.json")))
SCORE={r["TeamName"]: float(r.get("Score") or 0) for r in LB if r.get("TeamName")}
FILES=sorted(glob.glob(os.path.join(ROOT,"data","episodes","*.json")))
WORKERS=int(os.environ.get("FM_WORKERS","16"))

def sig(deck):
    return "|".join(f"{c}x{n}" for c,n in sorted(collections.Counter(deck).items()))

def one(f):
    try:
        d=json.load(open(f))
        names=(d.get("info") or {}).get("TeamNames") or []
        st=d.get("steps") or []
        if len(st)<2 or len(names)<2: return None
        out=[]
        for seat in (0,1):
            a=st[1][seat].get("action")
            if isinstance(a,list) and len(a)==60:
                out.append((names[seat], sig(a), tuple(sorted(a))))
        return out or None
    except Exception:
        return None

def main():
    print(f"mining {len(FILES)} episodes on {WORKERS} workers", flush=True)
    with mp.Pool(WORKERS) as p:
        res=[r for r in p.map(one, FILES, chunksize=8) if r]
    print(f"  parsed {len(res)} episodes with decks", flush=True)
    bands={"ours_700_880":(700,880), "high_1100+":(1100,9999), "all":(0,9999)}
    decks={}
    counts={b:collections.Counter() for b in bands}
    unrated=0
    for ep in res:
        for name,s,deck in ep:
            decks[s]=list(deck)
            sc=SCORE.get(name)
            if sc is None: unrated+=1; continue
            for b,(lo,hi) in bands.items():
                if lo<=sc<=hi: counts[b][s]+=1
    print(f"  unrated player-slots (team not on LB): {unrated}", flush=True)
    out={}
    for b in bands:
        tot=sum(counts[b].values())
        print(f"\n===== BAND {b}  ({tot} player-games) =====", flush=True)
        top=counts[b].most_common(10)
        out[b]={}
        for i,(s,n) in enumerate(top):
            print(f"    band_{i:02d}  share {100*n/max(tot,1):5.2f}%  n={n:4d}", flush=True)
            out[b][f"band_{i:02d}"]={"share":n/max(tot,1),"n":n,"deck":decks[s],"sig":s}
    json.dump(out, open(os.path.join(ROOT,"agent","field_decks_by_band.json"),"w"), indent=1)
    # how different is our band from the >=1100 field we optimised against?
    a=set(list(out["ours_700_880"])[:8]); 
    sa={out["ours_700_880"][k]["sig"] for k in out["ours_700_880"]}
    sb={out["high_1100+"][k]["sig"] for k in out["high_1100+"]}
    print(f"\n  top-10 deck overlap between our band and >=1100: {len(sa&sb)}/10", flush=True)

if __name__=="__main__": main()
