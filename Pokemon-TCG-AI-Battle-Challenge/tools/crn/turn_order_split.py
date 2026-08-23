"""Measure OUR agent's win-rate split by whether it actually went first or second.

Discussion 723591 (5,333 games): going first wins 55.2%, and 91.5% of agents grab
first when they win the toss. So first is available only ~54% of the time, while
second can be secured ~96% of the time. Their conjecture: "second is the
controllable state… a build that's strong going second might be worth investing in."

That is only actionable if our agent's second-turn deficit differs from the field's.
The arena's first_player_winrate column reports who-went-first overall, not our
split, so this measures ours directly using the engine's `firstPlayer` field.
"""
import os, sys, random, ctypes, json
from collections import Counter
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(ROOT,'agent'))
from paired_eval import build, _lib, StartData, SerialData

def play_track(seed, a0, a1, d0, d1, me_seat):
    cards=(ctypes.c_int*120)(*(list(d0)+list(d1)))
    st=_lib.CrnBattleStart(cards, ctypes.c_uint(seed & 0xFFFFFFFF or 1))
    if st.errorPlayer>=0: return None,None
    ptr=st.battlePtr; agents=(a0,a1); first=None
    try:
        for _ in range(6000):
            sd=_lib.GetBattleData(ptr)
            if not sd.json: return None,first
            obs=json.loads(ctypes.string_at(sd.json).decode('utf-8','replace'))
            obs['search_begin_input']=ctypes.string_at(sd.data,sd.count).decode('ascii')
            cur=obs.get('current') or {}
            fp=cur.get('firstPlayer')
            if first is None and fp in (0,1):
                first = (fp==me_seat)
            if cur.get('result',-1)>=0: return cur['result'],first
            sel=obs.get('select')
            if sel is None: return None,first
            n=len(sel.get('option') or [])
            if n==0: return None,first
            who=sd.selectPlayer if sd.selectPlayer in (0,1) else 0
            try: ch=agents[who](obs)
            except Exception: ch=[0]
            ch=[c for c in (ch or [0]) if isinstance(c,int) and 0<=c<n] or [0]
            arr=(ctypes.c_int*len(ch))(*ch)
            if _lib.Select(ptr,arr,len(ch))!=0: return 1-who,first
        return None,first
    finally: _lib.BattleFinish(ptr)

def main():
    import arena  # noqa
    cand=sys.argv[1]; nseeds=int(sys.argv[2]) if len(sys.argv)>2 else 30
    fa,da=build(cand)
    for oname in ('public-archaludon','public-alakazam'):
        fo,do=build(oname)
        rec=Counter()
        for s in [60000+i for i in range(nseeds)]:
            for seat in (0,1):
                random.seed(s)
                if seat==0: w,first=play_track(s,fa,fo,da,do,0); won=(w==0)
                else:       w,first=play_track(s,fo,fa,do,da,1); won=(w==1)
                if first is None: continue
                rec[('first' if first else 'second','W' if won else 'L')]+=1
        f_w,f_l=rec[('first','W')],rec[('first','L')]
        s_w,s_l=rec[('second','W')],rec[('second','L')]
        print(f'{cand} vs {oname}:')
        print(f'   went FIRST : {f_w}/{f_w+f_l} = {100*f_w/max(f_w+f_l,1):5.1f}%')
        print(f'   went SECOND: {s_w}/{s_w+s_l} = {100*s_w/max(s_w+s_l,1):5.1f}%')
        print(f'   asymmetry  : {100*f_w/max(f_w+f_l,1)-100*s_w/max(s_w+s_l,1):+5.1f} pts')
main()
