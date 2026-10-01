"""Paired evaluation of a neural checkpoint against scripted and frozen agents."""
import argparse,json,statistics
from pathlib import Path
from kaggle_environments import make
from .runtime import NeuralAgent,load_scripted


def play(checkpoint,opponent,seed,seat,steps=720):
    neural=NeuralAgent(checkpoint,"cpu",True)
    other=NeuralAgent(opponent,"cpu",True) if str(opponent).endswith(".pt") else load_scripted(opponent)
    env=make("kaggriculture",configuration={"episodeSteps":steps,"seed":seed},debug=False); state=env.reset(2)
    for _ in range(steps):
        acts=[None,None]; acts[seat]=neural(state[seat].observation); acts[1-seat]=other(state[1-seat].observation)
        state=env.step(acts)
        if all(str(s.status)=="DONE" for s in state): break
    mine=float(state[seat].reward or 0); theirs=float(state[1-seat].reward or 0)
    return {"seed":seed,"seat":seat,"mine":mine,"theirs":theirs,"score":1 if mine>theirs else (.5 if mine==theirs else 0),
            "clean":all(str(s.status)=="DONE" for s in state)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("checkpoint"); ap.add_argument("--opponents",nargs="*",default=["main.py"])
    ap.add_argument("--seeds",type=int,default=5); ap.add_argument("--steps",type=int,default=720); ap.add_argument("--out")
    a=ap.parse_args(); report={}
    for opp in a.opponents:
        rows=[play(a.checkpoint,opp,s,seat,a.steps) for s in range(a.seeds) for seat in (0,1)]
        report[opp]={"score":statistics.mean(x["score"] for x in rows),"mean_bank":statistics.mean(x["mine"] for x in rows),
                     "clean":all(x["clean"] for x in rows),"games":len(rows),"rows":rows}
        print(f"{opp:40s} score={report[opp]['score']:.3f} bank={report[opp]['mean_bank']:,.0f} clean={report[opp]['clean']}")
    if a.out: Path(a.out).write_text(json.dumps(report,indent=2))


if __name__=="__main__": main()

