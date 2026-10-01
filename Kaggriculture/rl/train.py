"""Distributed population self-play PPO trainer.

CPU workers run independent official-engine episodes. The GPU learner consumes
their trajectories, applies clipped PPO updates, and periodically freezes a
checkpoint into the historical opponent population.
"""
import argparse,json,math,multiprocessing as mp,random,shutil,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .config import TrainConfig
from .model import RecurrentPolicy
from .runtime import save_checkpoint,tensorize
from .worker import worker_loop


def batches(xs,n,rng):
    order=list(range(len(xs))); rng.shuffle(order)
    for i in range(0,len(order),n): yield [xs[j] for j in order[i:i+n]]


def learn(model,opt,transitions,cfg,device,rng):
    adv=np.asarray([x["advantage"] for x in transitions],np.float32); adv=(adv-adv.mean())/(adv.std()+1e-6)
    for x,a in zip(transitions,adv): x["norm_adv"]=float(a)
    stats=[]
    for _ in range(cfg.epochs):
        for rows in batches(transitions,cfg.batch_sequences,rng):
            batch=tensorize([x["encoded"] for x in rows],device)
            h=torch.as_tensor(np.stack([x["h0"] for x in rows]),device=device,dtype=torch.float32)
            ua=torch.as_tensor(np.stack([x["unit_actions"] for x in rows]),device=device,dtype=torch.long)
            ma=torch.as_tensor(np.stack([x["market_actions"] for x in rows]),device=device,dtype=torch.long)
            old=torch.tensor([x["old_logp"] for x in rows],device=device); ret=torch.tensor([x["return"] for x in rows],device=device)
            a=torch.tensor([x["norm_adv"] for x in rows],device=device)
            logp,entropy,value,_=model.evaluate_actions(batch,h,ua,ma)
            ratio=(logp-old).exp(); pg=-torch.minimum(ratio*a,ratio.clamp(1-cfg.clip_ratio,1+cfg.clip_ratio)*a).mean()
            vf=.5*(value-ret).square().mean(); ent=entropy.mean()
            loss=pg+cfg.value_coef*vf-cfg.entropy_coef*ent
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),cfg.max_grad_norm); opt.step()
            stats.append((loss.item(),pg.item(),vf.item(),ent.item()))
    return np.mean(stats,axis=0).tolist()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--updates",type=int,default=None); ap.add_argument("--actors",type=int,default=None)
    ap.add_argument("--device",default=None); ap.add_argument("--resume"); ap.add_argument("--smoke",action="store_true"); args=ap.parse_args()
    cfg=TrainConfig();
    if args.updates: cfg.updates=args.updates
    if args.actors: cfg.actors=args.actors
    if args.device: cfg.device=args.device
    if args.smoke: cfg.updates=1; cfg.actors=1; cfg.episode_steps=48; cfg.batch_sequences=32
    device=torch.device(cfg.device if cfg.device!="cuda" or torch.cuda.is_available() else "cpu")
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    model=RecurrentPolicy(cfg.hidden_size).to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate)
    start=0
    if args.resume:
        p=torch.load(args.resume,map_location=device,weights_only=False); model.load_state_dict(p["model"])
        if p.get("optimizer"): opt.load_state_dict(p["optimizer"])
        start=int(p.get("update",0))+1
    root=Path(cfg.checkpoint_dir); pop=root/"population"; pop.mkdir(parents=True,exist_ok=True); latest=root/"latest.pt"
    save_checkpoint(latest,model,opt,cfg,start)
    # Seed the league with the random initial policy. Without this, the first
    # policy sees only v9, receives an almost constant loss, and has no useful
    # ranking signal from which to bootstrap.
    initial=pop/"policy_000000.pt"
    if not initial.exists(): shutil.copy2(latest,initial)
    ctx=mp.get_context("spawn"); iq=ctx.Queue(); oq=ctx.Queue(); workers=[]
    for wid in range(cfg.actors):
        p=ctx.Process(target=worker_loop,args=(wid,iq,oq),daemon=True); p.start(); workers.append(p)
    rng=random.Random(cfg.seed)
    try:
        for update in range(start,cfg.updates):
            # Workers reload one immutable checkpoint per episode.
            save_checkpoint(latest,model,opt,cfg,update)
            for wid in range(cfg.actors):
                iq.put({"checkpoint":str(latest),"seed":cfg.seed+update*1000+wid,"config":cfg.to_dict(),
                        "opponent_path":"main.py","population_dir":str(pop)})
            transitions=[]; episodes=[]
            for _ in workers:
                wid,result,error=oq.get(timeout=3600)
                if error: raise RuntimeError(f"worker {wid}: {error}")
                traj,summary=result; transitions.extend(traj); episodes.append(summary)
            losses=learn(model,opt,transitions,cfg,device,rng)
            score=np.mean([(x["outcome"]+1)/2 for x in episodes]); bank=np.mean([x["bank"] for x in episodes])
            metrics={"score":float(score),"mean_bank":float(bank),"loss":losses,"episodes":episodes}
            print(json.dumps({"update":update,"transitions":len(transitions),**metrics},default=float),flush=True)
            save_checkpoint(latest,model,opt,cfg,update,metrics)
            if update%cfg.snapshot_every==0:
                shutil.copy2(latest,pop/f"policy_{update:06d}.pt")
    finally:
        for _ in workers: iq.put(None)
        for p in workers: p.join(timeout=5)


if __name__=="__main__": main()
