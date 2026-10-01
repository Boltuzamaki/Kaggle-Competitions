"""CPU rollout worker. Each process owns its engine and samples opponents."""
import random
from pathlib import Path
import numpy as np
import torch
from kaggle_environments import make
from .encoding import encode
from .runtime import NeuralAgent, load_scripted, tensorize
from .model import RecurrentPolicy
from .actions import decode_unit, decode_market


def _action(ua,ma,n):
    unit=[decode_unit(x) for x in ua[:n].tolist()]
    return {"farmer":unit[0],"hands":unit[1:],"market":decode_market(ma.tolist())}


def rollout(checkpoint,seed,config,opponent_path="main.py",population_dir=None):
    """One full episode; returns learner-seat transitions and terminal result."""
    rng=random.Random(seed); device="cpu"
    payload=torch.load(checkpoint,map_location=device,weights_only=False)
    model=RecurrentPolicy(payload.get("config",{}).get("hidden_size",256)); model.load_state_dict(payload["model"]); model.eval()
    candidates=[Path(opponent_path)]
    if population_dir:
        candidates += sorted(Path(population_dir).glob("*.pt"))
    chosen=rng.choice(candidates); neural=chosen.suffix==".pt"
    opponent=NeuralAgent(chosen,device="cpu",greedy=False) if neural else load_scripted(chosen)
    seat=rng.randrange(2); env=make("kaggriculture",configuration={"episodeSteps":config["episode_steps"],"seed":seed},debug=False)
    state=env.reset(2); h=model.initial_state(1); trajectory=[]
    for _ in range(config["episode_steps"]):
        obs=state[seat].observation
        e=encode(obs,config["max_units"]); batch=tensorize([e],device)
        h0=h.detach().cpu().numpy()[0]
        with torch.inference_mode(): ua,ma,lp,en,val,h=model.act(batch,h,greedy=False)
        learner_action=_action(ua[0],ma[0],e["num_units"])
        other_obs=state[1-seat].observation; other_action=opponent(other_obs)
        acts=[None,None]; acts[seat]=learner_action; acts[1-seat]=other_action
        trajectory.append({"encoded":e,"unit_actions":ua[0].cpu().numpy(),"market_actions":ma[0].cpu().numpy(),
                           "old_logp":float(lp.item()),"old_value":float(val.item()),"h0":h0})
        state=env.step(acts)
        if all(str(s.status)=="DONE" for s in state): break
    mine=float(state[seat].reward or 0); theirs=float(state[1-seat].reward or 0)
    outcome=1.0 if mine>theirs else (-1.0 if mine<theirs else 0.0)
    rewards=np.zeros(len(trajectory),np.float32); rewards[-1]=outcome
    values=np.asarray([x["old_value"] for x in trajectory],np.float32)
    adv=np.zeros_like(rewards); last=0.0
    for t in range(len(rewards)-1,-1,-1):
        nxt=values[t+1] if t+1<len(values) else 0.0
        delta=rewards[t]+config["gamma"]*nxt-values[t]
        last=delta+config["gamma"]*config["gae_lambda"]*last; adv[t]=last
    returns=adv+values
    for i,x in enumerate(trajectory): x["advantage"]=float(adv[i]); x["return"]=float(returns[i])
    return trajectory,{"outcome":outcome,"bank":mine,"opponent_bank":theirs,"seat":seat,"opponent":str(chosen)}


def worker_loop(worker_id,in_q,out_q):
    while True:
        job=in_q.get()
        if job is None: return
        try: out_q.put((worker_id,rollout(**job),None))
        except Exception as exc: out_q.put((worker_id,None,repr(exc)))

