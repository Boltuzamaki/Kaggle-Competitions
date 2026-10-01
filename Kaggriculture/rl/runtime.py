import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from .actions import decode_unit, decode_market
from .encoding import encode
from .model import RecurrentPolicy


def tensorize(items, device):
    keys=("board","global","units","unit_mask","market_mask")
    return {k:torch.as_tensor(np.stack([x[k] for x in items]),device=device,
                              dtype=torch.bool if "mask" in k else torch.float32) for k in keys}


class NeuralAgent:
    def __init__(self, checkpoint, device="cpu", greedy=True):
        payload=torch.load(checkpoint,map_location=device,weights_only=False)
        cfg=payload.get("config",{})
        self.model=RecurrentPolicy(cfg.get("hidden_size",256)).to(device)
        self.model.load_state_dict(payload["model"]); self.model.eval()
        self.device=device; self.greedy=greedy; self.states={}

    def reset(self): self.states.clear()

    @torch.inference_mode()
    def __call__(self,obs,config=None):
        pid=int(obs.get("player",0)); step=int(obs.get("step",0))
        if step==0: self.states.pop(pid,None)
        e=encode(obs); batch=tensorize([e],self.device)
        h=self.states.get(pid,self.model.initial_state(1,self.device))
        ua,ma,_,_,_,h=self.model.act(batch,h,self.greedy); self.states[pid]=h
        unit=[decode_unit(x) for x in ua[0,:e["num_units"]].tolist()]
        return {"farmer":unit[0],"hands":unit[1:],"market":decode_market(ma[0].tolist())}


def load_scripted(path):
    path=Path(path); name="scripted_"+path.stem+"_"+str(abs(hash(path.resolve())))
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    def call(obs):
        if int(obs.get("step",0))==0 and hasattr(mod,"_BRAINS"): mod._BRAINS.clear()
        return mod.agent(obs)
    return call


def save_checkpoint(path,model,optimizer,config,update,metrics=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict() if optimizer else None,
                "config":config.to_dict() if hasattr(config,"to_dict") else dict(config),
                "update":update,"metrics":metrics or {}},tmp)
    tmp.replace(path)

