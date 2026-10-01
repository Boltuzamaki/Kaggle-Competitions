"""Recurrent factorized actor-critic policy."""
import torch
from torch import nn
from torch.distributions import Categorical
from .actions import UNIT_ACTIONS, MARKET_ACTIONS


class RecurrentPolicy(nn.Module):
    def __init__(self, hidden=256, global_size=96, unit_features=20):
        super().__init__()
        self.hidden=hidden
        self.board=nn.Sequential(nn.Conv2d(24,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,48,3,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d((2,2)),nn.Flatten())
        self.global_net=nn.Sequential(nn.Linear(global_size,128),nn.SiLU())
        self.core=nn.GRUCell(48*4+128,hidden)
        self.unit_net=nn.Sequential(nn.Linear(hidden+unit_features,hidden),nn.SiLU(),nn.Linear(hidden,len(UNIT_ACTIONS)))
        self.market_embed=nn.Embedding(len(MARKET_ACTIONS),64)
        self.market_core=nn.GRUCell(hidden+64,hidden)
        self.market_head=nn.Linear(hidden,len(MARKET_ACTIONS))
        self.value=nn.Sequential(nn.Linear(hidden,128),nn.SiLU(),nn.Linear(128,1))

    def initial_state(self,batch,device=None):
        return torch.zeros(batch,self.hidden,device=device or next(self.parameters()).device)

    def forward_core(self,board,global_vec,state):
        z=torch.cat([self.board(board),self.global_net(global_vec)],-1)
        h=self.core(z,state)
        return h,self.value(h).squeeze(-1)

    @staticmethod
    def _dist(logits,mask):
        return Categorical(logits=logits.masked_fill(~mask,-1e9))

    def act(self,batch,state,greedy=False):
        h,value=self.forward_core(batch["board"],batch["global"],state)
        b,u,_=batch["units"].shape
        uh=h[:,None,:].expand(-1,u,-1)
        ud=self._dist(self.unit_net(torch.cat([uh,batch["units"]],-1)),batch["unit_mask"])
        ua=ud.logits.argmax(-1) if greedy else ud.sample()
        logp=ud.log_prob(ua).sum(-1); entropy=ud.entropy().sum(-1)
        prev=torch.zeros(b,dtype=torch.long,device=h.device); mh=h; market=[]
        alive=torch.ones(b,dtype=torch.bool,device=h.device)
        for slot in range(10):
            mh=self.market_core(torch.cat([h,self.market_embed(prev)],-1),mh)
            md=self._dist(self.market_head(mh),batch["market_mask"][:,slot])
            a=md.logits.argmax(-1) if greedy else md.sample()
            market.append(a); logp+=md.log_prob(a)*alive; entropy+=md.entropy()*alive
            alive=alive & (a!=0); prev=a
        return ua,torch.stack(market,1),logp,entropy,value,h

    def evaluate_actions(self,batch,state,unit_actions,market_actions):
        h,value=self.forward_core(batch["board"],batch["global"],state)
        b,u,_=batch["units"].shape; uh=h[:,None,:].expand(-1,u,-1)
        ud=self._dist(self.unit_net(torch.cat([uh,batch["units"]],-1)),batch["unit_mask"])
        logp=ud.log_prob(unit_actions).sum(-1); entropy=ud.entropy().sum(-1)
        prev=torch.zeros(b,dtype=torch.long,device=h.device); mh=h
        alive=torch.ones(b,dtype=torch.bool,device=h.device)
        for slot in range(10):
            mh=self.market_core(torch.cat([h,self.market_embed(prev)],-1),mh)
            md=self._dist(self.market_head(mh),batch["market_mask"][:,slot])
            a=market_actions[:,slot]; logp+=md.log_prob(a)*alive; entropy+=md.entropy()*alive
            alive=alive & (a!=0); prev=a
        return logp,entropy,value,h
