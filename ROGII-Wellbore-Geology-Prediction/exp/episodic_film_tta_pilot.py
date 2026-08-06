"""Legal fixed-120-well pilot: episodic TCN with prefix-only latent adaptation.

The shared network is trained only on development wells.  For each held-out well,
all network weights are frozen and an 8-D latent is optimized using rolling-origin
forecast losses wholly inside the organizer-visible TVT_input prefix.  Hidden TVT
is loaded only after predictions have been frozen for scoring.
"""
from pathlib import Path
import glob, json, sys, time
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/episodic_film_tta_pilot"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "exp"))
from random_prefix_complete_well_gpu_cv import load_well, episode, Block, L, rmse, OFF  # noqa

SEED = 260803
LATENT = 8


class FilmNet(nn.Module):
    def __init__(self, cin=20, width=64):
        super().__init__()
        self.stem = nn.Conv1d(cin, width, 1)
        self.blocks = nn.ModuleList([Block(width, d) for d in (1,2,4,8,16,32,64)])
        self.film = nn.ModuleList([nn.Linear(LATENT, 2*width) for _ in self.blocks])
        self.head = nn.Sequential(nn.Conv1d(width, 32, 3, padding=1), nn.SiLU(),
                                  nn.Conv1d(32, 1, 1))

    def forward(self, x, latent):
        h = self.stem(x)
        for block, affine in zip(self.blocks, self.film):
            h = block(h)
            scale, bias = affine(latent).chunk(2, dim=1)
            h = h * (1 + .10*torch.tanh(scale)[:, :, None]) + .10*bias[:, :, None]
        return self.head(h).squeeze(1)


def tensor_episode(s, cut, device):
    x, target, baseline, mask = flat_episode(s, int(cut))
    return (torch.tensor(x[None], device=device),
            torch.tensor(target[None], device=device), baseline, mask)


def flat_episode(s, cut):
    """Replace the unstable prefix-slope anchor with last-visible TVT."""
    x, old_target, old_base, mask = episode(s, cut)
    tvt = old_base + 20*old_target
    baseline = np.full(L, tvt[cut], dtype=np.float32)
    x[12] = 0
    gr=s["v"]["GR"]; gmed=np.median(gr[:cut+1])
    gs=1.4826*np.median(np.abs(gr[:cut+1]-gmed))+5
    for j,off in enumerate(OFF):
        x[13+j]=(gr-np.interp(tvt[cut]+off,s["tw_tvt"],s["tw_gr"]))/gs
    return x, ((tvt-baseline)/20).astype(np.float32), baseline, mask


def main(epochs=18):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    rng = np.random.RandomState(SEED)
    torch.manual_seed(SEED)
    seq = [s for s in map(load_well, sorted(glob.glob(str(ROOT / "data/train/*__horizontal_well.csv")))) if s]
    chosen = np.sort(rng.choice(len(seq), 120, replace=False))
    seq = [seq[i] for i in chosen]
    # Locked before outcomes: first 80 sampled wells develop the shared operator;
    # final 40 are a disjoint confirmation set.
    train_ids, valid_ids = np.arange(80), np.arange(80, 120)
    device = "cuda"
    model = FilmNet().to(device)
    train_latent = nn.Embedding(80, LATENT).to(device)
    nn.init.zeros_(train_latent.weight)
    opt = torch.optim.AdamW(list(model.parameters()) + list(train_latent.parameters()),
                            lr=1.2e-3, weight_decay=3e-4)
    started = time.time()
    for ep in range(epochs):
        model.train()
        for st in range(0, 80, 8):
            ids = rng.permutation(train_ids)[st:st+8]
            eps = [flat_episode(seq[i], rng.randint(max(32, int(.20*L)), int(.72*L))) for i in ids]
            x = torch.tensor(np.stack([e[0] for e in eps]), device=device)
            y = torch.tensor(np.stack([e[1] for e in eps]), device=device)
            mask = torch.tensor(np.stack([e[3] for e in eps]), device=device)
            p = model(x, train_latent(torch.tensor(ids, device=device)))
            suffix = 1-mask
            loss = ((p-y).square()*suffix).sum()/suffix.sum().clamp_min(1)
            loss += 1e-3*train_latent.weight.square().mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2); opt.step()
        if ep % 3 == 2:
            print(f"epoch={ep+1} loss={loss.item():.5f}", flush=True)

    model.eval()
    records=[]; truth=[]; frozen=[]; zero=[]
    for jj, i in enumerate(valid_ids):
        s=seq[i]; org=s["org_cut"]
        latent=torch.zeros(1,LATENT,device=device,requires_grad=True)
        aopt=torch.optim.Adam([latent],lr=.08)
        cuts=np.unique(np.linspace(max(24,int(.35*org)),max(25,int(.82*org)),6).astype(int))
        # Each target interval ends at org_cut: no held-out suffix label enters adaptation.
        for _ in range(35):
            losses=[]
            for cut in cuts:
                x,y,_,_=tensor_episode(s,cut,device); p=model(x,latent)
                hi=org+1; losses.append((p[:,cut+1:hi]-y[:,cut+1:hi]).square().mean())
            loss=torch.stack(losses).mean()+.01*latent.square().mean()
            aopt.zero_grad(); loss.backward(); aopt.step()
        with torch.no_grad():
            x,_,base,_=tensor_episode(s,org,device)
            pa=(base+20*model(x,latent).cpu().numpy()[0])
            p0=(base+20*model(x,torch.zeros_like(latent)).cpu().numpy()[0])
        n=len(s["h"]); pos=np.linspace(0,1,n); mk=np.arange(n)>s["org_ps"]
        # Freeze predictions before accessing hidden TVT for scoring.
        pa=np.interp(pos,np.linspace(0,1,L),pa)[mk]
        p0=np.interp(pos,np.linspace(0,1,L),p0)[mk]
        yt=s["h"].TVT.to_numpy(float)[mk]
        truth.append(yt); frozen.append(pa); zero.append(p0)
        records.append({"well":s["well"],"rows":int(mk.sum()),"latent_norm":float(latent.norm()),
                        "zero_rmse":rmse(yt,p0),"adapted_rmse":rmse(yt,pa)})
        print(f"confirm {jj+1}/40",flush=True)
    y,p,p0=map(np.concatenate,(truth,frozen,zero))
    grid=[]
    for a in (0,.1,.2,.3,.5,.7,1):
        q=(1-a)*p0+a*p
        grid.append({"adapt_blend":a,"rmse":rmse(y,q)})
    result={"seed":SEED,"sample_wells":120,"development_wells":80,"confirmation_wells":40,
            "epochs":epochs,"zero_latent_rmse":rmse(y,p0),"adapted_rmse":rmse(y,p),
            "best_diagnostic_blend":min(grid,key=lambda z:z["rmse"]),"grid":grid,
            "well_wins":int(sum(r["adapted_rmse"]<r["zero_rmse"] for r in records)),
            "runtime_seconds":time.time()-started,
            "legality":"shared weights exclude confirmation wells; TTA targets stop at visible-prefix end"}
    (OUT/"summary.json").write_text(json.dumps(result,indent=2))
    (OUT/"well_records.json").write_text(json.dumps(records,indent=2))
    print(json.dumps(result,indent=2))

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv)>1 else 18)
