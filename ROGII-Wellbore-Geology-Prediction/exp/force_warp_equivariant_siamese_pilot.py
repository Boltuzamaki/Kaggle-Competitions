"""Legal FORCE GR warp-equivariant pretraining + disjoint ROGII alignment pilot.

The external objective is correspondence, not generic reconstruction: two views of
the same FORCE interval receive independent acquisition distortions and the model
must identify the known relative depth displacement.  The pretrained convolution
weights then initialize the existing supervised horizontal/typewell Siamese pilot.
"""
from pathlib import Path
import io, json, zipfile
import numpy as np
import torch
import torch.nn.functional as F

import siamese_gr_patch_alignment_cv as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/force_warp_equivariant_siamese"
ZIP = ROOT / "external_data/force_2020/LAS_files_Force_2020_all_wells_train_test_blind_hidden_final.zip"


def load_force(max_wells=118):
    """Read GR only from LAS files without using any FORCE labels."""
    curves = []
    with zipfile.ZipFile(ZIP) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".las")][:max_wells]
        for name in names:
            text = z.read(name).decode("latin1")
            head, asc = text.split("~Ascii", 1)
            csec = head.split("~Curve", 1)[1].split("~Parameter", 1)[0]
            cols = [ln.split(".", 1)[0].strip() for ln in csec.splitlines() if "." in ln]
            if "GR" not in cols:
                continue
            a = np.loadtxt(io.StringIO(asc), dtype=np.float32)
            gr = a[:, cols.index("GR")]
            gr = gr[np.isfinite(gr) & (gr > -900)]
            if len(gr) >= 512:
                # FORCE spacing is ~0.5 ft; robust scaling is done per crop.
                curves.append(gr)
    return curves


def aug(x, rng):
    x = np.asarray(x, np.float32).copy()
    lo, hi = np.percentile(x, [1, 99]); x = np.clip(x, lo, hi)
    x = (x - np.median(x)) / max(np.percentile(x, 75)-np.percentile(x, 25), 2.)
    x = x * rng.uniform(.65, 1.45) + rng.normal(0, .25)
    if rng.random() < .7:
        k = int(rng.choice([3, 5, 7])); x = np.convolve(x, np.ones(k)/k, mode="same")
    x += rng.normal(0, rng.uniform(.02, .18), len(x))
    if rng.random() < .5:
        i = rng.integers(5, len(x)-6); x[i:i+rng.integers(2, 8)] = x[max(0, i-1)]
    return x.astype(np.float32)


def ssl_pretrain(curves, dev, epochs=2, steps=80, batch=96):
    enc = base.Encoder().to(dev)
    opt = torch.optim.AdamW(enc.parameters(), 1e-3, weight_decay=2e-4)
    rng = np.random.default_rng(7319); offsets = np.arange(-24, 25, 4); half = 20
    losses=[]; acc=[]
    for ep in range(epochs):
        for _ in range(steps):
            q, cand = [], []
            for _b in range(batch):
                g = curves[rng.integers(len(curves))]
                c = rng.integers(80, len(g)-81)
                # mild local stretch/compression before independent acquisition views
                scale = rng.uniform(.90, 1.10)
                pos = c + scale*np.arange(-half-24, half+25)
                src = np.interp(pos, np.arange(len(g)), g)
                query = aug(src[24:24+2*half+1], rng)
                cs=[]
                for d in offsets:
                    st=24+d; cs.append(aug(src[st:st+2*half+1], rng))
                q.append(np.stack([query, np.convolve(query,np.ones(5)/5,"same"),np.gradient(query)]))
                cand.append(np.stack([np.stack([v,np.convolve(v,np.ones(5)/5,"same"),np.gradient(v)]) for v in cs]))
            qt=torch.from_numpy(np.stack(q).astype(np.float32)).to(dev)
            ct=torch.from_numpy(np.stack(cand).astype(np.float32)).to(dev)
            zq=enc(qt); zc=enc(ct.flatten(0,1)).reshape(batch,len(offsets),-1)
            score=torch.einsum("bd,bkd->bk",zq,zc)*12
            target=torch.full((batch,),len(offsets)//2,device=dev,dtype=torch.long)
            loss=F.cross_entropy(score,target)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss)); acc.append(float((score.argmax(1)==target).float().mean()))
        print(f"ssl_epoch={ep} loss={np.mean(losses[-steps:]):.4f} acc={np.mean(acc[-steps:]):.4f}",flush=True)
    return enc.state_dict(), {"ssl_loss":float(np.mean(losses[-steps:])),"ssl_accuracy":float(np.mean(acc[-steps:]))}


def train_initialized(items, dev, state, epochs=5, max_station=140):
    model=base.Siamese().to(dev); model.he.load_state_dict(state); model.te.load_state_dict(state)
    opt=torch.optim.AdamW(model.parameters(),8e-4,weight_decay=2e-4); rng=np.random.default_rng(2026)
    for ep in range(epochs):
        samples=base.batches(items,max_station,rng); losses=[]; model.train()
        for st in range(0,len(samples),96):
            h,t,c=base.make_batch(items,samples[st:st+96],base.NEG); h,t,c=h.to(dev),t.to(dev),c.to(dev)
            score=model(h,t,c); target=torch.full((len(h),),len(base.NEG)//2,device=dev,dtype=torch.long)
            loss=F.cross_entropy(score,target,label_smoothing=.02)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2); opt.step(); losses.append(float(loss))
        print(f"finetune_epoch={ep} loss={np.mean(losses):.4f}",flush=True)
    return model


def main():
    # Reuse the exact fixed 200-well/fold-0 protocol of the rejected baseline.
    import glob, joblib, pandas as pd
    from sklearn.model_selection import GroupKFold
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(2026); OUT.mkdir(parents=True,exist_ok=True)
    curves=load_force(); state,sm=ssl_pretrain(curves,dev)
    feat=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl"); vo=np.asarray(joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]["lgb7"])
    v4={w:(vo[q.index].astype(np.float32),q.index.to_numpy()) for w,q in feat.groupby("well",sort=False)}
    paths=sorted(glob.glob(str(ROOT/"data/train/*__horizontal_well.csv")))[:200]
    wells=[q for q in (base.load(p,*v4[Path(p).name.split("__")[0]]) for p in paths) if q]
    ti,vi=list(GroupKFold(5).split(wells,groups=[w["well"] for w in wells]))[0]
    model=train_initialized([wells[i] for i in ti],dev,state); model.eval()
    se=bs=nr=top1=top3=0
    for j in vi:
        w=wells[j]; score=base.emissions(model,w,dev); pred=base.decode(score); m=w["rows"]>=w["ps"]
        res=w["truth"][w["rows"]]-(w["anchor"]+w["base"][w["rows"]]); se+=np.square(pred[m]-res[m]).sum(); bs+=np.square(res[m]).sum(); nr+=m.sum()
        true=np.argmin(abs(base.OFF[None,:]-res[m,None]),axis=1); rank=np.argsort(score[m],axis=1)[:,::-1]
        top1+=(rank[:,0]==true).sum(); top3+=np.any(rank[:,:3]==true[:,None],axis=1).sum()
    summary={**sm,"external":"FORCE 2020 GR only, CC BY 4.0","external_wells":len(curves),"valid_wells":len(vi),"rows":int(nr),"pooled_rmse":float(np.sqrt(se/nr)),"v4_rmse":float(np.sqrt(bs/nr)),"emission_top1":float(top1/nr),"emission_top3":float(top3/nr),"comparison_baseline_rmse":30.867419047750026,"comparison_baseline_top1":0.029398552533067134,"no_submission":True}
    torch.save(model.state_dict(),OUT/"fold0.pt"); (OUT/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
