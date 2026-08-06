"""Test-legal candidate selector using repeated backtests inside TVT_input.

For every well we hide the last 10/25/40/55 percent of the *visible* prefix,
fit simple continuation rules, and score them on the still-known TVT_input.
Those errors and stability diagnostics are then used to choose among the
honest Stack-V4 OOF candidate paths.  The outer selector validation is grouped
by complete well.  No TVT values from the scored suffix enter any feature.
"""
from pathlib import Path
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/prefix_backtest_selector"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "exp"))
from candidate_path_ranker_v2 import make_candidates, cand_features, rmse


def robust_poly(x, y, degree):
    z0, zs = x[-1], max(np.std(x), 1.)
    z = (x-z0)/zs
    keep = np.ones(len(x), bool)
    co = np.polyfit(z, y, degree)
    for _ in range(4):
        co = np.polyfit(z[keep], y[keep], degree)
        r = y-np.polyval(co, z)
        s = 1.4826*np.median(np.abs(r[keep]-np.median(r[keep])))+1e-4
        nk = np.abs(r) < 3*s
        if nk.sum() < degree+8 or np.array_equal(nk, keep):
            break
        keep = nk
    return co, z0, zs


def predict_rule(xtr, ytr, xva, rule):
    if rule == "hold":
        return np.full(len(xva), ytr[-1])
    degree = {"line": 1, "quad": 2, "cubic": 3}[rule]
    # Recent windows prevent the long build-section history dominating the
    # local continuation geometry.
    n = min(len(xtr), 600 if degree == 1 else 900)
    co, z0, zs = robust_poly(xtr[-n:], ytr[-n:], degree)
    return np.polyval(co, (xva-z0)/zs)


def prefix_diagnostics(well):
    f = ROOT / "data/train" / f"{well}__horizontal_well.csv"
    q = pd.read_csv(f, usecols=["MD", "TVT_input"])
    q = q[q.TVT_input.notna()]
    x, y = q.MD.to_numpy(float), q.TVT_input.to_numpy(float)
    rules = ["hold", "line", "quad", "cubic"]
    cuts = [.45, .60, .75, .90]
    feat, losses = {}, {r: [] for r in rules}
    for cut in cuts:
        n = max(20, min(len(x)-5, int(len(x)*cut)))
        for r in rules:
            try:
                loss = rmse(y[n:], predict_rule(x[:n], y[:n], x[n:], r))
            except Exception:
                loss = 1e3
            losses[r].append(min(loss, 1e3))
            feat[f"bt_{r}_{int(cut*100)}"] = loss
    for r, a in losses.items():
        a = np.asarray(a)
        feat[f"bt_{r}_mean"] = a.mean()
        feat[f"bt_{r}_last"] = a[-1]
        feat[f"bt_{r}_std"] = a.std()
        feat[f"bt_{r}_trend"] = np.polyfit(cuts, a, 1)[0]
    means = np.array([feat[f"bt_{r}_mean"] for r in rules])
    feat["bt_best"] = int(np.argmin(means))
    feat["bt_margin"] = float(np.partition(means, 1)[1]-means.min())
    feat["prefix_n"] = len(x)
    feat["prefix_span"] = np.ptp(x)
    # Extrapolations from all visible data: test-legal expectations about the
    # first/last hidden delta and local derivatives.
    horizon = max(1., np.ptp(x))
    for r in rules[1:]:
        n = min(len(x), 600 if r == "line" else 900)
        co, z0, zs = robust_poly(x[-n:], y[-n:], {"line":1,"quad":2,"cubic":3}[r])
        xx = np.array([x[-1]+1, x[-1]+horizon*.25, x[-1]+horizon*.5])
        pp = np.polyval(co, (xx-z0)/zs)-y[-1]
        feat[f"ex_{r}_near"], feat[f"ex_{r}_q1"], feat[f"ex_{r}_q2"] = pp
    # Recent observed slope and curvature stability.
    for n in (50, 150, 400):
        nn = min(n, len(x))
        co = np.polyfit(x[-nn:]-x[-1], y[-nn:]-y[-1], min(2, nn-1))
        feat[f"pre_slope{n}"] = co[-2] if len(co) >= 2 else 0.
        feat[f"pre_curv{n}"] = co[-3] if len(co) >= 3 else 0.
    return feat


def main(limit=0, folds=5):
    d = pd.read_pickle(ROOT/"r_v4b/train_feats.pkl").reset_index(drop=True)
    oo = joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]
    wells = sorted(d.well.unique())
    if limit:
        wells = list(np.random.RandomState(730).choice(
            wells, min(limit, len(wells)), replace=False))
        d = d[d.well.isin(wells)]
    rows, paths, truths = [], {}, {}
    for j, (w, g) in enumerate(d.groupby("well", sort=True)):
        pre = prefix_diagnostics(w)
        cs = make_candidates(g, oo)
        y = g.target.to_numpy(float)
        paths[w], truths[w] = cs, y
        x = g.d_md.to_numpy(float)
        h = max(np.ptp(x), 1.)
        for cid, p in cs.items():
            z = cand_features(g, p, cs, cid)
            z.update(pre)
            # Candidate compatibility with prefix-derived continuation.
            near = np.mean(p[:min(30, len(p))])
            q1 = p[min(len(p)-1, int(.25*len(p)))]
            q2 = p[min(len(p)-1, int(.50*len(p)))]
            slope = np.polyfit(x[:min(200,len(x))], p[:min(200,len(x))], 1)[0]
            z["cand_near"] = near
            z["cand_q1"] = q1
            z["cand_q2"] = q2
            z["cand_initial_slope"] = slope
            for r in ("line", "quad", "cubic"):
                z[f"compat_{r}_near"] = abs(near-pre[f"ex_{r}_near"])
                z[f"compat_{r}_q1"] = abs(q1-pre[f"ex_{r}_q1"])
                z[f"compat_{r}_q2"] = abs(q2-pre[f"ex_{r}_q2"])
            z.update(well=w, candidate=cid, label=rmse(y, p))
            rows.append(z)
        if (j+1) % 100 == 0:
            print("loaded", j+1, flush=True)
    tab = pd.DataFrame(rows)
    base = tab[tab.candidate=="weighted"].set_index("well").label
    tab["rank_label"] = tab.label-tab.well.map(base)
    names = [c for c in tab if c not in ("well","candidate","label","rank_label")]
    X = tab[names].replace([np.inf,-np.inf],np.nan).fillna(0)
    chosen, oracle, soft = {}, {}, {t:{} for t in (.05,.10,.20,.35,.50,1.0)}
    for fold, (tr, va) in enumerate(GroupKFold(folds).split(X, groups=tab.well)):
        m = LGBMRegressor(n_estimators=700, learning_rate=.02, num_leaves=15,
            min_child_samples=35, colsample_bytree=.75, reg_lambda=8,
            verbosity=-1, random_state=800+fold,
            n_jobs=max(1,os.cpu_count()//2))
        m.fit(X.iloc[tr], tab.rank_label.iloc[tr])
        z=tab.iloc[va][["well","candidate","label"]].copy()
        z["score"]=m.predict(X.iloc[va])
        for w,q in z.groupby("well"):
            chosen[w]=q.loc[q.score.idxmin(),"candidate"]
            oracle[w]=q.loc[q.label.idxmin(),"candidate"]
            # Continuous path weighting is less brittle than hard selection.
            # Scores are predicted RMSE regrets, so exp(-score/T) is a natural
            # fixed, label-free conversion to candidate weights.
            for t in soft:
                a=q.score.to_numpy(float)
                ww=np.exp(-(a-a.min())/t); ww/=ww.sum()
                soft[t][w]=sum(v*paths[w][c] for v,c in
                               zip(ww,q.candidate.to_numpy()))
    def pooled(sel):
        return rmse(np.concatenate([truths[w] for w in sel]),
                    np.concatenate([paths[w][c] for w,c in sel.items()]))
    baseline={w:"weighted" for w in truths}
    res={"wells":len(truths),"rows":sum(map(len,truths.values())),
         "baseline":pooled(baseline),"selector":pooled(chosen),
         "oracle":pooled(oracle)}
    for t, pp in soft.items():
        res[f"soft_t{t:g}"]=rmse(
            np.concatenate([truths[w] for w in pp]),
            np.concatenate([pp[w] for w in pp]))
    # Continuous affine residual correction learned only across outer wells.
    # This directly targets the datum/trend mistakes that discrete candidates
    # approximate with coarse +/- offsets.
    wr=[]
    for w in truths:
        one=tab[(tab.well==w)&(tab.candidate=="weighted")].iloc[0]
        ff={k:one[k] for k in names}
        y, p=truths[w], paths[w]["weighted"]
        u=np.linspace(-.5,.5,len(y))
        co=np.polyfit(u,y-p,1)
        ff.update(well=w,nrows=len(y),off=float(co[1]),tilt=float(co[0]))
        wr.append(ff)
    wt=pd.DataFrame(wr)
    WX=wt[names].replace([np.inf,-np.inf],np.nan).fillna(0)
    corrected={}
    for fold,(tr,va) in enumerate(GroupKFold(folds).split(WX,groups=wt.well)):
        preds=[]
        for target in ("off","tilt"):
            mm=LGBMRegressor(n_estimators=400,learning_rate=.02,num_leaves=9,
                min_child_samples=25,reg_lambda=12,verbosity=-1,
                random_state=990+fold,n_jobs=max(1,os.cpu_count()//2))
            mm.fit(WX.iloc[tr],wt[target].iloc[tr],
                   sample_weight=wt.nrows.iloc[tr])
            preds.append(mm.predict(WX.iloc[va]))
        for k,ix in enumerate(va):
            w=wt.well.iloc[ix]; u=np.linspace(-.5,.5,len(truths[w]))
            corrected[w]=paths[w]["weighted"]+preds[0][k]+preds[1][k]*u
    res["affine_correction"]=rmse(
        np.concatenate([truths[w] for w in corrected]),
        np.concatenate([corrected[w] for w in corrected]))
    res["gain"]=res["baseline"]-res["selector"]
    print(json.dumps(res,indent=2))
    tag="pilot" if limit else "full"
    (OUT/f"{tag}_summary.json").write_text(json.dumps(res,indent=2))
    pd.DataFrame({"well":chosen.keys(),"chosen":chosen.values(),
        "oracle":[oracle[w] for w in chosen]}).to_csv(OUT/f"{tag}_choices.csv",index=False)
    return res


if __name__=="__main__":
    main(int(sys.argv[1]) if len(sys.argv)>1 else 0)
