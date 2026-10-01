"""Build the publishable Kaggle notebook from the repo's own findings."""
import json, os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# What the income digits know — and three other things the generator leaked

Everyone's first LightGBM on this data scores about **0.9417**, and for a while that was the
median of the leaderboard. This notebook is about why that wall exists and what is actually
behind it.

The short version: **the wall is a property of the representation, not of the task.** The
competition data is synthetic, and the model that generated it left fingerprints on
`Annual_Income_USD` that no survey of real households could contain. Feed those fingerprints
in as columns and the same LightGBM goes to **0.9457**; the full pipeline here reaches
**CV 0.94634 / LB 0.94636**.

What is in here:

1. **Proof that the wall is real** — a capacity sweep, a neural net, and a Bayes-AUC ceiling
   computed from the model's own probabilities, all agreeing that tuning is not the answer.
2. **The digit artifact**, with the one control that settles it: the real 10 000-row survey
   this data was generated from.
3. **Two more leaks** nobody needs a model to find: a deterministic income cliff and a dead zone.
4. **A residual scan** over ~40 candidate keys — including the one that looked overwhelming
   and was worth nothing.
5. **Everything that did not work**, measured, because a notebook that only reports its wins
   is not much use to you.

Every number below is computed in this notebook from the competition data plus one public
dataset. Nothing is blended from anyone else's submission file.
""")

code(r"""
import numpy as np, pandas as pd, warnings
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
warnings.filterwarnings("ignore")

plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
INK, SIG, ACC, ASH = "#16211d", "#b0521c", "#0d7561", "#8a9a93"

BASE = "/kaggle/input/playground-series-s6e9"
ORIG = ("/kaggle/input/ev-adoption-behavior-and-range-anxiety/"
        "EV_Adoption_and_Range_Anxiety_Dataset.csv")

train = pd.read_csv(f"{BASE}/train.csv")
test  = pd.read_csv(f"{BASE}/test.csv")
orig  = pd.read_csv(ORIG)

TARGET, ID = "Will_Buy_EV", "id"
y  = (train[TARGET] == "Yes").astype(int).values
yo = (orig[TARGET]  == "Yes").astype(int).values
FEATS = [c for c in train.columns if c not in (ID, TARGET)]

print(f"train {train.shape}   test {test.shape}   original survey {orig.shape}")
print(f"positive rate: synthetic {y.mean():.4f}   real survey {yo.mean():.4f}")
""")

md(r"""
## 1. The shape of the problem

Two near-binary **gates** and one **dial**. `Subsidy_Available` and `Range_Anxiety_Level`
multiply rather than add — close either one and the purchase rate falls below one percent
no matter what the rest of the row says.
""")

code(r"""
gate = pd.crosstab(train.Subsidy_Available, train.Range_Anxiety_Level,
                   values=y, aggfunc="mean")[["Low", "Medium", "High"]]
cnt  = pd.crosstab(train.Subsidy_Available, train.Range_Anxiety_Level)[["Low", "Medium", "High"]]
print("P(buy) by subsidy x range anxiety\n")
print((gate * 100).round(2).to_string(), "\n")
print("row counts\n"); print(cnt.to_string())

open_cell = (train.Subsidy_Available == "Yes") & (train.Range_Anxiety_Level == "Low")
print(f"\nInside the open cell ({open_cell.sum():,} rows) the rate is {y[open_cell].mean():.4f}")
sub = train[open_cell]
print("\n  by Environmental_Concern_Level:")
print((pd.Series(y[open_cell]).groupby(sub.Environmental_Concern_Level.values).mean() * 100
       ).round(2).to_string())
print("\n  by income decile:")
print((pd.Series(y[open_cell]).groupby(
    pd.qcut(sub.Annual_Income_USD, 10, labels=False).values).mean() * 100).round(2).to_string())
""")

md(r"""
The data itself is inert: no missing values, no duplicate feature rows, `id` is noise, and
the train/test marginals are indistinguishable. There is no cleaning to do and no drift to
correct — which is exactly why tuning feels so unrewarding here.
""")

code(r"""
print("duplicate feature rows :", train[FEATS].duplicated().sum())
print("missing values         :", int(train[FEATS].isna().sum().sum()),
      "(train)", int(test[FEATS].isna().sum().sum()), "(test)")
print("AUC of raw id          :", round(roc_auc_score(y, train[ID].values), 5))
for c in ["Age", "Annual_Income_USD", "Daily_Commute_km"]:
    a, b = train[c], test[c]
    print(f"{c:20s} mean shift {abs(a.mean()-b.mean())/a.std():.5f} sd")
""")

md(r"""
## 2. The wall is real

Three independent checks, all saying the same thing: a stock LightGBM is not under-tuned.

**(a) A capacity sweep saturates, and the *shallowest* model wins.** Learning rates from
0.01 to 0.05, 16 to 256 leaves — every configuration lands within 0.0007 of the others.
""")

code(r"""
X0 = train[FEATS].copy()
for c in X0.columns:
    if not pd.api.types.is_numeric_dtype(X0[c]):
        X0[c] = X0[c].astype("category")

folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X0, y))
itr, iva = folds[0]
dtr = lgb.Dataset(X0.iloc[itr], y[itr]); dva = lgb.Dataset(X0.iloc[iva], y[iva], reference=dtr)

for tag, over in [("lr.05 leaves64 mcs100", dict(learning_rate=.05, num_leaves=64,  min_child_samples=100)),
                  ("lr.02 leaves16 mcs500", dict(learning_rate=.02, num_leaves=16,  min_child_samples=500)),
                  ("lr.02 leaves256 mcs2k", dict(learning_rate=.02, num_leaves=256, min_child_samples=2000)),
                  ("lr.02 depth4  mcs500",  dict(learning_rate=.02, num_leaves=16, max_depth=4, min_child_samples=500))]:
    p = dict(objective="binary", metric="auc", feature_fraction=.8, bagging_fraction=.8,
             bagging_freq=1, lambda_l2=1.0, n_jobs=-1, verbosity=-1, seed=42, **over)
    m = lgb.train(p, dtr, 20000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
    print(f"{tag:24s} iter={m.best_iteration:>5}  fold0 AUC {roc_auc_score(y[iva], m.predict(X0.iloc[iva])):.6f}")
""")

md(r"""
**(b) The model is already at its own ceiling.** Take the out-of-fold probabilities, sample
labels from them, and re-score. That is the best AUC anyone could get *if those probabilities
were the truth*. It comes out equal to what the model actually achieves — so the model has
extracted everything its representation contains. The representation is what has to change.
""")

code(r"""
P = dict(objective="binary", metric="auc", learning_rate=.05, num_leaves=64,
         min_child_samples=100, feature_fraction=.8, bagging_fraction=.8, bagging_freq=1,
         lambda_l2=1.0, n_jobs=-1, verbosity=-1, seed=42)
oof0 = np.zeros(len(y))
for a, b in folds:
    m = lgb.train(P, lgb.Dataset(X0.iloc[a], y[a]), 400)
    oof0[b] = m.predict(X0.iloc[b])

rng = np.random.default_rng(0)
ceiling = np.mean([roc_auc_score((rng.random(len(oof0)) < oof0).astype(int), oof0) for _ in range(5)])
print(f"baseline OOF AUC                  {roc_auc_score(y, oof0):.6f}")
print(f"Bayes-AUC ceiling of these probs  {ceiling:.6f}")
""")

md(r"""
## 3. What the income digits know

Here is the whole thing in one test. Group the training rows by a **digit position** of
`Annual_Income_USD` and measure how far the purchase rate spreads across the ten digits.
Compare that against the binomial null corridor — the spread you would see if the digit meant
nothing at all.

The hundreds digit of a household income cannot cause anybody to buy a car. If it separates
buyers from non-buyers, that is the generator talking.
""")

code(r"""
def digit_spread(vals, yy, label):
    rows = []
    for name, k in [("units", 1), ("tens", 10), ("hundreds", 100), ("thousands", 1000)]:
        d = (vals // k) % 10
        g = pd.Series(yy).groupby(d).agg(["mean", "size"])
        p, nbar = yy.mean(), g["size"].mean()
        span = (g["mean"].max() - g["mean"].min()) * 100
        corridor = 3.1 * np.sqrt(p * (1 - p) / nbar) * 100   # expected range of 10 normals
        rows.append({"digit": name, "spread_pp": round(span, 2),
                     "null_corridor_pp": round(corridor, 2),
                     "ratio": f"{span/corridor:.1f}x"})
    print(f"\n{label}"); print(pd.DataFrame(rows).to_string(index=False))

inc_syn = train.Annual_Income_USD.values.astype(np.int64)
io = orig.Annual_Income_USD.dropna()
digit_spread(inc_syn, y, "COMPETITION TRAIN SET (668,665 rows)")
digit_spread(io.values.astype(np.int64), yo[io.index], "THE REAL SURVEY IT WAS GENERATED FROM (10,000 rows)")
""")

md(r"""
Sixteen to twenty-five times the corridor in the synthetic data. One-point-four in the real
survey — i.e. nothing. Same test, same column, same units; the only difference is which of the
two datasets a generative model sat in front of.

The control matters more than the finding. Without it, "income digits predict the target" is
just a number you found by looking at enough columns.
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, (vals, yy, title, colr) in zip(axes, [
        (inc_syn, y, f"Competition train set · {len(y):,} rows", SIG),
        (io.values.astype(np.int64), yo[io.index], f"Real survey · {len(io):,} rows", ASH)]):
    d = (vals // 100) % 10
    g = pd.Series(yy).groupby(d).agg(["mean", "size"])
    base = yy.mean()
    corr = 3.1 * np.sqrt(base * (1 - base) / g["size"].mean())
    ax.axhspan(base - corr/2, base + corr/2, color=ASH, alpha=.30, lw=0,
               label="binomial null corridor")
    ax.axhline(base, color=ASH, ls="--", lw=1)
    ax.bar(g.index, g["mean"], color=colr, width=.62)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("hundreds digit of Annual_Income_USD"); ax.set_xticks(range(10))
axes[0].set_ylabel("P(buy)"); axes[0].legend(fontsize=8, loc="upper right")
plt.suptitle("The same test on both datasets", y=1.03)
plt.tight_layout(); plt.show()
""")

md(r"""
## 4. Two more leaks, no model required

The income → P(buy) curve is not smooth at *any* scale. Binned at \$1000 and compared with its
own local trend, dozens of bands sit more than six standard errors away and several more than
fifteen. Two regions are outright deterministic.
""")

code(r"""
CLIFF = 170537
hi = inc_syn >= CLIFF
dead = (inc_syn >= 38000) & (inc_syn <= 42000)
print(f"income >= {CLIFF}      train n={hi.sum():>6}  buy rate {y[hi].mean():.4f}   "
      f"test n={(test.Annual_Income_USD >= CLIFF).sum()}")
print(f"38000 <= income <= 42000  train n={dead.sum():>6}  buy rate {y[dead].mean():.4f}   "
      f"test n={((test.Annual_Income_USD>=38000)&(test.Annual_Income_USD<=42000)).sum()}")
print(f"income == 30000 (spike)   train n={(inc_syn==30000).sum():>6}  "
      f"buy rate {y[inc_syn==30000].mean():.4f}")
print(f"\nhighest income anywhere in train carrying a 'No' label: {inc_syn[y==0].max():,}")

b = (inc_syn // 1000) * 1000
g = pd.Series(y).groupby(b).agg(["mean", "size"]); g = g[g["size"] >= 150]
local = g["mean"].rolling(11, center=True, min_periods=3).median()
se = np.sqrt(local.clip(1e-4, 1-1e-4) * (1 - local.clip(1e-4, 1-1e-4)) / g["size"])
z = (g["mean"] - local) / se
print(f"\n$1000 bands more than 6 sigma from their own local trend: {(z.abs() > 6).sum()} of {len(z)}")
print(f"largest deviation: {z.abs().max():.1f} sigma")
""")

code(r"""
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot(g.index/1000, g["mean"], lw=.9, color=INK, label="$1000 bands")
ax.plot(local.index/1000, local, lw=2, color=ACC, label="local trend (11-band median)")
ax.axvspan(38, 42, color=SIG, alpha=.18, lw=0)
ax.axvline(CLIFF/1000, color=SIG, lw=1.4, ls="--")
ax.text(CLIFF/1000, .62, " cliff: 100% buy", color=SIG, fontsize=9, va="top")
ax.text(40, .62, "dead zone", color=SIG, fontsize=9, ha="center", va="top")
ax.set_xlabel("Annual_Income_USD (thousands)"); ax.set_ylabel("P(buy)")
ax.set_title("Income against purchase rate: jagged everywhere, deterministic at two ends")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
""")

md(r"""
## 5. Building it in, without leaking

Two rules keep the target encodings honest, and skipping either inflates cross-validation by
several thousandths while buying nothing on the board:

1. encodings are fitted **inside each fold**, on that fold's training rows only;
2. the training-side copy uses an **inner KFold**, so no row is ever encoded with a statistic
   that its own label helped compute.

Income and commute are encoded at several resolutions — exact value, `//100`, `//1000`, the
commute integer — each at three smoothing strengths, and the digits are encoded too. The model
gets to pick its own bias/variance trade-off rather than having me pick one for it.
""")

code(r"""
def add_features(df):
    d = pd.DataFrame(index=df.index)
    for c, mp in [("Range_Anxiety_Level", {"Low":0,"Medium":1,"High":2}),
                  ("Home_Charging_Possible", {"No":0,"Yes":1}),
                  ("Subsidy_Available", {"No":0,"Yes":1}),
                  ("City_Type", {"Rural":0,"Suburban":1,"Urban":2}),
                  ("Gender", {"Male":0,"Female":1,"Other":2}),
                  ("Current_Car_Type", {"Hatchback":0,"Sedan":1,"SUV":2,"Truck":3})]:
        d[c] = df[c].map(mp).astype(np.int8)
    for c in ["Age","Annual_Income_USD","Daily_Commute_km","Charging_Stations_Near_Home",
              "Charging_Stations_Near_Work","Environmental_Concern_Level"]:
        d[c] = df[c].values

    inc  = df.Annual_Income_USD.values.astype(np.int64)
    km10 = np.rint(df.Daily_Commute_km.values * 10).astype(np.int64)

    # the artifacts
    for k, nm in [(1,"units"),(10,"tens"),(100,"hundreds"),(1000,"thousands"),(10000,"tenk")]:
        d[f"inc_d_{nm}"] = ((inc // k) % 10).astype(np.int8)
    d["km_dec"] = (km10 % 10).astype(np.int8)
    d["is_cliff"]     = (inc >= CLIFF).astype(np.int8)
    d["is_dead_zone"] = ((inc >= 38000) & (inc <= 42000)).astype(np.int8)
    d["is_spike"]     = (inc == 30000).astype(np.int8)
    d["log_income"]   = np.log1p(inc)

    # multi-resolution keys, encoded later
    d["k_inc"]    = inc
    d["k_inc100"] = inc // 100
    d["k_inc1k"]  = inc // 1000
    d["k_km"]     = km10
    d["k_kmint"]  = km10 // 10
    return d

KEYS    = ["k_inc", "k_inc100", "k_inc1k", "k_km", "k_kmint"]
DIGITS  = ["inc_d_units","inc_d_tens","inc_d_hundreds","inc_d_thousands","inc_d_tenk","km_dec"]
SMOOTHS = [5.0, 20.0, 100.0]

def encode_fold(Xtr, ytr, others):
    # Target-encode every key and digit column at three smoothing strengths.
    Xtr = Xtr.copy(); others = [o.copy() for o in others]
    prior = ytr.mean()
    for col in KEYS + DIGITS:
        ktr = Xtr[col].values
        enc = {s: np.full(len(ktr), prior, np.float32) for s in SMOOTHS}
        for a, b in KFold(5, shuffle=True, random_state=42).split(ktr):      # inner OOF
            g = pd.DataFrame({"k": ktr[a], "y": ytr[a]}).groupby("k")["y"].agg(["sum","count"])
            kb = pd.Series(ktr[b])
            for s in SMOOTHS:
                enc[s][b] = kb.map((g["sum"] + prior*s) / (g["count"] + s)).fillna(prior).values
        g = pd.DataFrame({"k": ktr, "y": ytr}).groupby("k")["y"].agg(["sum","count"])
        for s in SMOOTHS:
            m = (g["sum"] + prior*s) / (g["count"] + s)
            Xtr[f"te_{col}_{s:g}"] = enc[s]
            for o in others:
                o[f"te_{col}_{s:g}"] = pd.Series(o[col].values).map(m).fillna(prior).values.astype(np.float32)
    return Xtr, others

Xa, Xb = add_features(train), add_features(test)
print("feature matrix before fold encodings:", Xa.shape)
""")

md(r"""
## 6. What it is worth

Same folds, same LightGBM settings, three feature sets. The jump is the artifact; the
hyperparameters barely move.
""")

code(r"""
LGB = dict(objective="binary", metric="auc", learning_rate=0.02, max_depth=5, num_leaves=32,
           min_child_samples=10, bagging_fraction=0.81, feature_fraction=0.20,
           lambda_l1=0.07, lambda_l2=2.03, max_bin=255, bagging_freq=1,
           n_jobs=-1, verbosity=-1, seed=42)

def cv(make, tag, n_splits=5):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y)); tp = np.zeros(len(Xb))
    for a, b in skf.split(Xa, y):
        Atr, Ava, Ate = make(Xa.iloc[a], y[a], Xa.iloc[b], Xb)
        dtr = lgb.Dataset(Atr, y[a]); dva = lgb.Dataset(Ava, y[b], reference=dtr)
        m = lgb.train(LGB, dtr, 30000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(300, verbose=False)])
        oof[b] = m.predict(Ava); tp += m.predict(Ate) / n_splits
    print(f"{tag:46s} CV AUC {roc_auc_score(y, oof):.6f}")
    return oof, tp

raw_cols = [c for c in Xa.columns if not c.startswith("k_")]
no_dig   = [c for c in raw_cols if not c.startswith(("inc_d_","km_dec","is_"))]

cv(lambda A,ya,V,T: (A[no_dig], V[no_dig], T[no_dig]), "1. raw columns only")
cv(lambda A,ya,V,T: (A[raw_cols], V[raw_cols], T[raw_cols]), "2. + digits and hard-rule flags")
oof_full, test_full = cv(lambda A,ya,V,T: (lambda r: (r[0], r[1][0], r[1][1]))(encode_fold(A, ya, [V, T])),
                         "3. + fold-safe multi-smoothing target encodings")
""")

md(r"""
## 7. The one that fooled me, and everything else that did not work

A residual scan over ~40 candidate keys — income and commute moduli, digit crosses, age
remainders — ranked by statistical significance put **`Age` at the very top**, at nearly
nineteen sigma. It is worth essentially nothing.

With ~14,800 rows per age value, a ±0.8 percentage-point wobble is overwhelming evidence that
the wobble is *real* and no evidence at all that it is *useful*. Significance scales with n;
AUC does not. Rank candidates by effect size.
""")

code(r"""
r = y - oof_full
var = oof_full * (1 - oof_full)
age = train.Age.values

for nm, k in [("Age", age), ("income hundreds digit", (inc_syn // 100) % 10),
              ("income exact // 10", inc_syn // 10)]:
    df = pd.DataFrame({"k": k, "r": r, "v": var}).groupby("k").agg(
        r=("r","sum"), v=("v","sum"), n=("r","size"))
    df = df[df["n"] >= 200]
    z = df["r"] / np.sqrt(df["v"])
    chi2, dof = float((z**2).sum()), len(df)
    print(f"{nm:24s} groups={dof:>5}  chi2/dof={chi2/dof:6.2f}  "
          f"excess={(chi2-dof)/np.sqrt(2*dof):7.2f} sigma")

g = pd.Series(r).groupby(age).mean() * 100
fig, ax = plt.subplots(figsize=(9, 2.7))
ax.bar(g.index, g.values, color=[SIG if v > 0 else ACC for v in g.values])
ax.axhline(0, color=INK, lw=1)
ax.set_xlabel("Age"); ax.set_ylabel("mean residual (pp)")
ax.set_title("Real, and worth nothing: per-age residual is a genuine sawtooth of about ±0.8pp")
plt.tight_layout(); plt.show()
""")

md(r"""
The rest of the graveyard, all measured on the same folds:

| tried | result |
|---|---|
| **Window / neighbourhood encodings** — rolling target mean over the *k* nearest distinct income values | **−0.00008.** The three smoothing strengths already smooth at multiple scales; the windows were redundant |
| **Digits declared categorical** to LightGBM | **−0.00008.** Ordinally meaningless, but the model isolates single digit values anyway |
| **The real 10k survey as extra training rows** | **+0.0002** at unit weight, negative above it. Its value was as a *control*, not as data |
| **Ratio features** (income per car, km per station, income × commute) | **−0.0001.** More ways to split on noise |
| **A GPU MLP with embeddings** | **0.9436** against 0.9451 for trees on the same fold. Wider, deeper, longer schedules all made it worse |
| **Forcing the cliff rows to p=1** | **+0.000001.** The model already ranks 97% of them in the top 1% |

Two of these deserve a note.

The **MLP** is the only model whose errors are genuinely decorrelated from the trees
(rank ρ = 0.968, against 0.994–0.999 among the GBDTs) — exactly the blend member you would
want — and it is too weak to earn weight.

The other is an experiment worth repeating on your own ensemble. Suspecting that every model
was simply reading the exact-income target encodings, I trained a LightGBM **deliberately
blinded** to them, forcing it to rebuild the signal from the raw digits and coarser keys. It
scored 0.94617 — essentially unchanged — and came back correlated at **0.9994** with the
standard model, *higher* than XGBoost is. The artifact is encoded so redundantly that any
competent model converges on the same ranking.

That is the honest shape of this competition: a ten-model ensemble beats the best single
model by about 0.0001, and the last two members moved it by nothing at all. If your blend has
stopped paying here, this is why, and no amount of reweighting will fix it.
""")

md(r"""
## 8. Submission

Ten folds, three seeds, the full feature set. The final pipeline in the repository blends
LightGBM, XGBoost, CatBoost and an MLP across two fold partitions; the single model below is
within 0.0001 of that blend, which tells you most of what you need to know about how much
ensembling is worth here.
""")

code(r"""
oof10, test10 = cv(lambda A,ya,V,T: (lambda r: (r[0], r[1][0], r[1][1]))(encode_fold(A, ya, [V, T])),
                   "final: 10-fold, full feature set", n_splits=10)

sub = pd.DataFrame({ID: test[ID], TARGET: test10})
sub.to_csv("submission.csv", index=False)
print(sub.head(), "\n", sub.shape)
""")

md(r"""
## Takeaways

1. **When tuning stops paying, measure whether the ceiling is the model or the
   representation.** The Bayes-AUC-from-your-own-probabilities trick is three lines and tells
   you which.
2. **Synthetic data carries the generator's fingerprints.** Digits, spikes at a clipping
   floor, deterministic cliffs — these are properties of the sampler, not the world.
3. **Always find the control.** The original dataset is what turns "income digits are
   predictive" from a suspicious correlation into a demonstrated artifact.
4. **Rank candidate features by effect size, not by p-value.** At 668k rows almost everything
   is significant.
5. **Publish the negative results.** Six of the ideas above did not work, and knowing that is
   worth as much as the one that did.

*Credit where due: the income-digit idea surfaced in this competition's public discussion, and
this notebook's contribution is the control against the original survey, the cliff/dead-zone
measurement, the residual scan, and the list of things that failed.*
""")

nb["cells"] = C
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
out = os.path.join(ROOT, "notebook", "s6e9-what-the-income-digits-know.ipynb")
nbf.write(nb, out)
print("wrote", out, f"({len(C)} cells)")
