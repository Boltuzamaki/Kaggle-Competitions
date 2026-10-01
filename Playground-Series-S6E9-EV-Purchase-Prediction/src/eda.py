"""EDA for Playground Series S6E9 - Predicting Electric Vehicle Purchases.

Writes: reports/eda_stats.json  and  reports/figures/*.png
"""
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIG, exist_ok=True)

TARGET = "Will_Buy_EV"
ID = "id"

sns.set_theme(style="whitegrid", palette="deep")
PRIMARY, ACCENT = "#2b6cb0", "#dd6b20"

train = pd.read_csv(os.path.join(DATA, "train.csv"))
test = pd.read_csv(os.path.join(DATA, "test.csv"))
sub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))

S = {}
S["shapes"] = {"train": list(train.shape), "test": list(test.shape), "submission": list(sub.shape)}
S["sample_submission_head"] = sub.head(3).to_dict("records")
S["target_name"] = TARGET
S["target_raw_values"] = {str(k): int(v) for k, v in train[TARGET].value_counts().items()}
S["positive_rate"] = float((train[TARGET] == "Yes").mean())

feats = [c for c in train.columns if c not in (ID, TARGET)]
num_cols = [c for c in feats if pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in feats if c not in num_cols]
S["features"] = feats
S["numeric_features"] = num_cols
S["categorical_features"] = cat_cols

# ---------- column-level profile ----------
prof = []
y = (train[TARGET] == "Yes").astype(int)
for c in feats:
    tr, te = train[c], test[c]
    row = {
        "column": c,
        "dtype": str(tr.dtype),
        "kind": "numeric" if c in num_cols else "categorical",
        "n_unique_train": int(tr.nunique(dropna=True)),
        "n_unique_test": int(te.nunique(dropna=True)),
        "missing_train": int(tr.isna().sum()),
        "missing_train_pct": round(float(tr.isna().mean() * 100), 3),
        "missing_test": int(te.isna().sum()),
        "missing_test_pct": round(float(te.isna().mean() * 100), 3),
    }
    if c in num_cols:
        d = tr.dropna()
        row.update({
            "min": float(d.min()), "p01": float(d.quantile(0.01)), "p25": float(d.quantile(0.25)),
            "median": float(d.median()), "mean": float(d.mean()), "p75": float(d.quantile(0.75)),
            "p99": float(d.quantile(0.99)), "max": float(d.max()), "std": float(d.std()),
            "skew": float(d.skew()),
            "test_min": float(te.dropna().min()), "test_max": float(te.dropna().max()),
        })
        # univariate signal: point-biserial style correlation with target
        m = tr.notna()
        row["corr_with_target"] = float(np.corrcoef(tr[m], y[m])[0, 1])
    else:
        vc = tr.value_counts(dropna=False)
        row["categories"] = [str(x) for x in vc.index.tolist()]
        row["category_counts"] = {str(k): int(v) for k, v in vc.items()}
        row["category_pos_rate"] = {
            str(k): round(float(v), 4)
            for k, v in y.groupby(tr.fillna("__NA__")).mean().items()
        }
    prof.append(row)
S["profile"] = prof

# ---------- duplicates / leakage checks ----------
S["duplicate_rows_train_excl_id"] = int(train.drop(columns=[ID]).duplicated().sum())
S["duplicate_feature_rows_train"] = int(train[feats].duplicated().sum())
S["id_overlap_train_test"] = int(len(set(train[ID]) & set(test[ID])))
S["id_range_train"] = [int(train[ID].min()), int(train[ID].max())]
S["id_range_test"] = [int(test[ID].min()), int(test[ID].max())]

# contradictory duplicates: same features, different label
dup = train[train[feats].duplicated(keep=False)]
if len(dup):
    g = dup.groupby(feats, dropna=False)[TARGET].nunique()
    S["contradictory_duplicate_groups"] = int((g > 1).sum())
    S["duplicated_feature_groups"] = int(len(g))
else:
    S["contradictory_duplicate_groups"] = 0
    S["duplicated_feature_groups"] = 0

# ---------- train/test distribution shift (adversarial-lite) ----------
shift = []
for c in num_cols:
    a, b = train[c].dropna(), test[c].dropna()
    shift.append({"column": c, "train_mean": float(a.mean()), "test_mean": float(b.mean()),
                  "abs_std_diff": float(abs(a.mean() - b.mean()) / (a.std() + 1e-9))})
for c in cat_cols:
    a = train[c].value_counts(normalize=True, dropna=False)
    b = test[c].value_counts(normalize=True, dropna=False)
    idx = a.index.union(b.index)
    tvd = float(0.5 * (a.reindex(idx, fill_value=0) - b.reindex(idx, fill_value=0)).abs().sum())
    shift.append({"column": c, "total_variation_distance": tvd})
S["train_test_shift"] = shift

# ---------- numeric correlation ----------
corr = train[num_cols + []].assign(**{TARGET: y}).corr(numeric_only=True)
S["correlation_matrix"] = corr.round(4).to_dict()

# ================= FIGURES =================
def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=110, bbox_inches="tight")
    plt.close(fig)

# 1. target balance
fig, ax = plt.subplots(figsize=(4.6, 3.4))
vc = train[TARGET].value_counts()
ax.bar(vc.index, vc.values, color=[PRIMARY, ACCENT], width=0.55)
for i, v in enumerate(vc.values):
    ax.text(i, v, f"{v:,}\n({v/len(train):.1%})", ha="center", va="bottom", fontsize=9)
ax.set_title("Target balance: Will_Buy_EV"); ax.set_ylabel("rows")
ax.set_ylim(0, vc.max() * 1.18)
save(fig, "01_target_balance.png")

# 2. numeric distributions train vs test
n = len(num_cols)
fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(14, 3.1 * ((n + 2) // 3)))
for ax, c in zip(axes.ravel(), num_cols):
    bins = min(50, max(10, train[c].nunique()))
    ax.hist(train[c].dropna(), bins=bins, density=True, alpha=0.55, label="train", color=PRIMARY)
    ax.hist(test[c].dropna(), bins=bins, density=True, alpha=0.55, label="test", color=ACCENT)
    ax.set_title(c, fontsize=10); ax.legend(fontsize=7)
for ax in axes.ravel()[n:]:
    ax.axis("off")
fig.suptitle("Numeric features: train vs test distributions", y=1.005)
save(fig, "02_numeric_train_vs_test.png")

# 3. numeric feature vs target (positive rate by bin)
fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(14, 3.1 * ((n + 2) // 3)))
for ax, c in zip(axes.ravel(), num_cols):
    if train[c].nunique() <= 15:
        gr = y.groupby(train[c]).agg(["mean", "size"])
        ax.plot(gr.index, gr["mean"], "o-", color=PRIMARY)
    else:
        b = pd.qcut(train[c], 20, duplicates="drop")
        gr = y.groupby(b, observed=True).mean()
        ax.plot(range(len(gr)), gr.values, "o-", color=PRIMARY)
        ax.set_xlabel("ventile")
    ax.axhline(y.mean(), ls="--", c="grey", lw=1)
    ax.set_title(f"{c} -> P(buy)", fontsize=10)
for ax in axes.ravel()[n:]:
    ax.axis("off")
fig.suptitle("Numeric features vs P(Will_Buy_EV = Yes)", y=1.005)
save(fig, "03_numeric_vs_target.png")

# 4. categorical positive rate
m = len(cat_cols)
fig, axes = plt.subplots((m + 2) // 3, 3, figsize=(14, 3.4 * ((m + 2) // 3)))
axes = np.atleast_1d(axes).ravel()
for ax, c in zip(axes, cat_cols):
    gr = y.groupby(train[c].fillna("NA")).agg(["mean", "size"]).sort_values("mean")
    ax.barh(gr.index.astype(str), gr["mean"], color=PRIMARY)
    ax.axvline(y.mean(), ls="--", c=ACCENT, lw=1.4)
    ax.set_title(f"{c}", fontsize=10); ax.set_xlabel("P(buy)")
for ax in axes[m:]:
    ax.axis("off")
fig.suptitle("Categorical features vs P(Will_Buy_EV = Yes)  (dashed = base rate)", y=1.005)
save(fig, "04_categorical_vs_target.png")

# 5. correlation heatmap
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
            annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
ax.set_title("Numeric feature correlations (incl. target)")
save(fig, "05_correlation.png")

# 6. missingness
miss = pd.DataFrame({
    "train": train[feats].isna().mean() * 100,
    "test": test[feats].isna().mean() * 100,
}).sort_values("train", ascending=False)
fig, ax = plt.subplots(figsize=(8, 4.5))
miss.plot.barh(ax=ax, color=[PRIMARY, ACCENT])
ax.set_xlabel("% missing"); ax.set_title("Missing values by column")
save(fig, "06_missingness.png")

with open(os.path.join(ROOT, "reports", "eda_stats.json"), "w") as f:
    json.dump(S, f, indent=2, default=str)

print("train", train.shape, "test", test.shape)
print("pos rate", S["positive_rate"])
print("numeric:", num_cols)
print("categorical:", cat_cols)
print("dup feature rows:", S["duplicate_feature_rows_train"],
      "contradictory groups:", S["contradictory_duplicate_groups"])
print("figures ->", FIG)
