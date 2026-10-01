"""Compute every number the HTML report quotes, into reports/report_data.json."""
import json, os
import numpy as np, pandas as pd
import common as C

ROOT = C.ROOT
tr = pd.read_csv(os.path.join(C.DATA, "train.csv"))
te = pd.read_csv(os.path.join(C.DATA, "test.csv"))
y = (tr[C.TARGET] == "Yes").astype(int).values
orig = pd.read_csv(os.path.join(C.DATA, "original", "EV_Adoption_and_Range_Anxiety_Dataset.csv"))
yo = (orig.Will_Buy_EV == "Yes").astype(int).values

D = {"n_train": len(tr), "n_test": len(te), "pos_rate": float(y.mean()),
     "n_pos": int(y.sum()), "n_feat": 13}

# ---- schema table -------------------------------------------------------
schema = []
DESC = {
 "Age": ("integer, 25-69", "Respondent age in years."),
 "Annual_Income_USD": ("integer USD, floored at 30 000", "Household income. Carries the generator artifact."),
 "Daily_Commute_km": ("float, 1 decimal, floored at 5.0", "One-way daily commute distance."),
 "Number_of_Cars_Owned": ("integer, 1-4", "Cars currently in the household."),
 "Charging_Stations_Near_Home": ("integer, 0-14", "Public chargers near home."),
 "Charging_Stations_Near_Work": ("integer, 0-19", "Public chargers near work."),
 "Environmental_Concern_Level": ("integer, 1-5", "Self-reported concern. Strongest single driver."),
 "Gender": ("Male / Female / Other", "No measurable effect on the target."),
 "City_Type": ("Urban / Suburban / Rural", "Mild effect; rural buys slightly more."),
 "Current_Car_Type": ("Hatchback / Sedan / SUV / Truck", "Mild; truck owners buy least."),
 "Home_Charging_Possible": ("Yes / No", "Can charge at home."),
 "Subsidy_Available": ("Yes / No", "Purchase subsidy. A hard gate."),
 "Range_Anxiety_Level": ("Low / Medium / High", "Worry about range. A hard gate."),
}
for c in [x for x in tr.columns if x not in ("id", C.TARGET)]:
    s = tr[c]
    num = pd.api.types.is_numeric_dtype(s)
    schema.append({
        "name": c, "kind": "numeric" if num else "categorical",
        "domain": DESC[c][0], "desc": DESC[c][1],
        "n_unique": int(s.nunique()), "missing": int(s.isna().sum()),
        "corr": (round(float(np.corrcoef(s, y)[0, 1]), 3) if num else None),
    })
D["schema"] = schema

# ---- categorical effects ------------------------------------------------
D["cat_effects"] = {}
for c in C.CAT_COLS:
    g = pd.Series(y).groupby(tr[c].values).agg(["mean", "size"])
    D["cat_effects"][c] = [{"level": str(k), "rate": float(v["mean"]), "n": int(v["size"])}
                           for k, v in g.iterrows()]

# ---- the two gates ------------------------------------------------------
ct = pd.crosstab(tr.Subsidy_Available, tr.Range_Anxiety_Level, values=y, aggfunc="mean")
cn = pd.crosstab(tr.Subsidy_Available, tr.Range_Anxiety_Level)
D["gate_grid"] = {"rows": ct.index.tolist(), "cols": ["Low", "Medium", "High"],
                  "rate": [[float(ct.loc[r, c]) for c in ["Low", "Medium", "High"]] for r in ct.index],
                  "n": [[int(cn.loc[r, c]) for c in ["Low", "Medium", "High"]] for r in ct.index]}

# ---- numeric effect curves ---------------------------------------------
D["env_curve"] = [{"x": int(k), "rate": float(v)} for k, v in
                  pd.Series(y).groupby(tr.Environmental_Concern_Level.values).mean().items()]
inc_d = pd.qcut(tr.Annual_Income_USD, 10, labels=False, duplicates="drop")
g = pd.Series(y).groupby(inc_d).mean()
mid = tr.groupby(inc_d).Annual_Income_USD.median()
D["income_curve"] = [{"x": float(mid[i]), "rate": float(g[i])} for i in g.index]
age_g = pd.Series(y).groupby(tr.Age.values).mean()
D["age_curve"] = [{"x": int(k), "rate": float(v)} for k, v in age_g.items()]

# ---- THE artifact: income digits ---------------------------------------
def digit_table(vals, yy):
    out = {}
    for nm, k in [("units", 1), ("tens", 10), ("hundreds", 100), ("thousands", 1000)]:
        d = (vals // k) % 10
        g = pd.Series(yy).groupby(d).agg(["mean", "size"])
        p, nbar = yy.mean(), g["size"].mean()
        out[nm] = {"rate": [float(v) for v in g["mean"]],
                   "n": [int(v) for v in g["size"]],
                   "span_pp": float((g["mean"].max() - g["mean"].min()) * 100),
                   "corridor_pp": float(3.1 * np.sqrt(p * (1 - p) / nbar) * 100)}
    return out

inc_s = tr.Annual_Income_USD.values.astype(np.int64)
io = orig.Annual_Income_USD.dropna()
D["digits_synth"] = digit_table(inc_s, y)
D["digits_orig"] = digit_table(io.values.astype(np.int64), yo[io.index])
D["orig_rows"] = len(orig)

# ---- train/test identity ------------------------------------------------
D["shift"] = []
for c in C.NUM_COLS:
    a, b = tr[c], te[c]
    D["shift"].append({"col": c, "train_mean": float(a.mean()), "test_mean": float(b.mean()),
                       "std_diff": float(abs(a.mean() - b.mean()) / a.std())})
D["dup_rows"] = int(tr[[c for c in tr.columns if c not in ("id", C.TARGET)]].duplicated().sum())
D["id_auc"] = float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, tr.id.values))
D["income_floor_share"] = float((tr.Annual_Income_USD == 30000).mean())
D["commute_floor_share"] = float((tr.Daily_Commute_km == 5.0).mean())

# ---- experiment log -----------------------------------------------------
exps = [json.loads(l) for l in open(os.path.join(ROOT, "artifacts", "experiments.jsonl"))]
lbp = os.path.join(ROOT, "artifacts", "lb_scores.json")
D["lb"] = json.load(open(lbp)) if os.path.exists(lbp) else {}
D["experiments"] = exps
bl = os.path.join(ROOT, "artifacts", "blend.json")
D["blend"] = json.load(open(bl)) if os.path.exists(bl) else None

json.dump(D, open(os.path.join(ROOT, "reports", "report_data.json"), "w"), indent=1)
print("wrote reports/report_data.json;", len(D["experiments"]), "experiments")
