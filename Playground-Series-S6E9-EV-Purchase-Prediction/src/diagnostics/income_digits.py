"""Verify: does the generator leak target signal into income's low-order digits?"""
import os, sys
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)
os.chdir(_SRC)   # so the ../data and ../artifacts paths below resolve
import numpy as np, pandas as pd
import common as C

tr = pd.read_csv("../data/train.csv"); te = pd.read_csv("../data/test.csv")
y = (tr.Will_Buy_EV == "Yes").astype(int).values
orig = pd.read_csv("../data/original/EV_Adoption_and_Range_Anxiety_Dataset.csv")
yo = (orig.Will_Buy_EV == "Yes").astype(int).values

def digit_report(vals, yy, label, col):
    print(f"\n--- {label} : {col} ---")
    for name, d in [("units", vals % 10), ("tens", (vals // 10) % 10),
                    ("hundreds", (vals // 100) % 10), ("thousands", (vals // 1000) % 10)]:
        g = pd.Series(yy).groupby(pd.Series(d).values).agg(["mean", "size"])
        span = (g["mean"].max() - g["mean"].min()) * 100
        # binomial corridor: expected max-min spread of 10 groups under the null
        p, nbar = yy.mean(), g["size"].mean()
        corridor = 3.1 * np.sqrt(p * (1 - p) / nbar) * 100   # ~ range of 10 normals
        flag = "  <== SIGNAL" if span > 2.5 * corridor else ""
        print(f"  {name:9s} span={span:6.2f}pp  null_corridor~{corridor:5.2f}pp  "
              f"ratio={span/corridor:5.1f}x{flag}")
        if flag:
            print("      rate by digit:", np.round(g["mean"].values * 100, 2).tolist())

inc = tr.Annual_Income_USD.values.astype(np.int64)
digit_report(inc, y, "SYNTHETIC train (668k)", "Annual_Income_USD")

# control: same test on the real survey, matched size
io = orig.Annual_Income_USD.dropna()
digit_report(io.values.astype(np.int64), yo[io.index], "ORIGINAL survey (10k)", "Annual_Income_USD")

# does the same happen for commute (1 decimal -> x10)?
km = np.round(tr.Daily_Commute_km.values * 10).astype(np.int64)
digit_report(km, y, "SYNTHETIC train", "Daily_Commute_km x10")

# is the digit distribution itself odd?
print("\nincome units-digit distribution (train):")
print(pd.Series(inc % 10).value_counts(normalize=True).sort_index().round(4).to_dict())
print("income last-3-digits ==0 share:", round(float((inc % 1000 == 0).mean()), 4))
print("test units-digit distribution matches?",
      round(float(np.abs(pd.Series(inc % 10).value_counts(normalize=True).sort_index().values -
      pd.Series(te.Annual_Income_USD.values.astype(np.int64) % 10).value_counts(normalize=True).sort_index().values).sum()), 5))
