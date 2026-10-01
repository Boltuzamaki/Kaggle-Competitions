"""Residual artifact scan: what signal is the current model still missing?

For each candidate integer key, test whether the mean OOF residual (y - p) varies
across key values by more than binomial noise. A large chi-square per degree of
freedom means untapped structure.
"""
import os, sys
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)
os.chdir(_SRC)   # so the ../data and ../artifacts paths below resolve
import numpy as np, pandas as pd
import common as C

X, Xt, y, tid = C.build(level="raw")
p = np.load("../artifacts/oof/exp02_lgbm_digits.npy")
r = y - p
var = p * (1 - p)     # per-row Bernoulli variance

inc = X.Annual_Income_USD.values.astype(np.int64)
km10 = np.rint(X.Daily_Commute_km.values * 10).astype(np.int64)
age = X.Age.values.astype(np.int64)

CAND = {
    "inc_units": inc % 10, "inc_tens": (inc//10) % 10, "inc_hund": (inc//100) % 10,
    "inc_thou": (inc//1000) % 10, "inc_tenk": (inc//10000) % 10,
    "inc_mod100": inc % 100, "inc_mod1000": inc % 1000,
    "inc_mod7": inc % 7, "inc_mod9": inc % 9, "inc_mod11": inc % 11,
    "inc_mod3": inc % 3, "inc_mod2": inc % 2, "inc_mod5": inc % 5,
    "inc_digitsum": sum((inc//k) % 10 for k in (1,10,100,1000,10000)),
    "inc_exact_bucket": inc // 10,
    "km_dec": km10 % 10, "km_units": (km10//10) % 10, "km_tens": (km10//100) % 10,
    "km_mod100": km10 % 100, "km_mod7": km10 % 7, "km_mod3": km10 % 3,
    "km_digitsum": sum((km10//k) % 10 for k in (1,10,100)),
    "age": age, "age_mod10": age % 10, "age_mod7": age % 7, "age_mod3": age % 3,
    "cars": X.Number_of_Cars_Owned.values, "env": X.Environmental_Concern_Level.values,
    "st_home": X.Charging_Stations_Near_Home.values, "st_work": X.Charging_Stations_Near_Work.values,
    "st_sum": (X.Charging_Stations_Near_Home + X.Charging_Stations_Near_Work).values,
    "gender": X.Gender.values, "city": X.City_Type.values, "cartype": X.Current_Car_Type.values,
    "inc_units_x_tens": (inc % 10) * 10 + (inc//10) % 10,
    "inc_units_x_hund": (inc % 10) * 10 + (inc//100) % 10,
    "inc_dec_x_km_dec": (inc % 10) * 10 + (km10 % 10),
    "inc_hund_x_env": ((inc//100) % 10) * 10 + X.Environmental_Concern_Level.values.astype(np.int64),
    "inc_units_x_age": (inc % 10) * 100 + age,
}

rows = []
for nm, k in CAND.items():
    k = np.asarray(k)
    df = pd.DataFrame({"k": k, "r": r, "v": var}).groupby("k").agg(
        r=("r", "sum"), v=("v", "sum"), n=("r", "size"))
    df = df[df["n"] >= 200]
    if len(df) < 2:
        continue
    z = df["r"] / np.sqrt(df["v"])          # standardised residual per group
    chi2 = float((z ** 2).sum()); dof = len(df)
    rows.append({"key": nm, "groups": dof, "chi2_per_dof": chi2 / dof,
                 "max_abs_z": float(z.abs().max()),
                 "excess_sigma": (chi2 - dof) / np.sqrt(2 * dof)})

res = pd.DataFrame(rows).sort_values("excess_sigma", ascending=False)
pd.set_option("display.width", 160)
print("Residual structure left in the exp02 model (excess_sigma > ~3 = real, untapped):")
print(res.head(25).to_string(index=False, float_format=lambda v: f"{v:9.2f}"))
