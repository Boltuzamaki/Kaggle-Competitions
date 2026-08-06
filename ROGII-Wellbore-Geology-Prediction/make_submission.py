"""Train on all train wells, predict the test wells, write submission.csv."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from wellbore_lib import (build_dataset, make_features, load_well, list_wells,
                          FEATURES, LGB_PARAMS, postprocess)

DATA = "data"

print("Building train dataset...")
train = build_dataset(f"{DATA}/train", has_target=True)
print("train rows", len(train))

model = lgb.LGBMRegressor(**LGB_PARAMS)
model.fit(train[FEATURES], train["d_tvt"])
print("model trained")

# sample_submission defines exactly which ids must be predicted
sample = pd.read_csv(f"{DATA}/sample_submission.csv")
test_wells = list_wells(f"{DATA}/test")
print("test wells", test_wells)

preds = []
for w in test_wells:
    h, tw = load_well(f"{DATA}/test", w)
    f = make_features(h, tw, w, has_target=False)        # uses TVT_input as anchor
    d = postprocess(model.predict(f[FEATURES]))
    f = f.assign(tvt=f["tvt_ps"] + d)
    preds.append(f[["id", "tvt"]])

pred_df = pd.concat(preds, ignore_index=True)
sub = sample[["id"]].merge(pred_df, on="id", how="left")
# any id we somehow miss -> fall back to its well's PS TVT (constant baseline)
missing = sub["tvt"].isna().sum()
sub["tvt"] = sub["tvt"].ffill().fillna(0.0)
sub.to_csv("submission.csv", index=False)
print(f"wrote submission.csv rows={len(sub)} missing_filled={missing}")
print(sub.head())
print(sub["tvt"].describe())
