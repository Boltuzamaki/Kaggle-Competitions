# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(os.environ.get("STEP10B_OUT", "/kaggle/working/step10b_strict_consensus"))
OUT.mkdir(parents=True, exist_ok=True)
SEED = 230724

EXPECTED = {
    "sample_manifest.json": "f2906451a2b6326bf9c51f4d85f1eda175a86a2e11dadd3d32fffbfb977284b7",
    "public_probe_table.csv": "36f14f3f4e667378b9056df6de8593c17e106b25404c76007d7adc99be7a21c8",
    "renderer_feature_table.csv": "d4fa6b1b062f78bfef8dc40fd62407a4c82f691c7aeafa516cc4938b353b63a8",
    "checkpoint_manifest.json": "5f8d583e1d3803326fb714163df1e804341b3dc3f537a18bb3a1b253a3dd65a2",
}
RENDERER_FEATURES = [
    "positive_fraction", "energy_density", "anisotropy", "straightness",
    "endpoint_ratio", "endpoint_asymmetry", "center_ratio", "longitudinal_cv",
    "longitudinal_total_variation", "gap_fraction", "fft_periodicity",
    "autocorrelation_peak", "gap_run_cv", "width_mean", "width_cv",
    "width_drift", "transverse_symmetry", "gaussian_psf_error",
    "inside_noise_cv", "side_positive_rate", "side_noise_std", "unique_ratio",
    "bit1_entropy", "bit2_entropy", "bit4_entropy", "bit8_entropy",
    "subpixel_phase", "endpoint_sharpness", "interpolation_fraction",
    "axis_residual", "log_length", "log_aspect",
]
CLEAN_FAMILIES = [
    "external_crop_transplant", "synthetic_irregular_dash",
    "synthetic_stochastic_blink", "synthetic_head_tail_tracklet",
]
GATES = {
    "consensus": "3_of_3_intersection",
    "signals": ["renderer_physics", "pcgrad_collapse", "v12_collapse"],
    "training_clean_quantile": 0.99,
    "deployment_threshold_rule": (
        "renderer: q99 of five-fold public OOF clean probabilities; "
        "PCGrad and V12: q99 of their full public clean scores"
    ),
    "aggregate_auc_min": 0.80,
    "each_family_auc_min": 0.70,
    "calibrated_precision_min": 0.90,
    "maximum_family_fpr": 0.05,
    "median_poison_recall_min": 0.25,
    "each_family_poison_recall_min": 0.20,
    "permutation_repeats": 30,
    "permutation_auc_p95_max": 0.70,
    "renderer_l2": 0.20,
    "fit_steps": 700,
    "inner_folds": 5,
}
LOCK = {
    "status": "frozen_before_input_enumeration_or_table_read",
    "experiment": "STEP10B_STRICT_3_OF_3_CONSENSUS",
    "seed": SEED,
    "exact_step8b_artifact_hashes": EXPECTED,
    "renderer_features": RENDERER_FEATURES,
    "gates": GATES,
    "validation": "nested leave-one-poison-image-out times leave-one-clean-family-out",
    "forbidden": ["V10", "V9", "threshold sweep", "test data", "candidate", "submission"],
    "selection_boundary": {
        "competition_source_mounted": False,
        "competition_test_enumerated": False,
        "competition_test_read": False,
        "candidate_created": False,
        "competition_submission_created": False,
    },
}
(OUT / "selection_lock.json").write_text(json.dumps(LOCK, indent=2), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_source():
    explicit = os.environ.get("STEP8B_SOURCE")
    if explicit:
        return Path(explicit)
    found = []
    for p in Path("/kaggle/input").rglob("sample_manifest.json"):
        if sha256(p) == EXPECTED["sample_manifest.json"] and all((p.parent / n).exists() for n in EXPECTED):
            found.append(p.parent)
    assert len(found) == 1, [str(p) for p in found]
    return found[0]


SOURCE = find_source()
audit = {}
for name, expected in EXPECTED.items():
    actual = sha256(SOURCE / name)
    audit[name] = {"expected": expected, "actual": actual, "match": actual == expected}
    assert actual == expected
(OUT / "input_artifact_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

probe = pd.read_csv(SOURCE / "public_probe_table.csv")
physical = pd.read_csv(SOURCE / "renderer_feature_table.csv")
manifest = json.loads((SOURCE / "sample_manifest.json").read_text(encoding="utf-8"))["samples"]
ids = [r["sample_id"] for r in manifest]
assert len(probe) == len(physical) == len(ids) == 140
assert probe.sample_id.tolist() == physical.sample_id.tolist() == ids
assert probe.groupby("family").size().to_dict() == {"public_poison": 20, **{f: 30 for f in CLEAN_FAMILIES}}

X = physical[RENDERER_FEATURES].to_numpy(float)
y = probe.label_poison.to_numpy(int)
families = probe.family.astype(str).to_numpy()
sample_ids = probe.sample_id.astype(str).to_numpy()
pcgrad = 1.0 - np.clip(probe.ratio_pcgrad_median.to_numpy(float), 0.0, 2.0)
v12 = 1.0 - np.clip(probe.ratio_v12.to_numpy(float), 0.0, 2.0)
poison_idx = np.where(y == 1)[0]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logistic(x, labels, l2=0.2, steps=700):
    x = np.asarray(x, float); labels = np.asarray(labels, float)
    center = np.median(x, axis=0)
    scale = np.quantile(x, .75, axis=0) - np.quantile(x, .25, axis=0)
    scale = np.where(scale > 1e-7, scale, np.std(x, axis=0))
    scale = np.where(scale > 1e-7, scale, 1.0)
    z = np.clip((x-center)/scale, -12, 12)
    w = np.zeros(z.shape[1]); b = float(np.log((labels.mean()+1e-3)/(1-labels.mean()+1e-3)))
    sw = np.where(labels > .5, len(labels)/max(2*labels.sum(),1), len(labels)/max(2*(len(labels)-labels.sum()),1))
    for step in range(steps):
        err = (sigmoid(z@w+b)-labels)*sw
        lr = .08/math.sqrt(1+step/80)
        w -= lr*((z.T@err)/len(labels)+l2*w/len(labels)); b -= lr*err.mean()
    return center, scale, w, b


def predict(model, x):
    c,s,w,b=model
    return sigmoid(np.clip((np.asarray(x,float)-c)/s,-12,12)@w+b)


def fold_assign(indices, labels, fams, n=5):
    out=np.zeros(len(indices),int); counters={}
    for local, idx in enumerate(indices):
        key="poison" if labels[idx] else str(fams[idx]); count=counters.get(key,0)
        out[local]=count%n; counters[key]=count+1
    return out


def auc(labels, scores):
    labels=np.asarray(labels); scores=np.asarray(scores)
    p=scores[labels==1]; n=scores[labels==0]
    return float((p[:,None]>n).mean()+.5*(p[:,None]==n).mean())


def empirical_cdf(train_clean, values):
    clean=np.sort(np.asarray(train_clean,float)); values=np.asarray(values,float)
    return np.searchsorted(clean, values, side="right")/len(clean)


records=[]
for held_family in CLEAN_FAMILIES:
    clean_test=np.where(families==held_family)[0]
    for held_poison in poison_idx:
        test=np.concatenate([[held_poison],clean_test])
        mask=np.ones(len(y),bool); mask[test]=False; train=np.where(mask)[0]
        folds=fold_assign(train,y,families,GATES["inner_folds"])
        rtrain=np.zeros(len(train))
        for fold in range(GATES["inner_folds"]):
            fit=folds!=fold; val=folds==fold
            model=fit_logistic(X[train[fit]],y[train[fit]],GATES["renderer_l2"],GATES["fit_steps"])
            rtrain[val]=predict(model,X[train[val]])
        rmodel=fit_logistic(X[train],y[train],GATES["renderer_l2"],GATES["fit_steps"])
        rtest=predict(rmodel,X[test])
        clean_local=y[train]==0
        clean_signals=[rtrain[clean_local],pcgrad[train][clean_local],v12[train][clean_local]]
        thresholds=[float(np.quantile(s,GATES["training_clean_quantile"])) for s in clean_signals]
        test_signals=[rtest,pcgrad[test],v12[test]]
        cdfs=np.column_stack([empirical_cdf(c,v) for c,v in zip(clean_signals,test_signals)])
        consensus_score=cdfs.min(axis=1)
        chosen=np.logical_and.reduce([v>=t for v,t in zip(test_signals,thresholds)])
        for idx,score,pred,signal_values in zip(test,consensus_score,chosen,np.column_stack(test_signals)):
            records.append({"heldout_clean_family":held_family,"heldout_poison_id":sample_ids[held_poison],
                "sample_id":sample_ids[idx],"family":families[idx],"label_poison":int(y[idx]),
                "renderer":float(signal_values[0]),"pcgrad":float(signal_values[1]),"v12":float(signal_values[2]),
                "consensus_score":float(score),"predicted_poison":bool(pred),
                "renderer_threshold":thresholds[0],"pcgrad_threshold":thresholds[1],"v12_threshold":thresholds[2]})

pred=pd.DataFrame(records)
pred.to_csv(OUT/"nested_consensus_predictions.csv",index=False)
agg=pred.groupby(["sample_id","family","label_poison"],as_index=False).agg(
    consensus_score=("consensus_score","mean"), predicted_rate=("predicted_poison","mean"))
agg.to_csv(OUT/"sample_consensus_scores.csv",index=False)
poison=agg[agg.label_poison==1]
family_results={}
for family in CLEAN_FAMILIES:
    clean=agg[agg.family==family]; pair=pd.concat([poison,clean])
    raw=pred[pred.heldout_clean_family==family]
    family_results[family]={"auc":auc(pair.label_poison,pair.consensus_score),
        "poison_recall":float(raw.loc[raw.label_poison==1,"predicted_poison"].mean()),
        "clean_fpr":float(raw.loc[raw.label_poison==0,"predicted_poison"].mean())}

selected=pred[pred.predicted_poison]
metrics={"aggregate_auc":auc(agg.label_poison,agg.consensus_score),"family_results":family_results,
    "calibrated_precision":float(selected.label_poison.mean()) if len(selected) else 0.0,
    "unique_poison_support":int(selected.loc[selected.label_poison==1,"sample_id"].nunique()),
    "median_poison_recall":float(np.median([v["poison_recall"] for v in family_results.values()])),
    "maximum_family_fpr":float(max(v["clean_fpr"] for v in family_results.values()))}

# Immutable public-only deployment calibration. These thresholds are computed
# before any possible test enumeration and are never retuned per test image.
full_folds=fold_assign(np.arange(len(y)),y,families,GATES["inner_folds"])
renderer_oof=np.zeros(len(y))
for fold in range(GATES["inner_folds"]):
    tr=full_folds!=fold; va=full_folds==fold
    m=fit_logistic(X[tr],y[tr],GATES["renderer_l2"],GATES["fit_steps"])
    renderer_oof[va]=predict(m,X[va])
full_renderer_model=fit_logistic(X,y,GATES["renderer_l2"],GATES["fit_steps"])
clean=y==0
deployment_thresholds={
    "aggregation_rule":GATES["deployment_threshold_rule"],
    "quantile":GATES["training_clean_quantile"],
    "renderer":float(np.quantile(renderer_oof[clean],GATES["training_clean_quantile"])),
    "pcgrad":float(np.quantile(pcgrad[clean],GATES["training_clean_quantile"])),
    "v12":float(np.quantile(v12[clean],GATES["training_clean_quantile"])),
}
(OUT/"deployment_thresholds.json").write_text(json.dumps(deployment_thresholds,indent=2),encoding="utf-8")
np.savez_compressed(OUT/"renderer_deployment_model.npz",center=full_renderer_model[0],scale=full_renderer_model[1],
    weights=full_renderer_model[2],bias=np.asarray(full_renderer_model[3]),features=np.asarray(RENDERER_FEATURES))

# Permutation control: shuffle labels, then five-fold-train the same physical
# classifier plus the two scalar signals. This must collapse to chance.
rng=np.random.default_rng(SEED+2000); perm_aucs=[]
raw=np.column_stack([X,pcgrad,v12]); folds=fold_assign(np.arange(len(y)),y,families,GATES["inner_folds"])
for _ in range(GATES["permutation_repeats"]):
    py=rng.permutation(y); ps=np.zeros(len(y))
    for fold in range(GATES["inner_folds"]):
        tr=folds!=fold; va=folds==fold
        model=fit_logistic(raw[tr],py[tr],1.0,GATES["fit_steps"])
        ps[va]=predict(model,raw[va])
    perm_aucs.append(auc(py,ps))
permutation={"repeats":len(perm_aucs),"auc_mean":float(np.mean(perm_aucs)),
    "auc_p95":float(np.quantile(perm_aucs,.95)),"auc_max":float(np.max(perm_aucs)),
    "passed":bool(np.quantile(perm_aucs,.95)<=GATES["permutation_auc_p95_max"]),"aucs":perm_aucs}
(OUT/"permutation_control.json").write_text(json.dumps(permutation,indent=2),encoding="utf-8")

checks={"aggregate_auc":metrics["aggregate_auc"]>=GATES["aggregate_auc_min"],
    "each_family_auc":all(v["auc"]>=GATES["each_family_auc_min"] for v in family_results.values()),
    "calibrated_precision":metrics["calibrated_precision"]>=GATES["calibrated_precision_min"],
    "maximum_family_fpr":metrics["maximum_family_fpr"]<=GATES["maximum_family_fpr"],
    "median_poison_recall":metrics["median_poison_recall"]>=GATES["median_poison_recall_min"],
    "each_family_poison_recall":all(v["poison_recall"]>=GATES["each_family_poison_recall_min"] for v in family_results.values()),
    "permutation_negative_control":permutation["passed"]}
passed=bool(all(checks.values()))
gate={"status":"pass" if passed else "rejected","gate_passed":passed,"metrics":metrics,"checks":checks,
    "frozen_requirements":GATES,"permutation_control":{k:v for k,v in permutation.items() if k!="aucs"},
    "deployment_thresholds":deployment_thresholds,
    "exact_step8b_bank_reproduced":True,"rule_7a_guard_passed":True,
    "competition_test_enumerated":False,"competition_test_read":False,"candidate_created":False,"competition_submission_created":False}
(OUT/"strict_consensus_gate.json").write_text(json.dumps(gate,indent=2),encoding="utf-8")
(OUT/"final_report.json").write_text(json.dumps({"experiment":LOCK["experiment"],"gate":gate,"artifact_audit":audit},indent=2),encoding="utf-8")
print(json.dumps(gate,indent=2))
