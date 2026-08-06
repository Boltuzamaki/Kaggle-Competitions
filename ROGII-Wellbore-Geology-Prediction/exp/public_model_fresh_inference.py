"""Strict fresh inference from the Pilkwang public model package.

This script deliberately refuses to read CSV/NPY prediction artifacts. It
builds features from official raw train/test files and loads model weights,
feature schema, and blend configuration only.
"""
from pathlib import Path
import argparse
import importlib.util
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_builder(package):
    p = package/"feature_builders/build_features.py"
    spec = importlib.util.spec_from_file_location("fresh_public_builder", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_features


def tree_predictions(frame, package, manifest):
    import joblib
    from catboost import CatBoostRegressor
    import lightgbm as lgb
    import xgboost as xgb
    schema = json.loads((package/"feature_builders/feature_columns.json").read_text())
    out = {}
    for entry in manifest["models"]:
        family = entry["model_family"]
        if family == "sequence_tcn":
            continue
        cols = schema[entry["feature_set"]]
        X = frame[cols].to_numpy(np.float32)
        path = package/entry["path"]
        if family == "xgb":
            model = xgb.XGBRegressor()
            model.load_model(path)
            pred = model.predict(X)
        elif family == "catboost":
            model = CatBoostRegressor()
            model.load_model(path)
            pred = model.predict(X)
        elif family == "hgb":
            # Package was serialized with an older sklearn that exposed the
            # loss extension as top-level ``_loss``.
            import sys
            import sklearn._loss.loss as sklearn_loss
            sys.modules.setdefault("_loss", sklearn_loss)
            model = joblib.load(path)
            pred = model.predict(np.nan_to_num(X))
        elif family == "lgb":
            model = lgb.Booster(model_file=str(path))
            pred = model.predict(X)
        else:
            raise ValueError(f"unsupported family {family}")
        out[entry["prediction_column"]] = np.asarray(pred, np.float32)
    return out


def tcn_prediction(frame, package):
    import torch
    from torch import nn
    ck = torch.load(package/"models/sequence_tcn_tcn_residual.pt",
                    map_location="cpu", weights_only=False)
    cfg = ck["config"]

    class Block(nn.Module):
        def __init__(self, cin, cout, dilation):
            super().__init__()
            pad = dilation*(cfg["kernel_size"]-1)//2
            self.conv1 = nn.Conv1d(cin, cout, cfg["kernel_size"],
                                   padding=pad, dilation=dilation)
            self.conv2 = nn.Conv1d(cout, cout, cfg["kernel_size"],
                                   padding=pad, dilation=dilation)
            self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
            self.drop = nn.Dropout(cfg["dropout"])
        def forward(self, x):
            y = self.drop(torch.relu(self.conv1(x)))
            y = self.drop(self.conv2(y))
            return torch.relu(y+self.skip(x))

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            blocks = []
            cin = cfg["feature_count"]
            for i in range(cfg["blocks"]):
                blocks.append(Block(cin, cfg["channels"], 2**i))
                cin = cfg["channels"]
            self.net = nn.Sequential(*blocks)
            self.head = nn.Conv1d(cin, 1, 1)
        def forward(self, x):
            return self.head(self.net(x)).squeeze(1)

    model = TCN()
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    cols = ck["feature_columns"]
    mean = np.asarray(ck["standardizer"]["mean"], np.float32)
    scale = np.asarray(ck["standardizer"]["scale"], np.float32)
    pred = np.empty(len(frame), np.float32)
    # IDs guarantee well grouping and row order; full suffix is observed.
    wells = frame["id"].astype(str).str.rsplit("_", n=1).str[0]
    with torch.inference_mode():
        for _, idx in frame.groupby(wells, sort=False).groups.items():
            pos = frame.index.get_indexer(idx)
            x = frame.iloc[pos][cols].to_numpy(np.float32)
            x = np.nan_to_num((x-mean)/(scale+1e-7))
            xt = torch.from_numpy(x.T[None])
            pred[pos] = model(xt).numpy()[0]
    return pred


def fresh_pilkwang_delta(data, package, sample=None):
    """Return (sample-aligned frame, delta) using weights and raw data only."""
    data, package = Path(data), Path(package)
    manifest = json.loads(
        (package/"metadata/model_package_manifest.json").read_text())
    if sample is None:
        sample = pd.read_csv(data/"sample_submission.csv", dtype={"id": str})
    else:
        sample = sample.copy()
        sample["id"] = sample.id.astype(str)
    frame = load_builder(package)(
        data_dir=data, sample_submission=sample, package_root=package,
        manifest=manifest)
    pred = tree_predictions(frame, package, manifest)
    pred["pred_delta_sequence_tcn_tcn_residual"] = tcn_prediction(frame, package)
    blend = json.loads((package/"stacking/blend_config.json").read_text())
    delta = sum(float(w)*pred[k] for k, w in blend["weights"].items())
    delta = float(blend["postprocess"]["alpha"])*delta
    # Apply documented delta-space well smoothing.
    from scipy.signal import savgol_filter
    wells = frame["id"].astype(str).str.rsplit("_", n=1).str[0]
    for _, idx in frame.groupby(wells, sort=False).groups.items():
        pos = frame.index.get_indexer(idx)
        n = len(pos); win = min(int(blend["postprocess"]["savgol_window"]), n)
        if win % 2 == 0:
            win -= 1
        if win >= 5:
            delta[pos] = savgol_filter(delta[pos], win, 2)
    return frame, np.asarray(delta, float), sorted(pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT/"data"))
    ap.add_argument("--package", default=str(ROOT/"exp/public_artifacts/pilkwang"))
    ap.add_argument("--out", default=str(
        ROOT/"exp/results/public_model_fresh_inference.csv"))
    args = ap.parse_args()
    data, package = Path(args.data), Path(args.package)
    # Explicit denylist: inference cannot silently consume saved predictions.
    forbidden = {"oof", "diagnostics", "submission.csv",
                 "test_base_predictions.csv", "base_test_predictions.npy"}
    assert not any(x in str(package/"feature_builders") for x in forbidden)
    sample = pd.read_csv(data/"sample_submission.csv", dtype={"id": str})
    frame, delta, model_names = fresh_pilkwang_delta(
        data, package, sample=sample)
    last = frame["last_known_TVT"].to_numpy(float)
    tvt = last + delta
    out = sample[["id"]].copy()
    out["tvt"] = tvt
    if len(out) != len(sample) or not np.isfinite(out.tvt).all():
        raise RuntimeError("fresh inference contract failed")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    audit = {"rows": len(out), "models": model_names,
             "read_prediction_artifacts": False,
             "hardcoded_test_ids": False,
             "features_built_from_raw_official_files": True,
             "tvt_min": float(out.tvt.min()), "tvt_max": float(out.tvt.max())}
    Path(args.out).with_suffix(".audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
