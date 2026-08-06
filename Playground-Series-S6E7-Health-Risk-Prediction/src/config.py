"""Loads config.yaml into a plain, dot-free nested dict, plus a couple of
derived conveniences (absolute paths, device flags). Single source of truth
for every script in src/.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = REPO_ROOT / "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = REPO_ROOT
    paths = cfg["paths"]
    for key, val in paths.items():
        paths[key] = str((root / val).resolve())

    data = cfg["data"]
    data["dir"] = str((root / data["dir"]).resolve())

    for d in [
        paths["artifacts_dir"],
        paths["features_dir"],
        paths["oof_dir"],
        paths["test_pred_dir"],
        paths["log_dir"],
        paths["submissions_dir"],
    ]:
        os.makedirs(d, exist_ok=True)

    cfg["_repo_root"] = str(root)
    return cfg


def is_gpu(cfg: dict) -> bool:
    return cfg["device"]["mode"].lower() == "gpu"


def gpu_id(cfg: dict) -> int:
    return int(cfg["device"].get("gpu_id", 0))
