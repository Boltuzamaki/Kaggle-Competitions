"""Build no-submit pairwise-advantage GPU kernels."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kaggle_remote" / "kernels"
DATASET = "boltuzamaki/ptcg-v12-pairwise-assets"


def notebook(seed: int, layers: int, name: str) -> dict:
    source = f'''from pathlib import Path
import glob, json, os, shutil, subprocess, sys

gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True, check=False).stdout
if "P100" in gpu_name:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

scripts = glob.glob("/kaggle/input/**/tools/rank/train_pairwise_advantage.py", recursive=True)
if scripts:
    root = Path(scripts[0]).parents[2]
else:
    assets = glob.glob("/kaggle/input/**/project.zip", recursive=True)
    if not assets:
        raise RuntimeError("pairwise asset dataset missing")
    root = Path("/kaggle/working/pairwise")
    shutil.unpack_archive(assets[0], root)
os.chdir(root)
command = [sys.executable, "tools/rank/train_pairwise_advantage.py",
    "--data", "data/v12_counterfactual_discordant.pkl",
    "--out", "/kaggle/working/{name}.pt", "--epochs", "48",
    "--batch", "256", "--seed", "{seed}", "--layers", "{layers}"]
subprocess.run(command, check=True)
checkpoint = torch.load("/kaggle/working/{name}.pt", map_location="cpu", weights_only=False)
summary = {{"experiment": "{name}", "seed": {seed}, "layers": {layers},
    "accuracy": checkpoint["accuracy"], "gate": checkpoint["gate"],
    "rows": checkpoint["rows"], "cuda": torch.cuda.is_available(),
    "submission_performed": False}}
Path("/kaggle/working/final_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
'''
    return {
        "cells": [{"cell_type": "markdown", "metadata": {},
                   "source": ["# Counterfactual pairwise advantage\n", "Research only; no submission.\n"]},
                  {"cell_type": "code", "metadata": {}, "execution_count": None,
                   "outputs": [], "source": source.splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"}, "accelerator": "GPU"},
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> None:
    for seed, layers in ((73, 2), (97, 5)):
        name = f"v12-pairwise-s{seed}-l{layers}"
        folder = KERNELS / f"gpu_{name.replace('-', '_')}"
        folder.mkdir(parents=True, exist_ok=True)
        metadata = {"id": f"boltuzamaki/ptcg-gpu-{name}",
                    "title": f"PTCG GPU {name}", "code_file": "notebook.ipynb",
                    "language": "python", "kernel_type": "notebook", "is_private": True,
                    "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
                    "keywords": ["research"], "dataset_sources": [DATASET],
                    "kernel_sources": [], "competition_sources": [], "model_sources": []}
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
        (folder / "notebook.ipynb").write_text(json.dumps(notebook(seed, layers, name), indent=1))
        print(folder)


if __name__ == "__main__":
    main()
