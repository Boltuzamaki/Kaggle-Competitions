"""Build Uza no-submit pairwise GPU kernels from the shared notebook builder."""
from __future__ import annotations

import json
from pathlib import Path

import build_kaggle_gpu_pairwise_wave as base


ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kaggle_remote" / "kernels"


def main() -> None:
    for seed, layers in ((23, 3), (131, 4)):
        name = f"v12-pairwise-s{seed}-l{layers}-uza"
        folder = KERNELS / f"gpu_{name.replace('-', '_')}"
        folder.mkdir(parents=True, exist_ok=True)
        metadata = {
            "id": f"divyanshuboltuzamaki/ptcg-gpu-{name}",
            "title": f"PTCG GPU {name}",
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": False,
            "keywords": ["research"],
            "dataset_sources": ["divyanshuboltuzamaki/ptcg-v12-pairwise-assets-uza"],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        }
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
        (folder / "notebook.ipynb").write_text(
            json.dumps(base.notebook(seed, layers, name), indent=1)
        )
        print(folder)


if __name__ == "__main__":
    main()
