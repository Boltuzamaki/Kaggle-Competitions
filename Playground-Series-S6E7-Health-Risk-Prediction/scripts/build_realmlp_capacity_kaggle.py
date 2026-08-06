"""Build a higher-capacity RealMLP experiment for Kaggle GPU."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kaggle_kernels/realmlp_seed2027_gpu/realmlp_seed2027_gpu.ipynb"
OUT = ROOT / "kaggle_kernels/realmlp_capacity9001_gpu"


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    replacements = {
        "seed_everything(2027)": "seed_everything(9001)",
        "torch.cuda.manual_seed_all(2027)": "torch.cuda.manual_seed_all(9001)",
        '"embed_dim":     8': '"embed_dim":     12',
        '"hidden_dims":   [512, 512, 512]': '"hidden_dims":   [768, 768, 512]',
        '"dropout":       0.06': '"dropout":       0.05',
        '"pbld_hidden_dim": 20': '"pbld_hidden_dim": 24',
        '"pbld_out_dim":    5': '"pbld_out_dim":    6',
        '"lr":               0.01': '"lr":               0.007',
        '"weight_decay":     0.013': '"weight_decay":     0.012',
        '"epochs":    3': '"epochs":    4',
        '"train_bs":  256': '"train_bs":  192',
        '"random_state": 2027': '"random_state": 9001',
        "SEED = 2027": "SEED = 9001",
        'EXPERIMENT_ID = "realmlp_seed2027_gpu_7fold_3epoch"': (
            'EXPERIMENT_ID = "realmlp_capacity9001_gpu_7fold_4epoch"'
        ),
    }
    for old, new in replacements.items():
        hits = 0
        for cell in notebook["cells"]:
            source = cell.get("source", [])
            text = "".join(source) if isinstance(source, list) else source
            count = text.count(old)
            if count:
                text = text.replace(old, new)
                cell["source"] = (
                    text.splitlines(keepends=True) if isinstance(source, list) else text
                )
                hits += count
        if hits != 1:
            raise RuntimeError(f"Expected one occurrence of {old!r}, found {hits}")

    OUT.mkdir(parents=True, exist_ok=True)
    notebook_name = "realmlp_capacity9001_gpu.ipynb"
    (OUT / notebook_name).write_text(json.dumps(notebook, ensure_ascii=True, indent=1))
    metadata = {
        "id": "boltuzamaki/health-risk-realmlp-capacity-9001-gpu",
        "title": "Health Risk RealMLP Capacity 9001 GPU",
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": ["playground-series-s6e7"],
        "docker_image": (
            "gcr.io/kaggle-private-byod/python@sha256:"
            "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
        ),
        "machine_shape": "NvidiaTeslaT4",
    }
    (OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Built {OUT / notebook_name}")


if __name__ == "__main__":
    main()
