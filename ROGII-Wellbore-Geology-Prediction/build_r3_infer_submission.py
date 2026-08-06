"""Package the already validated R3 honest output as a canonical Kaggle submission.

The R4 notebook contains the promoted R3 full-data inference path, but its
canonical submission.csv was configured as a contact/neural probe.  This
builder changes only the submission profile and kernel identity so Kaggle can
score the standalone R3 model.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = (
    ROOT
    / "kernels"
    / "neural_contact_probe_r4"
    / "rogii-r4-neural-contact-probe.ipynb"
)
OUT = ROOT / "kernels" / "r3_infer_submission"
NOTEBOOK = OUT / "rogii-r3-neural-physics-inference.ipynb"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence of {old!r}, found {count}")
    return text.replace(old, new)


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
for cell in notebook["cells"]:
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    if 'SUBMISSION_PROFILE = "contact_neural_probe"' in source:
        source = replace_once(
            source,
            'SUBMISSION_PROFILE = "contact_neural_probe"',
            'SUBMISSION_PROFILE = "previous_stack"',
        )
        cell["source"] = source.splitlines(keepends=True)

title = """# ROGII R3 Neural Physics Inference

Standalone full-data inference for the R3 model that passed grouped OOF
validation. It combines the repository's own residual physics-signal model
with a small distance-decayed trajectory-network correction. No public
prediction files, public notebook outputs, or external pretrained models are
used. The strict contact trajectory is retained only as an audit artifact;
the canonical submission is the original R3 neural-physics prediction.
"""
notebook["cells"][0] = {
    "cell_type": "markdown",
    "metadata": {},
    "source": title.splitlines(keepends=True),
}

OUT.mkdir(parents=True, exist_ok=True)
NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

metadata = {
    "id": "boltuzamaki/rogii-r3-neural-physics-inference",
    "title": "ROGII R3 Neural Physics Inference",
    "code_file": NOTEBOOK.name,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": False,
    "dataset_sources": [],
    "competition_sources": ["rogii-wellbore-geology-prediction"],
    "kernel_sources": [],
}
(OUT / "kernel-metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)
print(NOTEBOOK)
