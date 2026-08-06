"""Prepare (do not push) a Harshini + Pilkwang fresh-inference notebook."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
SRC = ROOT/"exp/results/harshini_kernel_source/rogii-harshini-scratch-stack.ipynb"
OUTDIR = ROOT/"notebooks/harshini_pilkwang_fresh"
OUTDIR.mkdir(parents=True, exist_ok=True)

nb = json.loads(SRC.read_text())
fresh = (ROOT/"exp/public_model_fresh_inference.py").read_text()
fresh = fresh.split('if __name__ == "__main__":')[0]
integration = (ROOT/"exp/harshini_pilkwang_integration.py").read_text()
fresh = fresh.replace("ROOT = Path(__file__).resolve().parents[1]",
                      "ROOT = Path('/kaggle/working')")
integration = integration.replace(
    "ROOT = Path(__file__).resolve().parents[1]",
    "ROOT = Path('/kaggle/working')")

def cell(source, tags):
    return {"cell_type": "code", "execution_count": None, "outputs": [],
            "metadata": {"tags": tags}, "source": source}

nb["cells"].append(cell(
    "# Fresh public-weight inference implementation. No saved prediction "
    "artifact is read.\n"+fresh,
    ["fresh-model-inference", "no-prediction-csv"]))
nb["cells"].append(cell(
    integration+"\n\n"
    "from pathlib import Path as _Path\n"
    "_data = _Path(DATA)\n"
    "_pkg_candidates = [\n"
    " _Path('/kaggle/input/datasets/pilkwang/rogii-model-package'),\n"
    " _Path('/kaggle/input/rogii-model-package')]\n"
    "_pkg = next((p for p in _pkg_candidates if p.exists()), None)\n"
    "if _pkg is None: raise RuntimeError('Pilkwang model package missing')\n"
    "_final, _audit = integrate(sub, _data, _pkg, fresh_pilkwang_delta)\n"
    "save_audited(_final, _audit, '/kaggle/working')\n"
    "sub = _final\n"
    "print(json.dumps(_audit, indent=2))\n",
    ["oof-ridge-integration", "submission-output"]))

nb_path = OUTDIR/"rogii-harshini-pilkwang-fresh.ipynb"
nb_path.write_text(json.dumps(nb))
meta = {
    "id": "boltuzamaki/rogii-harshini-pilkwang-fresh",
    "title": "ROGII Harshini Pilkwang Fresh Inference",
    "code_file": nb_path.name,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": False,
    "competition_sources": ["rogii-wellbore-geology-prediction"],
    "dataset_sources": ["pilkwang/rogii-model-package"],
    "kernel_sources": [],
    "model_sources": [],
}
(OUTDIR/"kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(nb_path)
