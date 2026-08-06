"""Build leakage-safe R5 experiments inspired by the Wiggle/Trend write-up."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import build_stack_contact_experiment as stack_base


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "kernels" / "wiggle_r5"
OWNER = "boltuzamaki"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one occurrence, found {count}: {old[:100]!r}")
    return text.replace(old, new, 1)


def splice(text: str, start: str, end: str, replacement: str) -> str:
    lo = text.find(start)
    hi = text.find(end, lo + len(start))
    if lo < 0 or hi < 0:
        raise RuntimeError(f"could not splice {start!r} -> {end!r}")
    return text[:lo] + replacement.rstrip() + "\n\n\n" + text[hi:]


def package_cpu(
    slug: str,
    title: str,
    intro: str,
    control: str,
    driver: str,
    *,
    pf: str | None = None,
    signals: str | None = None,
) -> None:
    notebook = {
        "cells": [
            stack_base.base.markdown(intro),
            stack_base.base.code(control, hidden=False),
            stack_base.base.code(pf or stack_base.base.PF),
            stack_base.base.code(stack_base.base.BEAM),
            stack_base.base.code(signals or stack_base.base.SIGNALS),
            stack_base.base.code(driver),
        ],
        "metadata": stack_base.base.NOTEBOOK["metadata"],
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"r5-{slug[-18:]}-{index}"
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{slug}:cell-{index}")
    metadata = {
        "id": f"{OWNER}/{slug}",
        "title": title,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "kernel_sources": [],
    }
    destination = OUT / slug
    destination.mkdir(parents=True, exist_ok=True)
    (destination / metadata["code_file"]).write_text(
        json.dumps(notebook, indent=1), encoding="utf-8"
    )
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def clean_control() -> str:
    control = replace_once(
        stack_base.CONTROL,
        'SUBMISSION_PROFILE = "contact_gated_anchor"  # previous_stack | contact_gated_anchor',
        'SUBMISSION_PROFILE = "previous_stack"  # strict target-copy-free profile',
    )
    return replace_once(control, "RUN_GROUP_CV = False", "RUN_GROUP_CV = True")


NO_CONTACT_BLOCK = r'''
# Strict audit profile: do not open a same-ID training well after test inference.
contact_rows = []
test["tvt_contact"] = test["tvt_pre_contact"].to_numpy(float)
for _well, _group in test.groupby("well", sort=False):
    contact_rows.append({
        "well": str(_well), "accepted": False,
        "reason": "disabled_target_copy_audit", "replaced_rows": 0,
    })
pd.DataFrame(contact_rows).to_csv(WORK / "guarded_contact_report.csv", index=False)
'''

clean_driver = splice(
    stack_base.DRIVER,
    "def contact_reconstruction(hw_train, tw_train, reference):",
    'sample = pd.read_csv(DATA / "sample_submission.csv")',
    NO_CONTACT_BLOCK,
)
package_cpu(
    "rogii-r5-clean-stack-audit",
    "ROGII R5 Clean Stack Audit",
    "# ROGII R5A: Strict no-contact baseline\n\nNo same-ID training well is opened after test inference. Five-fold grouped CV is enabled, and submission.csv is the honest residual stack.",
    clean_control(),
    clean_driver,
)


PREFIX_FEATURE_CONTACT = r'''
def contact_feature_path(hw_train, tw_train, reference):
    """Use the train twin's formation column, but never its TVT target."""
    if reference not in hw_train.columns:
        return None
    geology = tw_train.dropna(subset=["Geology", "TVT"]).copy()
    ref = geology.loc[geology["Geology"].astype(str) == reference, "TVT"]
    if ref.empty:
        return None
    ref_tvt = float(ref.min())
    raw = ref_tvt - (
        hw_train["Z"].to_numpy(float) - hw_train[reference].to_numpy(float)
    )
    return raw if np.isfinite(raw).sum() >= CONTACT_MIN_PHYS_ROWS else None


def apply_contact_guard(group):
    well = str(group["well"].iloc[0])
    train_hw_path = DATA / "train" / f"{well}__horizontal_well.csv"
    train_tw_path = DATA / "train" / f"{well}__typewell.csv"
    base = group["tvt_pre_contact"].to_numpy(float)
    report = {"well": well, "accepted": False, "replaced_rows": 0}
    if not train_hw_path.exists() or not train_tw_path.exists():
        report["reason"] = "no_same_well_feature_copy"
        return base, report

    hw_test, _ = load_pair(well, "test")
    hw_train = pd.read_csv(train_hw_path)
    tw_train = pd.read_csv(train_tw_path)
    raw = contact_feature_path(hw_train, tw_train, CONTACT_REFERENCE)
    if raw is None:
        report["reason"] = "contact_feature_unavailable"
        return base, report

    train_md = hw_train["MD"].to_numpy(float)
    finite = np.isfinite(train_md) & np.isfinite(raw)
    known = hw_test[hw_test["TVT_input"].notna()].copy()
    comparable = known.loc[
        known["MD"].between(np.nanmin(train_md[finite]), np.nanmax(train_md[finite]))
    ].copy()
    if len(comparable) < max(2 * CONTACT_MIN_PREFIX_ROWS, 100):
        report["reason"] = "too_few_prefix_rows"
        return base, report

    split = max(CONTACT_MIN_PREFIX_ROWS, int(0.70 * len(comparable)))
    fit = comparable.iloc[:split]
    audit = comparable.iloc[split:]
    fit_raw = np.interp(fit["MD"].to_numpy(float), train_md[finite], raw[finite])
    bias = float(np.nanmedian(fit["TVT_input"].to_numpy(float) - fit_raw))
    audit_raw = np.interp(audit["MD"].to_numpy(float), train_md[finite], raw[finite]) + bias
    audit_rmse = pooled_rmse(audit["TVT_input"].to_numpy(float), audit_raw)
    report.update({
        "fit_rows": int(len(fit)), "audit_rows": int(len(audit)),
        "prefix_audit_rmse": float(audit_rmse), "prefix_bias": bias,
        "uses_train_tvt": False, "reference": CONTACT_REFERENCE,
    })
    if audit_rmse > CONTACT_PREFIX_RMSE_LIMIT:
        report["reason"] = "heldout_prefix_failed"
        return base, report

    hidden_idx = group["id"].str.rsplit("_", n=1).str[-1].astype(int).to_numpy()
    hidden_md = hw_test["MD"].to_numpy(float)[hidden_idx]
    inside = (hidden_md >= np.nanmin(train_md[finite])) & (hidden_md <= np.nanmax(train_md[finite]))
    output = base.copy()
    output[inside] = np.interp(hidden_md[inside], train_md[finite], raw[finite]) + bias
    report.update({
        "accepted": True, "reason": "feature_only_prefix_verified",
        "replaced_rows": int(inside.sum()),
    })
    return output, report
'''

prefix_driver = splice(
    stack_base.DRIVER,
    "def contact_reconstruction(hw_train, tw_train, reference):",
    "contact_rows = []",
    PREFIX_FEATURE_CONTACT,
)
prefix_control = replace_once(
    stack_base.CONTROL,
    'SUBMISSION_PROFILE = "contact_gated_anchor"  # previous_stack | contact_gated_anchor',
    'SUBMISSION_PROFILE = "contact_gated_anchor"  # feature-only prefix-calibrated contact',
)
package_cpu(
    "rogii-r5-prefix-feature-contact",
    "ROGII R5 Prefix Feature Contact",
    "# ROGII R5B: Feature-only contact reconstruction\n\nThe formation column may come from an overlapping train well, but its true TVT is never read. Bias is fitted on the first 70% of visible test labels and must pass the remaining 30% before activation.",
    prefix_control,
    prefix_driver,
)


# A genuinely different, stiffer PF partner. Total seed count is held roughly
# constant so the experiment measures decorrelation rather than extra compute.
pf_ensemble = stack_base.base.PF
pf_ensemble = replace_once(
    pf_ensemble,
    "            n_particles=500, seed=0):",
    "            n_particles=500, seed=0, vn=VN, pn=PN, rp=RP, rr=RR):",
)
pf_ensemble = replace_once(
    pf_ensemble,
    "rate = MOM * rate + VN * rng.standard_normal(N)",
    "rate = MOM * rate + vn * rng.standard_normal(N)",
)
pf_ensemble = replace_once(
    pf_ensemble,
    "pos = pos + rate * dm + PN * rng.standard_normal(N)",
    "pos = pos + rate * dm + pn * rng.standard_normal(N)",
)
pf_ensemble = replace_once(
    pf_ensemble,
    "pos = pos[idx] + RP * rng.standard_normal(N)",
    "pos = pos[idx] + rp * rng.standard_normal(N)",
)
pf_ensemble = replace_once(
    pf_ensemble,
    "rate = rate[idx] + RR * rng.standard_normal(N)",
    "rate = rate[idx] + rr * rng.standard_normal(N)",
)
pf_ensemble = replace_once(
    pf_ensemble,
    "def pf_predict(hw, tw, n_particles=500, n_seeds=48, scale=5.0):",
    "def pf_predict(hw, tw, n_particles=500, n_seeds=48, scale=5.0, vn=VN, pn=PN, rp=RP, rr=RR):",
)
pf_ensemble = replace_once(
    pf_ensemble,
    "                                    ir, gs, n_particles, seed=s)",
    "                                    ir, gs, n_particles, seed=s, vn=vn, pn=pn, rp=rp, rr=rr)",
)

pf_call = (
    "pf = pf_predict(h, tw, n_particles=TRAIN_PF_PARTICLES if is_train else TEST_PF_PARTICLES, "
    "n_seeds=TRAIN_PF_SEEDS if is_train else TEST_PF_SEEDS, scale=12.0)"
)
pf_replacement = (
    "_nseed = max(4, (TRAIN_PF_SEEDS if is_train else TEST_PF_SEEDS) // 2)\n"
    "    pf_base = pf_predict(h, tw, n_particles=TRAIN_PF_PARTICLES if is_train else TEST_PF_PARTICLES, "
    "n_seeds=_nseed, scale=12.0)\n"
    "    pf_stiff = pf_predict(h, tw, n_particles=TRAIN_PF_PARTICLES if is_train else TEST_PF_PARTICLES, "
    "n_seeds=_nseed, scale=12.0, vn=0.0010, pn=0.0025, rp=0.05, rr=0.0005)\n"
    "    pf = 0.65 * pf_base + 0.35 * pf_stiff"
)
ensemble_signals = replace_once(stack_base.base.SIGNALS, pf_call, pf_replacement)
package_cpu(
    "rogii-r5-decorrelated-pf",
    "ROGII R5 Decorrelated PF",
    "# ROGII R5C: Decorrelated PF ensemble\n\nBlend the tuned PF with one half-noise/stiffer partner at 0.65/0.35 while holding total seeds approximately constant.",
    clean_control(),
    clean_driver,
    pf=pf_ensemble,
    signals=ensemble_signals,
)


PHYSICS_PROJECT_GROUP = r'''
def project_group(group):
    group = group.copy()
    well = str(group["well"].iloc[0])
    hw, _ = load_pair(well, "test")
    row_idx = group["id"].str.rsplit("_", n=1).str[-1].astype(int).to_numpy()
    md = hw["MD"].to_numpy(float)[row_idx]
    z = hw["Z"].to_numpy(float)[row_idx]
    known = hw[hw["TVT_input"].notna()]
    last = known.iloc[-1]
    anchor_u = float(last["TVT_input"]) + float(last["Z"])
    start_md = float(last["MD"])
    denom = max(float(hw["MD"].iloc[-1]) - start_md, 1e-6)
    s = (md - start_md) / denom
    raw = group["tvt_learned"].to_numpy(float)
    delta_u = raw + z - anchor_u
    fitted_delta_u = robust_polyfit(s, delta_u, 4)
    projected = anchor_u + fitted_delta_u - z
    warmup = np.clip((md - start_md) / 500.0, 0.0, 1.0)
    beta = 0.75 * warmup
    corrected = (1.0 - beta) * raw + beta * projected
    if len(corrected) >= 7:
        window = min(51, len(corrected) if len(corrected) % 2 else len(corrected) - 1)
        if window >= 7:
            corrected = savgol_filter(corrected, window, min(3, window - 1))
    group["tvt_projected"] = projected
    group["tvt_honest"] = corrected
    return group
'''

physics_driver = splice(
    stack_base.DRIVER,
    "def project_group(group):",
    "test = pd.concat(",
    PHYSICS_PROJECT_GROUP,
)
physics_driver = splice(
    physics_driver,
    "def contact_reconstruction(hw_train, tw_train, reference):",
    'sample = pd.read_csv(DATA / "sample_submission.csv")',
    NO_CONTACT_BLOCK,
)
package_cpu(
    "rogii-r5-physics-postprocess",
    "ROGII R5 Physics Postprocess",
    "# ROGII R5D: Exact trend-aware post-process\n\nApply degree-4 robust surface projection with beta=0.75, a 500-ft warm-up, and a 51-point Savitzky-Golay finish. Contact copying is disabled.",
    clean_control(),
    physics_driver,
)


l1_driver = replace_once(
    clean_driver,
    'objective="regression",',
    'objective="regression_l1",',
)
package_cpu(
    "rogii-r5-l1-residual",
    "ROGII R5 L1 Residual",
    "# ROGII R5E: L1 residual corrector\n\nA clean no-contact ablation changing only the LightGBM residual objective from L2 to L1.",
    clean_control(),
    l1_driver,
)


def package_gpu() -> None:
    source_path = ROOT / "kernels" / "neural_stack_integration_r3" / "rogii-current-neural-stack-integration-r3.ipynb"
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    notebook = copy.deepcopy(notebook)
    control = "".join(notebook["cells"][1]["source"])
    control = replace_once(control, "MLP_EPOCHS = 25", "MLP_EPOCHS = 35")
    control = replace_once(control, "MLP_PATIENCE = 5", "MLP_PATIENCE = 7")
    notebook["cells"][1]["source"] = control.splitlines(keepends=True)

    driver = "".join(notebook["cells"][5]["source"])
    old_model = '''class SurfaceMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 256), nn.GELU())
        self.blocks = nn.Sequential(*[ResidualBlock() for _ in range(4)])
        self.output = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, N_GRID))
    def forward(self, x):
        return self.output(self.blocks(self.input(x)))
'''
    new_model = '''class ConvResidual(nn.Module):
    def __init__(self, width=96, dropout=0.08):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, width), nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
            nn.Dropout(dropout), nn.Conv1d(width, width, 5, padding=2), nn.Dropout(dropout),
        )
    def forward(self, x):
        return x + self.net(x)


class SurfaceCNN(nn.Module):
    """Local CNN over level/slope/curvature with a strong visible anchor."""
    def __init__(self, input_dim):
        super().__init__()
        assert input_dim == 3 * N_GRID + 4
        self.stem = nn.Conv1d(3, 96, 7, padding=3)
        self.blocks = nn.Sequential(*[ConvResidual() for _ in range(6)])
        self.extra = nn.Sequential(nn.Linear(4, 96), nn.GELU(), nn.Linear(96, 96))
        self.head = nn.Sequential(nn.GroupNorm(8, 96), nn.GELU(), nn.Conv1d(96, 1, 1))
    def forward(self, x):
        seq = x[:, : 3 * N_GRID].reshape(-1, 3, N_GRID)
        extra = self.extra(x[:, 3 * N_GRID:]).unsqueeze(-1)
        hidden = self.blocks(self.stem(seq)) + extra
        return self.head(hidden).squeeze(1)
'''
    driver = replace_once(driver, old_model, new_model)
    driver = replace_once(driver, "model = SurfaceMLP(x_train.shape[1]).to(DEVICE)", "model = SurfaceCNN(x_train.shape[1]).to(DEVICE)")
    driver = replace_once(
        driver,
        '    "blend_decay_w020": stack_oof + 0.20 * decay * (geometry_oof - stack_oof),',
        '    "blend_decay_w020": stack_oof + 0.20 * decay * (geometry_oof - stack_oof),\n'
        '    "blend_decay_w030": stack_oof + 0.30 * decay * (geometry_oof - stack_oof),',
    )
    driver = driver.replace("rogii-neural-stack-r3", "rogii-surface-cnn-r5")
    notebook["cells"][5]["source"] = driver.splitlines(keepends=True)
    notebook["cells"][0]["source"] = (
        "# ROGII R5F: Anchored Surface CNN\n\n"
        "A grouped-CV GPU experiment implementing the write-up's simple local-CNN plus strong-anchor principle. "
        "The model forecasts only the smooth structural surface U=TVT+Z; the known -Z trajectory supplies the TVT wiggle. "
        "Blend weights through 0.30 are audited against the existing stack.\n"
    ).splitlines(keepends=True)
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"r5-surface-cnn-{index}"
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"surface-cnn:cell-{index}")
    metadata = {
        "id": f"{OWNER}/rogii-r5-anchored-surface-cnn",
        "title": "ROGII R5 Anchored Surface CNN",
        "code_file": "rogii-r5-anchored-surface-cnn.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "kernel_sources": [],
    }
    destination = OUT / "rogii-r5-anchored-surface-cnn"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / metadata["code_file"]).write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


package_gpu()
print("wrote", OUT)
