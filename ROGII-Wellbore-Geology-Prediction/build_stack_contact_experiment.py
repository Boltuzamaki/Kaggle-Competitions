"""Build the controlled previous-stack plus strict-contact Kaggle experiment."""
from __future__ import annotations

import json
from pathlib import Path

import build_contact_gated_training_notebook as base


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "kernels" / "stack_contact_v2"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one replacement, found {count}: {old[:80]!r}")
    return text.replace(old, new, 1)


INTRO = r"""# ROGII Previous Residual Stack + Strict Contact Gate

This controlled experiment changes only the non-contact fallback from the
8.940 contact-gated notebook. The fallback is restored to the earlier honest
physics-signal LightGBM stack:

\[
\widehat{TVT}_i=TVT_{last}+
\operatorname{smooth}\left(
\operatorname{clip}(0.95\,\widehat{\Delta TVT}_i,-60,60)
\right).
\]

The stack is trained end to end from the mounted competition train wells. Its
features include PF and beam paths, multiscale GR matching, trajectory
geometry, GR texture, and typewell residuals. No prediction CSV or pretrained
competition model is imported.

The same-well EGFDU reconstruction remains a strictly guarded final layer. It
is accepted only when at least 50 visible rows reproduce within 1.0 ft RMSE.
When that check fails or the train twin is unavailable, the residual stack is
left untouched.

Outputs:

* `submission_honest.csv`: previous residual-stack fallback only.
* `submission_contact_gated.csv`: fallback plus verified contact replacement.
* `submission.csv`: selected competition submission.
* Validation, contact, prefix, feature-importance, and integrity audits.
"""


CONTROL = replace_once(
    base.CONTROL,
    'SUBMISSION_PROFILE = "contact_gated_anchor"  # honest_anchor | contact_gated_anchor | visible_prefix_bounded',
    'SUBMISSION_PROFILE = "contact_gated_anchor"  # previous_stack | contact_gated_anchor',
)
CONTROL = replace_once(
    CONTROL,
    "LGB_ESTIMATORS = 900",
    "LGB_ESTIMATORS = 1200\nSTACK_SHRINK = 0.95\nSTACK_CLIP = 60.0",
)
CONTROL = replace_once(
    CONTROL,
    'if SUBMISSION_PROFILE not in {"honest_anchor", "contact_gated_anchor", "visible_prefix_bounded"}:',
    'if SUBMISSION_PROFILE not in {"previous_stack", "contact_gated_anchor"}:',
)


DRIVER = base.DRIVER
DRIVER = replace_once(
    DRIVER,
    'y = train["target_surface"].to_numpy(np.float32)',
    'y = train["target"].to_numpy(np.float32)',
)
DRIVER = replace_once(
    DRIVER,
    'ridge = Ridge(alpha=30.0).fit(scaler.transform(X[ridge_idx]), y[ridge_idx])',
    'ridge_surface_y = train["target_surface"].to_numpy(np.float32)\n'
    'ridge = Ridge(alpha=30.0).fit(\n'
    '    scaler.transform(X[ridge_idx]), ridge_surface_y[ridge_idx]\n'
    ')',
)
DRIVER = replace_once(DRIVER, 'learning_rate=0.025,', 'learning_rate=0.02,')
DRIVER = replace_once(DRIVER, 'colsample_bytree=0.75,', 'colsample_bytree=0.70,')
DRIVER = replace_once(DRIVER, 'reg_lambda=7.0,', 'reg_lambda=5.0,')
DRIVER = DRIVER.replace("lightgbm_surface_holdout", "previous_residual_stack_holdout")
DRIVER = DRIVER.replace("lightgbm_surface", "previous_residual_stack")
DRIVER = DRIVER.replace("pooled surface RMSE", "pooled residual-stack RMSE")
DRIVER = replace_once(
    DRIVER,
    "learned_test_ds = model.predict(Xt)",
    "learned_test_d = np.clip(\n"
    "    model.predict(Xt) * STACK_SHRINK, -STACK_CLIP, STACK_CLIP\n"
    ")",
)
DRIVER = replace_once(
    DRIVER,
    'test["learned_ds"] = learned_test_ds',
    'test["learned_d"] = learned_test_d',
)
DRIVER = replace_once(
    DRIVER,
    'test["tvt_learned"] = (\n'
    '    test["last_known_tvt"].astype(float) + test["learned_ds"] - test["d_z"]\n'
    ')',
    'test["tvt_learned"] = (\n'
    '    test["last_known_tvt"].astype(float) + test["learned_d"]\n'
    ')',
)
DRIVER = replace_once(
    DRIVER,
    '    group["tvt_honest"] = (\n'
    '        PROJECTED_ANCHOR_WEIGHT * group["tvt_projected"].to_numpy(float)\n'
    '        + (1.0 - PROJECTED_ANCHOR_WEIGHT)\n'
    '        * smooth(group["tvt_learned"].to_numpy(float))\n'
    '    )',
    '    # Controlled ablation: the previous residual stack is the complete\n'
    '    # non-contact fallback. Projection remains available for diagnostics.\n'
    '    group["tvt_honest"] = smooth(group["tvt_learned"].to_numpy(float))',
)
DRIVER = replace_once(
    DRIVER,
    'final_sub = honest_sub if SUBMISSION_PROFILE == "honest_anchor" else contact_sub',
    'final_sub = honest_sub if SUBMISSION_PROFILE == "previous_stack" else contact_sub',
)
DRIVER = replace_once(
    DRIVER,
    '    "profile": SUBMISSION_PROFILE,',
    '    "profile": SUBMISSION_PROFILE,\n'
    '    "fallback": "previous_residual_stack",\n'
    '    "stack_shrink": float(STACK_SHRINK),\n'
    '    "stack_clip": float(STACK_CLIP),',
)


NOTEBOOK = {
    "cells": [
        base.markdown(INTRO),
        base.code(CONTROL, hidden=False),
        base.code(base.PF),
        base.code(base.BEAM),
        base.code(base.SIGNALS),
        base.code(DRIVER),
    ],
    "metadata": base.NOTEBOOK["metadata"],
    "nbformat": 4,
    "nbformat_minor": 5,
}

METADATA = {
    "id": "boltuzamaki/rogii-contact-gated-stratigraphic-alignment",
    "title": "ROGII Contact-Gated Stratigraphic Alignment",
    "code_file": "rogii-previous-stack-contact-gate-v2.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": False,
    "dataset_sources": [],
    "competition_sources": ["rogii-wellbore-geology-prediction"],
    "kernel_sources": [],
}


OUT.mkdir(parents=True, exist_ok=True)
(OUT / METADATA["code_file"]).write_text(
    json.dumps(NOTEBOOK, indent=1), encoding="utf-8"
)
(OUT / "kernel-metadata.json").write_text(
    json.dumps(METADATA, indent=2), encoding="utf-8"
)
print("wrote", OUT)
