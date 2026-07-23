"""Execute Step 10C only through its public gate using downloaded artifacts."""

from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parents[1]
source = (here / "step10c_stress_consensus_inference.py").read_text(encoding="utf-8")
mapping = {
    'f"{ROOT}/neural-debris-step-8c-renderer-robustness/step8c_renderer_robustness/robustness_feature_table.csv"': repr(str(root / "local_validation/kaggle_step8c_renderer_robustness/output_v2/step8c_renderer_robustness/robustness_feature_table.csv")),
    'f"{ROOT}/neural-debris-step-8b-renderer-gate/step8b_renderer/public_probe_table.csv"': repr(str(root / "local_validation/kaggle_step8b_renderer/output_v3/step8b_renderer/public_probe_table.csv")),
    'f"{ROOT}/neural-debris-v18-canonical-recovery/submission_V18_0_exact_v15b.csv"': repr(str(root / "local_validation/output/control_v15b.csv")),
    'f"{ROOT}/neural-debris-v19-pcgrad-stability-veto/ndr_v19/per_box_diagnostics.csv"': repr(str(root / "forensics/kaggle_ndr_v19_pcgrad_stability/output_v1/ndr_v19/per_box_diagnostics.csv")),
    'f"{ROOT}/neural-debris-v11-v12-trajectory-anchor/artifacts_cands.npz"': repr(str(root / "kernels/experiments/ndr_v11_v12_trajectory_anchor/output_v1/artifacts_cands.npz")),
}
for old, new in mapping.items():
    if old not in source:
        raise RuntimeError(f"missing replacement: {old}")
    source = source.replace(old, new)
marker = "# %% [markdown]\n# ## Frozen deployment calibration"
source = source.replace(marker, 'print(json.dumps(public_audit, indent=2)); raise SystemExit(0)\n\n' + marker, 1)
exec(compile(source, str(here / "step10c_stress_consensus_inference.py"), "exec"), {})
