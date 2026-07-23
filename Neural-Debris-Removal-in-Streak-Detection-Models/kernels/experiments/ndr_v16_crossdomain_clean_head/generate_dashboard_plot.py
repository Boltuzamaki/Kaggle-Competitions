import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
audit = json.loads((Path(__file__).resolve().parent / "local_audit_v1.json").read_text(encoding="utf-8"))
gate = audit["gate"]

labels = ["External → synthetic", "Synthetic → external", "Required gate"]
values = [gate["external_to_synthetic_auc"], gate["synthetic_to_external_auc"], gate["minimum_required_auc"]]
colors = ["#ff6b6b", "#ff6b6b", "#62e7b4"]

plt.style.use("dark_background")
figure, axis = plt.subplots(figsize=(10.5, 5.4))
figure.patch.set_facecolor("#07100f")
axis.set_facecolor("#07100f")
bars = axis.barh(labels, values, color=colors, height=0.52)
axis.axvline(0.5, color="#f4b860", linestyle="--", linewidth=1.2, label="Random classifier")
axis.set_xlim(0, 1)
axis.set_xlabel("Held-out poison-vs-clean AUC", color="#b8c8c4")
axis.set_title("V16 cross-domain gate rejected the auxiliary clean head", loc="left", weight="bold", fontsize=15)
axis.grid(axis="x", alpha=0.12)
axis.tick_params(colors="#b8c8c4")
for spine in axis.spines.values():
    spine.set_color("#243532")
for bar, value in zip(bars, values):
    axis.text(value + 0.018, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", color="#e7f2ef", weight="bold")
axis.legend(frameon=False, loc="lower right")
figure.text(0.125, 0.015, "Both transfer directions were worse than random; all V16 outputs were safely reverted to the exact V12 anchor.", color="#91a6a1", fontsize=9)
figure.tight_layout(rect=(0, 0.05, 1, 1))
target = ROOT / "progress-dashboard/public/v16-crossdomain-gate.png"
figure.savefig(target, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
print(target)
