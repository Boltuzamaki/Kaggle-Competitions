import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).parent
REPORT = json.loads(
    (HERE / "audit_files_v3" / "ndr_v14" / "final_report.json").read_text(encoding="utf-8")
)
V13 = pd.read_csv(
    HERE.parent / "kaggle_ndr_v13_breakthrough" / "audit_files_v2" / "ndr_v13" / "per_box_diagnostics.csv"
)
V14 = pd.read_csv(HERE / "audit_files_v3" / "ndr_v14" / "per_box_diagnostics.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#07110f")
for ax in axes:
    ax.set_facecolor("#0d1b18")
    ax.tick_params(colors="#9fb3ae")
    for spine in ax.spines.values():
        spine.set_color("#284039")

ax = axes[0]
candidates = REPORT["pcgrad"]["candidates"]
x = [100 * row["poison_ratio"] for row in candidates]
y = [100 * row["retain_ratio"] for row in candidates]
ax.axvspan(0, 25, color="#62e7b4", alpha=0.07)
ax.axhspan(90, 115, color="#62e7b4", alpha=0.07)
ax.scatter(x, y, s=110, c=["#6ba1ff", "#f4b860", "#6ba1ff", "#62e7b4"], edgecolor="#e5f2ee")
for row, px, py in zip(candidates, x, y):
    ax.annotate(f"β={row['beta']:g}", (px, py), xytext=(7, 7), textcoords="offset points", color="#e5f2ee", fontsize=10)
ax.scatter([36.10], [109.36], marker="D", s=90, color="#ff6b6b", label="V13 projection")
ax.scatter([99.99], [99.99], marker="x", s=110, color="#f4b860", label="V13 task vector")
ax.axvline(25, color="#62e7b4", linestyle="--", linewidth=1)
ax.axhline(90, color="#62e7b4", linestyle="--", linewidth=1)
ax.set_xlim(0, 105)
ax.set_ylim(80, 112)
ax.set_xlabel("Poison confidence retained (%) - lower is better", color="#b9cbc6")
ax.set_ylabel("Real-streak confidence retained (%)", color="#b9cbc6")
ax.set_title("Repair trade-off on public-only controls", color="#e5f2ee", loc="left", weight="bold")
ax.legend(frameon=False, labelcolor="#b9cbc6", loc="lower right")
ax.grid(color="#284039", alpha=0.35, linewidth=0.7)

merged = V13.merge(V14, on=["image_id", "candidate"])
rows = ["V13 rank", "V13 projection", "V13 consensus"]
cols = ["V14 external rank", "V14 PCGrad", "V14 consensus"]
matrix = merged[["rank_poison", "projection_poison", "consensus_poison", "rank", "pcgrad", "consensus"]].corr().iloc[:3, 3:].to_numpy()
ax = axes[1]
image = ax.imshow(matrix, cmap="viridis", vmin=-0.25, vmax=1.0)
ax.set_xticks(range(3), cols, rotation=22, ha="right", color="#b9cbc6")
ax.set_yticks(range(3), rows, color="#b9cbc6")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", weight="bold")
ax.set_title("V14 adds a genuinely different signal", color="#e5f2ee", loc="left", weight="bold")
cb = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
cb.ax.tick_params(colors="#9fb3ae")
cb.outline.set_edgecolor("#284039")

fig.suptitle("V13 → V14: external-real retention changes the frontier", color="#62e7b4", fontsize=17, weight="bold", x=0.04, ha="left")
fig.text(0.04, 0.02, "Frozen public-only selection · 2,000 test IDs · no test-derived tuning · no Kaggle submission", color="#9fb3ae", fontsize=10)
fig.tight_layout(rect=(0.03, 0.06, 0.99, 0.92))
out = HERE.parent.parent / "progress-dashboard" / "public" / "v13-v14-breakthrough.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(out)
