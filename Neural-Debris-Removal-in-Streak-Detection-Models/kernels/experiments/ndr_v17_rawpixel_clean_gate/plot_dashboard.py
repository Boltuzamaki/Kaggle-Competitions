from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "progress-dashboard" / "public" / "v17-crossdomain-gate.png"

labels = ["External →\nsynthetic", "Synthetic →\nexternal"]
auc = [0.8059809028, 0.4965538194]
margin = [0.0178222656, -0.0046386719]

plt.style.use("dark_background")
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), facecolor="#07111f")
colors = ["#f4b860", "#ff6b6b"]

for axis in axes:
    axis.set_facecolor("#0c1a2b")
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#36506b")
    axis.tick_params(colors="#b8c7d9")
    axis.grid(axis="y", color="#2b4058", alpha=0.55, linewidth=0.7)

axes[0].bar(labels, auc, color=colors, width=0.58)
axes[0].axhline(0.75, color="#62e7b4", linestyle="--", linewidth=1.8, label="required 0.75")
axes[0].axhline(0.5, color="#95a7ba", linestyle=":", linewidth=1.2, label="chance 0.50")
axes[0].set_ylim(0, 1)
axes[0].set_title("Cross-domain AUC", color="white", fontsize=14, weight="bold")
axes[0].legend(frameon=False, fontsize=9, loc="lower left")
for idx, value in enumerate(auc):
    axes[0].text(idx, value + 0.025, f"{value:.4f}", ha="center", color="white", weight="bold")

axes[1].bar(labels, margin, color=colors, width=0.58)
axes[1].axhline(0.15, color="#62e7b4", linestyle="--", linewidth=1.8, label="required 0.15")
axes[1].axhline(0, color="#95a7ba", linewidth=1)
axes[1].set_ylim(-0.04, 0.19)
axes[1].set_title("Probability margin", color="white", fontsize=14, weight="bold")
axes[1].legend(frameon=False, fontsize=9, loc="upper right")
for idx, value in enumerate(margin):
    offset = 0.007 if value >= 0 else -0.015
    axes[1].text(idx, value + offset, f"{value:+.4f}", ha="center", color="white", weight="bold")

fig.suptitle("V17 raw-pixel gate: one partial pass, overall rejection", color="white", fontsize=17, weight="bold")
fig.text(0.5, 0.015, "Both AUC and margin had to pass in both directions before any test confidence could change.", ha="center", color="#aebed0", fontsize=10)
fig.tight_layout(rect=[0, 0.06, 1, 0.90])
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
print(OUT)
