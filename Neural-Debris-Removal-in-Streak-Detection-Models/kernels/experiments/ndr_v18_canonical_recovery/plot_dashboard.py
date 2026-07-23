from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "progress-dashboard" / "public" / "v18-canonical-recovery.png"

plt.style.use("dark_background")
fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), facecolor="#07111f")
for axis in axes:
    axis.set_facecolor("#0c1a2b")
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#36506b")
    axis.tick_params(colors="#b8c7d9")
    axis.grid(axis="y", color="#2b4058", alpha=0.55, linewidth=0.7)

directions = ["External →\nsynthetic", "Synthetic →\nexternal"]
auc = [0.8992534722, 0.9160590278]
axes[0].bar(directions, auc, color=["#62e7b4", "#62e7b4"], width=0.58)
axes[0].axhline(0.8, color="#f4b860", linestyle="--", linewidth=1.8, label="required 0.80")
axes[0].set_ylim(0, 1)
axes[0].set_title("Bidirectional transfer AUC", fontsize=14, weight="bold")
axes[0].legend(frameon=False, fontsize=9, loc="lower left")
for index, value in enumerate(auc):
    axes[0].text(index, value + 0.025, f"{value:.4f}", ha="center", color="white", weight="bold")

variants = ["V15_B", "V18_A", "V18_B", "V18_C", "V18_D", "V18_E"]
mass = [515.0, 759.3, 887.6, 970.5, 949.6, 914.6]
colors = ["#95a7ba", "#62e7b4", "#f4b860", "#ff6b6b", "#ff8f70", "#f2c14e"]
axes[1].bar(variants, mass, color=colors, width=0.68)
axes[1].set_title("Total confidence mass", fontsize=14, weight="bold")
axes[1].set_ylabel("confidence mass")
axes[1].set_ylim(0, 1050)
for index, value in enumerate(mass):
    axes[1].text(index, value + 22, f"{value:.0f}", ha="center", color="white", fontsize=9, weight="bold")

fig.suptitle("V18 canonical recovery: gate passed, first high-leverage promotion family", fontsize=17, weight="bold")
fig.text(0.5, 0.015, "V18_A promotes 1,650 incumbent-floor boxes to at most 0.21; no boxes are added or moved.", ha="center", color="#aebed0", fontsize=10)
fig.tight_layout(rect=[0, 0.06, 1, 0.90])
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
print(OUT)
