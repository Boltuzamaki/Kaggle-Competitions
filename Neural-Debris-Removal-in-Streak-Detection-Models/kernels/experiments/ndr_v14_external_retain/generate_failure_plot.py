from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), facecolor="#07110f")
for ax in axes:
    ax.set_facecolor("#0d1b18")
    ax.tick_params(colors="#9fb3ae")
    for spine in ax.spines.values():
        spine.set_color("#284039")

labels = ["V12 anchor", "V14_C"]
scores = [216.5399, 477.6549]
bars = axes[0].bar(labels, scores, color=["#62e7b4", "#ff6b6b"], width=0.58)
axes[0].set_title("Public leaderboard score", color="#e5f2ee", loc="left", weight="bold")
axes[0].set_ylabel("maCADD · lower is better", color="#b9cbc6")
axes[0].grid(axis="y", color="#284039", alpha=0.4)
for bar, value in zip(bars, scores):
    axes[0].text(bar.get_x() + bar.get_width()/2, value + 9, f"{value:.4f}", ha="center", color="#e5f2ee", weight="bold")
axes[0].set_ylim(0, 540)

tiers = ["Suppressed", "Weak 0.10", "Mid", "High ≥0.90"]
v12 = np.array([2936, 0, 656, 190])
v14 = np.array([3884, 3072, 1081, 261])
x = np.arange(len(tiers)); width = 0.36
axes[1].bar(x-width/2, v12, width, label="V12 anchor", color="#62e7b4")
axes[1].bar(x+width/2, v14, width, label="V14_C", color="#ff6b6b")
axes[1].set_xticks(x, tiers, rotation=16, ha="right", color="#b9cbc6")
axes[1].set_ylabel("Submitted boxes", color="#b9cbc6")
axes[1].set_title("V14_C inflated the weak-candidate bank", color="#e5f2ee", loc="left", weight="bold")
axes[1].grid(axis="y", color="#284039", alpha=0.4)
axes[1].legend(frameon=False, labelcolor="#b9cbc6")

fig.suptitle("Why V14_C failed: calibration did not transfer", color="#ff8d8d", fontsize=17, weight="bold", x=0.04, ha="left")
fig.text(0.04, 0.02, "8,298 boxes vs 3,995 · 3,072 weak detections promoted to 0.10 · score regressed by 261.1150", color="#9fb3ae", fontsize=10)
fig.tight_layout(rect=(0.03, 0.07, 0.99, 0.91))
out = Path(__file__).parents[2] / "progress-dashboard" / "public" / "v14-leaderboard-failure.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(out)
