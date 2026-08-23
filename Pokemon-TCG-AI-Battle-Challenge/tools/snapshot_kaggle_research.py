"""Publish a small Kaggle-only research status snapshot for the local dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "scratchpad" / "kaggle_remote_status.json"
SCRIPT_OUTPUT = ROOT / "dashboard" / "kaggle-status.js"

LANES = [
    {
        "id": "KAGGLE-CPU-01",
        "kernel": "boltuzamaki/ptcg-cpu-live-flat-w8-i16-v1",
        "title": "Flat root live field, budget 16",
        "resource": "CPU",
        "result": "232 wins, 246 losses, 2 draws across 480 games. Rejected at 48.33%.",
        "decision": "rejected",
    },
    {
        "id": "KAGGLE-CPU-02",
        "kernel": "boltuzamaki/ptcg-cpu-static-flat-w8-i40-v1",
        "title": "Flat root static gauntlet, budget 40",
        "resource": "CPU",
        "result": "211 wins and 109 losses across 320 games. Static lower bound was 60.58%.",
        "decision": "complete",
    },
    {
        "id": "KAGGLE-CPU-03",
        "kernel": "boltuzamaki/ptcg-cpu-live-flat-w8-i40-v1",
        "title": "Flat root live field, budget 40",
        "resource": "CPU",
        "result": "235 wins and 245 losses across 480 games. Rejected at 48.96%.",
        "decision": "rejected",
    },
    {
        "id": "KAGGLE-GPU-01",
        "kernel": "boltuzamaki/ptcg-gpu-v3-only-continuation-v1",
        "title": "Original v3-only learner continuation",
        "resource": "GPU request",
        "result": "Iteration 70 completed, but the training summary reported CUDA false.",
        "decision": "complete",
    },
    {
        "id": "KAGGLE-CPU-04",
        "kernel": "boltuzamaki/ptcg-cpu-gpu-iter70-holdout-v1",
        "title": "Iteration 70 frozen holdout",
        "resource": "CPU",
        "result": "5 wins in 120 games against v3 and 0 wins in 30 against random. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-47A",
        "kernel": "boltuzamaki/ptcg-cpu-original-v3-abomasnow-live-field",
        "title": "Original v3 policy with Abomasnow",
        "resource": "CPU",
        "result": "116 wins and 204 losses across 320 live-field games. Rejected at 36.25% with zero failures.",
        "decision": "rejected",
    },
    {
        "id": "EXP-47B",
        "kernel": "boltuzamaki/ptcg-cpu-original-v3-lucario-live-field",
        "title": "Original v3 policy with Lucario",
        "resource": "CPU",
        "result": "110 wins and 210 losses across 320 live-field games. Rejected at 34.38% with zero failures.",
        "decision": "rejected",
    },
    {
        "id": "EXP-47C",
        "kernel": "boltuzamaki/ptcg-cpu-original-v3-alakazam-live-field",
        "title": "Original v3 policy with Alakazam",
        "resource": "CPU",
        "result": "112 wins and 208 losses across 320 live-field games. Rejected at 35.0% with zero failures.",
        "decision": "rejected",
    },
    {
        "id": "EXP-47D",
        "kernel": "boltuzamaki/ptcg-cpu-original-v3-dragapult-live-field",
        "title": "Original v3 policy with Dragapult",
        "resource": "CPU",
        "result": "77 wins and 243 losses across 320 live-field games. Rejected at 24.06% with zero failures.",
        "decision": "rejected",
    },
    {
        "id": "EXP-48",
        "kernel": "boltuzamaki/ptcg-cpu-recovery-inventory-live-field",
        "title": "Recovery inventory planner",
        "resource": "CPU",
        "result": "98 wins and 142 losses across 240 live-field games. Rejected at 40.83% with zero failures.",
        "decision": "rejected",
    },
    {
        "id": "EXP-49",
        "kernel": "boltuzamaki/ptcg-cpu-end-turn-regret-live-field",
        "title": "End turn regret check",
        "resource": "CPU",
        "result": "97 wins and 143 losses across 240 live-field games. Rejected at 40.42% with zero failures.",
        "decision": "rejected",
    },
    {
        "id": "EXP-50",
        "kernel": "boltuzamaki/ptcg-cpu-bench-capacity-live-field",
        "title": "Bench capacity planner",
        "resource": "CPU",
        "result": "90 wins and 150 losses across 240 live-field games. Rejected at 37.5% with zero failures.",
        "decision": "rejected",
    },
    {
        "id": "EXP-51",
        "kernel": "boltuzamaki/ptcg-gpu-own-beam-distillation",
        "title": "Distill our own beam policy",
        "resource": "GPU request",
        "result": "GPU training finished with 99.05% Alakazam holdout agreement, but only 0.72 points gain over the sequenced teacher. Arena promotion was not earned.",
        "decision": "rejected",
    },
    {
        "id": "EXP-52",
        "kernel": "boltuzamaki/ptcg-cpu-evolution-stack-live-field",
        "title": "Evolution stack preservation",
        "resource": "CPU",
        "result": "97 wins and 143 losses across 240 live-field games, or 40.42%, with zero failures.",
        "decision": "rejected",
    },
    {
        "id": "EXP-53",
        "kernel": "boltuzamaki/ptcg-cpu-damage-breakpoint-live-field",
        "title": "Damage breakpoint planner",
        "resource": "CPU",
        "result": "99 wins and 141 losses across 240 live-field games, or 41.25%, with zero failures.",
        "decision": "rejected",
    },
    {
        "id": "EXP-54",
        "kernel": "boltuzamaki/ptcg-cpu-stadium-timing-live-field",
        "title": "Full Metal Lab timing model",
        "resource": "CPU",
        "result": "87 wins and 153 losses across 240 live-field games, or 36.25%, with zero failures.",
        "decision": "rejected",
    },
    {
        "id": "EXP-55",
        "kernel": "boltuzamaki/ptcg-cpu-healing-breakpoint-live-field",
        "title": "Healing value forecast",
        "resource": "CPU",
        "result": "96 wins and 144 losses across 240 live-field games, or 40.0%, with zero failures.",
        "decision": "rejected",
    },
    {
        "id": "EXP-56",
        "kernel": "boltuzamaki/ptcg-cpu-risk-retreat-live-field-v2",
        "profile": "bolt",
        "title": "Risk aware retreat selector, corrected override",
        "resource": "CPU",
        "result": "Corrected policy finished 78-162 across 240 live-field games, or 32.5%, with zero failures. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-57",
        "kernel": "boltuzamaki/ptcg-cpu-hand-quality-live-field-v3",
        "profile": "bolt",
        "title": "Hand quality estimator, corrected override",
        "resource": "CPU",
        "result": "Corrected policy finished 99-141 across 240 live-field games, or 41.25%, with zero failures. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-58",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-state-cache-comparison-v2",
        "profile": "uza",
        "title": "State transposition cache comparison",
        "resource": "CPU",
        "result": "Cache treatment improved p95 by 8.23% and scored 15-21 versus 13-23 for control, with zero failures. Rejected because speed gain missed the 15% gate.",
        "decision": "rejected",
    },
    {
        "id": "EXP-59",
        "kernel": "boltuzamaki/ptcg-cpu-selective-search-comparison",
        "profile": "bolt",
        "title": "Selective search trigger",
        "resource": "CPU",
        "result": "Gating removed 40.15% of p95 cost but scored 16-20 versus 17-19 for always-on search, with zero failures. Rejected because win rate fell.",
        "decision": "rejected",
    },
    {
        "id": "EXP-60",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-sequential-probability-test-v2",
        "profile": "uza",
        "title": "Sequential probability testing",
        "resource": "CPU",
        "result": "20,000 even-match simulations gave a 4.675% false-promotion rate, passing the 5% gate.",
        "decision": "complete",
    },
    {
        "id": "EXP-61",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-decision-invariant-check-v2",
        "profile": "uza",
        "title": "Decision invariant checker, corrected recheck",
        "resource": "CPU",
        "result": "Corrected wrapper passed 3,247 decisions across 64 games with zero missed overrides, failures, or game errors.",
        "decision": "complete",
    },
    {
        "id": "EXP-62",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-strategy-fusion-audit-v2",
        "profile": "uza",
        "title": "Strategy fusion detector",
        "resource": "CPU",
        "result": "Nine of 40 probed decisions preferred multiple actions across hidden samples; mean disagreement was 7.81%. This validates shared information-set aggregation as a real target.",
        "decision": "complete",
    },
    {
        "id": "EXP-63A",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-particle-belief-1x48",
        "profile": "uza",
        "title": "Particle belief search 1 by 48",
        "resource": "CPU",
        "result": "One hidden-state sample by 48 rollout steps scored 33-47 across 80 held-out live-deck games with zero failures. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-63B",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-particle-belief-3x16",
        "profile": "uza",
        "title": "Particle belief search 3 by 16",
        "resource": "CPU",
        "result": "Three hidden-state samples by 16 rollout steps scored 43-37 across 80 held-out live-deck games with zero failures. Promising, but the 95% lower bound remains below 50%, so fresh confirmation is required.",
        "decision": "conditional",
    },
    {
        "id": "EXP-63C",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-particle-belief-3x16-confirmation",
        "profile": "uza",
        "title": "Particle belief search fresh confirmation",
        "resource": "CPU",
        "result": "Fresh confirmation finished 114-126 across 240 games, or 47.5%, with a 41.27% Wilson lower bound and zero failures. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-64A",
        "kernel": "boltuzamaki/ptcg-gpu-own-policy-scale-17m",
        "profile": "bolt",
        "title": "Original policy model scale sweep, 1.7M",
        "resource": "GPU",
        "result": "T4 training completed with zero failures, but Alakazam holdout gain over the sequenced baseline was only 0.086 percentage points. Rejected before arena.",
        "decision": "rejected",
    },
    {
        "id": "EXP-64B",
        "kernel": "divyanshuboltuzamaki/ptcg-gpu-own-policy-scale-5m",
        "profile": "uza",
        "title": "Original policy model scale sweep, 5M",
        "resource": "GPU request",
        "result": "Uza requested GPU twice, including an explicit Nvidia Tesla T4 request, but Kaggle supplied a CPU runtime and CUDA was unavailable. Waiting for account-level GPU access.",
        "decision": "blocked",
    },
    {
        "id": "EXP-63D",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-belief-6x8-live-field",
        "profile": "uza",
        "title": "Particle belief allocation, 6 by 8",
        "resource": "CPU",
        "result": "Finished 114-126 across 240 games, or 47.5%, with zero failures. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-63E",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-belief-12x4-live-field",
        "profile": "uza",
        "title": "Particle belief allocation, 12 by 4",
        "resource": "CPU",
        "result": "Finished 98-142 across 240 games, or 40.83%, with zero failures. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-65",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-progressive-widening-live-field",
        "profile": "uza",
        "title": "Progressive widening search",
        "resource": "CPU",
        "result": "Finished 126-114 across 240 games, or 52.5%, with a 46.19% Wilson lower bound and zero failures. Inconclusive.",
        "decision": "conditional",
    },
    {
        "id": "EXP-66",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-risk-sensitive-live-field",
        "profile": "uza",
        "title": "Risk-sensitive rollout value",
        "resource": "CPU",
        "result": "Opened 132-108 across 240 games, or 55.0%, with a 48.68% Wilson lower bound and zero failures. Fresh confirmation is running.",
        "decision": "conditional",
    },
    {
        "id": "EXP-67",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-disagreement-fallback-live-field",
        "profile": "uza",
        "title": "Disagreement fallback ensemble",
        "resource": "CPU",
        "result": "Finished 117-123 across 240 games, or 48.75%, with zero failures. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-68",
        "kernel": "boltuzamaki/ptcg-gpu-dagger-own-oracle-v1",
        "profile": "bolt",
        "title": "DAgger with our own beam oracle",
        "resource": "GPU",
        "result": "Completed on T4, but Alakazam holdout was 15-25 and teacher-agreement gain was only 0.874 percentage points. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-66C",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-risk-sensitive-confirmation",
        "profile": "uza",
        "title": "Risk-sensitive fresh confirmation",
        "resource": "CPU",
        "result": "Fresh confirmation finished 236-244 across 480 games, or 49.17%, with a 44.72% Wilson lower bound and zero failures. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-69",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-chance-coupled-live-field-v2",
        "profile": "uza",
        "title": "Coupled chance sampling",
        "resource": "CPU",
        "result": "Finished 129-111 across 240 games, or 53.75%, with a 47.43% Wilson lower bound and zero failures. Fresh confirmation is running.",
        "decision": "conditional",
    },
    {
        "id": "EXP-70",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-adaptive-budget-live-field-v2",
        "profile": "uza",
        "title": "Adaptive search budget",
        "resource": "CPU",
        "result": "Finished 107-133 across 240 games, or 44.58%, with zero failures. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-71",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-risk-sensitive-weight-020",
        "profile": "uza",
        "title": "Risk-sensitive weight 0.20",
        "resource": "CPU",
        "result": "Weight 0.20 finished 122-118 across 240 games, or 50.83%, with zero failures. No promotion.",
        "decision": "rejected",
    },
    {
        "id": "EXP-72",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-risk-sensitive-weight-070",
        "profile": "uza",
        "title": "Risk-sensitive weight 0.70",
        "resource": "CPU",
        "result": "Weight 0.70 finished 115-125 across 240 games, or 47.92%, with zero failures. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-73",
        "kernel": "boltuzamaki/ptcg-gpu-pairwise-preference-v1",
        "profile": "bolt",
        "title": "Pairwise action preference model",
        "resource": "GPU",
        "result": "Completed on T4 with zero failures, but Alakazam holdout gain over the sequenced policy was only 0.55 percentage points. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-69C",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-chance-coupled-confirmation",
        "profile": "uza",
        "title": "Coupled chance fresh confirmation",
        "resource": "CPU",
        "result": "Fresh confirmation finished 253-227 across 480 games, or 52.71%, with a 48.24% Wilson lower bound, zero failures, and worst pairing p95 13.50 seconds. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-65C",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-progressive-confirmation",
        "profile": "uza",
        "title": "Progressive widening fresh confirmation",
        "resource": "CPU",
        "result": "Fresh confirmation finished 231-249 across 480 games, or 48.13%, with a 43.69% Wilson lower bound and zero failures. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-74",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-risk-sensitive-width-4",
        "profile": "uza",
        "title": "Risk-sensitive root width 4",
        "resource": "CPU",
        "result": "Finished 117-123 across 240 games, or 48.75%, with a 42.50% Wilson lower bound and zero failures. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-75",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-risk-sensitive-width-8",
        "profile": "uza",
        "title": "Risk-sensitive root width 8",
        "resource": "CPU",
        "result": "Finished 126-114 across 240 games, or 52.5%, with a 46.19% Wilson lower bound and zero failures. Fresh EXP-75C confirmation is running.",
        "decision": "conditional",
    },
    {
        "id": "EXP-76",
        "kernel": "boltuzamaki/ptcg-gpu-calibrated-uncertainty-v1",
        "profile": "bolt",
        "title": "Calibrated uncertainty gate",
        "resource": "GPU",
        "result": "Completed on T4, but holdout blended gain was only 0.71 percentage points at 87.9% coverage. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-77",
        "kernel": "boltuzamaki/ptcg-gpu-monotonic-policy-v1",
        "profile": "bolt",
        "title": "Monotonic rule constrained model",
        "resource": "GPU",
        "result": "Completed on T4 with zero lethal violations, but Alakazam holdout gain was only 0.63 percentage points. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-78",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-exploitability-matrix",
        "profile": "uza",
        "title": "Opponent exploitability matrix",
        "resource": "CPU",
        "result": "Chance-coupled led at 28-20, progressive finished 27-21, and the per-deck oracle reached 31-17 across the same 48 games. Zero failures; selector headroom is modest and exploratory.",
        "decision": "complete",
    },
    {
        "id": "EXP-75C",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-risk-width-8-confirmation",
        "profile": "uza",
        "title": "Risk-sensitive root width 8 confirmation",
        "resource": "CPU",
        "result": "Fresh 480-game confirmation of the 126-114 exploratory opening is running.",
        "decision": "active",
    },
    {
        "id": "EXP-79",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-seat-bias-correction",
        "profile": "uza",
        "title": "Seat bias correction",
        "resource": "CPU",
        "result": "Completed 384 games with zero failures. Absolute first-versus-second seat gaps were only 2.08 to 4.17 points across four original policies.",
        "decision": "complete",
    },
    {
        "id": "EXP-80",
        "kernel": "boltuzamaki/ptcg-gpu-matchup-conditioned-v1",
        "profile": "bolt",
        "title": "Matchup conditioned value model",
        "resource": "GPU",
        "result": "Completed on T4, but overall clean-room agreement gain was only 0.70 percentage points and no matchup gained one point. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-81",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-bayesian-deck-policy",
        "profile": "uza",
        "title": "Bayesian deck and policy tuning",
        "resource": "CPU",
        "result": "Thompson sampling is allocating games across five original legal decks and four original policies, followed by an internal holdout.",
        "decision": "active",
    },
    {
        "id": "EXP-82",
        "kernel": "boltuzamaki/ptcg-gpu-prioritized-failure-v1",
        "profile": "bolt",
        "title": "Prioritized rare failure replay",
        "resource": "GPU",
        "result": "Rare-decision accuracy improved from 81.25% to 87.5% while overall accuracy changed by -0.18 points. The learner gate passed; fresh EXP-82A arena screening is running.",
        "decision": "conditional",
    },
    {
        "id": "EXP-82A",
        "kernel": "boltuzamaki/ptcg-cpu-prioritized-model-screen",
        "profile": "bolt",
        "title": "Prioritized model live-field screen",
        "resource": "CPU",
        "result": "Finished 87-153 across 240 fresh games, or 36.25%, with a 30.43% Wilson lower bound and zero failures. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-83",
        "kernel": "boltuzamaki/ptcg-gpu-complexity-curriculum-v1",
        "profile": "bolt",
        "title": "Decision complexity curriculum",
        "resource": "GPU",
        "result": "Completed on T4, but hard-decision accuracy fell 1.16 points and overall accuracy fell 0.35 points against the shuffled control. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-84",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-within-match-adaptation",
        "profile": "uza",
        "title": "Within match opponent adaptation",
        "resource": "CPU",
        "result": "A policy map is being learned on frozen archetypes and compared with chance-coupled control on held-out live decks.",
        "decision": "active",
    },
    {
        "id": "EXP-85",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-opening-state-book",
        "profile": "uza",
        "title": "State cluster opening book",
        "resource": "CPU",
        "result": "An opening policy book is learning from our visible state clusters and game outcomes on eight decks, then testing on twelve disjoint decks.",
        "decision": "active",
    },
    {
        "id": "EXP-86",
        "kernel": "boltuzamaki/ptcg-gpu-evolutionary-weights-v1",
        "profile": "bolt",
        "title": "Evolutionary policy weights",
        "resource": "GPU",
        "result": "Completed on T4. Validation agreement gained only 0.22 points and Alakazam holdout agreement fell 0.08 points. Rejected.",
        "decision": "rejected",
    },
    {
        "id": "EXP-87",
        "kernel": "divyanshuboltuzamaki/ptcg-cpu-small-endgame-solver",
        "profile": "uza",
        "title": "Exact small endgame solver",
        "resource": "CPU",
        "result": "A bounded root-action-exhaustive endgame screen is running against chance-coupled control on twelve held-out decks.",
        "decision": "active",
    },
    {
        "id": "EXP-88",
        "kernel": "boltuzamaki/ptcg-cpu-resource-shadow-pricing",
        "profile": "bolt",
        "title": "Resource shadow pricing",
        "resource": "CPU",
        "result": "Three scarce-resource prices and a zero-price control are running on the same twelve held-out decks.",
        "decision": "active",
    },
    {
        "id": "EXP-89",
        "kernel": "boltuzamaki/ptcg-gpu-novelty-population-v1",
        "profile": "bolt",
        "title": "Novelty based policy population",
        "resource": "GPU",
        "result": "A diverse 12-member clean-room action population is being evolved on T4 to measure selector headroom.",
        "decision": "active",
    },
]


def profile_environment(profile: str) -> dict[str, str]:
    environment = dict(os.environ)
    if profile != "uza":
        environment.pop("KAGGLE_API_TOKEN", None)
        return environment
    token = environment.get("KAGGLE_UZA_TOKEN")
    if not token and sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                token = winreg.QueryValueEx(key, "KAGGLE_UZA_TOKEN")[0]
        except OSError:
            token = None
    if token:
        environment["KAGGLE_API_TOKEN"] = token
        environment.pop("KAGGLE_USERNAME", None)
        environment.pop("KAGGLE_KEY", None)
    return environment


def kernel_status(kernel: str, profile: str = "bolt") -> str:
    completed = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "status", kernel],
        cwd=ROOT,
        env=profile_environment(profile),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    text = f"{completed.stdout}\n{completed.stderr}"
    match = re.search(r'KernelWorkerStatus\.([A-Z_]+)', text)
    if match:
        return match.group(1).lower()
    if "404" in text or "not found" in text.lower():
        return "not_started"
    return "unknown"


def main() -> None:
    lanes = []
    for lane in LANES:
        item = dict(lane)
        item["status"] = kernel_status(
            item["kernel"], item.get("profile", "bolt")
        )
        item["url"] = f"https://www.kaggle.com/code/{item['kernel']}"
        lanes.append(item)

    counts = {
        "running": sum(item["status"] == "running" for item in lanes),
        "queued": sum(item["status"] in {"queued", "preparing"} for item in lanes),
        "complete": sum(item["status"] == "complete" for item in lanes),
        "failed": sum(item["status"] in {"error", "cancelled"} for item in lanes),
        "not_started": sum(item["status"] == "not_started" for item in lanes),
    }
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "compute_location": "Kaggle only, dual private profiles",
        "local_experiment_workers": "stopped",
        "submission_commands_allowed": False,
        "best_proven_candidate": {
            "name": "v3 shallow deterministic search",
            "public_score": 531.2,
            "submission_reference": 54744793,
        },
        "counts": counts,
        "lanes": lanes,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    SCRIPT_OUTPUT.write_text(
        "window.PTCG_KAGGLE_STATUS = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
