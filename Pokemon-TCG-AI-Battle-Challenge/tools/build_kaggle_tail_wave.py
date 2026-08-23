"""Build every remaining declared EXP-90 to EXP-95 remote notebook."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kaggle_remote" / "kernels"


def write_kernel(folder, metadata, cells, resource="CPU"):
    target = KERNELS / folder
    target.mkdir(parents=True, exist_ok=True)
    target.joinpath("kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    target.joinpath("notebook.ipynb").write_text(
        json.dumps(notebook(cells, resource), indent=1), encoding="utf-8"
    )


def metadata(identifier, title, owner, gpu=False, kernel_sources=None):
    return {
        "id": f"{owner}/{identifier}",
        "title": title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["research"],
        "dataset_sources": [
            (
                "boltuzamaki/ptcg-private-cpu-assets-bolt-v1"
                if owner == "boltuzamaki"
                else "divyanshuboltuzamaki/ptcg-research-assets-uza-v1"
            ),
            "kiyotah/cg-lib",
        ],
        "kernel_sources": kernel_sources or [],
        "competition_sources": [],
        "model_sources": [],
    }


def gpu_run(source_name, source, output, report):
    return f"""\
from pathlib import Path
import json, subprocess, sys, torch
if not torch.cuda.is_available():
    raise RuntimeError("GPU requested but CUDA is unavailable")
print("gpu", torch.cuda.get_device_name(0))
trainer = Path("/kaggle/working/ptcg/automation/jobs/{source_name}")
trainer.write_text({source!r}, encoding="utf-8")
completed = subprocess.run(
    [sys.executable, str(trainer), "--out", {output!r}], check=False
)
if completed.returncode:
    raise RuntimeError("{source_name} failed: " + str(completed.returncode))
payload = json.loads(Path({(output + "/" + report)!r}).read_text(encoding="utf-8"))
Path("/kaggle/working/final_summary.json").write_text(
    json.dumps(payload, indent=2), encoding="utf-8"
)
print(json.dumps(payload, indent=2))
"""


def main():
    distributional = (
        ROOT / "automation" / "jobs" / "train_distributional_value_gpu.py"
    ).read_text(encoding="utf-8")
    synthetic = (
        ROOT / "automation" / "jobs" / "train_synthetic_rare_gpu.py"
    ).read_text(encoding="utf-8")
    source_kernel = ["boltuzamaki/ptcg-gpu-pairwise-preference-v1"]
    write_kernel(
        "gpu_distributional_value_bolt",
        metadata(
            "ptcg-gpu-distributional-value-v1",
            "PTCG GPU distributional value V1",
            "boltuzamaki",
            True,
            source_kernel,
        ),
        [
            ("markdown", "# EXP-90 Distributional value prediction"),
            ("code", SETUP),
            (
                "code",
                gpu_run(
                    "train_distributional_value_gpu.py",
                    distributional,
                    "/kaggle/working/exp-90-distributional",
                    "distributional_value_report.json",
                ),
            ),
        ],
        "GPU",
    )
    write_kernel(
        "gpu_synthetic_rare_states_bolt",
        metadata(
            "ptcg-gpu-synthetic-rare-states-v1",
            "PTCG GPU synthetic rare states V1",
            "boltuzamaki",
            True,
            source_kernel,
        ),
        [
            ("markdown", "# EXP-91 Synthetic rare state generator"),
            ("code", SETUP),
            (
                "code",
                gpu_run(
                    "train_synthetic_rare_gpu.py",
                    synthetic,
                    "/kaggle/working/exp-91-synthetic",
                    "synthetic_rare_report.json",
                ),
            ),
        ],
        "GPU",
    )

    hidden_run = """\
from pathlib import Path
import json, subprocess, sys
out = "/kaggle/working/hidden_card_stress.json"
command = [
    sys.executable,
    "automation/jobs/strategy_fusion_audit.py",
    "--games", "16",
    "--samples", "32",
    "--max-decisions", "96",
    "--out", out,
]
completed = subprocess.run(command, check=False)
if completed.returncode:
    raise RuntimeError("hidden-card stress failed")
report = json.loads(Path(out).read_text(encoding="utf-8"))
report.update({
    "experiment": "EXP-92",
    "strategy_rank": 61,
    "strategy": "Adversarial hidden card stress test",
    "public_policy_code_used": False,
    "public_policy_actions_used_as_labels": False,
})
Path("/kaggle/working/final_summary.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))
"""
    write_kernel(
        "cpu_hidden_card_stress",
        metadata(
            "ptcg-cpu-hidden-card-stress",
            "PTCG CPU hidden card stress",
            "divyanshuboltuzamaki",
        ),
        [
            ("markdown", "# EXP-92 Adversarial hidden card stress"),
            ("code", SETUP),
            ("code", hidden_run),
        ],
    )

    metamorphic_run = """\
from copy import deepcopy
from pathlib import Path
import json
from kaggle_environments import make
import archaludon_policy, arena, search_agent
from decks import ALAKAZAM
checks = 0
mismatches = []
deck = archaludon_policy.ARCHALUDON_DECK
def checked(obs):
    global checks
    base = archaludon_policy.archaludon_sequenced_agent(obs, deck)
    select = obs.get("select") or {}
    options = select.get("option") or []
    if len(options) > 1 and (select.get("maxCount") or 1) == 1 and len(base) == 1:
        transformed = deepcopy(obs)
        transformed["select"]["option"] = list(reversed(options))
        other = archaludon_policy.archaludon_sequenced_agent(transformed, deck)
        checks += 1
        if len(other) != 1 or options[base[0]] != list(reversed(options))[other[0]]:
            mismatches.append({"turn": (obs.get("current") or {}).get("turn"), "options": len(options)})
    return base
opponent = arena.bind_deck(search_agent.agent, ALAKAZAM, search_module=search_agent)
pairing = arena.run_round_robin(make, [
    {"name":"metamorphic","agent":checked,"deck":deck},
    {"name":"v3-alakazam","agent":opponent,"deck":ALAKAZAM},
], 24)[0]
report = {
    "experiment":"EXP-93","strategy_rank":62,
    "strategy":"Metamorphic policy tests","checks":checks,
    "semantic_mismatches":len(mismatches),
    "mismatch_rate":len(mismatches)/max(checks,1),
    "examples":mismatches[:50],"pairing":arena.pairing_row(pairing),
    "public_policy_code_used":False,
    "public_policy_actions_used_as_labels":False,
    "submission_performed":False,
}
Path("/kaggle/working/final_summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
"""
    write_kernel(
        "cpu_metamorphic_tests",
        metadata(
            "ptcg-cpu-metamorphic-policy-tests",
            "PTCG CPU metamorphic policy tests",
            "divyanshuboltuzamaki",
        ),
        [
            ("markdown", "# EXP-93 Metamorphic policy tests"),
            ("code", SETUP),
            ("code", metamorphic_run),
        ],
    )

    fuzz_run = """\
from copy import deepcopy
from pathlib import Path
import json
from kaggle_environments import make
import archaludon_policy, arena, search_agent
from decks import MEGA_LUCARIO
checks = 0
errors = []
deck = archaludon_policy.ARCHALUDON_DECK
def checked(obs):
    global checks
    action = archaludon_policy.archaludon_sequenced_agent(obs, deck)
    if obs.get("select") is not None and checks < 2000:
        mutations = []
        a = deepcopy(obs); a.pop("remainingOverageTime", None); mutations.append(("missing-time",a))
        b = deepcopy(obs); (b.get("current") or {}).pop("logs", None); mutations.append(("missing-logs",b))
        c = deepcopy(obs); c["remainingOverageTime"] = 0; mutations.append(("zero-time",c))
        for name, mutated in mutations:
            checks += 1
            try:
                archaludon_policy.archaludon_sequenced_agent(mutated, deck)
            except Exception as exc:
                errors.append({"mutation":name,"error":type(exc).__name__,"message":str(exc)})
    return action
opponent = arena.bind_deck(search_agent.agent, MEGA_LUCARIO, search_module=search_agent)
pairing = arena.run_round_robin(make, [
    {"name":"fuzz-checked","agent":checked,"deck":deck},
    {"name":"v3-lucario","agent":opponent,"deck":MEGA_LUCARIO},
], 32)[0]
report = {
    "experiment":"EXP-94","strategy_rank":63,"strategy":"Game state fuzzing",
    "mutations_checked":checks,"exceptions":len(errors),"examples":errors[:50],
    "pairing":arena.pairing_row(pairing),
    "public_policy_code_used":False,
    "public_policy_actions_used_as_labels":False,
    "submission_performed":False,
}
Path("/kaggle/working/final_summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
"""
    write_kernel(
        "cpu_game_state_fuzzing",
        metadata(
            "ptcg-cpu-game-state-fuzzing",
            "PTCG CPU game state fuzzing",
            "divyanshuboltuzamaki",
        ),
        [
            ("markdown", "# EXP-94 Game state fuzzing"),
            ("code", SETUP),
            ("code", fuzz_run),
        ],
    )

    summary_rows = []
    for path in ROOT.glob("scratchpad/kaggle_exp*/final_summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        pairings = payload.get("pairings") or []
        if not pairings:
            for key in ("treatment", "book_result", "adaptive_live"):
                pairings.extend((payload.get(key) or {}).get("pairings") or [])
        for row in pairings:
            if row.get("games") and row.get("a_wins") is not None:
                summary_rows.append(
                    {
                        "experiment": payload.get("experiment", path.parent.name),
                        "deck": row.get("live_deck_signature", row.get("competitor_b")),
                        "wins": row["a_wins"],
                        "games": row["games"],
                    }
                )
    hierarchical_run = f"""\
from collections import defaultdict
from pathlib import Path
import json, math
rows = {summary_rows!r}
groups = defaultdict(lambda: {{"wins":0,"games":0,"decks":set()}})
for row in rows:
    group = groups[row["experiment"]]
    group["wins"] += row["wins"]
    group["games"] += row["games"]
    group["decks"].add(row["deck"])
alpha0, beta0 = 8.0, 8.0
ranking = []
for name, group in groups.items():
    alpha = alpha0 + group["wins"]
    beta = beta0 + group["games"] - group["wins"]
    mean = alpha / (alpha + beta)
    variance = alpha*beta/((alpha+beta)**2*(alpha+beta+1))
    ranking.append({{
        "candidate":name,"games":group["games"],"decks":len(group["decks"]),
        "posterior_mean":mean,
        "approx_lower95":max(0.0,mean-1.96*math.sqrt(variance)),
    }})
ranking.sort(key=lambda row:row["approx_lower95"],reverse=True)
report = {{
    "experiment":"EXP-95","strategy_rank":67,
    "strategy":"Hierarchical strength model","input_pairings":len(rows),
    "candidates":len(ranking),"ranking":ranking,
    "promotion_ready":False,
    "interpretation":"Retrospective shrinkage ranking only; selection-biased experiments cannot promote.",
    "public_policy_code_used":False,
    "public_policy_actions_used_as_labels":False,
    "submission_performed":False,
}}
Path("/kaggle/working/final_summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
"""
    write_kernel(
        "cpu_hierarchical_strength_bolt",
        metadata(
            "ptcg-cpu-hierarchical-strength",
            "PTCG CPU hierarchical strength",
            "boltuzamaki",
        ),
        [
            ("markdown", "# EXP-95 Hierarchical strength model"),
            ("code", hierarchical_run),
        ],
    )
    print("built EXP-90 through EXP-95")


if __name__ == "__main__":
    main()
