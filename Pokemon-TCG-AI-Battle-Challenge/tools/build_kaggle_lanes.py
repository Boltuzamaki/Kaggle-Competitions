"""Build private Kaggle research assets and CPU/GPU experiment notebooks."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
REMOTE = ROOT / "kaggle_remote"
ASSETS = REMOTE / "ptcg_research_assets_v1"
PROJECT = ASSETS / "project"
KERNELS = REMOTE / "kernels"


PROJECT_FILES = (
    "agent/archaludon_policy.py",
    "agent/beam_archaludon.py",
    "agent/candidate_agents.py",
    "agent/context_ranker.py",
    "agent/deck_variants.py",
    "agent/domain_policy.py",
    "agent/domain_policy_active_threat.py",
    "agent/domain_policy_no_retreat.py",
    "agent/hybrid_agent.py",
    "agent/information_search.py",
    "agent/opponent_beliefs.py",
    "agent/rlnet.py",
    "agent/search_agent.py",
    "tools/arena.py",
    "agent/meta_decks.py",
    "tools/train_rl.py",
    "automation/jobs/live_deck_gauntlet.py",
    "automation/jobs/sequential_probability_test.py",
    "automation/jobs/decision_invariant_check.py",
    "automation/jobs/strategy_fusion_audit.py",
    "automation/jobs/train_beam_distillation_gpu.py",
    "automation/jobs/evaluate_flat_static.py",
    "automation/jobs/combine_flat_static.py",
    "references/top_rankers/decks.py",
    "models/research_snapshots/exp42_iter30_v3only.pth",
    "models/research_snapshots/exp42_iter30_v3only.pth.arch.json",
    "models/research_snapshots/exp42_iter30_v3only.pth.ckpt",
)


def deck_from_replay(payload: dict, seat: int) -> list[int]:
    try:
        deck = payload["steps"][0][0]["visualize"][0]["action"][seat]
        if len(deck) == 60 and all(isinstance(card, int) for card in deck):
            return list(deck)
    except (KeyError, IndexError, TypeError):
        pass
    return []


def deck_signature(deck: list[int]) -> str:
    encoded = ",".join(map(str, sorted(deck))).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()[:12]


def build_deck_catalog() -> dict:
    records: dict[str, dict] = {}
    raw = ROOT / "data" / "live_episodes" / "raw"
    for path in sorted(raw.glob("**/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        names = payload.get("info", {}).get("TeamNames") or ["unknown", "unknown"]
        for seat in range(2):
            deck = deck_from_replay(payload, seat)
            if len(deck) != 60:
                continue
            signature = deck_signature(deck)
            record = records.setdefault(
                signature,
                {
                    "signature": signature,
                    "deck": deck,
                    "appearances": 0,
                    "agent_names": Counter(),
                },
            )
            record["appearances"] += 1
            record["agent_names"][str(names[seat])] += 1
    decks = sorted(
        records.values(),
        key=lambda item: (-item["appearances"], item["signature"]),
    )
    for record in decks:
        record["agent_names"] = dict(record["agent_names"])
    return {
        "decks": decks,
        "contains_public_policy_actions": False,
        "source": "visible deck lists from competition replay files",
    }


def notebook(cells: list[tuple[str, str]], accelerator: str | None = None) -> dict:
    result = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    if accelerator:
        result["metadata"]["accelerator"] = accelerator
    for cell_type, source in cells:
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if cell_type == "code":
            cell.update({"execution_count": None, "outputs": []})
        result["cells"].append(cell)
    return result


SETUP = """\
from pathlib import Path
import glob, json, os, shutil, sys

matches = glob.glob("/kaggle/input/**/tools/train_rl.py", recursive=True)
if not matches:
    archives = glob.glob("/kaggle/input/**/project.zip", recursive=True)
    if not archives:
        raise RuntimeError("PTCG research asset dataset is not attached")
    extract_root = Path("/kaggle/working/ptcg_asset_extract")
    extract_root.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(archives[0], extract_root)
    matches = glob.glob(str(extract_root / "**/tools/train_rl.py"), recursive=True)
    if not matches:
        raise RuntimeError(f"project.zip extracted without tools/train_rl.py: {archives[0]}")
asset_project = Path(matches[0]).parents[1]
work_project = Path("/kaggle/working/ptcg")
shutil.copytree(asset_project, work_project, dirs_exist_ok=True)
cg_paths = glob.glob("/kaggle/input/**/cg-lib", recursive=True)
project_paths = [
    work_project,
    work_project / "agent",
    work_project / "tools",
    work_project / "references" / "top_rankers",
]
for candidate in cg_paths:
    sys.path.insert(0, candidate)
for candidate in project_paths:
    sys.path.insert(0, str(candidate))
inherited_paths = [*cg_paths, *(str(path) for path in project_paths)]
if os.environ.get("PYTHONPATH"):
    inherited_paths.append(os.environ["PYTHONPATH"])
os.environ["PYTHONPATH"] = os.pathsep.join(inherited_paths)
os.chdir(work_project)
from cg.api import SelectContext, to_observation_class
print("project", work_project)
print("cg paths", [path for path in sys.path if path.endswith("cg-lib")])
print("cg api smoke", SelectContext.MAIN, callable(to_observation_class))
"""


def metadata(kernel_id: str, title: str, gpu: bool) -> dict:
    result = {
        "id": f"boltuzamaki/{kernel_id}",
        "title": title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["pokemon-tcg", "research", "clean-room"],
        "dataset_sources": [
            "boltuzamaki/ptcg-research-assets-v1",
            "kiyotah/cg-lib",
        ],
        "kernel_sources": [],
        "competition_sources": ["pokemon-tcg-ai-battle"],
        "model_sources": [],
    }
    if not gpu:
        result["machine_shape"] = "None"
    return result


def write_kernel(
    folder: str,
    kernel_id: str,
    title: str,
    gpu: bool,
    cells: list[tuple[str, str]],
) -> None:
    target = KERNELS / folder
    target.mkdir(parents=True, exist_ok=True)
    (target / "kernel-metadata.json").write_text(
        json.dumps(metadata(kernel_id, title, gpu), indent=2),
        encoding="utf-8",
    )
    (target / "notebook.ipynb").write_text(
        json.dumps(notebook(cells, "GPU" if gpu else None), indent=1),
        encoding="utf-8",
    )


def main() -> None:
    PROJECT.mkdir(parents=True, exist_ok=True)
    project_files = set(PROJECT_FILES)
    # arena.py registers many optional competitors at import time. Keep the
    # research bundle self-consistent as new top-level policy modules are added.
    for folder in ("agent", "tools"):
        project_files.update(
            str(path.relative_to(ROOT)) for path in (ROOT / folder).glob("*.py")
        )
    for relative in sorted(project_files):
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = PROJECT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    catalog = build_deck_catalog()
    catalog_path = PROJECT / "data" / "live_decks.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (ASSETS / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "PTCG Research Assets V1",
                "id": "boltuzamaki/ptcg-research-assets-v1",
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    gpu_code = """\
from pathlib import Path
import hashlib, json, shutil
import torch
import train_rl

source = Path("models/research_snapshots/exp42_iter30_v3only.pth")
output = Path("/kaggle/working/ptcg_gpu_v3only_iter70.pth")
for suffix in ("", ".arch.json", ".ckpt"):
    shutil.copy2(str(source) + suffix, str(output) + suffix)
arch = json.loads(Path(str(output) + ".arch.json").read_text())
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
train_rl.train(
    iterations=70,
    games=80,
    search=24,
    out_path=str(output),
    deck=train_rl.SAMPLE_DECK,
    resume=True,
    arch=arch,
    replay_capacity=50000,
    league_mix=1.0,
    league_opponents=("v3",),
)
summary = {
    "experiment": "KAGGLE-GPU-01",
    "model": str(output),
    "iterations_target": 70,
    "games_per_iteration": 80,
    "search": 24,
    "league": ["v3"],
    "cuda": torch.cuda.is_available(),
    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    "public_policy_actions_used_as_labels": False,
    "submission_performed": False,
}
Path("/kaggle/working/kaggle_gpu_01_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
"""
    write_kernel(
        "gpu_v3_continue",
        "ptcg-gpu-v3-only-continuation-v1",
        "PTCG GPU V3 Only Continuation V1",
        True,
        [
            ("markdown", "# PTCG GPU hard-opponent continuation\n\nResearch only. No submission command."),
            ("code", SETUP),
            ("code", gpu_code),
        ],
    )

    live_40 = """\
import subprocess, sys
subprocess.run([
    sys.executable, "automation/jobs/live_deck_gauntlet.py",
    "--candidate", "flatmc", "--flat-root-width", "8",
    "--flat-iterations", "40", "--games-per-deck", "24",
    "--max-decks", "20", "--deck-catalog", "data/live_decks.json",
    "--out", "/kaggle/working/kaggle_cpu_live_w8_i40",
], check=True)
"""
    write_kernel(
        "cpu_live_w8_i40",
        "ptcg-cpu-live-flat-w8-i40-v1",
        "PTCG CPU Live Flat W8 I40 V1",
        False,
        [
            ("markdown", "# PTCG live-field flat-root confirmation\n\nOriginal candidate against frozen local v3."),
            ("code", SETUP),
            ("code", live_40),
        ],
    )

    live_16 = """\
import subprocess, sys
subprocess.run([
    sys.executable, "automation/jobs/live_deck_gauntlet.py",
    "--candidate", "flatmc", "--flat-root-width", "8",
    "--flat-iterations", "16", "--games-per-deck", "24",
    "--max-decks", "20", "--deck-catalog", "data/live_decks.json",
    "--out", "/kaggle/working/kaggle_cpu_live_w8_i16",
], check=True)
"""
    write_kernel(
        "cpu_live_w8_i16",
        "ptcg-cpu-live-flat-w8-i16-v1",
        "PTCG CPU Live Flat W8 I16 V1",
        False,
        [
            ("markdown", "# PTCG lower-cost flat-root confirmation\n\nTests whether the cheaper search preserves field strength."),
            ("code", SETUP),
            ("code", live_16),
        ],
    )

    static_code = """\
import json, subprocess, sys
from pathlib import Path

reports = []
for opponent in ("abomasnow", "lucario", "alakazam", "dragapult"):
    output = f"/kaggle/working/kaggle_cpu_static_{opponent}.json"
    subprocess.run([
        sys.executable, "automation/jobs/evaluate_flat_static.py",
        "--opponent", opponent, "--games", "80",
        "--root-width", "8", "--iterations", "40", "--out", output,
    ], check=True)
    reports.append(output)
subprocess.run([
    sys.executable, "automation/jobs/combine_flat_static.py",
    *reports, "--out", "/kaggle/working/kaggle_cpu_static_decision.json",
], check=True)
"""
    write_kernel(
        "cpu_static_w8_i40",
        "ptcg-cpu-static-flat-w8-i40-v1",
        "PTCG CPU Static Flat W8 I40 V1",
        False,
        [
            ("markdown", "# PTCG four-archetype decisive flat-root test\n\nBalanced seats, zero automatic submissions."),
            ("code", SETUP),
            ("code", static_code),
        ],
    )

    manifest = {
        "asset_files": list(PROJECT_FILES) + ["data/live_decks.json"],
        "live_decks": len(catalog["decks"]),
        "kernels": [
            "boltuzamaki/ptcg-gpu-v3-only-continuation-v1",
            "boltuzamaki/ptcg-cpu-live-flat-w8-i40-v1",
            "boltuzamaki/ptcg-cpu-live-flat-w8-i16-v1",
            "boltuzamaki/ptcg-cpu-static-flat-w8-i40-v1",
        ],
        "public_agent_code_in_assets": False,
        "submission_commands_present": False,
    }
    (REMOTE / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
