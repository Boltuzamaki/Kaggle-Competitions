"""Build and audit the original Grimmsnarl submission archive. Never submits.

Mirrors the audit discipline of the earlier builders: an allowlisted member set,
a source smoke that proves the deck is engine-legal, an isolated extract-and-import
smoke, a forbidden-member scan (no public policy code, no model weights), and a
SHA-256 of the exact bytes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import tempfile


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_DIR = os.path.join(ROOT, "agent")
ENTRY = os.path.join(AGENT_DIR, "main_grimmsnarl.py")
CG_DIR = os.path.join(AGENT_DIR, "cg")
OUT = os.path.join(ROOT, "submission_grimmsnarl.tar.gz")
REPORT = os.path.join(ROOT, "scratchpad", "submission_grimmsnarl_audit.json")
DEPENDENCIES = (
    "grimmsnarl_policy.py",
    "domain_policy.py",
    "meta_decks.py",
)


def engine_members():
    members = []
    for root, dirs, files in os.walk(CG_DIR):
        dirs[:] = [name for name in dirs if name != "__pycache__"]
        for name in files:
            if name.endswith(".pyc"):
                continue
            source = os.path.join(root, name)
            archive_name = os.path.relpath(source, AGENT_DIR).replace(os.sep, "/")
            members.append((source, archive_name))
    return sorted(members, key=lambda item: item[1])


def source_smoke():
    sys.path.insert(0, AGENT_DIR)
    spec = importlib.util.spec_from_file_location("main_grimmsnarl", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    deck = module.agent({"select": None})
    if len(module.DECK) != 60 or len(deck) != 60:
        raise SystemExit("source deck-selection smoke failed")
    from cg.game import battle_finish, battle_start

    _, start = battle_start(module.DECK, module.DECK)
    try:
        if start.errorPlayer >= 0:
            raise SystemExit(f"deck legality error type {start.errorType}")
    finally:
        battle_finish()
    return module


def isolated_smoke(archive_path):
    with tempfile.TemporaryDirectory(prefix="ptcg_grimmsnarl_") as directory:
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(directory)
        code = (
            "import json,main; "
            "deck=main.agent({'select':None}); "
            "assert len(main.DECK)==60 and len(deck)==60; "
            "print(json.dumps({'deck':len(deck),'module':main.__file__}))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode:
            raise SystemExit(
                "isolated archive smoke failed:\n" + result.stdout + result.stderr
            )
        return result.stdout.strip()


def main():
    required = [ENTRY, CG_DIR, *(os.path.join(AGENT_DIR, name) for name in DEPENDENCIES)]
    for path in required:
        if not os.path.exists(path):
            raise SystemExit(f"missing required path: {path}")

    module = source_smoke()
    members = [(ENTRY, "main.py")]
    members.extend((os.path.join(AGENT_DIR, name), name) for name in DEPENDENCIES)
    members.extend(engine_members())
    with tarfile.open(OUT, "w:gz") as archive:
        for source, archive_name in members:
            archive.add(source, arcname=archive_name)

    with tarfile.open(OUT, "r:gz") as archive:
        names = sorted(archive.getnames())
    expected = sorted(name for _, name in members)
    if names != expected:
        raise SystemExit("archive membership differs from the allowlist")
    forbidden = ("public_agents", "rlnet", "model.pth", "candidate_agents",
                 "context_ranker", "romanrozen", "top_rankers")
    bad = [name for name in names if any(token in name.lower() for token in forbidden)]
    if bad:
        raise SystemExit(f"forbidden archive members: {bad}")

    isolated = isolated_smoke(OUT)
    with open(OUT, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    report = {
        "archive": OUT,
        "sha256": digest,
        "size_bytes": os.path.getsize(OUT),
        "deck_cards": len(module.DECK),
        "deck_archetype": "Marnie's Grimmsnarl ex (mined from Aug-03 top episodes)",
        "members": names,
        "member_count": len(names),
        "forbidden_members": bad,
        "source_smoke": "pass",
        "isolated_smoke": isolated,
        "public_policy_code_included": False,
        "submission_performed": False,
    }
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    print("Archive built and validated. No submission was performed.")


if __name__ == "__main__":
    main()
