"""Build and audit the promoted original-only no-Relicanth archive.

This command builds and validates locally. It never submits to Kaggle.
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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AGENT_DIR = os.path.join(_ROOT, "agent")
_ENTRY = os.path.join(_AGENT_DIR, "main_no_relic.py")
_POLICY = os.path.join(_AGENT_DIR, "archaludon_policy.py")
_CG_DIR = os.path.join(_AGENT_DIR, "cg")
_OUT = os.path.join(_ROOT, "submission_no_relic.tar.gz")
_REPORT = os.path.join(_ROOT, "scratchpad", "submission_no_relic_audit.json")


def _engine_members():
    members = []
    for root, dirs, files in os.walk(_CG_DIR):
        dirs[:] = [name for name in dirs if name != "__pycache__"]
        for name in files:
            if name.endswith(".pyc"):
                continue
            source = os.path.join(root, name)
            archive_name = os.path.relpath(source, _AGENT_DIR).replace(os.sep, "/")
            members.append((source, archive_name))
    return sorted(members, key=lambda item: item[1])


def _source_smoke():
    if _AGENT_DIR not in sys.path:
        sys.path.insert(0, _AGENT_DIR)
    spec = importlib.util.spec_from_file_location("main_no_relic", _ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if len(module.DECK) != 60:
        raise SystemExit(f"source deck has {len(module.DECK)} cards")
    if len(module.agent({"select": None})) != 60:
        raise SystemExit("source deck-selection smoke failed")
    from cg.game import battle_finish, battle_start

    _, start = battle_start(module.DECK, module.DECK)
    try:
        if start.errorPlayer >= 0:
            raise SystemExit(f"deck legality error type {start.errorType}")
    finally:
        battle_finish()
    return module.DECK


def _isolated_smoke(archive_path):
    with tempfile.TemporaryDirectory(prefix="ptcg_no_relic_") as directory:
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(directory)
        code = (
            "import json,main; "
            "d=main.agent({'select':None}); "
            "assert len(main.DECK)==60 and len(d)==60; "
            "print(json.dumps({'deck':len(d),'module':main.__file__}))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise SystemExit(
                "isolated archive smoke failed:\n" + result.stdout + result.stderr
            )
        return result.stdout.strip()


def main():
    for path in (_ENTRY, _POLICY, _CG_DIR):
        if not os.path.exists(path):
            raise SystemExit(f"missing required path: {path}")

    deck = _source_smoke()
    members = [(_ENTRY, "main.py"), (_POLICY, "archaludon_policy.py")]
    members.extend(_engine_members())

    with tarfile.open(_OUT, "w:gz") as archive:
        for source, archive_name in members:
            archive.add(source, arcname=archive_name)

    with tarfile.open(_OUT, "r:gz") as archive:
        names = sorted(archive.getnames())
    expected = sorted(name for _, name in members)
    if names != expected:
        raise SystemExit("archive membership differs from the allowlist")
    forbidden = ("public", "rlnet", "model.pth", "hybrid", "candidate_agents", "context_ranker")
    bad = [name for name in names if any(token in name.lower() for token in forbidden)]
    if bad:
        raise SystemExit(f"forbidden archive members: {bad}")

    isolated = _isolated_smoke(_OUT)
    digest = hashlib.sha256(open(_OUT, "rb").read()).hexdigest()
    report = {
        "archive": _OUT,
        "sha256": digest,
        "size_bytes": os.path.getsize(_OUT),
        "deck_cards": len(deck),
        "members": names,
        "member_count": len(names),
        "forbidden_members": bad,
        "source_smoke": "pass",
        "isolated_smoke": isolated,
        "submission_performed": False,
    }
    with open(_REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    print("Archive built and validated locally. No submission was performed.")


if __name__ == "__main__":
    main()
