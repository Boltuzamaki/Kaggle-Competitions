#!/usr/bin/env python
"""Push notebooks to Kaggle and submit them to the competition.

This is a *code competition* (``isKernelsSubmissionsOnly = true``): a submission
must reference a notebook version, so plain ``-f submission.csv`` is rejected.
The working flow is

    push  -> Kaggle runs the notebook -> submit that version

Usage
-----
    python scripts/kaggle_ops.py push   02_baseline_classical --gpu
    python scripts/kaggle_ops.py status 02_baseline_classical
    python scripts/kaggle_ops.py submit 02_baseline_classical -m "DoG baseline"

Kaggle rejects a code-competition submission without an explicit version number,
and the CLI has no command to list versions. ``push`` therefore records the
version it printed in ``.kaggle_versions.json`` and ``submit`` reuses it unless
``--version`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = REPO / "notebooks"
COMPETITION = "biohub-cell-tracking-during-development"
VERSION_STATE = REPO / ".kaggle_versions.json"
PROFILE_DIR = Path.home() / ".kaggle-profiles"

def use_profile(name: str | None) -> None:
    """Point the Kaggle CLI at ~/.kaggle-profiles/<name> for this process.

    Kaggle resolves credentials from KAGGLE_CONFIG_DIR, so switching accounts
    is just an env var - no need to move files around in ~/.kaggle.
    """
    if not name:
        return
    d = PROFILE_DIR / name
    if not (d / "credentials.json").is_file() and not (d / "kaggle.json").is_file():
        raise SystemExit(f"no credentials in {d}")
    os.environ["KAGGLE_CONFIG_DIR"] = str(d)

def _remember_version(kernel_id: str, version: int) -> None:
    state = json.loads(VERSION_STATE.read_text()) if VERSION_STATE.is_file() else {}
    state[kernel_id] = version
    VERSION_STATE.write_text(json.dumps(state, indent=2))

def _last_version(kernel_id: str) -> int | None:
    if not VERSION_STATE.is_file():
        return None
    return json.loads(VERSION_STATE.read_text()).get(kernel_id)

def kaggle_username() -> str:
    """The account the token *actually* authenticates as.

    The ``username`` field in credentials.json is just a label and can be
    stale: a profile here claimed ``divyanshuboltuzamaki`` while its OAuth
    token belonged to ``boltuzamaki``, so pushes failed with an opaque
    ``409 Conflict``. ``kaggle config view`` reports the resolved identity, so
    that wins; the file is only a fallback.
    """
    proc = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True)
    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            if line.strip().startswith("- username:"):
                user = line.split(":", 1)[1].strip()
                if user and user.lower() != "none":
                    return user

    cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    for path in [cfg / "credentials.json", cfg / "kaggle.json"]:
        if path.is_file():
            user = json.loads(path.read_text()).get("username")
            if user:
                return user
    raise SystemExit(f"Could not determine the Kaggle username (config dir: {cfg})")

def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)
    if check and proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode})")
    return proc

def resolve_notebook(name: str) -> Path:
    for candidate in [Path(name), NOTEBOOKS / name, NOTEBOOKS / f"{name}.ipynb"]:
        if candidate.is_file():
            return candidate
    raise SystemExit(f"notebook not found: {name} (looked in {NOTEBOOKS})")

def slug_for(nb_path: Path) -> str:
    """Kaggle slugs allow lowercase alphanumerics and dashes only."""
    stem = nb_path.stem.lower().replace("_", "-")
    return f"biohub-{stem}"

def cmd_push(args: argparse.Namespace) -> None:
    nb = resolve_notebook(args.notebook)
    user = kaggle_username()
    slug = args.slug or slug_for(nb)
    kernel_id = f"{user}/{slug}"

    # Kaggle derives the real URL slug from the *title*, not the id, and warns
    # (then silently uses the title) if they disagree. Deriving the title from
    # the slug keeps the pushed URL equal to `kernel_id`.
    meta = {
        "id": kernel_id,
        "title": args.title or slug.replace("-", " ").title(),
        "code_file": nb.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": bool(args.gpu),
        "enable_internet": bool(args.internet),
        "dataset_sources": args.dataset or [],
        "competition_sources": [COMPETITION],
        "kernel_sources": args.kernel_source or [],
    }
    if args.gpu:
        # Without this Kaggle may hand out a P100 (compute capability 6.0), and
        # its PyTorch build has no sm_60 kernels - every CUDA op dies with
        # "no kernel image is available for execution on the device". T4 is
        # sm_75 and is what every strong public notebook in this competition uses.
        meta["machine_shape"] = args.machine_shape

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(nb, tmp / nb.name)
        (tmp / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
        proc = run(["kaggle", "kernels", "push", "-p", str(tmp)])

    # Code-competition submissions require an explicit version number and the
    # CLI offers no way to list versions, so the number is captured from the
    # push output and remembered for `submit`.
    m = re.search(r"[Kk]ernel version (\d+) successfully pushed", proc.stdout or "")
    if m:
        _remember_version(kernel_id, int(m.group(1)))
        print(f"\npushed version {m.group(1)}")

    print(f"\n▶ https://www.kaggle.com/code/{user}/{slug}")
    print(f"  watch:  python scripts/kaggle_ops.py status {args.notebook}")
    print(f"  submit: python scripts/kaggle_ops.py submit {args.notebook} -m \"...\"")

def cmd_status(args: argparse.Namespace) -> None:
    nb = resolve_notebook(args.notebook)
    kernel_id = f"{kaggle_username()}/{args.slug or slug_for(nb)}"
    run(["kaggle", "kernels", "status", kernel_id], check=False)

def cmd_submit(args: argparse.Namespace) -> None:
    nb = resolve_notebook(args.notebook)
    kernel_id = f"{kaggle_username()}/{args.slug or slug_for(nb)}"

    version = args.version if args.version is not None else _last_version(kernel_id)
    if version is None:
        raise SystemExit(
            "No version known for this kernel. Pass --version N explicitly "
            "(the Kaggle CLI cannot list versions; `push` records the number it prints)."
        )
    cmd = ["kaggle", "competitions", "submit", "-c", COMPETITION,
           "-f", "submission.csv", "-k", kernel_id, "-m", args.message,
           "-v", str(version)]
    print(f"submitting version {version}")
    run(cmd)
    print("\nLeaderboard position:")
    run(["kaggle", "competitions", "submissions", "-c", COMPETITION], check=False)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("push", help="create or update the Kaggle notebook")
    p.add_argument("notebook")
    p.add_argument("--slug")
    p.add_argument("--title")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--machine-shape", default="NvidiaTeslaT4",
                   help="accelerator when --gpu is set (default NvidiaTeslaT4; "
                        "a P100 lacks sm_60 kernels in Kaggle's torch build)")
    p.add_argument("--internet", action="store_true")
    p.add_argument("--dataset", action="append", help="dataset source, repeatable")
    p.add_argument("--kernel-source", action="append")
    p.add_argument("--profile", help="account under ~/.kaggle-profiles/")
    p.set_defaults(func=cmd_push)

    p = sub.add_parser("status", help="show the last run's status")
    p.add_argument("notebook")
    p.add_argument("--slug")
    p.add_argument("--profile", help="account under ~/.kaggle-profiles/")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("submit", help="submit a notebook version to the leaderboard")
    p.add_argument("notebook")
    p.add_argument("--slug")
    p.add_argument("-v", "--version", type=int, help="default: the version recorded by the last push")
    p.add_argument("-m", "--message", required=True)
    p.add_argument("--profile", help="account under ~/.kaggle-profiles/")
    p.set_defaults(func=cmd_submit)

    args = ap.parse_args()
    use_profile(getattr(args, "profile", None))
    args.func(args)

if __name__ == "__main__":
    main()
