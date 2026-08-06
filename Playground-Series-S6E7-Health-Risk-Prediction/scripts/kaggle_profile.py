"""Safely switch the Kaggle CLI OAuth credential between local profiles.

Credential contents are never printed. The selected profile is copied
atomically to ~/.kaggle/credentials.json with owner-only permissions.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


HOME = Path.home()
ACTIVE = HOME / ".kaggle" / "credentials.json"
PROFILES = HOME / ".kaggle-profiles"


def username(path: Path) -> str:
    return str(json.loads(path.read_text(encoding="utf-8"))["username"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    args = parser.parse_args()
    source = PROFILES / args.profile / "credentials.json"
    if not source.is_file():
        raise SystemExit(f"Unknown Kaggle profile: {args.profile}")
    expected = username(source)
    ACTIVE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".credentials.", dir=ACTIVE.parent)
    try:
        with os.fdopen(fd, "wb") as target, source.open("rb") as src:
            target.write(src.read())
            target.flush()
            os.fsync(target.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, ACTIVE)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    actual = username(ACTIVE)
    if actual != expected:
        raise RuntimeError(f"Profile switch verification failed: {actual} != {expected}")
    print(actual)


if __name__ == "__main__":
    main()
