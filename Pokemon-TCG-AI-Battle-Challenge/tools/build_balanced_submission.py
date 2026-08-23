"""Build and audit the balanced no-Judge candidate without submitting it."""

from __future__ import annotations

import os

import build_no_relic_submission as builder

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

builder._ENTRY = os.path.join(_ROOT, "agent", "main_no_judge.py")
builder._OUT = os.path.join(_ROOT, "submission_balanced.tar.gz")
builder._REPORT = os.path.join(
    _ROOT, "scratchpad", "submission_balanced_audit.json"
)


if __name__ == "__main__":
    builder.main()
