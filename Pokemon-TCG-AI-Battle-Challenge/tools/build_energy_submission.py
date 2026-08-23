"""Build and audit the EXP-29 Energy-opportunity candidate without submitting."""

from __future__ import annotations

import os

import build_no_relic_submission as builder

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

builder._ENTRY = os.path.join(_ROOT, "agent", "main_energy_opportunity.py")
builder._OUT = os.path.join(_ROOT, "submission_energy_opportunity.tar.gz")
builder._REPORT = os.path.join(
    _ROOT, "scratchpad", "submission_energy_opportunity_audit.json"
)


if __name__ == "__main__":
    builder.main()
