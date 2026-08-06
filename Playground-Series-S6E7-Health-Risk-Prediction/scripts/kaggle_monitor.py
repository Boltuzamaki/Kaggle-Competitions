"""Poll all health-risk Kaggle kernels and collect completed outputs.

Designed for cron. It switches secured local OAuth profiles atomically, records
one JSONL event per kernel, downloads completed artifacts, and always restores
the primary profile. Repair and experiment-selection decisions are performed by
Codex from these authoritative logs/artifacts.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "artifacts" / "logs" / "kaggle_monitor.jsonl"
DOWNLOADS = ROOT / "kaggle_kernels" / "monitor_downloads"
SWITCH = ROOT / "scripts" / "kaggle_profile.py"
PRIMARY = "boltuzamaki"

KERNELS = {
    "boltuzamaki": [
        "health-risk-rule-ebm-cpu",
        "health-risk-ftt-balanced-seed-2027-gpu",
        "health-risk-ftt-balanced-seed-4242-gpu",
        "health-risk-realmlp-capacity-9001-gpu",
        "health-risk-tabpfn3-gpu",
        "health-risk-autogluon-cpu",
    ],
    "divyanshuboltuzamaki": [
        "health-risk-optuna-xgb-gpu",
        "health-risk-te-catboost-cpu",
        "health-risk-realmlp-seed-4242-gpu",
    ],
}


def run(*args: str, timeout: int = 45) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = (exc.stderr or "")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return subprocess.CompletedProcess(args, 124, "", f"{stderr}\nTIMEOUT")


def switch(profile: str) -> None:
    result = run("python3", str(SWITCH), profile)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)


def append(event: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()
    try:
        for owner, slugs in KERNELS.items():
            switch(owner)
            for slug in slugs:
                ref = f"{owner}/{slug}"
                status_result = run("kaggle", "kernels", "status", ref)
                text = (status_result.stdout + status_result.stderr).strip()
                status = next(
                    (
                        value
                        for value in ("COMPLETE", "ERROR", "RUNNING", "QUEUED")
                        if value in text
                    ),
                    "UNKNOWN",
                )
                event = {
                    "time": now,
                    "owner": owner,
                    "slug": slug,
                    "status": status,
                    "status_exit_code": status_result.returncode,
                }
                if status == "COMPLETE":
                    destination = DOWNLOADS / owner / slug
                    destination.mkdir(parents=True, exist_ok=True)
                    existing = [
                        p for p in destination.iterdir()
                        if p.is_file() and p.suffix in {".csv", ".json"}
                    ]
                    csv_files = [p for p in existing if p.suffix == ".csv"]
                    summary = destination / "training_summary.json"
                    cache_complete = (
                        summary.exists()
                        and summary.stat().st_size > 0
                        and len(csv_files) >= 2
                        and all(p.stat().st_size > 0 for p in csv_files)
                    )
                    if cache_complete:
                        event["output_exit_code"] = 0
                        event["output_cached"] = True
                    else:
                        output_result = run(
                            "kaggle", "kernels", "output", ref,
                            "-p", str(destination), "--force", timeout=240,
                        )
                        event["output_exit_code"] = output_result.returncode
                        if output_result.returncode:
                            event["output_error"] = output_result.stderr[-1000:]
                    event["files"] = sorted(
                        p.name for p in destination.iterdir() if p.is_file()
                    )
                    summary = destination / "training_summary.json"
                    if summary.exists() and summary.stat().st_size:
                        try:
                            event["training_summary"] = json.loads(
                                summary.read_text(encoding="utf-8")
                            )
                        except (json.JSONDecodeError, OSError) as exc:
                            event["summary_error"] = str(exc)
                elif status == "ERROR":
                    destination = DOWNLOADS / owner / slug
                    destination.mkdir(parents=True, exist_ok=True)
                    output_result = run(
                        "kaggle", "kernels", "output", ref,
                        "-p", str(destination), "--force", timeout=120,
                    )
                    event["output_exit_code"] = output_result.returncode
                    logs = sorted(destination.glob("*.log"))
                    if logs:
                        try:
                            event["log_tail"] = logs[-1].read_text(
                                encoding="utf-8", errors="replace"
                            )[-4000:]
                        except OSError as exc:
                            event["log_error"] = str(exc)
                append(event)
    finally:
        switch(PRIMARY)


if __name__ == "__main__":
    main()
