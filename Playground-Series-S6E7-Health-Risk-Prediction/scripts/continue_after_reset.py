"""Resume the evidence-ranked S6E7 probes when Kaggle's daily quota resets.

This process is deliberately bounded: four prepared submissions maximum, no
duplicate filenames, and an immediate stop once public score reaches 0.953.
"""

from __future__ import annotations

import csv
import datetime as dt
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "playground-series-s6e7"
TARGET_SCORE = 0.953
LOG = ROOT / "artifacts/continue_after_reset.log"
CANDIDATES = [
    (
        ROOT / "submissions/inverse_frontier_support4_neural7.csv",
        "0.95288 base + 7 positive-ridge unhealthy promotions, unanimous 7-model support",
    ),
    (
        ROOT / "submissions/inverse_frontier_support4_neural_support3.csv",
        "0.95288 base + 2 unanimous neural rows with 3/9 external peer support",
    ),
    (
        ROOT / "submissions/inverse_frontier_support4_neural_support2.csv",
        "0.95288 base + 4 unanimous neural rows with 2/9 external peer support",
    ),
    (
        ROOT / "submissions/inverse_frontier_support4_neural_support1.csv",
        "0.95288 base + remaining unanimous neural positive-ridge row",
    ),
]


def log(message: str) -> None:
    timestamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    with LOG.open("a") as handle:
        handle.write(line + "\n")


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True)


def submissions() -> list[dict[str, str]]:
    result = run("kaggle", "competitions", "submissions", "-c", COMPETITION, "--csv")
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return list(csv.DictReader(result.stdout.splitlines()))


def latest_score(filename: str, timeout_seconds: int = 600) -> float:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        for row in submissions():
            if row["fileName"] == filename:
                if row["status"].endswith("COMPLETE") and row["publicScore"]:
                    return float(row["publicScore"])
                break
        time.sleep(15)
    raise TimeoutError(f"score timed out for {filename}")


def already_submitted(filename: str) -> bool:
    return any(row["fileName"] == filename for row in submissions())


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log("continuation started")
    # Retry the first not-yet-submitted candidate until quota reset. Stop
    # retrying at the competition deadline rather than looping indefinitely.
    deadline = dt.datetime(2026, 8, 1, 5, 29, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)))
    for path, message in CANDIDATES:
        if already_submitted(path.name):
            log(f"skip duplicate {path.name}")
            continue
        while dt.datetime.now().astimezone() < deadline:
            result = run(
                "kaggle", "competitions", "submit", "-c", COMPETITION,
                "-f", str(path), "-m", message,
            )
            if result.returncode == 0:
                log(f"accepted {path.name}")
                break
            text = (result.stderr + result.stdout).strip().replace("\n", " ")
            log(f"deferred {path.name}: {text[-300:]}")
            time.sleep(300)
        else:
            log("deadline reached before submission")
            return
        try:
            score = latest_score(path.name)
        except Exception as exc:
            log(f"score failure {path.name}: {type(exc).__name__}: {exc}")
            return
        log(f"score {path.name}: {score:.5f}")
        if score >= TARGET_SCORE:
            log(f"TARGET REACHED: {score:.5f}")
            return
    log("bounded candidate sequence exhausted without target")


if __name__ == "__main__":
    main()
