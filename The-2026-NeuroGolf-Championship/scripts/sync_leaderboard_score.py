"""
sync_leaderboard_score.py -- pulls the latest COMPLETE Kaggle submission's real public score
and writes it to repairs/leaderboard_score.json, so the webapp can show the REAL confirmed
leaderboard number instead of the local tracker.db total (which is inflated by ~21 tasks
carrying a 25.0 placeholder sentinel for the negative-pads/checker-only-unmeasurable set --
that total is a fair-share ESTIMATE, not a verified score, and should never be presented as
equivalent to the real leaderboard).

Run any time, or after a submission finishes scoring:
    .venv/Scripts/python.exe scripts/sync_leaderboard_score.py
"""
import csv
import io
import json
import os
import subprocess
import sys

OUT_FILE = os.path.join("repairs", "leaderboard_score.json")


def main():
    r = subprocess.run(
        [sys.executable, "-m", "kaggle", "competitions", "submissions", "neurogolf-2026", "--csv"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("ERROR calling kaggle CLI:", r.stderr[:500])
        sys.exit(1)

    rows = list(csv.DictReader(io.StringIO(r.stdout)))
    latest_complete = next((row for row in rows if row["status"] == "SubmissionStatus.COMPLETE"), None)
    if latest_complete is None:
        print("No COMPLETE submission found yet.")
        sys.exit(1)

    data = {
        "public_score": float(latest_complete["publicScore"]),
        "private_score": (float(latest_complete["privateScore"]) if latest_complete["privateScore"] else None),
        "ref": latest_complete["ref"],
        "date": latest_complete["date"],
        "description": latest_complete["description"],
    }
    json.dump(data, open(OUT_FILE, "w"), indent=1)
    print(f"Synced: public_score={data['public_score']} (ref {data['ref']}, {data['date']})")
    print(f"Written to {OUT_FILE}")


if __name__ == "__main__":
    main()
