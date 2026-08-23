"""Combine isolated per-opponent exact-package safety artifacts."""

from __future__ import annotations

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    reports = []
    for path in args.inputs:
        with open(path, encoding="utf-8") as handle:
            reports.append(json.load(handle))
    hashes = {report["archive_sha256"] for report in reports}
    members = {tuple(report["archive_members"]) for report in reports}
    archives = {report["archive"] for report in reports}
    if len(hashes) != 1 or len(members) != 1 or len(archives) != 1:
        raise SystemExit("package identity differs between worker artifacts")

    opponents = []
    for report in reports:
        opponents.extend(report["opponents"])
    names = [item["opponent"] for item in opponents]
    if len(names) != len(set(names)):
        raise SystemExit("duplicate opponent result")
    total_games = sum(item["games"] for item in opponents)
    combined = {
        "archive": reports[0]["archive"],
        "archive_sha256": reports[0]["archive_sha256"],
        "archive_members": reports[0]["archive_members"],
        "total_games": total_games,
        "candidate_failures": sum(item["candidate_failures"] for item in opponents),
        "game_errors": sum(item["game_errors"] for item in opponents),
        "mean_match_s": sum(
            item["mean_match_s"] * item["games"] for item in opponents
        ) / total_games,
        "p95_match_s_conservative": max(item["p95_match_s"] for item in opponents),
        "opponents": sorted(opponents, key=lambda item: item["opponent"]),
        "parallel_workers": len(reports),
        "submission_performed": False,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2)
        handle.write("\n")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
