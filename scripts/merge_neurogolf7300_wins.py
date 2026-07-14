"""
Merges tasks from the neurogolf7300 comparison (scripts/audit_neurogolf7300.py output) into
repairs/, but ONLY where the comparison was clean: their_status == "ok" (passed our normal
checker with no negative-pads/banned-op/contrib-domain special-casing) and their points
genuinely beat ours. Anything that would need a rules-loophole (negative-pads-checker-only,
sanitize-fail, cost-unmeasurable, wrong-output) is skipped, not merged.

Backs up the full repairs/ directory first. Updates tracker.db notes per task with both
scores for the record.
"""
import functools
import json
import math
import os
import shutil
import sqlite3

print = functools.partial(print, flush=True)

REP = "repairs"
OTHER = os.path.join("submissions", "neurogolf7300")
COMPARE_FILE = os.path.join(REP, "neurogolf7300_compare.json")
BACKUP_DIR = "_disposable_logs_and_traces/backup_before_neurogolf7300_merge_2026-07-12"

data = json.load(open(COMPARE_FILE))

wins = []
for t_str, v in data.items():
    t = int(t_str)
    if v.get("their_status") == "ok" and v.get("delta") is not None and v["delta"] > 0:
        wins.append((t, v["our_pts"], v["their_pts"], v["delta"]))

wins.sort(key=lambda x: -x[3])
print(f"{len(wins)} clean wins to merge (status=='ok', delta>0)")
for t, our_pts, their_pts, delta in wins[:20]:
    print(f"  task{t:03d}: {our_pts:.4f} -> {their_pts:.4f} ({delta:+.4f})")
if len(wins) > 20:
    print(f"  ... and {len(wins)-20} more")

os.makedirs(BACKUP_DIR, exist_ok=True)
shutil.copytree(REP, os.path.join(BACKUP_DIR, "repairs"), dirs_exist_ok=True)
print(f"\nBacked up repairs/ to {BACKUP_DIR}/repairs")

con = sqlite3.connect(os.path.join(REP, "tracker.db"))
cur = con.cursor()

merged = 0
for t, our_pts, their_pts, delta in wins:
    src = os.path.join(OTHER, f"task{t:03d}.onnx")
    dst = os.path.join(REP, f"task{t:03d}.onnx")
    shutil.copy(src, dst)
    # their_pts wasn't floor-clamped (status=="ok" means a genuine measured cost), so the
    # points formula inverts cleanly: cost = exp(25 - pts).
    their_cost = round(math.exp(25.0 - their_pts))
    cur.execute("SELECT notes FROM tasks WHERE task=?", (t,))
    row = cur.fetchone()
    old_notes = row[0] if row else ""
    note = (old_notes or "") + f"""

[2026-07-12] MERGED from neurogolf7300 dataset (submissions/neurogolf7300/task{t:03d}.onnx),
per explicit user instruction after weighing the forum poison-pill/rules-risk warning. Only
merged because comparison was clean (passed our normal checker, no negative-pads/banned-op/
contrib-domain special-casing needed). pts {our_pts:.4f} -> {their_pts:.4f} ({delta:+.4f}),
cost back-derived from points as {their_cost} (exact cost not stored by the audit script)."""
    cur.execute("UPDATE tasks SET our_points=?, our_cost=?, notes=?, n_fail=0, source=? WHERE task=?",
                (their_pts, their_cost, note, "neurogolf7300", t))
    merged += 1

con.commit()
print(f"\nMerged {merged} tasks into repairs/ and tracker.db.")
