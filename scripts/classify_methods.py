"""
classify_methods.py -- tags every task's CURRENT repairs/taskNNN.onnx with a short,
mechanical "method" label based on its op-type signature, so future sessions can see
at a glance what technique is already in use per task instead of re-deriving it.

This is a heuristic classifier over op histograms (fast, no audits) -- it's meant as
a navigation aid, not a precise technical claim. Writes to tracker.db's new `method`
column (added if missing) and leaves `notes` untouched.

Run:  venv_scorer/Scripts/python.exe scripts/classify_methods.py
"""
import collections
import functools
import os
import sqlite3

import onnx

print = functools.partial(print, flush=True)

REP = "repairs"
DB_FILE = os.path.join(REP, "tracker.db")


def classify(g):
    ops = collections.Counter(n.op_type for n in g.node)
    n = len(g.node)

    if ops.get("Loop") or ops.get("Scan"):
        return "BANNED-OP present (needs fixing)"
    if ops.get("QLinearConv") or ops.get("ConvInteger") or ops.get("MatMulInteger"):
        return f"quantized-conv renderer ({n} nodes)"
    if ops.get("Einsum"):
        return f"Einsum-based remap ({n} nodes)"
    if ops.get("GridSample"):
        return f"GridSample warp/remap ({n} nodes)"
    if ops.get("BitShift", 0) >= 10 and (ops.get("BitwiseAnd", 0) + ops.get("BitwiseOr", 0)) >= 20:
        waves = "small" if n < 200 else ("large" if n > 1000 else "medium")
        return f"bit-packed BFS/flood-fill ({waves}, {n} nodes)"
    if ops.get("Conv", 0) >= 1 and n <= 15:
        return f"color-index Conv (1x1 channel-collapse lookup, {n} nodes)"
    if ops.get("MaxPool", 0) >= 3 and ops.get("Mul", 0) >= 3:
        return f"MaxPool-based dilation/gravity ({n} nodes)"
    if ops.get("TopK", 0) >= 1:
        return f"TopK-based match/select ({n} nodes)"
    if ops.get("ArgMax", 0) >= 1 and ops.get("Gather", 0) >= 3:
        return f"ArgMax+Gather object-select ({n} nodes)"
    if ops.get("ScatterND", 0) >= 1 or ops.get("ScatterElements", 0) >= 1:
        return f"Scatter-based paint/place ({n} nodes)"
    if ops.get("Slice", 0) >= 1 and ops.get("Pad", 0) >= 1 and n <= 10:
        return f"simple crop+pad ({n} nodes)"
    if n <= 6:
        return f"trivial ({n} nodes, top op {ops.most_common(1)})"

    top2 = ops.most_common(2)
    return f"other ({n} nodes, top ops {top2})"


def main():
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("PRAGMA table_info(tasks)")
    if "method" not in [r[1] for r in cur.fetchall()]:
        cur.execute("ALTER TABLE tasks ADD COLUMN method TEXT DEFAULT ''")

    tagged = 0
    for t in range(1, 401):
        path = os.path.join(REP, f"task{t:03d}.onnx")
        if not os.path.exists(path):
            continue
        try:
            m = onnx.load(path)
        except Exception as e:
            cur.execute("UPDATE tasks SET method=? WHERE task=?", (f"load-error: {str(e)[:80]}", t))
            continue
        label = classify(m.graph)
        cur.execute("UPDATE tasks SET method=? WHERE task=?", (label, t))
        tagged += 1

    con.commit()
    print(f"Tagged {tagged}/400 tasks with a method label.")

    cur.execute("SELECT method, COUNT(*) FROM tasks GROUP BY method ORDER BY COUNT(*) DESC")
    print("\nMethod distribution:")
    for label, cnt in cur.fetchall():
        print(f"  {cnt:>4}  {label}")


if __name__ == "__main__":
    main()
