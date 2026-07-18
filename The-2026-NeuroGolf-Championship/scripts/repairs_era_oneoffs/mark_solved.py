"""Mirrors the DB-write side of webapp/app.py::run_code() for a CLI workflow.
Verifies a candidate model.onnx (already saved to repairs/taskNNN.onnx) and, if it
genuinely passes (n_fail==0, cost measurable), marks the task 'ours' in tracker.db
and records a version row. Does NOT overwrite an existing 'ours' solution unless
--force is passed (see reference_predicted_fix_playbook: don't clobber a superior repair).

Usage:
    python repairs/mark_solved.py <task_num> [--force]
"""
import sys, os, sqlite3
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_task import audit, PROJ

REP = os.path.join(PROJ, "repairs")
DB_FILE = os.path.join(REP, "tracker.db")
USERCODE_DIR = os.path.join(REP, "user_code")


def db():
    c = sqlite3.connect(DB_FILE)
    c.row_factory = sqlite3.Row
    return c


def get_task(c, t):
    r = c.execute("SELECT * FROM tasks WHERE task=?", (t,)).fetchone()
    return dict(r)


if __name__ == "__main__":
    import onnx
    force = "--force" in sys.argv
    t = int(sys.argv[1])
    onnx_path = os.path.join(REP, f"task{t:03d}.onnx")
    if not os.path.exists(onnx_path):
        print(f"no {onnx_path}"); sys.exit(1)

    c = db()
    e = get_task(c, t)
    if e["state"] == "ours" and not force:
        print(f"task{t:03d} already marked 'ours' (points={e['our_points']}, cost={e['our_cost']}) — use --force to re-check/overwrite")
        sys.exit(0)

    model = onnx.load(onnx_path)
    nfail, cost, pts, msg = audit(model, t)
    if nfail is None or nfail != 0 or cost is None:
        print(f"task{t:03d}: NOT solved (n_fail={nfail}, msg={msg}) — not marking")
        c.execute("UPDATE tasks SET n_fail=?, state=? WHERE task=?",
                  (nfail, e["state"] if e["state"] == "ours" else "working", t))
        c.commit(); c.close()
        sys.exit(1)

    code_path = os.path.join(USERCODE_DIR, f"task{t:03d}.py")
    code = open(code_path, encoding="utf-8").read() if os.path.exists(code_path) else ""
    n_nodes = len(model.graph.node)
    c.execute("INSERT INTO versions(task,kind,content,points,cost,n_nodes,ts) VALUES(?,?,?,?,?,?,?)",
              (t, "code", code, round(pts, 3), int(cost), n_nodes, datetime.utcnow().isoformat(timespec="seconds")))
    c.execute("UPDATE tasks SET state='ours', our_points=?, our_cost=?, n_fail=0 WHERE task=?",
              (round(pts, 3), int(cost), t))
    c.commit(); c.close()
    print(f"task{t:03d}: SOLVED — points={pts:.3f} cost={cost} (base_points={e['base_points']}) -> marked 'ours'")
