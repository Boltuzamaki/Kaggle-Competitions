"""Self-serve checker: point it at a .py file you wrote (must set a top-level
`model` = onnx.ModelProto, same convention as repairs/user_code/*.py), and it will:

  1. Run the EXACT same audit the webapp UI runs (verify_task.audit): sanitize,
     onnx.checker full_check, load in onnxruntime, run every train+test+arc-gen
     example for the task, and measure cost the same way the real scorer does.
  2. Print n_fail / cost / points, and on failure the first few mismatching
     example indices (or the exception if the model itself errored/crashed).
  3. ONLY if it genuinely passes (n_fail==0 and cost measurable): save the
     .onnx to repairs/taskNNN.onnx, copy your source into repairs/user_code/taskNNN.py,
     insert a version row, and flip tracker.db tasks.state to 'ours' with the
     real points/cost - i.e. exactly what mark_solved.py / the UI's run_code()
     does. Nothing is written to the DB if it fails.

Won't overwrite an existing 'ours' entry unless --force (so you can't
accidentally clobber a better solve already recorded).

Usage:
    python repairs/check_and_promote.py <path/to/your_task_file.py> [task_num] [--force]

If task_num is omitted, it's inferred from the filename (e.g. task219.py -> 219).
"""
import sys, os, sqlite3, re, shutil
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_task import audit, load_model_from_path, PROJ

REP = os.path.join(PROJ, "repairs")
DB_FILE = os.path.join(REP, "tracker.db")
USERCODE_DIR = os.path.join(REP, "user_code")


def db():
    c = sqlite3.connect(DB_FILE)
    c.row_factory = sqlite3.Row
    return c


def get_task(c, t):
    r = c.execute("SELECT * FROM tasks WHERE task=?", (t,)).fetchone()
    if r is None:
        raise SystemExit(f"task {t} not found in tracker.db")
    return dict(r)


def infer_task_num(path):
    m = re.search(r"task0*(\d+)", os.path.basename(path))
    if not m:
        raise SystemExit(f"couldn't infer task number from filename '{path}' - pass it explicitly as arg 2")
    return int(m.group(1))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--force"]
    force = "--force" in sys.argv
    if not args:
        print(__doc__)
        sys.exit(2)
    src_path = args[0]
    t = int(args[1]) if len(args) > 1 else infer_task_num(src_path)

    import onnx

    c = db()
    e = get_task(c, t)
    if e["state"] == "ours" and not force:
        print(f"task{t:03d} already marked 'ours' (points={e['our_points']}, cost={e['our_cost']}) "
              f"- pass --force to re-check/overwrite")
        c.close()
        sys.exit(0)

    print(f"--- loading {src_path} as task{t:03d} ---")
    model = load_model_from_path(src_path)

    nfail, cost, pts, msg = audit(model, t)

    if nfail is None or nfail != 0 or cost is None:
        print(f"task{t:03d}: NOT solved (n_fail={nfail}, msg={msg}) - DB left untouched"
              if e["state"] == "ours" else
              f"task{t:03d}: NOT solved (n_fail={nfail}, msg={msg})")
        if e["state"] != "ours":
            c.execute("UPDATE tasks SET n_fail=?, state='working' WHERE task=?", (nfail, t))
            c.commit()
        c.close()
        sys.exit(1)

    onnx_path = os.path.join(REP, f"task{t:03d}.onnx")
    onnx.save(model, onnx_path)
    code_path = os.path.join(USERCODE_DIR, f"task{t:03d}.py")
    if os.path.abspath(src_path) != os.path.abspath(code_path):
        shutil.copyfile(src_path, code_path)
    code = open(code_path, encoding="utf-8").read()
    n_nodes = len(model.graph.node)

    c.execute("INSERT INTO versions(task,kind,content,points,cost,n_nodes,ts) VALUES(?,?,?,?,?,?,?)",
              (t, "code", code, round(pts, 3), int(cost), n_nodes, datetime.utcnow().isoformat(timespec="seconds")))
    c.execute("UPDATE tasks SET state='ours', our_points=?, our_cost=?, n_fail=0 WHERE task=?",
              (round(pts, 3), int(cost), t))
    c.commit()
    c.close()

    verdict = "BEATS" if pts > e["base_points"] else ("TIES" if pts == e["base_points"] else "below")
    print(f"task{t:03d}: SOLVED - points={pts:.3f} cost={cost} "
          f"(base_points={e['base_points']}, {verdict} baseline) -> saved {onnx_path}, marked 'ours'")
