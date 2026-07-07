import os
import sys
import glob
import onnx
import subprocess
import traceback

# Add webapp and data/neurogolf_utils to path
webapp_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, webapp_dir)

PROJ = os.path.dirname(webapp_dir)
os.environ["PROJECT_DIR"] = PROJ

from app import audit, set_task, get_task, all_tasks

PROJ = os.path.dirname(webapp_dir)
PRED = os.path.join(PROJ, "predicted")
REP = os.path.join(PROJ, "repairs")

def process_unsolved_tasks():
    tasks = all_tasks()
    unsolved = [t['task'] for t in tasks if t['state'] != 'ours']
    
    print(f"Found {len(unsolved)} unsolved tasks.")
    
    passed_count = 0
    failed_count = 0
    
    for t in unsolved:
        onnx_path = os.path.join(PRED, f"task{t:03d}.onnx")
        py_path = os.path.join(PRED, f"test_onnx_task{t:03d}.py")
        
        # If ONNX doesn't exist, but Python script does, run the Python script to generate it
        if not os.path.exists(onnx_path) and os.path.exists(py_path):
            print(f"Running {py_path} to generate model for task {t}...")
            try:
                subprocess.run([sys.executable, py_path], cwd=PRED, check=False, capture_output=True)
            except Exception as e:
                print(f"Error running script for task {t}: {e}")
        
        if os.path.exists(onnx_path):
            try:
                model = onnx.load(onnx_path)
                nfail, cost, pts, msg = audit(model, t)
                
                if nfail == 0 and cost is not None:
                    print(f"Task {t:03d} PASSED! Cost: {cost}, Points: {pts}")
                    # Save to repairs
                    repair_onnx = os.path.join(REP, f"task{t:03d}.onnx")
                    onnx.save(model, repair_onnx)
                    # Update DB
                    set_task(t, state="ours", our_points=round(pts, 3), our_cost=int(cost), n_fail=0)
                    passed_count += 1
                else:
                    print(f"Task {t:03d} FAILED audit. nfail: {nfail}, msg: {msg}")
                    failed_count += 1
            except Exception as e:
                print(f"Task {t:03d} ERROR during audit: {e}")
                failed_count += 1
        else:
            # print(f"Task {t:03d} has no .onnx or .py in predicted folder.")
            pass
            
    print(f"\nDone! Passed: {passed_count}, Failed: {failed_count}")

if __name__ == '__main__':
    process_unsolved_tasks()
