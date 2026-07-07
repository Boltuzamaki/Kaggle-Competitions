import os
import sys
import glob
import subprocess
import onnx
import traceback

webapp_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, webapp_dir)
os.environ["PROJECT_DIR"] = os.path.dirname(webapp_dir)

from app import audit, set_task
from universal_patcher import universal_patch

unsolved_list = [4, 5, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 27, 29, 42, 44, 46, 48, 54, 58, 65, 66, 69, 71, 74, 77, 80, 83, 84, 86, 90, 92, 93, 96, 101, 102, 105, 106, 107, 109, 111, 112, 115, 117, 118, 119, 123, 124, 125, 126, 129, 130, 133, 134, 137, 138, 141, 142, 143, 144, 145, 148, 151, 152, 153, 154, 156, 157, 158, 159, 161, 162, 163, 164, 165, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 308, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 338, 339, 340, 341, 342, 343, 344, 345, 346, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 381, 382, 383, 387, 388, 390, 391, 392, 393, 394, 396, 397, 398, 399, 400]

BANNED_OPS = ['NonZero', 'Loop', 'Scan', 'Unique', 'Compress', 'Sequence']

PROJ = os.environ["PROJECT_DIR"]
PRED = os.path.join(PROJ, "predicted")
REP = os.path.join(PROJ, "repairs")

banned_tasks = []
patched_tasks = []
passed_tasks = []
failed_tasks = []

def run_mass_patch():
    print(f"Starting mass patch over {len(unsolved_list)} tasks...")
    
    for t in unsolved_list:
        py_file = os.path.join(PRED, f"test_onnx_task{t:03d}.py")
        if not os.path.exists(py_file):
            print(f"[{t}] Python file not found, skipping...")
            failed_tasks.append(t)
            continue
            
        with open(py_file, 'r') as f:
            content = f.read()
            
        # Check for banned ops
        is_banned = False
        for b in BANNED_OPS:
            if b in content:
                print(f"[{t}] Contains banned op: {b}")
                banned_tasks.append(t)
                is_banned = True
                break
        
        if is_banned:
            continue
            
        # Execute the ORIGINAL script to get the raw onnx
        try:
            subprocess.run([sys.executable, py_file], cwd=PRED, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"[{t}] Raw script execution failed: {e}")
            failed_tasks.append(t)
            continue
            
        raw_onnx_path = os.path.join(PRED, f"task{t:03d}.onnx")
        patched_onnx_path = os.path.join(PRED, f"task{t:03d}_patched.onnx")
        
        if not os.path.exists(raw_onnx_path):
            print(f"[{t}] Did not produce ONNX file!")
            failed_tasks.append(t)
            continue
            
        try:
            universal_patch(raw_onnx_path, patched_onnx_path)
            patched_tasks.append(t)
        except Exception as e:
            print(f"[{t}] Universal Patcher failed: {e}")
            failed_tasks.append(t)
            continue
            
        # Audit the generated ONNX
        if os.path.exists(patched_onnx_path):
            try:
                model = onnx.load(patched_onnx_path)
                nfail, cost, pts, msg = audit(model, t)
                
                if nfail == 0 and cost is not None:
                    print(f"[{t}] PASSED AUDIT! Saving to repairs...")
                    import shutil
                    shutil.copy(patched_onnx_path, os.path.join(REP, f"task{t:03d}.onnx"))
                    set_task(t, state='ours', our_points=round(pts, 3), our_cost=int(cost), n_fail=0)
                    passed_tasks.append(t)
                else:
                    print(f"[{t}] Failed audit. nfail: {nfail}, msg: {msg}")
                    failed_tasks.append(t)
            except Exception as e:
                print(f"[{t}] Audit threw exception: {e}")
                failed_tasks.append(t)
        else:
            print(f"[{t}] Patched ONNX not found.")
            failed_tasks.append(t)
            
    print("\n--- Summary ---")
    print(f"Patched: {len(patched_tasks)}")
    print(f"Passed Audit: {len(passed_tasks)}")
    print(f"Failed Audit: {len(failed_tasks)}")
    print(f"Banned Ops (Need manual/subagent fix): {len(banned_tasks)} - {banned_tasks}")

if __name__ == '__main__':
    run_mass_patch()
