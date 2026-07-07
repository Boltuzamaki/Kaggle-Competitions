"""
Process unsolved tasks one by one:
1. Run the py file to generate ONNX
2. Check if it needs patching (I/O contract, padding)
3. Audit
4. If passes, push to UI
"""
import os, sys, json, numpy as np, onnx, copy, math, shutil, re, subprocess, traceback

PROJ = r'c:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship'
os.environ['PROJECT_DIR'] = PROJ
sys.path.insert(0, os.path.join(PROJ, 'webapp'))
sys.path.insert(0, os.path.join(PROJ, 'data', 'neurogolf_utils'))

from app import audit, set_task
from universal_patcher import universal_patch

PRED = os.path.join(PROJ, 'predicted')
DATA = os.path.join(PROJ, 'data')
REP = os.path.join(PROJ, 'repairs')

BANNED_OPS = {'NonZero', 'Loop', 'Scan', 'Unique', 'Compress'}

def find_py_file(t):
    """Find the python file for task t (handles both padded and unpadded naming)."""
    for name in [f'test_onnx_task{t}.py', f'test_onnx_task{t:03d}.py']:
        path = os.path.join(PRED, name)
        if os.path.exists(path):
            return path
    return None

def find_onnx_save_name(py_content):
    """Extract the onnx filename from the python script."""
    matches = re.findall(r"onnx\.save\([^,]+,\s*['\"]([^'\"]+\.onnx)['\"]", py_content)
    return matches[0] if matches else None

def check_banned_ops(py_content):
    """Check if the script uses banned ONNX ops."""
    for op in BANNED_OPS:
        if op in py_content:
            return op
    return None

def needs_patching(model):
    """Check if model needs I/O patching."""
    inp = model.graph.input[0]
    out = model.graph.output[0]
    
    # Check input
    inp_ok = (inp.name == 'input' and 
              inp.type.tensor_type.elem_type == onnx.TensorProto.FLOAT and
              len(inp.type.tensor_type.shape.dim) == 4)
    
    # Check output  
    out_ok = (out.name == 'output' and
              out.type.tensor_type.elem_type == onnx.TensorProto.FLOAT and
              len(out.type.tensor_type.shape.dim) == 4)
    
    return not (inp_ok and out_ok)

def process_task(t):
    """Process a single task. Returns (success, message)."""
    py_file = find_py_file(t)
    if not py_file:
        return False, "No Python file found"
    
    py_content = open(py_file).read()
    
    # Check banned ops
    banned = check_banned_ops(py_content)
    if banned:
        return False, f"BANNED OP: {banned}"
    
    # Find onnx save name
    onnx_name = find_onnx_save_name(py_content)
    if not onnx_name:
        return False, "Cannot find onnx.save() in script"
    
    # Run the script
    try:
        result = subprocess.run(
            [sys.executable, py_file], 
            cwd=PRED, capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            return False, f"Script failed: {result.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return False, "Script timed out"
    except Exception as e:
        return False, f"Script error: {e}"
    
    onnx_path = os.path.join(PRED, onnx_name)
    if not os.path.exists(onnx_path):
        return False, f"ONNX file {onnx_name} not created"
    
    # Load and check
    try:
        model = onnx.load(onnx_path)
    except Exception as e:
        return False, f"Cannot load ONNX: {e}"
    
    # If needs patching, patch it
    if needs_patching(model):
        patched_path = onnx_path.replace('.onnx', '_patched.onnx')
        try:
            universal_patch(onnx_path, patched_path)
            if os.path.exists(patched_path):
                model = onnx.load(patched_path)
                onnx_path = patched_path
            else:
                return False, "Patching failed - no output"
        except Exception as e:
            return False, f"Patching failed: {e}"
    
    # Audit
    try:
        nfail, cost, pts, msg = audit(model, t)
    except Exception as e:
        return False, f"Audit exception: {e}"
    
    if nfail == 0 and cost is not None:
        # SUCCESS! Save to repairs and push to UI
        repair_path = os.path.join(REP, f'task{t:03d}.onnx')
        shutil.copy(onnx_path, repair_path)
        set_task(t, state='ours', our_points=round(pts, 3), our_cost=int(cost), n_fail=0)
        return True, f"PASSED! pts={pts:.3f}, cost={cost}"
    elif nfail == 0 and cost is None:
        return False, f"nfail=0 but {msg}"
    else:
        return False, f"nfail={nfail}, msg={msg}"

def main():
    # Process ONE task at a time
    if len(sys.argv) > 1:
        task_num = int(sys.argv[1])
        print(f"=== Task {task_num} ===")
        success, message = process_task(task_num)
        status = "PASSED" if success else "FAILED"
        print(f"  {status}: {message}")
        return
    
    # Or process a batch
    unsolved = [5, 21, 27, 29, 42, 44, 46, 48, 65, 66, 69, 71, 74, 77, 80, 83, 84, 86, 90, 92, 93, 96, 101, 102, 105, 106, 109, 111, 112, 117, 118, 123, 124, 125, 126, 129, 130, 133, 134, 137, 138, 141, 142, 144, 145, 148, 151, 152, 153, 154, 156, 157, 158, 159, 161, 162, 163, 164, 165, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 308, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 338, 339, 340, 341, 342, 343, 344, 345, 346, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 376, 377, 378, 379, 381, 382, 383, 387, 388, 390, 391, 392, 393, 394, 396, 397, 398, 399, 400]
    
    passed = []
    failed = []
    banned = []
    
    for t in unsolved:
        print(f"=== Task {t} ===", flush=True)
        success, message = process_task(t)
        status = "PASSED" if success else "FAILED"
        print(f"  {status}: {message}", flush=True)
        
        if success:
            passed.append(t)
        elif "BANNED" in message:
            banned.append(t)
        else:
            failed.append(t)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {len(passed)} - {passed}")
    print(f"Failed: {len(failed)}")  
    print(f"Banned: {len(banned)} - {banned}")

if __name__ == '__main__':
    main()
