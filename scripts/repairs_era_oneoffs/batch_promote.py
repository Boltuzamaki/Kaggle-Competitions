"""Batch-promote predicted/test_onnx_taskNNN.py sources into repairs/user_code/taskNNN.py.

For every task that has a repaired ONNX but only a stub (or no file) in user_code/:
  - Reads predicted/test_onnx_taskNNN.py
  - Adapts it so the webapp can exec it and get a top-level `model` variable
  - Writes to repairs/user_code/taskNNN.py

Also handles autosolved tasks (from autosolved.json) by writing a descriptive stub.

Does NOT re-run audits or insert version rows — it's a pure file-copy operation.

Usage:
    python repairs/batch_promote.py [--dry-run]
"""
import os, re, sys, json

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPAIRS = os.path.join(PROJ, "repairs")
USERCODE = os.path.join(REPAIRS, "user_code")
PREDICTED = os.path.join(PROJ, "predicted")
AUTOSOLVED_FILE = os.path.join(REPAIRS, "autosolved.json")

DRY_RUN = "--dry-run" in sys.argv


def is_stub(content):
    """Check if the content is a stub/placeholder (just loads an onnx file)."""
    lines = [l.strip() for l in content.strip().splitlines() if l.strip() and not l.strip().startswith("#")]
    # Stub pattern: just "import onnx" + "model = onnx.load(...)"
    if len(lines) <= 3 and any("onnx.load" in l for l in lines):
        return True
    if "ALREADY SOLVED" in content:
        return True
    # Also catch the generic rule stubs
    if "solved by generic rule" in content:
        return True
    return False


def adapt_predicted_source(src_code, task_num):
    """Adapt a predicted/test_onnx_taskNNN.py so it sets a top-level `model` variable.

    The predicted files typically have one of these patterns:
      1. build_model() / create_model() that constructs and saves the model internally
      2. check_task() / main() that creates + tests the model

    We keep the full source as-is (for readability), and append a line that calls
    the builder function and captures the model. The webapp execs the whole thing
    and picks up `model` from the namespace.
    """
    # Find the main builder function name — look for the function called in __main__
    main_call = None
    if "if __name__" in src_code:
        # Extract the function call after if __name__ == '__main__':
        parts = src_code.split("if __name__")
        if len(parts) >= 2:
            main_block = parts[1]
            # Look for a function call like "build_model()" or "check_task()"
            m = re.search(r"^\s+(\w+)\(\)", main_block, re.MULTILINE)
            if m:
                main_call = m.group(1)

    # Find the function that actually BUILDS the model (returns or saves it)
    # Look for functions that contain "make_model" or "make_graph"
    builder_func = None
    func_defs = re.findall(r"^def (\w+)\(", src_code, re.MULTILINE)

    for func_name in func_defs:
        # Get the function body
        pattern = r"^def " + re.escape(func_name) + r"\(.*?\).*?(?=\n(?:def |if __name__|class )|\Z)"
        match = re.search(pattern, src_code, re.DOTALL | re.MULTILINE)
        if match:
            body = match.group()
            if "make_model" in body or "make_graph" in body:
                builder_func = func_name
                break

    if builder_func is None:
        # Fall back to the main_call function
        builder_func = main_call

    if builder_func is None:
        # Can't determine the builder — just include the source as-is with a note
        header = f"# Source code for task {task_num} (from predicted/test_onnx_task{task_num:03d}.py)\n"
        header += f"# NOTE: This file builds the ONNX model. The verified model is at repairs/task{task_num:03d}.onnx\n"
        header += f"import onnx\n"
        header += f"model = onnx.load(\"/project/repairs/task{task_num:03d}.onnx\")\n\n"
        header += "# --- Original source code below ---\n"
        header += "# " + src_code.replace("\n", "\n# ")
        return header

    # Check if the builder function returns the model or saves it internally
    match = re.search(r"^def " + re.escape(builder_func) + r"\(.*?\).*?(?=\n(?:def |if __name__|class )|\Z)",
                       src_code, re.DOTALL | re.MULTILINE)
    func_body = match.group() if match else ""

    has_return = bool(re.search(r"^\s+return\s+\w+", func_body, re.MULTILINE))
    has_save = "onnx.save" in func_body

    # Remove the if __name__ block and the onnx.save() call, then add model = builder()
    adapted = src_code

    # Strip the if __name__ block
    adapted = re.sub(r"\nif __name__\s*==\s*['\"]__main__['\"]:\s*\n.*", "", adapted, flags=re.DOTALL)

    if has_return:
        # Function returns the model — just call it
        adapted += f"\n\n# Build the model\nmodel = {builder_func}()\n"
    elif has_save:
        # Function saves internally but doesn't return — modify to capture
        # Replace onnx.save(model, ...) with a return, or just load after save
        # Safer: just add a wrapper
        adapted += f"\n\n# Build the model (the function saves it internally, so we load the result)\n"
        adapted += f"{builder_func}()\n"
        adapted += f"import glob\n"
        adapted += f"model = onnx.load(\"/project/repairs/task{task_num:03d}.onnx\")\n"
    else:
        # Unknown pattern — just call and load
        adapted += f"\n\n# Build the model\n"
        adapted += f"model = onnx.load(\"/project/repairs/task{task_num:03d}.onnx\")\n"

    # Add header comment
    header = f"# Source: predicted/test_onnx_task{task_num:03d}.py — ONNX graph construction code\n"
    header += f"# Verified model: repairs/task{task_num:03d}.onnx\n"

    return header + adapted.lstrip()


def autosolved_stub(task_num, rule_category):
    """Generate a stub for autosolved tasks (generic rule, no per-task script)."""
    return f"""# Task {task_num} — solved by generic rule: "{rule_category}"
# No per-task construction script exists. This task was solved by a rule-based
# category solver ({rule_category}) that handles multiple tasks at once.
# The verified model is at repairs/task{task_num:03d}.onnx
import onnx
model = onnx.load("/project/repairs/task{task_num:03d}.onnx")
"""


def main():
    os.makedirs(USERCODE, exist_ok=True)

    # Load autosolved
    autosolved = {}
    if os.path.exists(AUTOSOLVED_FILE):
        autosolved = {int(k): v for k, v in json.load(open(AUTOSOLVED_FILE)).items()}

    # Scan user_code for stubs
    promoted = 0
    skipped = 0
    auto_written = 0
    no_source = 0

    for t in range(1, 401):
        onnx_path = os.path.join(REPAIRS, f"task{t:03d}.onnx")
        uc_file = os.path.join(USERCODE, f"task{t:03d}.py")

        # Skip if no repaired ONNX
        if not os.path.exists(onnx_path):
            continue

        # Skip if user_code already has REAL code (not a stub)
        if os.path.exists(uc_file):
            content = open(uc_file, encoding="utf-8").read()
            if not is_stub(content):
                skipped += 1
                continue

        # Check for predicted source (might be 0-padded or not)
        pred_file = os.path.join(PREDICTED, f"test_onnx_task{t:03d}.py")
        if not os.path.exists(pred_file):
            pred_file = os.path.join(PREDICTED, f"test_onnx_task{t}.py")

        if os.path.exists(pred_file):
            src = open(pred_file, encoding="utf-8").read()
            adapted = adapt_predicted_source(src, t)
            if DRY_RUN:
                print(f"  [DRY-RUN] task{t:03d}: would write {len(adapted)} bytes to user_code/")
            else:
                open(uc_file, "w", encoding="utf-8").write(adapted)
                # Also update the draft to match
                draft_f = os.path.join(USERCODE, f"task{t:03d}.draft.py")
                if os.path.exists(draft_f):
                    draft_content = open(draft_f, encoding="utf-8").read()
                    if is_stub(draft_content):
                        open(draft_f, "w", encoding="utf-8").write(adapted)
            promoted += 1
        elif t in autosolved:
            stub = autosolved_stub(t, autosolved[t])
            if DRY_RUN:
                print(f"  [DRY-RUN] task{t:03d}: would write autosolved stub ({autosolved[t]})")
            else:
                open(uc_file, "w", encoding="utf-8").write(stub)
                draft_f = os.path.join(USERCODE, f"task{t:03d}.draft.py")
                if os.path.exists(draft_f):
                    draft_content = open(draft_f, encoding="utf-8").read()
                    if is_stub(draft_content):
                        open(draft_f, "w", encoding="utf-8").write(stub)
            auto_written += 1
        else:
            no_source += 1
            if DRY_RUN:
                print(f"  [DRY-RUN] task{t:03d}: no source found anywhere")

    print(f"\nDone! promoted={promoted}, autosolved={auto_written}, "
          f"already_had_code={skipped}, no_source={no_source}")
    if DRY_RUN:
        print("(dry run — no files were written)")


if __name__ == "__main__":
    main()
