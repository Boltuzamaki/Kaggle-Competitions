import sys, os, copy, json, math
from collections import Counter
sys.path.insert(0, r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\data\neurogolf_utils")
import onnx, onnxruntime, numpy as np
from onnx import helper, numpy_helper
import neurogolf_utils as ngu

BASE_DIR = r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\baseline_v22"
REPAIRS_DIR = r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\repairs"
TASK_DIR = r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\data"
MAX_PASSES = 10

REPAIRED = ["task045.onnx", "task127.onnx", "task384.onnx", "task135.onnx",
            "task146.onnx", "task149.onnx", "task240.onnx", "task347.onnx"]

# ---- kernel border trimming ----
# If a Conv/ConvTranspose kernel has an all-zero border row/col, it can be
# dropped and the corresponding pad shrunk by `dilation` on that side without
# changing the output (verified analytically + empirically on synthetic
# Conv and ConvTranspose cases). Only safe while the resulting pad stays >= 0.

def trim_weight_once(weight, pads, dilations):
    OC, IC, KH, KW = weight.shape
    h_begin, w_begin, h_end, w_end = pads
    dh, dw = dilations
    if KH > 1 and np.all(weight[:, :, 0, :] == 0) and h_begin - dh >= 0:
        return weight[:, :, 1:, :], [h_begin - dh, w_begin, h_end, w_end], "h_begin"
    if KH > 1 and np.all(weight[:, :, -1, :] == 0) and h_end - dh >= 0:
        return weight[:, :, :-1, :], [h_begin, w_begin, h_end - dh, w_end], "h_end"
    if KW > 1 and np.all(weight[:, :, :, 0] == 0) and w_begin - dw >= 0:
        return weight[:, :, :, 1:], [h_begin, w_begin - dw, h_end, w_end], "w_begin"
    if KW > 1 and np.all(weight[:, :, :, -1] == 0) and w_end - dw >= 0:
        return weight[:, :, :, :-1], [h_begin, w_begin, h_end, w_end - dw], "w_end"
    return None

def apply_kernel_trim_once(model):
    model = copy.deepcopy(model)
    init_map = {init.name: init for init in model.graph.initializer}
    ref_counts = Counter(inp for node in model.graph.node for inp in node.input if inp)
    total_reduction = 0
    for node in model.graph.node:
        if node.op_type not in ("Conv", "ConvTranspose") or len(node.input) < 2:
            continue
        wname = node.input[1]
        if wname not in init_map or ref_counts[wname] != 1:
            continue
        attrs = {a.name: a for a in node.attribute}
        if "pads" not in attrs:
            continue
        auto_pad = None
        for a in node.attribute:
            if a.name == "auto_pad":
                auto_pad = helper.get_attribute_value(a)
        if isinstance(auto_pad, bytes):
            auto_pad = auto_pad.decode()
        if auto_pad not in (None, "NOTSET", ""):
            continue

        weight = numpy_helper.to_array(init_map[wname])
        if weight.ndim != 4:
            continue
        pads = list(helper.get_attribute_value(attrs["pads"]))
        dilations = list(helper.get_attribute_value(attrs["dilations"])) if "dilations" in attrs else [1, 1]

        result = trim_weight_once(weight, pads, dilations)
        if result is None:
            continue
        new_weight, new_pads, _ = result

        ii = next(i for i, init in enumerate(model.graph.initializer) if init.name == wname)
        model.graph.initializer[ii].CopyFrom(numpy_helper.from_array(new_weight, name=wname))
        kept = [copy.deepcopy(a) for a in node.attribute if a.name not in {"kernel_shape", "pads"}]
        del node.attribute[:]
        node.attribute.extend(kept)
        node.attribute.extend([
            helper.make_attribute("kernel_shape", list(new_weight.shape[2:])),
            helper.make_attribute("pads", new_pads),
        ])
        total_reduction += weight.size - new_weight.size

    if total_reduction == 0:
        return None, 0
    onnx.checker.check_model(model)
    return model, total_reduction

def apply_kernel_trim(model, max_passes=MAX_PASSES):
    total = 0
    for _ in range(max_passes):
        new_model, reduction = apply_kernel_trim_once(model)
        if new_model is None:
            break
        model = new_model
        total += reduction
    return (model, total) if total > 0 else (None, 0)

# ---- official audit (matches proven-accurate recipe) ----

def audit_bytes(model_bytes, examples):
    try:
        model = onnx.load_model_from_string(model_bytes)
        sanitized = ngu.sanitize_model(model)
        if sanitized is None:
            return {"status": "sanitize_fail"}
        options = onnxruntime.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.profile_file_prefix = "trimprof"
        session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
        n_fail = 0
        for ex in examples["train"] + examples["test"] + examples["arc-gen"]:
            bench = ngu.convert_to_numpy(ex)
            if not bench: continue
            try:
                out = ngu.run_network(session, bench["input"])
                if not np.array_equal(out, bench["output"]): n_fail += 1
            except Exception:
                n_fail += 1
        trace_path = session.end_profiling()
        memory, params = ngu.score_network(sanitized, trace_path)
        try: os.remove(trace_path)
        except Exception: pass
        if memory is None or params is None or memory < 0 or params < 0:
            return {"status": "cost_fail", "n_fail": n_fail}
        cost = memory + params
        if n_fail > 0:
            return {"status": "incorrect", "n_fail": n_fail, "cost": cost}
        points = max(1.0, 25.0 - math.log(max(1.0, cost)))
        return {"status": "ok", "cost": cost, "points": points}
    except Exception:
        return {"status": "error"}

def main():
    improved = []
    checked = 0
    for t in range(1, 401):
        fn = f"task{t:03d}.onnx"
        path = os.path.join(REPAIRS_DIR, fn) if fn in REPAIRED else os.path.join(BASE_DIR, fn)
        model = onnx.load(path)
        has_conv = any(n.op_type in ("Conv", "ConvTranspose") for n in model.graph.node)
        if not has_conv:
            continue
        checked += 1

        cand_model, reduction = apply_kernel_trim(model)
        if cand_model is None:
            continue

        examples = json.load(open(os.path.join(TASK_DIR, f"task{t:03d}.json")))
        base_r = audit_bytes(model.SerializeToString(), examples)
        cand_r = audit_bytes(cand_model.SerializeToString(), examples)

        if cand_r["status"] != "ok":
            print(f"task{t:03d}: candidate INVALID after trim ({cand_r}) -- skipped, keeping original")
            continue
        if base_r["status"] != "ok":
            print(f"task{t:03d}: baseline wasn't ok?! ({base_r}) -- skipped")
            continue

        gain = cand_r["points"] - base_r["points"]
        if gain > 1e-9:
            out_path = os.path.join(REPAIRS_DIR, fn)
            onnx.save(cand_model, out_path)
            improved.append((t, base_r["points"], cand_r["points"], gain, reduction))
            print(f"task{t:03d}: {base_r['points']:.4f} -> {cand_r['points']:.4f} (+{gain:.4f}, params trimmed {reduction})")

        if t % 50 == 0:
            print(f"...task{t:03d} scanned (checked={checked}, improved={len(improved)})")

    print(f"\nChecked {checked} tasks with Conv/ConvTranspose nodes.")
    print(f"Improved: {len(improved)}")
    print(f"Total gain: {sum(g for _,_,_,g,_ in improved):.4f}")

if __name__ == "__main__":
    main()
