import sys, os, copy, json, math
sys.path.insert(0, r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\data\neurogolf_utils")
import onnx, onnxruntime, numpy as np
from onnx import helper, numpy_helper, TensorProto
import neurogolf_utils as ngu

BASE_DIR = r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\baseline_v22"
TASK_DIR = r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\data"
OUT_DIR = r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\repairs"

def find_node_by_output(model, out_name):
    return next(n for n in model.graph.node if n.output[0] == out_name)

def repair_grow(model, node_out_name, deficit_h, deficit_w):
    """ConvTranspose-style: clip end pads to 0 on the node, then zero-Pad the
    output by (deficit_h, deficit_w) at the end of spatial dims."""
    model = copy.deepcopy(model)
    node_idx = next(i for i, n in enumerate(model.graph.node) if n.output[0] == node_out_name)
    node = model.graph.node[node_idx]
    tmp_name = node_out_name + "__pre_pad"
    node.output[0] = tmp_name
    for a in node.attribute:
        if a.name == "pads":
            vals = list(a.ints)
            h_begin, w_begin = vals[0], vals[1]
            a.ints[:] = [h_begin, w_begin, 0, 0]

    pads_init = numpy_helper.from_array(
        np.array([0, 0, 0, 0, 0, 0, deficit_h, deficit_w], dtype=np.int64),
        name=node_out_name + "__pads")
    model.graph.initializer.append(pads_init)

    pad_node = helper.make_node(
        "Pad", [tmp_name, pads_init.name], [node_out_name], mode="constant",
    )
    model.graph.node.insert(node_idx + 1, pad_node)
    return model

def repair_crop(model, node_out_name, clipped_pads, crop_h, crop_w, orig_h, orig_w):
    """Conv-style: clip negative pads to 0, then Slice the output down to the
    original (oh, ow) window starting at (crop_h, crop_w)."""
    model = copy.deepcopy(model)
    node_idx = next(i for i, n in enumerate(model.graph.node) if n.output[0] == node_out_name)
    node = model.graph.node[node_idx]
    tmp_name = node_out_name + "__pre_crop"
    node.output[0] = tmp_name
    for a in node.attribute:
        if a.name == "pads":
            a.ints[:] = clipped_pads

    starts_init = numpy_helper.from_array(np.array([crop_h, crop_w], dtype=np.int64), name=node_out_name + "__starts")
    ends_init = numpy_helper.from_array(np.array([crop_h + orig_h, crop_w + orig_w], dtype=np.int64), name=node_out_name + "__ends")
    axes_init = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name=node_out_name + "__axes")
    model.graph.initializer.extend([starts_init, ends_init, axes_init])

    slice_node = helper.make_node(
        "Slice", [tmp_name, starts_init.name, ends_init.name, axes_init.name], [node_out_name],
    )
    model.graph.node.insert(node_idx + 1, slice_node)
    return model

def repair_pool_input_crop(model, pool_fixes):
    """MaxPool-with-negative-begin-pad style: the negative pad is a pure index
    shift into a kernel_shape=[2,2] window (verified: referenced positions
    never go out of bounds for this parameterization). Rather than slicing a
    near-full-size copy of the input (cheap-looking but actually materializes
    an almost-30x30x10 intermediate tensor per pool -- that's what made the
    first version of this fix the single most expensive task in the whole
    set), extract ONLY the exact 2x2-per-channel receptive field the pool
    reads via a strided Slice, then run a trivial kernel=2/dilation=1/pad=0
    MaxPool over that tiny tensor. Same result, ~200x cheaper per pool."""
    model = copy.deepcopy(model)
    for fix in pool_fixes:
        node_idx = next(i for i, n in enumerate(model.graph.node) if n.output[0] == fix["node_out_name"])
        node = model.graph.node[node_idx]
        data_in = node.input[0]
        crop_h, crop_w = fix["crop_h"], fix["crop_w"]
        dil_h, dil_w = fix["dil_h"], fix["dil_w"]
        field_name = fix["node_out_name"] + "__field"

        starts_init = numpy_helper.from_array(np.array([crop_h, crop_w], dtype=np.int64), name=field_name + "__starts")
        ends_init = numpy_helper.from_array(np.array([crop_h + dil_h + 1, crop_w + dil_w + 1], dtype=np.int64), name=field_name + "__ends")
        axes_init = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name=field_name + "__axes")
        steps_init = numpy_helper.from_array(np.array([dil_h, dil_w], dtype=np.int64), name=field_name + "__steps")
        model.graph.initializer.extend([starts_init, ends_init, axes_init, steps_init])

        slice_node = helper.make_node(
            "Slice", [data_in, starts_init.name, ends_init.name, axes_init.name, steps_init.name], [field_name],
        )
        model.graph.node.insert(node_idx, slice_node)
        node = next(n for n in model.graph.node if n is node)
        node.input[0] = field_name
        kept = [copy.deepcopy(a) for a in node.attribute if a.name not in {"pads", "dilations", "strides"}]
        del node.attribute[:]
        node.attribute.extend(kept)
        node.attribute.extend([
            helper.make_attribute("pads", [0, 0, 0, 0]),
            helper.make_attribute("dilations", [1, 1]),
            helper.make_attribute("strides", [1, 1]),
        ])
    return model

REPAIRS = {
    "task045.onnx": ("grow", dict(node_out_name="output", deficit_h=20, deficit_w=20)),
    "task127.onnx": ("grow", dict(node_out_name="output", deficit_h=23, deficit_w=19)),
    "task384.onnx": ("grow", dict(node_out_name="output", deficit_h=22, deficit_w=20)),
    "task135.onnx": ("crop", dict(node_out_name="output", clipped_pads=[3, 0, 0, 6], crop_h=0, crop_w=6, orig_h=30, orig_w=30)),
    "task146.onnx": ("crop", dict(node_out_name="checks", clipped_pads=[0, 0, 0, 0], crop_h=0, crop_w=0, orig_h=3, orig_w=1)),
    "task149.onnx": ("crop", dict(node_out_name="conv1_out", clipped_pads=[0, 0, 0, 0], crop_h=0, crop_w=0, orig_h=3, orig_w=3)),
    "task240.onnx": ("pool_crop", dict(pool_fixes=[
        dict(node_out_name="s0_pool", crop_h=1, crop_w=1, dil_h=16, dil_w=16),
        dict(node_out_name="s1_pool", crop_h=1, crop_w=3, dil_h=16, dil_w=12),
        dict(node_out_name="s2_pool", crop_h=3, crop_w=3, dil_h=12, dil_w=12),
        dict(node_out_name="s3_pool", crop_h=3, crop_w=5, dil_h=12, dil_w=8),
        dict(node_out_name="s4_pool", crop_h=5, crop_w=5, dil_h=8, dil_w=8),
        dict(node_out_name="s5_pool", crop_h=5, crop_w=7, dil_h=8, dil_w=4),
        dict(node_out_name="s6_pool", crop_h=7, crop_w=7, dil_h=4, dil_w=4),
    ])),
}

def audit_bytes(model_bytes, examples):
    try:
        model = onnx.load_model_from_string(model_bytes)
        sanitized = ngu.sanitize_model(model)
        if sanitized is None:
            return {"status": "sanitize_fail"}
        options = onnxruntime.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.profile_file_prefix = "fixprof"
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
    except Exception as e:
        import traceback
        return {"status": "error", "msg": traceback.format_exc()[-400:]}

def main():
    for fn, (kind, kwargs) in REPAIRS.items():
        task_num = int(fn[4:7])
        path = os.path.join(BASE_DIR, fn)
        model = onnx.load(path)
        if kind == "grow":
            fixed = repair_grow(model, **kwargs)
        elif kind == "crop":
            fixed = repair_crop(model, **kwargs)
        else:
            fixed = repair_pool_input_crop(model, **kwargs)

        # sanity: passes strict checker now
        try:
            onnx.checker.check_model(fixed, full_check=True)
            checker_ok = True
        except Exception as e:
            checker_ok = False
            print(f"{fn}: CHECKER STILL FAILS: {e}")

        examples = json.load(open(os.path.join(TASK_DIR, f"task{task_num:03d}.json")))
        result = audit_bytes(fixed.SerializeToString(), examples)
        print(f"{fn}: checker_ok={checker_ok}  audit={result}")

        if checker_ok and result["status"] == "ok":
            out_path = os.path.join(OUT_DIR, fn)
            onnx.save(fixed, out_path)
            print(f"  -> saved repaired model to {out_path}")

if __name__ == "__main__":
    main()
