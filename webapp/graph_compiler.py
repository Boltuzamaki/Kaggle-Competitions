"""Compile a JSON node-graph (built by the Visual Graph editor) into an onnx.ModelProto.

Graph JSON shape:
  {"nodes": [{"id": "n1", "op": "Input", "attrs": {}}, ...],
   "edges": [{"from": "n1", "fromPort": 0, "to": "n2", "toPort": 0}, ...]}

Every node has exactly one logical output (its "out" tensor); ports on the input
side are positional (port 0, 1, 2, ...) per the OP_INPUTS arity below. The graph
must contain exactly one Input node and exactly one Output node. Static shapes
only: contract is [1,10,30,30] FLOAT in, [1,10,30,30] FLOAT out, matching the
plain-Python code path in app.py.
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

DEFAULT_GRAPH = {
    "nodes": [
        {"id": "in", "op": "Input", "x": 40, "y": 140, "attrs": {}},
        {"id": "out", "op": "Output", "x": 380, "y": 140, "attrs": {}},
    ],
    "edges": [{"from": "in", "fromPort": 0, "to": "out", "toPort": 0}],
}

# number of ordered input ports each op consumes ('concat' is variable, resolved from edges)
OP_INPUTS = {
    "Input": 0, "Output": 1, "Constant": 0, "Cast": 1, "Identity": 1,
    "Equal": 2, "Greater": 2, "Less": 2, "Not": 1, "And": 2, "Or": 2, "Where": 3,
    "Add": 2, "Sub": 2, "Mul": 2, "Div": 2,
    "ReduceSum": 1, "ArgMax": 1,
    "RowIndex": 0, "ColIndex": 0,
    "Slice": 1, "Pad": 1, "Concat": None, "Transpose": 1, "Tile": 1, "Resize": 1, "Conv": 1,
}

DTYPE = {"FLOAT": TensorProto.FLOAT, "INT64": TensorProto.INT64,
         "INT32": TensorProto.INT32, "BOOL": TensorProto.BOOL}


class GraphCompileError(ValueError):
    pass


def _row_col_index_array(op):
    rows = np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)
    cols = np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)
    src = rows if op == "RowIndex" else cols
    return np.ascontiguousarray(np.broadcast_to(src, (1, 1, 30, 30)))


def _constant_array(attrs):
    shape = attrs.get("shape", [1, 10, 30, 30])
    # dtype defaults to float32 for hand-authored Constant nodes (feeding elementwise math);
    # imported initializers (e.g. Pad's int64 "pads") carry their real dtype explicitly.
    dtype = np.dtype(attrs["dtype"]) if attrs.get("dtype") else np.float32
    if "value" in attrs:
        return np.array(attrs["value"], dtype=dtype).reshape(shape)
    fill = attrs.get("fill", 0.0)
    return np.full(shape, fill, dtype=dtype)


def _to_dtype(v):
    if isinstance(v, str):
        return DTYPE.get(v.upper(), TensorProto.FLOAT)
    try:
        return int(v)
    except (TypeError, ValueError):
        return TensorProto.FLOAT


def _build_generic(op, ins, outs, attrs):
    """Faithful passthrough for a node imported from a real ONNX model: real wired
    ports, real outputs (possibly >1), real onnx attributes verbatim — no curated-op
    synthesis/defaults applied."""
    try:
        return [helper.make_node(op, ins, outs, **attrs)]
    except Exception as ex:
        raise GraphCompileError(f"imported op '{op}' failed to rebuild: {ex}")


def _build_op(op, nid, attrs, ins, outs, initializers, newname, imported=False):
    if imported:
        return _build_generic(op, ins, outs, attrs)
    out = outs[0]
    if op in ("Output", "Identity"):
        return [helper.make_node("Identity", [ins[0]], [out])]
    if op == "Cast":
        return [helper.make_node("Cast", [ins[0]], [out], to=_to_dtype(attrs.get("to", "FLOAT")))]
    if op in ("Equal", "Greater", "Less", "And", "Or", "Add", "Sub", "Mul", "Div"):
        return [helper.make_node(op, [ins[0], ins[1]], [out])]
    if op == "Not":
        return [helper.make_node("Not", [ins[0]], [out])]
    if op == "Where":
        return [helper.make_node("Where", [ins[0], ins[1], ins[2]], [out])]
    if op == "ReduceSum":
        axes = attrs.get("axes", [1]); keepdims = int(attrs.get("keepdims", 1))
        return [helper.make_node("ReduceSum", [ins[0]], [out], axes=axes, keepdims=keepdims)]
    if op == "ArgMax":
        axis = int(attrs.get("axis", 1)); keepdims = int(attrs.get("keepdims", 1))
        return [helper.make_node("ArgMax", [ins[0]], [out], axis=axis, keepdims=keepdims)]
    if op == "Transpose":
        return [helper.make_node("Transpose", [ins[0]], [out], perm=attrs.get("perm", [0, 1, 2, 3]))]
    if op == "Concat":
        return [helper.make_node("Concat", ins, [out], axis=int(attrs.get("axis", 1)))]
    if op == "Constant":
        initializers.append(numpy_helper.from_array(_constant_array(attrs), name=out))
        return []
    if op in ("RowIndex", "ColIndex"):
        initializers.append(numpy_helper.from_array(_row_col_index_array(op), name=out))
        return []
    if op == "Slice":
        starts = attrs.get("starts", [0, 0, 0, 0]); ends = attrs.get("ends", [1, 10, 30, 30])
        axes = attrs.get("axes", list(range(len(starts)))); steps = attrs.get("steps", [1] * len(starts))
        sn, en, an, stn = (newname(nid + s) for s in ("_starts", "_ends", "_axes", "_steps"))
        initializers.append(numpy_helper.from_array(np.array(starts, dtype=np.int64), name=sn))
        initializers.append(numpy_helper.from_array(np.array(ends, dtype=np.int64), name=en))
        initializers.append(numpy_helper.from_array(np.array(axes, dtype=np.int64), name=an))
        initializers.append(numpy_helper.from_array(np.array(steps, dtype=np.int64), name=stn))
        return [helper.make_node("Slice", [ins[0], sn, en, an, stn], [out])]
    if op == "Pad":
        pads = attrs.get("pads", [0, 0, 0, 0, 0, 0, 0, 0]); value = float(attrs.get("value", 0.0))
        mode = attrs.get("mode", "constant")
        pn, vn = newname(nid + "_pads"), newname(nid + "_value")
        initializers.append(numpy_helper.from_array(np.array(pads, dtype=np.int64), name=pn))
        initializers.append(numpy_helper.from_array(np.array(value, dtype=np.float32), name=vn))
        return [helper.make_node("Pad", [ins[0], pn, vn], [out], mode=mode)]
    if op == "Tile":
        repeats = attrs.get("repeats", [1, 1, 1, 1])
        rn = newname(nid + "_repeats")
        initializers.append(numpy_helper.from_array(np.array(repeats, dtype=np.int64), name=rn))
        return [helper.make_node("Tile", [ins[0], rn], [out])]
    if op == "Resize":
        sizes = attrs.get("sizes", [1, 10, 30, 30]); mode = attrs.get("mode", "nearest")
        roi_n, scales_n, sizes_n = (newname(nid + s) for s in ("_roi", "_scales", "_sizes"))
        initializers.append(numpy_helper.from_array(np.array([], dtype=np.float32), name=roi_n))
        initializers.append(numpy_helper.from_array(np.array([], dtype=np.float32), name=scales_n))
        initializers.append(numpy_helper.from_array(np.array(sizes, dtype=np.int64), name=sizes_n))
        return [helper.make_node("Resize", [ins[0], roi_n, scales_n, sizes_n], [out], mode=mode)]
    if op == "Conv":
        weight = attrs.get("weight")
        if weight is None:
            raise GraphCompileError(f"Conv node {nid} needs attrs.weight: a nested [Cout,Cin,kh,kw] list")
        w = np.array(weight, dtype=np.float32)
        if w.ndim != 4:
            raise GraphCompileError(f"Conv node {nid} weight must be 4D [Cout,Cin,kh,kw], got shape {w.shape}")
        pads = attrs.get("pads", [0, 0, 0, 0]); strides = attrs.get("strides", [1, 1])
        wn = newname(nid + "_weight")
        initializers.append(numpy_helper.from_array(w, name=wn))
        return [helper.make_node("Conv", [ins[0], wn], [out],
                                  kernel_shape=[w.shape[2], w.shape[3]], pads=pads, strides=strides)]
    raise GraphCompileError(f"unsupported op: {op}")


def compile_graph(graph):
    """graph: dict with 'nodes' and 'edges' (see module docstring). Returns onnx.ModelProto."""
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    if not nodes:
        raise GraphCompileError("graph has no nodes")
    by_id = {n["id"]: n for n in nodes}
    if len(by_id) != len(nodes):
        raise GraphCompileError("duplicate node ids")
    input_nodes = [n for n in nodes if n["op"] == "Input"]
    output_nodes = [n for n in nodes if n["op"] == "Output"]
    if len(input_nodes) != 1:
        raise GraphCompileError(f"graph must have exactly one Input node (found {len(input_nodes)})")
    if len(output_nodes) != 1:
        raise GraphCompileError(f"graph must have exactly one Output node (found {len(output_nodes)})")
    input_id, output_id = input_nodes[0]["id"], output_nodes[0]["id"]

    incoming = {}  # node_id -> {port_idx: (src_id, src_port)}
    for e in edges:
        src, dst = e.get("from"), e.get("to")
        sport, dport = int(e.get("fromPort", 0)), int(e.get("toPort", 0))
        if src not in by_id or dst not in by_id:
            raise GraphCompileError(f"edge references unknown node ({src} -> {dst})")
        incoming.setdefault(dst, {})[dport] = (src, sport)

    deps = {nid: sorted({src for (src, _) in incoming.get(nid, {}).values()}) for nid in by_id}
    order, perm_mark, temp_mark = [], set(), set()

    def visit(nid):
        if nid in perm_mark:
            return
        if nid in temp_mark:
            raise GraphCompileError(f"cycle detected at node {nid}")
        temp_mark.add(nid)
        for d in deps[nid]:
            visit(d)
        temp_mark.discard(nid); perm_mark.add(nid); order.append(nid)

    for nid in by_id:
        visit(nid)

    tensor_name = {input_id: ["input"]}  # node_id -> list of its output tensor names (by output port)
    onnx_nodes, initializers = [], []
    ctr = [0]

    def newname(prefix):
        ctr[0] += 1
        return f"{prefix}_{ctr[0]}"

    for nid in order:
        if nid == input_id:
            continue
        node = by_id[nid]
        op = node["op"]
        attrs = node.get("attrs") or {}
        imported = bool(node.get("imported"))
        ins_map = incoming.get(nid, {})
        if "n_in" in node:
            n_expected = node["n_in"]
        elif op == "Concat":
            n_expected = max(2, len(ins_map))
        else:
            arity = OP_INPUTS.get(op)
            if arity is None:
                raise GraphCompileError(f"unsupported op: {op}")
            n_expected = arity
        in_tensors = []
        for p in range(n_expected):
            if p not in ins_map:
                raise GraphCompileError(f"node {nid} ({op}) is missing a wire into input port {p}")
            src, sport = ins_map[p]
            if src not in tensor_name:
                raise GraphCompileError(f"node {nid} depends on {src} which has no resolved output "
                                         f"(disconnected from Input?)")
            if sport >= len(tensor_name[src]):
                raise GraphCompileError(f"node {nid} reads output port {sport} of {src}, which only "
                                         f"has {len(tensor_name[src])} output(s)")
            in_tensors.append(tensor_name[src][sport])
        n_out = 1 if nid == output_id else node.get("n_out", 1)
        out_names = ["output"] if nid == output_id else (
            [newname(nid)] if n_out == 1 else [newname(f"{nid}_o{i}") for i in range(n_out)])
        onnx_nodes.extend(_build_op(op, nid, attrs, in_tensors, out_names, initializers, newname, imported))
        tensor_name[nid] = out_names

    if output_id not in tensor_name or tensor_name[output_id] != ["output"]:
        raise GraphCompileError("Output node produced no tensor")

    # dtype normally FLOAT for hand-authored graphs; an imported graph may legitimately
    # declare FLOAT16 (a cost-saving trick some solved models use) — preserve it on round-trip.
    x = helper.make_tensor_value_info("input", _to_dtype(graph.get("input_dtype", "FLOAT")), [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", _to_dtype(graph.get("output_dtype", "FLOAT")), [1, 10, 30, 30])
    g = helper.make_graph(onnx_nodes, "usergraph", [x], [y], initializers)
    return helper.make_model(g, ir_version=10, opset_imports=[helper.make_opsetid("", 12)])


def _attr_to_value(a):
    v = helper.get_attribute_value(a)
    if isinstance(v, onnx.TensorProto):
        return numpy_helper.to_array(v).tolist()
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, list):
        return [x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x for x in v]
    return v


def onnx_to_graph(model):
    """Decompile an onnx.ModelProto into the editor's {nodes, edges} JSON, preserving the
    model's real wiring and attributes exactly, so it round-trips through compile_graph()
    (used to show/edit an already-solved task's actual saved ONNX in the Visual Graph tab)."""
    g = model.graph
    nodes, edges = [], []
    producer = {}  # real onnx tensor name -> (node_id, output_port_index)
    nodes.append({"id": "in", "op": "Input", "attrs": {}, "x": 30, "y": 260})
    producer[g.input[0].name] = ("in", 0)

    ctr = [0]

    def newid(prefix):
        ctr[0] += 1
        return f"{prefix}{ctr[0]}"

    ypos = 20
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        nid = newid("const")
        nodes.append({"id": nid, "op": "Constant",
                      "attrs": {"shape": list(arr.shape), "value": arr.tolist(), "dtype": arr.dtype.name},
                      "x": 30, "y": ypos})
        ypos += 90
        producer[init.name] = (nid, 0)

    xpos = 260
    for i, node in enumerate(g.node):
        if node.op_type == "Constant":
            val_attr = next((a for a in node.attribute if a.name == "value"), None)
            arr = numpy_helper.to_array(helper.get_attribute_value(val_attr)) if val_attr is not None \
                else np.zeros((1,), dtype=np.float32)
            nid = newid("const")
            nodes.append({"id": nid, "op": "Constant",
                          "attrs": {"shape": list(arr.shape), "value": arr.tolist(), "dtype": arr.dtype.name},
                          "x": 30, "y": ypos})
            ypos += 90
        else:
            nid = f"op{i}"
            attrs = {a.name: _attr_to_value(a) for a in node.attribute}
            node_json = {"id": nid, "op": node.op_type, "attrs": attrs, "imported": True,
                         "n_in": len(node.input), "x": xpos, "y": 40 + (i % 7) * 110}
            if len(node.output) != 1:
                node_json["n_out"] = len(node.output)
            nodes.append(node_json)
            xpos += 170
            for p, inp in enumerate(node.input):
                if not inp:
                    continue
                src = producer.get(inp)
                if src is None:
                    continue
                edges.append({"from": src[0], "fromPort": src[1], "to": nid, "toPort": p})
        for out_idx, out_name in enumerate(node.output):
            if out_name:
                producer[out_name] = (nid, out_idx)

    out_id = "out"
    nodes.append({"id": out_id, "op": "Output", "attrs": {}, "x": xpos + 120, "y": 260})
    src = producer.get(g.output[0].name)
    if src is not None:
        edges.append({"from": src[0], "fromPort": src[1], "to": out_id, "toPort": 0})
    return {"nodes": nodes, "edges": edges,
            "input_dtype": g.input[0].type.tensor_type.elem_type,
            "output_dtype": g.output[0].type.tensor_type.elem_type}
