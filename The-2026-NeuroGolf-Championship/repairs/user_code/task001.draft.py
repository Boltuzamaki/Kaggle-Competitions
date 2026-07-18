# --- Available: onnx, helper, numpy_helper, np, TensorProto. Assign final model to `model`. ---
# task001: direct ConvTranspose to output, exact background fill
# Expected cost on current scorer may be ~94-ish if final output is not charged.

F = TensorProto.FLOAT
inits = []

def T(name, arr, dtype=np.float32):
    inits.append(numpy_helper.from_array(np.array(arr, dtype=dtype), name))
    return name

T("one", [1.0])

W1 = np.zeros((1, 10, 3, 3), np.float32)
W1[0, 0] = 1.0
T("W1", W1)

x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])

nodes = [
    # grid = input[:, :, :3, :3]
    helper.make_node(
        "Slice",
        ["input"],
        ["grid"],
        starts=[0, 0],
        ends=[3, 3],
        axes=[2, 3],
    ),

    # ch0 = grid[:, 0:1, :, :]
    helper.make_node(
        "Slice",
        ["grid"],
        ["ch0"],
        starts=[0],
        ends=[1],
        axes=[1],
    ),

    # act = 1 - ch0
    helper.make_node("Sub", ["one", "ch0"], ["act"]),

    # tile-control tensor: active branch + inactive/background branch
    helper.make_node("Concat", ["act", "ch0"], ["X3"], axis=1),

    # 3x3 control map -> 10x10 control map
    helper.make_node(
        "Pad",
        ["X3"],
        ["X"],
        pads=[0, 0, 0, 0, 0, 0, 7, 7],
        mode="constant",
        value=0.0,
    ),

    # dynamic kernel:
    # branch 0 stamps the input grid
    # branch 1 stamps background channel 0
    helper.make_node("Concat", ["grid", "W1"], ["W"], axis=0),

    # IMPORTANT: this writes directly to graph output.
    helper.make_node(
        "ConvTranspose",
        ["X", "W"],
        ["output"],
        strides=[3, 3],
    ),
]

graph = helper.make_graph(nodes, "task001", [x], [y], inits)
model = helper.make_model(
    graph,
    ir_version=8,
    opset_imports=[helper.make_opsetid("", 9)],
)