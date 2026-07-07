# ARC task355: choose the base-region colour that owns the most sparse marker pixels.
# Input convention: one-hot [1,10,30,30]. Output: one-hot [1,10,30,30], with the 1x1 answer at [0,0].

import json
import collections
import numpy as np


def solve_355_numpy(grid):
    """Reference solver for raw ARC grids. Returns a 1x1 grid [[colour]]."""
    a = np.asarray(grid, dtype=np.int64)
    vals, cnts = np.unique(a, return_counts=True)

    # Sparse marker/defect colour: the least frequent colour in the image.
    marker = int(vals[np.argmin(cnts)])
    totals = {int(c): int(n) for c, n in zip(vals, cnts)}

    H, W = a.shape
    owner_scores = collections.Counter()

    # For every marker pixel, look in the 4 cardinal directions and vote for
    # the first non-marker colour seen in that direction. This resolves boundary
    # markers and adjacent marker clusters better than just looking at 8-neighbours.
    for r, c in zip(*np.where(a == marker)):
        votes = collections.Counter()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            k = 1
            while 0 <= r + dr * k < H and 0 <= c + dc * k < W:
                colour = int(a[r + dr * k, c + dc * k])
                if colour != marker:
                    votes[colour] += 1
                    break
                k += 1

        if votes:
            best_vote = max(votes.values())
            tied = [colour for colour, v in votes.items() if v == best_vote]
            # Tie-break by larger solid region area.
            owner = max(tied, key=lambda colour: totals[colour])
            owner_scores[owner] += 1

    best_score = max(owner_scores.values())
    tied = [colour for colour, v in owner_scores.items() if v == best_score]
    return [[max(tied, key=lambda colour: totals[colour])]]


def verify_numpy(task_path="/mnt/data/task355.json"):
    with open(task_path, "r") as f:
        task = json.load(f)
    failed = []
    for split in ["train", "test", "arc-gen"]:
        for i, ex in enumerate(task.get(split, [])):
            pred = solve_355_numpy(ex["input"])
            if pred != ex["output"]:
                failed.append((split, i, pred, ex["output"]))
    print(f"checked {sum(len(task.get(s, [])) for s in ['train', 'test', 'arc-gen'])} examples")
    print("failures:", len(failed))
    if failed:
        print(failed[:10])
    return failed


# ---------- ONNX builder ----------
# This mirrors solve_355_numpy using tensor ops only.
# Requires: pip install onnx  (and optionally onnxruntime for runtime verification)

def build_onnx_model():
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    F = TensorProto.FLOAT
    I64 = TensorProto.INT64

    def K(name, value, dtype):
        return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])

    # Directional ray kernels. For each colour independently, the convolution
    # computes sum(4^-distance) along one direction. Because 4^-d is larger than
    # the sum of all farther terms, ArgMax over colours gives the nearest
    # non-marker colour in that direction.
    def ray_kernel(direction):
        if direction in ("L", "R"):
            W = np.zeros((10, 1, 1, 30), np.float32)
            for d in range(1, 30):
                pos = 29 - d if direction == "L" else d
                W[:, 0, 0, pos] = np.float32(4.0 ** -d)
        else:
            W = np.zeros((10, 1, 30, 1), np.float32)
            for d in range(1, 30):
                pos = 29 - d if direction == "U" else d
                W[:, 0, pos, 0] = np.float32(4.0 ** -d)
        return W

    init = [
        K("axesHW", [2, 3], np.int64),
        K("axesC", [1], np.int64),
        K("R10", np.arange(10, dtype=np.int64).reshape(1, 10, 1, 1), np.int64),
        K("zero", [0.0], np.float32),
        K("one", [1.0], np.float32),
        K("big", [10000.0], np.float32),
        K("huge", [1.0e6], np.float32),
        K("outpad", [0, 0, 0, 0, 0, 0, 29, 29], np.int64),
        K("padval", [0.0], np.float32),
        K("WL", ray_kernel("L"), np.float32),
        K("WR", ray_kernel("R"), np.float32),
        K("WU", ray_kernel("U"), np.float32),
        K("WD", ray_kernel("D"), np.float32),
    ]

    nodes = []

    # Count pixels per colour and find the sparse marker colour.
    nodes += [
        helper.make_node("ReduceSum", ["input", "axesHW"], ["cnt"], keepdims=1),
        helper.make_node("Greater", ["cnt", "zero"], ["present"]),
        helper.make_node("Where", ["present", "cnt", "huge"], ["cntMasked"]),
        helper.make_node("ArgMin", ["cntMasked"], ["markerIdx"], axis=1, keepdims=1),
        helper.make_node("Equal", ["R10", "markerIdx"], ["markerEq"]),
        helper.make_node("Cast", ["markerEq"], ["markerOH"], to=F),
        helper.make_node("Sub", ["one", "markerOH"], ["notMarkerOH"]),
        helper.make_node("Mul", ["input", "notMarkerOH"], ["nonMarker"]),
        helper.make_node("Mul", ["input", "markerOH"], ["markerOnly"]),
        helper.make_node("ReduceSum", ["markerOnly", "axesC"], ["markerMask"], keepdims=1),
    ]

    # For each direction: nearest non-marker colour one-hot at every pixel.
    direction_specs = {
        "L": ("WL", [0, 29, 0, 0]),
        "R": ("WR", [0, 0, 0, 29]),
        "U": ("WU", [29, 0, 0, 0]),
        "D": ("WD", [0, 0, 29, 0]),
    }
    vote_names = []
    for d, (Wname, pads) in direction_specs.items():
        score = f"score{d}"
        valid_sum = f"validSum{d}"
        valid_bool = f"validBool{d}"
        valid = f"valid{d}"
        idx = f"idx{d}"
        eq = f"eq{d}"
        oh = f"oh{d}"
        vote = f"vote{d}"
        vote_names.append(vote)
        nodes += [
            helper.make_node("Conv", ["nonMarker", Wname], [score], pads=pads, group=10),
            helper.make_node("ReduceSum", [score, "axesC"], [valid_sum], keepdims=1),
            helper.make_node("Greater", [valid_sum, "zero"], [valid_bool]),
            helper.make_node("Cast", [valid_bool], [valid], to=F),
            helper.make_node("ArgMax", [score], [idx], axis=1, keepdims=1),
            helper.make_node("Equal", ["R10", idx], [eq]),
            helper.make_node("Cast", [eq], [oh], to=F),
            helper.make_node("Mul", [oh, valid], [vote]),
        ]

    nodes += [
        helper.make_node("Add", ["voteL", "voteR"], ["votesLR"]),
        helper.make_node("Add", ["voteU", "voteD"], ["votesUD"]),
        helper.make_node("Add", ["votesLR", "votesUD"], ["votes"]),

        # Owner per marker: most cardinal ray votes; ties by larger colour area.
        helper.make_node("Mul", ["votes", "big"], ["votesBig"]),
        helper.make_node("Add", ["votesBig", "cnt"], ["pixelScore"]),
        helper.make_node("ArgMax", ["pixelScore"], ["ownerIdx"], axis=1, keepdims=1),
        helper.make_node("Equal", ["R10", "ownerIdx"], ["ownerEq"]),
        helper.make_node("Cast", ["ownerEq"], ["ownerOH"], to=F),
        helper.make_node("Mul", ["ownerOH", "markerMask"], ["ownedMarkers"]),

        # Count marker ownership per colour; final answer is the max owner count.
        helper.make_node("ReduceSum", ["ownedMarkers", "axesHW"], ["ownedCnt"], keepdims=1),
        helper.make_node("Mul", ["ownedCnt", "big"], ["ownedBig"]),
        helper.make_node("Add", ["ownedBig", "cnt"], ["finalScore"]),
        helper.make_node("ArgMax", ["finalScore"], ["outIdx"], axis=1, keepdims=1),
        helper.make_node("Equal", ["R10", "outIdx"], ["outEq"]),
        helper.make_node("Cast", ["outEq"], ["out1x1"], to=F),
        helper.make_node("Pad", ["out1x1", "outpad", "padval"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task355", [x], [y], init)
    model = helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


# Required by repairs/check_and_promote.py / load_model_from_path.
# The checker execs this file and reads the global `model` variable.
model = build_onnx_model()
