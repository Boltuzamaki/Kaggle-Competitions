"""Search hybrid segmented/local propagation schedules for task002.

This is a research helper.  It exposes the packed state immediately before the
late propagation tail and compares exact aggregate states over every official
example.  Costs count charged uint32[1,1,20,1] tensors/nodes.
"""

from __future__ import annotations

import copy
import heapq
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "data" / "neurogolf_utils"))
import neurogolf_utils as ngu  # noqa: E402


DIRECTIONS = "LRUD"


def expose_states():
    model = copy.deepcopy(onnx.load(ROOT / "repairs" / "task002.onnx"))
    for name in ("safe_name_25", "safe_name_78", "safe_name_166"):
        model.graph.output.append(
            helper.make_tensor_value_info(name, TensorProto.UINT32, [1, 1, 20, 1])
        )
    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    task = json.loads((ROOT / "data" / "task002.json").read_text())
    examples = task["train"] + task["test"] + task["arc-gen"]
    values = [[], [], []]
    for example in examples:
        batch = ngu.convert_to_numpy(example)
        outputs = session.run(
            ["safe_name_25", "safe_name_78", "safe_name_166"],
            {"input": batch["input"]},
        )
        for target, output in zip(values, outputs):
            target.append(output.reshape(20))
    return tuple(np.stack(value).astype(np.uint32) for value in values)


def move(state, direction):
    if direction == "L":
        return state << np.uint32(1)
    if direction == "R":
        return state >> np.uint32(1)
    if direction == "U":
        return state[:, np.r_[0, 0:19]]
    if direction == "D":
        return state[:, np.r_[1:20, 19]]
    raise ValueError(direction)


def local(state, allowed, directions):
    result = state.copy()
    for direction in directions:
        result |= move(state, direction)
    return result & allowed


def segmented(state, allowed, direction, offsets):
    """Parallel prefix propagation matching the ONNX doubling network."""
    reach = state.copy()
    connection = allowed.copy()
    for offset in offsets:
        if direction == "L":
            shifted_reach = reach << np.uint32(offset)
            shifted_connection = connection << np.uint32(offset)
        elif direction == "R":
            shifted_reach = reach >> np.uint32(offset)
            shifted_connection = connection >> np.uint32(offset)
        elif direction == "U":
            indices = np.maximum(np.arange(20) - offset, 0)
            shifted_reach = reach[:, indices]
            shifted_connection = connection[:, indices]
        elif direction == "D":
            indices = np.minimum(np.arange(20) + offset, 19)
            shifted_reach = reach[:, indices]
            shifted_connection = connection[:, indices]
        else:
            raise ValueError(direction)
        reach |= connection & shifted_reach
        connection &= shifted_connection
    return reach & allowed


def key(state):
    return state.tobytes()


def exact(state, target):
    return np.array_equal(state, target)


def local_dijkstra(start, allowed, target, limit=68):
    """Find the least-cost exact tail made from arbitrary direction subsets."""
    subsets = []
    for size in range(1, 5):
        for subset in combinations(DIRECTIONS, size):
            subsets.append((2 * size + 1, "".join(subset)))
    start_key = key(start)
    queue = [(0, 0, start_key, start, [])]
    best = {start_key: 0}
    serial = 1
    while queue:
        cost, _, state_key, state, path = heapq.heappop(queue)
        if best[state_key] != cost:
            continue
        if exact(state, target):
            return cost, path
        for step_cost, subset in subsets:
            new_cost = cost + step_cost
            if new_cost > limit:
                continue
            new_state = local(state, allowed, subset)
            new_key = key(new_state)
            if new_key == state_key or best.get(new_key, limit + 1) <= new_cost:
                continue
            best[new_key] = new_cost
            heapq.heappush(
                queue,
                (new_cost, serial, new_key, new_state, path + [subset]),
            )
            serial += 1
    return None


def optimize_fixed_rounds(start, allowed, target, rounds, max_cost=None):
    """Exact DP over every direction subset that can survive alone.

    Monotonicity makes the individual-stage filter lossless: if deleting a
    direction with every other stage full misses the target, further deletions
    elsewhere cannot repair it.
    """
    subsets = []
    for size in range(1, 5):
        for subset in combinations(DIRECTIONS, size):
            subsets.append((2 * size + 1, "".join(subset)))

    full_states = [start]
    for _ in range(rounds):
        full_states.append(local(full_states[-1], allowed, DIRECTIONS))
    assert exact(full_states[-1], target)

    viable = []
    for stage in range(rounds):
        options = []
        for cost, subset in subsets:
            probe = local(full_states[stage], allowed, subset)
            for _ in range(stage + 1, rounds):
                probe = local(probe, allowed, DIRECTIONS)
            if exact(probe, target):
                options.append((cost, subset))
        viable.append(options)

    states = {key(start): (0, start, [])}
    for stage_index, options in enumerate(viable):
        next_states = {}
        for old_cost, state, path in states.values():
            for step_cost, subset in options:
                new_state = local(state, allowed, subset)
                new_key = key(new_state)
                new_cost = old_cost + step_cost
                remaining_floor = 3 * (rounds - stage_index - 1)
                if max_cost is not None and new_cost + remaining_floor > max_cost:
                    continue
                previous = next_states.get(new_key)
                if previous is None or new_cost < previous[0]:
                    next_states[new_key] = (new_cost, new_state, path + [subset])
        states = next_states
    solved = [value for value in states.values() if exact(value[1], target)]
    result = min(solved, key=lambda value: value[0]) if solved else None
    return result, [[subset for _, subset in options] for options in viable]


def main():
    allowed, start, target = expose_states()
    candidates = []
    current = start
    total_cost = 0
    # Existing scan templates use offsets 1,2,4 and cost 13 per direction.
    for label, directions in (("V", "UD"), ("H", "LR"), ("V", "UD")):
        for direction in directions:
            current = segmented(current, allowed, direction, (1, 2, 4))
            total_cost += 13
        candidates.append((total_cost, label, current.copy()))

    for prefix_cost, label, state in candidates:
        missing = int(np.unpackbits((state ^ target).view(np.uint8)).sum())
        probe = state
        full_rounds = None
        for round_index in range(11):
            if exact(probe, target):
                full_rounds = round_index
                break
            probe = local(probe, allowed, DIRECTIONS)
        print(
            f"prefix={label:5s} cost={prefix_cost:2d} missing_bits={missing:4d} "
            f"full_rounds={full_rounds}"
        )

    # One-direction scans: first report the number of ordinary full rounds.
    for direction in DIRECTIONS:
        state = segmented(start, allowed, direction, (1, 2, 4))
        probe = state
        full_rounds = None
        for round_index in range(11):
            if exact(probe, target):
                full_rounds = round_index
                break
            probe = local(probe, allowed, DIRECTIONS)
        missing = int(np.unpackbits((state ^ target).view(np.uint8)).sum())
        print(
            f"prefix={direction}7    cost=13 missing_bits={missing:4d} "
            f"full_rounds={full_rounds}"
        )
        if direction in "UD":
            result, _ = optimize_fixed_rounds(
                state, allowed, target, full_rounds, max_cost=54
            )
            print(
                f"optimized prefix={direction}7 result="
                f"{None if result is None else (result[0], result[2])}"
            )


if __name__ == "__main__":
    main()
