# --- task286: maze corridor 2-colouring by reachability + checkerboard parity.
# Colours 0=background, 8=walls. A short seed of two colours (e.g. 2/3) sits in one corridor.
# Rule: every non-wall cell 4-connected-reachable from a seed cell is painted; a 4-connected
# grid is bipartite by (r+c) parity, so the paint is a checkerboard of the two seed colours,
# phase fixed so the existing seeds keep their colours. Unreached cells & walls stay unchanged.
# Reachability is a multi-source flood-fill, unrolled (ONNX has no Loop) as plus-Conv dilations.
import numpy as np
F = TensorProto.FLOAT
ITERS = 80
SEED_CH = [1, 2, 3, 4, 5, 6, 7, 9]          # any colour that can act as a seed (not 0 bg, not 8 wall)
x = helper.make_tensor_value_info('input',  F, [1, 10, 30, 30])
y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])
def K(n, a): return numpy_helper.from_array(a, n)
rr, cc = np.meshgrid(np.arange(30), np.arange(30), indexing='ij')
even = ((rr + cc) % 2 == 0).astype(np.float32).reshape(1, 1, 30, 30)
odd  = ((rr + cc) % 2 == 1).astype(np.float32).reshape(1, 1, 30, 30)
inits = [
    K('one', np.array(1.0, np.float32)), K('zero', np.array(0.0, np.float32)),
    K('plus', np.array([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], np.float32)),
    K('C0', even), K('C1', odd), K('ax1', np.array([1], np.int64)),
]
for k in range(10):
    inits.append(K('s%d' % k, np.array([k], np.int64)))
    inits.append(K('e%d' % k, np.array([k + 1], np.int64)))

nodes = []
for k in range(10):                                                # split channels
    nodes.append(helper.make_node('Slice', ['input', 's%d' % k, 'e%d' % k, 'ax1'], ['ch%d' % k]))
nodes.append(helper.make_node('ReduceSum', ['input'], ['filled'], axes=[1], keepdims=1))
nodes.append(helper.make_node('Sub', ['filled', 'ch8'], ['free']))  # non-wall in-grid cells
nodes.append(helper.make_node('Sub', ['free', 'ch0'], ['seed0']))   # cells already a seed colour

prev = 'seed0'
for j in range(1, ITERS + 1):                                       # unrolled multi-source BFS
    nodes += [
        helper.make_node('Conv', [prev, 'plus'], ['d%d' % j], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node('Clip', ['d%d' % j, 'zero', 'one'], ['dc%d' % j]),
        helper.make_node('Mul', ['dc%d' % j, 'free'], ['r%d' % j]),
    ]
    prev = 'r%d' % j
reach = prev                                                        # reachable free cells

out_channels = {}
for k in SEED_CH:                                                   # paint each seed colour by parity
    ck = 'ch%d' % k
    nodes += [
        helper.make_node('Mul', [ck, 'C0'], ['ev%d' % k]),
        helper.make_node('ReduceSum', ['ev%d' % k], ['evs%d' % k], axes=[1, 2, 3], keepdims=1),
        helper.make_node('Clip', ['evs%d' % k, 'zero', 'one'], ['ev1_%d' % k]),
        helper.make_node('Mul', [ck, 'C1'], ['od%d' % k]),
        helper.make_node('ReduceSum', ['od%d' % k], ['ods%d' % k], axes=[1, 2, 3], keepdims=1),
        helper.make_node('Clip', ['ods%d' % k, 'zero', 'one'], ['od1_%d' % k]),
        helper.make_node('Mul', ['ev1_%d' % k, 'C0'], ['fpa%d' % k]),
        helper.make_node('Mul', ['od1_%d' % k, 'C1'], ['fpb%d' % k]),
        helper.make_node('Add', ['fpa%d' % k, 'fpb%d' % k], ['fp%d' % k]),
        helper.make_node('Mul', [reach, 'fp%d' % k], ['paint%d' % k]),
    ]
    out_channels[k] = 'paint%d' % k
nodes += [                                                          # unreached bg stays background
    helper.make_node('Sub', ['one', reach], ['notreach']),
    helper.make_node('Mul', ['ch0', 'notreach'], ['out0']),
]
out_channels[0] = 'out0'
out_channels[8] = 'ch8'                                             # walls unchanged
nodes.append(helper.make_node('Concat', [out_channels[k] for k in range(10)], ['output'], axis=1))

model = helper.make_model(helper.make_graph(nodes, 'task286', [x], [y], inits),
                          ir_version=10, opset_imports=[helper.make_opsetid('', 12)])
