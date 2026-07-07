# ARC task285 -- genuine from-scratch static-ONNX solve (Claude, verified vs train+test+arc-gen).
#
# RULE (verified numpy reference kept below for provenance):
#   The grid decomposes into 8-connected multi-colour components. Within a component the
#   colour with the most cells is the "source" shape; the remaining colours are partial
#   seeds of mirrored copies of the source arranged in an (up to) 2x2 block layout. Each
#   partial fragment is completed into the full mirrored copy (H-flip / V-flip / both),
#   coloured by the fragment colour seeded in that quadrant.
#
# STATIC-ONNX IMPLEMENTATION (no Loop/Scan/NonZero/If, no lookup tables, no memorised
# outputs -- only bounded vectorised tensor ops on the fixed 30x30 one-hot grid):
#   1. presence M = sum of colour channels 1..9; valid-grid mask = max over all 10 channels;
#      colour-id grid col = sum_k k*channel_k.
#   2. 8-connected component label R = min linear-index flood (unrolled 16x max/min-pool).
#   3. per-cell same-colour-same-component count via a 900x900 equality matrix; the source
#      cells are those whose count equals the per-component maximum (max-flood).
#   4. source bounding box (r0,r1,c0,c1) broadcast across each component by flooding the
#      row/col extremes seeded on the source cells.
#   5. classify each fragment cell into the H / V / HV quadrant relative to the source bbox;
#      flood per-quadrant existence flags and quadrant colours across the component.
#   6. reflection axes from the bbox edges (2*c1+1 / 2*c0-1 / 2*r1+1 / 2*r0-1); SCATTER each
#      source cell to its reflected target in every active quadrant (ScatterND, with a dump
#      row for inactive/out-of-range writes) to paint the completed copies.
#   7. rebuild the one-hot output and mask off the padding region with the valid-grid mask.
#
# --- verified numpy rule reference (ground truth) -------------------------------------
# import numpy as np
# from collections import defaultdict
# from itertools import product
# def solve_285_numpy(grid):
#     # 8-connected components -> source = max-count colour; other colours are partial
#     # mirror copies in a 2x2 layout; complete them with H/V/180 flips of the source.
#     ...  (see git history; behaviour reproduced exactly by the graph below)
# --------------------------------------------------------------------------------------

F = TensorProto.FLOAT
I64 = TensorProto.INT64
ITERS = 16
INF = 1e9

nodes = []
inits = []


def K(name, arr):
    inits.append(numpy_helper.from_array(np.asarray(arr), name))
    return name


def n(op, ins, outs, **kw):
    if isinstance(outs, str):
        outs = [outs]
    nodes.append(helper.make_node(op, ins if isinstance(ins, list) else [ins], outs, **kw))
    return outs[0]


# ---- constants ----
rr = np.zeros((1, 1, 30, 30), np.float32)
cc = np.zeros((1, 1, 30, 30), np.float32)
for i in range(30):
    rr[0, 0, i, :] = i
    cc[0, 0, :, i] = i
K('RR', rr)
K('CC', cc)
K('IDX', (rr * 30 + cc).astype(np.float32))
K('cw', np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
K('INF', np.float32(INF))
K('NINF', np.float32(-INF))
K('neg1', np.float32(-1.0))
K('zero', np.float32(0.0))
K('half', np.float32(0.5))
K('two', np.float32(2.0))
K('one', np.float32(1.0))
K('c29', np.float32(29.0))
K('c30f', np.float32(30.0))
K('pads1', np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64))
K('padrow', np.array([0, 0, 1, 0], np.int64))
K('s1', np.array([1], np.int64))
K('s10', np.array([10], np.int64))
K('ax1', np.array([1], np.int64))
K('sh_col1', np.array([900, 1], np.int64))
K('sh_row1', np.array([1, 900], np.int64))
K('sh_900', np.array([900], np.int64))
K('sh_9001', np.array([900, 1], np.int64))
K('sh_grid', np.array([1, 1, 30, 30], np.int64))
K('sh_3030', np.array([30, 30], np.int64))
K('r0slc', np.array([0], np.int64))
K('r30slc', np.array([30], np.int64))
K('ax0', np.array([0], np.int64))

# ---- basic fields ----
x = 'input'
n('ReduceMax', [x], 'validmask', axes=[1], keepdims=1)
n('Slice', [x, 's1', 's10', 'ax1'], 'ch19')
n('ReduceSum', ['ch19'], 'M', axes=[1], keepdims=1)
n('Greater', ['M', 'half'], 'pres')
n('Mul', [x, 'cw'], 'xw')
n('ReduceSum', ['xw'], 'col', axes=[1], keepdims=1)


def maxpool_nb(v, pfx):
    p = n('Pad', [v, 'pads1', 'NINF'], pfx + '_pad')
    return n('MaxPool', [p], pfx + '_mp', kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0])


def maxflood(seed, pfx):
    st = seed
    for i in range(ITERS):
        mp = maxpool_nb(st, '%s_%d' % (pfx, i))
        st = n('Where', ['pres', mp, st], '%s_st%d' % (pfx, i))
    return st


def minflood(seed, pfx):
    st = seed
    for i in range(ITERS):
        neg = n('Mul', [st, 'neg1'], '%s_ng%d' % (pfx, i))
        mp = maxpool_nb(neg, '%s_%d' % (pfx, i))
        nb = n('Mul', [mp, 'neg1'], '%s_nb%d' % (pfx, i))
        st = n('Where', ['pres', nb, st], '%s_st%d' % (pfx, i))
    return st


# component label R (8-connected min linear index)
n('Where', ['pres', 'IDX', 'INF'], 'seedR')
R = minflood('seedR', 'R')

# same-colour-same-component count matrix -> source detection
n('Reshape', [R, 'sh_col1'], 'Rc')
n('Reshape', [R, 'sh_row1'], 'Rr')
n('Reshape', ['col', 'sh_col1'], 'Cc2')
n('Reshape', ['col', 'sh_row1'], 'Cr')
n('Cast', ['pres'], 'presf', to=F)
n('Reshape', ['presf', 'sh_row1'], 'presrow')
n('Equal', ['Rc', 'Rr'], 'eqR')
n('Equal', ['Cc2', 'Cr'], 'eqC')
n('Cast', ['eqR'], 'eqRf', to=F)
n('Cast', ['eqC'], 'eqCf', to=F)
n('Mul', ['eqRf', 'eqCf'], 'eqRC')
n('Mul', ['eqRC', 'presrow'], 'm3')
n('ReduceSum', ['m3'], 'cntflat', axes=[1], keepdims=0)
n('Reshape', ['cntflat', 'sh_grid'], 'cnt')
n('Where', ['pres', 'cnt', 'NINF'], 'seedCM')
compmax = maxflood('seedCM', 'cm')
n('Sub', [compmax, 'half'], 'cmm')
n('Greater', ['cnt', 'cmm'], 'ismax')
n('And', ['pres', 'ismax'], 'SRC')

# source bounding box, broadcast across component
n('Where', ['SRC', 'RR', 'INF'], 'seed_r0')
r0 = minflood('seed_r0', 'r0')
n('Where', ['SRC', 'CC', 'INF'], 'seed_c0')
c0 = minflood('seed_c0', 'c0')
n('Where', ['SRC', 'RR', 'NINF'], 'seed_r1')
r1 = maxflood('seed_r1', 'r1')
n('Where', ['SRC', 'CC', 'NINF'], 'seed_c1')
c1 = maxflood('seed_c1', 'c1')

# fragments and reflection-side flags
n('Not', ['SRC'], 'notsrc')
n('And', ['pres', 'notsrc'], 'FRAG')


def flag(cond_bool, pfx):
    cf = n('Cast', [cond_bool], pfx + '_cf', to=F)
    seed = n('Where', ['pres', cf, 'NINF'], pfx + '_sd')
    fl = maxflood(seed, pfx + '_fl')
    return n('Greater', [fl, 'half'], pfx + '_b')


n('Greater', ['CC', c1], 'gtc1')
n('And', ['FRAG', 'gtc1'], 'frR')
HR = flag('frR', 'HR')
n('Less', ['CC', c0], 'ltc0')
n('And', ['FRAG', 'ltc0'], 'frL')
HL = flag('frL', 'HL')
n('Greater', ['RR', r1], 'gtr1')
n('And', ['FRAG', 'gtr1'], 'frD')
VD = flag('frD', 'VD')
n('Less', ['RR', r0], 'ltr0')
n('And', ['FRAG', 'ltr0'], 'frU')
VU = flag('frU', 'VU')

# quadrant occupancy + per-quadrant colour
n('GreaterOrEqual', ['RR', r0], 'ge_r0')
n('LessOrEqual', ['RR', r1], 'le_r1')
n('And', ['ge_r0', 'le_r1'], 'inrow')
n('GreaterOrEqual', ['CC', c0], 'ge_c0')
n('LessOrEqual', ['CC', c1], 'le_c1')
n('And', ['ge_c0', 'le_c1'], 'incol')
n('Not', ['inrow'], 'ninrow')
n('Not', ['incol'], 'nincol')
n('And', ['FRAG', 'inrow'], 't1')
n('And', ['t1', 'nincol'], 'Hq')
n('And', ['FRAG', 'incol'], 't2')
n('And', ['t2', 'ninrow'], 'Vq')
n('And', ['FRAG', 'ninrow'], 't3')
n('And', ['t3', 'nincol'], 'HVq')


def existflag(q_bool, pfx):
    cf = n('Cast', [q_bool], pfx + '_cf', to=F)
    seed = n('Where', ['pres', cf, 'NINF'], pfx + '_sd')
    fl = maxflood(seed, pfx + '_fl')
    return n('Greater', [fl, 'half'], pfx + '_b')


Hex = existflag('Hq', 'Hex')
Vex = existflag('Vq', 'Vex')
HVex = existflag('HVq', 'HVex')


def qcolor(q_bool, pfx):
    seed = n('Where', [q_bool, 'col', 'NINF'], pfx + '_sd')
    fl = maxflood(seed, pfx + '_fl')
    return n('Max', [fl, 'zero'], pfx + '_c')


cH = qcolor('Hq', 'cH')
cV = qcolor('Vq', 'cV')
cHV = qcolor('HVq', 'cHV')

# reflection axes
n('Mul', [c1, 'two'], 'c1x2')
n('Add', ['c1x2', 'one'], 'axR')
n('Mul', [c0, 'two'], 'c0x2')
n('Sub', ['c0x2', 'one'], 'axL')
n('Mul', [r1, 'two'], 'r1x2')
n('Add', ['r1x2', 'one'], 'axD')
n('Mul', [r0, 'two'], 'r0x2')
n('Sub', ['r0x2', 'one'], 'axU')
n('Where', [HL, 'axL', 'CC'], 'axV0')
n('Where', [HR, 'axR', 'axV0'], 'axisV')
n('Where', [VU, 'axU', 'RR'], 'axH0')
n('Where', [VD, 'axD', 'axH0'], 'axisH')

# base buffer [31,30] (row 30 = scatter dump)
n('Reshape', ['col', 'sh_3030'], 'colg')
n('Pad', ['colg', 'padrow', 'zero'], 'buf0')


def build_scatter(buf, trg, tcg, gate_bool, color, pfx):
    ge_r = n('GreaterOrEqual', [trg, 'zero'], pfx + '_ger')
    le_r = n('LessOrEqual', [trg, 'c29'], pfx + '_ler')
    ge_c = n('GreaterOrEqual', [tcg, 'zero'], pfx + '_gec')
    le_c = n('LessOrEqual', [tcg, 'c29'], pfx + '_lec')
    n('And', [ge_r, le_r], pfx + '_rr')
    n('And', [ge_c, le_c], pfx + '_ccx')
    n('And', [pfx + '_rr', pfx + '_ccx'], pfx + '_rng')
    n('And', [gate_bool, pfx + '_rng'], pfx + '_val')
    ti = n('Where', [pfx + '_val', trg, 'c30f'], pfx + '_ti')
    tj = n('Where', [pfx + '_val', tcg, 'zero'], pfx + '_tj')
    n('Reshape', [ti, 'sh_9001'], pfx + '_tir')
    n('Reshape', [tj, 'sh_9001'], pfx + '_tjr')
    n('Concat', [pfx + '_tir', pfx + '_tjr'], pfx + '_idxf', axis=1)
    n('Cast', [pfx + '_idxf'], pfx + '_idx', to=I64)
    n('Reshape', [color, 'sh_900'], pfx + '_upd')
    return n('ScatterND', [buf, pfx + '_idx', pfx + '_upd'], pfx + '_buf')


n('And', ['SRC', HVex], 'gHV')
n('And', ['SRC', Vex], 'gV')
n('And', ['SRC', Hex], 'gH')
n('Sub', ['axisH', 'RR'], 'trHV')
n('Sub', ['axisV', 'CC'], 'tcHV')
b1 = build_scatter('buf0', 'trHV', 'tcHV', 'gHV', cHV, 'HV')
b2 = build_scatter(b1, 'trHV', 'CC', 'gV', cV, 'Vs')
b3 = build_scatter(b2, 'RR', 'tcHV', 'gH', cH, 'Hs')

# rebuild one-hot output, mask padding
n('Slice', [b3, 'r0slc', 'r30slc', 'ax0'], 'bufg')
n('Reshape', ['bufg', 'sh_grid'], 'Gf')
chs = []
for k in range(10):
    K('k%d' % k, np.float32(k))
    n('Equal', ['Gf', 'k%d' % k], 'eqk%d' % k)
    n('Cast', ['eqk%d' % k], 'chk%d' % k, to=F)
    chs.append('chk%d' % k)
n('Concat', chs, 'oh', axis=1)
n('Mul', ['oh', 'validmask'], 'output')

xI = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
yO = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])
graph = helper.make_graph(nodes, 'task285', [xI], [yO], inits)
model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid('', 12)])
