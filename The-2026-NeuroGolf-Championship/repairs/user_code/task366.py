# task366 -- genuine rule-based ONNX solve (no lookup table / RLE / memorized grids).
#
# RULE (learned from train, generalizes to test+arc-gen):
#   The grid is split in half along its longer (even) axis into a "dense" half
#   (a background full of small 2-colour objects) and a "sparse" half (same
#   background with a few scattered anchor cells). The output is the sparse half
#   (moved to the top-left corner). Every dense object owns exactly one "anchor
#   colour" -- the object colour that also appears among the sparse anchors.
#   An object is copied (whole, both colours) onto the sparse half at the unique
#   translation that lands its anchor-coloured cells exactly on sparse anchor
#   cells of that colour. When several objects could match overlapping anchors,
#   the larger anchor-constellation wins (coverage/maximal-match), so sub-patterns
#   are not spuriously stamped. Purely geometric: correlation + coverage, no
#   reference to any example index or embedded grid data.
#
# Everything is fixed-shape, static, bounded (<=3 objects, orientation handled by
# a conditional transpose). Ops: Squeeze/Transpose/Where/ArgMax/ReduceX/Conv/
# MaxPool/Gather/Slice/Clip/Equal/Greater/Less/And/Not -- no Loop/If/NonZero/etc.

F = TensorProto.FLOAT; I = TensorProto.INT64; B = TensorProto.BOOL

nodes = []; inits = []; _c = [0]
def nm(p):
    _c[0] += 1; return f"{p}_{_c[0]}"
def K(arr, name=None):
    name = name or nm('k'); inits.append(numpy_helper.from_array(np.asarray(arr), name)); return name
def N(op, ins, outs=None, **kw):
    o = outs or [nm(op.lower())]
    nodes.append(helper.make_node(op, ins, o, **kw)); return o[0]

# constants
ri  = K(np.arange(30, dtype=np.int64), 'ri')                       # [30]
rri = K(np.arange(30, dtype=np.int64).reshape(30, 1), 'rri')       # [30,1]
cci = K(np.arange(30, dtype=np.int64).reshape(1, 30), 'cci')       # [1,30]
chan = K(np.arange(10, dtype=np.int64).reshape(10, 1, 1), 'chan')  # [10,1,1]
zero_i = K(np.array(0, np.int64), 'zero_i'); c29 = K(np.array(29, np.int64), 'c29')
c30_i = K(np.array(30, np.int64), 'c30_i'); c59 = K(np.array(59, np.int64), 'c59')
two_f = K(np.array(2.0, np.float32), 'two_f'); zero_f = K(np.array(0.0, np.float32), 'zero_f')
c29_a = K(np.array([29, 29], np.int64), 'c29_a'); cbig_a = K(np.array([-9999, -9999], np.int64), 'cbig_a')
cneg_a = K(np.array([-1, -1], np.int64), 'cneg_a'); ax01 = K(np.array([0, 1], np.int64), 'ax01')
sh_1_1_30_30 = K(np.array([1, 1, 30, 30], np.int64), 'sh11')
sh_30_30 = K(np.array([30, 30], np.int64), 'sh3030')
sh_10_900 = K(np.array([10, 900], np.int64), 'sh10900')
sh_10_30_30 = K(np.array([10, 30, 30], np.int64), 'sh10_30_30')
sh_900 = K(np.array([900], np.int64), 'sh900')
sh_1_30_30 = K(np.array([1, 30, 30], np.int64), 'sh1_30_30')
sh_3481 = K(np.array([3481], np.int64), 'sh3481')
ids_f = K((np.arange(30).reshape(30, 1) * 30 + np.arange(30).reshape(1, 30) + 1).astype(np.float32), 'ids_f')

inp = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
out = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])

oh  = N('Squeeze', ['input'], axes=[0])                 # [10,30,30]
ohT = N('Transpose', [oh], perm=[0, 2, 1])
occ0 = N('ReduceSum', [oh], axes=[0], keepdims=0)       # [30,30]
rowmax = N('ReduceMax', [occ0], axes=[1], keepdims=0)   # [30]
H = N('ReduceSum', [rowmax], keepdims=0)
colmax = N('ReduceMax', [occ0], axes=[0], keepdims=0)
W = N('ReduceSum', [colmax], keepdims=0)
orient = N('Greater', [H, W])                           # scalar bool: rows are the long axis
ohn = N('Where', [orient, oh, ohT])                     # normalise to vertical split
occ = N('ReduceSum', [ohn], axes=[0], keepdims=0)       # [30,30]

rmax = N('ReduceMax', [occ], axes=[1], keepdims=0)      # [30]
Hn = N('ReduceSum', [rmax], keepdims=1)                 # [1]
hh_f = N('Div', [Hn, two_f])
hh = N('Cast', [hh_f], to=I)                            # [1]
tmrow = N('Less', [rri, hh])                            # [30,1] bool
tmrow_f = N('Cast', [tmrow], to=F)
tmrow_b = N('Reshape', [tmrow_f, K(np.array([1, 30, 1], np.int64), 'sh_1_30_1')])  # [1,30,1]
ohtop = N('Mul', [ohn, tmrow_b])                       # top half in place
botidx = N('Clip', [N('Add', [ri, hh]), zero_i, c29])  # relocate bottom half up
ohbot0 = N('Gather', [ohn, botidx], axis=1)
ohbot = N('Mul', [ohbot0, tmrow_b])

occtop = N('ReduceSum', [ohtop], axes=[0], keepdims=0)
occbot = N('ReduceSum', [ohbot], axes=[0], keepdims=0)
cst_top = N('ReduceSum', [ohtop], axes=[1, 2], keepdims=0)  # [10] colour histogram
cst_bot = N('ReduceSum', [ohbot], axes=[1, 2], keepdims=0)
bgtop = N('ArgMax', [cst_top], axis=0, keepdims=1)         # most-common colour = bg
bgbot = N('ArgMax', [cst_bot], axis=0, keepdims=1)
nbt = N('Sub', [N('ReduceSum', [occtop], keepdims=0), N('ReduceMax', [cst_top], keepdims=0)])
nbb = N('Sub', [N('ReduceSum', [occbot], keepdims=0), N('ReduceMax', [cst_bot], keepdims=0)])
topsparse = N('LessOrEqual', [nbt, nbb])                # sparse = fewer non-bg cells
sp_oh = N('Where', [topsparse, ohtop, ohbot])
de_oh = N('Where', [topsparse, ohbot, ohtop])
socc = N('Where', [topsparse, occtop, occbot])
docc = N('Where', [topsparse, occbot, occtop])
sbg = N('Where', [topsparse, bgtop, bgbot])            # [1]
dbg = N('Where', [topsparse, bgbot, bgtop])

# dense object labelling via 8-connected max-label propagation (unrolled, no Loop)
de_G = N('ArgMax', [de_oh], axis=0, keepdims=0)        # [30,30]
de_ne = N('Not', [N('Equal', [de_G, dbg])])
dmask = N('Mul', [docc, N('Cast', [de_ne], to=F)])     # [30,30] non-bg mask
label = N('Mul', [dmask, ids_f])
for _ in range(12):
    l4 = N('Reshape', [label, sh_1_1_30_30])
    lp = N('MaxPool', [l4], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
    lp2 = N('Reshape', [lp, sh_30_30])
    label = N('Mul', [lp2, dmask])
# peel off up to 3 object masks by descending label id
objs = []; lab = label
for i in range(3):
    L = N('ReduceMax', [lab], keepdims=1)
    mi = N('Cast', [N('And', [N('Equal', [lab, L]), N('Greater', [lab, zero_f])])], to=F)
    objs.append(mi)
    lab = N('Mul', [lab, N('Sub', [K(np.array(1.0, np.float32), 'onef_%d' % i), mi])])

# sparse anchor masks per colour (drop sparse bg channel)
notbg_chan = N('Cast', [N('Not', [N('Equal', [chan, sbg])])], to=F)   # [10,1,1]
Sk_all = N('Mul', [sp_oh, notbg_chan])                 # [10,30,30]
skchan_c = N('Greater', [N('ReduceSum', [Sk_all], axes=[1, 2], keepdims=0), zero_f])  # [10] bool

# per object: pick anchor colour, correlate anchors against sparse anchors
covobjs = []
for i in range(3):
    mi = objs[i]
    mi3 = N('Reshape', [mi, sh_1_30_30])
    objoh = N('Mul', [de_oh, mi3])                     # [10,30,30] object one-hot
    objchan = N('ReduceSum', [objoh], axes=[1, 2], keepdims=0)  # [10]
    iscand = N('And', [N('Greater', [objchan, zero_f]), skchan_c])  # anchor colour selector
    iscand_f = N('Cast', [iscand], to=F)
    iscand3 = N('Reshape', [iscand_f, K(np.array([10, 1, 1], np.int64), 'sh1011_%d' % i)])
    Aik = N('ReduceSum', [N('Mul', [objoh, iscand3])], axes=[0], keepdims=0)   # anchor mask [30,30]
    Sksel = N('ReduceSum', [N('Mul', [Sk_all, iscand3])], axes=[0], keepdims=0)  # sparse anchors [30,30]
    nik = N('ReduceSum', [Aik], keepdims=1)            # anchor count
    placef = N('ReduceMax', [iscand_f], keepdims=1)    # 1 if object has an anchor colour
    Aik4 = N('Reshape', [Aik, sh_1_1_30_30])
    Sk4 = N('Reshape', [Sksel, sh_1_1_30_30])
    corr = N('Conv', [Sk4, Aik4], kernel_shape=[30, 30], pads=[29, 29, 29, 29], strides=[1, 1])  # [1,1,59,59]
    valid = N('Mul', [N('Cast', [N('Equal', [corr, nik])], to=F),
                      N('Reshape', [placef, K(np.array([1, 1, 1, 1], np.int64), 'sh1111_%d' % i)])])
    covobjs.append((Aik, Aik4, nik, valid, mi, objoh))

# coverage field: maxcov(s) = largest anchor-count of any valid match covering sparse cell s
maxcov = None
for (Aik, Aik4, nik, valid, mi, objoh) in covobjs:
    flipAik = N('Slice', [Aik, c29_a, cbig_a, ax01, cneg_a])   # reverse both axes
    flipAik4 = N('Reshape', [flipAik, sh_1_1_30_30])
    covd = N('Conv', [valid, flipAik4], kernel_shape=[30, 30], pads=[0, 0, 0, 0], strides=[1, 1])  # [1,1,30,30]
    covered = N('Cast', [N('Greater', [N('Reshape', [covd, sh_30_30]), zero_f])], to=F)
    contrib = N('Mul', [covered, N('Reshape', [nik, K(np.array([1], np.int64), 'sh1_%d' % _c[0])])])
    maxcov = contrib if maxcov is None else N('Max', [maxcov, contrib])

# per object: keep only translations whose anchors are not dominated by a larger match, then stamp
out_oh = sp_oh
for (Aik, Aik4, nik, valid, mi, objoh) in covobjs:
    nik1 = N('Reshape', [nik, K(np.array([1, 1], np.int64), 'sh11a_%d' % _c[0])])
    bigcov = N('Cast', [N('Greater', [maxcov, nik1])], to=F)
    bigcov4 = N('Reshape', [bigcov, sh_1_1_30_30])
    badc = N('Conv', [bigcov4, Aik4], kernel_shape=[30, 30], pads=[29, 29, 29, 29], strides=[1, 1])
    survive = N('Mul', [valid, N('Cast', [N('Equal', [badc, zero_f])], to=F)])  # [1,1,59,59]
    surv_flat = N('Reshape', [survive, sh_3481])
    place = N('ReduceMax', [surv_flat], keepdims=1)
    idx = N('ArgMax', [surv_flat], axis=0, keepdims=1)
    oy = N('Div', [idx, c59]); ox = N('Sub', [idx, N('Mul', [oy, c59])])
    tr = N('Sub', [oy, c29]); tc = N('Sub', [ox, c29])     # translation (dr,dc)
    prmap = N('Sub', [rri, tr]); pcmap = N('Sub', [cci, tc])
    vpr = N('And', [N('GreaterOrEqual', [prmap, zero_i]), N('Less', [prmap, c30_i])])
    vpc = N('And', [N('GreaterOrEqual', [pcmap, zero_i]), N('Less', [pcmap, c30_i])])
    valids = N('Cast', [N('And', [vpr, vpc])], to=F)       # [30,30] in-bounds mask
    srcpr = N('Clip', [prmap, zero_i, c29]); srcpc = N('Clip', [pcmap, zero_i, c29])
    srclin = N('Reshape', [N('Add', [N('Mul', [srcpr, c30_i]), srcpc]), sh_900])
    objoh_flat = N('Reshape', [objoh, sh_10_900])
    gathered = N('Reshape', [N('Gather', [objoh_flat, srclin], axis=1), sh_10_30_30])  # shifted object
    place_b = N('Reshape', [place, K(np.array([1, 1, 1], np.int64), 'shp_%d' % _c[0])])
    stamped = N('Mul', [N('Mul', [gathered, N('Reshape', [valids, sh_1_30_30])]), place_b])
    painted = N('ReduceSum', [stamped], axes=[0], keepdims=0)
    pmask = N('Reshape', [N('Greater', [painted, zero_f]), sh_1_30_30])
    out_oh = N('Where', [pmask, stamped, out_oh])

out_ohT = N('Transpose', [out_oh], perm=[0, 2, 1])
out_final = N('Where', [orient, out_oh, out_ohT])          # restore original orientation
N('Unsqueeze', [out_final], ['output'], axes=[0])

model = helper.make_model(
    helper.make_graph(nodes, 'task366', [inp], [out], inits),
    ir_version=10, opset_imports=[helper.make_opsetid('', 12)])
