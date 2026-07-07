import os, sys, math, json, copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import defaultdict, Counter
import onnx, onnxruntime
from onnx import helper, numpy_helper, TensorProto

TASK_DIR="data"; BASE_DIR="baseline_v22"; OUT_DIR="repairs"
sys.path.insert(0,"data/neurogolf_utils"); import neurogolf_utils as ngu

_ARC=[(0,0,0),(30,147,255),(250,61,49),(78,204,48),(255,221,0),(153,153,153),(229,59,163),(255,133,28),(136,216,241),(147,17,49)]
PAL=ListedColormap([tuple(c/255 for c in rgb) for rgb in _ARC])

def load_task(t): return json.load(open(f"{TASK_DIR}/task{t:03d}.json"))

def show_task(t, n=3):
    d=load_task(t); pairs=d["train"][:n]+d["test"][:1]
    labs=[f"train{i}" for i in range(len(d["train"][:n]))]+["TEST"]
    fig,ax=plt.subplots(2,len(pairs),figsize=(3*len(pairs),6))
    if len(pairs)==1: ax=ax.reshape(2,1)
    for k,ex in enumerate(pairs):
        ax[0,k].imshow(np.array(ex["input"]),cmap=PAL,vmin=0,vmax=9); ax[0,k].set_title(f"{labs[k]} in"); ax[0,k].set_xticks([]); ax[0,k].set_yticks([])
        ax[1,k].imshow(np.array(ex["output"]),cmap=PAL,vmin=0,vmax=9); ax[1,k].set_title("out"); ax[1,k].set_xticks([]); ax[1,k].set_yticks([])
    plt.suptitle(f"task{t:03d}",fontsize=11); plt.tight_layout(); plt.show()

def draw_onnx(path_or_model, ax=None, title=None, max_nodes=45):
    m=onnx.load(path_or_model) if isinstance(path_or_model,str) else path_or_model
    nodes=list(m.graph.node); own=ax is None
    if own: fig,ax=plt.subplots(figsize=(10,5))
    if len(nodes)>max_nodes:
        c=Counter(n.op_type for n in nodes).most_common()
        ax.barh([k for k,_ in c][::-1],[v for _,v in c][::-1],color="#8e44ad")
        ax.set_title(f"{title or ''} ({len(nodes)} nodes)",fontsize=9)
        if own: plt.tight_layout(); plt.show()
        return
    prod={o:i for i,n in enumerate(nodes) for o in n.output}
    preds={i:[prod[x] for x in n.input if x in prod] for i,n in enumerate(nodes)}
    layer={}
    def L(i):
        if i not in layer: layer[i]=0 if not preds[i] else max(L(p) for p in preds[i])+1
        return layer[i]
    for i in range(len(nodes)): L(i)
    byl=defaultdict(list)
    for i,l in layer.items(): byl[l].append(i)
    pos={}
    for l,ids in byl.items():
        for k,i in enumerate(ids): pos[i]=(l,-(k-(len(ids)-1)/2))
    for i in range(len(nodes)):
        x2,y2=pos[i]
        for p in preds[i]:
            x1,y1=pos[p]; ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="->",color="#bbb",lw=0.7))
    for i,n in enumerate(nodes):
        x,y=pos[i]; ax.text(x,y,n.op_type,ha="center",va="center",fontsize=6.5,bbox=dict(boxstyle="round,pad=0.3",fc="#dceefb",ec="#2980b9",lw=0.8))
    ax.set_title(f"{title or 'graph'} ({len(nodes)} nodes)",fontsize=9); ax.axis("off"); ax.set_xlim(-0.6,max(byl)+0.6)
    if own: plt.tight_layout(); plt.show()

def verify_and_score(model, t):
    """Run our model on ALL examples of task t; return (n_fail, cost, points)."""
    san=ngu.sanitize_model(copy.deepcopy(model))
    onnx.checker.check_model(san, full_check=True)
    o=onnxruntime.SessionOptions(); o.enable_profiling=True; o.log_severity_level=3
    o.graph_optimization_level=onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL; o.profile_file_prefix="vs"
    s=onnxruntime.InferenceSession(san.SerializeToString(),o)
    d=load_task(t); nf=0
    for ex in d["train"]+d["test"]+d["arc-gen"]:
        b=ngu.convert_to_numpy(ex)
        if not b: continue
        out=ngu.run_network(s,b["input"])
        if not np.array_equal(out,b["output"]): nf+=1
    tp=s.end_profiling(); mem,par=ngu.score_network(san,tp)
    try: os.remove(tp)
    except: pass
    cost=(mem+par) if (mem is not None and par is not None) else None
    pts=max(1.0,25.0-math.log(max(1.0,cost))) if cost is not None else None
    return nf, cost, pts
print("helpers ready")
---
show_task(1)
---
# Build our task001 ONNX from scratch
F=TensorProto.FLOAT
inits=[]
def K(name,arr): inits.append(numpy_helper.from_array(arr,name)); return name
K('s0',np.array([0,0],np.int64)); K('e0',np.array([3,3],np.int64)); K('ax',np.array([2,3],np.int64))
K('s0c',np.array([0],np.int64)); K('e0c',np.array([1],np.int64)); K('axc',np.array([1],np.int64))
K('one',np.array(1.0,np.float32)); K('rep',np.array([1,1,3,3],np.int64))
K('sh1',np.array([1,1,3,1,3,1],np.int64)); K('sh2',np.array([1,1,3,3,3,3],np.int64)); K('sh3',np.array([1,1,9,9],np.int64))
K('chpad',np.array([0,0,0,0,0,9,0,0],np.int64)); K('sppad',np.array([0,0,0,0,0,0,21,21],np.int64))
x=helper.make_tensor_value_info('input',F,[1,10,30,30]); y=helper.make_tensor_value_info('output',F,[1,10,30,30])
n=[
 helper.make_node('Slice',['input','s0','e0','ax'],['Xc']),
 helper.make_node('Slice',['Xc','s0c','e0c','axc'],['ch0']),
 helper.make_node('Sub',['one','ch0'],['A']),
 helper.make_node('Reshape',['A','sh1'],['Ar']),
 helper.make_node('Expand',['Ar','sh2'],['Ae']),
 helper.make_node('Reshape',['Ae','sh3'],['Abig']),
 helper.make_node('Tile',['Xc','rep'],['Xtiled']),
 helper.make_node('Mul',['Xtiled','Abig'],['Omul']),
 helper.make_node('Sub',['one','Abig'],['bg']),
 helper.make_node('Pad',['bg','chpad'],['bgfull']),
 helper.make_node('Add',['Omul','bgfull'],['O9']),
 helper.make_node('Pad',['O9','sppad'],['output']),
]
g=helper.make_graph(n,'task001',[x],[y],inits)
m001=helper.make_model(g,ir_version=10,opset_imports=[helper.make_opsetid('',12)])
onnx.checker.check_model(m001,full_check=True)
nf,cost,pts=verify_and_score(m001,1)
print(f"task001 ours: n_fail={nf}/268  cost={cost}  points={pts:.3f}")
onnx.save(m001, f"{OUT_DIR}/task001.onnx")
---
# Our graph vs hoangvux's baseline graph for task001
fig,ax=plt.subplots(1,2,figsize=(15,5))
draw_onnx(m001,ax=ax[0],title="task001 OURS")
draw_onnx(f"{BASE_DIR}/task001.onnx",ax=ax[1],title="task001 hoangvux baseline")
plt.tight_layout(); plt.show()
print("Baseline is cheaper (one big Einsum + fp16, tiny intermediates). Our golf target: match/beat it.")
---
show_task(2)
---
# baseline graph for task002 (hoangvux) — study it to see how the rule was compiled
draw_onnx(f"{BASE_DIR}/task002.onnx", title="task002 baseline")
---
show_task(3)
---
F=TensorProto.FLOAT; inits=[]
def K(n,a): inits.append(numpy_helper.from_array(a,n)); return n
K('ax',np.array([2,3],np.int64))
K('t_s',np.array([0,0],np.int64)); K('t_e',np.array([3,3],np.int64))
K('b_s',np.array([3,0],np.int64)); K('b_e',np.array([6,3],np.int64))
K('m_s',np.array([2,0],np.int64)); K('m_e',np.array([5,3],np.int64))
K('i_s',np.array([0,0],np.int64)); K('i_e',np.array([6,3],np.int64))
K('nine',np.array(9.0,np.float32)); K('one',np.array(1.0,np.float32)); K('two',np.array(2.0,np.float32))
K('eqshape',np.array([1,1,1,1],np.int64)); K('depth',np.array(10,np.int64)); K('vals',np.array([0.0,1.0],np.float32))
K('c1s',np.array([1],np.int64)); K('c1e',np.array([2],np.int64)); K('c1ax',np.array([1],np.int64))
K('pad',np.array([0,0,0,0,0,0,21,27],np.int64))
x=helper.make_tensor_value_info('input',F,[1,10,30,30]); y=helper.make_tensor_value_info('output',F,[1,10,30,30])
nodes=[
 helper.make_node('Slice',['input','t_s','t_e','ax'],['top']),
 helper.make_node('Slice',['input','b_s','b_e','ax'],['bot']),
 helper.make_node('Slice',['input','m_s','m_e','ax'],['mid']),
 helper.make_node('Slice',['input','i_s','i_e','ax'],['Ic']),
 helper.make_node('Mul',['top','bot'],['tb']),
 helper.make_node('ReduceSum',['tb'],['match'],keepdims=0),
 helper.make_node('Equal',['match','nine'],['eqb']),
 helper.make_node('Cast',['eqb'],['eqf'],to=TensorProto.FLOAT),
 helper.make_node('Reshape',['eqf','eqshape'],['eq']),
 helper.make_node('Sub',['one','eq'],['neq']),
 helper.make_node('Mul',['bot','eq'],['e1']),
 helper.make_node('Mul',['mid','neq'],['e2']),
 helper.make_node('Add',['e1','e2'],['extra']),
 helper.make_node('Concat',['Ic','extra'],['out9'],axis=2),
 helper.make_node('Slice',['out9','c1s','c1e','c1ax'],['ch1']),
 helper.make_node('Mul',['ch1','two'],['cv']),
 helper.make_node('Squeeze',['cv'],['cvs'],axes=[1]),
 helper.make_node('Cast',['cvs'],['cvi'],to=TensorProto.INT64),
 helper.make_node('OneHot',['cvi','depth','vals'],['oh'],axis=1),
 helper.make_node('Pad',['oh','pad'],['output']),
]
m003=helper.make_model(helper.make_graph(nodes,'task003',[x],[y],inits),ir_version=10,opset_imports=[helper.make_opsetid('',12)])
onnx.checker.check_model(m003,full_check=True)
nf,cost,pts=verify_and_score(m003,3); print(f'task003 ours: n_fail={nf}/265  cost={cost}  points={pts:.3f}')
onnx.save(m003,f'{OUT_DIR}/task003.onnx')
---
draw_onnx(m003, title='task003 OURS (conditional append + recolour)')
---
show_task(4)
---
# baseline graph for task004 (hoangvux) — study it to see how the rule was compiled
draw_onnx(f"{BASE_DIR}/task004.onnx", title="task004 baseline")
---
show_task(5)
---
# baseline graph for task005 (hoangvux) — study it to see how the rule was compiled
draw_onnx(f"{BASE_DIR}/task005.onnx", title="task005 baseline")
---
show_task(6)
---
F=TensorProto.FLOAT; inits=[]
def K(n,a): inits.append(numpy_helper.from_array(a,n)); return n
K('ls',np.array([0,0,0],np.int64)); K('le',np.array([1,3,3],np.int64)); K('ax',np.array([1,2,3],np.int64))
K('rs',np.array([0,0,4],np.int64)); K('re',np.array([1,3,7],np.int64))
K('one',np.array(1.0,np.float32)); K('two',np.array(2.0,np.float32))
K('depth',np.array(10,np.int64)); K('vals',np.array([0.0,1.0],np.float32)); K('pad',np.array([0,0,0,0,0,0,27,27],np.int64))
x=helper.make_tensor_value_info('input',F,[1,10,30,30]); y=helper.make_tensor_value_info('output',F,[1,10,30,30])
nodes=[
 helper.make_node('Slice',['input','ls','le','ax'],['L0']),
 helper.make_node('Slice',['input','rs','re','ax'],['R0']),
 helper.make_node('Sub',['one','L0'],['Lnz']),
 helper.make_node('Sub',['one','R0'],['Rnz']),
 helper.make_node('Mul',['Lnz','Rnz'],['ov']),
 helper.make_node('Mul',['ov','two'],['cv']),
 helper.make_node('Squeeze',['cv'],['cvs'],axes=[1]),
 helper.make_node('Cast',['cvs'],['cvi'],to=TensorProto.INT64),
 helper.make_node('OneHot',['cvi','depth','vals'],['oh'],axis=1),
 helper.make_node('Pad',['oh','pad'],['output']),
]
m006=helper.make_model(helper.make_graph(nodes,'task006',[x],[y],inits),ir_version=10,opset_imports=[helper.make_opsetid('',12)])
onnx.checker.check_model(m006,full_check=True)
nf,cost,pts=verify_and_score(m006,6); print(f'task006 ours: n_fail={nf}/266  cost={cost}  points={pts:.3f}')
onnx.save(m006,f'{OUT_DIR}/task006.onnx')
---
draw_onnx(m006, title='task006 OURS (left AND right -> 2)')
---
show_task(7)
---
# baseline graph for task007 (hoangvux) — study it to see how the rule was compiled
draw_onnx(f"{BASE_DIR}/task007.onnx", title="task007 baseline")
---
show_task(8)
---
# baseline graph for task008 (hoangvux) — study it to see how the rule was compiled
draw_onnx(f"{BASE_DIR}/task008.onnx", title="task008 baseline")
---
show_task(9)
---
# baseline graph for task009 (hoangvux) — study it to see how the rule was compiled
draw_onnx(f"{BASE_DIR}/task009.onnx", title="task009 baseline")
---
show_task(10)
---
# baseline graph for task010 (hoangvux) — study it to see how the rule was compiled
draw_onnx(f"{BASE_DIR}/task010.onnx", title="task010 baseline")
---
show_task(16)
---
idx=np.array([0,5,6,4,3,1,2,7,9,8],np.int64)
x=helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30]); y=helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,10,30,30])
m016=helper.make_model(helper.make_graph(
    [helper.make_node('Gather',['input','cidx'],['output'],axis=1)],
    'task016',[x],[y],[numpy_helper.from_array(idx,'cidx')]),
    ir_version=10,opset_imports=[helper.make_opsetid('',12)])
onnx.checker.check_model(m016,full_check=True)
nf,cost,pts=verify_and_score(m016,16); print(f'task016 ours: n_fail={nf}/267  cost={cost}  points={pts:.3f}')
onnx.save(m016,f'{OUT_DIR}/task016.onnx')
---
draw_onnx(m016, title='task016 OURS (single Gather recolour)')
---
show_task(11)
---
show_task(17)
---
# Generic rule builders (all static-shape, opset-legal). Each returns an onnx model.
F=TensorProto.FLOAT
def _mk(nodes,inits):
    x=helper.make_tensor_value_info('input',F,[1,10,30,30]); y=helper.make_tensor_value_info('output',F,[1,10,30,30])
    m=helper.make_model(helper.make_graph(nodes,'g',[x],[y],inits),ir_version=10,opset_imports=[helper.make_opsetid('',12)])
    onnx.checker.check_model(m,full_check=True); return m
def _A(g): return np.array(g,dtype=int)
def _exs(t): d=load_task(t); return [(_A(e['input']),_A(e['output'])) for e in d['train']+d['test']+d['arc-gen']]

def build_recolor(t):
    """color map -> if bijection over 10 channels use cheap Gather, else channel-matrix Conv."""
    m={}
    for i,o in _exs(t):
        for a,b in zip(i.flat,o.flat): m[int(a)]=int(b)
    eff={c:(m[c] if c in m else c) for c in range(10)}
    if len(set(eff.values()))==10:                       # bijection -> Gather (cost ~10)
        inv={v:k for k,v in eff.items()}; perm=np.array([inv[b] for b in range(10)],np.int64)
        return _mk([helper.make_node('Gather',['input','p'],['output'],axis=1)],[numpy_helper.from_array(perm,'p')])
    W=np.zeros((10,10,1,1),np.float32)                    # else channel-sum 1x1 Conv (cost ~100)
    for a in range(10): W[eff[a],a,0,0]=1.0
    return _mk([helper.make_node('Conv',['input','W'],['output'],kernel_shape=[1,1])],[numpy_helper.from_array(W,'W')])

def build_geom(t):
    exs=_exs(t); H,W=exs[0][0].shape
    T={'flip_h':lambda g:g[:,::-1],'flip_v':lambda g:g[::-1],'rot90':lambda g:np.rot90(g,1),
       'rot180':lambda g:np.rot90(g,2),'rot270':lambda g:np.rot90(g,3),'transpose':lambda g:g.T}
    name=next(n for n,f in T.items() if all(f(i).shape==o.shape and np.array_equal(f(i),o) for i,o in exs))
    rr=np.array(list(range(H-1,-1,-1))+list(range(H,30)),np.int64)
    cc=np.array(list(range(W-1,-1,-1))+list(range(W,30)),np.int64)
    nodes=[]; inits=[]
    def gth(idx,axis,nm,src): inits.append(numpy_helper.from_array(idx,nm+'i')); nodes.append(helper.make_node('Gather',[src,nm+'i'],[nm],axis=axis)); return nm
    if name=='flip_h': gth(cc,3,'output','input')
    elif name=='flip_v': gth(rr,2,'output','input')
    elif name=='rot180': gth(rr,2,'a','input'); gth(cc,3,'output','a')
    elif name=='transpose': nodes.append(helper.make_node('Transpose',['input'],['output'],perm=[0,1,3,2]))
    elif name=='rot90': nodes.append(helper.make_node('Transpose',['input'],['a'],perm=[0,1,3,2])); gth(rr,2,'output','a')
    elif name=='rot270': nodes.append(helper.make_node('Transpose',['input'],['a'],perm=[0,1,3,2])); gth(cc,3,'output','a')
    return _mk(nodes,inits), name

def build_upscale(t):
    exs=_exs(t); H,W=exs[0][0].shape; ky=exs[0][1].shape[0]//H; kx=exs[0][1].shape[1]//W
    I=[]; K=lambda n,a:(I.append(numpy_helper.from_array(a,n)) or n)
    K('s',np.array([0,0],np.int64)); K('e',np.array([H,W],np.int64)); K('ax',np.array([2,3],np.int64))
    K('sh1',np.array([1,10,H,1,W,1],np.int64)); K('sh2',np.array([1,10,H,ky,W,kx],np.int64)); K('sh3',np.array([1,10,H*ky,W*kx],np.int64))
    K('pad',np.array([0,0,0,0,0,0,30-H*ky,30-W*kx],np.int64))
    nodes=[helper.make_node('Slice',['input','s','e','ax'],['c']),helper.make_node('Reshape',['c','sh1'],['r1']),
           helper.make_node('Expand',['r1','sh2'],['ex']),helper.make_node('Reshape',['ex','sh3'],['up']),
           helper.make_node('Pad',['up','pad'],['output'])]
    return _mk(nodes,I),(ky,kx)

def build_crop(t):
    exs=_exs(t); ih,iw=exs[0][0].shape; oh,ow=exs[0][1].shape
    r0,c0=next((r,c) for r in range(ih-oh+1) for c in range(iw-ow+1) if all(np.array_equal(i[r:r+oh,c:c+ow],o) for i,o in exs))
    I=[]; K=lambda n,a:(I.append(numpy_helper.from_array(a,n)) or n)
    K('s',np.array([r0,c0],np.int64)); K('e',np.array([r0+oh,c0+ow],np.int64)); K('ax',np.array([2,3],np.int64))
    K('pad',np.array([0,0,0,0,0,0,30-oh,30-ow],np.int64))
    return _mk([helper.make_node('Slice',['input','s','e','ax'],['c']),helper.make_node('Pad',['c','pad'],['output'])],I),(r0,c0,oh,ow)

def build_show(t, kind):
    if kind=='recolor': m=build_recolor(t); detail=''
    elif kind=='geom': m,detail=build_geom(t)
    elif kind=='upscale': m,detail=build_upscale(t)
    elif kind=='crop': m,detail=build_crop(t)
    nf,cost,pts=verify_and_score(m,t)
    onnx.save(m,f'{OUT_DIR}/task{t:03d}.onnx')
    print(f'task{t:03d} [{kind} {detail}]: n_fail={nf}  cost={cost}  points={pts:.3f}')
    draw_onnx(m,title=f'task{t:03d} OURS ({kind})')
    return m
print("generic builders ready: build_recolor/build_geom/build_upscale/build_crop, build_show()")
---
show_task(135)
---
m135=build_show(135, 'crop')
---
show_task(87)
---
m087=build_show(87, 'geom')
---
show_task(140)
---
m140=build_show(140, 'geom')
---
show_task(179)
---
m179=build_show(179, 'geom')
---
show_task(380)
---
m380=build_show(380, 'geom')
---
show_task(223)
---
m223=build_show(223, 'upscale')
---
show_task(276)
---
m276=build_show(276, 'recolor')
---
show_task(309)
---
m309=build_show(309, 'recolor')
---
show_task(337)
---
m337=build_show(337, 'recolor')
---
show_task(385)
---
F=TensorProto.FLOAT; inits=[]
def K(n,a): inits.append(numpy_helper.from_array(a,n)); return n
K('bs',np.array([5,0],np.int64)); K('be',np.array([10,4],np.int64)); K('ax',np.array([2,3],np.int64))
K('rev',np.array([4,3,2,1,0],np.int64)); K('pad',np.array([0,0,0,0,0,0,20,26],np.int64))
x=helper.make_tensor_value_info('input',F,[1,10,30,30]); y=helper.make_tensor_value_info('output',F,[1,10,30,30])
m385=helper.make_model(helper.make_graph([
 helper.make_node('Slice',['input','bs','be','ax'],['bottom']),
 helper.make_node('Gather',['bottom','rev'],['top'],axis=2),
 helper.make_node('Concat',['top','bottom'],['out10'],axis=2),
 helper.make_node('Pad',['out10','pad'],['output'])],'task385',[x],[y],inits),ir_version=10,opset_imports=[helper.make_opsetid('',12)])
onnx.checker.check_model(m385,full_check=True)
nf,cost,pts=verify_and_score(m385,385); print(f'task385 ours: n_fail={nf}/265  cost={cost}  points={pts:.3f}')
onnx.save(m385,f'{OUT_DIR}/task385.onnx')
---
draw_onnx(m385, title='task385 OURS (reflect bottom up)')