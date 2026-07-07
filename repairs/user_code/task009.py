# --- task009: connect colour marks with lines (bidirectional CumSum "between two marks"),
# --- suppress the grid colour, only overwrite background, decode via ArgMax->OneHot.
# YOUR LOGIC WAS CORRECT (interior 265/265). One fix: the final OneHot filled the empty 30x30
# canvas with channel-0=1; multiply the output by a "content" mask (input had any channel set)
# so everything OUTSIDE the actual grid stays all-zero. Verified 265/265.
# (Cost 475298 > baseline 8410, so the submission keeps baseline; you still own it.)
import numpy as np
F = TensorProto.FLOAT
x = helper.make_tensor_value_info('input',  F, [1,10,30,30])
y = helper.make_tensor_value_info('output', F, [1,10,30,30])
def K(n,a): return numpy_helper.from_array(a, name=n)
inits = [
 K('c0_s',np.array([0],np.int64)), K('c0_e',np.array([1],np.int64)), K('c1_s',np.array([1],np.int64)), K('c1_e',np.array([10],np.int64)),
 K('ax_c',np.array([1],np.int64)), K('ax_r',np.array(2,np.int64)), K('ax_col',np.array(3,np.int64)), K('zero',np.array(0.0,np.float32)),
 K('two',np.array(2.0,np.float32)), K('depth9',np.array(9,np.int64)), K('depth10',np.array(10,np.int64)),
 K('oh_vals',np.array([0.0,1.0],np.float32)), K('oh_vals_rev',np.array([1.0,0.0],np.float32)),
 K('zero_pad',np.zeros((1,1,30,30),np.float32)), K('rd_axes23',np.array([2,3],np.int64)), K('rd_axes1',np.array([1],np.int64)),
]
nodes = [
 helper.make_node('Slice',['input','c0_s','c0_e','ax_c'],['ch0']),
 helper.make_node('Slice',['input','c1_s','c1_e','ax_c'],['colors']),
 # row-wise fill: cells between first & last mark (per colour)
 helper.make_node('CumSum',['colors','ax_col'],['cum_r']), helper.make_node('CumSum',['colors','ax_col'],['cum_r_rev'],reverse=1),
 helper.make_node('Greater',['cum_r','zero'],['gr_r']), helper.make_node('Greater',['cum_r_rev','zero'],['gr_r_rev']), helper.make_node('And',['gr_r','gr_r_rev'],['fill_r']),
 # col-wise fill
 helper.make_node('CumSum',['colors','ax_r'],['cum_c']), helper.make_node('CumSum',['colors','ax_r'],['cum_c_rev'],reverse=1),
 helper.make_node('Greater',['cum_c','zero'],['gr_c']), helper.make_node('Greater',['cum_c_rev','zero'],['gr_c_rev']), helper.make_node('And',['gr_c','gr_c_rev'],['fill_c']),
 helper.make_node('Or',['fill_r','fill_c'],['fill_all_bool']), helper.make_node('Cast',['fill_all_bool'],['fill_all'],to=TensorProto.FLOAT),
 # suppress the grid (most frequent) colour
 helper.make_node('ReduceSum',['colors','rd_axes23'],['counts'],keepdims=1), helper.make_node('ArgMax',['counts'],['grid_color_idx'],axis=1,keepdims=0),
 helper.make_node('OneHot',['grid_color_idx','depth9','oh_vals_rev'],['grid_mask'],axis=1),
 helper.make_node('Mul',['fill_all','grid_mask'],['valid_fills']), helper.make_node('Mul',['valid_fills','ch0'],['masked_fills']),
 helper.make_node('Concat',['zero_pad','masked_fills'],['masked_fills_10'],axis=1), helper.make_node('Mul',['masked_fills_10','two'],['adds']),
 helper.make_node('Add',['input','adds'],['scores']),
 helper.make_node('ArgMax',['scores'],['pred_idx'],axis=1,keepdims=0), helper.make_node('OneHot',['pred_idx','depth10','oh_vals'],['oh'],axis=1),
 # FIX: zero out the empty canvas (keep only cells where the input had content)
 helper.make_node('ReduceSum',['input','rd_axes1'],['content'],keepdims=1),
 helper.make_node('Mul',['oh','content'],['output']),
]
model = helper.make_model(helper.make_graph(nodes,'task009',[x],[y],inits),
                          ir_version=10, opset_imports=[helper.make_opsetid('',13)])
