# Source: predicted/test_onnx_task012.py — ONNX graph construction code
# Verified model: repairs/task012.onnx
import onnx
import onnx.helper as helper
from onnx import TensorProto
import numpy as np

def make_shift(node_name, input_name, direction):
    pads_name = node_name + "_pads"
    pad_node = helper.make_node("Pad", [input_name, pads_name], [node_name + "_padded"], name=node_name + "_pad")
    
    starts_name = node_name + "_starts"
    ends_name = node_name + "_ends"
    axes_name = node_name + "_axes"
    slice_node = helper.make_node("Slice", 
                                  [node_name + "_padded", starts_name, ends_name, axes_name], 
                                  [node_name], 
                                  name=node_name + "_slice")
    
    if direction == 'R':
        pads = [0, 0, 0, 1, 0, 0, 0, 0]
        starts = [0]
        ends = [30]
        axes = [3]
    elif direction == 'L':
        pads = [0, 0, 0, 0, 0, 0, 0, 1]
        starts = [1]
        ends = [31]
        axes = [3]
    elif direction == 'D':
        pads = [0, 0, 1, 0, 0, 0, 0, 0]
        starts = [0]
        ends = [30]
        axes = [2]
    elif direction == 'U':
        pads = [0, 0, 0, 0, 0, 0, 1, 0]
        starts = [1]
        ends = [31]
        axes = [2]
        
    init_pads = helper.make_tensor(pads_name, TensorProto.INT64, [8], pads)
    init_starts = helper.make_tensor(starts_name, TensorProto.INT64, [1], starts)
    init_ends = helper.make_tensor(ends_name, TensorProto.INT64, [1], ends)
    init_axes = helper.make_tensor(axes_name, TensorProto.INT64, [1], axes)
    
    return [pad_node, slice_node], [init_pads, init_starts, init_ends, init_axes]

def build_model():
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10, 30, 30])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10, 30, 30])
    
    nodes = []
    inits = []
    
    def make_tensor(name, dtype, shape, vals):
        inits.append(helper.make_tensor(name, dtype, shape, vals))
        
    make_tensor("zero_int", TensorProto.INT64, [1], [0])
    make_tensor("one_int", TensorProto.INT64, [1], [1])
    make_tensor("ten_int", TensorProto.INT64, [1], [10])
    make_tensor("axes_c", TensorProto.INT64, [1], [1])
    make_tensor("one_float", TensorProto.FLOAT, [1], [1.0])
    
    # bg = input[:, 0]
    nodes.append(helper.make_node("Slice", ["input", "zero_int", "one_int", "axes_c"], ["bg"]))
    # M_any = 1 - bg
    nodes.append(helper.make_node("Sub", ["one_float", "bg"], ["M_any"]))
    
    # is_center
    plus = [0,1,0, 1,1,1, 0,1,0]
    make_tensor("plus_kernel", TensorProto.FLOAT, [1, 1, 3, 3], plus)
    nodes.append(helper.make_node("Conv", ["M_any", "plus_kernel"], ["center_conv"], pads=[1,1,1,1]))
    
    make_tensor("four_point_nine", TensorProto.FLOAT, [1], [4.9])
    nodes.append(helper.make_node("Greater", ["center_conv", "four_point_nine"], ["is_center_bool"]))
    nodes.append(helper.make_node("Cast", ["is_center_bool"], ["is_center"], to=TensorProto.FLOAT))
    
    # shift is_center
    nU, iU = make_shift("is_center_U", "is_center", "U")
    nD, iD = make_shift("is_center_D", "is_center", "D")
    nL, iL = make_shift("is_center_L", "is_center", "L")
    nR, iR = make_shift("is_center_R", "is_center", "R")
    nodes.extend(nU + nD + nL + nR)
    inits.extend(iU + iD + iL + iR)
    
    # arm masks
    nodes.append(helper.make_node("Mul", ["is_center_U", "M_any"], ["is_top_arm"]))
    nodes.append(helper.make_node("Mul", ["is_center_D", "M_any"], ["is_bottom_arm"]))
    nodes.append(helper.make_node("Mul", ["is_center_L", "M_any"], ["is_left_arm"]))
    nodes.append(helper.make_node("Mul", ["is_center_R", "M_any"], ["is_right_arm"]))
    
    # center_c
    nodes.append(helper.make_node("Mul", ["input", "is_center"], ["center_c"]))
    
    # X_c
    make_tensor("shape_10_1_30_30", TensorProto.INT64, [4], [10, 1, 30, 30])
    nodes.append(helper.make_node("Reshape", ["center_c", "shape_10_1_30_30"], ["center_c_10"]))
    
    X_k = [1,0,0,0,1, 0,1,0,1,0, 0,0,1,0,0, 0,1,0,1,0, 1,0,0,0,1]
    make_tensor("X_kernel", TensorProto.FLOAT, [1, 1, 5, 5], X_k)
    nodes.append(helper.make_node("Conv", ["center_c_10", "X_kernel"], ["X_c_10"], pads=[2,2,2,2]))
    
    make_tensor("shape_1_10_30_30", TensorProto.INT64, [4], [1, 10, 30, 30])
    nodes.append(helper.make_node("Reshape", ["X_c_10", "shape_1_10_30_30"], ["X_c"]))
    
    # extended arms
    nodes.append(helper.make_node("Mul", ["input", "is_top_arm"], ["top_arm_c"]))
    nU2, iU2 = make_shift("top_arm_ext", "top_arm_c", "U")
    nodes.extend(nU2); inits.extend(iU2)
    
    nodes.append(helper.make_node("Mul", ["input", "is_bottom_arm"], ["bottom_arm_c"]))
    nD2, iD2 = make_shift("bottom_arm_ext", "bottom_arm_c", "D")
    nodes.extend(nD2); inits.extend(iD2)
    
    nodes.append(helper.make_node("Mul", ["input", "is_left_arm"], ["left_arm_c"]))
    nL2, iL2 = make_shift("left_arm_ext", "left_arm_c", "L")
    nodes.extend(nL2); inits.extend(iL2)
    
    nodes.append(helper.make_node("Mul", ["input", "is_right_arm"], ["right_arm_c"]))
    nR2, iR2 = make_shift("right_arm_ext", "right_arm_c", "R")
    nodes.extend(nR2); inits.extend(iR2)
    
    # Combine
    nodes.append(helper.make_node("Max", ["input", "X_c"], ["out_1"]))
    nodes.append(helper.make_node("Max", ["out_1", "top_arm_ext"], ["out_2"]))
    nodes.append(helper.make_node("Max", ["out_2", "bottom_arm_ext"], ["out_3"]))
    nodes.append(helper.make_node("Max", ["out_3", "left_arm_ext"], ["out_4"]))
    nodes.append(helper.make_node("Max", ["out_4", "right_arm_ext"], ["out_raw"]))
    
    # Recompute BG
    nodes.append(helper.make_node("Slice", ["out_raw", "one_int", "ten_int", "axes_c"], ["out_1_9"]))
    nodes.append(helper.make_node("ReduceSum", ["out_1_9", "axes_c"], ["sum_1_9"], keepdims=1))
    
    make_tensor("zero_float", TensorProto.FLOAT, [1], [0.0])
    nodes.append(helper.make_node("Clip", ["sum_1_9", "zero_float", "one_float"], ["sum_clipped"]))
    nodes.append(helper.make_node("Sub", ["one_float", "sum_clipped"], ["new_bg"]))
    
    nodes.append(helper.make_node("Concat", ["new_bg", "out_1_9"], ["output_full"], axis=1))
    
    # Mask to valid grid just in case
    nodes.append(helper.make_node("ReduceSum", ["input", "axes_c"], ["valid_grid_mask"], keepdims=1))
    nodes.append(helper.make_node("Mul", ["output_full", "valid_grid_mask"], ["output"]))
    
    graph = helper.make_graph(nodes, "task12_graph", [X], [Y], inits)
    model = helper.make_model(graph, producer_name='antigravity', ir_version=8, opset_imports=[helper.make_opsetid('', 13)])
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    onnx.save(model, "task012.onnx")
    print("Saved task012.onnx")


# Build the model (the function saves it internally, so we load the result)
build_model()
import glob
model = onnx.load("/project/repairs/task012.onnx")
