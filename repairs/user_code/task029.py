# Source: predicted/test_onnx_task029.py — ONNX graph construction code
# Verified model: repairs/task029.onnx
import onnx
from onnx import helper, TensorProto

def make_tensor(name, data_type, dims, vals):
    return helper.make_node("Constant", [], [name], value=helper.make_tensor(name+"_v", data_type, dims, vals))

def build_model():
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10, 30, 30])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10, 30, 30])

    nodes = []

    # Constants
    nodes.append(make_tensor("two_float", TensorProto.FLOAT, [1, 1, 1, 1], [2.0]))
    nodes.append(make_tensor("one_float", TensorProto.FLOAT, [1, 1, 1, 1], [1.0]))
    nodes.append(make_tensor("zero_float", TensorProto.FLOAT, [1, 1, 1, 1], [0.0]))
    
    # Exclude channel 0 from being a frame
    nodes.append(make_tensor("channel_mask", TensorProto.FLOAT, [1, 10, 1, 1], [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    
    r_indices = list(range(30))
    nodes.append(make_tensor("r_col", TensorProto.FLOAT, [1, 1, 1, 30], r_indices))
    nodes.append(make_tensor("r_row", TensorProto.FLOAT, [1, 1, 30, 1], r_indices))

    # valid_r and valid_c
    nodes.append(helper.make_node("ReduceMax", ["input"], ["valid_r"], axes=[3], keepdims=1))
    nodes.append(helper.make_node("ReduceMax", ["input"], ["valid_c"], axes=[2], keepdims=1))

    # h and w (still needed for Greater(h, 2) check)
    nodes.append(make_tensor("axes_23", TensorProto.INT64, [2], [2, 3]))
    nodes.append(helper.make_node("ReduceSum", ["valid_r", "axes_23"], ["h"], keepdims=1))
    nodes.append(helper.make_node("ReduceSum", ["valid_c", "axes_23"], ["w"], keepdims=1))

    # min_r, max_r, min_c, max_c
    nodes.append(helper.make_node("ArgMax", ["valid_r"], ["min_r_int64"], axis=2, keepdims=1))
    nodes.append(helper.make_node("ArgMax", ["valid_c"], ["min_c_int64"], axis=3, keepdims=1))
    nodes.append(helper.make_node("Cast", ["min_r_int64"], ["min_r"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Cast", ["min_c_int64"], ["min_c"], to=TensorProto.FLOAT))
    
    nodes.append(helper.make_node("Mul", ["valid_r", "r_row"], ["r_vals"]))
    nodes.append(helper.make_node("ReduceMax", ["r_vals"], ["max_r"], axes=[2], keepdims=1))
    nodes.append(helper.make_node("Mul", ["valid_c", "r_col"], ["c_vals"]))
    nodes.append(helper.make_node("ReduceMax", ["c_vals"], ["max_c_bound"], axes=[3], keepdims=1))

    # boundary mask construction
    nodes.append(helper.make_node("Equal", ["r_row", "min_r"], ["eq_min_r"]))
    nodes.append(helper.make_node("Equal", ["r_row", "max_r"], ["eq_max_r"]))
    nodes.append(helper.make_node("Or", ["eq_min_r", "eq_max_r"], ["B_r"]))

    nodes.append(helper.make_node("Equal", ["r_col", "min_c"], ["eq_min_c"]))
    nodes.append(helper.make_node("Equal", ["r_col", "max_c_bound"], ["eq_max_c"]))
    nodes.append(helper.make_node("Or", ["eq_min_c", "eq_max_c"], ["B_c"]))

    # GreaterOrEqual / LessOrEqual using Greater / Less / Equal
    nodes.append(helper.make_node("Greater", ["r_row", "min_r"], ["gt_min_r"]))
    nodes.append(helper.make_node("Or", ["gt_min_r", "eq_min_r"], ["in_r1"]))
    nodes.append(helper.make_node("Less", ["r_row", "max_r"], ["lt_max_r"]))
    nodes.append(helper.make_node("Or", ["lt_max_r", "eq_max_r"], ["in_r2"]))
    nodes.append(helper.make_node("And", ["in_r1", "in_r2"], ["in_r"]))

    nodes.append(helper.make_node("Greater", ["r_col", "min_c"], ["gt_min_c"]))
    nodes.append(helper.make_node("Or", ["gt_min_c", "eq_min_c"], ["in_c1"]))
    nodes.append(helper.make_node("Less", ["r_col", "max_c_bound"], ["lt_max_c"]))
    nodes.append(helper.make_node("Or", ["lt_max_c", "eq_max_c"], ["in_c2"]))
    nodes.append(helper.make_node("And", ["in_c1", "in_c2"], ["in_c"]))

    nodes.append(helper.make_node("Or", ["B_r", "B_c"], ["B_rc"]))
    nodes.append(helper.make_node("And", ["in_r", "in_c"], ["in_rc"]))
    nodes.append(helper.make_node("And", ["B_rc", "in_rc"], ["boundary_bool"]))
    nodes.append(helper.make_node("Cast", ["boundary_bool"], ["boundary_float"], to=TensorProto.FLOAT))

    # is_frame diff
    nodes.append(helper.make_node("Sub", ["input", "boundary_float"], ["diff1"]))
    nodes.append(helper.make_node("Abs", ["diff1"], ["diff"]))
    nodes.append(helper.make_node("ReduceSum", ["diff", "axes_23"], ["diff_sum"], keepdims=1))
    
    nodes.append(helper.make_node("Equal", ["diff_sum", "zero_float"], ["cond_sum"]))
    nodes.append(helper.make_node("Greater", ["h", "two_float"], ["cond_h"]))
    nodes.append(helper.make_node("Greater", ["w", "two_float"], ["cond_w"]))
    nodes.append(helper.make_node("And", ["cond_sum", "cond_h"], ["cond1"]))
    nodes.append(helper.make_node("And", ["cond1", "cond_w"], ["is_frame_bool"]))
    
    nodes.append(helper.make_node("Cast", ["is_frame_bool"], ["is_frame_float_all"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Mul", ["is_frame_float_all", "channel_mask"], ["is_frame_float"]))

    # frame_min_r, frame_min_c, frame_h, frame_w
    nodes.append(helper.make_node("Mul", ["min_r", "is_frame_float"], ["masked_min_r"]))
    nodes.append(helper.make_node("Mul", ["min_c", "is_frame_float"], ["masked_min_c"]))
    nodes.append(helper.make_node("Mul", ["h", "is_frame_float"], ["masked_h"]))
    nodes.append(helper.make_node("Mul", ["w", "is_frame_float"], ["masked_w"]))
    
    nodes.append(make_tensor("axes_1", TensorProto.INT64, [1], [1]))
    nodes.append(helper.make_node("ReduceSum", ["masked_min_r", "axes_1"], ["frame_min_r"], keepdims=1))
    nodes.append(helper.make_node("ReduceSum", ["masked_min_c", "axes_1"], ["frame_min_c"], keepdims=1))
    nodes.append(helper.make_node("ReduceSum", ["masked_h", "axes_1"], ["frame_h"], keepdims=1))
    nodes.append(helper.make_node("ReduceSum", ["masked_w", "axes_1"], ["frame_w"], keepdims=1))

    # Interior mask
    nodes.append(helper.make_node("Sub", ["frame_h", "one_float"], ["frame_h_minus_1"]))
    nodes.append(helper.make_node("Sub", ["frame_w", "one_float"], ["frame_w_minus_1"]))
    nodes.append(helper.make_node("Add", ["frame_min_r", "frame_h_minus_1"], ["interior_max_r"]))
    nodes.append(helper.make_node("Add", ["frame_min_c", "frame_w_minus_1"], ["interior_max_c"]))

    nodes.append(helper.make_node("Greater", ["r_row", "frame_min_r"], ["gt_r"]))
    nodes.append(helper.make_node("Less", ["r_row", "interior_max_r"], ["lt_r"]))
    nodes.append(helper.make_node("And", ["gt_r", "lt_r"], ["in_interior_r"]))
    
    nodes.append(helper.make_node("Greater", ["r_col", "frame_min_c"], ["gt_c"]))
    nodes.append(helper.make_node("Less", ["r_col", "interior_max_c"], ["lt_c"]))
    nodes.append(helper.make_node("And", ["gt_c", "lt_c"], ["in_interior_c"]))
    
    nodes.append(helper.make_node("And", ["in_interior_r", "in_interior_c"], ["interior_bool"]))
    nodes.append(helper.make_node("Cast", ["interior_bool"], ["interior_mask"], to=TensorProto.FLOAT))

    # Apply mask
    nodes.append(helper.make_node("Mul", ["input", "interior_mask"], ["interior_pixels"]))

    # Shift amounts
    nodes.append(helper.make_node("Add", ["frame_min_r", "one_float"], ["shift_r"]))
    nodes.append(helper.make_node("Add", ["frame_min_c", "one_float"], ["shift_c"]))

    # Shift matrices
    nodes.append(helper.make_node("Add", ["r_row", "shift_r"], ["shifted_r_row"]))
    nodes.append(helper.make_node("Add", ["r_row", "shift_c"], ["shifted_c_row"]))
    nodes.append(helper.make_node("Equal", ["r_col", "shifted_r_row"], ["P_r_bool"]))
    nodes.append(helper.make_node("Equal", ["r_col", "shifted_c_row"], ["P_c_bool"]))
    nodes.append(helper.make_node("Cast", ["P_r_bool"], ["P_r"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Cast", ["P_c_bool"], ["P_c"], to=TensorProto.FLOAT))

    # MatMul shift
    nodes.append(helper.make_node("Transpose", ["P_c"], ["P_c_T"], perm=[0, 1, 3, 2]))
    nodes.append(helper.make_node("MatMul", ["P_r", "interior_pixels"], ["shifted_rows"]))
    nodes.append(helper.make_node("MatMul", ["shifted_rows", "P_c_T"], ["output"]))

    graph = helper.make_graph(nodes, "task029", [X], [Y])
    model = helper.make_model(graph, producer_name='antigravity', ir_version=8, opset_imports=[helper.make_opsetid('', 13)])
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    onnx.save(model, "task029.onnx")
    print("Saved task029.onnx")


# Build the model (the function saves it internally, so we load the result)
build_model()
import glob
model = onnx.load("/project/repairs/task029.onnx")
