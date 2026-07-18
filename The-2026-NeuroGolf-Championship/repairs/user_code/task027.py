# Source: predicted/test_onnx_task027.py — ONNX graph construction code
# Verified model: repairs/task027.onnx
import onnx
from onnx import helper, TensorProto

def make_tensor(name, data_type, dims, vals):
    return helper.make_node("Constant", [], [name], value=helper.make_tensor(name+"_v", data_type, dims, vals))

def build_model():
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10, 30, 30])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10, 30, 30])

    nodes = []
    
    # Constants
    nodes.append(make_tensor("zero_float_scalar", TensorProto.FLOAT, [], [0.0]))
    nodes.append(make_tensor("one_float_scalar", TensorProto.FLOAT, [], [1.0]))
    
    r_indices = list(range(30))
    nodes.append(make_tensor("r_col", TensorProto.FLOAT, [1, 1, 1, 30], r_indices))
    nodes.append(make_tensor("r_row", TensorProto.FLOAT, [1, 1, 30, 1], r_indices))
    nodes.append(make_tensor("twenty_nine", TensorProto.FLOAT, [1, 1, 1, 1], [29.0]))
    nodes.append(make_tensor("one_float", TensorProto.FLOAT, [1, 1, 1, 1], [1.0]))
    nodes.append(make_tensor("two_float", TensorProto.FLOAT, [1, 1, 1, 1], [2.0]))
    
    # Slice channels 0, 1, 2
    nodes.append(make_tensor("axes_1", TensorProto.INT64, [1], [1]))
    
    nodes.append(make_tensor("starts_0", TensorProto.INT64, [1], [0]))
    nodes.append(make_tensor("ends_1", TensorProto.INT64, [1], [1]))
    nodes.append(helper.make_node("Slice", ["input", "starts_0", "ends_1", "axes_1"], ["in_0"]))

    nodes.append(make_tensor("starts_1", TensorProto.INT64, [1], [1]))
    nodes.append(make_tensor("ends_2", TensorProto.INT64, [1], [2]))
    nodes.append(helper.make_node("Slice", ["input", "starts_1", "ends_2", "axes_1"], ["in_1"]))

    nodes.append(make_tensor("starts_2", TensorProto.INT64, [1], [2]))
    nodes.append(make_tensor("ends_3", TensorProto.INT64, [1], [3]))
    nodes.append(helper.make_node("Slice", ["input", "starts_2", "ends_3", "axes_1"], ["in_2"]))
    
    nodes.append(make_tensor("starts_3", TensorProto.INT64, [1], [3]))
    nodes.append(make_tensor("ends_10", TensorProto.INT64, [1], [10]))
    nodes.append(helper.make_node("Slice", ["input", "starts_3", "ends_10", "axes_1"], ["in_3_9"]))

    blue_float = "in_1"

    # r_max_blue
    nodes.append(helper.make_node("Mul", [blue_float, "r_row"], ["r_masked"]))
    nodes.append(helper.make_node("ReduceMax", ["r_masked"], ["r_max"], axes=[2, 3], keepdims=1))

    # c_max_blue
    nodes.append(helper.make_node("Mul", [blue_float, "r_col"], ["c_masked"]))
    nodes.append(helper.make_node("ReduceMax", ["c_masked"], ["c_max"], axes=[2, 3], keepdims=1))

    # r_min_blue
    nodes.append(helper.make_node("Sub", ["twenty_nine", "r_row"], ["r_inv"]))
    nodes.append(helper.make_node("Mul", [blue_float, "r_inv"], ["r_inv_masked"]))
    nodes.append(helper.make_node("ReduceMax", ["r_inv_masked"], ["r_min_inv"], axes=[2, 3], keepdims=1))
    nodes.append(helper.make_node("Sub", ["twenty_nine", "r_min_inv"], ["r_min"]))

    # c_min_blue
    nodes.append(helper.make_node("Sub", ["twenty_nine", "r_col"], ["c_inv"]))
    nodes.append(helper.make_node("Mul", [blue_float, "c_inv"], ["c_inv_masked"]))
    nodes.append(helper.make_node("ReduceMax", ["c_inv_masked"], ["c_min_inv"], axes=[2, 3], keepdims=1))
    nodes.append(helper.make_node("Sub", ["twenty_nine", "c_min_inv"], ["c_min"]))

    # S_r = r_max - r_min + 1
    nodes.append(helper.make_node("Sub", ["r_max", "r_min"], ["S_r_sub"]))
    nodes.append(helper.make_node("Add", ["S_r_sub", "one_float"], ["S_r"]))
    
    # S_c = c_max - c_min + 1
    nodes.append(helper.make_node("Sub", ["c_max", "c_min"], ["S_c_sub"]))
    nodes.append(helper.make_node("Add", ["S_c_sub", "one_float"], ["S_c"]))
    
    # S = Max(S_r, S_c)
    nodes.append(helper.make_node("Max", ["S_r", "S_c"], ["S"]))

    # mapped_r_val = 2 * r_min + S - 1
    nodes.append(helper.make_node("Mul", ["r_min", "two_float"], ["r_min_2"]))
    nodes.append(helper.make_node("Add", ["r_min_2", "S"], ["mapped_r_add"]))
    nodes.append(helper.make_node("Sub", ["mapped_r_add", "one_float"], ["mapped_r_val"]))

    # mapped_c_val = 2 * c_max - S + 1
    nodes.append(helper.make_node("Mul", ["c_max", "two_float"], ["c_max_2"]))
    nodes.append(helper.make_node("Sub", ["c_max_2", "S"], ["mapped_c_sub"]))
    nodes.append(helper.make_node("Add", ["mapped_c_sub", "one_float"], ["mapped_c_val"]))

    # P_r
    nodes.append(helper.make_node("Add", ["r_row", "r_col"], ["r_plus_c"]))
    nodes.append(helper.make_node("Equal", ["r_plus_c", "mapped_r_val"], ["P_r_bool"]))
    nodes.append(helper.make_node("Cast", ["P_r_bool"], ["P_r"], to=TensorProto.FLOAT))

    # P_c
    nodes.append(helper.make_node("Equal", ["r_plus_c", "mapped_c_val"], ["P_c_bool"]))
    nodes.append(helper.make_node("Cast", ["P_c_bool"], ["P_c"], to=TensorProto.FLOAT))

    # mapped blue
    nodes.append(helper.make_node("MatMul", ["P_r", blue_float], ["blue_mapped_r"]))
    nodes.append(helper.make_node("MatMul", ["blue_mapped_r", "P_c"], ["blue_mapped"]))

    # red_float = blue_mapped * (1 - blue_float)
    nodes.append(helper.make_node("Sub", ["one_float", blue_float], ["blue_not_float"]))
    nodes.append(helper.make_node("Mul", ["blue_mapped", "blue_not_float"], ["red_float"]))

    # Output channels
    # out_0 = Clip(in_0 - red_float)
    nodes.append(helper.make_node("Sub", ["in_0", "red_float"], ["out_0_raw"]))
    nodes.append(helper.make_node("Clip", ["out_0_raw", "zero_float_scalar", "one_float_scalar"], ["out_0"]))
    
    # out_2 = Clip(in_2 + red_float)
    nodes.append(helper.make_node("Add", ["in_2", "red_float"], ["out_2_raw"]))
    nodes.append(helper.make_node("Clip", ["out_2_raw", "zero_float_scalar", "one_float_scalar"], ["out_2"]))

    nodes.append(helper.make_node("Concat", ["out_0", "in_1", "out_2", "in_3_9"], ["output_unmasked"], axis=1))
    nodes.append(helper.make_node("ReduceMax", ["input"], ["presence_mask"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("Mul", ["output_unmasked", "presence_mask"], ["output"]))

    graph = helper.make_graph(nodes, "task027", [X], [Y])
    model = helper.make_model(graph, producer_name='antigravity', ir_version=8, opset_imports=[helper.make_opsetid('', 13)])
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    onnx.save(model, "task027.onnx")
    print("Saved task027.onnx")


# Build the model (the function saves it internally, so we load the result)
build_model()
import glob
model = onnx.load("/project/repairs/task027.onnx")
