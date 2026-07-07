# Source: predicted/test_onnx_task021.py — ONNX graph construction code
# Verified model: repairs/task021.onnx
import onnx
import onnx.helper as helper
from onnx import TensorProto

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
    make_tensor("one_float", TensorProto.FLOAT, [1], [1.0])
    
    make_tensor("axes_1", TensorProto.INT64, [1], [1])
    make_tensor("axes_2_3", TensorProto.INT64, [2], [2, 3])
    make_tensor("axes_1_3", TensorProto.INT64, [2], [1, 3])
    make_tensor("axes_1_2", TensorProto.INT64, [2], [1, 2])
    make_tensor("axes_3", TensorProto.INT64, [1], [3])
    make_tensor("axes_2", TensorProto.INT64, [1], [2])
    
    # Slice input_1_9
    nodes.append(helper.make_node("Slice", ["input", "one_int", "ten_int", "axes_1"], ["input_1_9"]))
    
    # counts_1_9 = sum(input_1_9, axis=(2, 3))
    nodes.append(helper.make_node("ReduceSum", ["input_1_9", "axes_2_3"], ["counts_1_9"], keepdims=1))
    
    # bg_idx = argmax(counts_1_9, axis=1)
    nodes.append(helper.make_node("ArgMax", ["counts_1_9"], ["bg_idx"], axis=1, keepdims=1))
    
    # bg_channel_idx = bg_idx + 1 (int64)
    nodes.append(helper.make_node("Add", ["bg_idx", "one_int"], ["bg_channel_idx"]))
    
    # bg_onehot (1, 9, 1, 1)
    make_tensor("range_1_10", TensorProto.INT64, [1, 9, 1, 1], list(range(1, 10)))
    nodes.append(helper.make_node("Equal", ["range_1_10", "bg_channel_idx"], ["bg_onehot_bool"]))
    nodes.append(helper.make_node("Cast", ["bg_onehot_bool"], ["bg_onehot"], to=TensorProto.FLOAT))
    
    # input_bg = sum(input_1_9 * bg_onehot, axis=1)
    nodes.append(helper.make_node("Mul", ["input_1_9", "bg_onehot"], ["input_1_9_bg"]))
    nodes.append(helper.make_node("ReduceSum", ["input_1_9_bg", "axes_1"], ["input_bg"], keepdims=1))
    
    # valid_rows = max(input_1_9, axes=[1, 3])
    nodes.append(helper.make_node("ReduceMax", ["input_1_9"], ["valid_rows"], axes=[1, 3], keepdims=1))
    
    # bg_in_rows = max(input_bg, axes=[3])
    nodes.append(helper.make_node("ReduceMax", ["input_bg"], ["bg_in_rows"], axes=[3], keepdims=1))
    
    # is_h_line = valid_rows * (1.0 - bg_in_rows)
    nodes.append(helper.make_node("Sub", ["one_float", "bg_in_rows"], ["not_bg_in_rows"]))
    nodes.append(helper.make_node("Mul", ["valid_rows", "not_bg_in_rows"], ["is_h_line"]))
    
    # h_lines = sum(is_h_line, axes=[2, 3])
    nodes.append(helper.make_node("ReduceSum", ["is_h_line", "axes_2_3"], ["h_lines"], keepdims=1))
    
    # h_out = h_lines + 1.0
    nodes.append(helper.make_node("Add", ["h_lines", "one_float"], ["h_out"]))
    
    # valid_cols = max(input_1_9, axes=[1, 2])
    nodes.append(helper.make_node("ReduceMax", ["input_1_9"], ["valid_cols"], axes=[1, 2], keepdims=1))
    
    # bg_in_cols = max(input_bg, axes=[2])
    nodes.append(helper.make_node("ReduceMax", ["input_bg"], ["bg_in_cols"], axes=[2], keepdims=1))
    
    # is_v_line = valid_cols * (1.0 - bg_in_cols)
    nodes.append(helper.make_node("Sub", ["one_float", "bg_in_cols"], ["not_bg_in_cols"]))
    nodes.append(helper.make_node("Mul", ["valid_cols", "not_bg_in_cols"], ["is_v_line"]))
    
    # v_lines = sum(is_v_line, axes=[2, 3])
    nodes.append(helper.make_node("ReduceSum", ["is_v_line", "axes_2_3"], ["v_lines"], keepdims=1))
    
    # w_out = v_lines + 1.0
    nodes.append(helper.make_node("Add", ["v_lines", "one_float"], ["w_out"]))
    
    # valid_r_out
    make_tensor("r_indices", TensorProto.FLOAT, [1, 1, 30, 1], list(range(30)))
    nodes.append(helper.make_node("Less", ["r_indices", "h_out"], ["valid_r_bool"]))
    nodes.append(helper.make_node("Cast", ["valid_r_bool"], ["valid_r_out"], to=TensorProto.FLOAT))
    
    # valid_c_out
    make_tensor("c_indices", TensorProto.FLOAT, [1, 1, 1, 30], list(range(30)))
    nodes.append(helper.make_node("Less", ["c_indices", "w_out"], ["valid_c_bool"]))
    nodes.append(helper.make_node("Cast", ["valid_c_bool"], ["valid_c_out"], to=TensorProto.FLOAT))
    
    # valid_grid_out
    nodes.append(helper.make_node("Mul", ["valid_r_out", "valid_c_out"], ["valid_grid_out"]))
    
    # output_1_9
    nodes.append(helper.make_node("Mul", ["bg_onehot", "valid_grid_out"], ["output_1_9"]))
    
    # output_0
    nodes.append(helper.make_node("Sub", ["valid_grid_out", "valid_grid_out"], ["output_0"]))
    
    # output
    nodes.append(helper.make_node("Concat", ["output_0", "output_1_9"], ["output"], axis=1))
    
    graph = helper.make_graph(nodes, "task21_graph", [X], [Y], inits)
    model = helper.make_model(graph, producer_name='antigravity', ir_version=8, opset_imports=[helper.make_opsetid('', 13)])
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    onnx.save(model, "task021.onnx")
    print("Saved task021.onnx")


# Build the model (the function saves it internally, so we load the result)
build_model()
import glob
model = onnx.load("/project/repairs/task021.onnx")
