# Source: predicted/test_onnx_task014.py — ONNX graph construction code
# Verified model: repairs/task014.onnx
import onnx
import onnx.helper as helper
from onnx import TensorProto
import numpy as np

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
    make_tensor("thirty_int", TensorProto.INT64, [1], [30])
    make_tensor("huge_float", TensorProto.FLOAT, [1], [99999.0])
    make_tensor("huge_int", TensorProto.INT64, [1], [99999])
    make_tensor("one_float", TensorProto.FLOAT, [1], [1.0])
    
    make_tensor("axes_c", TensorProto.INT64, [1], [1])
    make_tensor("axes_h", TensorProto.INT64, [1], [2])
    make_tensor("axes_w", TensorProto.INT64, [1], [3])
    make_tensor("axes_hw", TensorProto.INT64, [2], [2, 3])
    
    # input_1_9
    nodes.append(helper.make_node("Slice", ["input", "one_int", "ten_int", "axes_c"], ["input_1_9"]))
    
    # 1. Find counts of all colors
    nodes.append(helper.make_node("ReduceSum", ["input_1_9", "axes_hw"], ["counts"], keepdims=1))
    
    # 2. counts_valid = Where(counts > 0, counts, 99999.0)
    make_tensor("zero_float", TensorProto.FLOAT, [1], [0.0])
    nodes.append(helper.make_node("Greater", ["counts", "zero_float"], ["counts_gt_zero"]))
    nodes.append(helper.make_node("Where", ["counts_gt_zero", "counts", "huge_float"], ["counts_valid"]))
    
    # 3. min_count
    nodes.append(helper.make_node("ReduceMin", ["counts_valid"], ["min_count"], axes=[1], keepdims=1))
    
    # 4. is_target
    nodes.append(helper.make_node("Equal", ["counts_valid", "min_count"], ["is_target"]))
    
    # 5. target_pixels
    nodes.append(helper.make_node("Cast", ["is_target"], ["is_target_float"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Mul", ["is_target_float", "input_1_9"], ["target_pixels"]))
    
    # 6. target_presence (1x1x30x30)
    nodes.append(helper.make_node("ReduceMax", ["target_pixels"], ["target_presence"], axes=[1], keepdims=1))
    
    # 7. target_presence_r and target_presence_c
    nodes.append(helper.make_node("ReduceMax", ["target_presence"], ["target_presence_r"], axes=[3], keepdims=1))
    nodes.append(helper.make_node("ReduceMax", ["target_presence"], ["target_presence_c"], axes=[2], keepdims=1))
    
    # 8. Get r_min, r_max
    make_tensor("zero_scalar", TensorProto.INT64, [], [0])
    make_tensor("thirty_scalar", TensorProto.INT64, [], [30])
    make_tensor("one_scalar", TensorProto.INT64, [], [1])
    nodes.append(helper.make_node("Range", ["zero_scalar", "thirty_scalar", "one_scalar"], ["range_30"]))
    
    make_tensor("shape_1_1_30_1", TensorProto.INT64, [4], [1, 1, 30, 1])
    nodes.append(helper.make_node("Reshape", ["range_30", "shape_1_1_30_1"], ["r_indices"]))
    
    nodes.append(helper.make_node("Cast", ["target_presence_r"], ["target_presence_r_int"], to=TensorProto.INT64))
    nodes.append(helper.make_node("Mul", ["target_presence_r_int", "r_indices"], ["valid_r_indices"]))
    nodes.append(helper.make_node("ReduceMax", ["valid_r_indices"], ["r_max"], axes=[2], keepdims=1))
    
    nodes.append(helper.make_node("Greater", ["target_presence_r_int", "zero_int"], ["target_presence_r_gt_zero"]))
    nodes.append(helper.make_node("Where", ["target_presence_r_gt_zero", "r_indices", "huge_int"], ["invalid_r_indices"]))
    nodes.append(helper.make_node("ReduceMin", ["invalid_r_indices"], ["r_min"], axes=[2], keepdims=1))
    
    # 9. Get c_min, c_max
    make_tensor("shape_1_1_1_30", TensorProto.INT64, [4], [1, 1, 1, 30])
    nodes.append(helper.make_node("Reshape", ["range_30", "shape_1_1_1_30"], ["c_indices"]))
    
    nodes.append(helper.make_node("Cast", ["target_presence_c"], ["target_presence_c_int"], to=TensorProto.INT64))
    nodes.append(helper.make_node("Mul", ["target_presence_c_int", "c_indices"], ["valid_c_indices"]))
    nodes.append(helper.make_node("ReduceMax", ["valid_c_indices"], ["c_max"], axes=[3], keepdims=1))
    
    nodes.append(helper.make_node("Greater", ["target_presence_c_int", "zero_int"], ["target_presence_c_gt_zero"]))
    nodes.append(helper.make_node("Where", ["target_presence_c_gt_zero", "c_indices", "huge_int"], ["invalid_c_indices"]))
    nodes.append(helper.make_node("ReduceMin", ["invalid_c_indices"], ["c_min"], axes=[3], keepdims=1))
    
    # 10. bbox_mask
    nodes.append(helper.make_node("GreaterOrEqual", ["r_indices", "r_min"], ["r_ge_min"]))
    nodes.append(helper.make_node("LessOrEqual", ["r_indices", "r_max"], ["r_le_max"]))
    nodes.append(helper.make_node("And", ["r_ge_min", "r_le_max"], ["mask_r"]))
    
    nodes.append(helper.make_node("GreaterOrEqual", ["c_indices", "c_min"], ["c_ge_min"]))
    nodes.append(helper.make_node("LessOrEqual", ["c_indices", "c_max"], ["c_le_max"]))
    nodes.append(helper.make_node("And", ["c_ge_min", "c_le_max"], ["mask_c"]))
    
    nodes.append(helper.make_node("And", ["mask_r", "mask_c"], ["bbox_mask_bool"]))
    nodes.append(helper.make_node("Cast", ["bbox_mask_bool"], ["bbox_mask"], to=TensorProto.FLOAT))
    
    # 11. cropped_1_9
    nodes.append(helper.make_node("Mul", ["input_1_9", "bbox_mask"], ["cropped_1_9"]))
    
    # 12. Translation matrices
    make_tensor("shape_30_1", TensorProto.INT64, [2], [30, 1])
    make_tensor("shape_1_30", TensorProto.INT64, [2], [1, 30])
    nodes.append(helper.make_node("Reshape", ["range_30", "shape_30_1"], ["i_indices"]))
    nodes.append(helper.make_node("Reshape", ["range_30", "shape_1_30"], ["j_indices"]))
    
    make_tensor("shape_1", TensorProto.INT64, [1], [1])
    nodes.append(helper.make_node("Reshape", ["r_min", "shape_1"], ["r_min_flat"]))
    nodes.append(helper.make_node("Add", ["i_indices", "r_min_flat"], ["i_plus_r"]))
    nodes.append(helper.make_node("Equal", ["j_indices", "i_plus_r"], ["Ty_bool"]))
    nodes.append(helper.make_node("Cast", ["Ty_bool"], ["Ty_float_2d"], to=TensorProto.FLOAT))
    
    make_tensor("shape_1_1_30_30", TensorProto.INT64, [4], [1, 1, 30, 30])
    nodes.append(helper.make_node("Reshape", ["Ty_float_2d", "shape_1_1_30_30"], ["Ty_float"]))
    
    nodes.append(helper.make_node("Reshape", ["c_min", "shape_1"], ["c_min_flat"]))
    nodes.append(helper.make_node("Add", ["i_indices", "c_min_flat"], ["i_plus_c"]))
    nodes.append(helper.make_node("Equal", ["j_indices", "i_plus_c"], ["Tx_bool"]))
    nodes.append(helper.make_node("Cast", ["Tx_bool"], ["Tx_float_2d"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Reshape", ["Tx_float_2d", "shape_1_1_30_30"], ["Tx_float"]))
    
    # Transpose last two dims of Tx
    nodes.append(helper.make_node("Transpose", ["Tx_float"], ["Tx_T"], perm=[0, 1, 3, 2]))
    
    # 13. shifted_1_9
    nodes.append(helper.make_node("MatMul", ["Ty_float", "cropped_1_9"], ["shifted_temp"]))
    nodes.append(helper.make_node("MatMul", ["shifted_temp", "Tx_T"], ["shifted_1_9"]))
    
    # 14. recompute bg
    nodes.append(helper.make_node("ReduceSum", ["shifted_1_9", "axes_c"], ["sum_1_9"], keepdims=1))
    nodes.append(helper.make_node("Sub", ["one_float", "sum_1_9"], ["new_bg"]))
    
    nodes.append(helper.make_node("Concat", ["new_bg", "shifted_1_9"], ["output_unmasked"], axis=1))
    
    # 15. Valid grid mask
    nodes.append(helper.make_node("Sub", ["r_max", "r_min"], ["h_minus_1"]))
    nodes.append(helper.make_node("Add", ["h_minus_1", "one_int"], ["h_tensor"]))
    nodes.append(helper.make_node("Sub", ["c_max", "c_min"], ["w_minus_1"]))
    nodes.append(helper.make_node("Add", ["w_minus_1", "one_int"], ["w_tensor"]))
    
    nodes.append(helper.make_node("Reshape", ["h_tensor", "shape_1"], ["h_flat"]))
    nodes.append(helper.make_node("Reshape", ["w_tensor", "shape_1"], ["w_flat"]))
    
    nodes.append(helper.make_node("Less", ["i_indices", "h_flat"], ["valid_r"]))
    nodes.append(helper.make_node("Less", ["j_indices", "w_flat"], ["valid_c"]))
    nodes.append(helper.make_node("And", ["valid_r", "valid_c"], ["valid_mask"]))
    nodes.append(helper.make_node("Cast", ["valid_mask"], ["valid_mask_float"], to=TensorProto.FLOAT))
    
    nodes.append(helper.make_node("Mul", ["output_unmasked", "valid_mask_float"], ["output"]))
    
    graph = helper.make_graph(nodes, "task14_graph", [X], [Y], inits)
    model = helper.make_model(graph, producer_name='antigravity', ir_version=8, opset_imports=[helper.make_opsetid('', 13)])
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    onnx.save(model, "task014.onnx")
    print("Saved task014.onnx")


# Build the model (the function saves it internally, so we load the result)
build_model()
import glob
model = onnx.load("/project/repairs/task014.onnx")
