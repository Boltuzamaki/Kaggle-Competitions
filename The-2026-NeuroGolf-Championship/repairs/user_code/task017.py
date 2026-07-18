# Source: predicted/test_onnx_task017.py — ONNX graph construction code
# Verified model: repairs/task017.onnx
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
    make_tensor("zero_float", TensorProto.FLOAT, [1], [0.0])
    make_tensor("one_float", TensorProto.FLOAT, [1], [1.0])
    
    make_tensor("axes_1", TensorProto.INT64, [1], [1])
    make_tensor("axes_2", TensorProto.INT64, [1], [2])
    make_tensor("axes_3", TensorProto.INT64, [1], [3])
    make_tensor("axes_2_3", TensorProto.INT64, [2], [2, 3])
    
    # 0. Slice input 1..9
    nodes.append(helper.make_node("Slice", ["input", "one_int", "ten_int", "axes_1"], ["input_1_9"]))
    
    # 1. Compute h and w
    nodes.append(helper.make_node("ReduceMax", ["input_1_9"], ["target_presence"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("ReduceMax", ["target_presence"], ["target_presence_r"], axes=[3], keepdims=1))
    nodes.append(helper.make_node("ReduceMax", ["target_presence"], ["target_presence_c"], axes=[2], keepdims=1))
    
    make_tensor("zero_scalar", TensorProto.INT64, [], [0])
    make_tensor("thirty_scalar", TensorProto.INT64, [], [30])
    make_tensor("one_scalar", TensorProto.INT64, [], [1])
    nodes.append(helper.make_node("Range", ["zero_scalar", "thirty_scalar", "one_scalar"], ["range_30"]))
    
    make_tensor("shape_1_1_30_1", TensorProto.INT64, [4], [1, 1, 30, 1])
    nodes.append(helper.make_node("Reshape", ["range_30", "shape_1_1_30_1"], ["r_indices"]))
    nodes.append(helper.make_node("Cast", ["target_presence_r"], ["target_presence_r_int"], to=TensorProto.INT64))
    nodes.append(helper.make_node("Mul", ["target_presence_r_int", "r_indices"], ["valid_r_indices"]))
    nodes.append(helper.make_node("ReduceMax", ["valid_r_indices"], ["r_max"], axes=[2], keepdims=1))
    nodes.append(helper.make_node("Add", ["r_max", "one_int"], ["h"]))
    
    make_tensor("shape_1_1_1_30", TensorProto.INT64, [4], [1, 1, 1, 30])
    nodes.append(helper.make_node("Reshape", ["range_30", "shape_1_1_1_30"], ["c_indices"]))
    nodes.append(helper.make_node("Cast", ["target_presence_c"], ["target_presence_c_int"], to=TensorProto.INT64))
    nodes.append(helper.make_node("Mul", ["target_presence_c_int", "c_indices"], ["valid_c_indices"]))
    nodes.append(helper.make_node("ReduceMax", ["valid_c_indices"], ["c_max"], axes=[3], keepdims=1))
    nodes.append(helper.make_node("Add", ["c_max", "one_int"], ["w"]))
    
    # 2. Iterate p
    make_tensor("shape_1_30", TensorProto.INT64, [2], [1, 30])
    nodes.append(helper.make_node("Reshape", ["range_30", "shape_1_30"], ["r_indices_1_30"]))
    nodes.append(helper.make_node("Reshape", ["range_30", "shape_1_30"], ["c_indices_1_30"]))
    
    make_tensor("accum_tiled", TensorProto.FLOAT, [1, 9, 30, 30], [0.0] * (1 * 9 * 30 * 30))
    make_tensor("accum_sel", TensorProto.FLOAT, [1, 1, 1, 1], [0.0])
    
    prev_tiled = "accum_tiled"
    prev_sel = "accum_sel"
    
    for p in range(2, 10):
        p_tensor = f"p_{p}"
        make_tensor(p_tensor, TensorProto.INT64, [1], [p])
        
        nodes.append(helper.make_node("Mod", ["r_indices_1_30", p_tensor], [f"r_mod_{p}"]))
        nodes.append(helper.make_node("Mod", ["c_indices_1_30", p_tensor], [f"c_mod_{p}"]))
        
        range_p = f"range_{p}"
        p_scalar = f"p_scalar_{p}"
        make_tensor(p_scalar, TensorProto.INT64, [], [p])
        nodes.append(helper.make_node("Range", ["zero_scalar", p_scalar, "one_scalar"], [range_p]))
        
        shape_p_1 = f"shape_{p}_1"
        make_tensor(shape_p_1, TensorProto.INT64, [2], [p, 1])
        i_indices = f"i_indices_{p}"
        nodes.append(helper.make_node("Reshape", [range_p, shape_p_1], [i_indices]))
        
        # mask_r_float
        mask_r = f"mask_r_{p}"
        mask_r_float = f"mask_r_float_{p}"
        nodes.append(helper.make_node("Equal", [f"r_mod_{p}", i_indices], [mask_r]))
        nodes.append(helper.make_node("Cast", [mask_r], [mask_r_float], to=TensorProto.FLOAT))
        
        # mask_c_float
        mask_c = f"mask_c_{p}"
        mask_c_float = f"mask_c_float_{p}"
        nodes.append(helper.make_node("Equal", [f"c_mod_{p}", i_indices], [mask_c]))
        nodes.append(helper.make_node("Cast", [mask_c], [mask_c_float], to=TensorProto.FLOAT))
        mask_c_float_T = f"mask_c_float_T_{p}"
        nodes.append(helper.make_node("Transpose", [mask_c_float], [mask_c_float_T], perm=[1, 0]))
        
        # base_counts = mask_r @ input_1_9 @ mask_c.T
        temp1 = f"temp1_{p}"
        base_counts = f"base_counts_{p}"
        nodes.append(helper.make_node("MatMul", [mask_r_float, "input_1_9"], [temp1]))
        nodes.append(helper.make_node("MatMul", [temp1, mask_c_float_T], [base_counts]))
        
        # base_presence
        base_presence = f"base_presence_{p}"
        base_presence_bool = f"base_presence_bool_{p}"
        nodes.append(helper.make_node("Greater", [base_counts, "zero_float"], [base_presence_bool]))
        nodes.append(helper.make_node("Cast", [base_presence_bool], [base_presence], to=TensorProto.FLOAT))
        
        # num_colors = sum(base_presence, axis=1)
        num_colors = f"num_colors_{p}"
        nodes.append(helper.make_node("ReduceSum", [base_presence, "axes_1"], [num_colors], keepdims=1))
        
        # is_valid = (num_colors == 1)
        is_valid = f"is_valid_{p}"
        is_valid_int = f"is_valid_int_{p}"
        nodes.append(helper.make_node("Equal", [num_colors, "one_float"], [is_valid]))
        nodes.append(helper.make_node("Cast", [is_valid], [is_valid_int], to=TensorProto.INT64))
        
        # all_valid
        all_valid_int = f"all_valid_int_{p}"
        all_valid_float = f"all_valid_float_{p}"
        nodes.append(helper.make_node("ReduceMin", [is_valid_int], [all_valid_int], axes=[2, 3], keepdims=1))
        nodes.append(helper.make_node("Cast", [all_valid_int], [all_valid_float], to=TensorProto.FLOAT))
        
        # tiled = mask_r.T @ base_presence @ mask_c
        mask_r_float_T = f"mask_r_float_T_{p}"
        nodes.append(helper.make_node("Transpose", [mask_r_float], [mask_r_float_T], perm=[1, 0]))
        temp2 = f"temp2_{p}"
        tiled = f"tiled_{p}"
        nodes.append(helper.make_node("MatMul", [mask_r_float_T, base_presence], [temp2]))
        nodes.append(helper.make_node("MatMul", [temp2, mask_c_float], [tiled]))
        
        # sel = all_valid * (1 - prev_sel)
        not_prev_sel = f"not_prev_sel_{p}"
        sel = f"sel_{p}"
        nodes.append(helper.make_node("Sub", ["one_float", prev_sel], [not_prev_sel]))
        nodes.append(helper.make_node("Mul", [all_valid_float, not_prev_sel], [sel]))
        
        # add to accum_tiled
        sel_tiled = f"sel_tiled_{p}"
        next_tiled = f"next_tiled_{p}"
        nodes.append(helper.make_node("Mul", [sel, tiled], [sel_tiled]))
        nodes.append(helper.make_node("Add", [prev_tiled, sel_tiled], [next_tiled]))
        
        # next_sel = prev_sel + sel
        next_sel = f"next_sel_{p}"
        nodes.append(helper.make_node("Add", [prev_sel, sel], [next_sel]))
        
        prev_tiled = next_tiled
        prev_sel = next_sel
        
    # 3. Output is prev_tiled
    final_tiled_1_9 = prev_tiled
    
    # 4. Valid grid mask
    make_tensor("shape_1", TensorProto.INT64, [1], [1])
    nodes.append(helper.make_node("Reshape", ["h", "shape_1"], ["h_flat"]))
    nodes.append(helper.make_node("Reshape", ["w", "shape_1"], ["w_flat"]))
    
    nodes.append(helper.make_node("Less", ["r_indices_1_30", "h_flat"], ["valid_r"]))
    valid_r_T = "valid_r_T"
    nodes.append(helper.make_node("Transpose", ["valid_r"], [valid_r_T], perm=[1, 0]))
    
    nodes.append(helper.make_node("Less", ["c_indices_1_30", "w_flat"], ["valid_c"]))
    
    nodes.append(helper.make_node("And", [valid_r_T, "valid_c"], ["valid_grid"]))
    nodes.append(helper.make_node("Cast", ["valid_grid"], ["valid_grid_float"], to=TensorProto.FLOAT))
    
    # Mask final_tiled_1_9
    nodes.append(helper.make_node("Mul", [final_tiled_1_9, "valid_grid_float"], ["output_1_9"]))
    
    nodes.append(helper.make_node("ReduceSum", ["output_1_9", "axes_1"], ["sum_1_9"], keepdims=1))
    nodes.append(helper.make_node("Sub", ["one_float", "sum_1_9"], ["output_0_unmasked"]))
    nodes.append(helper.make_node("Mul", ["output_0_unmasked", "valid_grid_float"], ["output_0"]))
    
    nodes.append(helper.make_node("Concat", ["output_0", "output_1_9"], ["output"], axis=1))
    
    graph = helper.make_graph(nodes, "task17_graph", [X], [Y], inits)
    model = helper.make_model(graph, producer_name='antigravity', ir_version=8, opset_imports=[helper.make_opsetid('', 13)])
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    onnx.save(model, "task017.onnx")
    print("Saved task017.onnx")


# Build the model (the function saves it internally, so we load the result)
build_model()
import glob
model = onnx.load("/project/repairs/task017.onnx")
