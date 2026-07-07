# Source: predicted/test_onnx_task011.py — ONNX graph construction code
# Verified model: repairs/task011.onnx
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
    make_tensor("five_int", TensorProto.INT64, [1], [5])
    make_tensor("six_int", TensorProto.INT64, [1], [6])
    make_tensor("ten_int", TensorProto.INT64, [1], [10])
    make_tensor("eleven_int", TensorProto.INT64, [1], [11])
    
    make_tensor("axes_hw", TensorProto.INT64, [2], [2, 3])
    make_tensor("axes_c", TensorProto.INT64, [1], [1])
    
    make_tensor("starts_0_0", TensorProto.INT64, [2], [0, 0])
    make_tensor("ends_11_11", TensorProto.INT64, [2], [11, 11])
    
    # 1. M_11 = Slice(input, starts=[0,0], ends=[11,11], axes=[2,3])
    nodes.append(helper.make_node("Slice", ["input", "starts_0_0", "ends_11_11", "axes_hw"], ["M_11"]))
    
    # 2. grid_mask_11
    nodes.append(helper.make_node("Slice", ["M_11", "five_int", "six_int", "axes_c"], ["grid_mask_11"]))
    
    # 3. colored_mask
    nodes.append(helper.make_node("Slice", ["M_11", "one_int", "five_int", "axes_c"], ["c1_4"]))
    nodes.append(helper.make_node("Slice", ["M_11", "six_int", "ten_int", "axes_c"], ["c6_9"]))
    nodes.append(helper.make_node("Concat", ["c1_4", "c6_9"], ["c_all"], axis=1))
    nodes.append(helper.make_node("ReduceSum", ["c_all", "axes_c"], ["colored_mask"], keepdims=1))
    
    # 4. count_3x3
    make_tensor("conv_weight", TensorProto.FLOAT, [1, 1, 3, 3], [1.0]*9)
    nodes.append(helper.make_node("Conv", ["colored_mask", "conv_weight"], ["count_3x3"], strides=[4, 4]))
    
    # 5. min_count
    nodes.append(helper.make_node("ReduceMin", ["count_3x3"], ["min_count"], axes=[2, 3], keepdims=1))
    
    # 6. is_template
    make_tensor("half", TensorProto.FLOAT, [1], [0.5])
    nodes.append(helper.make_node("Add", ["min_count", "half"], ["threshold"]))
    nodes.append(helper.make_node("Less", ["count_3x3", "threshold"], ["is_template_bool"]))
    nodes.append(helper.make_node("Cast", ["is_template_bool"], ["is_template_float"], to=TensorProto.FLOAT))
    
    # 7. template_mask_11
    nodes.append(helper.make_node("ConvTranspose", ["is_template_float", "conv_weight"], ["template_mask_11"], strides=[4, 4]))
    
    # 8. masked_M_11
    nodes.append(helper.make_node("Mul", ["M_11", "template_mask_11"], ["masked_M_11"]))
    
    # 9. Extract 9 blocks and sum
    block_names = []
    for r in range(3):
        for c in range(3):
            name = f"block_{r}_{c}"
            block_names.append(name)
            make_tensor(f"st_{r}_{c}", TensorProto.INT64, [2], [r*4, c*4])
            make_tensor(f"en_{r}_{c}", TensorProto.INT64, [2], [r*4+3, c*4+3])
            nodes.append(helper.make_node("Slice", ["masked_M_11", f"st_{r}_{c}", f"en_{r}_{c}", "axes_hw"], [name]))
    
    nodes.append(helper.make_node("Sum", block_names, ["template_3x3"]))
    
    # 10. Upsample to 12x12
    make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 4.0, 4.0])
    nodes.append(helper.make_node("Resize", ["template_3x3", "", "scales"], ["upsampled_12x12"], mode='nearest'))
    
    # 11. Slice to 11x11
    nodes.append(helper.make_node("Slice", ["upsampled_12x12", "starts_0_0", "ends_11_11", "axes_hw"], ["upsampled_11x11"]))
    
    # 12. Grid lines
    make_tensor("one_float", TensorProto.FLOAT, [1], [1.0])
    nodes.append(helper.make_node("Sub", ["one_float", "grid_mask_11"], ["not_grid_mask"]))
    nodes.append(helper.make_node("Mul", ["upsampled_11x11", "not_grid_mask"], ["upsampled_no_grid"]))
    
    make_tensor("zero_float", TensorProto.FLOAT, [1], [0.0])
    nodes.append(helper.make_node("Mul", ["grid_mask_11", "zero_float"], ["zero_c"]))
    
    grid_concat_inputs = ["zero_c"] * 5 + ["grid_mask_11"] + ["zero_c"] * 4
    nodes.append(helper.make_node("Concat", grid_concat_inputs, ["grid_ch_10"], axis=1))
    
    nodes.append(helper.make_node("Add", ["upsampled_no_grid", "grid_ch_10"], ["out_11"]))
    
    # 13. Background channel
    nodes.append(helper.make_node("Slice", ["out_11", "one_int", "ten_int", "axes_c"], ["out_1_9"]))
    nodes.append(helper.make_node("ReduceSum", ["out_1_9", "axes_c"], ["sum_1_9"], keepdims=1))
    nodes.append(helper.make_node("Sub", ["one_float", "sum_1_9"], ["bg_11"]))
    # Clip bg_11 to [0, 1] just in case
    nodes.append(helper.make_node("Clip", ["bg_11", "zero_float", "one_float"], ["bg_11_clipped"]))
    
    nodes.append(helper.make_node("Concat", ["bg_11_clipped", "out_1_9"], ["out_11_final"], axis=1))
    
    # 14. Pad to 30x30
    make_tensor("pads_30", TensorProto.INT64, [8], [0, 0, 0, 0, 0, 0, 19, 19])
    nodes.append(helper.make_node("Pad", ["out_11_final", "pads_30"], ["output_raw"]))
    
    # Apply valid grid mask just in case to strictly enforce zeroes outside 11x11 (except for channel 0 if needed? Wait!
    # convert_to_numpy leaves ALL channels as 0 outside the grid.
    # Pad with 0s does exactly this!)
    nodes.append(helper.make_node("Identity", ["output_raw"], ["output"]))
    
    graph = helper.make_graph(nodes, "task11_graph", [X], [Y], inits)
    model = helper.make_model(graph, producer_name='antigravity', ir_version=8, opset_imports=[helper.make_opsetid('', 13)])
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    onnx.save(model, "task011.onnx")
    print("Saved task011.onnx")


# Build the model (the function saves it internally, so we load the result)
build_model()
import glob
model = onnx.load("/project/repairs/task011.onnx")
