# Source: predicted/test_onnx_task044.py — ONNX graph construction code
# Verified model: repairs/task044.onnx
import onnx
from onnx import helper, TensorProto
import numpy as np

X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10, 30, 30])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10, 30, 30])

nodes = []
initializers = []

def make_init(name, data_type, dims, vals):
    if not any(i.name == name for i in initializers):
        initializers.append(helper.make_tensor(name, data_type, dims, vals))

# Common initializers
make_init("axes_1", TensorProto.INT64, [1], [1])
make_init("zero_float", TensorProto.FLOAT, [1, 1, 1, 1], [0.0])
make_init("one_float", TensorProto.FLOAT, [1, 1, 1, 1], [1.0])
make_init("nine_float", TensorProto.FLOAT, [1, 1, 1, 1], [29.0])
make_init("pads_9", TensorProto.INT64, [8], [0, 0, 29, 29, 0, 0, 29, 29])
make_init("pad_val_1", TensorProto.FLOAT, [], [1.0])

grid_19 = list(range(59))
make_init("grid_19_row", TensorProto.FLOAT, [1, 1, 59, 1], grid_19)
make_init("grid_19_col", TensorProto.FLOAT, [1, 1, 1, 59], grid_19)

grid_10 = list(range(30))
make_init("r_row_10", TensorProto.FLOAT, [1, 1, 30, 1], grid_10)
make_init("r_col_10", TensorProto.FLOAT, [1, 1, 1, 30], grid_10)

nodes.append(helper.make_node("Sub", ["r_row_10", "r_col_10"], ["r_diff"]))

# input_0 and input_5
make_init("starts_c0", TensorProto.INT64, [1], [0])
make_init("ends_c1", TensorProto.INT64, [1], [1])
nodes.append(helper.make_node("Slice", ["input", "starts_c0", "ends_c1", "axes_1"], ["in_0"]))

make_init("starts_c5", TensorProto.INT64, [1], [5])
make_init("ends_c6", TensorProto.INT64, [1], [6])
nodes.append(helper.make_node("Slice", ["input", "starts_c5", "ends_c6", "axes_1"], ["in_5"]))

# padded_not_0, padded_not_5
nodes.append(helper.make_node("Sub", ["one_float", "in_0"], ["not_0"]))
nodes.append(helper.make_node("Sub", ["one_float", "in_5"], ["not_5"]))
nodes.append(helper.make_node("Pad", ["not_0", "pads_9", "pad_val_1"], ["padded_not_0"]))
nodes.append(helper.make_node("Pad", ["not_5", "pads_9", "pad_val_1"], ["padded_not_5"]))

make_init("pads_1", TensorProto.INT64, [8], [0, 0, 1, 1, 0, 0, 1, 1])
make_init("zero_float_scalar", TensorProto.FLOAT, [], [0.0])
make_init("pads_10", TensorProto.INT64, [8], [0, 0, 30, 30, 0, 0, 30, 30])
nodes.append(helper.make_node("Pad", ["not_5", "pads_10", "pad_val_1"], ["padded_not_5_extra"]))

make_init("axes_23", TensorProto.INT64, [2], [2, 3])
make_init("axes_1_reduce", TensorProto.INT64, [1], [1])

translated_channels = []

for c in range(1, 10):
    if c == 5:
        translated_channels.append("in_5")
        continue
    
    c_str = str(c)
    make_init(f"starts_{c}", TensorProto.INT64, [1], [c])
    make_init(f"ends_{c}", TensorProto.INT64, [1], [c+1])
    nodes.append(helper.make_node("Slice", ["input", f"starts_{c}", f"ends_{c}", "axes_1"], [f"in_{c}"]))
    
    # dilated_c
    # dilated_c padded to avoid truncation
    nodes.append(helper.make_node("Pad", [f"in_{c}", "pads_1", "zero_float_scalar"], [f"in_c_padded_{c}"]))
    nodes.append(helper.make_node("MaxPool", [f"in_c_padded_{c}"], [f"dilated_c_padded_{c}"], kernel_shape=[3, 3], pads=[1,1,1,1]))
    nodes.append(helper.make_node("Sub", [f"dilated_c_padded_{c}", f"in_c_padded_{c}"], [f"bound_c_padded_{c}"]))
    
    # score
    nodes.append(helper.make_node("Conv", ["padded_not_0", f"in_{c}"], [f"score_0_{c}"], pads=[0,0,0,0]))
    nodes.append(helper.make_node("Conv", ["padded_not_5_extra", f"bound_c_padded_{c}"], [f"score_5_{c}"], pads=[0,0,0,0]))
    nodes.append(helper.make_node("Add", [f"score_0_{c}", f"score_5_{c}"], [f"score_{c}"]))
    
    nodes.append(helper.make_node("Equal", [f"score_{c}", "zero_float"], [f"score_match_{c}"]))
    
    nodes.append(helper.make_node("ReduceSum", [f"in_{c}", "axes_23"], [f"piece_size_{c}"], keepdims=1))
    nodes.append(helper.make_node("Greater", [f"piece_size_{c}", "zero_float"], [f"has_piece_{c}"]))
    
    nodes.append(helper.make_node("And", [f"score_match_{c}", f"has_piece_{c}"], [f"is_match_{c}"]))
    nodes.append(helper.make_node("Cast", [f"is_match_{c}"], [f"is_match_f_{c}"], to=TensorProto.FLOAT))
    
    nodes.append(helper.make_node("ReduceMax", [f"is_match_f_{c}"], [f"is_match_any_{c}"], axes=[2, 3], keepdims=1))
    nodes.append(helper.make_node("Mul", ["nine_float", f"is_match_any_{c}"], [f"offset_29_{c}"]))
    
    nodes.append(helper.make_node("Mul", [f"is_match_f_{c}", "grid_19_row"], [f"dy_shifted_{c}_mul"]))
    nodes.append(helper.make_node("ReduceSum", [f"dy_shifted_{c}_mul", "axes_23"], [f"dy_shifted_{c}"], keepdims=1))
    nodes.append(helper.make_node("Sub", [f"dy_shifted_{c}", f"offset_29_{c}"], [f"dy_{c}"]))
    
    nodes.append(helper.make_node("Mul", [f"is_match_f_{c}", "grid_19_col"], [f"dx_shifted_{c}_mul"]))
    nodes.append(helper.make_node("ReduceSum", [f"dx_shifted_{c}_mul", "axes_23"], [f"dx_shifted_{c}"], keepdims=1))
    nodes.append(helper.make_node("Sub", [f"dx_shifted_{c}", f"offset_29_{c}"], [f"dx_{c}"]))
    nodes.append(helper.make_node("Neg", [f"dx_{c}"], [f"neg_dx_{c}"]))
    
    nodes.append(helper.make_node("Equal", ["r_diff", f"dy_{c}"], [f"P_r_bool_{c}"]))
    nodes.append(helper.make_node("Equal", ["r_diff", f"neg_dx_{c}"], [f"P_c_bool_{c}"]))
    nodes.append(helper.make_node("Cast", [f"P_r_bool_{c}"], [f"P_r_{c}"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Cast", [f"P_c_bool_{c}"], [f"P_c_{c}"], to=TensorProto.FLOAT))
    
    nodes.append(helper.make_node("MatMul", [f"P_r_{c}", f"in_{c}"], [f"trans_tmp_{c}"]))
    nodes.append(helper.make_node("MatMul", [f"trans_tmp_{c}", f"P_c_{c}"], [f"translated_{c}"]))
    translated_channels.append(f"translated_{c}")

# output_unmasked
nodes.append(helper.make_node("Concat", ["in_0"] + translated_channels, ["output_unmasked"], axis=1))

# Since some translated pieces overwrite background, we need to recompute background!
# Or we can just let argmax handle it?
# In process_tasks.py, Argmax creates the final grid.
# If background is 1 and translated piece is 1, Argmax will pick background (0) because it's first!
# We MUST zero out the background where any piece is present!
nodes.append(helper.make_node("ReduceMax", ["output_unmasked"], ["any_color"], axes=[1], keepdims=1))
# Wait, any_color includes in_0 (which is 1 on bg). So any_color is always 1!
# We should only consider colors 1..9 for masking out background.
nodes.append(helper.make_node("Concat", translated_channels, ["color_channels"], axis=1))
nodes.append(helper.make_node("ReduceMax", ["color_channels"], ["color_mask"], axes=[1], keepdims=1))
nodes.append(helper.make_node("Sub", ["one_float", "color_mask"], ["bg_mask_full"]))
nodes.append(helper.make_node("ReduceMax", ["input"], ["valid_mask"], axes=[1], keepdims=1))
nodes.append(helper.make_node("Mul", ["bg_mask_full", "valid_mask"], ["new_in_0"]))

nodes.append(helper.make_node("Concat", ["new_in_0"] + translated_channels, ["final_output"], axis=1))

nodes.append(helper.make_node("Identity", ["final_output"], ["output"]))

graph = helper.make_graph(nodes, 'task044', [X], [Y], initializer=initializers)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
onnx.checker.check_model(model)
model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
for val in model.graph.value_info:
    if val.name in ["in_0"] + translated_channels + ["output_unmasked", "final_output"]:
        print(val.name, [d.dim_value for d in val.type.tensor_type.shape.dim])
print("Model saved.")
onnx.save(model, 'task044.onnx')


# Build the model (the function saves it internally, so we load the result)
make_init()
import glob
model = onnx.load("/project/repairs/task044.onnx")
