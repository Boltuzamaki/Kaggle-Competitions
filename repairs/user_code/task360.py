# Source: predicted/test_onnx_task360.py — ONNX graph construction code
# Verified model: repairs/task360.onnx
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    F32 = TensorProto.FLOAT
    I64 = TensorProto.INT64
    input_info = helper.make_tensor_value_info('input', F32, [1, 10, 30, 30])
    output_info = helper.make_tensor_value_info('output', F32, [1, 10, 30, 30])
    
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node(
            'Constant', [], [name],
            value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())
        ))
        
    make_const('axes_3', [3])
    make_const('axes_1', [1])
    make_const('c0', [0])
    make_const('c4', [4])
    make_const('c5', [5])
    make_const('c9', [9])
    
    nodes.append(helper.make_node(
        'Slice',
        ['input', 'c0', 'c4', 'axes_3'],
        ['left_half']
    ))
    
    nodes.append(helper.make_node(
        'Slice',
        ['input', 'c5', 'c9', 'axes_3'],
        ['right_half']
    ))
    
    make_const('step_minus1', [-1])
    make_const('c3', [3])
    make_const('c_minus100', [-100])
    nodes.append(helper.make_node(
        'Slice',
        ['right_half', 'c3', 'c_minus100', 'axes_3', 'step_minus1'],
        ['right_flipped']
    ))
    
    # We want to combine left and right_flipped using Max.
    # But since background channel is 0, Max(background(1), color(0)) = 1, and Max(background(0), color(1)) = 1.
    # We should slice channels 1 to 9
    make_const('c1', [1])
    make_const('c10', [10])
    
    nodes.append(helper.make_node(
        'Slice',
        ['left_half', 'c1', 'c10', 'axes_1'],
        ['left_color']
    ))
    
    nodes.append(helper.make_node(
        'Slice',
        ['right_flipped', 'c1', 'c10', 'axes_1'],
        ['right_color']
    ))
    
    nodes.append(helper.make_node(
        'Max',
        ['left_color', 'right_color'],
        ['combined_color']
    ))
    
    # Reconstruct channel 0: 1 - sum(channels 1..9)
    # Actually, we can just pad with 0 at channel 0, since eval_task uses argmax, and if combined_color has 0 everywhere, argmax returns 0 (which is background).
    # Wait, if we pad channel 0 with 0, and combined_color has 0 everywhere, argmax returns 0.
    # If combined_color has 1 at channel c, argmax returns c (because c > 0).
    # So we just need to prepend a slice of zeros for channel 0!
    
    make_const('axes_1_tensor', [1])
    nodes.append(helper.make_node(
        'ReduceSum',
        ['combined_color', 'axes_1_tensor'],
        ['sum_colors'],
        keepdims=1
    ))
    
    make_const('one_f32', [1.0], dtype=F32)
    nodes.append(helper.make_node(
        'Sub',
        ['one_f32', 'sum_colors'],
        ['chan0']
    ))
    
    nodes.append(helper.make_node(
        'Concat',
        ['chan0', 'combined_color'],
        ['concat_out'],
        axis=1
    ))
    
    nodes.append(helper.make_node(
        'ReduceMax',
        ['left_half'],
        ['mask'],
        axes=[1],
        keepdims=1
    ))
    
    nodes.append(helper.make_node(
        'Mul',
        ['concat_out', 'mask'],
        ['masked_out']
    ))
    
    make_const('pads', [0, 0, 0, 0, 0, 0, 0, 26])
    
    nodes.append(helper.make_node(
        'Pad',
        ['masked_out', 'pads'],
        ['output']
    ))
    
    output_info = helper.make_tensor_value_info('output', F32, ['batch', 10, 30, 30])
    
    graph = helper.make_graph(
        nodes,
        'task360_graph',
        [input_info],
        [output_info]
    )
    
    model = helper.make_model(graph, producer_name='task360_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task360.onnx')


# Build the model (the function saves it internally, so we load the result)
create_onnx_model()
import glob
model = onnx.load("/project/repairs/task360.onnx")
