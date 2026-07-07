import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import sys
import traceback

def remove_value_infos(graph):
    while len(graph.value_info) > 0:
        graph.value_info.pop()

def replace_shape_with_const(graph, input_name):
    # Replaces Shape(tensor) with Constant([1, 30, 30]) for the input tensor
    for node in graph.node:
        if node.op_type == 'Shape' and len(node.input) == 1 and node.input[0] == input_name:
            node.op_type = 'Constant'
            del node.input[:]
            val = np.array([1, 30, 30], dtype=np.int64)
            node.attribute.append(helper.make_attribute('value', helper.make_tensor(node.name+'_v', TensorProto.INT64, val.shape, val.tolist())))

def universal_patch(model_path, out_path):
    model = onnx.load(model_path)
    graph = model.graph
    
    F = TensorProto.FLOAT
    I64 = TensorProto.INT64
    
    remove_value_infos(graph)
    inits = {init.name: init for init in graph.initializer}
    
    old_input = graph.input[0]
    old_output = graph.output[0]
    
    # We must ensure final input is 'input' and final output is 'output'.
    # If the old graph used these names, we must rename them inside the graph.
    old_input_inner_name = old_input.name + "_inner" if old_input.name == 'input' else old_input.name
    old_output_inner_name = old_output.name + "_inner" if old_output.name == 'output' else old_output.name
    
    new_nodes = list(graph.node)
    
    # Rename occurrences
    for node in new_nodes:
        for i in range(len(node.input)):
            if node.input[i] == old_input.name:
                node.input[i] = old_input_inner_name
            if node.input[i] == old_output.name:
                node.input[i] = old_output_inner_name
        for i in range(len(node.output)):
            if node.output[i] == old_input.name:
                node.output[i] = old_input_inner_name
            if node.output[i] == old_output.name:
                node.output[i] = old_output_inner_name
                
    # 1. Front wrapper if needed
    if old_input.type.tensor_type.elem_type != F or len(old_input.type.tensor_type.shape.dim) != 4:
        new_in = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
        argmax_node = helper.make_node('ArgMax', ['input'], [old_input_inner_name], axis=1, keepdims=0) 
        new_nodes.insert(0, argmax_node)
        graph.input.remove(old_input)
        graph.input.insert(0, new_in)
    else:
        # If it was already F [1, 10, 30, 30], we just rename old_input_inner_name back to 'input' in nodes
        for node in new_nodes:
            for i in range(len(node.input)):
                if node.input[i] == old_input_inner_name:
                    node.input[i] = 'input'
        # And ensure static shape
        graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
        graph.input[0].type.tensor_type.shape.dim[1].dim_value = 10
        graph.input[0].type.tensor_type.shape.dim[2].dim_value = 30
        graph.input[0].type.tensor_type.shape.dim[3].dim_value = 30
        
    # 2. Back wrapper if needed
    if old_output.type.tensor_type.elem_type != F or len(old_output.type.tensor_type.shape.dim) != 4:
        depth_name = 'patch_depth'
        vals_name = 'patch_vals'
        if depth_name not in inits:
            graph.initializer.append(numpy_helper.from_array(np.array([10], dtype=np.int64), name=depth_name))
        if vals_name not in inits:
            graph.initializer.append(numpy_helper.from_array(np.array([0.0, 1.0], dtype=np.float32), name=vals_name))
            
        oh_node = helper.make_node('OneHot', [old_output_inner_name, depth_name, vals_name], ['oh_out'], axis=-1)
        trans_node = helper.make_node('Transpose', ['oh_out'], ['trans_out'], perm=[0, 3, 1, 2])
        
        new_nodes.append(oh_node)
        new_nodes.append(trans_node)
        raw_float_out = 'trans_out'
    else:
        raw_float_out = old_output_inner_name

    # 3. Padding mask
    rm_node = helper.make_node('ReduceMax', ['input'], ['presence_mask'], axes=[1], keepdims=1)
    mul_node = helper.make_node('Mul', [raw_float_out, 'presence_mask'], ['output'])
    
    new_nodes.append(rm_node)
    new_nodes.append(mul_node)
    
    new_out = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])
    graph.output.remove(old_output)
    graph.output.insert(0, new_out)
    
    del graph.node[:]
    graph.node.extend(new_nodes)
    
    # Fold Shape(old_input_inner_name)
    replace_shape_with_const(graph, old_input_inner_name)
    
    new_model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])
    
    try:
        inferred = onnx.shape_inference.infer_shapes(new_model, strict_mode=True, data_prop=True)
        onnx.save(inferred, out_path)
    except Exception as e:
        print("Shape inference failed:", e)
        # Save anyway so we can debug
        onnx.save(new_model, out_path)

if __name__ == '__main__':
    universal_patch(sys.argv[1], sys.argv[2])
