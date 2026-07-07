import os, sys, json
import onnx
from onnx import numpy_helper

def parse_onnx_to_graph(onnx_path):
    if not os.path.exists(onnx_path):
        return None
    
    model = onnx.load(onnx_path)
    graph = model.graph
    
    nodes = []
    edges = []
    
    x_pos = 100
    y_pos = 100
    
    # Track tensor producer to create edges
    # producer_map[tensor_name] = (node_id, port_idx)
    producer_map = {}
    
    # 1. Input Node
    for i, inp in enumerate(graph.input):
        node_id = inp.name
        nodes.append({
            "id": node_id,
            "op": "Input",
            "attrs": {},
            "position": {"x": 50, "y": y_pos + i*150}
        })
        producer_map[inp.name] = (node_id, 0)
        
    init_map = {}
    for init in graph.initializer:
        init_map[init.name] = numpy_helper.to_array(init)

    # 3. Nodes (process nodes first so we can consume initializers)
    consumed_inits = set()
    for idx, n in enumerate(graph.node):
        node_id = n.output[0] if n.output else f"n_{idx}"
        op = n.op_type
        
        attrs = {}
        for attr in n.attribute:
            if attr.type == onnx.AttributeProto.INTS:
                attrs[attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.INT:
                attrs[attr.name] = int(attr.i)
            elif attr.type == onnx.AttributeProto.FLOAT:
                attrs[attr.name] = float(attr.f)
            elif attr.type == onnx.AttributeProto.FLOATS:
                attrs[attr.name] = list(attr.floats)
            elif attr.type == onnx.AttributeProto.STRING:
                attrs[attr.name] = attr.s.decode('utf-8')

        inputs = list(n.input)
        
        def consume_init(inp_idx, attr_name):
            if len(inputs) > inp_idx and inputs[inp_idx] in init_map:
                name = inputs[inp_idx]
                val = init_map[name]
                if val.size == 1:
                    attrs[attr_name] = val.item()
                else:
                    attrs[attr_name] = val.tolist()
                consumed_inits.add(name)

        if op == "Slice":
            consume_init(1, "starts"); consume_init(2, "ends")
            consume_init(3, "axes"); consume_init(4, "steps")
            inputs = inputs[:1]
        elif op == "Pad":
            consume_init(1, "pads"); consume_init(2, "value")
            inputs = inputs[:1]
        elif op == "Tile":
            consume_init(1, "repeats")
            inputs = inputs[:1]
        elif op == "Resize":
            consume_init(3, "sizes") # inputs 1, 2 are roi, scales (usually empty)
            if len(inputs) > 1 and inputs[1] in init_map: consumed_inits.add(inputs[1])
            if len(inputs) > 2 and inputs[2] in init_map: consumed_inits.add(inputs[2])
            inputs = inputs[:1]
        elif op == "Conv":
            consume_init(1, "weight")
            inputs = inputs[:1]
                
        nodes.append({
            "id": node_id,
            "op": op,
            "attrs": attrs,
            "position": {"x": x_pos, "y": y_pos + idx * 80}
        })
        
        # Edges
        for port_idx, inp_name in enumerate(inputs):
            # We defer edge creation slightly because Constant nodes (initializers) aren't created yet
            edges.append({
                "from": inp_name, # Temporary, will map to producer_id
                "fromPort": 0,
                "to": node_id,
                "toPort": port_idx,
                "toPortStr": f"in{port_idx}"
            })
                
        for port_idx, out_name in enumerate(n.output):
            producer_map[out_name] = (node_id, port_idx)
            
    # 2. Initializers (Constants) - only those not consumed
    for name, val in init_map.items():
        if name in consumed_inits:
            continue
        if val.size < 1000:
            val_list = val.tolist()
        else:
            val_list = []
            
        nodes.append({
            "id": name,
            "op": "Constant",
            "attrs": {"shape": list(val.shape), "value": val_list},
            "position": {"x": 50, "y": y_pos + 400}
        })
        producer_map[name] = (name, 0)
        y_pos += 100
        
    x_pos += 200
    
    # 4. Output Node
    for i, out in enumerate(graph.output):
        node_id = out.name
        nodes.append({
            "id": node_id,
            "op": "Output",
            "attrs": {},
            "position": {"x": x_pos, "y": 200 + i*150}
        })
        
        if out.name in producer_map:
            src_id, src_port = producer_map[out.name]
            edges.append({
                "from": out.name, # Temporary
                "fromPort": 0,
                "to": node_id,
                "toPort": 0,
                "toPortStr": "input"
            })
            
    # Fixup edges
    final_edges = []
    for e in edges:
        inp_name = e["from"]
        if inp_name in producer_map:
            src_id, src_port = producer_map[inp_name]
            e["from"] = src_id
            e["fromPort"] = src_port
            final_edges.append(e)
        else:
            pass # Unresolved input (maybe an external input we didn't model? Shouldn't happen)

    return {"nodes": nodes, "edges": final_edges}
            
    return {"nodes": nodes, "edges": edges}

if __name__ == "__main__":
    g = parse_onnx_to_graph("../repairs/task005.onnx")
    print(json.dumps(g, indent=2))
