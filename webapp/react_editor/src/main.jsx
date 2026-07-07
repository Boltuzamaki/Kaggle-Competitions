import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { createRoot } from 'react-dom/client';
import ReactFlow, { Background, Controls, Handle, Position, addEdge, useNodesState, useEdgesState } from 'reactflow';
import 'reactflow/dist/style.css';
import './style.css';

const baseOps = ["Input", "Output", "Constant", "RowIndex", "ColIndex", "Cast", "Identity", "Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual", "Not", "And", "Or", "Xor", "Add", "Sub", "Mul", "Div", "Mod", "Min", "Max", "Sum", "Relu", "Abs", "Neg", "Floor", "Clip", "Sign", "Sqrt", "ReduceSum", "ReduceMax", "ReduceMin", "ArgMax", "Where", "Slice", "Pad", "Concat", "Transpose", "Tile", "Resize", "Conv"];
const inputSlots = {
  Cast: ["input"], Identity: ["input"], Not: ["input"], ReduceSum: ["input"], ArgMax: ["input"],
  Slice: ["input"], Pad: ["input"], Transpose: ["input"], Tile: ["input"], Resize: ["input"], Conv: ["input"], Output: ["input"],
  Equal: ["a", "b"], Greater: ["a", "b"], Less: ["a", "b"], GreaterOrEqual: ["a", "b"], LessOrEqual: ["a", "b"],
  And: ["a", "b"], Or: ["a", "b"], Xor: ["a", "b"], Add: ["a", "b"], Sub: ["a", "b"], Mul: ["a", "b"], Div: ["a", "b"], Mod: ["a", "b"],
  Min: ["a", "b"], Max: ["a", "b"], Sum: ["a", "b"], Relu: ["input"], Abs: ["input"], Neg: ["input"], Floor: ["input"], Clip: ["input"],
  Sign: ["input"], Sqrt: ["input"], ReduceMax: ["input"], ReduceMin: ["input"], Where: ["condition", "true", "false"], Concat: ["a", "b"]
};

function defaultAttrs(opType) {
  if (opType === "Cast") return { to: "1" };
  if (opType === "ReduceSum" || opType === "ReduceMax" || opType === "ReduceMin") return { axes: [2, 3], keepdims: 1 };
  if (opType === "ArgMax") return { axis: 1, keepdims: 1 };
  if (opType === "Clip") return { min: 0, max: 9 };
  if (opType === "Slice") return { starts: [0, 0, 0, 0], ends: [1, 1, 30, 30], axes: [0, 1, 2, 3], steps: [1, 1, 1, 1] };
  if (opType === "Pad") return { pads: [0, 0, 0, 0, 0, 0, 0, 0], value: 0 };
  if (opType === "Concat") return { axis: 1 };
  if (opType === "Transpose") return { perm: [0, 1, 3, 2] };
  if (opType === "Tile") return { repeats: [1, 1, 1, 1] };
  if (opType === "Resize") return { sizes: [1, 1, 30, 30], mode: "nearest" };
  if (opType === "Conv") return { weight_shape: [1, 1, 3, 3], weights: [1, 1, 1, 1, 1, 1, 1, 1, 1], pads: [1, 1, 1, 1], strides: [1, 1] };
  return {};
}

function slotsForOp(opType) {
  if (inputSlots[opType]) return inputSlots[opType];
  if (opType === "Input" || opType === "Constant") return [];
  return ["in0"];
}

function OpNode({ data, selected }) {
  const slots = data.inputSlots || inputSlots[data.opType] || [];
  return (
    <div className={`node ${selected ? "nodeSelected" : ""}`} style={{background:'rgba(20,26,38,.92)', border: selected ? '1px solid #22d3ee' : '1px solid rgba(255,255,255,.09)', padding:'8px', borderRadius:'10px', minWidth:'110px', boxShadow: selected ? '0 0 0 1px #22d3ee' : '0 4px 12px rgba(0,0,0,0.4)', color:'#e8ecf4', fontSize:'11px'}}>
      {slots.map((slot, index) => (
        <Handle key={slot} id={slot} type="target" position={Position.Left} style={{ top: `${((index + 1) / (slots.length + 1)) * 100}%`, background:'#4b5468', border:'2px solid #0b0f18', width:'11px', height:'11px' }} />
      ))}
      {slots.length > 0 && <div style={{fontSize:'9px', color:'#8b93a7', marginBottom:'4px'}}>{slots.join(" / ")}</div>}
      <div style={{fontWeight:'bold', marginBottom:'4px'}}>{data.opType}</div>
      <div style={{color:'#8b93a7'}}>{data.label}</div>
      <Handle type="source" position={Position.Right} style={{background:'#4b5468', border:'2px solid #0b0f18', width:'11px', height:'11px'}} />
    </div>
  );
}

const nodeTypes = { op: OpNode };

function ReactEditor() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [opSel, setOpSel] = useState("Constant");
  const [isMaximized, setIsMaximized] = useState(false);

  useEffect(() => {
    if (window.INITIAL_GRAPH && window.INITIAL_GRAPH.nodes) {
      const g = window.INITIAL_GRAPH;
      const rfNodes = g.nodes.map((n, i) => ({
        id: n.id,
        type: 'op',
        position: n.position || { x: 100 + (i * 150), y: 100 },
        data: { label: n.id, opType: n.op, attrs: n.attrs || {}, inputSlots: slotsForOp(n.op) }
      }));
      const rfEdges = (g.edges || []).map((e, i) => ({
        id: `e_${i}`,
        source: e.from,
        target: e.to,
        targetHandle: e.toPortStr || (slotsForOp(rfNodes.find(n => n.id === e.to)?.data?.opType)[e.toPort] || 'in0')
      }));
      setNodes(rfNodes);
      setEdges(rfEdges);
    } else {
      setNodes([
        { id: "input_1", type: "op", position: { x: 70, y: 80 }, data: { label: "input_1", opType: "Input", attrs:{}, inputSlots:[] } },
        { id: "output_1", type: "op", position: { x: 310, y: 80 }, data: { label: "output_1", opType: "Output", attrs:{}, inputSlots:['input'] } },
      ]);
    }
  }, []);

  const onConnect = useCallback((params) => setEdges((eds) => addEdge(params, eds)), [setEdges]);

  const addNode = () => {
    const opType = opSel;
    const count = nodes.filter(n => n.data.opType === opType).length + 1;
    const id = `${opType.toLowerCase()}_${count}`;
    const data = { label: id, opType, attrs: defaultAttrs(opType), inputSlots: slotsForOp(opType) };
    if (opType === "Constant") data.attrs = { value: "0" };
    setNodes(nds => [...nds, { id, type: 'op', position: { x: 150, y: 150 }, data }]);
  };

  const deleteSelected = () => {
    setNodes(nds => nds.filter(n => !n.selected));
    setEdges(eds => eds.filter(e => !e.selected));
    setSelectedNode(null);
  };

  useEffect(() => {
    const sel = nodes.find(n => n.selected);
    setSelectedNode(sel || null);
  }, [nodes]);

  const updateAttrs = (val) => {
    if (!selectedNode) return;
    try {
      const parsed = JSON.parse(val);
      setNodes(nds => nds.map(n => n.id === selectedNode.id ? { ...n, data: { ...n.data, attrs: parsed } } : n));
    } catch(e) { }
  };

  const compileGraph = () => {
    const backendNodes = nodes.map(n => ({
      id: n.id,
      op: n.data.opType,
      attrs: n.data.attrs || {},
      position: n.position
    }));
    const backendEdges = edges.map(e => {
      const targetNode = nodes.find(n => n.id === e.target);
      const slots = slotsForOp(targetNode?.data?.opType);
      const portIdx = Math.max(0, slots.indexOf(e.targetHandle));
      return {
        from: e.source,
        fromPort: 0,
        to: e.target,
        toPort: portIdx,
        toPortStr: e.targetHandle
      };
    });

    const payload = { graph: { nodes: backendNodes, edges: backendEdges } };
    
    const T = window.T || 1; 
    document.getElementById('result').innerHTML = '⏳ compiling visual graph...';
    fetch(`/api/compile_graph/${T}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(j => {
      if (window.showResult) window.showResult(j, true);
    })
    .catch(e => {
      document.getElementById('result').innerHTML = '<span class="bad">request failed: ' + e + '</span>';
    });
  };

  return (
    <div style={isMaximized ? {position:'fixed', top:0, left:0, width:'100vw', height:'100vh', zIndex:9999, background:'#05060c', padding:'20px', display:'flex', flexDirection:'column', gap:'10px'} : {display:'flex', flexDirection:'column', gap:'10px', height:'100%'}}>
      <div style={{display:'flex', gap:'8px', alignItems:'center'}}>
        <select value={opSel} onChange={e => setOpSel(e.target.value)} style={{background:'rgba(255,255,255,.045)', color:'#e8ecf4', padding:'6px 10px', borderRadius:'8px', border:'1px solid rgba(255,255,255,.09)'}}>
          {baseOps.map(op => <option key={op} value={op}>{op}</option>)}
        </select>
        <button onClick={addNode} className="btn ghost">+ add node</button>
        <button onClick={deleteSelected} className="btn ghost">🗑 delete selected</button>
        <button onClick={() => setIsMaximized(!isMaximized)} className="btn ghost">{isMaximized ? '⤓ minimize' : '⤢ maximize'}</button>
        <button onClick={compileGraph} className="btn run" style={{marginLeft:'auto'}}>▶ Compile &amp; Verify</button>
      </div>
      <div style={{display:'flex', gap:'10px', flex: isMaximized ? 1 : 'none', height: isMaximized ? 'auto' : '450px'}}>
        <div style={{flex:1, background:'#02040a', borderRadius:'12px', border:'1px solid rgba(255,255,255,.09)', overflow:'hidden'}}>
          <ReactFlow nodes={nodes} edges={edges} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect} nodeTypes={nodeTypes} fitView>
            <Background color="#fff" gap={24} size={1} variant="dots" style={{opacity:0.04}} />
            <Controls style={{background:'rgba(20,26,38,.92)', fill:'#fff', border:'1px solid rgba(255,255,255,.09)', display:'flex', flexDirection:'column'}} />
          </ReactFlow>
        </div>
        <div style={{width:'260px', background:'rgba(255,255,255,.045)', borderRadius:'12px', padding:'12px', border:'1px solid rgba(255,255,255,.09)'}}>
          <div style={{fontSize:'11px', color:'#8b93a7', marginBottom:'8px'}}>Node Attributes (JSON)</div>
          <textarea 
            style={{width:'100%', height:'350px', background:'#02040a', color:'#c9f2e6', padding:'10px', borderRadius:'8px', border:'1px solid rgba(255,255,255,.09)', fontFamily:'monospace', fontSize:'12px', resize:'none'}}
            disabled={!selectedNode}
            value={selectedNode ? JSON.stringify(selectedNode.data.attrs || {}, null, 2) : ''}
            onChange={e => updateAttrs(e.target.value)}
          />
        </div>
      </div>
    </div>
  );
}

const root = createRoot(document.getElementById('react-editor-root'));
root.render(<ReactEditor />);
