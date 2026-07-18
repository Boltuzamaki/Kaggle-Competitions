import onnxruntime as ort
import numpy as np
import json

sess = ort.InferenceSession('task366.onnx')
with open('data/task366.json') as f:
    d = json.load(f)

for ex_id, ex in enumerate(d['train']):
    grid = np.array(ex['input'])
    inp_oh = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            inp_oh[0, grid[r,c], r, c] = 1.0
            
    out = sess.run(None, {'input': inp_oh})[0][0]
    print(f'Train {ex_id}: r=15, c=0 -> {out[15,0]}')
