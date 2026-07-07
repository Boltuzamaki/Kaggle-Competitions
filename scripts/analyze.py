import json, numpy as np

with open('data/task313.json') as f: data = json.load(f)

# Test hypothesis: output = input shifted by (+1, +1) with wraparound
# Or: output[r][c] = input[r-1][c-1] (circular)
for split in ['train', 'test', 'arc-gen']:
    if split not in data: continue
    for i, eg in enumerate(data[split][:5]):
        inp = np.array(eg['input'])
        out = np.array(eg['output'])
        H, W = inp.shape
        # Try shift (+1, +1) with cyclic
        pred = np.roll(np.roll(inp, 1, axis=0), 1, axis=1)
        match = np.all(pred == out)
        if not match:
            # Try other shifts
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    pred2 = np.roll(np.roll(inp, dr, axis=0), dc, axis=1)
                    if np.all(pred2 == out):
                        print(f'{split}[{i}]: shift ({dr},{dc})')
                        break
        else:
            print(f'{split}[{i}]: shift (+1,+1) OK')
