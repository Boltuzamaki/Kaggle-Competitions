import json
import numpy as np

def analyze():
    with open('c:/Users/chand/OneDrive/Desktop/get_a_job/kaggle_competitions/The 2026 NeuroGolf Championship/data/task319.json') as f:
        data = json.load(f)
    
    for i, ex in enumerate(data['train']):
        inp = np.array(ex['input'])
        bg = inp[0,0]
        colors = [c for c in set(inp.flatten()) if c != bg]
        
        crops = {}
        for c in colors:
            mask = (inp == c)
            rows, cols = np.where(mask)
            crop = mask[rows.min():rows.max()+1, cols.min():cols.max()+1]
            crops[c] = crop
            
        print(f"Task {i}")
        for c1 in colors:
            for c2 in colors:
                if c1 >= c2: continue
                # check if crop1 is same as crop2 under D8
                cr1 = crops[c1]
                cr2 = crops[c2]
                match = False
                for k in range(4):
                    rot = np.rot90(cr1, k)
                    if rot.shape == cr2.shape and np.all(rot == cr2):
                        match = True
                    flp = np.fliplr(rot)
                    if flp.shape == cr2.shape and np.all(flp == cr2):
                        match = True
                if match:
                    print(f"  Shape {c1} is isomorphic to {c2}")

analyze()
