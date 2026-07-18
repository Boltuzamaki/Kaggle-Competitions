import json
import numpy as np
from scipy.ndimage import label

def count_holes(mask):
    # pad with 0 (background)
    padded = np.pad(mask, 1, constant_values=False)
    lbl, num = label(~padded)
    # the outside background is 1 component, so holes = num - 1
    return num - 1

def analyze():
    with open('c:/Users/chand/OneDrive/Desktop/get_a_job/kaggle_competitions/The 2026 NeuroGolf Championship/data/task319.json') as f:
        data = json.load(f)
    
    for i, ex in enumerate(data['train']):
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        bg = inp[0,0]
        
        colors = set(inp.flatten())
        
        match_c = -1
        c_props = {}
        for c in colors:
            if c == bg: continue
            mask = (inp == c)
            if not np.any(mask): continue
            
            rows, cols = np.where(mask)
            rmin, rmax = rows.min(), rows.max()
            cmin, cmax = cols.min(), cols.max()
            
            crop = mask[rmin:rmax+1, cmin:cmax+1]
            crop_full = inp[rmin:rmax+1, cmin:cmax+1]
            
            if crop_full.shape == out.shape and np.all(crop_full == out):
                match_c = c
                
            holes = count_holes(crop)
            c_props[c] = holes
            
        print(f"Task {i} Matches {match_c}")
        for c, h in c_props.items():
            mark = "*" if c == match_c else " "
            print(f"{mark} Color {c}: {h} holes")
        print()

analyze()
