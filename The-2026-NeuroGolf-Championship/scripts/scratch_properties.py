import json
import numpy as np
from scipy.ndimage import label

def get_symmetry(mask):
    h = np.allclose(mask, np.flipud(mask))
    v = np.allclose(mask, np.fliplr(mask))
    p = np.allclose(mask, np.flipud(np.fliplr(mask)))
    return h, v, p

def count_components(mask):
    lbl, num = label(mask)
    return num

def analyze():
    with open('c:/Users/chand/OneDrive/Desktop/get_a_job/kaggle_competitions/The 2026 NeuroGolf Championship/data/task319.json') as f:
        data = json.load(f)
    
    for i, ex in enumerate(data['train'] + data['test']):
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
                
            comp_c = count_components(crop)
            comp_bg = count_components(~crop)
            sym_h, sym_v, sym_p = get_symmetry(crop)
            
            c_props[c] = {
                'area': crop.size,
                'pixels': crop.sum(),
                'cc_fg': comp_c,
                'cc_bg': comp_bg,
                'sym_h': sym_h,
                'sym_v': sym_v,
                'sym_p': sym_p,
                'shape': crop.shape
            }
            
        print(f"Task {i} Matches {match_c}")
        for c, props in c_props.items():
            mark = "*" if c == match_c else " "
            print(f"{mark} Color {c}: {props}")
        print()

analyze()
