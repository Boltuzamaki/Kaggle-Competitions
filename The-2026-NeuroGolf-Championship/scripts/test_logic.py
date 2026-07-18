import json
import numpy as np

def solve(task_file):
    with open(task_file, 'r') as f:
        data = json.load(f)
        
    for split in ['train', 'test']:
        for idx, ex in enumerate(data[split]):
            inp = np.array(ex['input'])
            colors = np.unique(inp)
            c = [x for x in colors if x != 0][0]
            
            pts = np.argwhere(inp == c)
            
            best_center = None
            best_P = None
            best_M = None
            best_score = -1
            
            for r0 in np.arange(-10, 20, 0.5):
                for c0 in np.arange(-10, 20, 0.5):
                    # compute D for all pts
                    D = np.maximum(np.abs(2*pts[:,0] - 2*r0), np.abs(2*pts[:,1] - 2*c0)).astype(int)
                    vals = np.unique(D)
                    
                    if len(vals) < 2:
                        continue
                        
                    # Check if vals form an arithmetic progression
                    diffs = np.diff(vals)
                    if len(np.unique(diffs)) == 1:
                        P = diffs[0]
                        M = vals[0] % P
                        # Count how many background pixels violate this inside the bounding box of pts?
                        # Or just pick the one with highest P ?
                        score = P
                        if score > best_score:
                            best_score = score
                            best_center = (r0, c0)
                            best_P = P
                            best_M = M
            print(f"{split} {idx}: color {c}, center {best_center}, P {best_P}, M {best_M}")

solve('c:/Users/chand/OneDrive/Desktop/get_a_job/kaggle_competitions/The 2026 NeuroGolf Championship/data/task392.json')
