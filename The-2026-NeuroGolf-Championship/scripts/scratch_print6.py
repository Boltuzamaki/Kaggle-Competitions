import json
import numpy as np

def analyze():
    with open('c:/Users/chand/OneDrive/Desktop/get_a_job/kaggle_competitions/The 2026 NeuroGolf Championship/data/task319.json') as f:
        data = json.load(f)
    
    inp = np.array(data['train'][1]['input'])
    bg = inp[0,0]
    
    mask = (inp == 6)
    rows, cols = np.where(mask)
    crop = inp[rows.min():rows.max()+1, cols.min():cols.max()+1]
    
    print("Task 1 Shape 6:")
    for r in crop:
        print("".join("6" if c == 6 else "." for c in r))

analyze()
