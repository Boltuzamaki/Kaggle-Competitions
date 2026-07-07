import json

def analyze():
    with open('c:/Users/chand/OneDrive/Desktop/get_a_job/kaggle_competitions/The 2026 NeuroGolf Championship/data/task319.json') as f:
        data = json.load(f)
    
    for i, ex in enumerate(data['train']):
        inp = ex['input']
        print(f"Task {i} Input:")
        for r in inp:
            print("".join(str(c) if c != inp[0][0] else "." for c in r))
        print("\n")

analyze()
