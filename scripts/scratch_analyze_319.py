import json

def analyze():
    with open('c:/Users/chand/OneDrive/Desktop/get_a_job/kaggle_competitions/The 2026 NeuroGolf Championship/data/task319.json') as f:
        data = json.load(f)
    
    for split in ['train', 'test', 'arc-gen']:
        print(f"--- {split} ---")
        for i, ex in enumerate(data[split]):
            inp = ex['input']
            out = ex['output']
            
            colors = set()
            for r in inp:
                for c in r:
                    colors.add(c)
                    
            match_c = -1
            for c in colors:
                rmin, rmax, cmin, cmax = 100, -1, 100, -1
                for r in range(len(inp)):
                    for col in range(len(inp[0])):
                        if inp[r][col] == c:
                            rmin = min(rmin, r)
                            rmax = max(rmax, r)
                            cmin = min(cmin, col)
                            cmax = max(cmax, col)
                if rmin <= rmax:
                    out_match = True
                    if len(out) == rmax-rmin+1 and len(out[0]) == cmax-cmin+1:
                        for r in range(len(out)):
                            for col in range(len(out[0])):
                                if inp[rmin+r][cmin+col] != out[r][col]:
                                    out_match = False
                        if out_match:
                            match_c = c
            
            print(f"Example {i} matches Color {match_c}")
            if match_c != -1:
                for row in out:
                    print("".join(str(x) if x == match_c else "." for x in row))
            print()
                        
analyze()
