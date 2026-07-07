import json
import numpy as np
from scipy.signal import convolve2d

d = json.load(open('data/task018.json'))
kernel = np.ones((3, 3))

def apply_sym(grid, s):
    # grid is [C, H, W]
    if s == 0: return grid
    if s == 1: return np.rot90(grid, 1, axes=(1, 2))
    if s == 2: return np.rot90(grid, 2, axes=(1, 2))
    if s == 3: return np.rot90(grid, 3, axes=(1, 2))
    if s == 4: return np.flip(grid, axis=1)
    if s == 5: return np.flip(np.rot90(grid, 1, axes=(1, 2)), axis=1)
    if s == 6: return np.flip(np.rot90(grid, 2, axes=(1, 2)), axis=1)
    if s == 7: return np.flip(np.rot90(grid, 3, axes=(1, 2)), axis=1)

def solve(grid):
    grid = np.array(grid)
    H, W = grid.shape
    
    # 1. Classify colors
    main_colors = set()
    anchor_colors = set()
    for k in range(1, 10):
        mask = (grid == k).astype(int)
        if mask.sum() == 0: continue
        conv = convolve2d(mask, kernel, mode='same')
        if (conv[mask == 1] > 1).any():
            main_colors.add(k)
        else:
            anchor_colors.add(k)
            
    # 2. Separate anchors
    main_mask = np.isin(grid, list(main_colors)).astype(int)
    main_adj = convolve2d(main_mask, kernel, mode='same') > 0
    
    M = np.zeros((10, H, W))
    T = np.zeros((10, H, W))
    for k in anchor_colors:
        mask = (grid == k)
        M[k] = mask & main_adj
        T[k] = mask & ~main_adj
        
    MainGrid = np.zeros((10, H, W))
    for k in main_colors:
        MainGrid[k] = (grid == k)
        
    # 3. Voting with Local Overlap
    # O[s, dr, dc, r_t, c_t] = Match
    O = np.zeros((8, 2*max(H,W)+1, 2*max(H,W)+1, H, W))
    
    for s in range(8):
        M_s = apply_sym(M, s)
        H_m, W_m = M_s.shape[1], M_s.shape[2]
        for dr in range(-H_m, H+1):
            for dc in range(-W_m, W+1):
                # match for each target pixel
                match = np.zeros((H, W))
                for r in range(H_m):
                    for c in range(W_m):
                        r_t = r + dr
                        c_t = c + dc
                        if 0 <= r_t < H and 0 <= c_t < W:
                            match[r_t, c_t] = np.sum(M_s[:, r, c] * T[:, r_t, c_t])
                
                # Local overlap using 15x15 kernel
                # We can just sum within a large window
                # Actually, 15x15 is enough for ARC grids (max 30x30).
                # Models are rarely larger than 15x15.
                local_overlap = convolve2d(match, np.ones((15, 15)), mode='same')
                O[s, dr+H_m, dc+W_m] = local_overlap
                
    # 4. Map Target Anchors to best (s, dr, dc)
    best_O = np.zeros((H, W))
    for s in range(8):
        M_s = apply_sym(M, s)
        H_m, W_m = M_s.shape[1], M_s.shape[2]
        for dr in range(-H_m, H+1):
            for dc in range(-W_m, W+1):
                # Which target anchors does this cover?
                for r in range(H_m):
                    for c in range(W_m):
                        r_t = r + dr
                        c_t = c + dc
                        if 0 <= r_t < H and 0 <= c_t < W:
                            if np.sum(M_s[:, r, c] * T[:, r_t, c_t]) > 0:
                                overlap = O[s, dr+H_m, dc+W_m, r_t, c_t]
                                if overlap > best_O[r_t, c_t]:
                                    best_O[r_t, c_t] = overlap
                                        
    # 5. Transform Main
    OutGrid = np.zeros((10, H, W))
    for s in range(8):
        Main_s = apply_sym(MainGrid, s)
        M_s = apply_sym(M, s)
        H_m, W_m = M_s.shape[1], M_s.shape[2]
        for dr in range(-H_m, H+1):
            for dc in range(-W_m, W+1):
                
                # Check if this transformation is the BEST for AT LEAST ONE target anchor it covers
                is_best = False
                for r in range(H_m):
                    for c in range(W_m):
                        r_t = r + dr
                        c_t = c + dc
                        if 0 <= r_t < H and 0 <= c_t < W:
                            if np.sum(M_s[:, r, c] * T[:, r_t, c_t]) > 0:
                                overlap = O[s, dr+H_m, dc+W_m, r_t, c_t]
                                if overlap == best_O[r_t, c_t]:
                                    is_best = True
                                        
                if is_best:
                    # Apply this transformation to Main_s
                    for r in range(H_m):
                        for c in range(W_m):
                            r_t = r + dr
                            c_t = c + dc
                            if 0 <= r_t < H and 0 <= c_t < W:
                                OutGrid[:, r_t, c_t] = np.maximum(OutGrid[:, r_t, c_t], Main_s[:, r, c])
                                
    # Add back target anchors
    OutGrid = np.maximum(OutGrid, T)
    
    # Convert to 2D
    out_2d = np.zeros((H, W), dtype=int)
    for k in range(10):
        out_2d[OutGrid[k] > 0] = k
    return out_2d

for i, ex in enumerate(d['train'] + d['test']):
    out = solve(ex['input'])
    expected = np.array(ex['output'])
    if np.array_equal(out, expected):
        print(f'Example {i}: PASS')
    else:
        print(f'Example {i}: FAIL')

