import numpy as np
from scipy.signal import convolve2d

def get_mask(inp, val):
    return (inp == val).astype(np.float32)

def solve(inp):
    mask_3 = get_mask(inp, 3)
    
    cross_k = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.float32)
    conv_D = convolve2d(mask_3, cross_k, mode='same')
    D = conv_D * mask_3
    
    tb_k = np.array([[0,1,0],[0,0,0],[0,1,0]], dtype=np.float32)
    lr_k = np.array([[0,0,0],[1,0,1],[0,0,0]], dtype=np.float32)
    
    tb_sum = convolve2d(mask_3, tb_k, mode='same')
    lr_sum = convolve2d(mask_3, lr_k, mode='same')
    
    tb_f = (tb_sum == 2.0).astype(np.float32)
    lr_f = (lr_sum == 2.0).astype(np.float32)
    is_collinear = tb_f + lr_f
    
    D_eq_2_f = (D == 2.0).astype(np.float32)
    collinear_0_f = (is_collinear == 0.0).astype(np.float32)
    
    is_corner = D_eq_2_f * collinear_0_f * mask_3
    
    M = mask_3.reshape(1, -1)
    M_col = mask_3.reshape(-1, 1)
    valid_pairs = M_col * M
    
    B = np.zeros((900, 900), dtype=np.float32)
    for r in range(30):
        for c in range(30):
            i = r * 30 + c
            B[i, i] = 1.0
            if r > 0: B[i, i - 30] = 1.0
            if r < 29: B[i, i + 30] = 1.0
            if c > 0: B[i, i - 1] = 1.0
            if c < 29: B[i, i + 1] = 1.0
            
    A0 = B * valid_pairs
    
    curr_A = A0
    for _ in range(10):
        curr_A = np.clip(curr_A @ curr_A, 0.0, 1.0)
        
    D_row = D.reshape(1, -1)
    A_times_D = curr_A * D_row
    max_degree = np.max(A_times_D, axis=1, keepdims=True)
    
    C_row = is_corner.reshape(1, -1)
    A_times_C = curr_A * C_row
    total_corners = np.sum(A_times_C, axis=1, keepdims=True)
    
    is_deg_3 = max_degree >= 3.0
    is_cor_1 = total_corners == 1.0
    
    color_no_deg_3 = np.where(is_cor_1, 1.0, 6.0)
    color_raw = np.where(is_deg_3, 2.0, color_no_deg_3)
    color_flat = color_raw * M_col
    
    color = color_flat.reshape(30, 30)
    return color

import json
with open('data/task364.json', 'r') as f:
    task = json.load(f)

for i, ex in enumerate(task['train']):
    inp = np.zeros((30, 30), dtype=np.float32)
    inp_ex = np.array(ex['input'])
    inp[:inp_ex.shape[0], :inp_ex.shape[1]] = inp_ex
    
    out = np.zeros((30, 30), dtype=np.float32)
    out_ex = np.array(ex['output'])
    out[:out_ex.shape[0], :out_ex.shape[1]] = out_ex
    
    pred = solve(inp)
    
    print(f"Train {i}: match={(pred == out)[inp_ex > 0].all()}")
    for r in range(inp_ex.shape[0]):
        for c in range(inp_ex.shape[1]):
            if inp_ex[r,c] == 3 and pred[r,c] != out[r,c]:
                print(f"  Mismatch at {r},{c}: pred={pred[r,c]}, out={out[r,c]}")
