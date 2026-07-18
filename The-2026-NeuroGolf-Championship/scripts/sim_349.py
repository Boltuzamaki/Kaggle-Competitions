import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class Task349(nn.Module):
    def forward(self, x):
        c9 = x[:, 9:10, :, :]
        
        out_1 = torch.zeros_like(c9)
        out_3 = torch.zeros_like(c9)
        out_9 = c9.clone()
        
        rem_c9 = c9.clone()
        
        for N in range(30, 0, -1):
            pad_br = N - 1
            padded_c9 = F.pad(rem_c9, (0, pad_br, 0, pad_br))
            det_kernel = torch.ones(1, 1, N, N, dtype=torch.float32, device=x.device)
            conv = F.conv2d(padded_c9, det_kernel)
            hit_N = (conv > N*N - 0.5).float()
            
            cover_kernel = torch.ones(1, 1, N, N, dtype=torch.float32, device=x.device)
            cover_N = F.conv_transpose2d(hit_N, cover_kernel)
            cover_N = cover_N[:, :, :30, :30]
            cover_N = (cover_N > 0.5).float()
            
            rem_c9 = rem_c9 * (1.0 - cover_N)
            
            T = N // 2
            if T > 0:
                S = N + T
                K = N + 2 * T
                
                b_kernel = torch.ones(1, 1, K, K, dtype=torch.float32, device=x.device)
                b_out = F.conv_transpose2d(hit_N, b_kernel)
                b_out_shifted = b_out[:, :, T : T + 30, T : T + 30]
                out_3 = torch.max(out_3, (b_out_shifted > 0.5).float())
                
                s_kernel = torch.zeros(1, 1, S + 30, N, dtype=torch.float32, device=x.device)
                s_kernel[:, :, S : S + 30, :] = 1.0
                s_out = F.conv_transpose2d(hit_N, s_kernel)
                s_out_cropped = s_out[:, :, :30, :30]
                out_1 = torch.max(out_1, (s_out_cropped > 0.5).float())
        
        final_1 = torch.clamp(out_1 - out_3 - out_9, 0.0, 1.0)
        final_3 = torch.clamp(out_3 - out_9, 0.0, 1.0)
        final_9 = out_9
        
        final_0 = torch.clamp(1.0 - final_1 - final_3 - final_9, 0.0, 1.0)
        
        channels = []
        for i in range(10):
            if i == 0:
                channels.append(final_0)
            elif i == 1:
                channels.append(final_1)
            elif i == 3:
                channels.append(final_3)
            elif i == 9:
                channels.append(final_9)
            else:
                channels.append(torch.zeros_like(final_0))
                
        output = torch.cat(channels, dim=1)
        return output

with open("data/task349.json", "r") as f:
    data = json.load(f)

model = Task349()
model.eval()

def grid_to_tensor(grid):
    # grid is list of lists
    H = len(grid)
    W = len(grid[0])
    t = torch.zeros(1, 10, 30, 30)
    for r in range(H):
        for c in range(W):
            val = grid[r][c]
            t[0, val, r, c] = 1.0
    # The rest is background 0
    t[0, 0, H:, :] = 1.0
    t[0, 0, :, W:] = 1.0
    return t, H, W

def tensor_to_grid(t, H, W):
    grid = []
    # t is [1, 10, 30, 30]
    preds = torch.argmax(t, dim=1)[0] # [30, 30]
    for r in range(H):
        row = []
        for c in range(W):
            row.append(preds[r, c].item())
        grid.append(row)
    return grid

all_passed = True
for idx, example in enumerate(data['train']):
    t, H, W = grid_to_tensor(example['input'])
    with torch.no_grad():
        out_t = model(t)
    pred_grid = tensor_to_grid(out_t, H, W)
    if pred_grid == example['output']:
        print(f"Train {idx} passed!")
    else:
        print(f"Train {idx} FAILED!")
        all_passed = False

for idx, example in enumerate(data['test']):
    t, H, W = grid_to_tensor(example['input'])
    with torch.no_grad():
        out_t = model(t)
    pred_grid = tensor_to_grid(out_t, H, W)
    if pred_grid == example['output']:
        print(f"Test {idx} passed!")
    else:
        print(f"Test {idx} FAILED!")
        all_passed = False

if all_passed:
    print("ALL PASSED!")
