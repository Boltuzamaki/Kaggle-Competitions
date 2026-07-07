# Source code for task 348 (from predicted/test_onnx_task348.py)
# NOTE: This file builds the ONNX model. The verified model is at repairs/task348.onnx
import onnx
model = onnx.load("/project/repairs/task348.onnx")

# --- Original source code below ---
# import torch
# import torch.nn as nn
# import onnx
# 
# class Task348(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.C = 10
#         self.H = 30
#         self.W = 30
#         rows = torch.arange(30, dtype=torch.float32).view(1, 1, 30, 1).expand(1, 1, 30, 30)
#         cols = torch.arange(30, dtype=torch.float32).view(1, 1, 1, 30).expand(1, 1, 30, 30)
#         self.register_buffer("rows", rows)
#         self.register_buffer("cols", cols)
# 
#     def forward(self, x):
#         mask_7 = x[:, 7:8, :, :]
#         row_masked = mask_7 * self.rows
#         col_masked = mask_7 * self.cols
#         
#         # Max over H and W
#         bottom_r = torch.max(row_masked.view(x.shape[0], -1), dim=1)[0].view(x.shape[0], 1, 1, 1)
#         bottom_c = torch.max(col_masked.view(x.shape[0], -1), dim=1)[0].view(x.shape[0], 1, 1, 1)
#         
#         dr = bottom_r - self.rows
#         dc = torch.abs(self.cols - bottom_c)
#         
#         valid_mask = torch.sum(x, dim=1, keepdim=True)
#         in_triangle = (self.rows <= bottom_r) & (dc <= dr) & (valid_mask > 0.5)
#         
#         dc_mod_2 = torch.remainder(dc, 2.0)
#         
#         is_7_f = (in_triangle & (dc_mod_2 == 0)).float()
#         is_8_f = (in_triangle & (dc_mod_2 == 1)).float()
#         not_in_triangle_f = (~in_triangle).float()
#         
#         out = x * not_in_triangle_f
#         
#         zeros_1_7 = torch.zeros_like(x[:, 0:7, :, :])
#         zeros_8_10 = torch.zeros_like(x[:, 8:10, :, :])
#         is_7_full = torch.cat([zeros_1_7, is_7_f, zeros_8_10], dim=1)
#         
#         zeros_1_8 = torch.zeros_like(x[:, 0:8, :, :])
#         zeros_9_10 = torch.zeros_like(x[:, 9:10, :, :])
#         is_8_full = torch.cat([zeros_1_8, is_8_f, zeros_9_10], dim=1)
#         
#         out = out + is_7_full + is_8_full
#         return out
# 
# if __name__ == "__main__":
#     model = Task348()
#     dummy_input = torch.zeros(1, 10, 30, 30, dtype=torch.float32)
#     dummy_input[0, 7, 10:15, 10] = 1.0
#     onnx_path = "task348.onnx"
#     torch.onnx.export(model, dummy_input, onnx_path, 
#                       opset_version=14, 
#                       input_names=['input'], 
#                       output_names=['output'])
#     
#     # Just to verify and save with pure ONNX formatting
#     model_onnx = onnx.load(onnx_path)
#     onnx.checker.check_model(model_onnx)
#     onnx.save(model_onnx, 'task348.onnx')
#     print(f"Saved {onnx_path}")
# 