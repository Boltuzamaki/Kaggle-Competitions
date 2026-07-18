import torch
import torch.nn as nn
import onnx

class DummyModel(nn.Module):
    def forward(self, x):
        return x * 2

model = DummyModel()
dummy_input = torch.randn(1, 10, 30, 30)
torch.onnx.export(model, dummy_input, "test_dummy.onnx", opset_version=15, input_names=['input'], output_names=['output'])
print("PyTorch exported ONNX successfully.")
