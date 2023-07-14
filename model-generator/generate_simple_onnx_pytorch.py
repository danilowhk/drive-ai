import torch
import torch.nn as nn
import torch.onnx

# Define model
model = nn.Sequential(
    nn.Linear(in_features=20, out_features=10),
    nn.ReLU(),
    nn.Linear(in_features=10, out_features=5)
)

# Generate input sample
x = torch.randn(1, 20)

# Export model to ONNX
torch.onnx.export(model, 
                  x,
                  "model.onnx",
                  opset_version=11)