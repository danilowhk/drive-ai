import torch
import torch.nn as nn
import torch.onnx

# Define model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

model = RNN(10, 20, 5)

# Generate input sample
x = torch.randn(1, 10)
hidden = torch.zeros(1, 20)

# Export model to ONNX
torch.onnx.export(model, 
                  (x, hidden),
                  "model_rnn.onnx",
                  opset_version=11)
