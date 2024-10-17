import torch.nn as nn
import torch

def calculate_capacity(model):
  total = 0
  for p in model.parameters():
    total += p.numel()
  return total

n_features = 8
n_classes = 2

input_tensor = torch.Tensor([[3, 4, 6, 2, 3, 6, 8, 9]])

# Create a neural network with less than 120 parameters
model = nn.Sequential(nn.Linear(n_features,4),
                      nn.Linear(4,8),
                      nn.Linear(8,4),
                      nn.Linear(4,n_classes))
output = model(input_tensor)

print(calculate_capacity(model))