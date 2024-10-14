import torch
import torch.nn as nn

input_tensor = torch.Tensor([[2, 3, 6, 7, 9, 3, 2, 1]])

# Implement a small neural network with exactly two linear layers
model = nn.Sequential(nn.Linear(8, 4),
                      nn.Linear(4, 1)
                     )

output = model(input_tensor)
print(output)