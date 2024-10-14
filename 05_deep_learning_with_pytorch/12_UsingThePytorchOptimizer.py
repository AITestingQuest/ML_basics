import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss

lr = 0.001

model = nn.Sequential(nn.Linear(16, 8),
                      nn.Linear(8, 4),
                      nn.Linear(4, 2))

pred_np = np.array(
    [[ 0.5011, -0.3752]]
)

target_np = np.array(
    [[1., 0.]]
)

pred = torch.tensor(pred_np)
target = torch.tensor(target_np)

criterion = CrossEntropyLoss()

# Create the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001)

loss = criterion(pred, target)
loss.backward()

# Update the model's parameters using the optimizer
optimizer.step()
