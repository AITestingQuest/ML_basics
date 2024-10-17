import torch.nn as nn

model = nn.Sequential(nn.Linear(16, 4),
                      nn.Linear(4, 2),
                      nn.Linear(2, 1))

total = 0

# Calculate the number of parameters in the model
for parameter in model.parameters():
  total += parameter.numel()

print(total)