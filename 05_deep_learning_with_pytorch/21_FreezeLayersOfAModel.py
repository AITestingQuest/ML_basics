import torch.nn as nn
import torch

model = nn.Sequential(nn.Linear(8,16),
                      nn.Linear(16,32),
                      nn.Linear(32,10))

print(model)

for name, param in model.named_parameters():
    print(name)
    print(param)

for name, param in model.named_parameters():    
  
    # Check if the parameters belong to the first layer
    if name == '0.bias' or name == '0.weight':
      
        # Freeze the parameters
        param.requires_grad = False
  
    # Check if the parameters belong to the second layer
    if name == '1.bias' or name == '1.weight':
      
        # Freeze the parameters
        param.requires_grad = False
