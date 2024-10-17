import numpy as np
import torch
import torch.nn as nn

# Create a leaky relu function in PyTorch
leaky_relu_pytorch = nn.LeakyReLU(0.05)

x = torch.tensor(-3.0)
# Call the above function on the tensor x
output = leaky_relu_pytorch(x)
print(output)