import torch.nn as nn
import torch
import numpy as np

weight_np = np.array(
    [
        [1.5317, -0.6213, -1.4732,  0.4708,  0.4061, -0.0167,  0.6464,  0.2901,
          0.8319],
        [ 0.5644,  0.9839, -1.4630, -0.1886,  1.6151,  1.3351,  0.6182,  1.1764,
         -0.2068]
    ]
)

bias_np = np.array(
    [-2.2455, -0.9470]
)

preds_np = np.array(
    [[-0.6612,  1.2721]]
)

target_np = np.array(
    [[1., 0.]]
)


weight = torch.tensor(weight_np, requires_grad=True)
bias = torch.tensor(bias_np, requires_grad=True)
preds = torch.tensor(preds_np)
target = torch.tensor(target_np)

print(weight)
print(bias)
print(preds)
print(target)

# Something should to be done to make the tensors with these attributes exactly
# In [1]:
# print(weight)
# tensor([[ 1.5317, -0.6213, -1.4732,  0.4708,  0.4061, -0.0167,  0.6464,  0.2901,
#           0.8319],
#         [ 0.5644,  0.9839, -1.4630, -0.1886,  1.6151,  1.3351,  0.6182,  1.1764,
#          -0.2068]], requires_grad=True)
# In [2]:
# print(bias)
# tensor([-2.2455, -0.9470], requires_grad=True)
# In [3]:
# print(preds)
# tensor([[-0.6612,  1.2721]], grad_fn=<AddBackward0>)
# In [4]:
# print(target)
# tensor([[1., 0.]])

criterion = nn.CrossEntropyLoss()

# Calculate the loss
loss = criterion(preds, target)

# Compute the gradients of the loss
loss.backward()

# Display gradients of the weight and bias tensors in order
print(weight.grad)
print(bias.grad)