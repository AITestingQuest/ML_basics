import torch.nn as nn
import torch


input_tensor_sigmoid = torch.tensor([[0.8]])
input_tensor_softmax = torch.tensor([[1.0, -6.0, 2.5, -0.3, 1.2, 0.8]])

# Create a sigmoid function and apply it on input_tensor
sigmoid = nn.Sigmoid()
probability_sigmoid = sigmoid(input_tensor_sigmoid)
print(probability_sigmoid)

softmax = nn.Softmax(1)
probability_softmax = softmax(input_tensor_softmax)
print(probability_softmax)