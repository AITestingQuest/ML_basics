import torch
import numpy as np

array_a = np.array([[1, 1, 1],[2, 3, 4],[4, 5, 6]])
array_b = np.array([[7, 5, 4],[2, 2, 8],[6, 3, 8]])

print(array_a)
print(array_b)

# Create two tensors from the arrays
tensor_a = torch.tensor(array_a)
tensor_b = torch.tensor(array_b)

# Subtract tensor_b from tensor_a 
tensor_c = tensor_a - tensor_b
print(tensor_c)

# Multiply each element of tensor_a with each element of tensor_b
tensor_d = tensor_a * tensor_b
print(tensor_d)

# Add tensor_c to tensor_d
tensor_e = tensor_c + tensor_d
print(tensor_e)