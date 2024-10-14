# Import PyTorch
import torch

list_a = [1, 2, 3, 4]

# Create a tensor from list_a
tensor_a = torch.tensor(list_a)

print(tensor_a)

# Display the tensor device
print(tensor_a.device)

# Display the tensor data type
print(tensor_a.dtype)