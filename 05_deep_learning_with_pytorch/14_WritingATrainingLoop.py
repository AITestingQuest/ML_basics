# Writing a training loop
# In scikit-learn, the whole training loop is contained in the .fit() method. In PyTorch, however, you implement the loop manually. While this provides control over loop's content, it requires a custom implementation.
#
# You will write a training loop every time you train a deep learning model with PyTorch, which you'll practice in this exercise. The show_results() function provided will display some sample ground truth and the model predictions.
# 
# The package imports provided are: pandas as pd, torch, torch.nn as nn, torch.optim as optim, as well as DataLoader and TensorDataset from torch.utils.data.
# 
# The following variables have been created: dataloader, containing the dataloader; model, containing the neural network; criterion, containing the loss function, nn.MSELoss(); optimizer, containing the SGD optimizer; and num_epochs, containing the number of epochs.
# 
# Loop over the number of epochs and the dataloader
# for i in range(num_epochs):
#   for data in dataloader:
#     # Set the gradients to zero
#     optimizer.zero_grad()
#     # Run a forward pass
#     feature, target = data
#     prediction = model(feature)    
#     # Calculate the loss
#     loss = criterion(prediction, target)    
#     # Compute the gradients
#     loss.backward()
#     # Update the model's parameters
#     optimizer.step()
# show_results(model, dataloader)