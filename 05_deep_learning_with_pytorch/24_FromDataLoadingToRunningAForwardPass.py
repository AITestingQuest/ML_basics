import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import torch.nn as nn

dataframe = pd.read_csv("../00_resources/water_potability.csv")
print(dataframe.head())

features_df = dataframe[["ph","Sulfate","Conductivity","Organic_carbon"]]
print(features_df)
X = features_df.to_numpy()
target_df = dataframe[["Potability"]]
print(target_df)
y = target_df.to_numpy()

# Load the different columns into two PyTorch tensors
features = torch.tensor(dataframe[['ph', 'Sulfate', 'Conductivity', 'Organic_carbon']].to_numpy()).float()
target = torch.tensor(dataframe['Potability'].to_numpy()).float()

# Create a dataset from the two generated tensors
dataset = TensorDataset(features, target)

# Create a dataloader using the above dataset
dataloader = DataLoader(dataset, batch_size = 2, shuffle = True)
x, y = next(iter(dataloader))

# Create a model using the nn.Sequential API
model = nn.Sequential(nn.Linear(4,2),
                      nn.Linear(2,1))
output = model(features)
print(output)
