import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

df = pd.read_csv('../00_resources/Grains/seeds.csv', header=None)
samples = df.iloc[:, :-1].values
variety_numbers = df.iloc[:, -1].values

# Create a TSNE instance: model
model = TSNE(learning_rate=200)

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(samples)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs,ys,c=variety_numbers)
plt.show()
