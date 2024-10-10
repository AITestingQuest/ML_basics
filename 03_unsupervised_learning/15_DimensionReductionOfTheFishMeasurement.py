import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../00_resources/fish.csv', header=None)
samples = df.loc[:,1:].values

# Create scaler: scaler
scaler = StandardScaler()
scaled_samples = scaler.fit_transform(samples)

# print(samples)
# print(scaled_samples)

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)


