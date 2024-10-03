# Perform the necessary imports
from matplotlib.streamplot import StreamplotSet
import scipy as sp
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

source_csv = "../00_resources/fish.csv"
with open(source_csv) as f:
    species = [row.split(",")[0] for row in f]
np_array = np.genfromtxt(source_csv, delimiter=",")
samples = np_array[:,1:6]

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans)

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({"labels": labels, "species": species})

# Create crosstab: ct
ct = pd.crosstab(df["labels"], df["species"])

# Display ct
print(ct)