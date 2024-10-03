# Import Normalizer
from shutil import move
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
import csv

with open("../00_resources/company-stock-movements-2010-2015-incl.csv") as f:
    companies = [row.split(",")[0] for row in f]

with open("../00_resources/company-stock-movements-2010-2015-incl.csv") as f:
    source_list = list(csv.reader(f,delimiter=","))

companies = companies[1:]
source_list = source_list[1:]

np_array = np.array(source_list)
np_array = np_array[:,1:]

movements = np_array.astype(float)

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)

# Predict the cluster labels: labels
labels = pipeline.predict(movements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print(df.sort_values('labels'))