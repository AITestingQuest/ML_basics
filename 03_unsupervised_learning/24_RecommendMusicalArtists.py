from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

df = pd.read_csv('../00_resources/Musical artists/scrobbler-small-sample.csv')
artists1 = df.sort_values(['artist_offset', 'user_offset'], ascending=[True, True])
row_ind = np.array(artists1['artist_offset'])
col_ind = np.array(artists1['user_offset'])
data1 = np.array(artists1['playcount'])
artists = coo_matrix((data1, (row_ind, col_ind)))

df = pd.read_csv('../00_resources/Musical artists/artists.csv', header=None)
artist_names = df.values.reshape(111).tolist()

# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()

# Create an NMF model: nmf
nmf = NMF(n_components=20)

# Create a Normalizer: normalizer
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler, nmf, normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)

# Create a DataFrame: df
df_norm = pd.DataFrame(norm_features, index=artist_names)

# Select row of 'Bruce Springsteen': artist
artist = df_norm.loc['Bruce Springsteen']

# Compute cosine similarities: similarities
similarities = df_norm.dot(artist)

# Display those with highest cosine similarity
print(similarities.nlargest())
