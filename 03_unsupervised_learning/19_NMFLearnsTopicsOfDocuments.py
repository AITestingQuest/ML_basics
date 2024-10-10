from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.decomposition import NMF

df = pd.read_csv('../00_resources/Wikipedia articles/wikipedia-vectors.csv', index_col=0)
articles = csr_matrix(df.transpose())
titles = list(df.columns)

# Create an NMF instance: model
model = NMF(n_components=6)

# Fit the model to articles
model.fit(articles)

words = []
with open('../00_resources/Wikipedia articles/wikipedia-vocabulary-utf8.txt') as f:
    words = f.read().splitlines()

# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_, columns=words )

# Print the shape of the DataFrame
print(components_df.shape)

# Select row 3: component
component = components_df.iloc[3]

# Print result of nlargest
print(component.nlargest())



