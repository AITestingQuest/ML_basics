import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster


df = pd.read_csv('../00_resources/Grains/seeds.csv', header=None)
df[7] = df[7].map({1:'Kama wheat', 2:'Rosa wheat', 3:'Canadian wheat'})
df.head()

samples = df.iloc[:, :-1].values
varieties = df.iloc[:, -1].values

mergings = linkage(samples, method='complete')

# Use fcluster to extract labels: labels
labels = fcluster(mergings, 6, criterion='distance')

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)