import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

df = pd.read_csv('../00_resources/eurovision-2016.csv')
samples = df.iloc[:,2:7].values[:42]
country_names = df.iloc[:, 1].values[:42]
# Calculate the linkage: mergings
mergings = linkage(samples, method='single')

# Plot the dendrogram
dendrogram(mergings,
           labels=country_names,
           leaf_rotation=90,
           leaf_font_size=6)
plt.show()
