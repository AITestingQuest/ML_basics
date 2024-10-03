import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


df = pd.read_csv('../00_resources/Grains/seeds.csv', header=None)
df[7] = df[7].map({1:'Kama wheat', 2:'Rosa wheat', 3:'Canadian wheat'})
df.head()

samples = df.iloc[:, :-1].values
varieties = df.iloc[:, -1].values

mergings = linkage(samples, method='complete')

dendrogram(mergings,
            labels=varieties,
            leaf_rotation=90,
            leaf_font_size=6)
plt.show()