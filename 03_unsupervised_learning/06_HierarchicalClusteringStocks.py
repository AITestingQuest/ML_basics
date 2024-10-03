import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, dendrogram

with open("../00_resources/company-stock-movements-2010-2015-incl.csv") as f:
    companies = [row.split(",")[0] for row in f]

with open("../00_resources/company-stock-movements-2010-2015-incl.csv") as f:
    source_list = list(csv.reader(f,delimiter=","))

companies = companies[1:]
source_list = source_list[1:]

np_array = np.array(source_list)
np_array = np_array[:,1:]

movements = np_array.astype(float)

#print(movements)
#print(companies)

# Normalize the movements: normalized_movements
normalized_movements = normalize(movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements, method='complete')

# Plot the dendrogram
dendrogram(mergings,
           labels=companies,
           leaf_rotation=90,
           leaf_font_size=6)
plt.show()