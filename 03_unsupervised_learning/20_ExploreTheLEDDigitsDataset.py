# Import pyplot
from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv('../00_resources/lcd-digits.csv', header=None)
samples = df.values

# Select the 0th row: digit
digit = samples[0]

# Print digit
print(digit)

# Reshape digit to a 13x8 array: bitmap
bitmap = digit.reshape(13,8)

# Print bitmap
print(bitmap)

# Use plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()