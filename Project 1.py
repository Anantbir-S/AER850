import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load your CSV file
csv_file = 'Project_1_Data.csv'
df = pd.read_csv(csv_file)

# Standardize the data for X, Y, Z, and Step columns
scaler = StandardScaler()
df[['X', 'Y', 'Z', 'Step']] = scaler.fit_transform(df[['X', 'Y', 'Z', 'Step']])

# 1. 3D scatter plot (X, Y, Z vs Step)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
sc = ax1.scatter(df['X'], df['Y'], df['Z'], c=df['Step'], cmap='viridis')
plt.colorbar(sc, ax=ax1, label='Step')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Scatter Plot of X, Y, Z with Step')
plt.show()

# Calculate the correlation matrix
correlation_matrix = df[['X', 'Y', 'Z', 'Step']].corr()

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Matrix of X, Y, Z, and Step')
plt.show()

# # 3. Box plot (Distribution of Coordinates)
# fig3, ax3 = plt.subplots()
# sns.boxplot(data=df[['X', 'Y', 'Z']], ax=ax3)
# ax3.set_title('Box Plot of Coordinates')
# plt.show()

# # 4. Pair plot (2D relations) - Separate figure
# sns.pairplot(df[['X', 'Y', 'Z', 'Step']], hue='Step')
# plt.show()
