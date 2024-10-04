# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:17:45 2024

@author: anant
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


# Load your CSV file
csv_file = 'Project_1_Data.csv'
df = pd.read_csv(csv_file)

# Calculate the correlation matrix for X, Y, Z, and Step columns
correlation_matrix = df[['X', 'Y', 'Z', 'Step']].corr()
plt.figure()
# Plot the heatmap for the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, square=True, linewidths=0.5)

# Add a title
plt.title('Correlation Matrix of X, Y, Z, and Step', fontsize=15)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
sc = ax1.scatter(df['X'], df['Y'], df['Z'], c=df['Step'], cmap='viridis')
plt.colorbar(sc, ax=ax1, label='Step')
ax1.set_xlabel('X')
ax1.set_ylabel('Step')
ax1.set_zlabel('Z')
ax1.set_title('3D Scatter Plot of X, Y, Z with Step')
plt.show()

coordinates = df[['X', 'Y', 'Z']].values
stepvalues = df['Step'].values

x_train, x_test, y_train, y_test=train_test_split(coordinates,stepvalues, test_size=0.2, random_state=500953158)

