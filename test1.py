# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:17:45 2024

@author: anant
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Load your CSV file
csv_file = 'Project_1_Data.csv'
df = pd.read_csv(csv_file)

#plotting the coordinates vs step function
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
sc = ax1.scatter(df['X'], df['Y'], df['Z'], c=df['Step'], cmap='viridis')
plt.colorbar(sc, ax=ax1, label='Step')
ax1.set_xlabel('X')
ax1.set_ylabel('Step')
ax1.set_zlabel('Z')
ax1.set_title('3D Scatter Plot of X, Y, Z with Step')
plt.show()


# Calculate the correlation matrix for X, Y, Z, and Step columns
correlation_matrix = df[['X', 'Y', 'Z', 'Step']].corr()
plt.figure()
# Plot the heatmap for the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, square=True, linewidths=0.5)

# Add a title
plt.title('Correlation Matrix of X, Y, Z, and Step', fontsize=15)



coordinates = df[['X', 'Y', 'Z']].values
stepvalues = df['Step'].values

x_train, x_test, y_train, y_test=train_test_split(coordinates,stepvalues, test_size=0.2, random_state=500953158)

#1 Linear Regression
my_model1 = LinearRegression()
my_model1.fit(x_train, y_train)
y_pred_train1 = my_model1.predict(x_train)

for i in range(5):
    print("Predictions:", y_pred_train1[i], "Actual values:", y_train[i])

mae_train1 = mean_absolute_error(y_pred_train1, y_train)
print("Model 1 training MAE is: ", round(mae_train1,2))


"""Training Second Model"""
my_model2 = RandomForestRegressor(n_estimators=10, random_state=500953158)
my_model2.fit(x_train, y_train)
y_pred_train2 = my_model2.predict(x_train)
mae_train2 = mean_absolute_error(y_pred_train2, y_train)
print("Model 2 training MAE is: ", round(mae_train2,2))

for i in range(10):
    print("Mode 1 Predictions:",
          round(y_pred_train1[i],2),
          "Mode 2 Predictions:|",
          round(y_pred_train2[i],2),
          "Actual values:",
          round(y_train[i],2))

# """GridSearchCV"""
# param_grid = {
#     'n_estimators': [10, 30, 50],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2']
# }
# my_model3 = RandomForestRegressor(random_state=500953158)
# grid_search = GridSearchCV(my_model3, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=1)
# grid_search.fit(x_train, y_train)
# best_params = grid_search.best_params_
# print("Best Hyperparameters:", best_params)
# best_model3 = grid_search.best_estimator_