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
from sklearn.tree import DecisionTreeRegressor


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

x_train, x_test, y_train, y_test = train_test_split(coordinates, stepvalues, test_size=0.2, random_state=500953158)



"""Model 1: Linear Regression"""
my_model1 = LinearRegression()
my_model1.fit(x_train, y_train)
y_pred_train1 = my_model1.predict(x_train)

for i in range(5):
    print("Predictions:", round(y_pred_train1[i], 2),
          "Actual values:", round(y_train[i], 2))
print("\n==============================\n")  
mae_train1 = mean_absolute_error(y_pred_train1, y_train)
print("Model 1 training MAE is: ", round(mae_train1, 2))




"""Model 2: Random Forest"""
my_model2 = RandomForestRegressor(n_estimators=10,
                                  
min_samples_leaf=2,
                                  random_state=500953158)
my_model2.fit(x_train, y_train)
y_pred_train2 = my_model2.predict(x_train)
mae_train2 = mean_absolute_error(y_pred_train2, y_train)
print("Model 2 training MAE is: ", round(mae_train2, 2))
print("\n==============================\n")

for i in range(5):
    print("Mode 1 Predictions: ",
          round(y_pred_train1[i], 2),
          "Mode 2 Predictions: ",
          round(y_pred_train2[i], 2),
          "Actual values:",
          round(y_train[i], 2))
    
print("\n==============================\n")

"""GridSearchCV"""
param_grid = {
    'n_estimators': [10, 30, 50, 100],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
my_model2_1 = RandomForestRegressor(random_state=500953158)
grid_search = GridSearchCV(my_model2_1, 
                           param_grid, 
                           cv=5, 
                           scoring='neg_mean_absolute_error', 
                           n_jobs=4)

grid_search.fit(x_train, y_train)
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)
print("\n==============================\n")
best_model3 = grid_search.best_estimator_

# ============================
# Adding Test Data Evaluation
# ============================

# Test Model 1 (Linear Regression) on the test data
y_pred_test1 = my_model1.predict(x_test)

# Calculate MAE on the test data for Model 1
mae_test1 = mean_absolute_error(y_test, y_pred_test1)


# Test Model 2 (Random Forest) on the test data
y_pred_test2 = my_model2.predict(x_test)

# Calculate MAE on the test data for Model 2
mae_test2 = mean_absolute_error(y_test, y_pred_test2)

# Optional: Compare the first 10 predictions for both models
for i in range(5):
    print("Test Data - Mode 1 Predictions:",
          round(y_pred_test1[i], 2),
          "Mode 2 Predictions:|",
          round(y_pred_test2[i], 2),
          "Actual values:",
          round(y_test[i], 2))
print("\n==============================\n")
print("Model 1 (Linear Regression) Test MAE is: ", round(mae_test1, 2))
print("Model 2 (Random Forest) Test MAE is: ", round(mae_test2, 2))
print("\n==============================\n")
y_pred_test = best_model3.predict(x_test)


# Calculate the test Mean Absolute Error (MAE)
mae_test = mean_absolute_error(y_test, y_pred_test)
print("Test MAE of the best model is:", round(mae_test, 2))


# # ============================
# # Plotting Test Predictions vs Actual Values
# # ============================

# plt.figure(figsize=(10, 6))

# # Plot actual values
# plt.plot(range(len(y_test)), y_test, label='Actual Values', color='b', marker='o')

# # Plot predictions from Model 1 (Linear Regression)
# plt.plot(range(len(y_pred_test1)), y_pred_test1, label='Model 1 Predictions (Linear Regression)', color='g', linestyle='--', marker='x')

# # Plot predictions from Model 2 (Random Forest)
# plt.plot(range(len(y_pred_test2)), y_pred_test2, label='Model 2 Predictions (Random Forest)', color='r', linestyle='--', marker='d')

# # Add title and labels
# plt.title('Test Predictions vs Actual Values')
# plt.xlabel('Data Point Index')
# plt.ylabel('Step Value')

# # Add a legend
# plt.legend()

# # Show the plot
# plt.show()