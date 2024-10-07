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
from sklearn.model_selection import cross_val_score


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

# Import cross_val_score if not already imported


# Define a scoring function
scoring = 'neg_mean_absolute_error'

# Cross-validation for Linear Regression before training
cv_scores1 = cross_val_score(LinearRegression(), coordinates, stepvalues, cv=5, scoring=scoring)
cv_mae1 = -cv_scores1.mean()  # Negate because scoring returns negative MAE
print(f"Cross-validation MAE for Linear Regression: {round(cv_mae1, 2)}")

# Cross-validation for Random Forest before training
cv_scores2 = cross_val_score(RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_split=10, 
                                                   min_samples_leaf=5, max_features='sqrt', random_state=500953158), 
                             coordinates, stepvalues, cv=5, scoring=scoring)
cv_mae2 = -cv_scores2.mean()
print(f"Cross-validation MAE for Random Forest: {round(cv_mae2, 2)}")

# Cross-validation for Decision Tree before training
cv_scores3 = cross_val_score(DecisionTreeRegressor(max_depth=5, min_samples_split=10, min_samples_leaf=5, 
                                                   random_state=500953158), 
                             coordinates, stepvalues, cv=5, scoring=scoring)
cv_mae3 = -cv_scores3.mean()
print(f"Cross-validation MAE for Decision Tree: {round(cv_mae3, 2)}")

# Proceed with your original code for fitting the models and evaluating on test data

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
my_model2 = RandomForestRegressor(n_estimators=50,  # Number of trees
                                  max_depth=5,      # Reduce tree depth
                                  min_samples_split=10,  # Increase the minimum number of samples to split a node
                                  min_samples_leaf=5,    # Increase the minimum number of samples per leaf node
                                  max_features='sqrt',   # Consider more features per split
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
best_model2_1 = grid_search.best_estimator_

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
y_pred_test = best_model2_1.predict(x_test)


# Calculate the test Mean Absolute Error (MAE)
mae_test = mean_absolute_error(y_test, y_pred_test)
print("Test MAE of the best model is:", round(mae_test, 2))

# ============================
# Adding Model 3: Decision Tree Regressor
# ============================
param_grid_tree = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
my_model3 = DecisionTreeRegressor(max_depth=5,           # Reduce tree depth
                                  min_samples_split=10,  # Increase the minimum number of samples to split a node
                                  min_samples_leaf=5,    # Increase the minimum number of samples per leaf node
                                  random_state=500953158)
my_model3.fit(x_train, y_train)
y_pred_train3 = my_model3.predict(x_train)

# Evaluate training MAE
mae_train3 = mean_absolute_error(y_pred_train3, y_train)
print("Updated Model 3 training MAE is: ", round(mae_train3, 2))

# Test Model 3 on the test data
y_pred_test3 = my_model3.predict(x_test)
mae_test3 = mean_absolute_error(y_test, y_pred_test3)
print("Updated Model 3 test MAE is: ", round(mae_test3, 2))

# ============================
# Adding Model 3.1: Decision Tree Regressor with GridSearchCV
# ============================

# Define the parameter grid for GridSearchCV
param_grid_tree = {
    'max_depth': [5, 10, 20, None],  # Experiment with different tree depths
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4, 5]  # Minimum number of samples in a leaf node
}

# Initialize the Decision Tree Regressor for GridSearchCV
my_model3_1 = DecisionTreeRegressor(random_state=500953158)

# Perform Grid Search with 5-fold cross-validation
grid_search_tree = GridSearchCV(my_model3_1, 
                                param_grid_tree, 
                                cv=5,  # 5-fold cross-validation
                                scoring='neg_mean_absolute_error',  # Scoring based on Mean Absolute Error
                                n_jobs=4)  # Use multiple cores for faster processing

# Fit the model using GridSearchCV
grid_search_tree.fit(x_train, y_train)

# Get the best hyperparameters
best_params_tree = grid_search_tree.best_params_
print("Best Hyperparameters for Decision Tree (GridSearchCV):", best_params_tree)
print("\n==============================\n")

# Get the best model from GridSearchCV
best_model3_1 = grid_search_tree.best_estimator_

# Evaluate the model on the training data
y_pred_train3_1 = best_model3_1.predict(x_train)
mae_train3_1 = mean_absolute_error(y_pred_train3_1, y_train)
print("Best Model 3.1 (Decision Tree with GridSearchCV) training MAE is: ", round(mae_train3_1, 2))

# Evaluate the model on the test data
y_pred_test3_1 = best_model3_1.predict(x_test)
mae_test3_1 = mean_absolute_error(y_test, y_pred_test3_1)
print("Best Model 3.1 (Decision Tree with GridSearchCV) test MAE is: ", round(mae_test3_1, 2))