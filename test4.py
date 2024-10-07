# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:38:16 2024

@author: anant
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
import numpy as np

# Load your CSV file
csv_file = 'Project_1_Data.csv'
df = pd.read_csv(csv_file)

# Plotting the coordinates vs step function
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
plt.title('Correlation Matrix of X, Y, Z, and Step', fontsize=15)

# Features and target
coordinates = df[['X', 'Y', 'Z']].values
stepvalues = df['Step'].values

# Split the data into training and testing sets to prevent data leakage
x_train, x_test, y_train, y_test = train_test_split(coordinates, stepvalues,shuffle=True, test_size=0.2, random_state=500953158)

# Define a scoring function
neg_mae = 'neg_mean_absolute_error'

# ============================
# Model 1: Linear Regression
# ============================
cv_scores1 = cross_val_score(LinearRegression(), x_train, y_train, cv=5, scoring=neg_mae)
cv_mae1 = -cv_scores1.mean()  # Negate because scoring returns negative MAE
print("Model 1 Cross-Validation Mean Absolute Error (CV MAE):", round(cv_mae1, 2))

my_model1 = LinearRegression()
my_model1.fit(x_train, y_train)
y_pred_test1 = my_model1.predict(x_test)
mae_test1 = mean_absolute_error(y_test, y_pred_test1)
print("Model 1 (Linear Regression) Test MAE is: ", round(mae_test1, 2))
print("\n========================================================\n")

# ============================
# Model 2: Random Forest (Consistent Parameters)
# ============================

# Define the Random Forest model with consistent hyperparameters for both cross-validation and final training
my_model2 = RandomForestRegressor(n_estimators=100,  # Increased number of trees for better generalization
                                  max_depth=2,      # Limit depth of trees to prevent overfitting
                                  min_samples_split=2,  # Increase minimum samples to split a node
                                  min_samples_leaf=2,   # Increase minimum samples in leaf nodes
                                  max_features='sqrt',  # Use square root of features for splits
                                  random_state=500953158)

# Perform 5-fold cross-validation using the same model configuration
cv_scores2 = cross_val_score(my_model2, x_train, y_train, cv=5, scoring=neg_mae)

# Calculate Cross-Validation MAE
cv_mae2 = -cv_scores2.mean()  # Negate because scoring returns negative MAE
print("Model 2 Cross-Validation Mean Absolute Error (CV MAE):", round(cv_mae2, 2))

# Now train the final model on the full training set using the same hyperparameters
my_model2.fit(x_train, y_train)

# Make predictions on the test set
y_pred_test2 = my_model2.predict(x_test)

# Calculate the Mean Absolute Error on the test set
mae_test2 = mean_absolute_error(y_test, y_pred_test2)
print("Model 2 (Random Forest) Test MAE is: ", round(mae_test2, 2))
print("\n========================================================\n")

# ============================
# Adding Model 2.1: Random Forest with GridSearchCV
# ============================
param_grid_rf = {
    'n_estimators': [10, 30, 50, 100],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

my_model2_1 = RandomForestRegressor(random_state=500953158)
grid_search_rf = GridSearchCV(my_model2_1, param_grid_rf, cv=5, scoring='neg_mean_absolute_error', n_jobs=4)

grid_search_rf.fit(x_train, y_train)
best_params_rf = grid_search_rf.best_params_
print("Best Hyperparameters for Random Forest (GridSearchCV):", best_params_rf)

best_model2_1 = grid_search_rf.best_estimator_
y_pred_test2_1 = best_model2_1.predict(x_test)
mae_test2_1 = mean_absolute_error(y_test, y_pred_test2_1)
print("\nBest Model 2.1 (Random Forest with GridSearchCV) Test MAE is: ", round(mae_test2_1, 2))
print("\n========================================================\n")

# ============================
# Model 2.2: Random Forest with RandomizedSearchCV (Organized)
# ============================

# Define the hyperparameter grid for RandomizedSearchCV
param_dist_rf = {
    'n_estimators': [10, 50, 100, 200],      # Number of trees in the forest
    'max_depth': [None, 5, 10, 20],          # Maximum depth of the tree
    'min_samples_split': [2, 5, 10, 20],     # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4, 10],       # Minimum number of samples required to be at a leaf node
    'max_features': ['sqrt', 'log2', None]   # Number of features to consider for the best split
}

# Define the Random Forest Regressor model
my_model2_2 = RandomForestRegressor(random_state=500953158)

# Apply RandomizedSearchCV with 5-fold cross-validation
random_search_rf = RandomizedSearchCV(
    estimator=my_model2_2,                  # The Random Forest model
    param_distributions=param_dist_rf,      # The hyperparameter grid
    n_iter=10,                              # Number of random combinations to try
    cv=5,                                   # 5-fold cross-validation
    scoring='neg_mean_absolute_error',      # Scoring function (Negative MAE)
    n_jobs=4,                               # Use multiple cores for faster processing
    random_state=500953158                  # Reproducibility
)

# Fit the RandomizedSearchCV model
random_search_rf.fit(x_train, y_train)

# Get the best hyperparameters from RandomizedSearchCV
best_params_rf_random = random_search_rf.best_params_
print("Best Hyperparameters for Random Forest (RandomizedSearchCV):", best_params_rf_random)

# Use the best estimator to make predictions
best_model2_2 = random_search_rf.best_estimator_
y_pred_test2_2 = best_model2_2.predict(x_test)

# Calculate the Mean Absolute Error for the test set
mae_test2_2 = mean_absolute_error(y_test, y_pred_test2_2)
print("\nBest Model 2.2 (Random Forest with RandomizedSearchCV) Test MAE is: ", round(mae_test2_2, 2))
print("\n========================================================\n")
# ============================
# Model 3: Decision Tree Regressor
# ============================

# Define the Decision Tree Regressor
my_model3 = DecisionTreeRegressor(max_depth=3, 
                                  min_samples_split=50, 
                                  min_samples_leaf=10, 
                                  random_state=500953158)

# Perform 5-fold cross-validation using the defined model
cv_scores_model3 = cross_val_score(my_model3, x_train, y_train, cv=5, scoring=neg_mae)

# Now train the final model on the full training set using the same hyperparameters
my_model3.fit(x_train, y_train)

# Make predictions on the test set
y_pred_test3 = my_model3.predict(x_test)

# Calculate the Mean Absolute Error on the test set
mae_test3 = mean_absolute_error(y_test, y_pred_test3)
# Calculate the Cross-Validation Mean Absolute Error (CV MAE)
cv_mae3 = -cv_scores_model3.mean()  # Negate because scoring returns negative MAE
print("Model 3 Cross-Validation Mean Absolute Error (CV MAE):", round(cv_mae3, 2))
print("Model 3 (Decision Tree) Test MAE is: ", round(mae_test3, 2))
print("\n========================================================\n")

# ============================
# Adding Model 3.1: Decision Tree Regressor with GridSearchCV
# ============================
param_grid_tree = {
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 5]
}

my_model3_1 = DecisionTreeRegressor(random_state=500953158)
grid_search_tree = GridSearchCV(my_model3_1, 
                                param_grid_tree, 
                                cv=5, 
                                scoring='neg_mean_absolute_error', 
                                n_jobs=4)

grid_search_tree.fit(x_train, y_train)
best_params_tree = grid_search_tree.best_params_
print("Best Hyperparameters for Decision Tree (GridSearchCV):", best_params_tree)

best_model3_1 = grid_search_tree.best_estimator_
y_pred_test3_1 = best_model3_1.predict(x_test)
mae_test3_1 = mean_absolute_error(y_test, y_pred_test3_1)
print("\nBest Model 3.1 (Decision Tree with GridSearchCV) Test MAE is: ", round(mae_test3_1, 2))

print("\n========================================================\n")

#===================
#Confusion MAtrix
#===================

# Adjust the binning process
bins = np.linspace(df['Step'].min(), df['Step'].max(), 5)

# Bin the 'Step' values, setting labels=False ensures the binning process assigns numeric bins
df['Step_binned'] = pd.cut(df['Step'], bins=bins, labels=False, include_lowest=True)

# Train-test split on binned 'Step' values (for classification purposes)
y_train_binned = pd.cut(y_train, bins=bins, labels=False, include_lowest=True)
y_test_binned = pd.cut(y_test, bins=bins, labels=False, include_lowest=True)

# Convert NaN values to a specific bin (e.g., -1)
y_train_binned = np.nan_to_num(y_train_binned, nan=-1)
y_test_binned = np.nan_to_num(y_test_binned, nan=-1)

# Train the Decision Tree model (already done above)
my_model3.fit(x_train, y_train)

# Predict the continuous 'Step' values
y_pred_test3 = my_model3.predict(x_test)

# Bin the predicted 'Step' values
y_pred_test3_binned = pd.cut(y_pred_test3, bins=bins, labels=False, include_lowest=True)

# Convert NaN values to a specific bin (e.g., -1) in predictions
y_pred_test3_binned = np.nan_to_num(y_pred_test3_binned, nan=-1)

# Create the confusion matrix
conf_matrix = confusion_matrix(y_test_binned, y_pred_test3_binned)

# Plot the confusion matrix using seaborn's heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)

# Add labels and title
plt.title("Confusion Matrix for Decision Tree Regressor (Binned)", fontsize=15)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)

# Show the plot
plt.show()