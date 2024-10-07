# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:58:54 2024

@author: anant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

# Load your CSV file
csv_file = 'Project_1_Data.csv'
df = pd.read_csv(csv_file)

# Plotting the coordinates vs step function (no change here)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
sc = ax1.scatter(df['X'], df['Y'], df['Z'], c=df['Step'], cmap='viridis')
plt.colorbar(sc, ax=ax1, label='Step')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Scatter Plot of X, Y, Z with Step')
plt.show()

# Bin the 'Step' values into classes
# You can customize the number of bins or categories based on your specific use case
bins = np.linspace(df['Step'].min(), df['Step'].max(), 4)  # 3 bins: low, medium, high
df['Step_binned'] = pd.cut(df['Step'], bins=bins, labels=[0, 1, 2], include_lowest=True)  # Label as 0, 1, 2

# Calculate the correlation matrix for X, Y, Z, and Step columns (optional)
correlation_matrix = df[['X', 'Y', 'Z', 'Step_binned']].corr()
plt.figure()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, square=True, linewidths=0.5)
plt.title('Correlation Matrix of X, Y, Z, and Step Binned', fontsize=15)

# Features and target
coordinates = df[['X', 'Y', 'Z']].values
step_binned = df['Step_binned'].values

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(coordinates, step_binned, test_size=0.2, random_state=500953158)

# ============================
# Model 1: Logistic Regression
# ============================
my_model1 = LogisticRegression(max_iter=200)
my_model1.fit(x_train, y_train)
y_pred_test1 = my_model1.predict(x_test)

# Evaluate the Logistic Regression model
accuracy1 = accuracy_score(y_test, y_pred_test1)
print("Model 1 (Logistic Regression) Test Accuracy:", round(accuracy1, 2))

print("\nClassification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_test1))

# ============================
# Model 2: Random Forest Classifier
# ============================
my_model2 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=500953158)
my_model2.fit(x_train, y_train)
y_pred_test2 = my_model2.predict(x_test)

# Evaluate the Random Forest model
accuracy2 = accuracy_score(y_test, y_pred_test2)
print("\nModel 2 (Random Forest Classifier) Test Accuracy:", round(accuracy2, 2))

print("\nClassification Report for Random Forest:")
print(classification_report(y_test, y_pred_test2))

# ============================
# Model 3: Decision Tree Classifier
# ============================
my_model3 = DecisionTreeClassifier(max_depth=5, random_state=500953158)
my_model3.fit(x_train, y_train)
y_pred_test3 = my_model3.predict(x_test)

# Evaluate the Decision Tree model
accuracy3 = accuracy_score(y_test, y_pred_test3)
print("\nModel 3 (Decision Tree Classifier) Test Accuracy:", round(accuracy3, 2))

print("\nClassification Report for Decision Tree:")
print(classification_report(y_test, y_pred_test3))

# ============================
# Confusion Matrix for Model 3 (Decision Tree)
# ============================
conf_matrix = confusion_matrix(y_test, y_pred_test3)

# Plot confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix for Decision Tree Classifier", fontsize=15)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.show()

# ============================
# Optional: Hyperparameter Tuning with GridSearchCV
# ============================
param_grid_tree = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_tree = GridSearchCV(DecisionTreeClassifier(random_state=500953158),
                                param_grid_tree, cv=5, scoring='accuracy', n_jobs=4)

grid_search_tree.fit(x_train, y_train)
best_params_tree = grid_search_tree.best_params_
print("\nBest Hyperparameters for Decision Tree (GridSearchCV):", best_params_tree)

best_model_tree = grid_search_tree.best_estimator_
y_pred_test_best_tree = best_model_tree.predict(x_test)
accuracy_best_tree = accuracy_score(y_test, y_pred_test_best_tree)
print("\nBest Decision Tree Classifier Test Accuracy after GridSearchCV:", round(accuracy_best_tree, 2))
