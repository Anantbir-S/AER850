# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:56:12 2024

@author: Anant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay

# Load your dataset
csv_file = 'Project_1_Data.csv'
df = pd.read_csv(csv_file)

# Preparing the data
X = df.drop('Step', axis=1)
y = df['Step']

# Stratified train-test split to maintain class distribution
x_train, x_test, y_train, y_test = train_test_split(X, y, 
                                                    stratify=y, 
                                                    shuffle=True, 
                                                    test_size=0.2, 
                                                    random_state=500953158)

# Combine training features and target for plotting
train_data = pd.DataFrame(x_train, columns=['X', 'Y', 'Z'])
train_data['Step'] = y_train.values

# 3D Scatter Plot using only the training data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(train_data['X'], 
                train_data['Y'],
                train_data['Z'], 
                c=train_data['Step'], 
                cmap='viridis', 
                label='Training Data')

plt.colorbar(sc, ax=ax, label='Step')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot of Training Data (X, Y, Z) with Step')

# Correlation matrix using only the training data
correlation_matrix = train_data[['X', 'Y', 'Z', 'Step']].corr()
plt.figure()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, square=True, linewidths=0.5)
plt.title('Correlation Matrix of Training Data (X, Y, Z, and Step)', fontsize=15)
plt.show()

# Train a Random Forest classifier with GridSearchCV
print("======= Random Forest =======\n")
rf_classifier = RandomForestClassifier(random_state=500953158)

# Parameter grid for GridSearchCV
param_grid_rf = {
    'n_estimators': [10, 20, 30, 40],
    'max_depth': [1, 2, 3],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3],
    'bootstrap': [True]
}

# Perform Grid Search with cross-validation
grid_search_rf = GridSearchCV(estimator=rf_classifier, 
                              param_grid=param_grid_rf, 
                              cv=5, 
                              n_jobs=-1)

grid_search_rf.fit(x_train, y_train)

# Best parameters from grid search
print(f"Best parameters from Grid Search: {grid_search_rf.best_params_}")

# Predict using the best Random Forest model from grid search
y_pred_rf_grid = grid_search_rf.predict(x_test)

# Calculate and print F1 score
f1_grid_search = f1_score(y_test, y_pred_rf_grid, average='weighted')
print(f"F1 Score for the best Random Forest model from Grid Search: {f1_grid_search:.4f}")

# SVM Classifier model with GridSearchCV
print("\n======= SVM Classifier =======\n")
svm_classifier = SVC(random_state=500953158)

# Parameter grid for GridSearchCV
param_grid_svm = {
    'C': [1]  # Single value for the regularization parameter
}

# Perform Grid Search with cross-validation
grid_search_svm = GridSearchCV(estimator=svm_classifier,
                               param_grid=param_grid_svm, 
                               cv=5,
                               n_jobs=-1)

grid_search_svm.fit(x_train, y_train)

# Best parameters from Grid Search
print(f"Best parameters from Grid Search: {grid_search_svm.best_params_}")

# Predict using the best SVM model from Grid Search
y_pred_svm_grid = grid_search_svm.predict(x_test)

# Calculate and print F1 score
f1_grid_search_svm = f1_score(y_test, y_pred_svm_grid, average='weighted')
print(f"F1 Score for the best SVM model from Grid Search: {f1_grid_search_svm:.4f}")

# Decision Tree Classifier with RandomizedSearchCV
print("\n======= Decision Tree Classifier (Randomized Search) =======\n")
dt_classifier = DecisionTreeClassifier(random_state=500953158)

# Parameter distribution for RandomizedSearchCV
param_dist_dt = {
    'criterion': ['gini'],
    'splitter': ['best', 'random'],
    'max_depth': [1, 2, 3],
    'min_samples_split': [10, 12, 15],
    'min_samples_leaf': [5, 7, 9],
    'class_weight': [None]
}

# RandomizedSearchCV for Decision Tree
random_search_dt = RandomizedSearchCV(estimator=dt_classifier, param_distributions=param_dist_dt, n_iter=5, cv=5, n_jobs=-1, random_state=500953158)
random_search_dt.fit(x_train, y_train)

# Predict using the best Decision Tree model from Randomized Search
y_pred_dt_random = random_search_dt.predict(x_test)

# Calculate and print F1 score
f1_random_search_dt = f1_score(y_test, y_pred_dt_random, average='weighted')
print(f"F1 Score for the best Decision Tree model from Randomized Search: {f1_random_search_dt:.4f}")

# Stacking Classifier
print("\nStacking Classifier\n")
estimators = [
    ('rf', grid_search_rf.best_estimator_),  # Random Forest from Grid Search
    ('svm', grid_search_svm.best_estimator_)  # SVM from Grid Search
]

# Stacking Classifier with Logistic Regression as final estimator
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# Train the Stacking Classifier
stacking_clf.fit(x_train, y_train)

# Predict using the Stacking Classifier
y_pred_stacking = stacking_clf.predict(x_test)

# Calculate and print F1 Score for the Stacked Model
f1_stacking = f1_score(y_test, y_pred_stacking, average='weighted')
print(f"F1 Score for the Stacked Model: {f1_stacking:.4f}")

# Confusion matrix for the Stacked Model
confusion_matrix_stacked = confusion_matrix(y_test, y_pred_stacking)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_stacked)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Stacked Model")
plt.show()

# Save the Stacked Model
joblib.dump(stacking_clf, 'stacking_model.joblib')
print("Stacking model saved to 'stacking_model.joblib'.")

new_coordinates = np.array([[9.375, 3.0625, 1.51], 
                            [6.995, 5.125, 0.3875], 
                            [0, 3.0625, 1.93], 
                            [9.4, 3, 1.8], 
                            [9.4, 3, 1.3]])

# Convert new_coordinates to a pandas DataFrame with correct column names
new_coordinates_df = pd.DataFrame(new_coordinates, columns=['X', 'Y', 'Z'])

# Predict the maintenance steps for the given coordinates using the stacked classifier
predicted_steps = stacking_clf.predict(new_coordinates_df)

# Output the predicted maintenance steps for the given coordinates
print("Predicted Maintenance Steps for the given coordinates:")
for coords, step in zip(new_coordinates, predicted_steps):
    print(f"Coordinates {coords} -> Predicted Maintenance Step: {step}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the original training data
sc = ax.scatter(train_data['X'], train_data['Y'], train_data['Z'], c=train_data['Step'], cmap='viridis', label='Training Data')
plt.colorbar(sc, ax=ax, label='Step')

# Plot the new predicted points in red
ax.scatter(new_coordinates[:, 0], new_coordinates[:, 1], new_coordinates[:, 2], 
           c='red', label='Predicted Data', s=100, edgecolors='k')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot with Training Data and Predicted Points')

# Show legend to differentiate between original and predicted points
ax.legend()

# Show the updated plot
plt.show()