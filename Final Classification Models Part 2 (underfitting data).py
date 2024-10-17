# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:56:12 2024

@author: Anant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load your dataset
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

correlation_matrix = df[['X', 'Y', 'Z', 'Step']].corr()
plt.figure()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, square=True, linewidths=0.5)
plt.title('Correlation Matrix of X, Y, Z, and Step', fontsize=15)

# Preparing the data
X = df.drop('Step', axis=1)
y = df['Step']

x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=500953158)

# Train a Random Forest classifier
print("======= Random Forest =======\n")
rf_classifier = RandomForestClassifier(n_estimators=100, 
                                       max_depth=2, 
                                       random_state=500953158)

# Define parameter grid for GridSearch
param_grid_rf = {
    'n_estimators': [10,20, 30, 40],  # Fewer trees to make the model simpler
    'max_depth': [1,2,3],  # Shallower trees
    'min_samples_split': [2,4,6],  # Require more samples to split nodes
    'min_samples_leaf': [1,2,3],  # Require more samples at each leaf node
    'bootstrap': [True]  # Keep bootstrap to introduce randomness and generalization
}

# Perform Grid Search with cross-validation
grid_search_rf = GridSearchCV(estimator=rf_classifier, 
                              param_grid=param_grid_rf, 
                              cv=5, 
                              n_jobs=-1)

grid_search_rf.fit(x_train, y_train)

# Best parameters from grid search
print(f"Best parameters from grid search: {grid_search_rf.best_params_}")

# Predict using the best Random Forest model from grid search
y_pred_rf_grid = grid_search_rf.predict(x_test)

# Train the initial Random Forest model without Grid Search for comparison
rf_classifier.fit(x_train, y_train)
y_pred_rf = rf_classifier.predict(x_test)

# Calculate and print F1 scores for both models
f1_initial = f1_score(y_test, y_pred_rf, average='weighted')
f1_grid_search = f1_score(y_test, y_pred_rf_grid, average='weighted')

print(f"F1 Score for the initial Random Forest model: {f1_initial:.4f}")
print(f"F1 Score for the best Random Forest model from grid search: {f1_grid_search:.4f}")

# SVM Classifier model
print("\n======= SVM Classifier =======\n")
svm_classifier = SVC(random_state=500953158)

# Define parameter grid for GridSearch
param_grid_svm = {
    'C': [0.01, 0.1, 0.2, 0.5],  # Reduced regularization parameter for simpler models
    'kernel': ['linear'],  # Limiting to a simpler kernel
    'gamma': ['scale']  # Keeping a simple gamma for the linear case
}

# Perform Grid Search with cross-validation
grid_search_svm = GridSearchCV(estimator=svm_classifier, 
                               param_grid=param_grid_svm, 
                               cv=5, 
                               n_jobs=-1)

# Fit the model with grid search on training data
grid_search_svm.fit(x_train, y_train)

# Best parameters from grid search
print(f"Best parameters from grid search: {grid_search_svm.best_params_}")

# Predict using the best SVM model from grid search
y_pred_svm_grid = grid_search_svm.predict(x_test)

# Train the initial SVM model without Grid Search for comparison
svm_classifier.fit(x_train, y_train)
y_pred_svm = svm_classifier.predict(x_test)

# Calculate and print F1 scores for both models
f1_initial_svm = f1_score(y_test, y_pred_svm, average='weighted')
f1_grid_search_svm = f1_score(y_test, y_pred_svm_grid, average='weighted')

print(f"F1 Score for the initial SVM model: {f1_initial_svm:.4f}")
print(f"F1 Score for the best SVM model from grid search: {f1_grid_search_svm:.4f}")

# Simple DecisionTreeClassifier - No Random Forest Parameters
print("\n======= Decision Tree Classifier (Randomized Search) =======\n")
dt_classifier = DecisionTreeClassifier(max_depth=3, 
                                       min_samples_split=3, 
                                       min_samples_leaf=3, 
                                       random_state=500953158)

# Parameter grid specific to Decision Tree (no Random Forest parameters)
param_dist_dt = {
    'criterion': ['gini'],  # Limiting to one simpler criterion
    'splitter': ['best', 'random'],  # Keeping only the 'best' splitter
    'max_depth': [1, 2, 3],  # Reducing tree depth for simplicity
    'min_samples_split': [10, 12, 15],  # Increasing minimum samples required to split
    'min_samples_leaf': [5, 7, 9],  # Requiring more samples per leaf for less complexity
    'class_weight': [None]  # Avoid balancing weights to keep it simple
}

# RandomizedSearchCV for Decision Tree (no n_estimators or bootstrap)
random_search_dt = RandomizedSearchCV(estimator=dt_classifier, 
                                      param_distributions=param_dist_dt, 
                                      n_iter=5, 
                                      cv=5, 
                                      n_jobs=-1, 
                                      random_state=500953158)

# Fit the model
random_search_dt.fit(x_train, y_train)

# Predict using the best Decision Tree model from Randomized Search
y_pred_dt_random = random_search_dt.predict(x_test)

# Train the initial Decision Tree model without Randomized Search for comparison
dt_classifier.fit(x_train, y_train)
y_pred_dt = dt_classifier.predict(x_test)

# Calculate and print F1 scores for both models
f1_initial_dt = f1_score(y_test, y_pred_dt, average='weighted')
f1_random_search_dt = f1_score(y_test, y_pred_dt_random, average='weighted')

print(f"F1 Score for the initial Decision Tree model: {f1_initial_dt:.4f}")
print(f"F1 Score for the best Decision Tree model from Randomized Search: {f1_random_search_dt:.4f}")

# Stacking Classifier
print("\nStacking Classifier\n")
estimators = [
    ('rf', rf_classifier),  # Random Forest
    ('dt', dt_classifier)   # Decision Tree
]

# Stacking Classifier with Logistic Regression as final estimator
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# Train the stacked model
stacking_clf.fit(x_train, y_train)

# Evaluate the stacked model
y_pred_stacking = stacking_clf.predict(x_test)

# Calculate and print F1 Score
f1_stacking = f1_score(y_test, y_pred_stacking, average='weighted')
print(f"F1 Score for the Stacked Model: {f1_stacking:.4f}")


# Generate confusion matrix for the stacked model
confusion_matrix_stacked = confusion_matrix(y_test, y_pred_stacking)

# Display the confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_stacked)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Stacked Model")
plt.show()





# Predict using the stacked model on the training data
y_pred_stacking_train = stacking_clf.predict(x_train)

# Generate confusion matrix for the training data
confusion_matrix_stacked_train = confusion_matrix(y_train, y_pred_stacking_train)

# Display the confusion matrix using ConfusionMatrixDisplay
disp_train = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_stacked_train)
disp_train.plot(cmap='Blues')
plt.title("Confusion Matrix for Stacked Model (Training Data)")
plt.show()

joblib.dump(stacking_clf, 'stacking_model.joblib')
print("Stacking model saved to 'stacking_model.joblib'.")

# Load the model back from the file to demonstrate reusability
loaded_stacking_model = joblib.load('stacking_model.joblib')
print("Stacking model loaded from 'stacking_model.joblib'.")
print("\n============================\n")

# Given set of random coordinates for maintenance step prediction
new_coordinates = np.array([[9.375, 3.0625, 1.51], 
                            [6.995, 5.125, 0.3875], 
                            [0, 3.0625, 1.93], 
                            [9.4, 3, 1.8], 
                            [9.4, 3, 1.3]])

# Predict maintenance steps using the loaded model
predicted_steps = loaded_stacking_model.predict(new_coordinates)

# Output the predicted maintenance steps for the given coordinates
print("Predicted Maintenance Steps for given coordinates:")
for coords, step in zip(new_coordinates, predicted_steps):
    print(f"Coordinates {coords} -> Predicted Maintenance Step: {step}")
    



# Initial 3D Scatter Plot of the original dataset (already plotted earlier)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
sc = ax1.scatter(df['X'], df['Y'], df['Z'], c=df['Step'], cmap='viridis', label='Original Data')
plt.colorbar(sc, ax=ax1, label='Step')

# Set labels and title
ax1.set_xlabel('X')
ax1.set_ylabel('Step')
ax1.set_zlabel('Z')
ax1.set_title('3D Scatter Plot of X, Y, Z with Step')

# Adding predicted coordinates in a different color (e.g., red)
predicted_coordinates = np.array([[9.375, 3.0625, 1.51], 
                                  [6.995, 5.125, 0.3875], 
                                  [0, 3.0625, 1.93], 
                                  [9.4, 3, 1.8], 
                                  [9.4, 3, 1.3]])

# Plot the new predicted points in red
ax1.scatter(predicted_coordinates[:, 0], 
            predicted_coordinates[:, 1], 
            predicted_coordinates[:, 2], 
            c='red', label='Predicted Data', s=100, edgecolors='k')

# Show legend to differentiate between original and predicted points
ax1.legend()

# Show the updated plot
plt.show()

