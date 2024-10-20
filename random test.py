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
from sklearn.metrics import f1_score, confusion_matrix, precision_score, accuracy_score, recall_score
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
                                                    shuffle=True, 
                                                    test_size=0.2, 
                                                    random_state=500953158)

# Combine training features and target for plotting
train_data = pd.DataFrame(x_train, columns=['X', 'Y', 'Z'])
train_data['Step'] = y_train.values

""" PLotting the Data and the Correlation Matrix """

# Plot histograms for X, Y, Z, and Step
plt.figure(figsize=(12, 10))

# Histogram for X
plt.subplot(2, 2, 1)
sns.histplot(df['X'], kde=True, bins=20, color='blue')
plt.title('Histogram of X')

# Histogram for Y
plt.subplot(2, 2, 2)
sns.histplot(df['Y'], kde=True, bins=20, color='green')
plt.title('Histogram of Y')

# Histogram for Z
plt.subplot(2, 2, 3)
sns.histplot(df['Z'], kde=True, bins=20, color='red')
plt.title('Histogram of Z')

# Histogram for Step
plt.subplot(2, 2, 4)
sns.histplot(df['Step'], kde=False, bins=13, color='purple')
plt.title('Histogram of Step (Maintenance Steps)')

plt.tight_layout()
plt.show()

# 3D Scatter Plot using only the training data
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Adjust marker size and add transparency for better clarity
sc = ax.scatter(train_data['X'], 
                train_data['Y'], 
                train_data['Z'], 
                c=train_data['Step'], 
                cmap='viridis', 
                s=50,       # Adjusted size of the markers
                alpha=0.8,  # Slight transparency for better visibility
                edgecolors='w',  # White edge color for markers
                linewidth=0.5,   # Thinner edge for better marker distinction
                label='Training Data')

# Adding a color bar with the label
cbar = plt.colorbar(sc, ax=ax, label='Step')
cbar.set_ticks(range(int(train_data['Step'].min()), int(train_data['Step'].max())+1))

# Set axis labels
ax.set_xlabel('X', fontsize=12, labelpad=10)
ax.set_ylabel('Y', fontsize=12, labelpad=10)
ax.set_zlabel('Z', fontsize=12, labelpad=10)

# Set axis limits for better view control
ax.set_xlim(train_data['X'].min(), train_data['X'].max())
ax.set_ylim(train_data['Y'].min(), train_data['Y'].max())
ax.set_zlim(train_data['Z'].min(), train_data['Z'].max())

# Set a clean title
ax.set_title('3D Scatter Plot of Training Data (X, Y, Z) with Step', fontsize=14, pad=20)

# Rotate the view for a better 3D effect
ax.view_init(elev=30, azim=120)

# Show the plot
plt.show()

# Correlation matrix using only the training data
correlation_matrix = train_data[['X', 'Y', 'Z', 'Step']].corr()
plt.figure()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, square=True, linewidths=0.5)
plt.title('Correlation Matrix of Training Data (X, Y, Z, and Step)', fontsize=15)
plt.show()

"""======================= Logistic Regression Classifier (Training & Test Data) ====================================================================""" 

# Logistic Regression Classifier with GridSearchCV
print("\n======= Logistic Regression Classifier =======\n")

# Define the Logistic Regression model
logistic = LogisticRegression(multi_class='ovr', solver='saga', max_iter=5000)  

# Set up the hyperparameter grid for Logistic Regression
param_grid_lr = {
    'C': [0.0001, 0.001, 0.01, 0.1, 1],   
    'penalty': ['l1', 'l2']
}

# Perform Grid Search with cross-validation
grid_search_lr = GridSearchCV(estimator=logistic, 
                              param_grid=param_grid_lr, 
                              cv=10, 
                              n_jobs=-1)

# Fit the Logistic Regression model using Grid Search
grid_search_lr.fit(x_train, y_train)

# Best parameters from Grid Search
print(f"Best parameters from Grid Search: {grid_search_lr.best_params_}\n")

# Predict on the test and train data
y_pred_lr_grid_test = grid_search_lr.predict(x_test)
y_pred_lr_grid_train = grid_search_lr.predict(x_train)

# Test metrics
f1_grid_search_lr_test = f1_score(y_test, y_pred_lr_grid_test, average='weighted')
recall_lr_test = recall_score(y_test, y_pred_lr_grid_test, average='weighted')
precision_lr_test = precision_score(y_test, y_pred_lr_grid_test, average='weighted')
accuracy_lr_test = accuracy_score(y_test, y_pred_lr_grid_test)

# Train metrics
f1_grid_search_lr_train = f1_score(y_train, y_pred_lr_grid_train, average='weighted')
recall_lr_train = recall_score(y_train, y_pred_lr_grid_train, average='weighted')
precision_lr_train = precision_score(y_train, y_pred_lr_grid_train, average='weighted')
accuracy_lr_train = accuracy_score(y_train, y_pred_lr_grid_train)

# Print Test Metrics
print(f"F1 Score for Logistic Regression (Test Data): {f1_grid_search_lr_test:.4f}")
print(f"Recall for Logistic Regression (Test Data): {recall_lr_test:.4f}")
print(f"Precision for Logistic Regression (Test Data): {precision_lr_test:.4f}")
print(f"Accuracy for Logistic Regression (Test Data): {accuracy_lr_test:.4f}\n")

# Print Train Metrics
print(f"F1 Score for Logistic Regression (Training Data): {f1_grid_search_lr_train:.4f}")
print(f"Recall for Logistic Regression (Training Data): {recall_lr_train:.4f}")
print(f"Precision for Logistic Regression (Training Data): {precision_lr_train:.4f}")
print(f"Accuracy for Logistic Regression (Training Data): {accuracy_lr_train:.4f}\n")

'''================== Random Forest Classifier (Training & Test Data) ==============================================================================='''

print("======= Random Forest Classifier =======\n")
rf_classifier = RandomForestClassifier(random_state=500953158)

# Parameter grid for GridSearchCV
param_grid_rf = {
    'n_estimators': [50, 70, 100],  # Increased number of trees
    'max_depth': [1, 2, 3, 4],  # Moderate depth to balance complexity
    'min_samples_split': [2, 5, 8],  # Ensure sufficient data for splits
    'min_samples_leaf': [1, 2, 4],  # Control the size of leaf nodes
    'bootstrap': [True]  # Standard bootstrap
}

# Perform Grid Search with cross-validation
grid_search_rf = GridSearchCV(estimator=rf_classifier, 
                              param_grid=param_grid_rf, 
                              cv=5, 
                              n_jobs=-1)

grid_search_rf.fit(x_train, y_train)

# Best parameters from grid search
print(f"Best parameters from Grid Search: {grid_search_rf.best_params_}\n")

# Predict on the test and train data
y_pred_rf_grid_test = grid_search_rf.predict(x_test)
y_pred_rf_grid_train = grid_search_rf.predict(x_train)

# Test metrics
f1_grid_search_rf_test = f1_score(y_test, y_pred_rf_grid_test, average='weighted')
recall_rf_test = recall_score(y_test, y_pred_rf_grid_test, average='weighted')
precision_rf_test = precision_score(y_test, y_pred_rf_grid_test, average='weighted')
accuracy_rf_test = accuracy_score(y_test, y_pred_rf_grid_test)

# Train metrics
f1_grid_search_rf_train = f1_score(y_train, y_pred_rf_grid_train, average='weighted')
recall_rf_train = recall_score(y_train, y_pred_rf_grid_train, average='weighted')
precision_rf_train = precision_score(y_train, y_pred_rf_grid_train, average='weighted')
accuracy_rf_train = accuracy_score(y_train, y_pred_rf_grid_train)

# Print Test Metrics
print(f"F1 Score for Random Forest (Test Data): {f1_grid_search_rf_test:.4f}")
print(f"Recall for Random Forest (Test Data): {recall_rf_test:.4f}")
print(f"Precision for Random Forest (Test Data): {precision_rf_test:.4f}")
print(f"Accuracy for Random Forest (Test Data): {accuracy_rf_test:.4f}\n")

# Print Train Metrics
print(f"F1 Score for Random Forest (Training Data): {f1_grid_search_rf_train:.4f}")
print(f"Recall for Random Forest (Training Data): {recall_rf_train:.4f}")
print(f"Precision for Random Forest (Training Data): {precision_rf_train:.4f}")
print(f"Accuracy for Random Forest (Training Data): {accuracy_rf_train:.4f}\n")

'''================== SVM Classifier (Training & Test Data) ==============================================================================='''

print("\n======= SVM Classifier =======\n")
svm_classifier = SVC(random_state=500953158)

# Parameter grid for GridSearchCV
param_grid_svm = {
    'C': [1]  # Single value for the regularization parameter
}

# Perform Grid Search with cross-validation
grid_search_svm = GridSearchCV(estimator=svm_classifier,
                               param_grid=param_grid_svm, 
                               cv=3,
                               n_jobs=-1)

grid_search_svm.fit(x_train, y_train)

# Best parameters from Grid Search
print(f"Best parameters from Grid Search: {grid_search_svm.best_params_}\n")

# Predict on the test and train data
y_pred_svm_grid_test = grid_search_svm.predict(x_test)
y_pred_svm_grid_train = grid_search_svm.predict(x_train)

# Test metrics
f1_grid_search_svm_test = f1_score(y_test, y_pred_svm_grid_test, average='weighted')
recall_svm_test = recall_score(y_test, y_pred_svm_grid_test, average='weighted')
precision_svm_test = precision_score(y_test, y_pred_svm_grid_test, average='weighted')
accuracy_svm_test = accuracy_score(y_test, y_pred_svm_grid_test)

# Train metrics
f1_grid_search_svm_train = f1_score(y_train, y_pred_svm_grid_train, average='weighted')
recall_svm_train = recall_score(y_train, y_pred_svm_grid_train, average='weighted')
precision_svm_train = precision_score(y_train, y_pred_svm_grid_train, average='weighted')
accuracy_svm_train = accuracy_score(y_train, y_pred_svm_grid_train)

# Print Test Metrics
print(f"F1 Score for SVM (Test Data): {f1_grid_search_svm_test:.4f}")
print(f"Recall for SVM (Test Data): {recall_svm_test:.4f}")
print(f"Precision for SVM (Test Data): {precision_svm_test:.4f}")
print(f"Accuracy for SVM (Test Data): {accuracy_svm_test:.4f}\n")

# Print Train Metrics
print(f"F1 Score for SVM (Training Data): {f1_grid_search_svm_train:.4f}")
print(f"Recall for SVM (Training Data): {recall_svm_train:.4f}")
print(f"Precision for SVM (Training Data): {precision_svm_train:.4f}")
print(f"Accuracy for SVM (Training Data): {accuracy_svm_train:.4f}\n")

'''================== Decision Tree Classifier (Training & Test Data) ==============================================================================='''

print("\n======= Decision Tree Classifier (Training Data) =======\n")
dt_classifier = DecisionTreeClassifier(random_state=500953158)

# Parameter distribution for RandomizedSearchCV
param_dist_dt = {
    'criterion': ['gini'],
    'splitter': ['best', 'random'],
    'max_depth': [1, 2, 3, 4],
    'min_samples_split': [10, 12, 15, 20],
    'min_samples_leaf': [5, 7, 9],
    'class_weight': [None]
}

# RandomizedSearchCV for Decision Tree
random_search_dt = RandomizedSearchCV(estimator=dt_classifier, 
                                      param_distributions=param_dist_dt, 
                                      n_iter=5, 
                                      cv=5, 
                                      n_jobs=-1, 
                                      random_state=500953158)
random_search_dt.fit(x_train, y_train)

# Predict on the test and train data
y_pred_dt_random_test = random_search_dt.predict(x_test)
y_pred_dt_random_train = random_search_dt.predict(x_train)

# Test metrics
f1_random_search_dt_test = f1_score(y_test, y_pred_dt_random_test, average='weighted')
recall_dt_test = recall_score(y_test, y_pred_dt_random_test, average='weighted')
precision_dt_test = precision_score(y_test, y_pred_dt_random_test, average='weighted')
accuracy_dt_test = accuracy_score(y_test, y_pred_dt_random_test)

# Train metrics
f1_random_search_dt_train = f1_score(y_train, y_pred_dt_random_train, average='weighted')
recall_dt_train = recall_score(y_train, y_pred_dt_random_train, average='weighted')
precision_dt_train = precision_score(y_train, y_pred_dt_random_train, average='weighted')
accuracy_dt_train = accuracy_score(y_train, y_pred_dt_random_train)

# Print Test Metrics
print(f"F1 Score for Decision Tree (Test Data): {f1_random_search_dt_test:.4f}")
print(f"Recall for Decision Tree (Test Data): {recall_dt_test:.4f}")
print(f"Precision for Decision Tree (Test Data): {precision_dt_test:.4f}")
print(f"Accuracy for Decision Tree (Test Data): {accuracy_dt_test:.4f}\n")

# Print Train Metrics
print(f"F1 Score for Decision Tree (Training Data): {f1_random_search_dt_train:.4f}")
print(f"Recall for Decision Tree (Training Data): {recall_dt_train:.4f}")
print(f"Precision for Decision Tree (Training Data): {precision_dt_train:.4f}")
print(f"Accuracy for Decision Tree (Training Data): {accuracy_dt_train:.4f}\n")

# Predict using the best Random Forest model from Grid Search
y_pred_rf_grid = grid_search_rf.predict(x_test)

# Generate confusion matrix for the Random Forest Classifier
confusion_matrix_rf = confusion_matrix(y_test, y_pred_rf_grid)

# Display the confusion matrix
disp_rf = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_rf)

# Plot the confusion matrix with a color map
disp_rf.plot(cmap='Blues')
plt.title("Confusion Matrix for Random Forest Classifier")
plt.show()

'''================== Confusion Matrices and Visualization for Stacked Classifier ==========================================================='''

# Stacking Classifier
estimators = [
    ('rf', grid_search_rf.best_estimator_),  # Random Forest from Grid Search
    ('lr', grid_search_lr.best_estimator_)   # Logistic Regression from Grid Search
]
# Stacking Classifier with Logistic Regression as final estimator
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=500))

# Train the Stacking Classifier
stacking_clf.fit(x_train, y_train)

# Predict using the Stacking Classifier
y_pred_stacking = stacking_clf.predict(x_test)

# Calculate and print F1 Score for the Stacked Model
f1_stacking = f1_score(y_test, y_pred_stacking, average='weighted')
print(f"\nF1 Score for the Stacked Model: {f1_stacking:.4f}\n")

# Recall score
recall_stacking = recall_score(y_test, y_pred_stacking, average='weighted')
print(f"Recall for Stacked Classifier: {recall_stacking:.4f}\n")

# Precision score
precision_stacking = precision_score(y_test, y_pred_stacking, average='weighted')
print(f"Precision for Stacked Classifier: {precision_stacking:.4f}\n")

# Accuracy score
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
print(f"Accuracy for Stacked Classifier: {accuracy_stacking:.4f}\n")

# Confusion matrix for the Stacked Model
confusion_matrix_stacked = confusion_matrix(y_test, y_pred_stacking)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_stacked)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Stacked Model")
plt.show()

# Load the saved Stacked Model
stacking_clf = joblib.load('stacking_model.joblib')
print("Stacking model loaded from 'stacking_model.joblib'.")

# Define new coordinates
new_coordinates = np.array([[9.375, 3.0625, 1.51], 
                            [6.995, 5.125, 0.3875], 
                            [0, 3.0625, 1.93], 
                            [9.4, 3, 1.8], 
                            [9.4, 3, 1.3]])

# Convert new_coordinates to a pandas DataFrame with correct column names
new_coordinates_df = pd.DataFrame(new_coordinates, columns=['X', 'Y', 'Z'])

# Predict the maintenance steps for the given coordinates using the loaded stacked classifier
predicted_steps = stacking_clf.predict(new_coordinates_df)

# Output the predicted maintenance steps for the given coordinates
print("\nPredicted Maintenance Steps for the given coordinates:\n")
for coords, step in zip(new_coordinates, predicted_steps):
    print(f"Coordinates {coords} -> Predicted Maintenance Step: {step}")

# Visualization (same as before)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the original training data (this assumes you have access to `train_data`)
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

