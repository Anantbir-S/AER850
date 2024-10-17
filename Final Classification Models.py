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
# Plot the heatmap for the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, square=True, linewidths=0.5)
plt.title('Correlation Matrix of X, Y, Z, and Step', fontsize=15)

X = df.drop('Step', axis=1)
y = df['Step']

x_train, x_test, y_train, y_test = train_test_split(X, y,shuffle=True, test_size=0.2, random_state=500953158)

# Train a Random Forest classifier

print("======= Random Forest =======\n")
rf_classifier = RandomForestClassifier(n_estimators=100,
                                       max_depth=2, 
                                       random_state=500953158)

# Define parameter grid for GridSearch
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Perform Grid Search with cross-validation
grid_search_rf = GridSearchCV(estimator=rf_classifier, 
                              param_grid=param_grid, 
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
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear', 'rbf'],  # Kernel type
    'gamma': ['scale', 'auto'],  # Kernel coefficient
    'degree': [2, 3, 4],  # Only for polynomial kernel, so will be used when kernel='poly'
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
dt_classifier = DecisionTreeClassifier(
    max_depth=2,  # You can adjust this value
    min_samples_split=20,  # Require more samples to split nodes
    min_samples_leaf=5,  # Force larger leaf sizes
    random_state=500953158
)

# Parameter grid specific to Decision Tree (no Random Forest parameters)
param_dist_dt = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'class_weight': [None, 'balanced']
}

# RandomizedSearchCV for Decision Tree (no n_estimators or bootstrap)
random_search_dt = RandomizedSearchCV(
    estimator=dt_classifier,
    param_distributions=param_dist_dt,
    n_iter=10,  # Number of random parameter combinations to test
    cv=5,  # Cross-validation
    verbose=2,
    n_jobs=-1,  # Use all cores
    random_state=500953158
)

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