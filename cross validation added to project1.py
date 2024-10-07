import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

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
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, square=True, linewidths=0.5)
plt.title('Correlation Matrix of X, Y, Z, and Step', fontsize=15)

# Define features and labels
coordinates = df[['X', 'Y', 'Z']].values
stepvalues = df['Step'].values

x_train, x_test, y_train, y_test = train_test_split(coordinates, stepvalues, test_size=0.2, random_state=500953158)

"""Cross-Validation for Linear Regression (Model 1)"""
cv_scores_model1 = cross_val_score(LinearRegression(), 
                                   x_train, 
                                   y_train, 
                                   cv=5, 
                                   scoring='neg_mean_absolute_error')
cv_mae1 = -cv_scores_model1.mean()
print("Model 1 Cross-Validation Mean Absolute Error (CV MAE):", round(cv_mae1, 2))
print("\n==============================\n")

"""Cross-Validation for Random Forest (Model 2)"""
cv_scores_model2 = cross_val_score(RandomForestRegressor(
                    n_estimators=50, 
                    max_depth=5, 
                    min_samples_split=10, 
                    min_samples_leaf=5, 
                    max_features='sqrt', 
                    random_state=500953158), 
                    x_train, y_train, cv=5, scoring='neg_mean_absolute_error')

cv_mae2 = -cv_scores_model2.mean()
print("Model 2 Cross-Validation Mean Absolute Error (CV MAE):", round(cv_mae2, 2))
print("\n==============================\n")

"""GridSearchCV for Random Forest"""
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
y_pred_test1 = LinearRegression().fit(x_train, y_train).predict(x_test)
mae_test1 = mean_absolute_error(y_test, y_pred_test1)

# Test Model 2 (Random Forest) on the test data
y_pred_test2 = RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_split=10, min_samples_leaf=5, max_features='sqrt', random_state=500953158).fit(x_train, y_train).predict(x_test)
mae_test2 = mean_absolute_error(y_test, y_pred_test2)

# Compare the first 5 predictions for both models
for i in range(5):
    print("Test Data - Model 1 Predictions:",
          round(y_pred_test1[i], 2),
          "Model 2 Predictions:",
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
cv_scores_model3 = cross_val_score(DecisionTreeRegressor(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=500953158), 
                                   x_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_mae3 = -cv_scores_model3.mean()
print("Model 3 Cross-Validation Mean Absolute Error (CV MAE):", round(cv_mae3, 2))
print("\n==============================\n")

# ============================
# Adding Model 3.1: Decision Tree Regressor with GridSearchCV
# ============================

param_grid_tree = {
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 5]
}

my_model3_1 = DecisionTreeRegressor(random_state=500953158)
grid_search_tree = GridSearchCV(my_model3_1, param_grid_tree, cv=5, scoring='neg_mean_absolute_error', n_jobs=4)

grid_search_tree.fit(x_train, y_train)
best_params_tree = grid_search_tree.best_params_
print("Best Hyperparameters for Decision Tree (GridSearchCV):", best_params_tree)
print("\n==============================\n")

best_model3_1 = grid_search_tree.best_estimator_
y_pred_test3_1 = best_model3_1.predict(x_test)
mae_test3_1 = mean_absolute_error(y_test, y_pred_test3_1)
print("Best Model 3.1 (Decision Tree with GridSearchCV) test MAE is: ", round(mae_test3_1, 2))