import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


#fxn train test split 
from sklearn.model_selection import train_test_split


df = pd.read_csv("housing.csv")

# print(df.columns)

# x_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
#        'total_bedrooms', 'population', 'households', 'median_income',
#        'median_house_value', 'ocean_proximity']

# y_column = ['median_house_value']

# x = df[x_columns] #independent variables, predictor 
# y = df[y_column] #outcome measure, dependent variable


# One Hot Encoding
my_encoder = OneHotEncoder(sparse_output=False)
my_encoder.fit(df[['ocean_proximity']])

encoded_data = my_encoder.transform(df[['ocean_proximity']])
category_names = my_encoder.get_feature_names_out()

encoded_data_df = pd.DataFrame(encoded_data, columns= category_names)


print(encoded_data)
print(category_names)

df = pd.concat([df, encoded_data_df], axis=1)
df = df.drop('ocean_proximity', axis=1)
df.to_csv("test.csv")


# Define X and Y



x_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income',
        'median_house_value', 'ocean_proximity', 'ocean_proximity_<1H OCEAN' 'ocean_proximity_INLAND'
         'ocean_proximity_ISLAND' 'ocean_proximity_NEAR BAY'
         'ocean_proximity_NEAR OCEAN']

y_column = ['median_house_value']

X=x_columns
y=y_column


# Lets now train and make test data sets
# Train / Test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#calling the fxn
train_test_split(X, y)