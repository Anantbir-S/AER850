# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
file_Name= "housing.csv"
df=pd.read_csv(file_Name)

#print datarame column info
print (df.columns)

#how to acces dataframe columns
x=df["population"]

#how to access dataframe rows
third_row = df.loc[2]

a_row = df.loc[8]
a_row['total_rooms']
