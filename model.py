# Upload files to Google CoLab
from google.colab import files
uploaded = files.upload()
for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

# Import required libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor

# Convert CSV files to Pandas Dataframes
train_data=pd.read_csv("yds_train2018.csv")
test_data=pd.read_csv("yds_test2018.csv")
promo_exp=pd.read_csv("promotional_expense.csv")
sub=pd.read_csv("sample_submission.csv")

# Drop columns not required for training model
train_data.drop("Merchant_ID" , axis=1, inplace=True)
train_data.drop("Week" , axis=1, inplace=True)
train_data.drop("S_No" , axis=1, inplace=True)
test_data.drop("S_No" , axis=1, inplace=True)

# Calculate total sales for all mercahnts and then drop duplicate rows
train_data['Total'] = train_data.groupby(['Year','Month','Product_ID','Country'])['Sales'].transform('sum')
train_data.drop("Sales" , axis=1, inplace=True)
new_train_data=train_data.drop_duplicates()

# One-Hot encode the countries
final_train = pd.get_dummies(new_train_data,prefix=['Country'])
final_test=pd.get_dummies(test_data,prefix=['Country'])

# Prepare training and test data for training the model
predictor_cols=["Year","Month","Product_ID","Country_Argentina","Country_Belgium","Country_Columbia","Country_Denmark","Country_England","Country_Finland"]
train_X=final_train[predictor_cols]
train_Y=final_train.Total
test_X=final_test[predictor_cols]

# Define the model
my_model = RandomForestRegressor()
my_model.fit(train_X, train_Y)

# Predict required values for test data and structure the submission file
predicted_sales = my_model.predict(test_X)
test_data['Sales']=predicted_sales
test=pd.read_csv("yds_test2018.csv")
test_data['S_No']=test.S_No
test_data=test_data.reindex(columns=['S_No','Year','Month','Product_ID','Country','Sales'])

# Convert to CSV and download from Google CoLab
test_data.to_csv('yds_submission2018.csv', index=False)
files.download('yds_submission2018.csv')